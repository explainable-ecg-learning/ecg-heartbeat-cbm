"""
data.py — Unified ECG data loading for the heartbeat CBM project.

Provides two PyTorch Datasets built from PTB-XL
(/nfs/data8/schlegel/git/ecg-cbm/data/ptb-xl):

  PTBXLDataset      — full 12-lead records, diagnostic superclass labels + concept
                      annotations from concepts.parquet. Used for record-level tasks.

  HeartbeatDataset  — beat-segmented subsequences for heartbeat completeness
                      classification. Labels and concept pseudo-labels are derived
                      from neurokit2 delineation.

Also exposes signal utilities (bandpass_filter, z_normalize, augment) and the
delineation helpers used to build HeartbeatDataset offline.

Data path convention (override via config):
    PTB-XL root:     /nfs/data8/schlegel/git/ecg-cbm/data/ptb-xl
    Concepts file:   /nfs/data8/schlegel/git/ecg-cbm/data/concepts.parquet
"""

from __future__ import annotations

import ast
import json
import logging
import pickle
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import wfdb
from scipy.signal import butter, filtfilt, find_peaks
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

import neurokit2 as nk

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

PTBXL_ROOT      = Path("/nfs/data8/schlegel/git/ecg-cbm/data/ptb-xl")
CONCEPTS_FILE   = Path("/nfs/data8/schlegel/git/ecg-cbm/data/concepts.parquet")

SUPERCLASS_MAP: Dict[str, int] = {"NORM": 0, "MI": 1, "STTC": 2, "CD": 3, "HYP": 4}
NUM_SUPERCLASSES = len(SUPERCLASS_MAP)

CONCEPT_COLUMNS: List[str] = [
    "p_present", "p_amplitude", "p_duration", "p_morphology",
    "q_depth", "r_amplitude", "s_depth", "qrs_duration",
    "t_amplitude", "t_symmetry", "t_morphology", "st_deviation",
    "rr_mean", "rr_std", "pr_interval", "qt_interval",
    "qtc_interval", "heart_rate", "regularity", "snr",
]
NUM_RECORD_CONCEPTS = len(CONCEPT_COLUMNS)

BEAT_CONCEPT_NAMES: List[str] = ["P", "Q", "R", "S", "T", "U", "PR", "QRS", "ST", "QT"]
NUM_BEAT_CONCEPTS = 10


# ─────────────────────────────────────────────────────────────────────────────
# Signal utilities
# ─────────────────────────────────────────────────────────────────────────────

def bandpass_filter(
    signal: np.ndarray,
    sampling_rate: int = 500,
    lowcut: float = 0.5,
    highcut: float = 40.0,
    order: int = 4,
) -> np.ndarray:
    """Zero-phase bandpass Butterworth filter. signal: (..., T)."""
    nyq = 0.5 * sampling_rate
    b, a = butter(order, [lowcut / nyq, highcut / nyq], btype="band")
    return filtfilt(b, a, signal, axis=-1).astype(signal.dtype)


def z_normalize(signal: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Per-channel z-score normalisation. signal: (C, T)."""
    mean = signal.mean(axis=-1, keepdims=True)
    std  = signal.std(axis=-1, keepdims=True) + eps
    return (signal - mean) / std


def minmax_normalize(signal: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Per-channel normalisation to [-1, 1]. signal: (C, T)."""
    mx = np.abs(signal).max(axis=-1, keepdims=True) + eps
    return signal / mx


def augment(
    signal: np.ndarray,
    t_fixed: int = 500,
    temporal_jitter: int = 10,
    amp_min: float = 0.8,
    amp_max: float = 1.2,
    noise_std: float = 0.01,
    lead_dropout_prob: float = 0.1,
    lead_dropout_max: int = 2,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """Training augmentations: temporal jitter, amplitude scaling, noise, lead dropout."""
    if rng is None:
        rng = np.random.default_rng()
    C, T = signal.shape
    shift     = rng.integers(-temporal_jitter, temporal_jitter + 1)
    start     = max(0, -shift)
    src_start = max(0, shift)
    length    = min(T - src_start, t_fixed - start)
    out = np.zeros((C, t_fixed), dtype=np.float32)
    if length > 0:
        out[:, start : start + length] = signal[:, src_start : src_start + length]
    else:
        out[:, : min(T, t_fixed)] = signal[:, : min(T, t_fixed)]
    out = out * rng.uniform(amp_min, amp_max)
    out = out + rng.standard_normal(out.shape).astype(np.float32) * noise_std
    if lead_dropout_prob > 0 and rng.random() < lead_dropout_prob:
        n_drop = rng.integers(1, lead_dropout_max + 1)
        out[rng.choice(C, size=min(n_drop, C), replace=False)] = 0.0
    return out


# ─────────────────────────────────────────────────────────────────────────────
# R-peak detection — multi-lead consensus (adapted from DisentangledECG)
# ─────────────────────────────────────────────────────────────────────────────

def _detect_r_peaks_multilead(
    signal: np.ndarray,
    sampling_rate: int = 500,
    lead_idx: int = 1,
    min_lead_consensus: int = 6,
) -> np.ndarray:
    """
    Multi-lead consensus R-peak detection.

    1. Primary detection on `lead_idx` with nk.ecg_clean + nk.ecg_peaks.
    2. Searches all other leads for peaks not already close to a primary peak;
       adds a candidate only when ≥ `min_lead_consensus` leads independently
       agree on it (and amplitude / neighbour-amplitude checks pass).
    3. Removes weak peaks whose amplitude is < 25 % of both neighbours and
       < 0.2 in absolute terms.
    """
    # signal: (C, T) — transpose to (T, C) for lead indexing
    sig = signal.T  # (T, C)
    threshold = sampling_rate // 5  # min distance to call two peaks "different"

    # ── Primary detection ────────────────────────────────────────────────────
    try:
        cleaned = nk.ecg_clean(sig[:, lead_idx], sampling_rate=sampling_rate)
        _, info  = nk.ecg_peaks(cleaned, sampling_rate=sampling_rate)
        r_peaks  = np.array(
            sorted(int(p) for p in info["ECG_R_Peaks"] if not np.isnan(p)),
            dtype=int,
        )
    except Exception:
        return np.array([], dtype=int)

    if len(r_peaks) <= 2:
        return r_peaks

    # ── Cross-validate with remaining leads ──────────────────────────────────
    num_leads = sig.shape[1]
    candidate_list: list = []  # (peak_idx, lead, amplitude)

    for lead in range(num_leads):
        if lead == lead_idx:
            continue
        try:
            cleaned_l = nk.ecg_clean(sig[:, lead], sampling_rate=sampling_rate)
            _, info_l  = nk.ecg_peaks(cleaned_l, sampling_rate=sampling_rate)
            other = [int(p) for p in info_l["ECG_R_Peaks"] if not np.isnan(p)]
            for cand in other:
                if not np.any(np.abs(r_peaks - cand) < threshold):
                    candidate_list.append((cand, lead, abs(float(sig[cand, lead]))))
        except Exception:
            continue

    # Keep only candidates seen in ≥ min_lead_consensus leads
    candidate_list.sort(key=lambda x: x[0])
    i = 0
    while i < len(candidate_list):
        group = [candidate_list[i]]
        j = i + 1
        while j < len(candidate_list) and abs(candidate_list[j][0] - candidate_list[i][0]) < threshold:
            group.append(candidate_list[j])
            j += 1
        if len({g[1] for g in group}) >= min_lead_consensus:
            best = max(group, key=lambda x: x[2])
            p = best[0]
            a = float(sig[p, lead_idx])
            if a >= -0.1:
                pos = int(np.searchsorted(r_peaks, p))
                if 0 < pos < len(r_peaks):
                    a_prev = abs(float(sig[r_peaks[pos - 1], lead_idx]))
                    a_next = abs(float(sig[r_peaks[pos],     lead_idx]))
                    if abs(a) >= 0.4 * a_prev and abs(a) >= 0.4 * a_next:
                        r_peaks = np.unique(np.append(r_peaks, p))
        i = j

    # ── Remove weak peaks ────────────────────────────────────────────────────
    amps = np.abs(sig[r_peaks, lead_idx]).astype(np.float32)
    keep = np.ones(len(r_peaks), dtype=bool)
    for i in range(1, len(r_peaks) - 1):
        a = float(amps[i])
        if (
            a < 0.25 * float(amps[i - 1])
            and a < 0.25 * float(amps[i + 1])
            and a < 0.2
            and float(amps[i - 1]) > 0.1
            and float(amps[i + 1]) > 0.1
        ):
            keep[i] = False
    return r_peaks[keep]


# ─────────────────────────────────────────────────────────────────────────────
# Per-beat masked normalisation (adapted from DisentangledECG)
# ─────────────────────────────────────────────────────────────────────────────

def _masked_normalize_beat(
    beat: np.ndarray,
    mask: np.ndarray,
    eps: float = 1e-6,
) -> np.ndarray:
    """
    Per-channel mean-variance normalisation using only the natural beat region.

    beat : (C, T) float32
    mask : (T,)  float32  — 1 inside natural boundary, 0 outside

    The masked-out region is preserved as zeros after normalisation.
    """
    m     = mask.astype(np.float32)[np.newaxis, :]          # (1, T)
    denom = float(np.maximum(m.sum(), 1.0))
    mean  = (beat * m).sum(axis=1, keepdims=True) / denom   # (C, 1)
    var   = (((beat - mean) ** 2) * m).sum(axis=1, keepdims=True) / denom
    std   = np.sqrt(np.maximum(var, eps))                   # (C, 1)
    return ((beat - mean) / std * m).astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Delineation (neurokit2 wrapper)
# ─────────────────────────────────────────────────────────────────────────────

def _detect_u_wave(
    ecg_signal: np.ndarray,
    t_offsets: np.ndarray,
    next_p_onsets: np.ndarray,
    sampling_rate: int = 500,
    prominence_fraction: float = 0.02,
) -> list:
    """Heuristic U-wave detector. Returns list of peak indices (or NaN) per beat."""
    threshold = prominence_fraction * np.ptp(ecg_signal)
    u_peaks: list = []
    for t_off, next_p in zip(t_offsets, next_p_onsets):
        if np.isnan(t_off):
            u_peaks.append(np.nan)
            continue
        t_int = int(t_off)
        end   = int(next_p) if not np.isnan(next_p) else min(t_int + int(0.2 * sampling_rate), len(ecg_signal))
        window = ecg_signal[t_int:end]
        if len(window) < 10:
            u_peaks.append(np.nan)
            continue
        peaks, _ = find_peaks(window, prominence=threshold)
        u_peaks.append(t_int + int(peaks[0]) if len(peaks) > 0 else np.nan)
    return u_peaks


def delineate_record(
    ecg_signal: np.ndarray,
    sampling_rate: int = 500,
    lead_idx: int = 1,
) -> dict:
    """
    Full neurokit2 delineation of a (12, T) or 1-D ECG record.

    Uses multi-lead consensus R-peak detection (see _detect_r_peaks_multilead).
    Returns dict: {r_peaks, delineation, u_peaks, quality_ok}.
    """
    lead_signal = ecg_signal[lead_idx] if ecg_signal.ndim == 2 else ecg_signal
    try:
        # Multi-lead consensus R-peak detection
        if ecg_signal.ndim == 2:
            r_peaks = _detect_r_peaks_multilead(ecg_signal, sampling_rate, lead_idx)
        else:
            # Fallback for 1-D input
            cleaned = nk.ecg_clean(lead_signal, sampling_rate=sampling_rate)
            _, rpeaks_info = nk.ecg_peaks(cleaned, sampling_rate=sampling_rate)
            r_peaks = np.array([int(p) for p in rpeaks_info["ECG_R_Peaks"] if not np.isnan(p)], dtype=int)

        if len(r_peaks) < 3:
            return {"r_peaks": r_peaks, "delineation": {}, "u_peaks": [], "quality_ok": False}
        rr = np.diff(r_peaks)
        if np.any(rr < 300) or np.any(rr > 2000):
            return {"r_peaks": r_peaks, "delineation": {}, "u_peaks": [], "quality_ok": False}
        cleaned = nk.ecg_clean(lead_signal, sampling_rate=sampling_rate)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            _, waves = nk.ecg_delineate(cleaned, rpeaks=r_peaks, sampling_rate=sampling_rate, method="dwt")
        delin = {k: np.array(v, dtype=float) for k, v in waves.items()}
    except Exception:
        return {"r_peaks": np.array([], dtype=int), "delineation": {}, "u_peaks": [], "quality_ok": False}

    t_offsets    = delin.get("ECG_T_Offsets",  np.full(len(r_peaks), np.nan))
    p_onsets     = delin.get("ECG_P_Onsets",   np.full(len(r_peaks), np.nan))
    next_p_onsets = np.concatenate([p_onsets[1:], [np.nan]])
    u_peaks = _detect_u_wave(lead_signal, t_offsets, next_p_onsets, sampling_rate)
    return {"r_peaks": r_peaks, "delineation": delin, "u_peaks": np.asarray(u_peaks, dtype=float), "quality_ok": True}


def get_concept_labels(
    subseq_start: int,
    subseq_end: int,
    delineation: dict,
    r_peaks: np.ndarray,
    u_peaks,
) -> Tuple[List[int], List[int]]:
    """
    Derive beat concept vector and confidence weights for a window [start, end].

    Returns
    -------
    concepts : [P, Q, R, S, T, U, PR, QRS, ST, QT] — 10 binary values
    weights  : 10 binary values (0 = delineation failed for this concept)
    """
    def _get(key):
        v = delineation.get(key, np.array([]))
        return v if v is not None else np.array([])

    def in_window(indices) -> bool:
        for idx in indices:
            if not (isinstance(idx, float) and np.isnan(idx)):
                if subseq_start <= int(idx) <= subseq_end:
                    return True
        return False

    def span_in_window(starts, ends) -> bool:
        for s, e in zip(starts, ends):
            if not (np.isnan(s) or np.isnan(e)) and subseq_start <= int(s) and int(e) <= subseq_end:
                return True
        return False

    def has_valid(indices) -> bool:
        return any(not (isinstance(v, float) and np.isnan(v)) for v in indices)

    c_P   = int(in_window(_get("ECG_P_Peaks")))
    c_Q   = int(in_window(_get("ECG_Q_Peaks")))
    c_R   = int(in_window(r_peaks))
    c_S   = int(in_window(_get("ECG_S_Peaks")))
    c_T   = int(in_window(_get("ECG_T_Peaks")))
    c_U   = int(in_window(u_peaks if u_peaks is not None else []))
    c_PR  = int(span_in_window(_get("ECG_P_Onsets"),  _get("ECG_R_Onsets")))
    c_QRS = int(span_in_window(_get("ECG_R_Onsets"),  _get("ECG_R_Offsets")))
    c_ST  = int(span_in_window(_get("ECG_R_Offsets"), _get("ECG_T_Offsets")))
    c_QT  = int(span_in_window(_get("ECG_R_Onsets"),  _get("ECG_T_Offsets")))

    w_P   = int(has_valid(_get("ECG_P_Peaks")))
    w_Q   = int(has_valid(_get("ECG_Q_Peaks")))
    w_R   = 1
    w_S   = int(has_valid(_get("ECG_S_Peaks")))
    w_T   = int(has_valid(_get("ECG_T_Peaks")))
    w_U   = 1
    w_PR  = int(has_valid(_get("ECG_P_Onsets")) and has_valid(_get("ECG_R_Onsets")))
    w_QRS = int(has_valid(_get("ECG_R_Onsets")) and has_valid(_get("ECG_R_Offsets")))
    w_ST  = int(has_valid(_get("ECG_R_Offsets")) and has_valid(_get("ECG_T_Offsets")))
    w_QT  = int(has_valid(_get("ECG_R_Onsets")) and has_valid(_get("ECG_T_Offsets")))

    return (
        [c_P, c_Q, c_R, c_S, c_T, c_U, c_PR, c_QRS, c_ST, c_QT],
        [w_P, w_Q, w_R, w_S, w_T, w_U, w_PR, w_QRS, w_ST, w_QT],
    )


# ─────────────────────────────────────────────────────────────────────────────
# Beat extraction
# ─────────────────────────────────────────────────────────────────────────────

def _window(signal: np.ndarray, start: int, end: int, t_fixed: int) -> np.ndarray:
    C, T = signal.shape
    chunk = signal[:, max(0, start) : min(T, end)]
    out = np.zeros((C, t_fixed), dtype=np.float32)
    out[:, : min(chunk.shape[1], t_fixed)] = chunk[:, : t_fixed]
    return out


def extract_beats(
    signal: np.ndarray,
    delin: dict,
    t_fixed: int = 500,
    beat_window_pre: int = 250,
    beat_window_post: int = 250,
    neg_oversample_factor: float = 1.0,
    rng: Optional[np.random.Generator] = None,
) -> List[Tuple[np.ndarray, int, List[int], List[int]]]:
    """
    R-peak aligned beat extraction with natural boundary masking.

    Positives: symmetric window centred on R-peak; signal is normalised using
    only the natural beat region (midpoints to neighbouring peaks), following
    the DisentangledECG approach.

    Negatives: 4 strategies (truncation, late-start, random-crop, inter-beat)
    using standard per-channel z-normalisation.
    """
    if rng is None:
        rng = np.random.default_rng()
    if not delin.get("quality_ok", False):
        return []

    r_peaks     = delin["r_peaks"]
    delineation = delin["delineation"]
    u_peaks     = delin.get("u_peaks", np.array([]))
    C, T        = signal.shape
    samples: list = []

    # Symmetric half-widths (mirrors DisentangledECG centering)
    half_pre  = t_fixed // 2
    half_post = t_fixed - half_pre

    # ── Positives (skip first and last — no neighbour on both sides) ─────────
    for i in range(1, len(r_peaks) - 1):
        r     = int(r_peaks[i])
        start = r - half_pre
        end   = r + half_post
        if start < 0 or end > T:
            continue

        # Natural boundaries: midpoints to neighbouring R-peaks
        prev_r    = int(r_peaks[i - 1])
        next_r    = int(r_peaks[i + 1])
        nat_start = (r + prev_r) // 2
        nat_end   = (r + next_r) // 2

        # Build mask in window-local coordinates
        mask = np.zeros(t_fixed, dtype=np.float32)
        rel_s = int(np.clip(nat_start - start, 0, t_fixed))
        rel_e = int(np.clip(nat_end   - start, 0, t_fixed))
        if rel_e > rel_s:
            mask[rel_s:rel_e] = 1.0

        subseq = _window(signal, start, end, t_fixed)
        subseq = _masked_normalize_beat(subseq, mask)

        concepts, weights = get_concept_labels(start, end, delineation, r_peaks, u_peaks)
        samples.append((subseq, 1, concepts, weights))

    n_per_strategy = max(1, int(round(len(samples) / 4 * neg_oversample_factor)))

    def _znorm(s: np.ndarray) -> np.ndarray:
        mean = s.mean(axis=-1, keepdims=True)
        std  = s.std(axis=-1, keepdims=True) + 1e-8
        return ((s - mean) / std).astype(np.float32)

    # Strategy 1: early truncation
    for _ in range(n_per_strategy):
        if not len(r_peaks): break
        r = int(r_peaks[rng.integers(0, len(r_peaks))])
        start = r - beat_window_pre
        cut   = r + int(rng.uniform(0.3, 0.6) * beat_window_post)
        if start < 0 or cut > T: continue
        subseq = _znorm(_window(signal, start, cut, t_fixed))
        concepts, weights = get_concept_labels(start, cut, delineation, r_peaks, u_peaks)
        samples.append((subseq, 0, concepts, weights))

    # Strategy 2: late start
    for _ in range(n_per_strategy):
        if not len(r_peaks): break
        r = int(r_peaks[rng.integers(0, len(r_peaks))])
        late_start = r - rng.integers(0, beat_window_pre)
        end        = r + beat_window_post
        if late_start < 0 or end > T: continue
        subseq = _znorm(_window(signal, late_start, end, t_fixed))
        concepts, weights = get_concept_labels(late_start, end, delineation, r_peaks, u_peaks)
        samples.append((subseq, 0, concepts, weights))

    # Strategy 3: random crop (no R-peak)
    for _ in range(n_per_strategy):
        start = rng.integers(0, max(1, T - t_fixed))
        end   = start + t_fixed
        if end > T: continue
        if any(start <= int(r) <= end for r in r_peaks): continue
        subseq = _znorm(_window(signal, start, end, t_fixed))
        concepts, weights = get_concept_labels(start, end, delineation, r_peaks, u_peaks)
        samples.append((subseq, 0, concepts, weights))

    # Strategy 4: inter-beat baseline
    for _ in range(n_per_strategy):
        if len(r_peaks) < 2: break
        idx   = rng.integers(0, len(r_peaks) - 1)
        mid   = (int(r_peaks[idx]) + int(r_peaks[idx + 1])) // 2
        start = mid - t_fixed // 2
        end   = mid + t_fixed // 2
        if start < 0 or end > T: continue
        subseq = _znorm(_window(signal, start, end, t_fixed))
        concepts, weights = get_concept_labels(start, end, delineation, r_peaks, u_peaks)
        samples.append((subseq, 0, concepts, weights))

    return samples


def extract_sliding_windows(
    signal: np.ndarray,
    delin: dict,
    t_fixed: int = 500,
    stride: int = 50,
) -> List[Tuple[np.ndarray, int, List[int], List[int]]]:
    """Dense sliding window extraction. Label = 1 if full QT interval inside window."""
    if not delin.get("quality_ok", False):
        return []
    C, T = signal.shape
    r_peaks    = delin["r_peaks"]
    delineation = delin["delineation"]
    u_peaks    = delin.get("u_peaks", np.array([]))
    samples: list = []
    for start in range(0, T - t_fixed + 1, stride):
        end = start + t_fixed
        subseq = signal[:, start:end].copy().astype(np.float32)
        concepts, weights = get_concept_labels(start, end, delineation, r_peaks, u_peaks)
        task_label = int(concepts[7])   # c_QRS = heartbeat present
        samples.append((subseq, task_label, concepts, weights))
    return samples


# ─────────────────────────────────────────────────────────────────────────────
# PTBXLDataset — full-record diagnostic + concept dataset
# ─────────────────────────────────────────────────────────────────────────────

class PTBXLDataset(Dataset):
    """
    PTB-XL 12-lead ECG dataset.

    Each item: {signal (C,T), age, sex, pathology (5,), concepts (20,), concept_mask (20,), record_id}

    Use `PTBXLDataset.make_splits(data_path)` to get (train, val, test).
    """

    def __init__(
        self,
        records: pd.DataFrame,
        data_path: Path,
        sampling_rate: int = 500,
        concepts_df: Optional[pd.DataFrame] = None,
        lead: Optional[int] = None,
    ) -> None:
        self.records       = records.copy()
        self.data_path     = Path(data_path)
        self.sampling_rate = sampling_rate
        self.filename_col  = "filename_hr" if sampling_rate == 500 else "filename_lr"
        self.concepts_df   = concepts_df
        self.lead          = lead
        self.ecg_ids       = self.records.index.tolist()

    @classmethod
    def make_splits(
        cls,
        data_path: str | Path = PTBXL_ROOT,
        sampling_rate: int = 500,
        val_fold: int = 9,
        test_fold: int = 10,
        concept_file: Optional[str | Path] = CONCEPTS_FILE,
        lead: Optional[int] = None,
    ) -> Tuple["PTBXLDataset", "PTBXLDataset", "PTBXLDataset"]:
        """Build train / val / test from PTB-XL fold assignments (folds 1-8 / 9 / 10)."""
        data_path = Path(data_path)
        meta = pd.read_csv(data_path / "ptbxl_database.csv", index_col="ecg_id", dtype={"scp_codes": str})
        meta["age"] = meta["age"].fillna(meta["age"].median())

        scp = pd.read_csv(data_path / "scp_statements.csv", index_col=0)
        scp = scp[scp["diagnostic"] == 1.0]
        code_to_sc: Dict[str, str] = scp["diagnostic_class"].dropna().to_dict()

        def _pathology(scp_str: str) -> np.ndarray:
            label = np.zeros(NUM_SUPERCLASSES, dtype=np.float32)
            try:
                codes = ast.literal_eval(scp_str)
            except (ValueError, SyntaxError):
                return label
            for code, likelihood in codes.items():
                sc = code_to_sc.get(code)
                if sc and sc in SUPERCLASS_MAP and likelihood > 0:
                    label[SUPERCLASS_MAP[sc]] = 1.0
            return label

        meta["pathology"] = meta["scp_codes"].apply(_pathology)

        concepts_df: Optional[pd.DataFrame] = None
        if concept_file is not None and Path(concept_file).exists():
            raw = pd.read_parquet(concept_file)
            if "ecg_id" in raw.columns:
                raw = raw.set_index("ecg_id")
            cols = [c for c in CONCEPT_COLUMNS if c in raw.columns]
            concepts_df = raw[cols].copy()

        train_folds = [f for f in range(1, 11) if f not in (val_fold, test_fold)]
        make = lambda subset: cls(subset, data_path, sampling_rate, concepts_df, lead)
        return (
            make(meta[meta["strat_fold"].isin(train_folds)]),
            make(meta[meta["strat_fold"] == val_fold]),
            make(meta[meta["strat_fold"] == test_fold]),
        )

    def __len__(self) -> int:
        return len(self.ecg_ids)

    def __getitem__(self, idx: int) -> Dict:
        ecg_id = self.ecg_ids[idx]
        row    = self.records.loc[ecg_id]
        try:
            signal, _ = wfdb.rdsamp(str(self.data_path / row[self.filename_col]))
            signal = signal.T.astype(np.float32)                    # (12, T)
        except Exception:
            T = 5000 if self.sampling_rate == 500 else 1000
            signal = np.zeros((12, T), dtype=np.float32)
        signal = minmax_normalize(signal)
        if self.lead is not None:
            signal = signal[self.lead : self.lead + 1, :]

        age       = np.float32(row["age"])
        sex       = int(row["sex"])
        pathology = row["pathology"].astype(np.float32)

        if self.concepts_df is not None and ecg_id in self.concepts_df.index:
            raw_c = self.concepts_df.loc[ecg_id, CONCEPT_COLUMNS].values.astype(np.float32)
        else:
            raw_c = np.full(NUM_RECORD_CONCEPTS, np.nan, dtype=np.float32)
        mask     = ~np.isnan(raw_c)
        concepts = np.where(mask, raw_c, 0.0).astype(np.float32)

        return {
            "signal":       torch.from_numpy(signal),
            "age":          torch.tensor(age),
            "sex":          torch.tensor(sex, dtype=torch.long),
            "pathology":    torch.from_numpy(pathology),
            "concepts":     torch.from_numpy(concepts),
            "concept_mask": torch.from_numpy(mask),
            "record_id":    int(ecg_id),
        }


# ─────────────────────────────────────────────────────────────────────────────
# HeartbeatDataset — beat-level completeness classification
# ─────────────────────────────────────────────────────────────────────────────

class HeartbeatDataset(Dataset):
    """
    Beat-level dataset for heartbeat completeness classification.

    Each item: {signal (C,T), task_label, concept_labels (10,), concept_weights (10,), record_id}

    Build with `HeartbeatDataset.from_records(data_path)` (slow; uses neurokit2).
    Cache with `ds.save_cache(dir, split)` / `HeartbeatDataset.from_cache(dir, split)`.
    """

    def __init__(
        self,
        samples: List[dict],
        augment_fn=None,
        active_concepts: Optional[List[int]] = None,
        model_lead: Optional[int] = None,
        model_leads: Optional[List[int]] = None,
    ) -> None:
        self.samples         = samples
        self.augment_fn      = augment_fn
        self.active_concepts = active_concepts
        self.model_lead      = model_lead
        self.model_leads     = model_leads

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict:
        s      = self.samples[idx]
        signal = s["signal"].astype(np.float32)

        if self.augment_fn is not None:
            signal = self.augment_fn(signal)

        if self.model_leads is not None:
            signal = signal[np.array(self.model_leads), :]
        elif self.model_lead is not None:
            signal = signal[self.model_lead : self.model_lead + 1, :]

        concepts = np.asarray(s["concept_labels"],  dtype=np.float32)
        weights  = np.asarray(s["concept_weights"], dtype=np.float32)
        if self.active_concepts is not None:
            concepts = concepts[self.active_concepts]
            weights  = weights[self.active_concepts]

        return {
            "signal":          torch.from_numpy(signal),
            "task_label":      torch.tensor(s["task_label"], dtype=torch.long),
            "concept_labels":  torch.from_numpy(concepts),
            "concept_weights": torch.from_numpy(weights),
            "record_id":       s["record_id"],
        }

    def label_distribution(self) -> dict:
        labels = [s["task_label"] for s in self.samples]
        n1 = sum(labels)
        return {"complete": n1, "incomplete": len(labels) - n1, "total": len(labels)}

    # ── Persistence ──────────────────────────────────────────────────────────

    def save_cache(self, cache_dir: str | Path, split: str) -> None:
        cache_dir = Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        with open(cache_dir / f"{split}.pkl", "wb") as f:
            pickle.dump(self.samples, f, protocol=pickle.HIGHEST_PROTOCOL)
        logger.info(f"Saved {split} cache ({len(self.samples)} samples) → {cache_dir}")

    @classmethod
    def from_cache(cls, cache_dir: str | Path, split: str) -> "HeartbeatDataset":
        path = Path(cache_dir) / f"{split}.pkl"
        with open(path, "rb") as f:
            samples = pickle.load(f)
        logger.info(f"Loaded {split} cache ({len(samples)} samples) from {path}")
        return cls(samples)

    # ── Full build from raw PTB-XL records ───────────────────────────────────

    @classmethod
    def from_records(
        cls,
        data_path: str | Path = PTBXL_ROOT,
        sampling_rate: int = 500,
        t_fixed: int = 500,
        beat_window_pre: int = 100,
        beat_window_post: int = 400,
        delineation_cache_dir: Optional[str | Path] = None,
        val_fold: int = 9,
        test_fold: int = 10,
        lead_idx: int = 1,
        neg_oversample_factor: float = 1.0,
        use_sliding_window: bool = False,
        stride: int = 50,
        seed: int = 42,
    ) -> Tuple["HeartbeatDataset", "HeartbeatDataset", "HeartbeatDataset"]:
        """
        Build train/val/test from PTB-XL.
        Preprocesses (bandpass + z-normalise), delineates (neurokit2), extracts beats.
        Uses delineation_cache_dir to persist intermediate results (recommended).
        """
        data_path = Path(data_path)
        rng       = np.random.default_rng(seed)
        meta      = pd.read_csv(data_path / "ptbxl_database.csv", index_col="ecg_id")
        filename_col = "filename_hr" if sampling_rate == 500 else "filename_lr"

        train_folds = [f for f in range(1, 11) if f not in (val_fold, test_fold)]
        splits = {
            "train": meta[meta["strat_fold"].isin(train_folds)],
            "val":   meta[meta["strat_fold"] == val_fold],
            "test":  meta[meta["strat_fold"] == test_fold],
        }
        all_samples: Dict[str, list] = {k: [] for k in splits}

        for split_name, subset in splits.items():
            logger.info(f"Processing {split_name} ({len(subset)} records)…")
            for ecg_id, row in tqdm(subset.iterrows(), total=len(subset), desc=split_name):
                try:
                    raw, _ = wfdb.rdsamp(str(data_path / row[filename_col]))
                    signal = bandpass_filter(raw.T.astype(np.float32), sampling_rate)
                    # No record-level z-normalisation — per-beat masked normalisation
                    # is applied inside extract_beats for positives.

                    delin_result = _load_or_compute_delin(
                        ecg_id, signal, sampling_rate, lead_idx, delineation_cache_dir
                    )

                    if use_sliding_window:
                        beats = extract_sliding_windows(signal, delin_result, t_fixed, stride)
                    else:
                        beats = extract_beats(
                            signal, delin_result, t_fixed,
                            beat_window_pre, beat_window_post, neg_oversample_factor, rng,
                        )

                    for subseq, task_label, concepts, weights in beats:
                        all_samples[split_name].append({
                            "signal":          subseq,
                            "task_label":      task_label,
                            "concept_labels":  concepts,
                            "concept_weights": weights,
                            "record_id":       int(ecg_id),
                        })
                except Exception as e:
                    logger.warning(f"Skipped ecg_id={ecg_id}: {e}")

        return (
            cls(all_samples["train"]),
            cls(all_samples["val"]),
            cls(all_samples["test"]),
        )


def _load_or_compute_delin(
    ecg_id: int,
    signal: np.ndarray,
    sampling_rate: int,
    lead_idx: int,
    cache_dir: Optional[Path],
) -> dict:
    if cache_dir is not None:
        cache_dir = Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_file = cache_dir / f"{ecg_id}.pkl"
        if cache_file.exists():
            with open(cache_file, "rb") as f:
                return pickle.load(f)

    result = delineate_record(signal, sampling_rate, lead_idx)

    if cache_dir is not None:
        with open(cache_dir / f"{ecg_id}.pkl", "wb") as f:
            pickle.dump(result, f, protocol=pickle.HIGHEST_PROTOCOL)
    return result


# ─────────────────────────────────────────────────────────────────────────────
# DataLoader factory
# ─────────────────────────────────────────────────────────────────────────────

def make_dataloaders(
    train_ds: Dataset,
    val_ds: Dataset,
    test_ds: Dataset,
    batch_size: int = 256,
    num_workers: int = 4,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    common = dict(batch_size=batch_size, num_workers=num_workers, pin_memory=True,
                  persistent_workers=num_workers > 0)
    return (
        DataLoader(train_ds, shuffle=True,  **common),
        DataLoader(val_ds,   shuffle=False, **common),
        DataLoader(test_ds,  shuffle=False, **common),
    )
