"""
data.py — Unified ECG data loading and preprocessing for the heartbeat CBM/CEM project.

Provides two PyTorch Datasets built from PTB-XL:
  PTBXLDataset      — full 12-lead records, 5 diagnostic superclass labels + 20 record-level
                      morphological/rhythm concepts from concepts.parquet.
  HeartbeatDataset  — beat-level 12-lead windows for heartbeat completeness classification
                      with 10 wave/interval concepts derived from neurokit2 delineation.
"""

from __future__ import annotations

import ast
import hashlib
import json
import logging
import pickle
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import neurokit2 as nk
import numpy as np
import pandas as pd
import torch
import wfdb
from scipy.signal import butter, filtfilt, find_peaks
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

PTBXL_ROOT    = Path("/nfs/data8/schlegel/git/ecg-cbm/data/ptb-xl")
CONCEPTS_FILE = Path("/nfs/data8/schlegel/git/ecg-cbm/data/concepts.parquet")

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
CACHE_SCHEMA_VERSION = 2


# ─────────────────────────────────────────────────────────────────────────────
# Path & cache helpers
# ─────────────────────────────────────────────────────────────────────────────

def resolve_data_paths(cfg: dict) -> Tuple[Path, Path]:
    """Resolve (ptbxl_path, concept_parquet_path) from a config dictionary."""
    data_cfg = cfg.get("data", {})
    data_dir = data_cfg.get("data_dir", None)

    raw_data_path = Path(data_cfg.get("data_path", str(PTBXL_ROOT)))
    if data_dir and not raw_data_path.is_absolute():
        data_path = Path(data_dir) / raw_data_path
    else:
        data_path = raw_data_path

    raw_concept = Path(data_cfg.get("concept_file", str(CONCEPTS_FILE)))
    if data_dir and not raw_concept.is_absolute():
        concept_path = Path(data_dir) / raw_concept
    else:
        concept_path = raw_concept

    return data_path, concept_path


def dataset_cache_key(data_cfg: dict) -> Tuple[str, dict]:
    """Compute a deterministic cache key from dataset-shaping config fields."""
    spec = {
        "schema_v":              CACHE_SCHEMA_VERSION,
        "sampling_rate":         int(data_cfg.get("sampling_rate", 500)),
        "t_fixed":               int(data_cfg.get("t_fixed", 500)),
        "lead":                  int(data_cfg.get("lead", 1)),
        "val_fold":              int(data_cfg.get("val_fold", 9)),
        "test_fold":             int(data_cfg.get("test_fold", 10)),
        "beat_window_pre":       int(data_cfg.get("beat_window_pre", 250)),
        "beat_window_post":      int(data_cfg.get("beat_window_post", 250)),
        "pos_jitter":            int(data_cfg.get("pos_jitter", 60)),
        "use_masked_norm":       bool(data_cfg.get("use_masked_norm", False)),
        "neg_oversample_factor": float(data_cfg.get("neg_oversample_factor", 1.0)),
        "use_sliding_window":    bool(data_cfg.get("use_sliding_window", False)),
        "stride":                int(data_cfg.get("stride", 50)),
    }
    delineator = str(data_cfg.get("delineator", "neurokit2")).lower()
    if delineator != "neurokit2":
        spec["delineator"] = delineator
    digest = hashlib.sha1(json.dumps(spec, sort_keys=True).encode("utf-8")).hexdigest()[:10]
    return digest, spec


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
    """Per-channel z-score normalisation. signal: (..., T)."""
    mean = signal.mean(axis=-1, keepdims=True)
    std  = signal.std(axis=-1, keepdims=True) + eps
    return ((signal - mean) / std).astype(np.float32)


def minmax_normalize(signal: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Per-channel normalisation to [-1, 1]. signal: (C, T)."""
    mx = np.abs(signal).max(axis=-1, keepdims=True) + eps
    return (signal / mx).astype(np.float32)


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
    """Training augmentations: edge-replicated temporal jitter, scaling, noise, lead dropout."""
    if rng is None:
        rng = np.random.default_rng()
    C, T = signal.shape
    shift = int(rng.integers(-temporal_jitter, temporal_jitter + 1)) if temporal_jitter > 0 else 0
    if shift == 0 and T == t_fixed:
        out = signal.astype(np.float32, copy=True)
    else:
        # Edge-pad instead of zero-pad so jitter never introduces artificial 0-flatlines
        src = signal[:, :t_fixed] if T >= t_fixed else np.pad(signal, ((0, 0), (0, t_fixed - T)), mode="edge")
        if shift > 0:
            out = np.concatenate([src[:, shift:], np.repeat(src[:, -1:], shift, axis=1)], axis=1)
        elif shift < 0:
            s = -shift
            out = np.concatenate([np.repeat(src[:, :1], s, axis=1), src[:, :-s]], axis=1)
        else:
            out = src.copy()
        out = out.astype(np.float32)

    out = out * float(rng.uniform(amp_min, amp_max))
    if noise_std > 0:
        out = out + rng.standard_normal(out.shape).astype(np.float32) * noise_std
    if lead_dropout_prob > 0 and rng.random() < lead_dropout_prob:
        n_drop = int(rng.integers(1, lead_dropout_max + 1))
        out[rng.choice(C, size=min(n_drop, C), replace=False)] = 0.0
    return out


# ─────────────────────────────────────────────────────────────────────────────
# R-peak detection — multi-lead consensus
# ─────────────────────────────────────────────────────────────────────────────

def _detect_r_peaks_multilead(
    signal: np.ndarray,
    sampling_rate: int = 500,
    lead_idx: int = 1,
    min_lead_consensus: int = 6,
) -> np.ndarray:
    """Multi-lead consensus R-peak detection."""
    sig = signal.T  # (T, C)
    threshold = sampling_rate // 5

    try:
        cleaned = nk.ecg_clean(sig[:, lead_idx], sampling_rate=sampling_rate)
        _, info = nk.ecg_peaks(cleaned, sampling_rate=sampling_rate)
        r_peaks = np.array(
            sorted(int(p) for p in info["ECG_R_Peaks"] if not np.isnan(p)),
            dtype=int,
        )
    except Exception:
        return np.array([], dtype=int)

    if len(r_peaks) <= 2:
        return r_peaks

    num_leads = sig.shape[1]
    candidate_list: list = []

    for lead in range(num_leads):
        if lead == lead_idx:
            continue
        try:
            cleaned_l = nk.ecg_clean(sig[:, lead], sampling_rate=sampling_rate)
            _, info_l = nk.ecg_peaks(cleaned_l, sampling_rate=sampling_rate)
            other = [int(p) for p in info_l["ECG_R_Peaks"] if not np.isnan(p)]
            for cand in other:
                if not np.any(np.abs(r_peaks - cand) < threshold):
                    candidate_list.append((cand, lead, abs(float(sig[cand, lead]))))
        except Exception:
            continue

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


def _masked_normalize_beat(
    beat: np.ndarray,
    mask: np.ndarray,
    eps: float = 1e-6,
) -> np.ndarray:
    """Legacy per-channel mean-variance normalisation using only the natural beat region."""
    m     = mask.astype(np.float32)[np.newaxis, :]
    denom = float(np.maximum(m.sum(), 1.0))
    mean  = (beat * m).sum(axis=1, keepdims=True) / denom
    var   = (((beat - mean) ** 2) * m).sum(axis=1, keepdims=True) / denom
    std   = np.sqrt(np.maximum(var, eps))
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
    """Full neurokit2 delineation of a (12, T) or 1-D ECG record."""
    lead_signal = ecg_signal[lead_idx] if ecg_signal.ndim == 2 else ecg_signal
    try:
        if ecg_signal.ndim == 2:
            r_peaks = _detect_r_peaks_multilead(ecg_signal, sampling_rate, lead_idx)
        else:
            cleaned = nk.ecg_clean(lead_signal, sampling_rate=sampling_rate)
            _, rpeaks_info = nk.ecg_peaks(cleaned, sampling_rate=sampling_rate)
            r_peaks = np.array([int(p) for p in rpeaks_info["ECG_R_Peaks"] if not np.isnan(p)], dtype=int)

        if len(r_peaks) < 3:
            return {"r_peaks": r_peaks, "delineation": {}, "u_peaks": [], "quality_ok": False}
        # Physiological RR bounds scaled by sampling_rate (0.3s–4.0s → 150–2000 samples @ 500Hz)
        min_rr = int(0.30 * sampling_rate)
        max_rr = int(4.00 * sampling_rate)
        rr = np.diff(r_peaks)
        if np.any(rr < min_rr) or np.any(rr > max_rr):
            return {"r_peaks": r_peaks, "delineation": {}, "u_peaks": [], "quality_ok": False}

        cleaned = nk.ecg_clean(lead_signal, sampling_rate=sampling_rate)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            _, waves = nk.ecg_delineate(cleaned, rpeaks=r_peaks, sampling_rate=sampling_rate, method="dwt")
        delin = {k: np.array(v, dtype=float) for k, v in waves.items()}
    except Exception:
        return {"r_peaks": np.array([], dtype=int), "delineation": {}, "u_peaks": [], "quality_ok": False}

    t_offsets     = delin.get("ECG_T_Offsets", np.full(len(r_peaks), np.nan))
    p_onsets      = delin.get("ECG_P_Onsets",  np.full(len(r_peaks), np.nan))
    next_p_onsets = np.concatenate([p_onsets[1:], [np.nan]])
    u_peaks = _detect_u_wave(lead_signal, t_offsets, next_p_onsets, sampling_rate)
    return {
        "r_peaks":     r_peaks,
        "delineation": delin,
        "u_peaks":     np.asarray(u_peaks, dtype=float),
        "quality_ok":  True,
    }


def get_concept_labels(
    subseq_start: int,
    subseq_end: int,
    delineation: dict,
    r_peaks: np.ndarray,
    u_peaks,
) -> Tuple[List[int], List[int]]:
    """
    Derive beat concept vector and *beat-level* confidence weights for [subseq_start, subseq_end].

    Fixes the previous record-level weighting bug where a single delineated beat
    in a 10s strip caused failed beats elsewhere in the strip to receive weight=1, label=0.
    """
    n_beats = len(r_peaks)

    def _get(key: str) -> np.ndarray:
        v = delineation.get(key, None)
        if v is None:
            return np.array([], dtype=float)
        return np.asarray(v, dtype=float)

    def in_window(indices) -> bool:
        for idx in indices:
            if not (isinstance(idx, (float, np.floating)) and np.isnan(idx)):
                if subseq_start <= int(idx) <= subseq_end:
                    return True
        return False

    def span_in_window(starts: np.ndarray, ends: np.ndarray) -> bool:
        for s, e in zip(starts, ends):
            if not (np.isnan(s) or np.isnan(e)) and subseq_start <= int(s) and int(e) <= subseq_end:
                return True
        return False

    # Identify beats whose R-peak lies in the window (or adjacent within half-window margin)
    win_len = max(1, subseq_end - subseq_start)
    margin = win_len // 2
    beats_in_win = [b for b, r in enumerate(r_peaks) if subseq_start <= int(r) <= subseq_end]
    nearby_beats = beats_in_win if beats_in_win else [
        b for b, r in enumerate(r_peaks) if (subseq_start - margin) <= int(r) <= (subseq_end + margin)
    ]

    def wave_weight(concept_val: int, arr: np.ndarray) -> int:
        if concept_val == 1:
            return 1
        if n_beats == 0 or len(arr) != n_beats:
            return 0
        if not nearby_beats:
            return 1
        # Concept is 0: only confident if delineation succeeded for all beats overlapping the window
        return int(all(not np.isnan(arr[b]) for b in nearby_beats))

    def span_weight(concept_val: int, starts: np.ndarray, ends: np.ndarray) -> int:
        if concept_val == 1:
            return 1
        if n_beats == 0 or len(starts) != n_beats or len(ends) != n_beats:
            return 0
        if not nearby_beats:
            return 1
        return int(all(not (np.isnan(starts[b]) or np.isnan(ends[b])) for b in nearby_beats))

    p_peaks   = _get("ECG_P_Peaks")
    q_peaks   = _get("ECG_Q_Peaks")
    s_peaks   = _get("ECG_S_Peaks")
    t_peaks   = _get("ECG_T_Peaks")
    p_onsets  = _get("ECG_P_Onsets")
    r_onsets  = _get("ECG_R_Onsets")
    r_offsets = _get("ECG_R_Offsets")
    t_offsets = _get("ECG_T_Offsets")
    u_arr     = np.asarray(u_peaks if u_peaks is not None else [], dtype=float)

    c_P   = int(in_window(p_peaks))
    c_Q   = int(in_window(q_peaks))
    c_R   = int(in_window(r_peaks))
    c_S   = int(in_window(s_peaks))
    c_T   = int(in_window(t_peaks))
    c_U   = int(in_window(u_arr))
    c_PR  = int(span_in_window(p_onsets,  r_onsets))
    c_QRS = int(span_in_window(r_onsets,  r_offsets))
    c_ST  = int(span_in_window(r_offsets, t_offsets))
    c_QT  = int(span_in_window(r_onsets,  t_offsets))

    w_P   = wave_weight(c_P, p_peaks)
    w_Q   = wave_weight(c_Q, q_peaks)
    w_R   = 1 if n_beats > 0 else 0
    w_S   = wave_weight(c_S, s_peaks)
    w_T   = wave_weight(c_T, t_peaks)
    # If u_peaks is None (e.g. ECGdeli/Zenodo soft segmentation), mask U-wave out (w_U=0)
    w_U   = 0 if u_peaks is None else (1 if c_U == 1 else wave_weight(0, t_offsets))
    w_PR  = span_weight(c_PR,  p_onsets,  r_onsets)
    w_QRS = span_weight(c_QRS, r_onsets,  r_offsets)
    w_ST  = span_weight(c_ST,  r_offsets, t_offsets)
    w_QT  = span_weight(c_QT,  r_onsets,  t_offsets)

    return (
        [c_P, c_Q, c_R, c_S, c_T, c_U, c_PR, c_QRS, c_ST, c_QT],
        [w_P, w_Q, w_R, w_S, w_T, w_U, w_PR, w_QRS, w_ST, w_QT],
    )


def is_complete_heartbeat_window(concepts: List[int], weights: List[int]) -> bool:
    """
    Physiological completeness criterion for a window:
      - Ventricular complex + repolarisation present: QT interval (or R + T waves if offsets NaN)
      - Atrial activation present when delineated: PR interval or P wave (waived if P/PR undelineated)
    """
    has_ventricular = bool(concepts[9]) or (bool(concepts[2]) and bool(concepts[4]) and weights[9] == 0)
    has_atrial = bool(concepts[6]) or bool(concepts[0]) or (weights[0] == 0 and weights[6] == 0)
    return has_ventricular and has_atrial


# ─────────────────────────────────────────────────────────────────────────────
# Beat extraction
# ─────────────────────────────────────────────────────────────────────────────

def _window(signal: np.ndarray, start: int, end: int, t_fixed: int) -> np.ndarray:
    C, T = signal.shape
    chunk = signal[:, max(0, start) : min(T, end)]
    if chunk.shape[1] == t_fixed:
        return chunk.astype(np.float32, copy=True)
    out = np.zeros((C, t_fixed), dtype=np.float32)
    out[:, : min(chunk.shape[1], t_fixed)] = chunk[:, :t_fixed]
    return out


def extract_beats(
    signal: np.ndarray,
    delin: dict,
    t_fixed: int = 500,
    beat_window_pre: int = 250,
    beat_window_post: int = 250,
    neg_oversample_factor: float = 1.0,
    rng: Optional[np.random.Generator] = None,
    pos_jitter: int = 60,
    use_masked_norm: bool = False,
) -> List[Tuple[np.ndarray, int, List[int], List[int]]]:
    """
    Extract positive (complete) and negative (incomplete) `t_fixed` windows.

    By default (`use_masked_norm=False`), both positives and negatives are genuine
    continuous `t_fixed`-sample slices normalised identically with `z_normalize`,
    eliminating zero-padding/flatline shortcuts and matching sliding-window inference.
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

    p_onsets  = delineation.get("ECG_P_Onsets",  np.full(len(r_peaks), np.nan))
    t_offsets = delineation.get("ECG_T_Offsets", np.full(len(r_peaks), np.nan))

    # ── Positives ────────────────────────────────────────────────────────────
    for i in range(1, len(r_peaks) - 1):
        r = int(r_peaks[i])
        jitter = int(rng.integers(-pos_jitter, pos_jitter + 1)) if pos_jitter > 0 else 0
        start = r - beat_window_pre + jitter
        end   = start + t_fixed

        # Ensure beat's own delineated P-onset..T-offset remains inside the window
        if i < len(p_onsets) and not np.isnan(p_onsets[i]):
            start = min(start, int(p_onsets[i]) - 5)
            end = start + t_fixed
        if i < len(t_offsets) and not np.isnan(t_offsets[i]):
            if end < int(t_offsets[i]) + 5:
                end = int(t_offsets[i]) + 5
                start = end - t_fixed

        if start < 0 or end > T:
            continue

        subseq = _window(signal, start, end, t_fixed)
        if use_masked_norm:
            prev_r    = int(r_peaks[i - 1])
            next_r    = int(r_peaks[i + 1])
            nat_start = (r + prev_r) // 2
            nat_end   = (r + next_r) // 2
            mask = np.zeros(t_fixed, dtype=np.float32)
            rel_s = int(np.clip(nat_start - start, 0, t_fixed))
            rel_e = int(np.clip(nat_end   - start, 0, t_fixed))
            if rel_e > rel_s:
                mask[rel_s:rel_e] = 1.0
            subseq = _masked_normalize_beat(subseq, mask)
        else:
            subseq = z_normalize(subseq)

        concepts, weights = get_concept_labels(start, end, delineation, r_peaks, u_peaks)
        samples.append((subseq, 1, concepts, weights))

    n_per_strategy = max(1, int(round(max(1, len(samples)) / 4 * neg_oversample_factor)))

    # Strategy 1: early truncation — continuous t_fixed window ending right after R-peak (cuts off T/QT)
    added = 0
    for _ in range(n_per_strategy * 4):
        if added >= n_per_strategy or not len(r_peaks):
            break
        r   = int(r_peaks[rng.integers(0, len(r_peaks))])
        end = r + int(rng.integers(15, max(20, int(0.22 * t_fixed))))
        start = end - t_fixed
        if start < 0 or end > T:
            continue
        concepts, weights = get_concept_labels(start, end, delineation, r_peaks, u_peaks)
        if is_complete_heartbeat_window(concepts, weights):
            continue
        subseq = z_normalize(_window(signal, start, end, t_fixed))
        samples.append((subseq, 0, concepts, weights))
        added += 1

    # Strategy 2: late start — continuous t_fixed window starting at/after Q/R (cuts off P/PR/QRS-onset)
    added = 0
    for _ in range(n_per_strategy * 4):
        if added >= n_per_strategy or not len(r_peaks):
            break
        r     = int(r_peaks[rng.integers(0, len(r_peaks))])
        start = r + int(rng.integers(-20, 45))
        end   = start + t_fixed
        if start < 0 or end > T:
            continue
        concepts, weights = get_concept_labels(start, end, delineation, r_peaks, u_peaks)
        if is_complete_heartbeat_window(concepts, weights):
            continue
        subseq = z_normalize(_window(signal, start, end, t_fixed))
        samples.append((subseq, 0, concepts, weights))
        added += 1

    # Strategy 3: random crop containing no R-peak
    added = 0
    for _ in range(n_per_strategy * 6):
        if added >= n_per_strategy or T < t_fixed:
            break
        start = int(rng.integers(0, max(1, T - t_fixed + 1)))
        end   = start + t_fixed
        if any(start <= int(r) <= end for r in r_peaks):
            continue
        concepts, weights = get_concept_labels(start, end, delineation, r_peaks, u_peaks)
        subseq = z_normalize(_window(signal, start, end, t_fixed))
        samples.append((subseq, 0, concepts, weights))
        added += 1

    # Strategy 4: inter-beat midpoint window (verified not to contain a full tachycardia beat)
    added = 0
    for _ in range(n_per_strategy * 4):
        if added >= n_per_strategy or len(r_peaks) < 2:
            break
        idx   = int(rng.integers(0, len(r_peaks) - 1))
        mid   = (int(r_peaks[idx]) + int(r_peaks[idx + 1])) // 2
        start = mid - t_fixed // 2
        end   = start + t_fixed
        if start < 0 or end > T:
            continue
        concepts, weights = get_concept_labels(start, end, delineation, r_peaks, u_peaks)
        if is_complete_heartbeat_window(concepts, weights):
            continue
        subseq = z_normalize(_window(signal, start, end, t_fixed))
        samples.append((subseq, 0, concepts, weights))
        added += 1

    return samples


def extract_sliding_windows(
    signal: np.ndarray,
    delin: dict,
    t_fixed: int = 500,
    stride: int = 50,
) -> List[Tuple[np.ndarray, int, List[int], List[int]]]:
    """Dense sliding window extraction with per-window z-normalisation and unified completeness label."""
    if not delin.get("quality_ok", False):
        return []
    C, T = signal.shape
    r_peaks     = delin["r_peaks"]
    delineation = delin["delineation"]
    u_peaks     = delin.get("u_peaks", np.array([]))
    samples: list = []
    for start in range(0, T - t_fixed + 1, stride):
        end = start + t_fixed
        subseq = z_normalize(signal[:, start:end].astype(np.float32))
        concepts, weights = get_concept_labels(start, end, delineation, r_peaks, u_peaks)
        task_label = int(is_complete_heartbeat_window(concepts, weights))
        samples.append((subseq, task_label, concepts, weights))
    return samples


# ─────────────────────────────────────────────────────────────────────────────
# PTBXLDataset — full-record diagnostic + concept dataset
# ─────────────────────────────────────────────────────────────────────────────

class PTBXLDataset(Dataset):
    """
    PTB-XL 12-lead ECG dataset for record-level diagnostic classification + 20 clinical concepts.
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
            signal = signal.T.astype(np.float32)
        except Exception:
            T = 5000 if self.sampling_rate == 500 else 1000
            signal = np.zeros((12, T), dtype=np.float32)
        signal = z_normalize(bandpass_filter(signal, self.sampling_rate))
        if self.lead is not None:
            signal = signal[self.lead : self.lead + 1, :]

        age       = np.float32(row["age"])
        sex       = int(row["sex"])
        pathology = row["pathology"].astype(np.float32)

        if self.concepts_df is not None and ecg_id in self.concepts_df.index:
            raw_c = self.concepts_df.reindex(columns=CONCEPT_COLUMNS).loc[ecg_id].values.astype(np.float32)
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
    Each item: {signal (C,T), task_label, concept_labels (K,), concept_weights (K,), record_id}
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
        labels = [int(s["task_label"]) for s in self.samples]
        n1 = sum(labels)
        return {"complete": n1, "incomplete": len(labels) - n1, "total": len(labels)}

    def apply_config_views(self, cfg: dict) -> "HeartbeatDataset":
        """Apply model_lead / model_leads / active_concepts from config in-place."""
        data_cfg = cfg.get("data", {})
        model_leads = data_cfg.get("model_leads", None)
        model_lead  = data_cfg.get("model_lead", None)
        if model_leads is not None:
            self.model_leads = model_leads
        elif model_lead is not None:
            self.model_lead = model_lead
        active_concepts = cfg.get("active_concepts", None)
        if active_concepts is not None:
            self.active_concepts = active_concepts
        return self

    def save_cache(self, cache_dir: str | Path, split: str, meta: Optional[dict] = None) -> None:
        cache_dir = Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        with open(cache_dir / f"{split}.pkl", "wb") as f:
            pickle.dump(self.samples, f, protocol=pickle.HIGHEST_PROTOCOL)
        if meta is not None:
            (cache_dir / "cache_meta.json").write_text(json.dumps(meta, indent=2))
        logger.info(f"Saved {split} cache ({len(self.samples)} samples) → {cache_dir}")

    @classmethod
    def from_cache(cls, cache_dir: str | Path, split: str) -> "HeartbeatDataset":
        path = Path(cache_dir) / f"{split}.pkl"
        with open(path, "rb") as f:
            samples = pickle.load(f)
        logger.info(f"Loaded {split} cache ({len(samples)} samples) from {path}")
        return cls(samples)

    @classmethod
    def from_records(
        cls,
        data_path: str | Path = PTBXL_ROOT,
        sampling_rate: int = 500,
        t_fixed: int = 500,
        beat_window_pre: int = 250,
        beat_window_post: int = 250,
        delineation_cache_dir: Optional[str | Path] = None,
        val_fold: int = 9,
        test_fold: int = 10,
        lead_idx: int = 1,
        neg_oversample_factor: float = 1.0,
        use_sliding_window: bool = False,
        stride: int = 50,
        pos_jitter: int = 60,
        use_masked_norm: bool = False,
        seed: int = 42,
        build_workers: int = 16,
        delineator: str = "neurokit2",
        soft_seg_path: Optional[str | Path] = None,
    ) -> Tuple["HeartbeatDataset", "HeartbeatDataset", "HeartbeatDataset"]:
        """Build train/val/test splits from raw PTB-XL records (parallelised over I/O)."""
        from concurrent.futures import ThreadPoolExecutor

        data_path = Path(data_path)
        meta      = pd.read_csv(data_path / "ptbxl_database.csv", index_col="ecg_id")
        filename_col = "filename_hr" if sampling_rate == 500 else "filename_lr"
        if delineation_cache_dir is not None:
            Path(delineation_cache_dir).mkdir(parents=True, exist_ok=True)

        soft_seg_archive = None
        if delineator == "soft_seg":
            from soft_seg import SoftSegArchive, extract_fiducials_from_soft_seg
            ss_path = Path(soft_seg_path) if soft_seg_path else (data_path.parent / "ptbxl_segmentations_8bit.zip")
            soft_seg_archive = SoftSegArchive(ss_path)
            logger.info(f"Using Zenodo 7610236 (ECGdeli U-Net) soft segmentations from {ss_path} ({len(soft_seg_archive)} records)")

        train_folds = [f for f in range(1, 11) if f not in (val_fold, test_fold)]
        splits = {
            "train": meta[meta["strat_fold"].isin(train_folds)],
            "val":   meta[meta["strat_fold"] == val_fold],
            "test":  meta[meta["strat_fold"] == test_fold],
        }
        all_samples: Dict[str, list] = {k: [] for k in splits}

        def _process_one(item):
            ecg_id, rel_fn = item
            rec_rng = np.random.default_rng(seed + int(ecg_id))
            try:
                if soft_seg_archive is not None:
                    if ecg_id not in soft_seg_archive:
                        return []
                    soft_seg = soft_seg_archive.load(ecg_id)
                    delin_result = extract_fiducials_from_soft_seg(soft_seg, lead_idx=lead_idx)
                    if not delin_result.get("quality_ok", False):
                        return []
                    raw, _ = wfdb.rdsamp(str(data_path / rel_fn))
                    signal = bandpass_filter(raw.T.astype(np.float32), sampling_rate)
                else:
                    raw, _ = wfdb.rdsamp(str(data_path / rel_fn))
                    signal = bandpass_filter(raw.T.astype(np.float32), sampling_rate)
                    delin_result = _load_or_compute_delin(
                        ecg_id, signal, sampling_rate, lead_idx, delineation_cache_dir
                    )
                if use_sliding_window:
                    beats = extract_sliding_windows(signal, delin_result, t_fixed, stride)
                else:
                    beats = extract_beats(
                        signal, delin_result, t_fixed,
                        beat_window_pre, beat_window_post,
                        neg_oversample_factor, rec_rng,
                        pos_jitter=pos_jitter,
                        use_masked_norm=use_masked_norm,
                    )
                return [
                    {
                        "signal":          subseq,
                        "task_label":      task_label,
                        "concept_labels":  concepts,
                        "concept_weights": weights,
                        "record_id":       int(ecg_id),
                    }
                    for subseq, task_label, concepts, weights in beats
                ]
            except Exception as e:
                logger.warning(f"Skipped ecg_id={ecg_id}: {e}")
                return []

        try:
            for split_name, subset in splits.items():
                items = list(zip(subset.index.tolist(), subset[filename_col].tolist()))
                logger.info(f"Processing {split_name} ({len(items)} records, delineator={delineator}, workers={build_workers})…")
                with ThreadPoolExecutor(max_workers=max(1, build_workers)) as pool:
                    for rec_samples in tqdm(pool.map(_process_one, items), total=len(items), desc=split_name):
                        all_samples[split_name].extend(rec_samples)
        finally:
            if soft_seg_archive is not None:
                soft_seg_archive.close()

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
        cache_file = Path(cache_dir) / f"{ecg_id}.pkl"
        if cache_file.exists():
            with open(cache_file, "rb") as f:
                return pickle.load(f)

    result = delineate_record(signal, sampling_rate, lead_idx)

    if cache_dir is not None:
        cdir = Path(cache_dir)
        cdir.mkdir(parents=True, exist_ok=True)
        with open(cdir / f"{ecg_id}.pkl", "wb") as f:
            pickle.dump(result, f, protocol=pickle.HIGHEST_PROTOCOL)
    return result


def load_or_build_datasets(
    cfg: dict,
    splits: Tuple[str, ...] = ("train", "val", "test"),
) -> Dict[str, HeartbeatDataset]:
    """
    Unified dataset loader with config-fingerprinted caching.

    - Stores new caches under `<checkpoint_dir>/dataset_cache/<config_hash>/` so changing
      `lead`, `use_sliding_window`, `use_masked_norm`, etc. never silently loads stale data.
    - Falls back to legacy `<checkpoint_dir>/dataset_cache/<split>.pkl` when evaluating
      older runs whose config lacks `use_masked_norm` and `pos_jitter`.
    """
    data_cfg  = cfg.get("data", {})
    train_cfg = cfg.get("training", {})
    base_cache = Path(train_cfg.get("checkpoint_dir", "checkpoints")) / "dataset_cache"

    is_legacy_cfg = ("use_masked_norm" not in data_cfg) and ("pos_jitter" not in data_cfg)
    cache_key, spec = dataset_cache_key(data_cfg)
    keyed_cache = base_cache / cache_key

    # Check keyed cache first, then legacy root cache for legacy run configs
    if all((keyed_cache / f"{s}.pkl").exists() for s in splits):
        target_cache = keyed_cache
        need_build = False
    elif is_legacy_cfg and all((base_cache / f"{s}.pkl").exists() for s in splits):
        logger.info(f"Using legacy dataset cache at {base_cache} for legacy run config.")
        target_cache = base_cache
        need_build = False
    else:
        target_cache = keyed_cache
        need_build = True

    if not need_build:
        out = {s: HeartbeatDataset.from_cache(target_cache, s).apply_config_views(cfg) for s in splits}
        return out

    data_path, _ = resolve_data_paths(cfg)
    lead_idx = int(data_cfg.get("lead", 1))
    sr       = int(data_cfg.get("sampling_rate", 500))
    # Reuse legacy `delineation/` cache if lead==0 and sr==500, otherwise lead-specific dir
    legacy_delin = base_cache / "delineation"
    if lead_idx == 0 and sr == 500 and legacy_delin.exists():
        delin_dir = legacy_delin
    else:
        delin_dir = base_cache / f"delineation_sr{sr}_lead{lead_idx}"

    delineator = str(data_cfg.get("delineator", "neurokit2")).lower()
    raw_ss = data_cfg.get("soft_seg_path", None)
    if raw_ss:
        data_dir = data_cfg.get("data_dir", None)
        ss_path = Path(raw_ss) if Path(raw_ss).is_absolute() or not data_dir else (Path(data_dir) / raw_ss)
    else:
        ss_path = None

    logger.info(f"Building dataset (cache key={cache_key}, delineator={delineator}) from {data_path}…")
    train_ds, val_ds, test_ds = HeartbeatDataset.from_records(
        data_path             = data_path,
        sampling_rate         = sr,
        t_fixed               = int(data_cfg.get("t_fixed", 500)),
        beat_window_pre       = int(data_cfg.get("beat_window_pre", 250)),
        beat_window_post      = int(data_cfg.get("beat_window_post", 250)),
        delineation_cache_dir = delin_dir,
        val_fold              = int(data_cfg.get("val_fold", 9)),
        test_fold             = int(data_cfg.get("test_fold", 10)),
        lead_idx              = lead_idx,
        neg_oversample_factor = float(data_cfg.get("neg_oversample_factor", 1.0)),
        use_sliding_window    = bool(data_cfg.get("use_sliding_window", False)),
        stride                = int(data_cfg.get("stride", 50)),
        pos_jitter            = int(data_cfg.get("pos_jitter", 60)),
        use_masked_norm       = bool(data_cfg.get("use_masked_norm", False)),
        delineator            = delineator,
        soft_seg_path         = ss_path,
    )
    built = {"train": train_ds, "val": val_ds, "test": test_ds}
    for sp_name, ds_obj in built.items():
        ds_obj.save_cache(keyed_cache, sp_name, meta=spec)
        ds_obj.apply_config_views(cfg)

    return {s: built[s] for s in splits}


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
    common = dict(
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
    )
    return (
        DataLoader(train_ds, shuffle=True,  **common),
        DataLoader(val_ds,   shuffle=False, **common),
        DataLoader(test_ds,  shuffle=False, **common),
    )
