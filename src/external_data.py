"""
external_data.py — External 12-lead ECG dataset loaders, PTB-XL-aligned
preprocessing, SNOMED-CT → superclass mapping, and wave-span clinical interval
extraction for zero-shot OOD evaluation (e.g., Georgia 12-Lead ECG from
explainable-ecg-learning/ecg-data-repo).
"""

from __future__ import annotations

from fractions import Fraction
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import wfdb
from scipy.signal import resample_poly
from torch.utils.data import Dataset

from data import NUM_SUPERCLASSES, SUPERCLASS_MAP, bandpass_filter, z_normalize
from dense_seg import WAVE6_NAMES

CANONICAL_LEADS: List[str] = [
    "I", "II", "III", "AVR", "AVL", "AVF",
    "V1", "V2", "V3", "V4", "V5", "V6",
]

# SNOMED-CT codes mapped to PTB-XL 5 diagnostic superclasses
SNOMED_CD_CODES = frozenset({
    "270492004",  # IAVB (1st degree AV block)
    "195042002",  # IIAVB (2nd degree AV block)
    "27885002",   # CHB (complete heart block)
    "233917008",  # AVB
    "164909002",  # LBBB
    "251120003",  # ILBBB
    "59118001",   # RBBB
    "713427006",  # CRBBB
    "713426002",  # IRBBB
    "445118002",  # LAnFB
    "445211001",  # LPFB
    "698252002",  # NSIVCB
    "6374002",    # BBB
    "74390002",   # WPW
})

SNOMED_STTC_CODES = frozenset({
    "164934002",  # TAb
    "59931005",   # TInv
    "428750005",  # NSSTTA
    "429622005",  # STD
    "164931005",  # STE
    "164930006",  # STIAb
    "55930002",   # STC
    "426434006",  # AnMIs
    "425419005",  # IIs
    "425623009",  # LIs
    "164861001",  # MIs
    "413444003",  # AMIs
    "111975006",  # LQT
})

SNOMED_HYP_CODES = frozenset({
    "164873001",       # LVH
    "89792004",        # RVH
    "266249003",       # VH
    "67741000119109",  # LAE
    "195126007",       # AH
    "253352002",       # LAA
    "253339007",       # RAAb
    "446358003",       # RAH
    "446813000",       # LAH
})

SNOMED_PURE_MI_CODES = frozenset({
    "164865005",  # MI
    "164867002",  # OldMI
    "57054005",   # AMI
    "54329005",   # AnMI
})

SNOMED_MI_CODES = SNOMED_PURE_MI_CODES | frozenset({"164917005"})  # + QAb (abnormal Q-wave)

SNOMED_NSR_CODE = "426783006"
SNOMED_MAJOR_ARRHYTHMIA_CODES = frozenset({
    "164889003",  # AF
    "164890007",  # AFL
    "17338001",   # VPB
    "284470004",  # PAC
    "426761007",  # SVT
    "164895002",  # VTach
    "251266004",  # VPP (ventricular pacing)
    "251268003",  # AP (atrial pacing)
})

# Canonical cardiac-cycle adjacency for wave6 = {0:P, 1:PQ, 2:QRS, 3:ST, 4:T, 5:TP}
STRICT_FORWARD_TRANSITIONS = frozenset({
    (0, 1), (0, 2), (1, 2), (2, 3), (2, 4),
    (3, 4), (4, 5), (4, 0), (5, 0), (5, 2),
})
CANONICAL_TRANSITIONS = STRICT_FORWARD_TRANSITIONS | frozenset({(1, 5), (0, 5)})


def _parse_code_collection(codes: Sequence[str] | str | None) -> List[str]:
    if codes is None:
        return []
    if isinstance(codes, str):
        s = codes.strip()
        if not s or s.lower() == "nan":
            return []
        sep = "|" if "|" in s else ","
        return [c.strip() for c in s.split(sep) if c.strip()]
    return [str(c).strip() for c in codes if str(c).strip()]


def canonicalize_leads(signal: np.ndarray, sig_names: Sequence[str]) -> np.ndarray:
    """
    Reorder a 12-lead ECG array (12, T) so channel order matches PTB-XL canonical
    order: [I, II, III, AVR, AVL, AVF, V1, V2, V3, V4, V5, V6].
    Case-insensitive matching on lead names.
    """
    arr = np.asarray(signal)
    if arr.ndim != 2 or arr.shape[0] != len(sig_names):
        raise ValueError(
            f"Expected signal shape (n_leads, T) matching len(sig_names)={len(sig_names)}, "
            f"got {arr.shape}"
        )
    name_to_idx: Dict[str, int] = {}
    for i, name in enumerate(sig_names):
        key = str(name).strip().upper()
        if key not in name_to_idx:
            name_to_idx[key] = i

    missing = [lead for lead in CANONICAL_LEADS if lead not in name_to_idx]
    if missing:
        raise ValueError(f"Missing canonical ECG leads: {missing} (got {list(sig_names)})")

    order = [name_to_idx[lead] for lead in CANONICAL_LEADS]
    return arr[order].astype(np.float32, copy=False)


def preprocess_to_dense_100hz(
    signal: np.ndarray,
    fs: int = 500,
    target_sr: int = 100,
    target_sec: float = 10.0,
) -> np.ndarray:
    """
    Convert a raw canonical (12, T) recording at `fs` Hz into a PTB-XL-matched
    (12, target_sec * target_sr) float32 tensor at `target_sr` Hz using:
      1. NaN/Inf sanitization
      2. 10.0 s duration standardization (center-crop or edge-pad)
      3. Anti-aliased polyphase resampling (scipy.signal.resample_poly)
      4. Zero-phase 0.5-40 Hz Butterworth bandpass + per-lead z-normalization
    """
    sig = np.nan_to_num(np.asarray(signal, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    if sig.ndim != 2 or sig.shape[0] != len(CANONICAL_LEADS):
        raise ValueError(f"Expected canonical shape (12, T), got {sig.shape}")

    n_src_target = int(round(target_sec * fs))
    cur_len = sig.shape[-1]
    if cur_len > n_src_target:
        start = (cur_len - n_src_target) // 2
        sig = sig[:, start : start + n_src_target]
    elif cur_len < n_src_target:
        pad_total = n_src_target - cur_len
        pad_left = pad_total // 2
        pad_right = pad_total - pad_left
        sig = np.pad(sig, ((0, 0), (pad_left, pad_right)), mode="edge")

    if fs != target_sr:
        frac = Fraction(int(target_sr), int(fs))
        sig_rs = resample_poly(sig, frac.numerator, frac.denominator, axis=-1).astype(np.float32)
    else:
        sig_rs = sig

    n_dst_target = int(round(target_sec * target_sr))
    if sig_rs.shape[-1] > n_dst_target:
        sig_rs = sig_rs[:, :n_dst_target]
    elif sig_rs.shape[-1] < n_dst_target:
        sig_rs = np.pad(sig_rs, ((0, 0), (0, n_dst_target - sig_rs.shape[-1])), mode="edge")

    sig_filt = bandpass_filter(sig_rs, sampling_rate=target_sr, lowcut=0.5, highcut=40.0)
    return z_normalize(sig_filt)


def map_snomed_to_superclasses(
    snomed_codes: Sequence[str] | str,
    include_qab_in_mi: bool = True,
) -> np.ndarray:
    """
    Map CinC/Georgia SNOMED-CT diagnosis codes into PTB-XL 5-superclass binary
    targets: [NORM, MI, STTC, CD, HYP].
    NORM is assigned iff Sinus Rhythm (426783006) is present with NO structural/
    conduction/ischemic/hypertrophy code and NO major arrhythmia code.
    """
    codes = set(_parse_code_collection(snomed_codes))
    out = np.zeros(NUM_SUPERCLASSES, dtype=np.float32)

    mi_codes = SNOMED_MI_CODES if include_qab_in_mi else SNOMED_PURE_MI_CODES
    if codes & mi_codes:
        out[SUPERCLASS_MAP["MI"]] = 1.0
    if codes & SNOMED_STTC_CODES:
        out[SUPERCLASS_MAP["STTC"]] = 1.0
    if codes & SNOMED_CD_CODES:
        out[SUPERCLASS_MAP["CD"]] = 1.0
    if codes & SNOMED_HYP_CODES:
        out[SUPERCLASS_MAP["HYP"]] = 1.0

    has_pathology = out[1:].sum() > 0
    has_arrhythmia = bool(codes & SNOMED_MAJOR_ARRHYTHMIA_CODES)
    if (SNOMED_NSR_CODE in codes) and not has_pathology and not has_arrhythmia:
        out[SUPERCLASS_MAP["NORM"]] = 1.0

    return out


def extract_wave_spans_and_intervals(
    cls_seq: np.ndarray,
    fs: int = 100,
) -> Dict[str, Any]:
    """
    Extract contiguous wave spans and clinical ECG intervals from a predicted
    1D `wave6` hard segmentation sequence of shape (T,) with labels
    {0:P, 1:PQ, 2:QRS, 3:ST, 4:T, 5:TP}.
    """
    seq = np.asarray(cls_seq, dtype=np.int64).ravel()
    T = int(seq.size)
    dt_ms = 1000.0 / float(fs)
    min_glitch_samples = max(1, int(round(0.02 * fs)))  # <20 ms noise glitch filter for P/QRS/T

    spans: Dict[str, List[Tuple[int, int, float]]] = {name: [] for name in WAVE6_NAMES}
    if T > 0:
        changes = np.flatnonzero(seq[1:] != seq[:-1]) + 1
        starts = np.concatenate(([0], changes))
        ends = np.concatenate((changes, [T]))
        for s, e in zip(starts.tolist(), ends.tolist()):
            cls_idx = int(seq[s])
            if 0 <= cls_idx < len(WAVE6_NAMES):
                cname = WAVE6_NAMES[cls_idx]
                if cname in ("P", "QRS", "T") and (e - s) < min_glitch_samples:
                    continue
                center = 0.5 * (s + e - 1)
                spans[cname].append((s, e, center))

    # Duty cycles across full timeline
    duty_cycles = {
        name: float((seq == idx).mean()) if T > 0 else 0.0
        for idx, name in enumerate(WAVE6_NAMES)
    }

    # Transition ordering consistency
    if T > 1:
        diff_mask = seq[1:] != seq[:-1]
        n_trans = int(diff_mask.sum())
        if n_trans > 0:
            src = seq[:-1][diff_mask].tolist()
            dst = seq[1:][diff_mask].tolist()
            n_canon = sum((a, b) in CANONICAL_TRANSITIONS for a, b in zip(src, dst))
            n_strict = sum((a, b) in STRICT_FORWARD_TRANSITIONS for a, b in zip(src, dst))
            canon_rate = float(n_canon / n_trans)
            strict_rate = float(n_strict / n_trans)
        else:
            canon_rate = 1.0
            strict_rate = 1.0
    else:
        n_trans = 0
        canon_rate = 1.0
        strict_rate = 1.0

    # Valid beats = QRS spans within physiological beat width [40 ms, 300 ms]
    qrs_beats = [
        (s, e, c)
        for (s, e, c) in spans["QRS"]
        if 40.0 <= (e - s) * dt_ms <= 300.0
    ]
    n_beats = len(qrs_beats)
    if n_beats >= 2:
        rr_sec = np.diff([b[2] for b in qrs_beats]) / float(fs)
        med_rr = float(np.median(rr_sec))
        pred_hr_bpm = float(60.0 / med_rr) if med_rr > 1e-6 else float("nan")
    else:
        pred_hr_bpm = float("nan")

    # Beat-anchored intervals
    p_spans = spans["P"]
    t_spans = spans["T"]
    qrs_durs_ms: List[float] = []
    pr_list_ms: List[float] = []
    qt_list_ms: List[float] = []
    beats_with_p = 0

    for bi, (qs, qe, _) in enumerate(qrs_beats):
        dur_ms = (qe - qs) * dt_ms
        if 40.0 <= dur_ms <= 260.0:
            qrs_durs_ms.append(dur_ms)

        prev_q_end = qrs_beats[bi - 1][1] if bi > 0 else -1
        next_q_start = qrs_beats[bi + 1][0] if (bi + 1) < n_beats else T + 1

        # Preceding P-wave (no intervening QRS, onset within 350 ms before QRS onset)
        cand_p = [
            (ps, pe, pc)
            for (ps, pe, pc) in p_spans
            if ps >= prev_q_end and ps < qs and 0.0 < (qs - ps) * dt_ms <= 350.0
        ]
        if cand_p:
            beats_with_p += 1
            ps, _, _ = cand_p[-1]
            pr_val = (qs - ps) * dt_ms
            if 80.0 <= pr_val <= 400.0:
                pr_list_ms.append(pr_val)

        # Following T-wave (no intervening QRS, T offset within 600 ms after QRS onset)
        cand_t = [
            (ts, te, tc)
            for (ts, te, tc) in t_spans
            if te <= next_q_start and ts >= qe - int(round(0.02 * fs)) and 0.0 < (te - qs) * dt_ms <= 600.0
        ]
        if cand_t:
            _, te, _ = cand_t[0]
            qt_val = (te - qs) * dt_ms
            if 200.0 <= qt_val <= 650.0:
                qt_list_ms.append(qt_val)

    return {
        "spans": spans,
        "valid_qrs_beats": qrs_beats,
        "n_beats": n_beats,
        "pred_hr_bpm": pred_hr_bpm,
        "duty_cycles": duty_cycles,
        "n_transitions": n_trans,
        "canonical_transition_rate": canon_rate,
        "strict_forward_transition_rate": strict_rate,
        "qrs_ms": float(np.median(qrs_durs_ms)) if qrs_durs_ms else float("nan"),
        "pr_ms": float(np.median(pr_list_ms)) if pr_list_ms else float("nan"),
        "qt_ms": float(np.median(qt_list_ms)) if qt_list_ms else float("nan"),
        "p_present_beat_frac": float(beats_with_p / n_beats) if n_beats > 0 else float("nan"),
    }


def load_georgia_metadata(
    georgia_root: str | Path = "data/georgia_ecg",
    max_error_severity: Optional[int] = 0,
    severity_tiers: Optional[Iterable[str]] = None,
    require_10s_500hz: bool = True,
) -> pd.DataFrame:
    """
    Load audited metadata table from Georgia ECG (`quality/error_labels.csv`) and
    filter to valid records.
    """
    root = Path(georgia_root)
    csv_path = root / "quality" / "error_labels.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Georgia error_labels.csv not found at {csv_path}")

    df = pd.read_csv(csv_path)
    mask = df["err_file_missing_or_corrupt"].fillna(1).astype(int) == 0
    if require_10s_500hz:
        mask &= (df["fs"].fillna(0).astype(int) == 500) & (df["n_samples"].fillna(0).astype(int) == 5000)

    if severity_tiers is not None:
        allowed = set(severity_tiers)
        mask &= df["severity_tier"].astype(str).isin(allowed)
    elif max_error_severity is not None:
        tier_num = df["severity_tier"].astype(str).str.slice(0, 1).astype(int)
        mask &= tier_num <= int(max_error_severity)

    sub = df.loc[mask].copy().reset_index(drop=True)
    sub["snomed_codes"] = sub["snomed_codes"].fillna("").astype(str)
    sub["label_names"] = sub["label_names"].fillna("").astype(str)
    return sub


class GeorgiaDenseDataset(Dataset):
    """
    PyTorch Dataset serving Georgia 12-Lead ECG recordings aligned to PTB-XL's
    canonical lead order and 100 Hz (or optional raw 500 Hz Lead II) representation.
    """

    def __init__(
        self,
        metadata_df: pd.DataFrame,
        georgia_root: str | Path = "data/georgia_ecg",
        target_sr: int = 100,
        return_500hz_lead2: bool = False,
        include_qab_in_mi: bool = True,
    ) -> None:
        self.df = metadata_df.reset_index(drop=True)
        self.data_dir = Path(georgia_root) / "data"
        self.target_sr = int(target_sr)
        self.return_500hz_lead2 = bool(return_500hz_lead2)
        self.include_qab_in_mi = bool(include_qab_in_mi)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self.df.iloc[idx]
        rec_path = str(self.data_dir / str(row["record_path"]))
        raw, fields = wfdb.rdsamp(rec_path)
        fs = int(fields.get("fs", row.get("fs", 500)))
        sig_names = fields.get("sig_name", CANONICAL_LEADS)

        canon = canonicalize_leads(raw.T, sig_names)  # (12, T) @ fs
        sig_100 = preprocess_to_dense_100hz(canon, fs=fs, target_sr=self.target_sr, target_sec=10.0)
        pathology = map_snomed_to_superclasses(row["snomed_codes"], include_qab_in_mi=self.include_qab_in_mi)

        item: Dict[str, Any] = {
            "signal_100": torch.from_numpy(sig_100),
            "pathology": torch.from_numpy(pathology),
            "record_id": str(row["record_id"]),
            "snomed_codes": str(row["snomed_codes"]),
            "label_names": str(row["label_names"]),
            "est_hr_bpm": float(row["est_hr_bpm"]) if pd.notna(row["est_hr_bpm"]) else float("nan"),
            "severity_tier": str(row["severity_tier"]),
        }
        if self.return_500hz_lead2:
            lead2 = np.nan_to_num(canon[1].astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
            item["signal_500_lead2"] = torch.from_numpy(lead2)
        return item
