"""
soft_seg.py — Loader and cross-delineator benchmark for PTB-XL Soft Segmentations
(Zenodo 7610236, U-Net distilled from KIT-IBT/ECGdeli).

Reads (1000, 12, 24) uint8 soft segmentation arrays directly from
`ptbxl_segmentations_8bit.zip` (or an extracted folder) without unpacking 21k files,
maps the 24 ECGdeli/U-Net classes to our 10 heartbeat CBM/CEM concepts, and runs:
  1. Inter-delineator agreement: NeuroKit2 vs. Zenodo 7610236 (ECGdeli U-Net)
  2. Zero-shot cross-delineator evaluation of NeuroKit2-trained CBM & CEM against
     Zenodo 7610236 / ECGdeli labels on PTB-XL Test Fold 10
  3. Sliding-window visual comparison: ECG waveform + Zenodo 24-class soft segmentation
     heatmap/curves + CBM vs. CEM sliding-window concept activations

CLI:
    python src/soft_seg.py --zip /nfs/data8/schlegel/git/ecg-cbm/data/ptbxl_segmentations_8bit.zip \
        --runs runs/cbm_v2 runs/cbm_randint_v2 runs/cem_v2 --n-records 200 --out-dir runs/soft_seg_eval
"""

from __future__ import annotations

import argparse
import io
import json
import logging
import threading
import zipfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import wfdb
import yaml
from scipy.signal import find_peaks
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from tqdm import tqdm

from data import (
    bandpass_filter,
    delineate_record,
    get_concept_labels,
    is_complete_heartbeat_window,
    resolve_data_paths,
    z_normalize,
)
from models import CONCEPT_NAMES, build_model

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# Official 24 classes from Zenodo 7610236 README (Wagner, Mehari et al.; ECGdeli labels)
SOFT_SEG_CLASSES: List[str] = [
    "P-Onset", "(p", "P-Peak", "p)", "P-Offset", "pq",
    "QRS-Onset", "(q", "Q-Peak", "qr", "R-Peak", "rs", "S-Peak", "q)", "QRS-Offset",
    "qj", "J-point", "jt",
    "T-Onset", "(t", "T-Peak", "t)", "T-Offset", "tp",
]
SOFT_SEG_IDX: Dict[str, int] = {name: i for i, name in enumerate(SOFT_SEG_CLASSES)}

# Wave body channel groups (for per-timestep soft segmentation visualization)
WAVE_CHANNEL_GROUPS: Dict[str, List[int]] = {
    "P":   [0, 1, 2, 3, 4],           # P-Onset .. P-Offset
    "QRS": [6, 7, 8, 9, 10, 11, 12, 13, 14],  # QRS-Onset .. QRS-Offset
    "ST":  [15, 16, 17],              # qj, J-point, jt
    "T":   [18, 19, 20, 21, 22],      # T-Onset .. T-Offset
}

# Concepts shared between our 10-concept CBM/CEM and Zenodo/ECGdeli (U-wave is absent in ECGdeli)
SHARED_CONCEPT_INDICES: List[int] = [0, 1, 2, 3, 4, 6, 7, 8, 9]


# ─────────────────────────────────────────────────────────────────────────────
# Fast Zip / Directory Loader for Zenodo 7610236
# ─────────────────────────────────────────────────────────────────────────────

class SoftSegArchive:
    """Lazy reader for Zenodo 7610236 (`ptbxl_segmentations_8bit.zip` or extracted folder)."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._zf: Optional[zipfile.ZipFile] = None
        self._lock = threading.Lock()
        self._member_map: Dict[int, str] = {}

        if self.path.is_file() and self.path.suffix == ".zip":
            self._zf = zipfile.ZipFile(self.path, "r")
            for name in self._zf.namelist():
                if name.endswith("/") or name.startswith("__MACOSX"):
                    continue
                stem = Path(name).stem
                # Filenames may be "00009.npy", "9.npy", "00009_lr.npy", etc.
                digits = "".join(ch for ch in stem.split("_")[0] if ch.isdigit())
                if digits:
                    self._member_map[int(digits)] = name
        elif self.path.is_dir():
            for p in self.path.rglob("*.npy"):
                digits = "".join(ch for ch in p.stem.split("_")[0] if ch.isdigit())
                if digits:
                    self._member_map[int(digits)] = str(p)
        else:
            raise FileNotFoundError(f"Soft segmentation archive not found: {self.path}")

    def __contains__(self, ecg_id: int) -> bool:
        return int(ecg_id) in self._member_map

    def __len__(self) -> int:
        return len(self._member_map)

    def load(self, ecg_id: int) -> np.ndarray:
        """Load float32 soft segmentation probabilities in [0, 1] of shape (1000, 12, 24)."""
        key = int(ecg_id)
        if key not in self._member_map:
            raise KeyError(f"ecg_id={key} not in soft segmentation archive")
        member = self._member_map[key]
        if self._zf is not None:
            with self._lock:
                raw_bytes = self._zf.read(member)
            arr = np.load(io.BytesIO(raw_bytes), allow_pickle=True)
        else:
            arr = np.load(member, allow_pickle=True)
        arr = np.asarray(arr)
        if arr.dtype == np.uint8:
            return arr.astype(np.float32) / 255.0
        return arr.astype(np.float32)

    def close(self) -> None:
        if self._zf is not None:
            self._zf.close()
            self._zf = None


# ─────────────────────────────────────────────────────────────────────────────
# Mapping Zenodo 7610236 (ECGdeli U-Net) Soft Masks → Fiducials & Window Concepts
# ─────────────────────────────────────────────────────────────────────────────

def soft_seg_to_wave_curves(soft_seg: np.ndarray, lead_idx: Optional[int] = None) -> Dict[str, np.ndarray]:
    """
    Collapse (1000, 12, 24) soft probabilities into per-timestep wave envelopes (1000,)
    for P, QRS, ST, T (either for a single lead or 12-lead mean consensus).
    """
    seg = soft_seg[:, lead_idx, :] if lead_idx is not None else soft_seg.mean(axis=1)  # (1000, 24)
    return {
        name: np.clip(seg[:, idxs].sum(axis=-1), 0.0, 1.0)
        for name, idxs in WAVE_CHANNEL_GROUPS.items()
    }


def extract_fiducials_from_soft_seg(
    soft_seg: np.ndarray,
    lead_idx: Optional[int] = 0,
    peak_thresh: float = 0.15,
    min_rr_samples: int = 30,
) -> dict:
    """
    Extract ECGdeli-style beat-aligned fiducials from a (1000, 12, 24) soft segmentation
    at 100 Hz and rescale sample indices ×5 to 500 Hz coordinates for direct comparison
    with NeuroKit2 and 500 Hz sliding windows.
    """
    # Use 12-lead consensus for R-peaks (like multi-lead ECGdeli), and chosen lead (or consensus) for waves
    seg_all = soft_seg.mean(axis=1)  # (1000, 24)
    seg_ref = soft_seg[:, lead_idx, :] if lead_idx is not None else seg_all

    r_prob = seg_all[:, SOFT_SEG_IDX["R-Peak"]]
    r_peaks_100, _ = find_peaks(r_prob, height=peak_thresh, distance=min_rr_samples)
    if len(r_peaks_100) < 2:
        return {"r_peaks": np.array([], dtype=int), "delineation": {}, "u_peaks": np.array([]), "quality_ok": False}

    def _find_near(channel_name: str, lo_off: int, hi_off: int, use_all: bool = False) -> np.ndarray:
        ch = (seg_all if use_all else seg_ref)[:, SOFT_SEG_IDX[channel_name]]
        out = np.full(len(r_peaks_100), np.nan, dtype=float)
        for b, r in enumerate(r_peaks_100):
            s = max(0, int(r) + lo_off)
            e = min(len(ch), int(r) + hi_off + 1)
            if e <= s:
                continue
            seg_win = ch[s:e]
            idx_max = int(np.argmax(seg_win))
            if float(seg_win[idx_max]) >= peak_thresh * 0.65:
                out[b] = float((s + idx_max) * 5)  # scale 100 Hz → 500 Hz
        return out

    delin = {
        "ECG_P_Onsets":  _find_near("P-Onset",    -32, -8),
        "ECG_P_Peaks":   _find_near("P-Peak",     -28, -6),
        "ECG_Q_Peaks":   _find_near("Q-Peak",     -10,  0),
        "ECG_R_Onsets":  _find_near("QRS-Onset",  -12,  0, use_all=True),
        "ECG_S_Peaks":   _find_near("S-Peak",       0, 10),
        "ECG_R_Offsets": _find_near("QRS-Offset",   0, 14, use_all=True),
        "ECG_T_Peaks":   _find_near("T-Peak",       8, 45),
        "ECG_T_Offsets": _find_near("T-Offset",    14, 58, use_all=True),
    }

    return {
        "r_peaks":     (r_peaks_100 * 5).astype(int),
        "delineation": delin,
        "u_peaks":     None,  # ECGdeli/Zenodo has no U-wave class → masked out via w_U=0
        "quality_ok":  True,
    }


def soft_window_concept_probs(
    soft_seg: np.ndarray,
    start_500: int,
    end_500: int,
    lead_idx: Optional[int] = 0,
) -> np.ndarray:
    """
    Compute continuous [0, 1] soft window-level concept scores directly from the
    (1000, 12, 24) U-Net probability map (without hard thresholding) for visualization.
    Returns vector of length 10 aligned with CONCEPT_NAMES.
    """
    s = max(0, start_500 // 5)
    e = min(soft_seg.shape[0], end_500 // 5)
    seg_ref = (soft_seg[s:e, lead_idx, :] if lead_idx is not None else soft_seg[s:e].mean(axis=1))
    seg_all = soft_seg[s:e].mean(axis=1)

    if seg_ref.shape[0] == 0:
        return np.zeros(10, dtype=np.float32)

    p_max   = float(seg_ref[:, SOFT_SEG_IDX["P-Peak"]].max())
    q_max   = float(seg_ref[:, SOFT_SEG_IDX["Q-Peak"]].max())
    r_max   = float(seg_all[:, SOFT_SEG_IDX["R-Peak"]].max())
    s_max   = float(seg_ref[:, SOFT_SEG_IDX["S-Peak"]].max())
    t_max   = float(seg_ref[:, SOFT_SEG_IDX["T-Peak"]].max())
    u_max   = 0.0  # not annotated by ECGdeli/Zenodo

    # Soft interval presence: min(max_onset, max_offset) respecting temporal ordering
    def _ordered_span_prob(ch_start: str, ch_end: str, use_all: bool = False) -> float:
        src = seg_all if use_all else seg_ref
        a = src[:, SOFT_SEG_IDX[ch_start]]
        b = src[:, SOFT_SEG_IDX[ch_end]]
        # For every onset time t1, best offset at t2 > t1 (prefix max of a paired with b)
        pref_a = np.maximum.accumulate(a)
        if len(b) < 2:
            return 0.0
        return float(np.minimum(pref_a[:-1], b[1:]).max())

    pr_p  = _ordered_span_prob("P-Onset",    "QRS-Onset",  use_all=False)
    qrs_p = _ordered_span_prob("QRS-Onset",  "QRS-Offset", use_all=True)
    st_p  = _ordered_span_prob("QRS-Offset", "T-Offset",   use_all=False)
    qt_p  = _ordered_span_prob("QRS-Onset",  "T-Offset",   use_all=True)

    return np.array([p_max, q_max, r_max, s_max, t_max, u_max, pr_p, qrs_p, st_p, qt_p], dtype=np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Cross-Delineator Benchmark & Visualization CLI
# ─────────────────────────────────────────────────────────────────────────────

def _safe_auroc(y: np.ndarray, p: np.ndarray) -> float:
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(roc_auc_score(y, p))


def run_benchmark(
    zip_path: Path,
    run_dirs: List[Path],
    cfg_path: Path,
    n_records: int,
    stride: int,
    out_dir: Path,
    device: str,
) -> dict:
    with open(cfg_path) as f:
        base_cfg = yaml.safe_load(f)
    data_path, _ = resolve_data_paths(base_cfg)
    sr       = int(base_cfg["data"].get("sampling_rate", 500))
    t_fixed  = int(base_cfg["data"].get("t_fixed", 500))
    lead_idx = int(base_cfg["data"].get("lead", 0))
    delin_cache = Path(base_cfg["training"].get("checkpoint_dir", "checkpoints")) / "dataset_cache" / "delineation"

    meta = pd.read_csv(data_path / "ptbxl_database.csv", index_col="ecg_id")
    test_ids = meta[meta["strat_fold"] == 10].index.tolist()

    archive = SoftSegArchive(zip_path)
    logger.info(f"Opened Zenodo soft segmentation archive: {len(archive)} records")

    # Load all requested CBM/CEM models
    models: Dict[str, torch.nn.Module] = {}
    for rdir in run_dirs:
        with open(rdir / "config.yaml") as f:
            mcfg = yaml.safe_load(f)
        m = build_model(mcfg).to(device)
        st = torch.load(rdir / "checkpoints" / "best_model.pt", map_location=device, weights_only=False)
        m.load_state_dict(st["model"])
        m.eval()
        models[rdir.name] = m

    nk_task_all, ss_task_all = [], []
    nk_conc_all, nk_w_all = [], []
    ss_conc_all, ss_w_all = [], []
    model_task_preds: Dict[str, List[np.ndarray]] = {k: [] for k in models}
    model_conc_preds: Dict[str, List[np.ndarray]] = {k: [] for k in models}

    processed = 0
    for ecg_id in tqdm(test_ids, desc="Cross-delineator eval"):
        if processed >= n_records:
            break
        if ecg_id not in archive:
            continue
        try:
            soft_seg = archive.load(ecg_id)
            ss_delin = extract_fiducials_from_soft_seg(soft_seg, lead_idx=lead_idx)
            if not ss_delin["quality_ok"]:
                continue

            fn = meta.loc[ecg_id, "filename_hr"]
            raw, _ = wfdb.rdsamp(str(data_path / fn))
            sig = bandpass_filter(raw.T.astype(np.float32), sampling_rate=sr)

            # Load cached NeuroKit2 delineation if present
            pkl = delin_cache / f"{ecg_id}.pkl"
            if pkl.exists():
                import pickle
                with open(pkl, "rb") as f:
                    nk_delin = pickle.load(f)
            else:
                nk_delin = delineate_record(sig, sampling_rate=sr, lead_idx=lead_idx)
            if not nk_delin.get("quality_ok", False):
                continue

            starts = list(range(0, sig.shape[1] - t_fixed + 1, stride))
            windows = np.stack([z_normalize(sig[:, s : s + t_fixed]) for s in starts], axis=0).astype(np.float32)

            nk_c, nk_w, nk_t = [], [], []
            ss_c, ss_w, ss_t = [], [], []
            for s in starts:
                c1, w1 = get_concept_labels(s, s + t_fixed, nk_delin["delineation"], nk_delin["r_peaks"], nk_delin.get("u_peaks"))
                c2, w2 = get_concept_labels(s, s + t_fixed, ss_delin["delineation"], ss_delin["r_peaks"], ss_delin.get("u_peaks"))
                nk_c.append(c1); nk_w.append(w1); nk_t.append(int(is_complete_heartbeat_window(c1, w1)))
                ss_c.append(c2); ss_w.append(w2); ss_t.append(int(is_complete_heartbeat_window(c2, w2)))

            nk_task_all.append(np.array(nk_t)); ss_task_all.append(np.array(ss_t))
            nk_conc_all.append(np.array(nk_c)); nk_w_all.append(np.array(nk_w))
            ss_conc_all.append(np.array(ss_c)); ss_w_all.append(np.array(ss_w))

            xb = torch.from_numpy(windows).to(device)
            with torch.no_grad():
                for name, m in models.items():
                    tp, cp, _ = m(xb)
                    model_task_preds[name].append(tp.squeeze(-1).cpu().numpy())
                    model_conc_preds[name].append(cp.cpu().numpy())

            processed += 1
        except Exception as exc:
            logger.warning(f"Skipping ecg_id={ecg_id}: {exc}")

    archive.close()

    nk_t = np.concatenate(nk_task_all)
    ss_t = np.concatenate(ss_task_all)
    nk_c = np.concatenate(nk_conc_all, axis=0)
    nk_w = np.concatenate(nk_w_all, axis=0)
    ss_c = np.concatenate(ss_conc_all, axis=0)
    ss_w = np.concatenate(ss_w_all, axis=0)

    def _score_against(target_t, target_c, target_w, pred_t, pred_c) -> dict:
        out = {
            "task/accuracy": float(accuracy_score(target_t, (pred_t > 0.5).astype(int))),
            "task/f1":       float(f1_score(target_t, (pred_t > 0.5).astype(int), zero_division=0)),
            "task/auroc":    _safe_auroc(target_t, pred_t),
        }
        f1s = []
        for k in SHARED_CONCEPT_INDICES:
            valid = target_w[:, k] > 0
            if valid.sum() < 10:
                continue
            fk = float(f1_score(target_c[valid, k], (pred_c[valid, k] > 0.5).astype(int), zero_division=0))
            out[f"concept/{CONCEPT_NAMES[k]}/f1"] = fk
            out[f"concept/{CONCEPT_NAMES[k]}/auroc"] = _safe_auroc(target_c[valid, k], pred_c[valid, k])
            f1s.append(fk)
        out["concept/shared_macro_f1"] = float(np.mean(f1s)) if f1s else float("nan")
        return out

    # 1. Inter-delineator agreement: NeuroKit2 vs Zenodo/ECGdeli
    inter_delin = {
        "n_records": processed,
        "n_windows": int(len(nk_t)),
        "task_agreement_acc": float(accuracy_score(ss_t, nk_t)),
        "task_agreement_f1":  float(f1_score(ss_t, nk_t, zero_division=0)),
        "per_concept_f1": {},
    }
    for k in SHARED_CONCEPT_INDICES:
        valid = (nk_w[:, k] > 0) & (ss_w[:, k] > 0)
        if valid.sum() >= 10:
            inter_delin["per_concept_f1"][CONCEPT_NAMES[k]] = float(
                f1_score(ss_c[valid, k], nk_c[valid, k], zero_division=0)
            )
    inter_delin["shared_macro_f1"] = float(np.mean(list(inter_delin["per_concept_f1"].values())))

    summary = {"inter_delineator_nk_vs_ecgdeli": inter_delin, "models": {}}
    for name in models:
        pt = np.concatenate(model_task_preds[name])
        pc = np.concatenate(model_conc_preds[name], axis=0)
        summary["models"][name] = {
            "vs_neurokit2":     _score_against(nk_t, nk_c, nk_w, pt, pc),
            "vs_zenodo_ecgdeli": _score_against(ss_t, ss_c, ss_w, pt, pc),
        }

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "soft_seg_comparison.json").write_text(json.dumps(summary, indent=2))
    logger.info(f"Saved cross-delineator benchmark → {out_dir / 'soft_seg_comparison.json'}")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare CBM/CEM against Zenodo 7610236 (ECGdeli U-Net) soft segmentations")
    parser.add_argument("--zip",       required=True, help="Path to ptbxl_segmentations_8bit.zip or extracted folder")
    parser.add_argument("--runs",      nargs="+", required=True, help="Trained run directories to evaluate")
    parser.add_argument("--config",    default="config.yaml")
    parser.add_argument("--n-records", type=int, default=200, help="Number of test-fold records to benchmark")
    parser.add_argument("--stride",    type=int, default=25,  help="Sliding-window stride (samples @ 500 Hz)")
    parser.add_argument("--out-dir",   default="runs/soft_seg_eval")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    res = run_benchmark(
        zip_path=Path(args.zip),
        run_dirs=[Path(r) for r in args.runs],
        cfg_path=Path(args.config),
        n_records=args.n_records,
        stride=args.stride,
        out_dir=Path(args.out_dir),
        device=device,
    )
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
