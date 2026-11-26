"""
experiments/eval_external_georgia.py — Zero-shot external validation of PTB-XL
Dense CBM/CEM (and window SoftSeg CBM/CEM) on Georgia 12-Lead ECG
(10,344 US clinical 500 Hz ECGs from explainable-ecg-learning/ecg-data-repo).

Evaluates 4 suites:
  1. Full 0_clean Georgia cohort (n=8,770 @ 100 Hz): wave duty-cycles, canonical
     cardiac-cycle ordering, HR MAE vs. Georgia est_hr_bpm, cross-model Dice/Acc
     agreement, SNOMED-CT clinical morphology (AF P-drop, BBB QRS widening,
     IAVB PR prolongation, LQT QT), and 5-superclass pathology transfer.
  2. NeuroKit2 500 Hz DWT delineation benchmark: Georgia OOD (n=1,000) vs.
     PTB-XL Fold-10 in-domain baseline (n=500) at ±150 ms & ±50 ms + Span Dice.
  3. Uncertainty & noise sensitivity: Bnd/Int σ ratio (Georgia vs. PTB-XL) and
     mean σ(t) shift across 0_clean / 2_moderate_anomaly / 3_severe_error.
  4. Window-level 500 Hz zero-shot check: cbm_softseg_masked & cem_softseg on
     200 clean Georgia records (completeness Accuracy/AUROC & concept F1).

Outputs:
  - experiments/results/external_georgia_eval.json
  - experiments/results/external_georgia_showcase.npz (3 curated clinical ECGs)
"""

from __future__ import annotations

import ast
import json
import os
import sys
import time
import warnings
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import neurokit2 as nk
import numpy as np
import pandas as pd
import torch
import wfdb
import yaml
from scipy.stats import mannwhitneyu, pearsonr
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from data import (
    BEAT_CONCEPT_NAMES as WINDOW_CONCEPT_NAMES,
    bandpass_filter,
    delineate_record,
    extract_beats,
    z_normalize,
)
from dense_seg import (
    SUPERCLASS_NAMES,
    WAVE6_NAMES,
    _multilabel_ece_brier,
    build_dense_model,
)
from external_data import (
    CANONICAL_LEADS,
    GeorgiaDenseDataset,
    canonicalize_leads,
    extract_wave_spans_and_intervals,
    load_georgia_metadata,
)
from models import build_model
from soft_seg import SHARED_CONCEPT_INDICES
from utils import set_seed

DENSE_RUNS: List[Tuple[str, str]] = [
    ("dense_cbm_detached_s42", "DenseCBM (Detached)"),
    ("dense_cbm_joint_s42",    "DenseCBM (Joint)"),
    ("dense_cem_s42",          "DenseCEM"),
    ("dense_multitask_s42",    "DenseMultiTask"),
    ("dense_blackbox_s42",     "DenseBlackBox"),
]

WINDOW_RUNS: List[Tuple[str, str]] = [
    ("cbm_softseg_masked", "CBM (SoftSeg Masked)"),
    ("cem_softseg",        "CEM (SoftSeg)"),
]

GEORGIA_ROOT = REPO_ROOT / "data" / "georgia_ecg"
PTBXL_ROOT = Path("/nfs/data8/schlegel/git/ecg-cbm/data/ptb-xl")
PTBXL_TEST_NPZ = REPO_ROOT / "checkpoints" / "dense_softseg_cache" / "wave6_all_sr100" / "test.npz"
OUT_JSON = REPO_ROOT / "experiments" / "results" / "external_georgia_eval.json"
OUT_SHOWCASE = REPO_ROOT / "experiments" / "results" / "external_georgia_showcase.npz"


def _load_dense_model(run_name: str, device: str) -> torch.nn.Module:
    run_dir = REPO_ROOT / "models" / run_name
    cfg = yaml.safe_load((run_dir / "config.yaml").read_text())
    model = build_dense_model(cfg)
    state = torch.load(run_dir / "checkpoints" / "best_model.pt", map_location="cpu", weights_only=False)
    model.load_state_dict(state["model"], strict=True)
    model.to(device).eval()
    return model


def _load_window_model(run_name: str, device: str) -> torch.nn.Module:
    run_dir = REPO_ROOT / "models" / run_name
    cfg = yaml.safe_load((run_dir / "config.yaml").read_text())
    model = build_model(cfg)
    state = torch.load(run_dir / "checkpoints" / "best_model.pt", map_location="cpu", weights_only=False)
    model.load_state_dict(state["model"], strict=True)
    model.to(device).eval()
    return model


@torch.no_grad()
def _infer_dense_batched(
    model: torch.nn.Module,
    signals: np.ndarray,
    device: str,
    batch_size: int = 256,
    use_bf16: bool = True,
    return_seg: bool = True,
) -> Dict[str, Any]:
    N, _, T = signals.shape
    amp_en = use_bf16 and device.startswith("cuda") and torch.cuda.is_bf16_supported()
    task_probs_l = []
    cls_seq = np.empty((N, T), dtype=np.uint8) if return_seg else None
    prob_sum = np.zeros(6, dtype=np.float64) if return_seg else None
    total_ts = 0

    for start in range(0, N, batch_size):
        end = min(N, start + batch_size)
        xb = torch.from_numpy(signals[start:end]).to(device, non_blocking=True)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=amp_en):
            out = model(xb)
        tp = torch.sigmoid(out["task_logits"].float()).cpu().numpy()
        task_probs_l.append(tp)
        if return_seg:
            cp = out["concept_probs"].float()  # (B, 6, T)
            cls_seq[start:end] = cp.argmax(dim=1).cpu().numpy().astype(np.uint8)
            prob_sum += cp.sum(dim=(0, 2)).cpu().numpy().astype(np.float64)
            total_ts += (end - start) * T

    res: Dict[str, Any] = {"task_probs": np.concatenate(task_probs_l, axis=0)}
    if return_seg:
        res["cls_seq"] = cls_seq
        res["soft_duty"] = (prob_sum / max(1, total_ts)).tolist()
    return res


def _preload_georgia(df: pd.DataFrame, target_sr: int = 100, workers: int = 16) -> Dict[str, Any]:
    ds = GeorgiaDenseDataset(df, georgia_root=GEORGIA_ROOT, target_sr=target_sr, return_500hz_lead2=False)
    n = len(ds)
    signals = np.empty((n, 12, 1000), dtype=np.float32)
    pathology = np.empty((n, 5), dtype=np.float32)

    def _get(idx: int):
        item = ds[idx]
        return idx, item["signal_100"].numpy(), item["pathology"].numpy()

    with ThreadPoolExecutor(max_workers=workers) as pool:
        for idx, sig, patho in pool.map(_get, range(n)):
            signals[idx] = sig
            pathology[idx] = patho

    return {
        "signals": signals,
        "pathology": pathology,
        "df": df,
    }


def _extract_intervals_worker(cls_row: np.ndarray) -> Dict[str, Any]:
    info = extract_wave_spans_and_intervals(cls_row, fs=100)
    # Strip large raw spans dict to keep IPC tiny for 8,770 records
    return {
        "n_beats": info["n_beats"],
        "pred_hr_bpm": info["pred_hr_bpm"],
        "duty_cycles": info["duty_cycles"],
        "canonical_transition_rate": info["canonical_transition_rate"],
        "strict_forward_transition_rate": info["strict_forward_transition_rate"],
        "qrs_ms": info["qrs_ms"],
        "pr_ms": info["pr_ms"],
        "qt_ms": info["qt_ms"],
        "p_present_beat_frac": info["p_present_beat_frac"],
    }


def _code_mask(codes_series: pd.Series, target_codes: set[str], exclude_codes: Optional[set[str]] = None) -> np.ndarray:
    res = np.zeros(len(codes_series), dtype=bool)
    for i, s in enumerate(codes_series):
        cset = {c.strip() for c in str(s).split("|") if c.strip()}
        if cset & target_codes:
            if exclude_codes and (cset & exclude_codes):
                continue
            res[i] = True
    return res


def _cohort_stat_pair(
    values: np.ndarray,
    pos_mask: np.ndarray,
    neg_mask: np.ndarray,
    higher_is_pos: bool = True,
) -> Dict[str, Any]:
    vp = values[pos_mask]
    vn = values[neg_mask]
    vp = vp[np.isfinite(vp)]
    vn = vn[np.isfinite(vn)]
    if len(vp) < 5 or len(vn) < 5:
        return {"n_pos": int(len(vp)), "n_neg": int(len(vn)), "auroc": float("nan")}
    y = np.concatenate([np.ones(len(vp), dtype=np.int32), np.zeros(len(vn), dtype=np.int32)])
    s = np.concatenate([vp, vn]) if higher_is_pos else -np.concatenate([vp, vn])
    auc = float(roc_auc_score(y, s))
    _, pval = mannwhitneyu(vp, vn, alternative="greater" if higher_is_pos else "less")
    return {
        "n_pos": int(len(vp)),
        "n_neg": int(len(vn)),
        "pos_mean": float(np.mean(vp)),
        "pos_median": float(np.median(vp)),
        "pos_iqr": [float(np.percentile(vp, 25)), float(np.percentile(vp, 75))],
        "neg_mean": float(np.mean(vn)),
        "neg_median": float(np.median(vn)),
        "neg_iqr": [float(np.percentile(vn, 25)), float(np.percentile(vn, 75))],
        "auroc": auc,
        "mannwhitney_p": float(pval),
    }


# ─────────────────────────────────────────────────────────────────────────────
# NeuroKit2 500 Hz DWT Worker (Suite 2)
# ─────────────────────────────────────────────────────────────────────────────

def _nk2_dwt_worker(args: Tuple[str, int]) -> Optional[Dict[str, Any]]:
    rec_path, fs_expected = args
    try:
        raw, fields = wfdb.rdsamp(rec_path)
        fs = int(fields.get("fs", fs_expected))
        canon = canonicalize_leads(raw.T, fields.get("sig_name", CANONICAL_LEADS))
        lead2 = np.nan_to_num(canon[1].astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            cleaned = nk.ecg_clean(lead2, sampling_rate=fs)
            _, rinfo = nk.ecg_peaks(cleaned, sampling_rate=fs)
            r_peaks = np.array([int(p) for p in rinfo["ECG_R_Peaks"] if not np.isnan(p)], dtype=int)
            if len(r_peaks) < 3:
                return None
            _, dwt = nk.ecg_delineate(cleaned, r_peaks, sampling_rate=fs, method="dwt")
    except Exception:
        return None

    scale = 100.0 / float(fs)

    def _to_100(vals) -> np.ndarray:
        if vals is None:
            return np.array([], dtype=np.float32)
        return np.array([float(v) * scale for v in vals if v is not None and not np.isnan(v)], dtype=np.float32)

    def _spans_100(onsets, offsets) -> np.ndarray:
        mask = np.zeros(1000, dtype=bool)
        if onsets is None or offsets is None:
            return mask
        for a, b in zip(onsets, offsets):
            if a is None or b is None or np.isnan(a) or np.isnan(b):
                continue
            ia = max(0, int(round(float(a) * scale)))
            ib = min(1000, int(round(float(b) * scale)) + 1)
            if ib > ia:
                mask[ia:ib] = True
        return mask

    return {
        "r_peaks_100": np.asarray(r_peaks, dtype=np.float32) * scale,
        "p_peaks_100": _to_100(dwt.get("ECG_P_Peaks")),
        "t_peaks_100": _to_100(dwt.get("ECG_T_Peaks")),
        "p_mask_100":  _spans_100(dwt.get("ECG_P_Onsets"), dwt.get("ECG_P_Offsets")),
        "qrs_mask_100": _spans_100(dwt.get("ECG_R_Onsets"), dwt.get("ECG_R_Offsets")),
        "t_mask_100":  _spans_100(dwt.get("ECG_T_Onsets"), dwt.get("ECG_T_Offsets")),
    }


def _match_peaks_1to1(pred_centers: List[float], gt_peaks: np.ndarray, tol_samples: float) -> Tuple[int, int, int, List[float]]:
    """Greedy closest-pair 1-to-1 matching in 100 Hz samples."""
    n_p = len(pred_centers)
    n_g = len(gt_peaks)
    if n_p == 0 or n_g == 0:
        return 0, n_p, n_g, []
    pairs = []
    for i, pc in enumerate(pred_centers):
        for j, gp in enumerate(gt_peaks):
            d = abs(pc - float(gp))
            if d <= tol_samples:
                pairs.append((d, i, j))
    pairs.sort(key=lambda x: x[0])
    used_p, used_g = set(), set()
    errs_ms = []
    for d, i, j in pairs:
        if i not in used_p and j not in used_g:
            used_p.add(i)
            used_g.add(j)
            errs_ms.append(d * 10.0)  # 100 Hz -> 10 ms/sample
    return len(errs_ms), n_p, n_g, errs_ms


def _eval_nk2_delineation(
    cls_seq: np.ndarray,
    nk2_items: List[Optional[Dict[str, Any]]],
    is_af_afl: np.ndarray,
) -> Dict[str, Any]:
    wave_cfg = [
        ("QRS", "r_peaks_100", "qrs_mask_100", False),
        ("P",   "p_peaks_100", "p_mask_100",   True),   # evaluate P on non-AF/AFL
        ("T",   "t_peaks_100", "t_mask_100",   False),
    ]
    cls_idx = {"P": 0, "QRS": 2, "T": 4}
    out: Dict[str, Any] = {}

    spans_all = [extract_wave_spans_and_intervals(cls_seq[i], fs=100)["spans"] for i in range(len(cls_seq))]

    for wname, peak_key, mask_key, skip_af in wave_cfg:
        c_i = cls_idx[wname]
        w_res: Dict[str, Any] = {}
        for tol_ms, tol_samp in ((150, 15.0), (50, 5.0)):
            tp = pred_tot = gt_tot = 0
            errs_ms: List[float] = []
            inter_sum = union_sum = n_span_recs = 0
            for i, nk_item in enumerate(nk2_items):
                if nk_item is None:
                    continue
                if skip_af and is_af_afl[i]:
                    continue
                centers = [sp[2] for sp in spans_all[i][wname]]
                m, np_, ng_, e_l = _match_peaks_1to1(centers, nk_item[peak_key], tol_samp)
                tp += m
                pred_tot += np_
                gt_tot += ng_
                errs_ms.extend(e_l)

                if tol_ms == 150:
                    nk_mask = nk_item[mask_key]
                    if nk_mask.any():
                        pm = (cls_seq[i] == c_i)
                        inter_sum += int(np.logical_and(pm, nk_mask).sum())
                        union_sum += int(pm.sum() + nk_mask.sum())
                        n_span_recs += 1

            prec = tp / max(1, pred_tot)
            rec = tp / max(1, gt_tot)
            f1 = (2 * prec * rec) / max(1e-8, prec + rec)
            w_res[f"tol_{tol_ms}ms"] = {
                "precision": float(prec),
                "recall": float(rec),
                "f1": float(f1),
                "mae_ms": float(np.mean(errs_ms)) if errs_ms else float("nan"),
                "median_ae_ms": float(np.median(errs_ms)) if errs_ms else float("nan"),
                "tp": int(tp),
                "pred_total": int(pred_tot),
                "gt_total": int(gt_tot),
            }
            if tol_ms == 150:
                w_res["span_dice"] = float((2.0 * inter_sum) / max(1, union_sum))
                w_res["n_span_records"] = int(n_span_recs)
        out[wname] = w_res
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Suite 3: Uncertainty Evaluation
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def _eval_uncertainty_subset(
    model: torch.nn.Module,
    signals: np.ndarray,
    device: str,
    sub_bs: int = 32,
) -> Dict[str, Any]:
    N = len(signals)
    rec_means = np.empty(N, dtype=np.float32)
    bnd_vals: List[float] = []
    int_vals: List[float] = []

    for start in range(0, N, sub_bs):
        end = min(N, start + sub_bs)
        xb = torch.from_numpy(signals[start:end]).to(device, non_blocking=True)
        unc = model.predict_with_uncertainty(xb, n_mc_epistemic=10, n_mc_aleatoric=16, use_bf16=True)
        p_mean = unc["concept_probs_mean"].cpu().numpy()           # (B, 6, T)
        p_std = unc["concept_total_std"].mean(dim=1).cpu().numpy() # (B, T)
        rec_means[start:end] = p_std.mean(axis=1)

        pred_cls = p_mean.argmax(axis=1)
        is_bnd = np.zeros_like(pred_cls, dtype=bool)
        is_bnd[:, 1:] = pred_cls[:, 1:] != pred_cls[:, :-1]
        is_bnd[:, :-1] |= pred_cls[:, :-1] != pred_cls[:, 1:]
        if np.any(is_bnd) and np.any(~is_bnd):
            bnd_vals.append(float(p_std[is_bnd].mean()))
            int_vals.append(float(p_std[~is_bnd].mean()))

    bnd_m = float(np.mean(bnd_vals)) if bnd_vals else float("nan")
    int_m = float(np.mean(int_vals)) if int_vals else float("nan")
    return {
        "record_mean_stds": rec_means,
        "mean_std": float(rec_means.mean()),
        "median_std": float(np.median(rec_means)),
        "bnd_std": bnd_m,
        "int_std": int_m,
        "bnd_vs_int": bnd_m / max(int_m, 1e-8) if np.isfinite(bnd_m) and np.isfinite(int_m) else float("nan"),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Suite 4: Window-Level 500 Hz Check
# ─────────────────────────────────────────────────────────────────────────────

def _extract_georgia_windows_worker(args: Tuple[str, int]) -> List[Tuple[np.ndarray, int, List[int], List[int]]]:
    rec_path, seed = args
    try:
        raw, fields = wfdb.rdsamp(rec_path)
        fs = int(fields.get("fs", 500))
        canon = canonicalize_leads(raw.T, fields.get("sig_name", CANONICAL_LEADS))
        canon = np.nan_to_num(canon.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
        sig_filt = bandpass_filter(canon, sampling_rate=fs, lowcut=0.5, highcut=40.0)
        delin = delineate_record(sig_filt, sampling_rate=fs, lead_idx=1)
        rng = np.random.default_rng(seed)
        return extract_beats(
            sig_filt, delin,
            t_fixed=500, beat_window_pre=250, beat_window_post=250,
            neg_oversample_factor=1.0, rng=rng, pos_jitter=40, use_masked_norm=False,
        )
    except Exception:
        return []


# ─────────────────────────────────────────────────────────────────────────────
# Main Orchestrator
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    t_start = time.perf_counter()
    set_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
    print(f"=== Georgia 12-Lead ECG Zero-Shot Evaluation (device={device}) ===")

    # ── Load PTB-XL Fold 10 cache & reference metadata ───────────────────────
    ptb_test = np.load(PTBXL_TEST_NPZ)
    ptb_signals = ptb_test["signals"].astype(np.float32)  # (2198, 12, 1000)
    ptb_seg = ptb_test["seg_u8"].astype(np.float32) / 255.0
    ptb_seg /= np.maximum(ptb_seg.sum(axis=1, keepdims=True), 1e-6)
    ptb_ref_duty = {
        name: float(ptb_seg[:, k, :].mean()) for k, name in enumerate(WAVE6_NAMES)
    }
    ptb_ids = ptb_test["ecg_ids"]

    # ── Load Georgia 0_clean metadata & preload 8,770 signals ────────────────
    df_clean = load_georgia_metadata(GEORGIA_ROOT, max_error_severity=0, require_10s_500hz=True)
    n_clean = len(df_clean)
    print(f"[Suite 1] Preloading {n_clean} Georgia 0_clean records @ 100 Hz...")
    t0 = time.perf_counter()
    pre = _preload_georgia(df_clean, target_sr=100, workers=16)
    signals = pre["signals"]
    pathology = pre["pathology"]
    print(f"          Preloaded in {time.perf_counter() - t0:.1f}s | shape={signals.shape}")

    # Pure MI mask (without QAb) for transparent reporting
    pure_mi_y = _code_mask(df_clean["snomed_codes"], {"164865005", "164867002", "57054005", "54329005"}).astype(np.float32)
    mask_nsr = _code_mask(df_clean["snomed_codes"], {"426783006"}, exclude_codes={"164889003", "164890007"})
    mask_af = _code_mask(df_clean["snomed_codes"], {"164889003"})
    mask_lbbb = _code_mask(df_clean["snomed_codes"], {"164909002"})
    mask_rbbb = _code_mask(df_clean["snomed_codes"], {"59118001", "713427006"})
    mask_bbb = mask_lbbb | mask_rbbb
    mask_iavb = _code_mask(df_clean["snomed_codes"], {"270492004"})
    mask_lqt = _code_mask(df_clean["snomed_codes"], {"111975006"})
    est_hr = pd.to_numeric(df_clean["est_hr_bpm"], errors="coerce").to_numpy(dtype=np.float64)

    print(
        f"          Clinical cohorts: NSR={mask_nsr.sum()}, AF={mask_af.sum()}, "
        f"LBBB={mask_lbbb.sum()}, RBBB/CRBBB={mask_rbbb.sum()}, IAVB={mask_iavb.sum()}, LQT={mask_lqt.sum()}"
    )
    print(
        "          Superclass positives: "
        + ", ".join(f"{name}={int(pathology[:, i].sum())}" for i, name in enumerate(SUPERCLASS_NAMES))
        + f" (pure_MI={int(pure_mi_y.sum())})"
    )

    models: Dict[str, torch.nn.Module] = {}
    cls_seqs: Dict[str, np.ndarray] = {}
    suite1: Dict[str, Any] = {
        "n_records": int(n_clean),
        "ptbxl_fold10_ref_duty": ptb_ref_duty,
        "cohort_counts": {
            "NSR_non_af": int(mask_nsr.sum()),
            "AF": int(mask_af.sum()),
            "LBBB": int(mask_lbbb.sum()),
            "RBBB_CRBBB": int(mask_rbbb.sum()),
            "BBB_any": int(mask_bbb.sum()),
            "IAVB": int(mask_iavb.sum()),
            "LQT": int(mask_lqt.sum()),
            "pure_MI": int(pure_mi_y.sum()),
            "superclasses": {name: int(pathology[:, i].sum()) for i, name in enumerate(SUPERCLASS_NAMES)},
        },
        "models": {},
    }

    with ProcessPoolExecutor(max_workers=16) as proc_pool:
        for run_name, display_name in DENSE_RUNS:
            model = _load_dense_model(run_name, device)
            models[run_name] = model
            has_seg = (run_name != "dense_blackbox_s42")
            inf = _infer_dense_batched(model, signals, device=device, batch_size=256, return_seg=has_seg)
            tp = inf["task_probs"]

            # Pathology zero-shot AUROC + ECE/Brier
            per_cls_auc = {}
            aucs_5 = []
            aucs_4 = []
            for c, cname in enumerate(SUPERCLASS_NAMES):
                yt = pathology[:, c]
                n_pos = int(yt.sum())
                auc = float(roc_auc_score(yt, tp[:, c])) if len(np.unique(yt)) >= 2 else float("nan")
                per_cls_auc[cname] = {"auroc": auc, "n_pos": n_pos}
                if np.isfinite(auc):
                    aucs_5.append(auc)
                    if cname != "MI":
                        aucs_4.append(auc)
            pure_mi_auc = float(roc_auc_score(pure_mi_y, tp[:, 1])) if len(np.unique(pure_mi_y)) >= 2 else float("nan")
            ece, brier = _multilabel_ece_brier(tp, pathology)

            m_entry: Dict[str, Any] = {
                "display_name": display_name,
                "pathology": {
                    "macro_auroc_5": float(np.mean(aucs_5)),
                    "macro_auroc_4_ex_mi": float(np.mean(aucs_4)),
                    "per_class": per_cls_auc,
                    "pure_mi_auroc": pure_mi_auc,
                    "ece": float(ece),
                    "brier": float(brier),
                },
            }

            if has_seg:
                cseq = inf["cls_seq"]
                cls_seqs[run_name] = cseq
                rows = list(proc_pool.map(_extract_intervals_worker, cseq, chunksize=64))

                duty_means = {
                    name: float(np.mean([r["duty_cycles"][name] for r in rows])) for name in WAVE6_NAMES
                }
                canon_rate = float(np.mean([r["canonical_transition_rate"] for r in rows]))
                strict_rate = float(np.mean([r["strict_forward_transition_rate"] for r in rows]))

                pred_hr = np.array([r["pred_hr_bpm"] for r in rows], dtype=np.float64)
                valid_hr = np.isfinite(pred_hr) & np.isfinite(est_hr) & (est_hr >= 30.0) & (est_hr <= 220.0)
                hr_diff = np.abs(pred_hr[valid_hr] - est_hr[valid_hr])
                hr_r, _ = pearsonr(pred_hr[valid_hr], est_hr[valid_hr])

                p_duty = np.array([r["duty_cycles"]["P"] for r in rows], dtype=np.float64)
                p_frac = np.array([r["p_present_beat_frac"] for r in rows], dtype=np.float64)
                qrs_ms = np.array([r["qrs_ms"] for r in rows], dtype=np.float64)
                pr_ms = np.array([r["pr_ms"] for r in rows], dtype=np.float64)
                qt_ms = np.array([r["qt_ms"] for r in rows], dtype=np.float64)

                m_entry["segmentation"] = {
                    "hard_duty": duty_means,
                    "soft_duty": {name: float(v) for name, v in zip(WAVE6_NAMES, inf["soft_duty"])},
                    "canonical_transition_rate": canon_rate,
                    "strict_forward_transition_rate": strict_rate,
                    "hr_agreement": {
                        "n_valid": int(valid_hr.sum()),
                        "coverage": float(valid_hr.mean()),
                        "mae_bpm": float(np.mean(hr_diff)),
                        "median_ae_bpm": float(np.median(hr_diff)),
                        "pearson_r": float(hr_r),
                        "within_5bpm": float(np.mean(hr_diff <= 5.0)),
                    },
                    "clinical_morphology": {
                        "af_vs_nsr_p_duty": _cohort_stat_pair(p_duty, mask_af, mask_nsr, higher_is_pos=False),
                        "af_vs_nsr_p_beat_frac": _cohort_stat_pair(p_frac, mask_af, mask_nsr, higher_is_pos=False),
                        "lbbb_vs_nsr_qrs_ms": _cohort_stat_pair(qrs_ms, mask_lbbb, mask_nsr, higher_is_pos=True),
                        "rbbb_vs_nsr_qrs_ms": _cohort_stat_pair(qrs_ms, mask_rbbb, mask_nsr, higher_is_pos=True),
                        "bbb_vs_nsr_qrs_ms": _cohort_stat_pair(qrs_ms, mask_bbb, mask_nsr, higher_is_pos=True),
                        "iavb_vs_nsr_pr_ms": _cohort_stat_pair(pr_ms, mask_iavb, mask_nsr, higher_is_pos=True),
                        "lqt_vs_nsr_qt_ms": _cohort_stat_pair(qt_ms, mask_lqt, mask_nsr, higher_is_pos=True),
                    },
                }

            suite1["models"][run_name] = m_entry
            print(
                f"  - {display_name:20s}: Path4={m_entry['pathology']['macro_auroc_4_ex_mi']:.4f} "
                + (
                    f"| Canon={m_entry['segmentation']['canonical_transition_rate']*100:.2f}% "
                    f"| HR MAE={m_entry['segmentation']['hr_agreement']['mae_bpm']:.2f} bpm "
                    f"(r={m_entry['segmentation']['hr_agreement']['pearson_r']:.3f})"
                    if has_seg else "(No seg)"
                )
            )

    # Cross-model segmentation agreement across 8.77M Georgia timestamps
    seg_names = [r for r, _ in DENSE_RUNS if r != "dense_blackbox_s42"]
    agreement_matrix: Dict[str, Dict[str, Dict[str, float]]] = {}
    for a in seg_names:
        agreement_matrix[a] = {}
        ca = cls_seqs[a]
        for b in seg_names:
            cb = cls_seqs[b]
            acc = float((ca == cb).mean())
            dices = []
            for k in range(6):
                ak = (ca == k)
                bk = (cb == k)
                inter = int(np.logical_and(ak, bk).sum())
                denom = int(ak.sum() + bk.sum())
                dices.append((2.0 * inter + 1e-4) / (denom + 1e-4))
            agreement_matrix[a][b] = {"acc": acc, "macro_dice": float(np.mean(dices))}
    suite1["cross_model_agreement"] = agreement_matrix

    # ── Curate 3 Clinical Showcase Records (NSR, AF, LBBB) & Save NPZ ────────
    _save_showcase_records(df_clean, signals, cls_seqs, models, device, mask_nsr, mask_af, mask_lbbb, est_hr)

    # ── Suite 2: NeuroKit2 500 Hz DWT Benchmark (Georgia n=1,000 vs PTB-XL n=500)
    print("\n[Suite 2] Running NeuroKit2 500 Hz DWT benchmark (Georgia n=1,000 vs PTB-XL Fold-10 n=500)...")
    rng = np.random.default_rng(42)
    ga_idx = np.sort(rng.choice(n_clean, size=min(1000, n_clean), replace=False))
    ga_args = [(str(GEORGIA_ROOT / "data" / str(df_clean.iloc[i]["record_path"])), 500) for i in ga_idx]
    ga_af_afl = _code_mask(df_clean.iloc[ga_idx]["snomed_codes"], {"164889003", "164890007"})
    ga_sig_sub = signals[ga_idx]

    ptb_meta = pd.read_csv(PTBXL_ROOT / "ptbxl_database.csv", index_col="ecg_id", dtype={"scp_codes": str})
    ptb_sel = np.sort(rng.choice(len(ptb_ids), size=min(500, len(ptb_ids)), replace=False))
    ptb_sel_ids = ptb_ids[ptb_sel]
    ptb_args = [(str(PTBXL_ROOT / str(ptb_meta.loc[int(eid), "filename_hr"])), 500) for eid in ptb_sel_ids]
    ptb_af_afl = np.array([
        ("AFIB" in ast.literal_eval(str(ptb_meta.loc[int(eid), "scp_codes"])) or
         "AFLT" in ast.literal_eval(str(ptb_meta.loc[int(eid), "scp_codes"])))
        for eid in ptb_sel_ids
    ], dtype=bool)
    ptb_sig_sub = ptb_signals[ptb_sel]

    t1 = time.perf_counter()
    with ProcessPoolExecutor(max_workers=16) as pool:
        ga_nk2 = list(pool.map(_nk2_dwt_worker, ga_args, chunksize=16))
        ptb_nk2 = list(pool.map(_nk2_dwt_worker, ptb_args, chunksize=16))
    print(
        f"          NK2 DWT finished in {time.perf_counter() - t1:.1f}s | "
        f"valid Georgia={sum(x is not None for x in ga_nk2)}/{len(ga_nk2)}, "
        f"PTB-XL={sum(x is not None for x in ptb_nk2)}/{len(ptb_nk2)}"
    )

    suite2: Dict[str, Any] = {
        "georgia_n_sampled": int(len(ga_idx)),
        "georgia_n_delineated": int(sum(x is not None for x in ga_nk2)),
        "ptbxl_n_sampled": int(len(ptb_sel)),
        "ptbxl_n_delineated": int(sum(x is not None for x in ptb_nk2)),
        "models": {},
    }
    for run_name, display_name in DENSE_RUNS:
        if run_name == "dense_blackbox_s42":
            continue
        ptb_inf = _infer_dense_batched(models[run_name], ptb_sig_sub, device=device, batch_size=128, return_seg=True)
        ga_cseq_sub = cls_seqs[run_name][ga_idx]
        ptb_cseq_sub = ptb_inf["cls_seq"]

        ga_nk_res = _eval_nk2_delineation(ga_cseq_sub, ga_nk2, ga_af_afl)
        ptb_nk_res = _eval_nk2_delineation(ptb_cseq_sub, ptb_nk2, ptb_af_afl)
        suite2["models"][run_name] = {
            "display_name": display_name,
            "georgia": ga_nk_res,
            "ptbxl_fold10": ptb_nk_res,
        }
        print(
            f"  - {display_name:20s} | PTB-XL F1@150 (QRS/P/T) = "
            f"{ptb_nk_res['QRS']['tol_150ms']['f1']:.3f}/{ptb_nk_res['P']['tol_150ms']['f1']:.3f}/{ptb_nk_res['T']['tol_150ms']['f1']:.3f} | "
            f"Georgia F1@150 = "
            f"{ga_nk_res['QRS']['tol_150ms']['f1']:.3f}/{ga_nk_res['P']['tol_150ms']['f1']:.3f}/{ga_nk_res['T']['tol_150ms']['f1']:.3f}"
        )

    # ── Suite 3: Boundary Uncertainty & Noise Sensitivity ─────────────────────
    print("\n[Suite 3] Evaluating MC boundary uncertainty & severe artifact detection...")
    df_mod = load_georgia_metadata(GEORGIA_ROOT, severity_tiers=["2_moderate_anomaly"], require_10s_500hz=True)
    df_sev = load_georgia_metadata(GEORGIA_ROOT, severity_tiers=["3_severe_error"], require_10s_500hz=True)
    clean_u_idx = np.sort(rng.choice(n_clean, size=min(500, n_clean), replace=False))
    mod_idx = np.sort(rng.choice(len(df_mod), size=min(200, len(df_mod)), replace=False))

    sig_clean_u = signals[clean_u_idx]
    sig_mod_u = _preload_georgia(df_mod.iloc[mod_idx].reset_index(drop=True), target_sr=100, workers=8)["signals"]
    sig_sev_u = _preload_georgia(df_sev.reset_index(drop=True), target_sr=100, workers=8)["signals"]

    suite3: Dict[str, Any] = {
        "n_ptbxl": int(len(ptb_sig_sub)),
        "n_clean": int(len(sig_clean_u)),
        "n_moderate": int(len(sig_mod_u)),
        "n_severe": int(len(sig_sev_u)),
        "models": {},
    }
    for run_name, display_name in DENSE_RUNS:
        if run_name == "dense_blackbox_s42":
            continue
        m = models[run_name]
        u_ptb = _eval_uncertainty_subset(m, ptb_sig_sub, device, sub_bs=32)
        u_cln = _eval_uncertainty_subset(m, sig_clean_u, device, sub_bs=32)
        u_mod = _eval_uncertainty_subset(m, sig_mod_u, device, sub_bs=32)
        u_sev = _eval_uncertainty_subset(m, sig_sev_u, device, sub_bs=32)

        y_det = np.concatenate([np.zeros(len(sig_clean_u), dtype=np.int32), np.ones(len(sig_sev_u), dtype=np.int32)])
        s_det = np.concatenate([u_cln["record_mean_stds"], u_sev["record_mean_stds"]])
        sev_auroc = float(roc_auc_score(y_det, s_det))

        suite3["models"][run_name] = {
            "display_name": display_name,
            "ptbxl_bnd_vs_int": u_ptb["bnd_vs_int"],
            "ptbxl_mean_std": u_ptb["mean_std"],
            "georgia_clean_bnd_vs_int": u_cln["bnd_vs_int"],
            "georgia_clean_mean_std": u_cln["mean_std"],
            "georgia_moderate_mean_std": u_mod["mean_std"],
            "georgia_severe_mean_std": u_sev["mean_std"],
            "severe_vs_clean_auroc": sev_auroc,
        }
        print(
            f"  - {display_name:20s} | Bnd/Int σ: PTB={u_ptb['bnd_vs_int']:.2f}x, GA={u_cln['bnd_vs_int']:.2f}x | "
            f"σ(clean→mod→sev)={u_cln['mean_std']:.4f}→{u_mod['mean_std']:.4f}→{u_sev['mean_std']:.4f} "
            f"(Sev-Det AUROC={sev_auroc:.3f})"
        )

    # ── Suite 4: Window-Level 500 Hz Zero-Shot Check ─────────────────────────
    print("\n[Suite 4] Evaluating 500 Hz Window-Level CBM/CEM on 200 Georgia records...")
    win_rec_idx = np.sort(rng.choice(n_clean, size=min(200, n_clean), replace=False))
    win_args = [(str(GEORGIA_ROOT / "data" / str(df_clean.iloc[i]["record_path"])), int(42 + k)) for k, i in enumerate(win_rec_idx)]
    win_samples: List[Tuple[np.ndarray, int, List[int], List[int]]] = []
    with ProcessPoolExecutor(max_workers=16) as pool:
        for res in pool.map(_extract_georgia_windows_worker, win_args, chunksize=8):
            win_samples.extend(res)

    w_sigs = np.stack([s[0] for s in win_samples], axis=0)
    w_y = np.array([s[1] for s in win_samples], dtype=np.int32)
    w_c = np.array([s[2] for s in win_samples], dtype=np.float32)
    w_w = np.array([s[3] for s in win_samples], dtype=np.float32)

    suite4: Dict[str, Any] = {
        "n_records": int(len(win_rec_idx)),
        "n_windows": int(len(w_sigs)),
        "n_positive": int((w_y == 1).sum()),
        "n_negative": int((w_y == 0).sum()),
        "models": {},
    }
    for run_name, display_name in WINDOW_RUNS:
        wm = _load_window_model(run_name, device)
        tp_l, cp_l = [], []
        with torch.no_grad():
            for start in range(0, len(w_sigs), 256):
                xb = torch.from_numpy(w_sigs[start : start + 256]).to(device)
                tp, cp, _ = wm(xb)
                tp_l.append(tp.squeeze(-1).cpu().numpy())
                cp_l.append(cp.cpu().numpy())
        wtp = np.concatenate(tp_l, axis=0)
        wcp = np.concatenate(cp_l, axis=0)

        w_acc = float(accuracy_score(w_y, (wtp >= 0.5).astype(int)))
        w_f1 = float(f1_score(w_y, (wtp >= 0.5).astype(int)))
        w_auc = float(roc_auc_score(w_y, wtp))

        c_metrics = {}
        c_f1s = []
        for k in SHARED_CONCEPT_INDICES:
            cname = WINDOW_CONCEPT_NAMES[k]
            valid = w_w[:, k] > 0
            if int(valid.sum()) >= 10 and len(np.unique(w_c[valid, k])) >= 2:
                yt_k = (w_c[valid, k] >= 0.5).astype(int)
                yp_k = (wcp[valid, k] >= 0.5).astype(int)
                fk = float(f1_score(yt_k, yp_k, zero_division=0))
                ak = float(accuracy_score(yt_k, yp_k))
                uk = float(roc_auc_score(yt_k, wcp[valid, k]))
                c_metrics[cname] = {"f1": fk, "accuracy": ak, "auroc": uk, "n_valid": int(valid.sum())}
                c_f1s.append(fk)

        suite4["models"][run_name] = {
            "display_name": display_name,
            "task_accuracy": w_acc,
            "task_f1": w_f1,
            "task_auroc": w_auc,
            "shared_concept_macro_f1": float(np.mean(c_f1s)) if c_f1s else float("nan"),
            "concepts": c_metrics,
        }
        print(
            f"  - {display_name:20s} | Windows={len(w_sigs)} | Task Acc={w_acc:.4f}, "
            f"AUROC={w_auc:.4f} | Concept Macro-F1={np.mean(c_f1s):.4f}"
        )

    payload = {
        "dataset": "Georgia 12-Lead ECG (PhysioNet/CinC 2020, ecg-data-repo)",
        "suite1_full_clean_cohort": suite1,
        "suite2_neurokit2_delineation": suite2,
        "suite3_uncertainty_and_noise": suite3,
        "suite4_window_500hz_zeroshot": suite4,
        "elapsed_seconds": float(time.perf_counter() - t_start),
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(payload, indent=2))
    print(f"\nSaved evaluation report → {OUT_JSON} (total time {time.perf_counter() - t_start:.1f}s)")


def _save_showcase_records(
    df_clean: pd.DataFrame,
    signals: np.ndarray,
    cls_seqs: Dict[str, np.ndarray],
    models: Dict[str, torch.nn.Module],
    device: str,
    mask_nsr: np.ndarray,
    mask_af: np.ndarray,
    mask_lbbb: np.ndarray,
    est_hr: np.ndarray,
) -> None:
    """Select 3 representative clean Georgia ECGs (NSR, AF, LBBB) & save masks/uncertainty."""
    cbm_seq = cls_seqs["dense_cbm_detached_s42"]

    def _pick_best(cand_mask: np.ndarray, criterion: str) -> int:
        idxs = np.flatnonzero(cand_mask)
        best_i = int(idxs[0])
        best_score = -1e9
        for idx in idxs[:300]:
            info = extract_wave_spans_and_intervals(cbm_seq[idx], fs=100)
            hr_ok = np.isfinite(est_hr[idx]) and np.isfinite(info["pred_hr_bpm"]) and abs(est_hr[idx] - info["pred_hr_bpm"]) < 4.0
            if not hr_ok:
                continue
            if criterion == "nsr":
                if np.isfinite(info["pr_ms"]) and 120 <= info["pr_ms"] <= 200 and (info["p_present_beat_frac"] or 0) > 0.85:
                    score = info["canonical_transition_rate"] + (info["p_present_beat_frac"] or 0)
                    if score > best_score:
                        best_score, best_i = score, int(idx)
            elif criterion == "af":
                pfrac = info["p_present_beat_frac"] if np.isfinite(info["p_present_beat_frac"]) else 1.0
                score = info["canonical_transition_rate"] + (1.0 - pfrac)
                if pfrac < 0.25 and score > best_score:
                    best_score, best_i = score, int(idx)
            elif criterion == "lbbb":
                qrs = info["qrs_ms"]
                if np.isfinite(qrs) and qrs >= 120.0:
                    score = info["canonical_transition_rate"] + min(qrs, 190.0) / 200.0
                    if score > best_score:
                        best_score, best_i = score, int(idx)
        return best_i

    picks = {
        "nsr":  _pick_best(mask_nsr, "nsr"),
        "af":   _pick_best(mask_af, "af"),
        "lbbb": _pick_best(mask_lbbb, "lbbb"),
    }

    sel_idx = [picks["nsr"], picks["af"], picks["lbbb"]]
    xb = torch.from_numpy(signals[sel_idx]).to(device)
    with torch.no_grad():
        unc_cbm = models["dense_cbm_detached_s42"].predict_with_uncertainty(xb, n_mc_epistemic=12, n_mc_aleatoric=16, use_bf16=True)
        unc_cem = models["dense_cem_s42"].predict_with_uncertainty(xb, n_mc_epistemic=12, n_mc_aleatoric=16, use_bf16=True)

    # Also run NK2 on the 3 raw 500 Hz Lead II signals for overlay dots
    nk_p = np.full((3, 30), -1.0, dtype=np.float32)
    nk_r = np.full((3, 30), -1.0, dtype=np.float32)
    nk_t = np.full((3, 30), -1.0, dtype=np.float32)
    rec_ids = []
    labels = []
    for slot, idx in enumerate(sel_idx):
        row = df_clean.iloc[idx]
        rec_ids.append(str(row["record_id"]))
        labels.append(str(row["label_names"]))
        nk_item = _nk2_dwt_worker((str(GEORGIA_ROOT / "data" / str(row["record_path"])), 500))
        if nk_item is not None:
            for arr, key in ((nk_p, "p_peaks_100"), (nk_r, "r_peaks_100"), (nk_t, "t_peaks_100")):
                vals = nk_item[key][:30]
                arr[slot, : len(vals)] = vals

    OUT_SHOWCASE.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        OUT_SHOWCASE,
        tags=np.array(["NSR", "AF", "LBBB"]),
        record_ids=np.array(rec_ids),
        label_names=np.array(labels),
        lead2_100hz=signals[sel_idx, 1, :],
        cbm_probs=unc_cbm["concept_probs_mean"].cpu().numpy(),
        cbm_std=unc_cbm["concept_total_std"].mean(dim=1).cpu().numpy(),
        cbm_task=unc_cbm["task_probs_mean"].cpu().numpy(),
        cem_probs=unc_cem["concept_probs_mean"].cpu().numpy(),
        cem_std=unc_cem["concept_total_std"].mean(dim=1).cpu().numpy(),
        cem_task=unc_cem["task_probs_mean"].cpu().numpy(),
        nk_p=nk_p,
        nk_r=nk_r,
        nk_t=nk_t,
    )
    print(f"Saved 3 showcase records → {OUT_SHOWCASE} ({rec_ids})")


if __name__ == "__main__":
    main()
