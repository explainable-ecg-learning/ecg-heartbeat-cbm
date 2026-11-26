"""
explorer/build_vercel.py — Export a zero-cold-start, static Vercel deployment
bundle under `explorer/vercel/` from existing trained checkpoints, PTB-XL Fold 10,
and Georgia 12-Lead (OOD) ECG records.

Run:
    CUDA_VISIBLE_DEVICES=0 .venv_new/bin/python explorer/build_vercel.py
"""

from __future__ import annotations

import json
import math
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
EXPLORER_DIR = REPO_ROOT / "explorer"
SRC_DIR = REPO_ROOT / "src"
for p in (str(EXPLORER_DIR), str(SRC_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

import app as exp_app  # noqa: E402
from data import (  # noqa: E402
    bandpass_filter,
    get_concept_labels,
    is_complete_heartbeat_window,
    z_normalize,
)
from dense_seg import (  # noqa: E402
    SUPERCLASS_NAMES,
    WAVE6_NAMES,
    collapse_soft_seg_targets,
    get_dense_concept_names,
)
from external_data import extract_wave_spans_and_intervals  # noqa: E402
from models import LOG_VAR_MAX  # noqa: E402
from soft_seg import soft_seg_to_wave_curves  # noqa: E402

VERCEL_DIR = EXPLORER_DIR / "vercel"
DATA_OUT = VERCEL_DIR / "data"
REC_OUT = DATA_OUT / "records"

WINDOW_EXPORT_IDS = ["cem_softseg", "cbm_softseg_masked", "blackbox_softseg"]
DENSE_EXPORT_IDS = [
    "dense_cem_s42",
    "dense_cbm_detached_s42",
    "dense_cbm_joint_s42",
    "dense_multitask_s42",
    "dense_blackbox_s42",
]
STATIC_STRIDE = 25


def _round_list(arr: np.ndarray, decimals: int) -> list:
    return np.round(np.asarray(arr, dtype=np.float64), decimals).tolist()


def _quantize_u8_dict(arr: np.ndarray, names: List[str], omit_tp: bool = True) -> Dict[str, List[int]]:
    """Quantize (K, T) float probabilities/stds into 0..255 ints, omitting redundant TP."""
    out: Dict[str, List[int]] = {}
    for i, name in enumerate(names):
        if omit_tp and name == "TP":
            continue
        q = np.clip(np.round(np.asarray(arr[i], dtype=np.float64) * 255.0), 0, 255).astype(np.int32)
        out[name] = q.tolist()
    return out


def _compute_seg_spans(probs_w6: np.ndarray, thresh: float = 0.35) -> List[List[int]]:
    """Collapse (6, T) wave6 probabilities into contiguous [start, end, wave_idx] (wave_idx 0..4) spans."""
    wave_probs = np.asarray(probs_w6[:5], dtype=np.float32)  # P, PQ, QRS, ST, T
    T = wave_probs.shape[1]
    best_vals = wave_probs.max(axis=0)
    best_idx = wave_probs.argmax(axis=0)
    active = best_vals > thresh

    spans: List[List[int]] = []
    cur_idx = -1
    run_start = 0
    for t in range(T):
        w = int(best_idx[t]) if active[t] else -1
        if w != cur_idx:
            if cur_idx >= 0 and t > run_start:
                spans.append([run_start, t, cur_idx])
            cur_idx = w
            run_start = t
    if cur_idx >= 0 and T > run_start:
        spans.append([run_start, T, cur_idx])
    return spans


def _select_curated_records() -> Tuple[List[int], List[str]]:
    """Select 15 PTB-XL Fold-10 records and 15 Georgia 12-Lead (OOD) records across classes and tiers."""
    ptb_meta = exp_app._get_meta()
    fold10 = ptb_meta[ptb_meta["strat_fold"] == 10]
    archive = exp_app._get_soft_seg_archive()

    def _ptb_valid(eid_i: int) -> bool:
        if eid_i not in fold10.index:
            return False
        if archive is not None and eid_i not in archive:
            return False
        return bool(exp_app._load_delineation_any(eid_i, 500, 0, dataset="ptbxl").get("quality_ok", False))

    ptb_picks: List[int] = []
    for eid_i in (9, 8215, 65, 63, 116, 162):
        if eid_i not in ptb_picks and _ptb_valid(eid_i):
            ptb_picks.append(eid_i)

    ptb_tag_pools = {
        tag: [int(e) for e in fold10[fold10["_tag_set"].apply(lambda s, t=tag: t in s)].index.tolist()]
        for tag in ("NORM", "AFIB", "BBB", "MI", "STTC", "HYP")
    }
    ptb_cursor = {tag: 0 for tag in ptb_tag_pools}
    ptb_tags = ("NORM", "AFIB", "BBB", "MI", "STTC", "HYP")
    while len(ptb_picks) < 15:
        progressed = False
        for tag in ptb_tags:
            if len(ptb_picks) >= 15:
                break
            pool = ptb_tag_pools[tag]
            idx = ptb_cursor[tag]
            while idx < len(pool):
                eid_i = pool[idx]
                idx += 1
                if eid_i not in ptb_picks and _ptb_valid(eid_i):
                    ptb_picks.append(eid_i)
                    progressed = True
                    break
            ptb_cursor[tag] = idx
        if not progressed:
            break

    if len(ptb_picks) < 15:
        for eid in fold10.index.tolist():
            eid_i = int(eid)
            if eid_i not in ptb_picks and _ptb_valid(eid_i):
                ptb_picks.append(eid_i)
                if len(ptb_picks) >= 15:
                    break

    ga_meta = exp_app._get_georgia_meta()
    ga_picks: List[str] = [
        str(rid) for rid in ("E00021", "E00005", "E10019", "E00067", "E00010", "E00006")
        if rid in ga_meta.index
    ]

    ga_clean = ga_meta[ga_meta["severity_tier"].astype(str) == "0_clean"]
    for tag in ("AF", "LBBB", "RBBB", "IAVB", "LQT", "NSR", "MI", "STTC"):
        if len(ga_picks) >= 10:
            break
        cands = ga_clean[ga_clean["_tag_set"].apply(lambda s, t=tag: t in s)].index.tolist()
        for rid in cands:
            r_str = str(rid)
            if r_str in ga_picks:
                continue
            if exp_app._load_delineation_any(r_str, 500, 1, dataset="georgia").get("quality_ok", False):
                ga_picks.append(r_str)
                break

    for tier_name, tier_quota in (("1_minor_warning", 2), ("2_moderate_anomaly", 2), ("3_severe_error", 1)):
        tier_df = ga_meta[ga_meta["severity_tier"].astype(str) == tier_name]
        added = 0
        for rid in tier_df.index.tolist():
            r_str = str(rid)
            if r_str not in ga_picks:
                ga_picks.append(r_str)
                added += 1
                if added >= tier_quota or len(ga_picks) >= 15:
                    break

    if len(ga_picks) < 15:
        for rid in ga_clean.index.tolist() + ga_meta.index.tolist():
            r_str = str(rid)
            if r_str not in ga_picks:
                ga_picks.append(r_str)
                if len(ga_picks) >= 15:
                    break

    return ptb_picks[:15], ga_picks[:15]


def _export_window_head(model: torch.nn.Module) -> dict:
    mtype = getattr(model, "model_type", "cbm")
    inactive = [int(i) for i in getattr(model, "inactive_concepts", ()) or ()]
    if mtype == "blackbox":
        return {"model_type": "blackbox"}
    bias = float(model.classifier.linear.bias.data[0].item())
    if mtype == "cem":
        return {
            "model_type":        "cem",
            "bias":              round(bias, 6),
            "inactive_concepts": inactive,
        }
    use_unc = bool(getattr(model.classifier, "use_uncertainty", True))
    w = model.classifier.linear.weight.data[0].detach().cpu().numpy()
    K = model.num_concepts
    return {
        "model_type":        "cbm",
        "bias":              round(bias, 6),
        "use_uncertainty":   use_unc,
        "w_prob":            [round(float(v), 6) for v in w[:K]],
        "w_var":             ([round(float(v), 6) for v in w[K : 2 * K]] if use_unc else []),
        "inactive_concepts": inactive,
    }


@torch.no_grad()
def _compute_window_sweep_with_export(
    model: torch.nn.Module,
    cfg: dict,
    windows_np: np.ndarray,
    device: str,
) -> Tuple[dict, np.ndarray]:
    data_cfg = cfg.get("data", {})
    normed = exp_app._select_model_leads(z_normalize(windows_np), data_cfg)
    xb = torch.from_numpy(normed).to(device)
    mtype = getattr(model, "model_type", "cbm")
    K = model.num_concepts
    N = xb.shape[0]

    if mtype in ("cbm", "cem"):
        z = model.encoder(xb)
        t_mean, t_std, _, c_probs, c_var = model.mc_task_from_z(z, n_mc=40)
        c_std = c_var.clamp(min=0.0).sqrt()
    else:
        tp, c_probs, _ = model(xb)
        t_mean = tp.squeeze(-1)
        t_std = torch.zeros_like(t_mean)
        c_std = None

    # Batched MC-Dropout Epistemic uncertainty across all sweep windows (chunked bs=64)
    mc_t_std_parts, mc_t_epi_parts, mc_c_std_parts = [], [], []
    n_ale = 20 if mtype in ("cbm", "cem") else 1
    for s in range(0, N, 64):
        u = model.predict_with_uncertainty(xb[s : s + 64], n_mc_epistemic=10, n_mc_aleatoric=n_ale)
        mc_t_std_parts.append(u["task_prob_std"].cpu().numpy())
        mc_t_epi_parts.append(u["task_epistemic"].cpu().numpy())
        if mtype in ("cbm", "cem"):
            mc_c_std_parts.append(u["concept_probs_std"].cpu().numpy())

    t_std_mc = np.concatenate(mc_t_std_parts, axis=0)
    t_epi_mc = np.concatenate(mc_t_epi_parts, axis=0)

    if mtype == "blackbox":
        entry: Dict[str, Any] = {
            "model_type":       mtype,
            "task_probs":       _round_list(t_mean.cpu().numpy(), 4),
            "task_stds":        _round_list(t_std.cpu().numpy(), 4),
            "task_stds_mc":     _round_list(t_std_mc, 4),
            "task_epistemic_mc": _round_list(t_epi_mc, 4),
            "concept_probs":    None,
            "concept_stds":     None,
            "concept_stds_mc":  None,
        }
    else:
        c_std_np = c_std.cpu().numpy()
        c_std_mc_np = np.concatenate(mc_c_std_parts, axis=0)
        entry = {
            "model_type":        mtype,
            "_cstd_scale":       255,
            "task_probs":        _round_list(t_mean.cpu().numpy(), 4),
            "task_stds":         _round_list(t_std.cpu().numpy(), 4),
            "task_stds_mc":      _round_list(t_std_mc, 4),
            "task_epistemic_mc": _round_list(t_epi_mc, 4),
            "concept_probs":     _round_list(c_probs.cpu().numpy(), 3),
            "concept_stds":      np.clip(np.round(c_std_np * 255.0), 0, 255).astype(np.int32).tolist(),
            "concept_stds_mc":   np.clip(np.round(c_std_mc_np * 255.0), 0, 255).astype(np.int32).tolist(),
        }

    if mtype == "cem":
        c_pos, c_neg, _, _ = model.concept_layer.compute_embeddings_and_scores(z)
        emb_size = model.concept_layer.emb_size
        w = model.classifier.linear.weight.data[0].view(1, K, emb_size)
        a_pos = (c_pos * w).sum(dim=-1).cpu().numpy()
        a_neg = (c_neg * w).sum(dim=-1).cpu().numpy()
        entry["a_pos"] = _round_list(a_pos, 4)
        entry["a_neg"] = _round_list(a_neg, 4)
    elif mtype == "cbm":
        _, c_lv = model.concept_layer(z)
        c_vars = torch.exp(c_lv.clamp(-25.0, LOG_VAR_MAX)).cpu().numpy()
        entry["c_vars"] = _round_list(c_vars, 5)

    return entry, xb.cpu().numpy()


def _verify_intervention_parity(
    model: torch.nn.Module,
    head: dict,
    sw_entry: dict,
    xb_np: np.ndarray,
    device: str,
    run_id: str,
) -> None:
    """Assert exported browser linear math matches PyTorch `model.intervene()`."""
    mtype = head["model_type"]
    if mtype == "blackbox":
        return
    idx = min(20, xb_np.shape[0] - 1)
    K = len(sw_entry["concept_probs"][idx])
    int_dict = {0: 1.0, 7: 0.0, 8: 1.0, 4: 0.0}  # force P=1, QRS=0, ST=1, T=0

    # PyTorch reference
    x1 = torch.from_numpy(xb_np[idx : idx + 1]).to(device)
    lbl = torch.zeros(1, K, device=device)
    msk = torch.zeros(1, K, device=device)
    for k, val in int_dict.items():
        lbl[0, k] = val
        msk[0, k] = 1.0
    pt_prob = float(model.intervene(x1, lbl, msk)[0, 0].item())

    # Exported JS-equivalent linear formula
    inactive = set(head.get("inactive_concepts", []))
    c_probs = sw_entry["concept_probs"][idx]
    logit = float(head["bias"])
    if mtype == "cem":
        a_pos = sw_entry["a_pos"][idx]
        a_neg = sw_entry["a_neg"][idx]
        for k in range(K):
            p_eff = 0.0 if k in inactive else int_dict.get(k, c_probs[k])
            logit += p_eff * a_pos[k] + (1.0 - p_eff) * a_neg[k]
    else:
        w_p = head["w_prob"]
        w_v = head["w_var"]
        c_vars = sw_entry["c_vars"][idx]
        v_zero = math.exp(-20.0)
        for k in range(K):
            if k in inactive:
                p_eff, v_eff = 0.0, 0.0
            elif k in int_dict:
                p_eff, v_eff = int_dict[k], v_zero
            else:
                p_eff, v_eff = c_probs[k], c_vars[k]
            logit += w_p[k] * p_eff + (w_v[k] * v_eff if head["use_uncertainty"] else 0.0)

    exp_prob = 1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, logit))))
    diff = abs(pt_prob - exp_prob)
    assert diff < 1e-3, f"Parity failed for {run_id}: pt={pt_prob:.6f} vs exp={exp_prob:.6f} (diff={diff:.6f})"
    print(f"  [Parity OK] {run_id:20s}: PyTorch={pt_prob:.5f} vs Exported={exp_prob:.5f} (|Δ|={diff:.2e})")


@torch.no_grad()
def _compute_dense_export(
    run_id: str,
    ecg_id: int | str,
    ds_name: str,
    raw_ss: Optional[np.ndarray],
    device: str,
) -> dict:
    _, model, cfg = exp_app.get_dense_model_and_cfg(run_id)
    sr = int(cfg.get("data", {}).get("sampling_rate", 100))
    c_mode = str(cfg.get("data", {}).get("concept_mode", "wave6"))
    c_names = list(cfg.get("concept_names", get_dense_concept_names(c_mode)))
    mtype = getattr(model, "model_type", "dense_cem")
    m_has_seg = (mtype != "dense_blackbox")
    m_can_int = mtype in ("dense_cbm", "dense_cem")

    sig = z_normalize(exp_app._load_filtered_signal_any(ecg_id, sr, dataset=ds_name))
    T = sig.shape[-1]
    x = torch.from_numpy(sig).unsqueeze(0).to(device)

    base_out = model(x)
    base_tp = torch.sigmoid(base_out["task_logits"][0].float()).cpu().numpy()
    base_patho = {name: round(float(base_tp[i]), 4) for i, name in enumerate(SUPERCLASS_NAMES)}

    if not m_has_seg:
        unc = model.predict_with_uncertainty(x, n_mc_epistemic=8, n_mc_aleatoric=1)
        tp_std_mc = unc["task_probs_std"][0].cpu().numpy()
        return {
            "model_type":             mtype,
            "has_seg":                False,
            "supports_interventions": False,
            "concept_names":          c_names,
            "base_pathology":         base_patho,
            "pathology_std":          {name: 0.0 for name in SUPERCLASS_NAMES},
            "pathology_std_mc":       {name: round(float(tp_std_mc[i]), 4) for i, name in enumerate(SUPERCLASS_NAMES)},
        }

    cp = base_out["concept_probs"][0].float().cpu().numpy()
    lv = base_out["concept_log_vars"][0].float().cpu().numpy()
    cs = np.clip(cp * (1.0 - cp) * np.exp(0.5 * np.clip(lv, -10.0, 4.0)), 0.0, 0.5)

    unc = model.predict_with_uncertainty(x, n_mc_epistemic=8, n_mc_aleatoric=12)
    cs_mc = unc["concept_total_std"][0].cpu().numpy()
    tp_std_mc = unc["task_probs_std"][0].cpu().numpy()

    intervals = None
    if c_mode == "wave6" and cp.shape[0] == 6:
        iv = extract_wave_spans_and_intervals(cp.argmax(axis=0), fs=sr)
        intervals = {
            "qrs_ms":      exp_app._safe_round(iv.get("qrs_ms"), 1),
            "pr_ms":       exp_app._safe_round(iv.get("pr_ms"), 1),
            "qt_ms":       exp_app._safe_round(iv.get("qt_ms"), 1),
            "pred_hr_bpm": exp_app._safe_round(iv.get("pred_hr_bpm"), 1),
            "canon_cycle": exp_app._safe_round(iv.get("canonical_transition_rate"), 4),
        }

    interventions: Dict[str, Dict[str, float]] = {}
    interventions_win: Dict[str, List[List[float]]] = {}
    if m_can_int:
        n_bins = max(1, math.ceil(T / 100))
        x_b = x.expand(n_bins, -1, -1)
        base_cp_b = base_out["concept_probs"].expand(n_bins, -1, -1)
        bin_slices = [(b * 100, min(T, (b + 1) * 100)) for b in range(n_bins)]

        if raw_ss is not None:
            gt_arr = collapse_soft_seg_targets(raw_ss, concept_mode=c_mode, lead_idx=None)
            il_gt = torch.from_numpy(gt_arr).unsqueeze(0).to(device)
            im_gt = torch.ones((1, 1, T), device=device)
            tp_gt = torch.sigmoid(model(x, intervene_labels=il_gt, intervene_mask=im_gt)["task_logits"][0].float()).cpu().numpy()
            interventions["gt"] = {name: round(float(tp_gt[i]), 4) for i, name in enumerate(SUPERCLASS_NAMES)}

            il_gt_b = il_gt.expand(n_bins, -1, -1)
            im_gt_b = torch.zeros((n_bins, 1, T), device=device)
            for b, (s0, s1) in enumerate(bin_slices):
                im_gt_b[b, :, s0:s1] = 1.0
            tp_gt_w = torch.sigmoid(model(x_b, intervene_labels=il_gt_b, intervene_mask=im_gt_b)["task_logits"].float()).cpu().numpy()
            interventions_win["gt"] = _round_list(tp_gt_w, 3)

        for k, cname in enumerate(c_names):
            # Full-record ablate k -> 0
            il_a = base_out["concept_probs"].clone()
            il_a[:, k, :] = 0.0
            im_a = torch.zeros_like(il_a)
            im_a[:, k, :] = 1.0
            tp_a = torch.sigmoid(model(x, intervene_labels=il_a, intervene_mask=im_a)["task_logits"][0].float()).cpu().numpy()
            interventions[f"ablate:{cname}"] = {name: round(float(tp_a[i]), 4) for i, name in enumerate(SUPERCLASS_NAMES)}

            # Full-record force k -> 1
            il_f = torch.zeros_like(base_out["concept_probs"])
            il_f[:, k, :] = 1.0
            im_f = torch.ones_like(il_f)
            tp_f = torch.sigmoid(model(x, intervene_labels=il_f, intervene_mask=im_f)["task_logits"][0].float()).cpu().numpy()
            interventions[f"force:{cname}"] = {name: round(float(tp_f[i]), 4) for i, name in enumerate(SUPERCLASS_NAMES)}

            # Batched 1.0 s bin window-scoped interventions (n_bins=10 parallel forward passes)
            il_aw = base_cp_b.clone()
            im_aw = torch.zeros_like(il_aw)
            il_fw = base_cp_b.clone()
            im_fw = torch.zeros_like(il_fw)
            for b, (s0, s1) in enumerate(bin_slices):
                il_aw[b, k, s0:s1] = 0.0
                im_aw[b, k, s0:s1] = 1.0
                il_fw[b, :, s0:s1] = 0.0
                il_fw[b, k, s0:s1] = 1.0
                im_fw[b, :, s0:s1] = 1.0
            tp_aw = torch.sigmoid(model(x_b, intervene_labels=il_aw, intervene_mask=im_aw)["task_logits"].float()).cpu().numpy()
            tp_fw = torch.sigmoid(model(x_b, intervene_labels=il_fw, intervene_mask=im_fw)["task_logits"].float()).cpu().numpy()
            interventions_win[f"ablate:{cname}"] = _round_list(tp_aw, 3)
            interventions_win[f"force:{cname}"] = _round_list(tp_fw, 3)

    seg_spans = _compute_seg_spans(cp) if (c_mode == "wave6" and cp.shape[0] == 6) else None
    std_line_q = np.clip(np.round(cs.mean(axis=0) * 10000.0), 0, 10000).astype(np.int32).tolist()
    std_line_mc_q = np.clip(np.round(cs_mc.mean(axis=0) * 10000.0), 0, 10000).astype(np.int32).tolist()

    return {
        "model_type":             mtype,
        "has_seg":                True,
        "supports_interventions": m_can_int,
        "concept_names":          c_names,
        "_prob_scale":            255,
        "_std_scale":             10000,
        "concept_probs":          _quantize_u8_dict(cp, c_names, omit_tp=True),
        "concept_stds":           _quantize_u8_dict(cs, c_names, omit_tp=True),
        "seg_spans":              seg_spans,
        "mean_std_timeline":      std_line_q,
        "mean_std_timeline_mc":   std_line_mc_q,
        "intervals":              intervals,
        "base_pathology":         base_patho,
        "pathology_std":          {name: 0.0 for name in SUPERCLASS_NAMES},
        "pathology_std_mc":       {name: round(float(tp_std_mc[i]), 4) for i, name in enumerate(SUPERCLASS_NAMES)},
        "interventions":          interventions,
        "interventions_win":      interventions_win,
    }


def main() -> None:
    t0 = time.perf_counter()
    device = exp_app.DEVICE
    print(f"=== Building Static Vercel Explorer Bundle (device={device}) ===")

    if VERCEL_DIR.exists():
        # Preserve vercel.json / README.md if re-running, rebuild assets & data cleanly
        for sub in ("static", "data"):
            if (VERCEL_DIR / sub).exists():
                shutil.rmtree(VERCEL_DIR / sub)
    REC_OUT.mkdir(parents=True, exist_ok=True)

    # 1. Discover & filter models
    all_win_meta = exp_app.list_models()
    win_runs = exp_app.discover_runs()
    win_ids = [mid for mid in WINDOW_EXPORT_IDS if mid in win_runs]
    win_entries = [e for e in all_win_meta["models"] if e["id"] in win_ids]

    all_dense_meta = exp_app.list_dense_models()
    dense_runs = exp_app.discover_dense_runs()
    dense_ids = [mid for mid in DENSE_EXPORT_IDS if mid in dense_runs]
    dense_entries = [e for e in all_dense_meta["models"] if e["id"] in dense_ids]

    print(f"Window models: {win_ids}")
    print(f"Dense models:  {dense_ids}")

    window_heads: Dict[str, dict] = {}
    summaries: Dict[str, dict] = {}
    for mid in win_ids:
        _, wm, _ = exp_app.get_model_and_cfg(mid)
        window_heads[mid] = _export_window_head(wm)
        summaries[mid] = exp_app.model_summary(mid)

    # 2. Select 15 PTB-XL + 15 Georgia curated records
    ptb_picks, ga_picks = _select_curated_records()
    print(f"Curated PTB-XL ({len(ptb_picks)}): {ptb_picks}")
    print(f"Curated Georgia ({len(ga_picks)}): {ga_picks}")

    archive = exp_app._get_soft_seg_archive()
    records_by_dataset: Dict[str, List[dict]] = {"ptbxl": [], "georgia": []}

    ptb_meta = exp_app._get_meta()
    for eid in ptb_picks:
        row = ptb_meta.loc[eid]
        records_by_dataset["ptbxl"].append({
            "ecg_id":  int(eid),
            "dataset": "ptbxl",
            "fold":    int(row.get("strat_fold", 10)),
            "age":     float(row["age"]) if not np.isnan(row.get("age", np.nan)) else 0.0,
            "sex":     int(row["sex"]) if not np.isnan(row.get("sex", np.nan)) else 0,
            "dx":      row["_dx_list"],
            "tags":    sorted(row["_tag_set"]),
        })

    ga_meta = exp_app._get_georgia_meta()
    for rid in ga_picks:
        row = ga_meta.loc[rid]
        records_by_dataset["georgia"].append({
            "ecg_id":        str(rid),
            "dataset":       "georgia",
            "severity_tier": str(row["severity_tier"]),
            "age":           float(row["age"]) if not np.isnan(row.get("age", np.nan)) else 0.0,
            "sex":           int(row["_sex_int"]),
            "sex_label":     str(row["_sex_label"]),
            "dx":            row["_dx_list"],
            "label_names":   row["_snomed_labels"],
            "tags":          sorted(row["_tag_set"]),
            "est_hr_bpm":    exp_app._safe_round(row.get("est_hr_bpm"), 1),
        })

    # 3. Export each record JSON
    parity_verified = False
    all_targets = [("ptbxl", eid) for eid in ptb_picks] + [("georgia", rid) for rid in ga_picks]
    for ds_name, eid in all_targets:
        rec_payload = exp_app.get_record(str(eid), model_id=win_ids[0], dataset=ds_name)
        # Re-downsample 250 Hz -> 125 Hz (ds=4 vs 500 Hz) & drop duplicate soft_seg payload
        sig_ds2 = np.asarray(rec_payload["signal"], dtype=np.float32)
        sig_ds4 = sig_ds2[:, ::2]
        rec_payload["downsample"] = 4
        rec_payload["n_samples_ds"] = int(sig_ds4.shape[1])
        rec_payload["signal"] = _round_list(sig_ds4, 2)
        rec_payload["soft_seg"] = None

        # Filtered 500 Hz windows (stride = 25)
        sig_500 = exp_app._load_filtered_signal_any(eid, 500, dataset=ds_name)
        _, T500 = sig_500.shape
        t_fixed = 500
        starts = list(range(0, T500 - t_fixed + 1, STATIC_STRIDE))
        windows_np = np.stack([sig_500[:, s : s + t_fixed] for s in starts], axis=0)

        # GT delineation per window
        delin = exp_app._load_delineation_any(eid, 500, 0, dataset=ds_name)
        gt_concepts, gt_task = None, None
        if delin.get("quality_ok", False):
            gc, gt = [], []
            for s in starts:
                c, w = get_concept_labels(s, s + t_fixed, delin["delineation"], delin["r_peaks"], delin.get("u_peaks"))
                gc.append(c)
                gt.append(int(is_complete_heartbeat_window(c, w)))
            gt_concepts, gt_task = gc, gt

        sweeps: Dict[str, dict] = {}
        for mid in win_ids:
            _, wm, wcfg = exp_app.get_model_and_cfg(mid)
            sw_entry, xb_np = _compute_window_sweep_with_export(wm, wcfg, windows_np, device)
            sw_entry["stride"] = STATIC_STRIDE
            sw_entry["t_fixed"] = t_fixed
            sw_entry["sampling_rate"] = 500
            sw_entry["starts"] = starts
            sw_entry["concept_names"] = exp_app._concept_names_for_cfg(wcfg)
            sweeps[mid] = sw_entry

            if not parity_verified and mid in ("cem_softseg", "cbm_softseg_masked"):
                _verify_intervention_parity(wm, window_heads[mid], sw_entry, xb_np, device, mid)

        if not parity_verified:
            parity_verified = True

        # Dense per-timestamp models (100 Hz)
        raw_ss = archive.load(int(eid)) if (ds_name == "ptbxl" and archive is not None and int(eid) in archive) else None
        dense_gt_probs = None
        dense_gt_spans = None
        if raw_ss is not None:
            gt_w6 = collapse_soft_seg_targets(raw_ss, concept_mode="wave6", lead_idx=None)
            dense_gt_probs = _quantize_u8_dict(gt_w6, WAVE6_NAMES, omit_tp=True)
            dense_gt_spans = _compute_seg_spans(gt_w6)

        dense_map: Dict[str, dict] = {}
        for dmid in dense_ids:
            dense_map[dmid] = _compute_dense_export(dmid, eid, ds_name, raw_ss, device)

        out_obj = {
            "record":            rec_payload,
            "sweep_gt_task":     gt_task,
            "sweep_gt_concepts": gt_concepts,
            "sweeps":            sweeps,
            "dense_prob_scale":  255,
            "dense_gt_probs":    dense_gt_probs,
            "dense_gt_spans":    dense_gt_spans,
            "dense":             dense_map,
        }
        out_file = REC_OUT / f"{ds_name}_{eid}.json"
        out_file.write_text(json.dumps(out_obj, separators=(",", ":"), allow_nan=False))
        print(f"  Exported {out_file.name:24s} ({out_file.stat().st_size / 1024:.1f} KB)")

    # 4. Write manifest.json & external_benchmark.json
    datasets_payload = exp_app.list_datasets()
    for d in datasets_payload["datasets"]:
        if d["id"] == "ptbxl":
            d["splits"] = [
                {"value": "10",  "label": "Test (Fold 10)"},
                {"value": "all", "label": "All Curated"},
            ]
        elif d["id"] == "georgia":
            d["splits"] = [
                {"value": "0_clean",            "label": "Clean (0_clean)"},
                {"value": "1_minor_warning",    "label": "Minor Warning (1)"},
                {"value": "2_moderate_anomaly", "label": "Moderate Anomaly (2)"},
                {"value": "3_severe_error",     "label": "Severe Error (3)"},
                {"value": "all",                "label": "All Curated"},
            ]

    manifest = {
        "default_dataset": datasets_payload["default_dataset"],
        "datasets":        datasets_payload["datasets"],
        "models": {
            "default_model":   "cem_softseg" if "cem_softseg" in win_ids else win_ids[0],
            "compare_default": "cbm_softseg_masked" if "cbm_softseg_masked" in win_ids else None,
            "models":          win_entries,
        },
        "window_heads": window_heads,
        "summaries":    summaries,
        "dense_models": {
            "default_model":    "dense_cem_s42" if "dense_cem_s42" in dense_ids else dense_ids[0],
            "compare_default":  "dense_cbm_detached_s42" if "dense_cbm_detached_s42" in dense_ids else None,
            "blackbox_default": "dense_blackbox_s42" if "dense_blackbox_s42" in dense_ids else None,
            "models":           dense_entries,
        },
        "records_by_dataset": records_by_dataset,
    }
    manifest_file = DATA_OUT / "manifest.json"
    manifest_file.write_text(json.dumps(manifest, separators=(",", ":"), allow_nan=False))

    ext_bench_file = DATA_OUT / "external_benchmark.json"
    ext_payload = exp_app.get_external_benchmark()
    ext_bench_file.write_text(json.dumps(ext_payload, separators=(",", ":"), allow_nan=False))

    # 5. Copy static assets & rewrite index.html cleanly for static Vercel serving
    shutil.copytree(EXPLORER_DIR / "static", VERCEL_DIR / "static")
    html = (EXPLORER_DIR / "index.html").read_text()
    preload_tags = (
        '<link rel="preload" href="./data/manifest.json" as="fetch" crossorigin="anonymous" />\n  '
        '<link rel="preload" href="./data/records/ptbxl_9.json" as="fetch" crossorigin="anonymous" />\n  '
    )
    html = html.replace(
        '<link rel="stylesheet" href="/static/style.css"',
        preload_tags + '<link rel="stylesheet" href="./static/style.css"',
    )
    html = html.replace('href="/static/style.css"', 'href="./static/style.css"')
    html = html.replace('src="/static/vendor/d3.v7.min.js"', 'src="./static/vendor/d3.v7.min.js"')
    html = html.replace(
        'src="/static/main.js"></script>',
        'src="./static/main.js"></script>',
    ).replace(
        '<script src="./static/main.js"></script>',
        '<script>window.EXPLORER_STATIC_MODE = true;</script>\n  <script src="./static/main.js"></script>',
    )
    (VERCEL_DIR / "index.html").write_text(html)

    # 6. Write vercel.json & README.md
    vercel_cfg = {
        "$schema": "https://openapi.vercel.sh/vercel.json",
        "cleanUrls": True,
        "headers": [
            {
                "source": "/static/vendor/(.*)",
                "headers": [{"key": "Cache-Control", "value": "public, max-age=31536000, immutable"}],
            },
            {
                "source": "/static/(.*)",
                "headers": [{"key": "Cache-Control", "value": "public, max-age=3600, stale-while-revalidate=86400"}],
            },
            {
                "source": "/data/(.*)",
                "headers": [{"key": "Cache-Control", "value": "public, max-age=86400, stale-while-revalidate=604800"}],
            },
        ],
    }
    (VERCEL_DIR / "vercel.json").write_text(json.dumps(vercel_cfg, indent=2) + "\n")

    readme_txt = """# ECG Heartbeat CBM & CEM Explorer — Vercel Static Deployment

Zero-cold-start, static CDN build of the interactive **ECG Heartbeat CBM & CEM Explorer** supporting both **PTB-XL Fold 10** and **Georgia 12-Lead ECG (Zero-Shot OOD)** cohorts (`30` curated clinical recordings across `NORM`, `AFIB`, `LBBB/RBBB`, `MI`, `STTC`, `HYP`, `IAVB`, `LQT`, and all 4 Georgia quality tiers).

## Why Static + Exact Linear Head Export?

PyTorch CPU wheels exceed Vercel's `250 MB` serverless limit and trigger slow cold starts. Instead, [explorer/build_vercel.py](../build_vercel.py) exports:
1. Precomputed `500 Hz` sliding-window concept probabilities (`stride = 25` samples = `50 ms`), batched **MC-Dropout Epistemic** decompositions, exact per-window **CEM** positive/negative logits `(a⁺, a⁻)`, and **CBM** linear classifier weights `(w_prob, w_var, b)` so **all test-time window concept interventions execute deterministically in browser JavaScript with `< 1e-4` error relative to PyTorch**.
2. Precomputed `100 Hz` **DenseCEM / DenseCBM / DenseMultiTask / DenseBlackBox** `uint8`-quantized soft segmentations, precomputed wave-shading spans, MC-Dropout boundary uncertainty `σ(t)`, clinical `PR / QRS / QT / HR` intervals, and both full-record and `1.0 s` window-scoped wave-mask interventions (`ablate` / `force` / `Apply GT Masks`).

You can also connect a live FastAPI GPU/CPU backend at any time by appending `?api=https://your-fastapi-host` to the deployed Vercel URL.

## Deploy to Vercel

### Option 1 — Vercel Dashboard (Recommended)
1. Push this repository to GitHub/GitLab/Bitbucket.
2. Import the repository in the Vercel Dashboard.
3. Set **Root Directory** to `explorer/vercel`, leave **Framework Preset** as `Other`, leave **Build Command** empty, and click **Deploy**.

### Option 2 — Vercel CLI
```bash
cd explorer/vercel
npx vercel --prod
```

## Regenerating the Static Bundle

Whenever checkpoints, showcase records, or benchmark summaries change, regenerate `explorer/vercel/`:

```bash
CUDA_VISIBLE_DEVICES=0 .venv_new/bin/python explorer/build_vercel.py
```

And preview locally before deploying:

```bash
python3 -m http.server 8041 --directory explorer/vercel
```
"""
    (VERCEL_DIR / "README.md").write_text(readme_txt)

    total_kb = sum(p.stat().st_size for p in VERCEL_DIR.rglob("*") if p.is_file()) / 1024.0
    print(f"\nDone! Built explorer/vercel/ in {time.perf_counter() - t0:.1f}s | Total uncompressed size: {total_kb / 1024:.2f} MB")


if __name__ == "__main__":
    main()
