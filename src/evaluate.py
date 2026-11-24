"""
evaluate.py — Comprehensive evaluation for ECG Heartbeat CBM, CEM, and Black-Box models.

Computes:
  - Task performance: accuracy, F1, precision, recall, AUROC
  - Calibration: Brier score, Expected Calibration Error (ECE)
  - Masked per-concept metrics: F1, accuracy, AUROC (strictly over valid delineations w_k > 0)
  - Validation losses (total, task, concept, consistency) when a criterion is supplied
  - Test-time Concept Intervention curves (uncertainty-guided vs random concept ordering)
  - Epistemic/Aleatoric uncertainty quality & selective error-detection AUROC

CLI usage:
    python src/evaluate.py --run-dir runs/my_run [--split test] [--interventions] [--uncertainty]
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import yaml
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch.utils.data import DataLoader
from tqdm import tqdm

from data import load_or_build_datasets
from losses import CombinedCBMLoss
from models import CONCEPT_NAMES, build_model

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Calibration helper
# ─────────────────────────────────────────────────────────────────────────────

def expected_calibration_error(probs: np.ndarray, labels: np.ndarray, n_bins: int = 15) -> float:
    """Compute Expected Calibration Error (ECE) for binary probabilities."""
    probs = np.asarray(probs, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.float64)
    if len(probs) == 0:
        return 0.0
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(probs)
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (probs >= lo) & (probs <= hi if hi == 1.0 else probs < hi)
        if not np.any(mask):
            continue
        conf = probs[mask].mean()
        acc  = labels[mask].mean()
        ece += (mask.sum() / n) * abs(acc - conf)
    return float(ece)


def _safe_auroc(labels: np.ndarray, probs: np.ndarray) -> float:
    try:
        if len(np.unique(labels)) < 2:
            return float("nan")
        return float(roc_auc_score(labels, probs))
    except ValueError:
        return float("nan")


# ─────────────────────────────────────────────────────────────────────────────
# Standard evaluation (task + masked concepts + calibration + optional loss)
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    loader: DataLoader,
    device: str,
    concept_names: Optional[List[str]] = None,
    criterion: Optional[CombinedCBMLoss] = None,
    epoch: int = 0,
    stage: str = "joint",
) -> Dict[str, float]:
    """Compute task, calibration, loss, and confidence-masked per-concept metrics."""
    model.eval()
    concept_names = concept_names or CONCEPT_NAMES
    is_blackbox = getattr(model, "model_type", "cbm") == "blackbox"

    task_probs_all, task_labels_all = [], []
    c_probs_all, c_labels_all, c_weights_all = [], [], []
    loss_sums = {"total": 0.0, "task": 0.0, "concept": 0.0, "consistency": 0.0}
    n_batches = 0

    for batch in tqdm(loader, desc="Eval", leave=False):
        x         = batch["signal"].to(device)
        y_task    = batch["task_label"].to(device)
        y_conc    = batch["concept_labels"].to(device)
        w_conc    = batch["concept_weights"].to(device)

        if criterion is not None and hasattr(model, "forward_with_logits"):
            task_prob, c_probs, c_log_vars, c_logits = model.forward_with_logits(x)
            losses = criterion(
                task_prob, c_probs, c_log_vars, c_logits,
                y_task, y_conc, w_conc,
                epoch=epoch, stage=stage, compute_concepts=not is_blackbox,
            )
            for k in loss_sums:
                v = losses[k]
                loss_sums[k] += float(v.item() if torch.is_tensor(v) else v)
            n_batches += 1
        else:
            task_prob, c_probs, _ = model(x)

        task_probs_all.append(task_prob.squeeze(-1).cpu())
        task_labels_all.append(y_task.cpu())
        c_probs_all.append(c_probs.cpu())
        c_labels_all.append(y_conc.cpu())
        c_weights_all.append(w_conc.cpu())

    task_probs   = torch.cat(task_probs_all).numpy()
    task_labels  = torch.cat(task_labels_all).numpy()
    c_probs_mat  = torch.cat(c_probs_all).numpy()
    c_labels_mat = torch.cat(c_labels_all).numpy()
    c_weights_mat = torch.cat(c_weights_all).numpy()

    metrics: Dict[str, float] = {}
    if criterion is not None and n_batches > 0:
        for k, val in loss_sums.items():
            metrics[f"val/loss_{k}"] = val / n_batches

    task_preds = (task_probs > 0.5).astype(int)
    metrics["task/accuracy"]  = float(accuracy_score(task_labels, task_preds))
    metrics["task/f1"]        = float(f1_score(task_labels, task_preds, zero_division=0))
    metrics["task/precision"] = float(precision_score(task_labels, task_preds, zero_division=0))
    metrics["task/recall"]    = float(recall_score(task_labels, task_preds, zero_division=0))
    metrics["task/auroc"]     = _safe_auroc(task_labels, task_probs)
    metrics["task/brier"]     = float(brier_score_loss(task_labels, np.clip(task_probs, 0.0, 1.0)))
    metrics["task/ece"]       = expected_calibration_error(task_probs, task_labels)

    if not is_blackbox:
        K = min(c_probs_mat.shape[1], c_labels_mat.shape[1])
        concept_f1s, concept_accs = [], []
        for k in range(K):
            name = concept_names[k] if k < len(concept_names) else f"C{k}"
            valid = c_weights_mat[:, k] > 0
            if not np.any(valid):
                continue
            probs_k  = c_probs_mat[valid, k]
            preds_k  = (probs_k > 0.5).astype(int)
            labels_k = c_labels_mat[valid, k].astype(int)

            f1_k  = float(f1_score(labels_k, preds_k, zero_division=0))
            acc_k = float(accuracy_score(labels_k, preds_k))
            metrics[f"concept/{name}/f1"]       = f1_k
            metrics[f"concept/{name}/accuracy"] = acc_k
            metrics[f"concept/{name}/auroc"]    = _safe_auroc(labels_k, probs_k)
            concept_f1s.append(f1_k)
            concept_accs.append(acc_k)

        if concept_f1s:
            metrics["concept/macro_f1"]       = float(np.mean(concept_f1s))
            metrics["concept/macro_accuracy"] = float(np.mean(concept_accs))

    return metrics


# ─────────────────────────────────────────────────────────────────────────────
# Test-time Concept Intervention evaluation (CBM & CEM)
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate_interventions(
    model: torch.nn.Module,
    loader: DataLoader,
    device: str,
    n_random_trials: int = 3,
    seed: int = 42,
) -> Dict[str, list]:
    """
    Measure task accuracy, F1, and AUROC as 0..K concepts are intervened upon
    using ground-truth concept labels (restricted to valid delineations `w_k > 0`).

    Compares:
      - Uncertainty-ordered intervention: intervenes first on concepts with highest
        combined predictive entropy + heteroscedastic std.
      - Random-ordered intervention: averaged over `n_random_trials` random orderings.
    """
    model.eval()
    K = model.num_concepts
    rng = np.random.default_rng(seed)

    # Pre-collect latent representations so sweeping 0..K interventions is fast
    z_list, y_task_list, y_conc_list, w_conc_list = [], [], [], []
    c_probs_list, c_lv_list = [], []

    for batch in tqdm(loader, desc="Interventions", leave=False):
        x = batch["signal"].to(device)
        z = model.encoder(x)
        _, c_probs, c_lv = model(x)
        z_list.append(z.cpu())
        c_probs_list.append(c_probs.cpu())
        c_lv_list.append(c_lv.cpu())
        y_task_list.append(batch["task_label"])
        y_conc_list.append(batch["concept_labels"])
        w_conc_list.append(batch["concept_weights"])

    z_all       = torch.cat(z_list, dim=0)
    c_probs_all = torch.cat(c_probs_list, dim=0)
    c_lv_all    = torch.cat(c_lv_list, dim=0)
    y_task_np   = torch.cat(y_task_list, dim=0).numpy()
    y_conc_all  = torch.cat(y_conc_list, dim=0)
    w_conc_all  = torch.cat(w_conc_list, dim=0)
    N = z_all.size(0)

    # Combined concept uncertainty score per (sample, concept); invalid concepts get -inf so they go last
    p_clamp = c_probs_all.clamp(1e-6, 1.0 - 1e-6)
    entropy = -(p_clamp * torch.log(p_clamp) + (1.0 - p_clamp) * torch.log(1.0 - p_clamp))
    aleatoric_std = torch.exp(0.5 * c_lv_all.clamp(-20.0, 6.0))
    unc_score = entropy + 0.25 * aleatoric_std
    unc_score = torch.where(w_conc_all > 0, unc_score, torch.full_like(unc_score, -1e9))
    unc_order = torch.argsort(unc_score, dim=1, descending=True)   # (N, K)

    def _predict_from_z_with_mask(mask_full: torch.Tensor, batch_size: int = 1024) -> np.ndarray:
        preds = []
        mtype = getattr(model, "model_type", "cbm")
        for s in range(0, N, batch_size):
            zb  = z_all[s : s + batch_size].to(device)
            yb  = y_conc_all[s : s + batch_size].to(device)
            mb  = mask_full[s : s + batch_size].to(device)
            if mtype == "cbm":
                cp, clv = model.concept_layer(zb)
                mask_f = mb.float()
                cp_i   = (1.0 - mask_f) * cp + mask_f * yb.float()
                clv_i  = torch.where(mb.bool(), torch.full_like(clv, -20.0), clv)
                tp = model.classifier(cp_i, clv_i)
            elif mtype == "cem":
                mixed, _, _, _ = model.concept_layer.forward_all(
                    zb, intervene_labels=yb, intervene_mask=mb
                )
                tp = model.classifier(mixed)
            else:
                tp = torch.sigmoid(model.classifier(zb))
            preds.append(tp.squeeze(-1).cpu().numpy())
        return np.concatenate(preds)

    results = {
        "num_intervened":  list(range(K + 1)),
        "uncertainty_acc": [], "uncertainty_f1": [], "uncertainty_auroc": [],
        "random_acc":      [], "random_f1":      [], "random_auroc":      [],
    }

    valid_bool = w_conc_all > 0
    for m in range(K + 1):
        # 1. Uncertainty-guided top-m valid concepts
        if m == 0:
            u_mask = torch.zeros_like(w_conc_all)
        else:
            top_idx = unc_order[:, :m]
            u_mask = torch.zeros_like(w_conc_all)
            u_mask.scatter_(1, top_idx, 1.0)
            u_mask = u_mask * valid_bool.float()

        u_probs = _predict_from_z_with_mask(u_mask)
        u_preds = (u_probs > 0.5).astype(int)
        results["uncertainty_acc"].append(float(accuracy_score(y_task_np, u_preds)))
        results["uncertainty_f1"].append(float(f1_score(y_task_np, u_preds, zero_division=0)))
        results["uncertainty_auroc"].append(_safe_auroc(y_task_np, u_probs))

        # 2. Random ordering (averaged over trials)
        if m == 0 or m == K:
            r_mask = torch.zeros_like(w_conc_all) if m == 0 else valid_bool.float()
            r_probs = _predict_from_z_with_mask(r_mask) if m == K else u_probs
            r_preds = (r_probs > 0.5).astype(int)
            results["random_acc"].append(float(accuracy_score(y_task_np, r_preds)))
            results["random_f1"].append(float(f1_score(y_task_np, r_preds, zero_division=0)))
            results["random_auroc"].append(_safe_auroc(y_task_np, r_probs))
        else:
            accs, f1s, aurocs = [], [], []
            for _ in range(n_random_trials):
                noise = torch.from_numpy(rng.random((N, K)).astype(np.float32))
                noise = torch.where(valid_bool, noise, torch.full_like(noise, -1e9))
                r_order = torch.argsort(noise, dim=1, descending=True)
                r_mask = torch.zeros_like(w_conc_all)
                r_mask.scatter_(1, r_order[:, :m], 1.0)
                r_mask = r_mask * valid_bool.float()
                r_probs = _predict_from_z_with_mask(r_mask)
                r_preds = (r_probs > 0.5).astype(int)
                accs.append(float(accuracy_score(y_task_np, r_preds)))
                f1s.append(float(f1_score(y_task_np, r_preds, zero_division=0)))
                aurocs.append(_safe_auroc(y_task_np, r_probs))
            results["random_acc"].append(float(np.mean(accs)))
            results["random_f1"].append(float(np.mean(f1s)))
            results["random_auroc"].append(float(np.nanmean(aurocs)))

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Uncertainty quality evaluation
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate_uncertainty_quality(
    model: torch.nn.Module,
    loader: DataLoader,
    device: str,
    reliable_concepts: Optional[List[int]] = None,
    n_mc_epistemic: int = 15,
    n_mc_aleatoric: int = 30,
) -> Dict[str, float]:
    """Quantify uncertainty decomposition and selective error-detection AUROC."""
    reliable_concepts = reliable_concepts if reliable_concepts is not None else [2, 4, 7, 9]
    task_means, task_stds, task_ep, task_al, labels_all = [], [], [], [], []
    c_al_all = []

    for batch in tqdm(loader, desc="Uncertainty Eval", leave=False):
        x = batch["signal"].to(device)
        out = model.predict_with_uncertainty(x, n_mc_epistemic, n_mc_aleatoric)
        task_means.append(out["task_prob_mean"].cpu().numpy())
        task_stds.append(out["task_prob_std"].cpu().numpy())
        task_ep.append(out["task_epistemic"].cpu().numpy())
        task_al.append(out["task_aleatoric"].cpu().numpy())
        c_al_all.append(out["concept_aleatoric"].cpu().numpy())
        labels_all.append(batch["task_label"].numpy())

    t_mean = np.concatenate(task_means)
    t_std  = np.concatenate(task_stds)
    t_ep   = np.concatenate(task_ep)
    t_al   = np.concatenate(task_al)
    c_al   = np.concatenate(c_al_all, axis=0)
    y_true = np.concatenate(labels_all)

    errors = ((t_mean > 0.5).astype(int) != y_true).astype(int)
    res: Dict[str, float] = {
        "uncertainty/task_std_mean":        float(t_std.mean()),
        "uncertainty/task_epistemic_mean":  float(t_ep.mean()),
        "uncertainty/task_aleatoric_mean":  float(t_al.mean()),
        "uncertainty/error_detection_auroc": _safe_auroc(errors, t_std),
    }

    if getattr(model, "model_type", "cbm") != "blackbox" and c_al.shape[1] > 0:
        K = c_al.shape[1]
        rel_idx = [i for i in reliable_concepts if i < K]
        noisy_idx = [i for i in range(K) if i not in rel_idx]
        if rel_idx and noisy_idx:
            rel_val   = float(c_al[:, rel_idx].mean())
            noisy_val = float(c_al[:, noisy_idx].mean())
            res["uncertainty/concept_aleatoric_reliable"] = rel_val
            res["uncertainty/concept_aleatoric_noisy"]    = noisy_val
            res["uncertainty/noisy_vs_reliable_ratio"]    = noisy_val / max(rel_val, 1e-8)

    return res


# ─────────────────────────────────────────────────────────────────────────────
# Faithfulness (Counterfactual Flip Rate) & Leakage (Oracle Impurity Score)
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate_faithfulness_and_leakage(
    model: torch.nn.Module,
    loader: DataLoader,
    device: str,
    max_ois_samples: int = 4000,
    seed: int = 42,
) -> Dict[str, float]:
    """
    Evaluate causal faithfulness and concept representation purity:
      - Counterfactual flip rate: fraction of predicted-Complete windows that flip to
        Incomplete when forcing concepts→0 (and predicted-Incomplete that flip to
        Complete when forcing concepts→1).
      - Oracle Impurity Score (OIS, Zarlenga et al. 2023): mean excess off-diagonal
        AUROC when predicting concept j from concept i's bottleneck representation
        compared to predicting concept j from ground-truth concept i alone (lower = purer).
    """
    from sklearn.linear_model import LogisticRegression

    mtype = getattr(model, "model_type", "cbm")
    if mtype == "blackbox":
        return {
            "faithfulness/flip_rate_to_incomplete": 0.0,
            "faithfulness/flip_rate_to_complete":   0.0,
            "faithfulness/counterfactual_flip_rate": 0.0,
            "faithfulness/mean_prob_shift":         0.0,
        }

    model.eval()
    base_probs_l, zero_probs_l, one_probs_l = [], [], []
    reps_l, y_conc_l, w_conc_l = [], [], []

    for batch in tqdm(loader, desc="Faithfulness & OIS", leave=False):
        x      = batch["signal"].to(device)
        y_conc = batch["concept_labels"]
        w_conc = batch["concept_weights"].to(device)
        z      = model.encoder(x)

        p_base, _, _ = model(x)
        zeros = torch.zeros_like(w_conc)
        ones  = torch.ones_like(w_conc)
        p_zero = model.intervene(x, zeros, w_conc)
        p_one  = model.intervene(x, ones,  w_conc)

        base_probs_l.append(p_base.squeeze(-1).cpu().numpy())
        zero_probs_l.append(p_zero.squeeze(-1).cpu().numpy())
        one_probs_l.append(p_one.squeeze(-1).cpu().numpy())

        if hasattr(model, "get_concept_representations"):
            reps_l.append(model.get_concept_representations(z).cpu().numpy())
            y_conc_l.append(y_conc.numpy())
            w_conc_l.append(w_conc.cpu().numpy())

    p_base = np.concatenate(base_probs_l)
    p_zero = np.concatenate(zero_probs_l)
    p_one  = np.concatenate(one_probs_l)

    pred_pos = p_base > 0.5
    pred_neg = ~pred_pos
    flip_to_inc = float((p_zero[pred_pos] < 0.5).mean()) if np.any(pred_pos) else float("nan")
    flip_to_cmp = float((p_one[pred_neg]  > 0.5).mean()) if np.any(pred_neg) else float("nan")
    shifts = []
    if np.any(pred_pos):
        shifts.append(float((p_base[pred_pos] - p_zero[pred_pos]).mean()))
    if np.any(pred_neg):
        shifts.append(float((p_one[pred_neg] - p_base[pred_neg]).mean()))

    metrics: Dict[str, float] = {
        "faithfulness/flip_rate_to_incomplete":  flip_to_inc,
        "faithfulness/flip_rate_to_complete":    flip_to_cmp,
        "faithfulness/counterfactual_flip_rate": float(np.nanmean([flip_to_inc, flip_to_cmp])),
        "faithfulness/mean_prob_shift":          float(np.mean(shifts)) if shifts else 0.0,
    }

    # Oracle Impurity Score (OIS)
    if reps_l:
        reps = np.concatenate(reps_l, axis=0)      # (N, K, m)
        yc   = np.concatenate(y_conc_l, axis=0)    # (N, K)
        wc   = np.concatenate(w_conc_l, axis=0)    # (N, K)
        N, K, _ = reps.shape

        rng = np.random.default_rng(seed)
        if N > max_ois_samples:
            sel = rng.choice(N, size=max_ois_samples, replace=False)
            reps, yc, wc = reps[sel], yc[sel], wc[sel]
            N = max_ois_samples

        perm = rng.permutation(N)
        split = N // 2
        tr, te = perm[:split], perm[split:]

        excess_scores = []
        for i in range(K):
            Xi_tr, Xi_te = reps[tr, i, :], reps[te, i, :]
            Ci_tr, Ci_te = yc[tr, i : i + 1], yc[te, i : i + 1]
            for j in range(K):
                if i == j:
                    continue
                valid_tr = (wc[tr, i] > 0) & (wc[tr, j] > 0)
                valid_te = (wc[te, i] > 0) & (wc[te, j] > 0)
                if valid_tr.sum() < 20 or valid_te.sum() < 20:
                    continue
                yj_tr = yc[tr[valid_tr], j].astype(int)
                yj_te = yc[te[valid_te], j].astype(int)
                if len(np.unique(yj_tr)) < 2 or len(np.unique(yj_te)) < 2:
                    continue
                try:
                    clf_rep = LogisticRegression(max_iter=200, C=1.0)
                    clf_rep.fit(Xi_tr[valid_tr], yj_tr)
                    p_rep = clf_rep.predict_proba(Xi_te[valid_te])[:, 1]
                    auc_rep = roc_auc_score(yj_te, p_rep)

                    clf_orc = LogisticRegression(max_iter=200, C=1.0)
                    clf_orc.fit(Ci_tr[valid_tr], yj_tr)
                    p_orc = clf_orc.predict_proba(Ci_te[valid_te])[:, 1]
                    auc_orc = roc_auc_score(yj_te, p_orc)

                    excess_scores.append(max(0.0, float(auc_rep - auc_orc)))
                except Exception:
                    continue

        if excess_scores:
            metrics["leakage/ois"] = float(np.mean(excess_scores))

    return metrics


def format_metrics(metrics: Dict[str, float]) -> str:
    return "\n".join(f"  {k}: {v:.4f}" for k, v in sorted(metrics.items()))


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description="Evaluate trained ECG CBM / CEM checkpoint")
    parser.add_argument("--run-dir",       required=True,         help="Training run directory")
    parser.add_argument("--checkpoint",    default="best_model.pt", help="Checkpoint filename inside <run-dir>/checkpoints")
    parser.add_argument("--split",         default="test",        choices=["train", "val", "test"])
    parser.add_argument("--interventions", action="store_true",   help="Also compute concept intervention curve")
    parser.add_argument("--uncertainty",   action="store_true",   help="Also compute MC uncertainty quality metrics")
    parser.add_argument("--faithfulness",  action="store_true",   help="Also compute counterfactual flip rate and OIS leakage")
    args = parser.parse_args()

    run_dir   = Path(args.run_dir)
    ckpt_path = run_dir / "checkpoints" / args.checkpoint
    cfg_path  = run_dir / "config.yaml"
    if not cfg_path.exists():
        cfg_path = Path(__file__).resolve().parent.parent / "config.yaml"
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Device: {device} | Checkpoint: {ckpt_path} | Split: {args.split}")

    model = build_model(cfg).to(device)
    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(state["model"])
    model.eval()

    ds = load_or_build_datasets(cfg, splits=(args.split,))[args.split]
    data_cfg = cfg.get("data", {})
    loader = DataLoader(
        ds,
        batch_size=int(data_cfg.get("batch_size", 256)),
        shuffle=False,
        num_workers=int(data_cfg.get("num_workers", 4)),
        pin_memory=torch.cuda.is_available(),
    )

    concept_names = cfg.get("concept_names", CONCEPT_NAMES)
    metrics = evaluate(model, loader, device=device, concept_names=concept_names)

    if args.uncertainty:
        unc_metrics = evaluate_uncertainty_quality(
            model, loader, device=device,
            reliable_concepts=cfg.get("training", {}).get("reliable_concepts", [2, 4, 7, 9]),
        )
        metrics.update(unc_metrics)

    if args.faithfulness or args.interventions:
        faith_metrics = evaluate_faithfulness_and_leakage(model, loader, device=device)
        metrics.update(faith_metrics)

    logger.info(f"Results on [{args.split}] ({len(ds)} samples):\n" + format_metrics(metrics))

    out_path = run_dir / f"eval_{args.split}_metrics.json"
    with open(out_path, "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"Saved metrics → {out_path}")

    if args.interventions and getattr(model, "model_type", "cbm") != "blackbox":
        int_res = evaluate_interventions(model, loader, device=device)
        int_path = run_dir / f"intervention_{args.split}.json"
        with open(int_path, "w") as f:
            json.dump(int_res, f, indent=2)
        logger.info(
            f"Intervention accuracy (0 → {model.num_concepts} concepts): "
            f"{int_res['uncertainty_acc'][0]:.4f} → {int_res['uncertainty_acc'][-1]:.4f} "
            f"(saved to {int_path})"
        )


if __name__ == "__main__":
    main()
