"""
visualize.py — Visualise training results, uncertainty, and interventions for CBM / CEM.

Produces (all saved to <run-dir>/figures/):
  1. training_curves.png            — train/val loss & metric curves from train.log
  2. metrics_bar_<split>.png        — task + masked per-concept F1/accuracy bar chart
  3. roc_curve_<split>.png          — task-level ROC curve
  4. concept_heatmap_<split>.png    — predicted concept probs vs GT on random samples
  5. concept_confusion_<split>.png  — masked per-concept confusion matrices
  6. uncertainty_scatter_<split>.png— epistemic & aleatoric task uncertainty vs confidence
  7. intervention_curve_<split>.png — test-time concept intervention curve (Uncertainty vs Random)
  8. signal_examples_<split>.png    — ECG traces with concept annotations
"""

from __future__ import annotations

import argparse
import json
import logging
import re
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import wfdb
import yaml
from matplotlib.colors import LinearSegmentedColormap
from sklearn.metrics import accuracy_score, auc, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score, roc_curve
from torch.utils.data import DataLoader
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

PALETTE = {
    "blue":   "#4C72B0",
    "orange": "#DD8452",
    "green":  "#55A868",
    "red":    "#C44E52",
    "purple": "#8172B3",
    "grey":   "#8C8C8C",
}

LEAD_NAMES = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]

plt.rcParams.update({
    "figure.dpi":        150,
    "font.size":         10,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.grid":         True,
    "grid.alpha":        0.3,
    "grid.linestyle":    "--",
})


# ─────────────────────────────────────────────────────────────────────────────
# 1. Training curves
# ─────────────────────────────────────────────────────────────────────────────

def _parse_train_log(log_path: Path) -> Dict[str, List]:
    """Parse train.log into sequential step series (supports single- and multi-stage logs)."""
    epoch_re = re.compile(r"Epoch\s+(\d+)")
    loss_re  = re.compile(r"(?<!Val)Loss=([\d.]+)")
    vloss_re = re.compile(r"ValLoss=([\d.]+)")
    acc_re   = re.compile(r"ValAcc=([\d.]+)")
    auroc_re = re.compile(r"ValAUROC=([\d.]+)")
    cf1_re   = re.compile(r"ConceptF1=([\d.]+)")
    lam_re   = re.compile(r"λ_c=([\d.]+)")

    steps, losses, val_losses, accs, aurocs, cf1s, lams = [], [], [], [], [], [], []
    step_idx = 0
    with open(log_path) as f:
        for line in f:
            if not epoch_re.search(line):
                continue
            lm = loss_re.search(line)
            am = acc_re.search(line)
            rm = auroc_re.search(line)
            if not (lm and am and rm):
                continue
            step_idx += 1
            vlm = vloss_re.search(line)
            cm  = cf1_re.search(line)
            km  = lam_re.search(line)
            steps.append(step_idx)
            losses.append(float(lm.group(1)))
            val_losses.append(float(vlm.group(1)) if vlm else None)
            accs.append(float(am.group(1)))
            aurocs.append(float(rm.group(1)))
            cf1s.append(float(cm.group(1)) if cm else None)
            lams.append(float(km.group(1)) if km else None)
    return {
        "step": steps, "loss": losses, "val_loss": val_losses,
        "val_acc": accs, "val_auroc": aurocs, "concept_f1": cf1s, "lambda_c": lams,
    }


def plot_training_curves(run_dir: Path, out_dir: Path) -> None:
    log_path = run_dir / "train.log"
    if not log_path.exists():
        logger.warning("train.log not found — skipping training curves")
        return

    data = _parse_train_log(log_path)
    if not data["step"]:
        logger.warning("No epoch entries found in train.log")
        return

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    fig.suptitle("Training Curves", fontsize=13, fontweight="bold")

    ax = axes[0]
    ax.plot(data["step"], data["loss"], color=PALETTE["blue"], linewidth=1.8, label="Train Loss")
    if any(v is not None for v in data["val_loss"]):
        vl_x = [s for s, v in zip(data["step"], data["val_loss"]) if v is not None]
        vl_y = [v for v in data["val_loss"] if v is not None]
        ax.plot(vl_x, vl_y, color=PALETTE["red"], linewidth=1.6, linestyle="--", label="Val Loss")
        ax.legend(fontsize=9)
    ax.set_xlabel("Epoch"); ax.set_ylabel("Loss"); ax.set_title("Loss")

    ax = axes[1]
    ax.plot(data["step"], data["val_acc"],   color=PALETTE["green"],  linewidth=1.8, label="Val Accuracy")
    ax.plot(data["step"], data["val_auroc"], color=PALETTE["orange"], linewidth=1.8, label="Val AUROC", linestyle="--")
    if any(v is not None and v > 0 for v in data["concept_f1"]):
        cf_x = [s for s, v in zip(data["step"], data["concept_f1"]) if v is not None]
        cf_y = [v for v in data["concept_f1"] if v is not None]
        ax.plot(cf_x, cf_y, color=PALETTE["purple"], linewidth=1.5, linestyle=":", label="Concept Macro-F1")
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("Epoch"); ax.set_ylabel("Score"); ax.set_title("Validation Metrics")
    ax.legend(fontsize=8)

    ax = axes[2]
    lams = [v for v in data["lambda_c"] if v is not None]
    if lams:
        ax.plot(data["step"][: len(lams)], lams, color=PALETTE["purple"], linewidth=1.8)
        ax.set_xlabel("Epoch"); ax.set_ylabel("λ_concept"); ax.set_title("Concept Loss Weight")

    fig.tight_layout()
    out = out_dir / "training_curves.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved {out}")


# ─────────────────────────────────────────────────────────────────────────────
# 2. Metrics bar chart
# ─────────────────────────────────────────────────────────────────────────────

def plot_metrics_bar(metrics: dict, concept_names: List[str], out_dir: Path, split: str) -> None:
    concept_f1s = [metrics.get(f"concept/{c}/f1", 0.0) for c in concept_names]
    concept_acc = [metrics.get(f"concept/{c}/accuracy", 0.0) for c in concept_names]

    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    fig.suptitle(f"Model Metrics — {split} split", fontsize=13, fontweight="bold")

    ax = axes[0]
    task_keys   = ["accuracy", "f1", "precision", "recall", "auroc"]
    task_values = [metrics.get(f"task/{k}", 0.0) for k in task_keys]
    bars = ax.bar(task_keys, task_values, color=PALETTE["blue"], alpha=0.85, width=0.5)
    for bar, v in zip(bars, task_values):
        if not np.isnan(v):
            ax.text(bar.get_x() + bar.get_width() / 2, v + 0.01, f"{v:.3f}",
                    ha="center", va="bottom", fontsize=9)
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("Score")
    ax.set_title("Task Metrics")

    ax = axes[1]
    x = np.arange(len(concept_names))
    b1 = ax.bar(x - 0.2, concept_f1s, 0.35, label="F1",       color=PALETTE["orange"], alpha=0.85)
    b2 = ax.bar(x + 0.2, concept_acc, 0.35, label="Accuracy", color=PALETTE["green"],  alpha=0.85)
    for idx_c, c in enumerate(concept_names):
        if f"concept/{c}/f1" not in metrics:
            ax.text(x[idx_c], 0.04, "N/A", ha="center", va="bottom", fontsize=7.5, color=PALETTE["grey"])
            continue
        for bar in (b1[idx_c], b2[idx_c]):
            v = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, v + 0.01, f"{v:.2f}",
                    ha="center", va="bottom", fontsize=7, rotation=45)
    ax.set_xticks(x); ax.set_xticklabels(concept_names, fontsize=9)
    ax.set_ylim(0, 1.25)
    ax.set_ylabel("Score"); ax.set_title("Masked Per-Concept Metrics")
    ax.legend(fontsize=9)

    fig.tight_layout()
    out = out_dir / f"metrics_bar_{split}.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved {out}")


# ─────────────────────────────────────────────────────────────────────────────
# 3. ROC curve
# ─────────────────────────────────────────────────────────────────────────────

def plot_roc(task_probs: np.ndarray, task_labels: np.ndarray, out_dir: Path, split: str) -> None:
    if len(np.unique(task_labels)) < 2:
        return
    fpr, tpr, _ = roc_curve(task_labels, task_probs)
    roc_auc = auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(fpr, tpr, color=PALETTE["blue"], linewidth=2, label=f"AUC = {roc_auc:.4f}")
    ax.plot([0, 1], [0, 1], color=PALETTE["grey"], linestyle="--", linewidth=1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(f"Task ROC Curve — {split}")
    ax.legend(loc="lower right", fontsize=10)
    ax.set_xlim(-0.02, 1.02); ax.set_ylim(-0.02, 1.05)

    fig.tight_layout()
    out = out_dir / f"roc_curve_{split}.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved {out}")


# ─────────────────────────────────────────────────────────────────────────────
# 4. Concept heatmap
# ─────────────────────────────────────────────────────────────────────────────

def plot_concept_heatmap(
    c_probs: np.ndarray,
    c_labels: np.ndarray,
    concept_names: List[str],
    out_dir: Path,
    split: str,
    n_samples: int = 50,
) -> None:
    rng = np.random.default_rng(0)
    idx = rng.choice(len(c_probs), size=min(n_samples, len(c_probs)), replace=False)
    idx = np.sort(idx)

    probs_sub  = c_probs[idx]
    labels_sub = c_labels[idx]

    order = np.argsort(labels_sub.sum(axis=1))[::-1]
    probs_sub  = probs_sub[order]
    labels_sub = labels_sub[order]

    cmap_pred = LinearSegmentedColormap.from_list("pred", ["#f7f7f7", PALETTE["blue"]])
    cmap_gt   = LinearSegmentedColormap.from_list("gt",   ["#f7f7f7", PALETTE["orange"]])

    fig, axes = plt.subplots(1, 2, figsize=(13, max(4, n_samples * 0.18)), sharey=True)
    fig.suptitle(f"Concept Heatmap — {split} ({len(idx)} samples)", fontsize=13, fontweight="bold")

    for ax, data, cmap, title in zip(
        axes,
        [probs_sub, labels_sub],
        [cmap_pred, cmap_gt],
        ["Predicted Probabilities", "Ground-Truth Labels"],
    ):
        im = ax.imshow(data, aspect="auto", vmin=0, vmax=1, cmap=cmap, interpolation="nearest")
        ax.set_xticks(range(len(concept_names)))
        ax.set_xticklabels(concept_names, fontsize=9)
        ax.set_ylabel("Sample index")
        ax.set_title(title)
        plt.colorbar(im, ax=ax, fraction=0.03, pad=0.03)

    fig.tight_layout()
    out = out_dir / f"concept_heatmap_{split}.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved {out}")


# ─────────────────────────────────────────────────────────────────────────────
# 5. Masked per-concept confusion matrices
# ─────────────────────────────────────────────────────────────────────────────

def plot_concept_confusion(
    c_probs: np.ndarray,
    c_labels: np.ndarray,
    concept_names: List[str],
    out_dir: Path,
    split: str,
    c_weights: Optional[np.ndarray] = None,
) -> None:
    K = c_probs.shape[1]
    ncols = 5
    nrows = (K + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 3, nrows * 3))
    fig.suptitle(f"Per-Concept Confusion Matrices (Valid Delineations) — {split}", fontsize=13, fontweight="bold")

    axes_flat = axes.flatten() if hasattr(axes, "flatten") else [axes]
    cmap = LinearSegmentedColormap.from_list("cm", ["#ffffff", PALETTE["blue"]])

    for k in range(K):
        ax = axes_flat[k]
        valid = (c_weights[:, k] > 0) if c_weights is not None else np.ones(len(c_probs), dtype=bool)
        title_k = concept_names[k] if k < len(concept_names) else f"C{k}"
        if not np.any(valid):
            ax.imshow(np.zeros((2, 2)), cmap="Greys", vmin=0, vmax=1)
            ax.text(0.5, 0.5, "N/A\n(masked)", ha="center", va="center", fontsize=9, color=PALETTE["grey"])
            ax.set_xticks([]); ax.set_yticks([])
            ax.set_title(title_k, fontsize=10)
            ax.grid(False)
            continue
        preds  = (c_probs[valid, k] > 0.5).astype(int)
        labels = c_labels[valid, k].astype(int)
        cm = confusion_matrix(labels, preds, labels=[0, 1])
        cm_norm = cm.astype(float) / (cm.sum(axis=1, keepdims=True) + 1e-8)
        ax.imshow(cm_norm, cmap=cmap, vmin=0, vmax=1)
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f"{cm[i,j]}\n({cm_norm[i,j]:.1%})",
                        ha="center", va="center", fontsize=8,
                        color="white" if cm_norm[i, j] > 0.5 else "black")
        ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
        ax.set_xticklabels(["Pred 0", "Pred 1"], fontsize=8)
        ax.set_yticklabels(["GT 0",   "GT 1"],   fontsize=8)
        ax.set_title(title_k, fontsize=10)
        ax.grid(False)

    for k in range(K, len(axes_flat)):
        axes_flat[k].set_visible(False)

    fig.tight_layout()
    out = out_dir / f"concept_confusion_{split}.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved {out}")


# ─────────────────────────────────────────────────────────────────────────────
# 6. Uncertainty scatter
# ─────────────────────────────────────────────────────────────────────────────

def plot_uncertainty_scatter(
    task_probs_mean: np.ndarray,
    task_epistemic: np.ndarray,
    task_aleatoric: np.ndarray,
    task_labels: np.ndarray,
    out_dir: Path,
    split: str,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    fig.suptitle(f"Task Uncertainty — {split}", fontsize=13, fontweight="bold")

    for ax, unc, title in zip(
        axes,
        [task_epistemic, task_aleatoric],
        ["Epistemic Uncertainty", "Aleatoric Uncertainty"],
    ):
        for label, color, name in [(0, PALETTE["red"], "Incomplete"), (1, PALETTE["blue"], "Complete")]:
            mask = task_labels == label
            ax.scatter(task_probs_mean[mask], unc[mask], s=12, alpha=0.35,
                       color=color, label=name, rasterized=True)
        ax.set_xlabel("Mean Task Probability")
        ax.set_ylabel("Uncertainty (std)")
        ax.set_title(title)
        ax.legend(fontsize=9, markerscale=2)

    fig.tight_layout()
    out = out_dir / f"uncertainty_scatter_{split}.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved {out}")


# ─────────────────────────────────────────────────────────────────────────────
# 7. Test-time Concept Intervention curve
# ─────────────────────────────────────────────────────────────────────────────

def plot_intervention_curve(int_data: dict, out_dir: Path, split: str) -> None:
    ks = int_data["num_intervened"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    fig.suptitle(f"Test-Time Concept Interventions — {split}", fontsize=13, fontweight="bold")

    ax = axes[0]
    ax.plot(ks, int_data["uncertainty_acc"], marker="o", linewidth=2, color=PALETTE["blue"], label="Uncertainty-guided")
    ax.plot(ks, int_data["random_acc"],      marker="s", linewidth=1.8, linestyle="--", color=PALETTE["orange"], label="Random order")
    ax.set_xlabel("Number of Intervened Concepts")
    ax.set_ylabel("Task Accuracy")
    ax.set_title("Accuracy vs. Interventions")
    ax.set_xticks(ks)
    ax.legend(fontsize=9)

    ax = axes[1]
    ax.plot(ks, int_data["uncertainty_f1"], marker="o", linewidth=2, color=PALETTE["green"], label="Uncertainty-guided")
    ax.plot(ks, int_data["random_f1"],      marker="s", linewidth=1.8, linestyle="--", color=PALETTE["red"], label="Random order")
    ax.set_xlabel("Number of Intervened Concepts")
    ax.set_ylabel("Task F1")
    ax.set_title("F1 Score vs. Interventions")
    ax.set_xticks(ks)
    ax.legend(fontsize=9)

    fig.tight_layout()
    out = out_dir / f"intervention_curve_{split}.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved {out}")


# ─────────────────────────────────────────────────────────────────────────────
# 8. ECG signal examples
# ─────────────────────────────────────────────────────────────────────────────

def _choose_diverse_indices(task_probs: np.ndarray, n_samples: int, seed: int = 42) -> List[int]:
    rng = np.random.default_rng(seed)
    confident = np.where(np.abs(task_probs - 0.5) > 0.35)[0]
    uncertain = np.where(np.abs(task_probs - 0.5) <= 0.15)[0]
    n_each = n_samples // 2
    chosen: List[int] = []
    if len(confident) >= n_each:
        chosen.extend(rng.choice(confident, n_each, replace=False).tolist())
    else:
        chosen.extend(confident.tolist())
    if len(uncertain) >= n_samples - len(chosen):
        chosen.extend(rng.choice(uncertain, n_samples - len(chosen), replace=False).tolist())
    else:
        chosen.extend(uncertain.tolist())
    remaining = n_samples - len(chosen)
    if remaining > 0:
        pool = list(set(range(len(task_probs))) - set(chosen))
        if pool:
            chosen.extend(rng.choice(pool, min(remaining, len(pool)), replace=False).tolist())
    return chosen[:n_samples]


def plot_signal_examples(
    signals: np.ndarray,
    task_probs: np.ndarray,
    task_labels: np.ndarray,
    c_probs: np.ndarray,
    c_labels: np.ndarray,
    concept_names: List[str],
    out_dir: Path,
    split: str,
    n_samples: int = 8,
    lead_idx: int = 1,
    sampling_rate: int = 500,
) -> None:
    chosen = _choose_diverse_indices(task_probs, n_samples)
    if not chosen:
        return

    T = signals.shape[-1]
    t_axis = np.arange(T) / sampling_rate * 1000

    fig = plt.figure(figsize=(16, len(chosen) * 1.8))
    fig.suptitle(f"ECG Signal Examples — {split} (lead {lead_idx})", fontsize=13, fontweight="bold")
    gs = gridspec.GridSpec(len(chosen), 2, figure=fig, width_ratios=[2, 1], hspace=0.6, wspace=0.3)

    for row, i in enumerate(chosen):
        sig = signals[i, lead_idx] if signals.shape[1] > lead_idx else signals[i, 0]

        ax_sig = fig.add_subplot(gs[row, 0])
        correct = int(task_probs[i] > 0.5) == int(task_labels[i])
        color = PALETTE["blue"] if int(task_labels[i]) == 1 else PALETTE["red"]
        ax_sig.plot(t_axis, sig, color=color, linewidth=0.9, alpha=0.9)
        ax_sig.set_xlim(t_axis[0], t_axis[-1])
        ax_sig.set_ylabel("z-score", fontsize=7)
        title = (
            f"GT={'Complete' if task_labels[i] else 'Incomplete'}  "
            f"Pred={task_probs[i]:.2f}  {'✓' if correct else '✗'}"
        )
        ax_sig.set_title(title, fontsize=8)
        if row == len(chosen) - 1:
            ax_sig.set_xlabel("Time (ms)", fontsize=8)
        else:
            ax_sig.set_xticklabels([])
        ax_sig.tick_params(labelsize=7)
        ax_sig.grid(True, alpha=0.2)

        ax_bar = fig.add_subplot(gs[row, 1])
        K = len(concept_names)
        ypos = np.arange(K)
        ax_bar.barh(ypos, c_probs[i, :K], height=0.45, color=PALETTE["blue"], alpha=0.7, label="Pred")
        ax_bar.scatter(c_labels[i, :K], ypos, color=PALETTE["orange"], s=25, zorder=5, label="GT")
        ax_bar.axvline(0.5, color=PALETTE["grey"], linewidth=0.8, linestyle="--")
        ax_bar.set_yticks(ypos); ax_bar.set_yticklabels(concept_names, fontsize=7)
        ax_bar.set_xlim(-0.05, 1.1)
        ax_bar.set_xlabel("Prob", fontsize=7)
        ax_bar.tick_params(labelsize=7)
        ax_bar.grid(True, alpha=0.2, axis="x")
        if row == 0:
            ax_bar.legend(fontsize=7, loc="lower right")

    out = out_dir / f"signal_examples_{split}.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved {out}")


def plot_signal_examples_full_lead(
    signals: np.ndarray,
    task_probs: np.ndarray,
    task_labels: np.ndarray,
    c_probs: np.ndarray,
    c_labels: np.ndarray,
    concept_names: List[str],
    out_dir: Path,
    split: str,
    n_samples: int = 4,
    sampling_rate: int = 500,
) -> None:
    chosen = _choose_diverse_indices(task_probs, n_samples)
    n_leads = signals.shape[1]
    lead_names = LEAD_NAMES[:n_leads] if n_leads <= len(LEAD_NAMES) else [f"L{l}" for l in range(n_leads)]
    T = signals.shape[-1]
    t_axis = np.arange(T) / sampling_rate * 1000

    for sample_num, i in enumerate(chosen):
        correct = int(task_probs[i] > 0.5) == int(task_labels[i])
        gt_str  = "Complete" if task_labels[i] else "Incomplete"
        title   = f"GT={gt_str}  Pred={task_probs[i]:.2f}  {'✓' if correct else '✗'}  [{split} sample {i}]"

        fig, axes = plt.subplots(n_leads, 1, figsize=(14, n_leads * 1.1), sharex=True)
        fig.suptitle(title, fontsize=11, fontweight="bold")

        for l_idx in range(n_leads):
            ax = axes[l_idx] if n_leads > 1 else axes
            color = PALETTE["blue"] if int(task_labels[i]) == 1 else PALETTE["red"]
            ax.plot(t_axis, signals[i, l_idx], color=color, linewidth=0.7, alpha=0.9)
            ax.set_ylabel(lead_names[l_idx], fontsize=7, rotation=0, labelpad=22)
            ax.tick_params(labelsize=6)
            ax.set_xlim(t_axis[0], t_axis[-1])
            ax.grid(True, alpha=0.2)

        (axes[-1] if n_leads > 1 else axes).set_xlabel("Time (ms)", fontsize=8)
        K = len(concept_names)
        pred_str = "  ".join(
            f"{concept_names[k]}={'✓' if c_probs[i, k] > 0.5 else '✗'}({c_probs[i, k]:.2f})"
            for k in range(min(K, c_probs.shape[1]))
        )
        fig.text(0.5, 0.005, pred_str, ha="center", fontsize=6.5, color=PALETTE["grey"])
        fig.tight_layout(rect=[0, 0.02, 1, 1])
        out = out_dir / f"signal_full_lead_{split}_sample{sample_num:02d}.png"
        fig.savefig(out, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"Saved {out}")


# ─────────────────────────────────────────────────────────────────────────────
# Sliding-window dataset builder & inference helpers
# ─────────────────────────────────────────────────────────────────────────────

def build_sliding_window_subset(cfg: dict, n_records: int = 30):
    """Build a sliding-window HeartbeatDataset from the first `n_records` test records."""
    from data import (
        HeartbeatDataset,
        _load_or_compute_delin,
        bandpass_filter,
        extract_sliding_windows,
        resolve_data_paths,
    )

    data_cfg = cfg.get("data", {})
    data_path, _ = resolve_data_paths(cfg)
    sampling_rate = int(data_cfg.get("sampling_rate", 500))
    t_fixed       = int(data_cfg.get("t_fixed", 500))
    stride        = int(data_cfg.get("stride", 50))
    test_fold     = int(data_cfg.get("test_fold", 10))
    lead_idx      = int(data_cfg.get("lead", 1))
    base_cache    = Path(cfg.get("training", {}).get("checkpoint_dir", "checkpoints")) / "dataset_cache"
    delin_cache   = base_cache / "delineation" if (lead_idx == 0 and (base_cache / "delineation").exists()) else base_cache / f"delineation_sr{sampling_rate}_lead{lead_idx}"

    filename_col = "filename_hr" if sampling_rate == 500 else "filename_lr"
    meta = pd.read_csv(data_path / "ptbxl_database.csv", index_col="ecg_id")
    test_records = meta[meta["strat_fold"] == test_fold].iloc[:n_records]

    samples: list = []
    for ecg_id, row in test_records.iterrows():
        try:
            raw, _ = wfdb.rdsamp(str(data_path / row[filename_col]))
            signal = bandpass_filter(raw.T.astype("float32"), sampling_rate)
            delin  = _load_or_compute_delin(ecg_id, signal, sampling_rate, lead_idx, delin_cache)
            for subseq, task_label, concepts, weights in extract_sliding_windows(signal, delin, t_fixed, stride):
                samples.append({
                    "signal":          subseq,
                    "task_label":      task_label,
                    "concept_labels":  concepts,
                    "concept_weights": weights,
                    "record_id":       int(ecg_id),
                })
        except Exception as exc:
            logger.warning(f"Skipped ecg_id={ecg_id} (sliding window): {exc}")

    logger.info(f"Built sliding-window subset: {len(samples)} windows from {n_records} records")
    return HeartbeatDataset(samples).apply_config_views(cfg)


def _forward_pass(
    model,
    ds,
    data_cfg: dict,
    use_uncertainty: bool,
    n_mc_epistemic: int,
    n_mc_aleatoric: int,
    desc: str = "Inference",
) -> dict:
    loader = DataLoader(
        ds,
        batch_size=int(data_cfg.get("batch_size", 256)),
        shuffle=False,
        num_workers=int(data_cfg.get("num_workers", 4)),
        pin_memory=torch.cuda.is_available(),
    )
    task_probs_all, task_labels_all = [], []
    c_probs_all, c_labels_all, c_weights_all = [], [], []
    signals_all = []
    task_epistemic_all, task_aleatoric_all = [], []

    device = next(model.parameters()).device
    model.eval()
    with torch.no_grad():
        for batch in tqdm(loader, desc=desc):
            x      = batch["signal"].to(device)
            y_task = batch["task_label"]
            y_conc = batch["concept_labels"]
            w_conc = batch["concept_weights"]

            if use_uncertainty:
                out = model.predict_with_uncertainty(x, n_mc_epistemic, n_mc_aleatoric)
                task_prob = out["task_prob_mean"]
                c_prob    = out["concept_probs_mean"]
                task_epistemic_all.append(out["task_epistemic"].cpu().numpy())
                task_aleatoric_all.append(out["task_aleatoric"].cpu().numpy())
            else:
                task_prob, c_prob, _ = model(x)
                task_prob = task_prob.squeeze(-1)

            task_probs_all.append(task_prob.cpu().numpy())
            task_labels_all.append(y_task.numpy())
            c_probs_all.append(c_prob.cpu().numpy())
            c_labels_all.append(y_conc.numpy())
            c_weights_all.append(w_conc.numpy())
            signals_all.append(x.cpu().numpy())

    return {
        "loader":         loader,
        "task_probs":     np.concatenate(task_probs_all),
        "task_labels":    np.concatenate(task_labels_all),
        "c_probs":        np.concatenate(c_probs_all),
        "c_labels":       np.concatenate(c_labels_all),
        "c_weights":      np.concatenate(c_weights_all),
        "signals":        np.concatenate(signals_all),
        "task_epistemic": np.concatenate(task_epistemic_all) if task_epistemic_all else None,
        "task_aleatoric": np.concatenate(task_aleatoric_all) if task_aleatoric_all else None,
    }


def main() -> None:
    from data import load_or_build_datasets
    from evaluate import evaluate_interventions
    from models import build_model
    from train import Checkpointer

    parser = argparse.ArgumentParser(description="Visualise heartbeat CBM / CEM results")
    parser.add_argument("--run-dir",          required=True,       help="Path to training run directory")
    parser.add_argument("--split",            default="test",      help="Dataset split to evaluate")
    parser.add_argument("--n-samples",        type=int, default=8, help="Number of ECG signal examples to plot")
    parser.add_argument("--no-uncertainty",   action="store_true", help="Skip MC-uncertainty forward pass")
    parser.add_argument("--no-interventions", action="store_true", help="Skip intervention curve plot")
    parser.add_argument("--n-mc-epistemic",   type=int, default=20, help="MC epistemic samples")
    parser.add_argument("--n-mc-aleatoric",   type=int, default=30, help="MC aleatoric samples")
    parser.add_argument("--lead",             type=int, default=1,  help="Lead to show in signal plot")
    parser.add_argument("--full-lead",        action="store_true", help="Also generate per-sample 12-lead signal plots")
    parser.add_argument("--sliding-window",   action="store_true", help="Also plot sliding-window subset examples")
    parser.add_argument("--sw-n-records",     type=int, default=30, help="Test records for sliding window")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    out_dir = run_dir / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg_path = run_dir / "config.yaml"
    if not cfg_path.exists():
        cfg_path = Path(__file__).resolve().parent.parent / "config.yaml"
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)
    concept_names: List[str] = cfg.get("concept_names", ["P", "Q", "R", "S", "T", "U", "PR", "QRS", "ST", "QT"])
    active = cfg.get("active_concepts", None)
    if active is not None:
        concept_names = [concept_names[i] for i in active]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Device: {device} | Output: {out_dir}")

    plot_training_curves(run_dir, out_dir)

    ckpt_path = run_dir / "checkpoints" / "best_model.pt"
    model = build_model(cfg).to(device)
    Checkpointer.load(ckpt_path, model, device=device)
    is_blackbox = getattr(model, "model_type", "cbm") == "blackbox"

    ds = load_or_build_datasets(cfg, splits=(args.split,))[args.split]
    use_uncertainty = not args.no_uncertainty
    preds = _forward_pass(
        model, ds, cfg.get("data", {}), use_uncertainty,
        args.n_mc_epistemic, args.n_mc_aleatoric, desc=f"Inference ({args.split})",
    )

    task_probs  = preds["task_probs"]
    task_labels = preds["task_labels"]
    c_probs     = preds["c_probs"]
    c_labels    = preds["c_labels"]
    c_weights   = preds["c_weights"]
    signals     = preds["signals"]

    task_preds = (task_probs > 0.5).astype(int)
    metrics: dict = {
        "task/accuracy":  accuracy_score(task_labels, task_preds),
        "task/f1":        f1_score(task_labels, task_preds, zero_division=0),
        "task/precision": precision_score(task_labels, task_preds, zero_division=0),
        "task/recall":    recall_score(task_labels, task_preds, zero_division=0),
        "task/auroc":     roc_auc_score(task_labels, task_probs) if len(np.unique(task_labels)) > 1 else float("nan"),
    }
    if not is_blackbox:
        K = min(c_probs.shape[1], c_labels.shape[1])
        for k in range(K):
            name = concept_names[k] if k < len(concept_names) else f"C{k}"
            valid = c_weights[:, k] > 0
            if not np.any(valid):
                continue
            preds_k = (c_probs[valid, k] > 0.5).astype(int)
            metrics[f"concept/{name}/f1"]       = f1_score(c_labels[valid, k].astype(int), preds_k, zero_division=0)
            metrics[f"concept/{name}/accuracy"] = accuracy_score(c_labels[valid, k].astype(int), preds_k)

    plot_metrics_bar(metrics, concept_names, out_dir, args.split)
    plot_roc(task_probs, task_labels, out_dir, args.split)

    if not is_blackbox:
        plot_concept_heatmap(c_probs, c_labels, concept_names, out_dir, args.split)
        plot_concept_confusion(c_probs, c_labels, concept_names, out_dir, args.split, c_weights=c_weights)

        if not args.no_interventions:
            int_path = run_dir / f"intervention_{args.split}.json"
            if int_path.exists():
                with open(int_path) as f:
                    int_data = json.load(f)
            else:
                int_data = evaluate_interventions(model, preds["loader"], device=device)
                with open(int_path, "w") as f:
                    json.dump(int_data, f, indent=2)
            plot_intervention_curve(int_data, out_dir, args.split)

    if use_uncertainty and preds["task_epistemic"] is not None:
        plot_uncertainty_scatter(
            task_probs, preds["task_epistemic"], preds["task_aleatoric"],
            task_labels, out_dir, args.split,
        )

    sr = int(cfg.get("data", {}).get("sampling_rate", 500))
    plot_signal_examples(
        signals, task_probs, task_labels, c_probs, c_labels,
        concept_names, out_dir, args.split,
        n_samples=args.n_samples, lead_idx=args.lead, sampling_rate=sr,
    )

    if args.full_lead:
        plot_signal_examples_full_lead(
            signals, task_probs, task_labels, c_probs, c_labels,
            concept_names, out_dir, args.split,
            n_samples=min(args.n_samples, 4), sampling_rate=sr,
        )

    if args.sliding_window:
        sw_ds = build_sliding_window_subset(cfg, n_records=args.sw_n_records)
        sw_preds = _forward_pass(
            model, sw_ds, cfg.get("data", {}), use_uncertainty,
            args.n_mc_epistemic, args.n_mc_aleatoric, desc="Inference (sliding window)",
        )
        plot_signal_examples(
            sw_preds["signals"], sw_preds["task_probs"], sw_preds["task_labels"],
            sw_preds["c_probs"], sw_preds["c_labels"],
            concept_names, out_dir, split=f"{args.split}_sw",
            n_samples=args.n_samples, lead_idx=args.lead, sampling_rate=sr,
        )
        if args.full_lead:
            plot_signal_examples_full_lead(
                sw_preds["signals"], sw_preds["task_probs"], sw_preds["task_labels"],
                sw_preds["c_probs"], sw_preds["c_labels"],
                concept_names, out_dir, split=f"{args.split}_sw",
                n_samples=min(args.n_samples, 4), sampling_rate=sr,
            )

    logger.info(f"All figures saved to: {out_dir}")


if __name__ == "__main__":
    main()
