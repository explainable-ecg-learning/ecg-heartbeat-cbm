"""
visualize.py — Visualise training results and model behaviour for the heartbeat CBM.

Usage:
    python visualize.py --run-dir runs/20260422_120000
    python visualize.py --run-dir runs/20260422_120000 --split test --n-samples 8
    python visualize.py --run-dir runs/20260422_120000 --no-uncertainty

Produces (all saved to <run-dir>/figures/):
  1. training_curves.png       — loss curves from train.log
  2. metrics_bar.png           — task + per-concept F1 bar chart
  3. concept_heatmap.png       — predicted concept probs on random test samples
  4. roc_curve.png             — task-level ROC curve
  5. concept_confusion.png     — per-concept confusion matrix (predicted vs GT)
  6. uncertainty_scatter.png   — task uncertainty vs task confidence scatter
  7. signal_examples.png       — ECG traces with concept annotations (n-samples)
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
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import torch
import yaml
from matplotlib.colors import LinearSegmentedColormap
from sklearn.metrics import roc_curve, auc, confusion_matrix
from torch.utils.data import DataLoader
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Style
# ─────────────────────────────────────────────────────────────────────────────

PALETTE = {
    "blue":   "#4C72B0",
    "orange": "#DD8452",
    "green":  "#55A868",
    "red":    "#C44E52",
    "purple": "#8172B3",
    "grey":   "#8C8C8C",
}
CONCEPT_COLORS = list(PALETTE.values())

plt.rcParams.update({
    "figure.dpi":       150,
    "font.size":        10,
    "axes.spines.top":  False,
    "axes.spines.right":False,
    "axes.grid":        True,
    "grid.alpha":       0.3,
    "grid.linestyle":   "--",
})


# ─────────────────────────────────────────────────────────────────────────────
# 1. Training curves
# ─────────────────────────────────────────────────────────────────────────────

def _parse_train_log(log_path: Path) -> Dict[str, List]:
    """Parse train.log into epoch-level series."""
    epoch_re  = re.compile(r"Epoch\s+(\d+)")
    loss_re   = re.compile(r"Loss=([\d.]+)")
    acc_re    = re.compile(r"ValAcc=([\d.]+)")
    auroc_re  = re.compile(r"ValAUROC=([\d.]+)")
    lam_re    = re.compile(r"λ_c=([\d.]+)")

    epochs, losses, accs, aurocs, lams = [], [], [], [], []
    with open(log_path) as f:
        for line in f:
            em = epoch_re.search(line)
            if not em:
                continue
            lm = loss_re.search(line)
            am = acc_re.search(line)
            rm = auroc_re.search(line)
            km = lam_re.search(line)
            if not (lm and am and rm):
                continue
            epochs.append(int(em.group(1)))
            losses.append(float(lm.group(1)))
            accs.append(float(am.group(1)))
            aurocs.append(float(rm.group(1)))
            lams.append(float(km.group(1)) if km else None)
    return {"epoch": epochs, "loss": losses, "val_acc": accs, "val_auroc": aurocs, "lambda_c": lams}


def plot_training_curves(run_dir: Path, out_dir: Path) -> None:
    log_path = run_dir / "train.log"
    if not log_path.exists():
        logger.warning("train.log not found — skipping training curves")
        return

    data = _parse_train_log(log_path)
    if not data["epoch"]:
        logger.warning("No epoch entries found in train.log")
        return

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    fig.suptitle("Training Curves", fontsize=13, fontweight="bold")

    ax = axes[0]
    ax.plot(data["epoch"], data["loss"], color=PALETTE["blue"], linewidth=1.8)
    ax.set_xlabel("Epoch"); ax.set_ylabel("Train Loss"); ax.set_title("Train Loss")

    ax = axes[1]
    ax.plot(data["epoch"], data["val_acc"],   color=PALETTE["green"],  linewidth=1.8, label="Val Accuracy")
    ax.plot(data["epoch"], data["val_auroc"], color=PALETTE["orange"], linewidth=1.8, label="Val AUROC", linestyle="--")
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("Epoch"); ax.set_ylabel("Score"); ax.set_title("Validation Metrics")
    ax.legend(fontsize=9)

    ax = axes[2]
    lams = [v for v in data["lambda_c"] if v is not None]
    if lams:
        ax.plot(data["epoch"][: len(lams)], lams, color=PALETTE["purple"], linewidth=1.8)
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

    # Task metrics
    ax = axes[0]
    task_keys   = ["accuracy", "f1", "precision", "recall", "auroc"]
    task_values = [metrics.get(f"task/{k}", 0.0) for k in task_keys]
    bars = ax.bar(task_keys, task_values, color=PALETTE["blue"], alpha=0.85, width=0.5)
    for bar, v in zip(bars, task_values):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.01, f"{v:.3f}",
                ha="center", va="bottom", fontsize=9)
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("Score")
    ax.set_title("Task Metrics")

    # Per-concept F1
    ax = axes[1]
    x = np.arange(len(concept_names))
    b1 = ax.bar(x - 0.2, concept_f1s, 0.35, label="F1",       color=PALETTE["orange"], alpha=0.85)
    b2 = ax.bar(x + 0.2, concept_acc, 0.35, label="Accuracy",  color=PALETTE["green"],  alpha=0.85)
    for bars in (b1, b2):
        for bar in bars:
            v = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, v + 0.01, f"{v:.2f}",
                    ha="center", va="bottom", fontsize=7, rotation=45)
    ax.set_xticks(x); ax.set_xticklabels(concept_names, fontsize=9)
    ax.set_ylim(0, 1.25)
    ax.set_ylabel("Score"); ax.set_title("Per-Concept Metrics")
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

    # Sort rows by task completeness (sum of concept labels)
    order = np.argsort(labels_sub.sum(axis=1))[::-1]
    probs_sub  = probs_sub[order]
    labels_sub = labels_sub[order]

    cmap_pred = LinearSegmentedColormap.from_list("pred", ["#f7f7f7", PALETTE["blue"]])
    cmap_gt   = LinearSegmentedColormap.from_list("gt",   ["#f7f7f7", PALETTE["orange"]])

    fig, axes = plt.subplots(1, 2, figsize=(13, max(4, n_samples * 0.18)), sharey=True)
    fig.suptitle(f"Concept Heatmap — {split} ({n_samples} samples)", fontsize=13, fontweight="bold")

    for ax, data, cmap, title in zip(
        axes,
        [probs_sub, labels_sub],
        [cmap_pred, cmap_gt],
        ["Predicted Probabilities", "Ground-Truth Labels"],
    ):
        im = ax.imshow(data, aspect="auto", vmin=0, vmax=1, cmap=cmap,
                       interpolation="nearest")
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
# 5. Per-concept confusion matrices
# ─────────────────────────────────────────────────────────────────────────────

def plot_concept_confusion(
    c_probs: np.ndarray,
    c_labels: np.ndarray,
    concept_names: List[str],
    out_dir: Path,
    split: str,
) -> None:
    K = c_probs.shape[1]
    ncols = 5
    nrows = (K + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 3, nrows * 3))
    fig.suptitle(f"Per-Concept Confusion Matrices — {split}", fontsize=13, fontweight="bold")

    axes_flat = axes.flatten() if hasattr(axes, "flatten") else [axes]
    cmap = LinearSegmentedColormap.from_list("cm", ["#ffffff", PALETTE["blue"]])

    for k in range(K):
        ax = axes_flat[k]
        preds  = (c_probs[:, k] > 0.5).astype(int)
        labels = c_labels[:, k].astype(int)
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
        ax.set_title(concept_names[k] if k < len(concept_names) else f"C{k}", fontsize=10)
        ax.grid(False)

    for k in range(K, len(axes_flat)):
        axes_flat[k].set_visible(False)

    fig.tight_layout()
    out = out_dir / f"concept_confusion_{split}.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved {out}")


# ─────────────────────────────────────────────────────────────────────────────
# 6. Uncertainty scatter (requires MC-uncertainty pass)
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
# 7. ECG signal examples with concept annotations
# ─────────────────────────────────────────────────────────────────────────────

def plot_signal_examples(
    signals: np.ndarray,              # (N, C, T)
    task_probs: np.ndarray,           # (N,)
    task_labels: np.ndarray,          # (N,)
    c_probs: np.ndarray,              # (N, K)
    c_labels: np.ndarray,             # (N, K)
    concept_names: List[str],
    out_dir: Path,
    split: str,
    n_samples: int = 8,
    lead_idx: int = 1,
    sampling_rate: int = 500,
) -> None:
    rng = np.random.default_rng(42)
    # Pick n/2 correct and n/2 uncertain predictions for variety
    confident   = np.where(np.abs(task_probs - 0.5) > 0.35)[0]
    uncertain   = np.where(np.abs(task_probs - 0.5) <= 0.15)[0]
    n_each = n_samples // 2
    chosen = []
    if len(confident) >= n_each:
        chosen.extend(rng.choice(confident, n_each, replace=False).tolist())
    else:
        chosen.extend(confident.tolist())
    if len(uncertain) >= n_samples - len(chosen):
        chosen.extend(rng.choice(uncertain, n_samples - len(chosen), replace=False).tolist())
    else:
        chosen.extend(uncertain.tolist())
    # Pad with random if needed
    remaining = n_samples - len(chosen)
    if remaining > 0:
        pool = list(set(range(len(signals))) - set(chosen))
        chosen.extend(rng.choice(pool, min(remaining, len(pool)), replace=False).tolist())

    chosen = chosen[:n_samples]

    T = signals.shape[-1]
    t_axis = np.arange(T) / sampling_rate * 1000   # ms

    fig = plt.figure(figsize=(16, n_samples * 1.8))
    fig.suptitle(f"ECG Signal Examples — {split} (lead {lead_idx})", fontsize=13, fontweight="bold")
    gs = gridspec.GridSpec(n_samples, 2, figure=fig, width_ratios=[2, 1], hspace=0.6, wspace=0.3)

    for row, i in enumerate(chosen):
        sig = signals[i, lead_idx] if signals.shape[1] > lead_idx else signals[i, 0]

        # Signal trace
        ax_sig = fig.add_subplot(gs[row, 0])
        correct = int(task_probs[i] > 0.5) == int(task_labels[i])
        color = PALETTE["blue"] if int(task_labels[i]) == 1 else PALETTE["red"]
        ax_sig.plot(t_axis, sig, color=color, linewidth=0.9, alpha=0.9)
        ax_sig.set_xlim(t_axis[0], t_axis[-1])
        ax_sig.set_ylabel("mV", fontsize=7)
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
        for sp in ("top", "right"):
            ax_sig.spines[sp].set_visible(False)

        # Concept bar
        ax_bar = fig.add_subplot(gs[row, 1])
        K = len(concept_names)
        ypos = np.arange(K)
        ax_bar.barh(ypos, c_probs[i, :K], height=0.45, color=PALETTE["blue"], alpha=0.7, label="Pred")
        gt_markers = c_labels[i, :K]
        ax_bar.scatter(gt_markers, ypos, color=PALETTE["orange"], s=25, zorder=5, label="GT")
        ax_bar.axvline(0.5, color=PALETTE["grey"], linewidth=0.8, linestyle="--")
        ax_bar.set_yticks(ypos); ax_bar.set_yticklabels(concept_names, fontsize=7)
        ax_bar.set_xlim(-0.05, 1.1)
        ax_bar.set_xlabel("Prob", fontsize=7)
        ax_bar.tick_params(labelsize=7)
        ax_bar.grid(True, alpha=0.2, axis="x")
        for sp in ("top", "right"):
            ax_bar.spines[sp].set_visible(False)
        if row == 0:
            ax_bar.legend(fontsize=7, loc="lower right")

    out = out_dir / f"signal_examples_{split}.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved {out}")


# ─────────────────────────────────────────────────────────────────────────────
# 7b. ECG signal examples — all 12 leads
# ─────────────────────────────────────────────────────────────────────────────

LEAD_NAMES = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]


def plot_signal_examples_full_lead(
    signals: np.ndarray,              # (N, C, T)
    task_probs: np.ndarray,           # (N,)
    task_labels: np.ndarray,          # (N,)
    c_probs: np.ndarray,              # (N, K)
    c_labels: np.ndarray,             # (N, K)
    concept_names: List[str],
    out_dir: Path,
    split: str,
    n_samples: int = 4,
    sampling_rate: int = 500,
) -> None:
    """Plot all 12 ECG leads for a small selection of examples."""
    rng = np.random.default_rng(42)
    confident = np.where(np.abs(task_probs - 0.5) > 0.35)[0]
    uncertain = np.where(np.abs(task_probs - 0.5) <= 0.15)[0]
    n_each = n_samples // 2
    chosen: list = []
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
        pool = list(set(range(len(signals))) - set(chosen))
        chosen.extend(rng.choice(pool, min(remaining, len(pool)), replace=False).tolist())
    chosen = chosen[:n_samples]

    n_leads = signals.shape[1]
    lead_names = LEAD_NAMES[:n_leads] if n_leads <= len(LEAD_NAMES) else [f"L{l}" for l in range(n_leads)]
    T = signals.shape[-1]
    t_axis = np.arange(T) / sampling_rate * 1000  # ms

    for sample_num, i in enumerate(chosen):
        correct = int(task_probs[i] > 0.5) == int(task_labels[i])
        gt_str  = "Complete" if task_labels[i] else "Incomplete"
        title   = (
            f"GT={gt_str}  Pred={task_probs[i]:.2f}  {'✓' if correct else '✗'}  "
            f"[{split}  sample {i}]"
        )

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
            for sp in ("top", "right"):
                ax.spines[sp].set_visible(False)

        axes[-1].set_xlabel("Time (ms)", fontsize=8)

        # Add concept annotation as text below title
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
# Sliding-window dataset builder (subset of test fold for visualisation)
# ─────────────────────────────────────────────────────────────────────────────

def build_sliding_window_subset(cfg: dict, n_records: int = 30) -> "HeartbeatDataset":
    """Build a HeartbeatDataset from the test fold using sliding windows.

    Only the first *n_records* test-fold records are processed to keep it fast.
    Delineation results are read from the existing cache when available.
    """
    import pickle
    import wfdb
    import pandas as pd
    from pathlib import Path as _Path
    from data import (
        HeartbeatDataset, PTBXL_ROOT,
        bandpass_filter, extract_sliding_windows,
        _load_or_compute_delin,
    )

    data_cfg       = cfg.get("data", {})
    data_path      = _Path(data_cfg.get("data_path", str(PTBXL_ROOT)))
    sampling_rate  = data_cfg.get("sampling_rate", 500)
    t_fixed        = data_cfg.get("t_fixed", 500)
    stride         = data_cfg.get("stride", 50)
    test_fold      = data_cfg.get("test_fold", 10)
    lead_idx       = data_cfg.get("lead", 1)
    delin_cache    = _Path(cfg.get("training", {}).get("checkpoint_dir", "checkpoints")) / "dataset_cache" / "delineation"

    filename_col = "filename_hr" if sampling_rate == 500 else "filename_lr"
    meta = pd.read_csv(data_path / "ptbxl_database.csv", index_col="ecg_id")
    test_records = meta[meta["strat_fold"] == test_fold].iloc[:n_records]

    samples: list = []
    for ecg_id, row in test_records.iterrows():
        try:
            raw, _ = wfdb.rdsamp(str(data_path / row[filename_col]))
            signal = bandpass_filter(raw.T.astype("float32"), sampling_rate)
            delin  = _load_or_compute_delin(ecg_id, signal, sampling_rate, lead_idx, delin_cache)
            windows = extract_sliding_windows(signal, delin, t_fixed, stride)
            for subseq, task_label, concepts, weights in windows:
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
    return HeartbeatDataset(samples)


# ─────────────────────────────────────────────────────────────────────────────
# Main inference + plotting pipeline
# ─────────────────────────────────────────────────────────────────────────────

def run_inference(
    run_dir: Path,
    cfg: dict,
    split: str,
    device: str,
    use_uncertainty: bool,
    n_mc_epistemic: int,
    n_mc_aleatoric: int,
) -> dict:
    """Run model on dataset split; return dict of raw predictions."""
    from data import HeartbeatDataset, make_dataloaders, PTBXL_ROOT
    from models import build_model, CONCEPT_NAMES
    from train import Checkpointer

    ckpt_path = run_dir / "checkpoints" / "best_model.pt"
    model = build_model(cfg).to(device)
    Checkpointer.load(ckpt_path, model, device=device)
    logger.info(f"Loaded checkpoint: {ckpt_path}")

    data_cfg  = cfg.get("data", {})
    cache_dir = Path(cfg.get("training", {}).get("checkpoint_dir", "checkpoints")) / "dataset_cache"

    if (cache_dir / f"{split}.pkl").exists():
        ds = HeartbeatDataset.from_cache(cache_dir, split)
    else:
        logger.warning(f"Cache for '{split}' not found — building from raw records (slow)…")
        train_ds, val_ds, test_ds = HeartbeatDataset.from_records(
            data_path     = data_cfg.get("data_path", str(PTBXL_ROOT)),
            sampling_rate = data_cfg.get("sampling_rate", 500),
            t_fixed       = data_cfg.get("t_fixed", 500),
            beat_window_pre  = data_cfg.get("beat_window_pre",  100),
            beat_window_post = data_cfg.get("beat_window_post", 400),
            val_fold  = data_cfg.get("val_fold",  9),
            test_fold = data_cfg.get("test_fold", 10),
            lead_idx  = data_cfg.get("lead", 1),
        )
        ds = {"train": train_ds, "val": val_ds, "test": test_ds}[split]

    model_leads = data_cfg.get("model_leads", None)
    model_lead  = data_cfg.get("model_lead", None)
    if model_leads is not None:
        ds.model_leads = model_leads
    elif model_lead is not None:
        ds.model_lead = model_lead

    return _forward_pass(
        model, ds, data_cfg, use_uncertainty,
        n_mc_epistemic, n_mc_aleatoric,
        desc=f"Inference ({split})",
    )


def _forward_pass(
    model,
    ds,
    data_cfg: dict,
    use_uncertainty: bool,
    n_mc_epistemic: int,
    n_mc_aleatoric: int,
    desc: str = "Inference",
) -> dict:
    """Run model forward pass over a dataset and return collected arrays."""
    loader = DataLoader(
        ds,
        batch_size=data_cfg.get("batch_size", 256),
        shuffle=False,
        num_workers=data_cfg.get("num_workers", 4),
        pin_memory=True,
    )
    task_probs_all, task_labels_all = [], []
    c_probs_all, c_labels_all = [], []
    signals_all = []
    task_epistemic_all, task_aleatoric_all = [], []

    device = next(model.parameters()).device
    model.eval()
    with torch.no_grad():
        for batch in tqdm(loader, desc=desc):
            x      = batch["signal"].to(device)
            y_task = batch["task_label"]
            y_conc = batch["concept_labels"]

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
            signals_all.append(x.cpu().numpy())

    return {
        "task_probs":     np.concatenate(task_probs_all),
        "task_labels":    np.concatenate(task_labels_all),
        "c_probs":        np.concatenate(c_probs_all),
        "c_labels":       np.concatenate(c_labels_all),
        "signals":        np.concatenate(signals_all),
        "task_epistemic": np.concatenate(task_epistemic_all) if task_epistemic_all else None,
        "task_aleatoric": np.concatenate(task_aleatoric_all) if task_aleatoric_all else None,
    }


def run_inference_sliding_window(
    run_dir: Path,
    cfg: dict,
    device: str,
    use_uncertainty: bool,
    n_mc_epistemic: int,
    n_mc_aleatoric: int,
    n_records: int = 30,
) -> dict:
    """Build a sliding-window subset from the test fold and run inference."""
    from models import build_model
    from train import Checkpointer

    ckpt_path = run_dir / "checkpoints" / "best_model.pt"
    model = build_model(cfg).to(device)
    Checkpointer.load(ckpt_path, model, device=device)
    logger.info(f"Loaded checkpoint (sliding window): {ckpt_path}")

    ds = build_sliding_window_subset(cfg, n_records=n_records)
    data_cfg = cfg.get("data", {})
    return _forward_pass(
        model, ds, data_cfg, use_uncertainty,
        n_mc_epistemic, n_mc_aleatoric,
        desc="Inference (sliding window)",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualise heartbeat CBM results")
    parser.add_argument("--run-dir",       required=True,          help="Path to training run directory")
    parser.add_argument("--split",         default="test",         help="Dataset split to evaluate (default: test)")
    parser.add_argument("--n-samples",     type=int, default=8,    help="Number of ECG signal examples to plot")
    parser.add_argument("--no-uncertainty",action="store_true",    help="Skip MC-uncertainty forward pass (faster)")
    parser.add_argument("--n-mc-epistemic",type=int, default=20,   help="MC epistemic samples")
    parser.add_argument("--n-mc-aleatoric",type=int, default=30,   help="MC aleatoric samples")
    parser.add_argument("--lead",          type=int, default=1,    help="Lead to show in signal plot (default: 1)")
    parser.add_argument("--full-lead",     action="store_true",    help="Also generate per-sample 12-lead signal plots")
    parser.add_argument("--sliding-window",action="store_true",    help="Also plot examples from a sliding-window dataset")
    parser.add_argument("--sw-n-records",  type=int, default=30,   help="Number of test records to use for sliding window (default: 30)")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    out_dir = run_dir / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Load config ───────────────────────────────────────────────────────────
    cfg_path = run_dir / "config.yaml"
    if not cfg_path.exists():
        cfg_path = Path(__file__).parent / "config.yaml"
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)
    concept_names: List[str] = cfg.get("concept_names", ["P","Q","R","S","T","U","PR","QRS","ST","QT"])

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Device: {device}  |  Output: {out_dir}")

    # ── 1. Training curves (log-parsing only, no GPU needed) ─────────────────
    plot_training_curves(run_dir, out_dir)

    # ── Metrics JSON (if already saved from run_eval.sh) ─────────────────────
    metrics_path = run_dir / f"eval_{args.split}_metrics.json"
    if not metrics_path.exists():
        metrics_path = run_dir / "test_metrics.json"

    use_uncertainty = not args.no_uncertainty

    # ── Run inference ─────────────────────────────────────────────────────────
    preds = run_inference(
        run_dir, cfg, args.split, device,
        use_uncertainty, args.n_mc_epistemic, args.n_mc_aleatoric,
    )

    task_probs  = preds["task_probs"]
    task_labels = preds["task_labels"]
    c_probs     = preds["c_probs"]
    c_labels    = preds["c_labels"]
    signals     = preds["signals"]

    # Recompute metrics fresh
    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
    task_preds = (task_probs > 0.5).astype(int)
    metrics: dict = {
        "task/accuracy":  accuracy_score(task_labels, task_preds),
        "task/f1":        f1_score(task_labels, task_preds, zero_division=0),
        "task/precision": precision_score(task_labels, task_preds, zero_division=0),
        "task/recall":    recall_score(task_labels, task_preds, zero_division=0),
    }
    try:
        metrics["task/auroc"] = roc_auc_score(task_labels, task_probs)
    except ValueError:
        metrics["task/auroc"] = float("nan")
    K = c_probs.shape[1]
    for k in range(min(K, c_labels.shape[1])):
        name = concept_names[k] if k < len(concept_names) else f"C{k}"
        preds_k = (c_probs[:, k] > 0.5).astype(int)
        metrics[f"concept/{name}/f1"]       = f1_score(c_labels[:, k].astype(int), preds_k, zero_division=0)
        metrics[f"concept/{name}/accuracy"] = accuracy_score(c_labels[:, k].astype(int), preds_k)

    # ── 2-6. Plots ────────────────────────────────────────────────────────────
    plot_metrics_bar(metrics, concept_names, out_dir, args.split)
    plot_roc(task_probs, task_labels, out_dir, args.split)
    plot_concept_heatmap(c_probs, c_labels, concept_names, out_dir, args.split)
    plot_concept_confusion(c_probs, c_labels, concept_names, out_dir, args.split)

    if use_uncertainty and preds["task_epistemic"] is not None:
        plot_uncertainty_scatter(
            task_probs, preds["task_epistemic"], preds["task_aleatoric"],
            task_labels, out_dir, args.split,
        )

    # ── 7. Signal examples ────────────────────────────────────────────────────
    plot_signal_examples(
        signals, task_probs, task_labels, c_probs, c_labels,
        concept_names, out_dir, args.split,
        n_samples=args.n_samples,
        lead_idx=args.lead,
        sampling_rate=cfg.get("data", {}).get("sampling_rate", 500),
    )

    # ── 7b. Full 12-lead signal examples ─────────────────────────────────────
    if args.full_lead:
        plot_signal_examples_full_lead(
            signals, task_probs, task_labels, c_probs, c_labels,
            concept_names, out_dir, args.split,
            n_samples=min(args.n_samples, 4),
            sampling_rate=cfg.get("data", {}).get("sampling_rate", 500),
        )

    # ── 8. Sliding-window examples ────────────────────────────────────────────
    if args.sliding_window:
        logger.info(f"Building sliding-window subset ({args.sw_n_records} records)…")
        sw_preds = run_inference_sliding_window(
            run_dir, cfg, device,
            use_uncertainty, args.n_mc_epistemic, args.n_mc_aleatoric,
            n_records=args.sw_n_records,
        )
        sw_sr = cfg.get("data", {}).get("sampling_rate", 500)
        plot_signal_examples(
            sw_preds["signals"], sw_preds["task_probs"], sw_preds["task_labels"],
            sw_preds["c_probs"], sw_preds["c_labels"],
            concept_names, out_dir, split=f"{args.split}_sw",
            n_samples=args.n_samples,
            lead_idx=args.lead,
            sampling_rate=sw_sr,
        )
        if args.full_lead:
            plot_signal_examples_full_lead(
                sw_preds["signals"], sw_preds["task_probs"], sw_preds["task_labels"],
                sw_preds["c_probs"], sw_preds["c_labels"],
                concept_names, out_dir, split=f"{args.split}_sw",
                n_samples=min(args.n_samples, 4),
                sampling_rate=sw_sr,
            )

    logger.info(f"\nAll figures saved to: {out_dir}")


if __name__ == "__main__":
    main()
