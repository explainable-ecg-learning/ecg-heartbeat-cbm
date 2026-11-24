"""
experiments/plot_seg_comparison.py — Generate the 2-panel editorial figure
comparing per-timestamp ECG wave segmentation across dense model families
(reads experiments/results/summary.json → writes assets/segmentation_comparison.png).
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
SUMMARY_PATH = REPO_ROOT / "experiments" / "results" / "summary.json"
OUT_PATH = REPO_ROOT / "assets" / "segmentation_comparison.png"


def main() -> None:
    summary = json.loads(SUMMARY_PATH.read_text())
    ms = summary["dense_multiseed"]
    dm = summary["dense_models"]

    plt.rcParams.update({
        "figure.dpi":        150,
        "font.size":         9.5,
        "axes.spines.top":   False,
        "axes.spines.right": False,
        "axes.edgecolor":    "#d0d0d0",
        "axes.grid":         True,
        "grid.color":        "#e2e2e2",
        "grid.alpha":        0.75,
        "grid.linestyle":    "--",
        "figure.facecolor":  "#ffffff",
        "axes.facecolor":    "#ffffff",
    })

    families = [
        ("dense_cbm_detached", "DenseCBM (Detached)", "#2d7a56"),
        ("dense_multitask",    "DenseMultiTask",      "#2b5b84"),
        ("dense_cem",          "DenseCEM",            "#b83b2a"),
        ("dense_cbm_joint",    "DenseCBM (Joint)",    "#7c5295"),
    ]
    waves = ["P", "PQ", "QRS", "ST", "T", "TP"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.5, 4.8), gridspec_kw={"width_ratios": [1.32, 1.0]})

    # ── Left panel: Per-Wave Hard Dice (mean ± std across 3 seeds) ────────────
    x = np.arange(len(waves))
    width = 0.19
    offsets = np.linspace(-1.5 * width, 1.5 * width, len(families))

    for idx, (fkey, label, col) in enumerate(families):
        st = ms[fkey]
        means = [st[f"seg/{w}/hard_dice"]["mean"] for w in waves]
        stds  = [st[f"seg/{w}/hard_dice"]["std"]  for w in waves]
        ax1.bar(
            x + offsets[idx], means, width=width * 0.92,
            yerr=stds, capsize=2.2, error_kw={"elinewidth": 1.0, "ecolor": "#333333"},
            color=col, alpha=0.88, label=label, zorder=3,
        )

    ax1.set_xticks(x)
    ax1.set_xticklabels(waves, fontweight="bold")
    ax1.set_ylim(0.70, 1.015)
    ax1.set_ylabel("Hard Dice (vs. Zenodo 7610236)")
    ax1.set_title("Per-Wave Hard Dice across 2.20M Timestamps (n=3 seeds)", fontsize=10.5, fontweight="bold", pad=42)
    ax1.legend(
        loc="lower center", bbox_to_anchor=(0.5, 1.005),
        ncol=2, frameon=False, fontsize=8.5, columnspacing=1.2,
    )

    # ── Right panel: Soft Fidelity (KL ↓) vs. Seg Error-Detection AUROC ↑ ──────
    pts = [
        ("DenseCBM (Detached)", ms["dense_cbm_detached"], "#2d7a56", "o", (0.0006, -0.0005), "left"),
        ("DenseMultiTask",      ms["dense_multitask"],    "#2b5b84", "s", (-0.0006, 0.0085), "center"),
        ("DenseCBM (Joint)",    ms["dense_cbm_joint"],    "#7c5295", "D", (0.0002, -0.0072), "center"),
        ("DenseCEM",            ms["dense_cem"],          "#b83b2a", "^", (-0.0021, 0.0005), "right"),
    ]

    for label, st, col, marker, (dx, dy), ha in pts:
        kl_m, kl_s = st["seg/mean_kl"]["mean"], st["seg/mean_kl"]["std"]
        ed_m, ed_s = st["uncertainty/seg_error_detection_auroc"]["mean"], st["uncertainty/seg_error_detection_auroc"]["std"]
        bi = st["uncertainty/boundary_vs_interior"]["mean"]
        ax2.errorbar(
            kl_m, ed_m, xerr=kl_s, yerr=ed_s,
            fmt=marker, markersize=8.5, color=col, ecolor=col,
            elinewidth=1.2, capsize=3, zorder=4,
        )
        ax2.annotate(
            f"{label}\n({bi:.2f}× Bnd/Int σ)",
            xy=(kl_m, ed_m), xytext=(kl_m + dx, ed_m + dy),
            ha=ha, va="center", fontsize=8.2, fontweight="bold", color=col,
        )

    # Heteroscedastic ablation single run
    nh = dm["dense_cem_nohetero"]
    nh_kl, nh_ed, nh_bi = nh["seg/mean_kl"], nh["uncertainty/seg_error_detection_auroc"], nh["uncertainty/boundary_vs_interior"]
    cem_kl = ms["dense_cem"]["seg/mean_kl"]["mean"]
    cem_ed = ms["dense_cem"]["uncertainty/seg_error_detection_auroc"]["mean"]

    ax2.scatter([nh_kl], [nh_ed], marker="v", s=64, color="#767676", zorder=4)
    ax2.annotate(
        f"DenseCEM (No Hetero)\n({nh_bi:.2f}× Bnd/Int σ)",
        xy=(nh_kl, nh_ed), xytext=(nh_kl + 0.0007, nh_ed + 0.0028),
        ha="left", va="center", fontsize=8.0, color="#555555",
    )
    ax2.annotate(
        "", xy=(cem_kl, cem_ed), xytext=(nh_kl, nh_ed),
        arrowprops=dict(arrowstyle="->", linestyle="--", color="#b83b2a", alpha=0.6, lw=1.2, shrinkA=6, shrinkB=6),
    )

    ax2.set_xlim(0.0205, 0.0392)
    ax2.set_ylim(0.840, 0.897)
    ax2.set_xlabel("Soft-Mask KL Divergence ↓  (left = sharper masks)")
    ax2.set_ylabel("Seg Error-Detection AUROC ↑")
    ax2.set_title("Mask Sharpness vs. Boundary Uncertainty Calibration", fontsize=10.5, fontweight="bold", pad=42)

    plt.tight_layout()
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PATH, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {OUT_PATH} ({OUT_PATH.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
