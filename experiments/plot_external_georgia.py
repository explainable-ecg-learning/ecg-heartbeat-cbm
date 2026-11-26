"""
experiments/plot_external_georgia.py — Generate the multi-panel editorial figure
for zero-shot external validation on Georgia 12-Lead ECG:
  - Top 3 panels: 4.0 s Lead-II zooms (NSR, RBBB|AF, LBBB) with zero-shot
    DenseCBM (Detached) wave bands, NeuroKit2 fiducial dots, and DenseCEM vs.
    DenseCBM boundary uncertainty σ(t).
  - Bottom-Left: PTB-XL (In-Domain) vs. Georgia (Zero-Shot OOD) NeuroKit2 F1@150ms.
  - Bottom-Right: Zero-shot clinical interval separation (QRS for LBBB/RBBB,
    PR for IAVB, QT for LQT vs. NSR) across all 8,770 clean Georgia ECGs.

Writes: assets/external_georgia_segmentation.png
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from external_data import extract_wave_spans_and_intervals

EVAL_JSON = REPO_ROOT / "experiments" / "results" / "external_georgia_eval.json"
SHOWCASE_NPZ = REPO_ROOT / "experiments" / "results" / "external_georgia_showcase.npz"
OUT_PATH = REPO_ROOT / "assets" / "external_georgia_segmentation.png"

WAVE_COLORS = {
    "P":   "#4c8bf5",
    "PQ":  "#88b7d5",
    "QRS": "#d95f43",
    "ST":  "#f0a830",
    "T":   "#3ca675",
}


def main() -> None:
    eval_data = json.loads(EVAL_JSON.read_text())
    sc = np.load(SHOWCASE_NPZ)

    plt.rcParams.update({
        "figure.dpi":        150,
        "font.size":         9.2,
        "axes.spines.top":   False,
        "axes.spines.right": False,
        "axes.edgecolor":    "#cfcfcf",
        "axes.grid":         True,
        "grid.color":        "#e5e5e5",
        "grid.alpha":        0.75,
        "grid.linestyle":    "--",
        "figure.facecolor":  "#ffffff",
        "axes.facecolor":    "#ffffff",
    })

    fig = plt.figure(figsize=(15.0, 11.4))
    top_bot = gridspec.GridSpec(
        2, 1, height_ratios=[3.2, 1.35], hspace=0.36,
        top=0.915, bottom=0.055, left=0.06, right=0.97,
    )
    outer = gridspec.GridSpecFromSubplotSpec(
        3, 1, subplot_spec=top_bot[0], hspace=0.36,
    )

    fig.suptitle(
        "Zero-Shot External Validation on Georgia 12-Lead ECG (PTB-XL-Trained Models)",
        fontsize=13.5, fontweight="bold", y=0.975, color="#1a1a1a",
    )

    # ── Top 3 panels: 4.0 s Lead-II zooms + σ(t) strips ──────────────────────
    t_start, t_end = 100, 500  # 1.0s .. 5.0s @ 100 Hz
    t_sec = np.arange(t_start, t_end) / 100.0
    tags = [str(x) for x in sc["tags"]]
    rec_ids = [str(x) for x in sc["record_ids"]]
    raw_labels = [str(x) for x in sc["label_names"]]

    panel_titles = [
        f"Georgia {rec_ids[0]} — Normal Sinus Rhythm ({raw_labels[0].replace('|', ', ')})",
        f"Georgia {rec_ids[1]} — Atrial Fibrillation + RBBB ({raw_labels[1].replace('|', ', ')})",
        f"Georgia {rec_ids[2]} — Left Bundle Branch Block ({raw_labels[2].replace('|', ', ')})",
    ]

    for slot in range(3):
        sub_gs = gridspec.GridSpecFromSubplotSpec(
            2, 1, subplot_spec=outer[slot], height_ratios=[2.25, 1.0], hspace=0.08
        )
        ax_w = fig.add_subplot(sub_gs[0])
        ax_u = fig.add_subplot(sub_gs[1], sharex=ax_w)

        sig = sc["lead2_100hz"][slot]
        cbm_p = sc["cbm_probs"][slot]
        cbm_cls = cbm_p.argmax(axis=0)
        info = extract_wave_spans_and_intervals(cbm_cls, fs=100)

        # Shade contiguous wave spans inside [t_start, t_end]
        for wname, wcol in WAVE_COLORS.items():
            for s, e, _ in info["spans"][wname]:
                if e <= t_start or s >= t_end:
                    continue
                xs = max(s, t_start) / 100.0
                xe = min(e, t_end) / 100.0
                ax_w.axvspan(xs, xe, color=wcol, alpha=0.22, lw=0)

        ax_w.plot(t_sec, sig[t_start:t_end], color="#1f2933", lw=1.25, zorder=3)

        # NeuroKit2 fiducial dots inside window
        for arr_key, marker, col, lbl in (
            ("nk_p", "o", "#1e5bbf", "NK2 P"),
            ("nk_r", "^", "#b83b2a", "NK2 R"),
            ("nk_t", "s", "#227a52", "NK2 T"),
        ):
            pts = sc[arr_key][slot]
            pts = pts[(pts >= t_start) & (pts < t_end)]
            if len(pts):
                idx_i = np.clip(np.round(pts).astype(int), t_start, t_end - 1)
                ax_w.scatter(pts / 100.0, sig[idx_i], marker=marker, s=26, color=col, zorder=5, edgecolors="white", linewidths=0.4)

        p_conf = float(cbm_p[0][cbm_cls == 0].mean()) if np.any(cbm_cls == 0) else 0.0
        if tags[slot] == "NSR":
            note = f"PR = {info['pr_ms']:.0f} ms · QRS = {info['qrs_ms']:.0f} ms · P-conf = {p_conf:.2f}"
        elif tags[slot] == "AF":
            note = f"Irregular RR · P-conf = {p_conf:.2f} · QRS = {info['qrs_ms']:.0f} ms · CEM P(CD) = {sc['cem_task'][slot, 3]:.2f}"
        else:
            note = f"Widened QRS = {info['qrs_ms']:.0f} ms · PR = {info['pr_ms']:.0f} ms · CEM P(CD) = {sc['cem_task'][slot, 3]:.2f}"

        ax_w.set_title(panel_titles[slot], loc="left", fontsize=10.0, fontweight="bold", pad=6, color="#1a1a1a")
        ax_w.set_title(note, loc="right", fontsize=8.8, fontweight="bold", pad=6, color="#374151")
        ax_w.set_ylabel("Lead II (z)", fontsize=8.5)
        ax_w.set_xlim(t_start / 100.0, (t_end - 1) / 100.0)
        y_lo, y_hi = np.percentile(sig[t_start:t_end], [0.5, 99.5])
        pad_y = max(0.35, 0.22 * (y_hi - y_lo))
        ax_w.set_ylim(y_lo - pad_y * 0.65, y_hi + pad_y)
        plt.setp(ax_w.get_xticklabels(), visible=False)

        # Shared wave legend on top-most panel only
        if slot == 0:
            handles = [mpatches.Patch(color=c, alpha=0.38, label=w) for w, c in WAVE_COLORS.items()]
            handles.extend([
                plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="#1e5bbf", markersize=5.5, label="NK2 P"),
                plt.Line2D([0], [0], marker="^", color="w", markerfacecolor="#b83b2a", markersize=6.0, label="NK2 R"),
                plt.Line2D([0], [0], marker="s", color="w", markerfacecolor="#227a52", markersize=5.5, label="NK2 T"),
            ])
            ax_w.legend(
                handles=handles, loc="lower center", bbox_to_anchor=(0.5, 1.24),
                ncol=8, frameon=False, fontsize=8.5, columnspacing=1.1, handletextpad=0.4,
            )

        # Uncertainty strip
        cbm_u = sc["cbm_std"][slot, t_start:t_end]
        cem_u = sc["cem_std"][slot, t_start:t_end]
        ax_u.plot(t_sec, cem_u, color="#b83b2a", lw=1.1, label="DenseCEM σ(t)")
        ax_u.plot(t_sec, cbm_u, color="#2d7a56", lw=1.0, alpha=0.88, linestyle="--", label="DenseCBM (Det.) σ(t)")
        ax_u.fill_between(t_sec, 0, cem_u, color="#b83b2a", alpha=0.12)
        ax_u.set_ylabel("σ(t)", fontsize=8.3)
        ax_u.set_ylim(0.0, max(0.045, float(max(cem_u.max(), cbm_u.max())) * 1.18))
        if slot == 0:
            ax_u.legend(loc="upper right", ncol=2, frameon=False, fontsize=7.8)
        if slot == 2:
            ax_u.set_xlabel("Time (seconds @ 100 Hz)", fontsize=9.0, labelpad=4)
        else:
            plt.setp(ax_u.get_xticklabels(), visible=False)

    # ── Bottom row: 2 side-by-side benchmark panels ───────────────────────────
    bot_gs = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=top_bot[1], width_ratios=[1.12, 1.0], wspace=0.28)
    ax_bl = fig.add_subplot(bot_gs[0])
    ax_br = fig.add_subplot(bot_gs[1])

    # Bottom-Left: PTB-XL (In-Domain) vs Georgia (OOD) NK2 F1 @ ±150ms
    s2 = eval_data["suite2_neurokit2_delineation"]["models"]
    model_specs = [
        ("dense_cbm_detached_s42", "DenseCBM (Detached)", "#2d7a56"),
        ("dense_cem_s42",          "DenseCEM",            "#b83b2a"),
        ("dense_cbm_joint_s42",    "DenseCBM (Joint)",    "#7c5295"),
        ("dense_multitask_s42",    "DenseMultiTask",      "#2b5b84"),
    ]
    groups = [
        ("PTB-XL QRS",  "ptbxl_fold10", "QRS"),
        ("Georgia QRS", "georgia",      "QRS"),
        ("PTB-XL P",    "ptbxl_fold10", "P"),
        ("Georgia P",   "georgia",      "P"),
        ("PTB-XL T",    "ptbxl_fold10", "T"),
        ("Georgia T",   "georgia",      "T"),
    ]
    gx = np.arange(len(groups))
    bw = 0.19
    offs = np.linspace(-1.5 * bw, 1.5 * bw, len(model_specs))

    for mi, (mkey, mlabel, mcol) in enumerate(model_specs):
        vals = [s2[mkey][dom][wv]["tol_150ms"]["f1"] for _, dom, wv in groups]
        bars = ax_bl.bar(gx + offs[mi], vals, width=bw * 0.90, color=mcol, alpha=0.88, label=mlabel, zorder=3)
        if mi in (0, 1):
            for b, v in zip(bars, vals):
                ax_bl.text(
                    b.get_x() + b.get_width() / 2.0, v + 0.006, f"{v:.3f}",
                    ha="center", va="bottom", rotation=90, fontsize=7.0, color=mcol, fontweight="bold",
                )

    ax_bl.set_xticks(gx)
    ax_bl.set_xticklabels([g[0] for g in groups], fontsize=8.4, fontweight="bold")
    ax_bl.set_ylim(0.70, 1.07)
    ax_bl.set_ylabel("Peak F1 vs. NeuroKit2 DWT (±150 ms)")
    ax_bl.set_title("Wave Detection: PTB-XL (In-Domain) vs. Georgia (Zero-Shot OOD)", fontsize=10.0, fontweight="bold", pad=32)
    ax_bl.legend(loc="lower center", bbox_to_anchor=(0.5, 1.005), ncol=2, frameon=False, fontsize=8.0, columnspacing=1.0)

    # Bottom-Right: Clinical interval median + IQR separation across 8,770 clean Georgia ECGs
    cm_cbm = eval_data["suite1_full_clean_cohort"]["models"]["dense_cbm_detached_s42"]["segmentation"]["clinical_morphology"]
    cm_cem = eval_data["suite1_full_clean_cohort"]["models"]["dense_cem_s42"]["segmentation"]["clinical_morphology"]

    cohorts = [
        ("QT: NSR\n(n=1,701)",   cm_cbm["lqt_vs_nsr_qt_ms"],   "neg", "#5f6c7b", None),
        ("QT: LQT\n(n=1,337)",   cm_cbm["lqt_vs_nsr_qt_ms"],   "pos", "#2b5b84", f"AUC {cm_cem['lqt_vs_nsr_qt_ms']['auroc']:.2f}"),
        ("PR: NSR\n(n=1,701)",   cm_cbm["iavb_vs_nsr_pr_ms"],  "neg", "#5f6c7b", None),
        ("PR: IAVB\n(n=670)",    cm_cbm["iavb_vs_nsr_pr_ms"],  "pos", "#7c5295", f"AUC {cm_cem['iavb_vs_nsr_pr_ms']['auroc']:.2f}"),
        ("QRS: NSR\n(n=1,701)",  cm_cbm["lbbb_vs_nsr_qrs_ms"], "neg", "#5f6c7b", None),
        ("QRS: RBBB\n(n=516)",   cm_cbm["rbbb_vs_nsr_qrs_ms"], "pos", "#d97706", f"AUC {cm_cbm['rbbb_vs_nsr_qrs_ms']['auroc']:.2f}"),
        ("QRS: LBBB\n(n=223)",   cm_cbm["lbbb_vs_nsr_qrs_ms"], "pos", "#b83b2a", f"AUC {cm_cbm['lbbb_vs_nsr_qrs_ms']['auroc']:.3f}"),
    ]

    yc = np.arange(len(cohorts))
    for i, (lbl, st, side, col, badge) in enumerate(cohorts):
        med = st[f"{side}_median"]
        q25, q75 = st[f"{side}_iqr"]
        ax_br.barh(i, med, height=0.56, color=col, alpha=0.82, zorder=3)
        ax_br.errorbar(med, i, xerr=[[med - q25], [q75 - med]], fmt="none", ecolor="#1a1a1a", elinewidth=1.4, capsize=3.2, zorder=4)
        txt = f"{med:.0f} ms" + (f"  ({badge})" if badge else "")
        ax_br.text(max(med, q75) + 7.0, i, txt, va="center", ha="left", fontsize=8.1, fontweight="bold", color=col)

    ax_br.axvline(120, color="#b83b2a", linestyle=":", lw=1.0, alpha=0.65)
    ax_br.axvline(200, color="#7c5295", linestyle=":", lw=1.0, alpha=0.65)
    ax_br.set_yticks(yc)
    ax_br.set_yticklabels([c[0] for c in cohorts], fontsize=8.1, fontweight="bold")
    ax_br.set_xlim(0, 585)
    ax_br.set_xlabel("Zero-Shot Predicted Interval Duration (ms, Median ± IQR)")
    ax_br.set_title("Clinical Morphology Validation on 8,770 Georgia ECGs", fontsize=10.0, fontweight="bold", pad=32)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PATH, dpi=150)
    plt.close(fig)
    print(f"Saved figure → {OUT_PATH}")


if __name__ == "__main__":
    main()
