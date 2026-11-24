"""
experiments/collect_results.py — Aggregate evaluation metrics across all trained
runs in models/ (including multi-seed mean±std, 95% bootstrap CIs, ablations,
and cross-delineator benchmarks) into experiments/results/summary.{json,md}.
"""

from __future__ import annotations

import json
import re
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import numpy as np

REPO_ROOT   = Path(__file__).resolve().parent.parent
MODELS_DIR  = REPO_ROOT / "models"
RESULTS_DIR = REPO_ROOT / "experiments" / "results"

SEED_RE = re.compile(r"^(.*)_s(\d+)$")


def _read_json(path: Path):
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


def _fmt(val, digits: int = 4, suffix: str = "") -> str:
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return "—"
    return f"{val:.{digits}f}{suffix}"


def _aggregate_seeds(runs: Dict[str, dict]) -> Dict[str, dict]:
    """Group runs matching `<family>_s<seed>` and compute mean ± std across seeds."""
    grouped: Dict[str, List[dict]] = defaultdict(list)
    for name, metrics in runs.items():
        m = SEED_RE.match(name)
        if m:
            grouped[m.group(1)].append(metrics)

    agg: Dict[str, dict] = {}
    for family, items in sorted(grouped.items()):
        if len(items) < 2:
            continue
        keys = set().union(*[d.keys() for d in items])
        stats: Dict[str, dict] = {"n_seeds": len(items)}
        for k in sorted(keys):
            vals = [float(d[k]) for d in items if k in d and isinstance(d[k], (int, float)) and not np.isnan(d[k])]
            if len(vals) == len(items):
                stats[k] = {
                    "mean": float(np.mean(vals)),
                    "std":  float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
                }
        agg[family] = stats
    return agg


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    summary: Dict[str, dict] = {
        "window_models": {},
        "dense_models": {},
        "dense_multiseed": {},
        "pathology_models": {},
        "pathology_multiseed": {},
        "cross_delineator": None,
    }

    # 1. Window-level runs
    window_names = [
        "cbm_v2", "cbm_randint_v2", "cem_v2",
        "cbm_softseg", "cem_softseg", "cbm_softseg_masked", "cem_softseg_masked", "blackbox_softseg",
    ]
    for name in window_names:
        rdir = MODELS_DIR / name
        m = _read_json(rdir / "eval_test_metrics.json") or _read_json(rdir / "test_metrics.json")
        if m:
            summary["window_models"][name] = m
            sub = RESULTS_DIR / ("window_softseg" if "softseg" in name else "window_neurokit")
            sub.mkdir(parents=True, exist_ok=True)
            for fname in ("eval_test_metrics.json", "intervention_test.json"):
                if (rdir / fname).exists():
                    shutil.copy2(rdir / fname, sub / f"{name}_{fname}")

    # 2. Per-timestamp dense runs (including baselines, ablations, and multi-seed)
    for rdir in sorted(MODELS_DIR.glob("dense_*")):
        m = _read_json(rdir / "eval_test_metrics.json")
        if m:
            summary["dense_models"][rdir.name] = m
            sub = RESULTS_DIR / "dense_softseg"
            sub.mkdir(parents=True, exist_ok=True)
            shutil.copy2(rdir / "eval_test_metrics.json", sub / f"{rdir.name}_eval_test_metrics.json")

    summary["dense_multiseed"] = _aggregate_seeds(summary["dense_models"])

    # 3. Downstream 5-class pathology runs
    for rdir in sorted(MODELS_DIR.glob("pathology_*")):
        m = _read_json(rdir / "test_metrics.json")
        if m:
            summary["pathology_models"][rdir.name] = m
            sub = RESULTS_DIR / "pathology"
            sub.mkdir(parents=True, exist_ok=True)
            shutil.copy2(rdir / "test_metrics.json", sub / f"{rdir.name}_test_metrics.json")

    summary["pathology_multiseed"] = _aggregate_seeds(summary["pathology_models"])

    # 4. Cross-delineator benchmark
    cross = _read_json(RESULTS_DIR / "soft_seg_eval_all" / "soft_seg_comparison.json")
    if cross:
        summary["cross_delineator"] = cross

    (RESULTS_DIR / "summary.json").write_text(json.dumps(summary, indent=2))

    # Build clean Markdown summary
    lines: List[str] = ["# Experiment Results Summary (PTB-XL Test Fold 10)", ""]

    if summary["window_models"]:
        lines += [
            "## 1. Window-Level Heartbeat Models",
            "",
            "| Run | Task Acc | Task AUROC | ECE ↓ | Concept Macro-F1 | Flip Rate ↑ | OIS ↓ | Err-Det AUROC |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
        for name, m in summary["window_models"].items():
            lines.append(
                f"| `{name}` | {_fmt(m.get('task/accuracy'))} | "
                f"{_fmt(m.get('task/auroc'))} | {_fmt(m.get('task/ece'))} | "
                f"{_fmt(m.get('concept/macro_f1'))} | "
                f"{_fmt(m.get('faithfulness/counterfactual_flip_rate'))} | "
                f"{_fmt(m.get('leakage/ois'))} | "
                f"{_fmt(m.get('uncertainty/error_detection_auroc'))} |"
            )
        lines.append("")

    if summary["cross_delineator"]:
        cd = summary["cross_delineator"]
        ceil = cd["inter_delineator_nk_vs_ecgdeli"]["task_agreement_acc"]
        lines += [
            f"## 2. Four-Way Dense Sliding-Window Cross-Delineator Benchmark (NK2 ↔ ECGdeli ceiling: `{ceil:.4f}`)",
            "",
            "| Model | vs. Zenodo Acc | vs. Zenodo AUROC | vs. Zenodo Macro-F1 (`QT`) | vs. NeuroKit2 Acc | vs. NeuroKit2 Macro-F1 |",
            "|---|---:|---:|---:|---:|---:|",
        ]
        for name, d in cd.get("models", {}).items():
            zs, nk = d["vs_zenodo_ecgdeli"], d["vs_neurokit2"]
            lines.append(
                f"| `{name}` | {zs['task/accuracy']:.4f} | {zs['task/auroc']:.4f} | "
                f"{zs['concept/shared_macro_f1']:.4f} ({zs['concept/QT/f1']:.4f}) | "
                f"{nk['task/accuracy']:.4f} | {nk['concept/shared_macro_f1']:.4f} |"
            )
        lines.append("")

    if summary["dense_multiseed"]:
        lines += [
            "## 3a. Per-Timestamp Dense 1D U-Net — Multi-Seed Summary (`mean ± std`, 100 Hz · 2.20M timestamps)",
            "",
            "| Model Family (n) | Seg Acc | Hard Macro-Dice | Pathology AUROC | Path ECE ↓ | Task Err-Det ↑ | GT-Mask Interv. | Excess Leakage Δ | Seg Err-Det ↑ | Bnd/Int σ | Ablate `ST` Shift |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
        def _ms(st: dict, key: str, d: int = 4, suf: str = "") -> str:
            if key not in st:
                return "—"
            return f"{st[key]['mean']:.{d}f} ± {st[key]['std']:.{d}f}{suf}"

        for fam, st in summary["dense_multiseed"].items():
            lines.append(
                f"| `{fam}` (n={st['n_seeds']}) | {_ms(st, 'seg/argmax_acc')} | "
                f"{_ms(st, 'seg/macro_hard_dice')} | {_ms(st, 'pathology/macro_auroc')} | "
                f"{_ms(st, 'pathology/macro_ece')} | {_ms(st, 'uncertainty/task_error_detection_auroc')} | "
                f"{_ms(st, 'pathology/intervened_macro_auroc')} | "
                f"{_ms(st, 'leakage/bottleneck_excess_auroc')} | "
                f"{_ms(st, 'uncertainty/seg_error_detection_auroc')} | "
                f"{_ms(st, 'uncertainty/boundary_vs_interior', 2, '×')} | "
                f"{_ms(st, 'counterfactual/ablate_ST_task_shift', 3)} |"
            )
        lines.append("")

    if summary["dense_models"]:
        lines += [
            "## 3b. Per-Timestamp Dense 1D U-Net — All Individual Runs & Ablations",
            "",
            "| Run | Seg Acc | Hard Macro-Dice | Soft MAE ↓ | Pathology AUROC (95% CI) | Path ECE ↓ | Task Err-Det ↑ | GT-Mask Interv. | Mask/BN Leak Δ | Seg Err-Det ↑ | Bnd/Int σ | Ablate `ST` |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
        for name, m in summary["dense_models"].items():
            p_auc = _fmt(m.get("pathology/macro_auroc"))
            if "pathology/macro_auroc_ci95_lo" in m:
                p_auc += f" [{m['pathology/macro_auroc_ci95_lo']:.3f}, {m['pathology/macro_auroc_ci95_hi']:.3f}]"
            if "leakage/soft_mask_excess_auroc" in m:
                lk_str = f"{m['leakage/soft_mask_excess_auroc']:+.3f} / {m['leakage/bottleneck_excess_auroc']:+.3f}"
            else:
                lk_str = "—"
            lines.append(
                f"| `{name}` | {_fmt(m.get('seg/argmax_acc'))} | {_fmt(m.get('seg/macro_hard_dice'))} | "
                f"{_fmt(m.get('seg/mean_mae'))} | {p_auc} | "
                f"{_fmt(m.get('pathology/macro_ece'))} | {_fmt(m.get('uncertainty/task_error_detection_auroc'))} | "
                f"{_fmt(m.get('pathology/intervened_macro_auroc'))} | {lk_str} | "
                f"{_fmt(m.get('uncertainty/seg_error_detection_auroc'))} | "
                f"{_fmt(m.get('uncertainty/boundary_vs_interior'), 2, '×')} | "
                f"{_fmt(m.get('counterfactual/ablate_ST_task_shift'), 3)} |"
            )
        lines.append("")

    if summary["pathology_models"]:
        lines += [
            "## 4. Record-Level 5-Class PTB-XL Pathology Models (20 clinical concepts)",
            "",
            "| Run | Macro-AUROC (95% CI) | NORM | MI | STTC | CD | HYP | Concept R² | Intervened AUROC |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
        for name, m in summary["pathology_models"].items():
            p_auc = _fmt(m.get("pathology/macro_auroc"))
            if "pathology/macro_auroc_ci95_lo" in m:
                p_auc += f" [{m['pathology/macro_auroc_ci95_lo']:.3f}, {m['pathology/macro_auroc_ci95_hi']:.3f}]"
            lines.append(
                f"| `{name}` | {p_auc} | {_fmt(m.get('pathology/NORM/auroc'))} | "
                f"{_fmt(m.get('pathology/MI/auroc'))} | {_fmt(m.get('pathology/STTC/auroc'))} | "
                f"{_fmt(m.get('pathology/CD/auroc'))} | {_fmt(m.get('pathology/HYP/auroc'))} | "
                f"{_fmt(m.get('concept/mean_r2'), 3)} | {_fmt(m.get('pathology/intervened_macro_auroc'))} |"
            )
        lines.append("")

    (RESULTS_DIR / "summary.md").write_text("\n".join(lines))
    print(f"Saved aggregated results → {RESULTS_DIR / 'summary.json'} & {RESULTS_DIR / 'summary.md'}")


if __name__ == "__main__":
    main()
