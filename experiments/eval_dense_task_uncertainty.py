"""
experiments/eval_dense_task_uncertainty.py — Evaluate pathology calibration
(ECE, Brier) and MC-Dropout task error-detection AUROC across all trained
models/dense_* runs (including DenseBlackBox & DenseMultiTask) and merge the
new metrics into each run's eval_test_metrics.json without retraining.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict

import torch
import yaml
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from data import resolve_data_paths  # noqa: E402
from dense_seg import (  # noqa: E402
    build_dense_model,
    evaluate_dense,
    evaluate_dense_task_uncertainty,
    get_dense_concept_names,
    load_or_build_dense_datasets,
)
from utils import set_seed  # noqa: E402


def main() -> None:
    models_dir = REPO_ROOT / "models"
    with open(REPO_ROOT / "config.yaml") as f:
        root_cfg = yaml.safe_load(f)

    data_path, _ = resolve_data_paths(root_cfg)
    raw_ss = root_cfg.get("data", {}).get("soft_seg_path", "ptbxl_segmentations_8bit.zip")
    data_dir = root_cfg.get("data", {}).get("data_dir", str(data_path.parent))
    soft_seg_path = Path(raw_ss) if Path(raw_ss).is_absolute() else (Path(data_dir) / raw_ss)
    cache_root = Path(root_cfg.get("training", {}).get("checkpoint_dir", "checkpoints")) / "dense_softseg_cache"
    if not cache_root.is_absolute():
        cache_root = REPO_ROOT / cache_root

    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    test_loaders: Dict[str, DataLoader] = {}

    def _get_test_loader(concept_mode: str) -> DataLoader:
        if concept_mode not in test_loaders:
            splits = load_or_build_dense_datasets(
                data_path=data_path,
                soft_seg_path=soft_seg_path,
                cache_root=cache_root,
                concept_mode=concept_mode,
            )
            test_loaders[concept_mode] = DataLoader(
                splits["test"],
                batch_size=128,
                shuffle=False,
                num_workers=4,
                pin_memory=torch.cuda.is_available(),
            )
        return test_loaders[concept_mode]

    run_dirs = sorted(models_dir.glob("dense_*"))
    for rdir in run_dirs:
        cfg_path = rdir / "config.yaml"
        ckpt_path = rdir / "checkpoints" / "best_model.pt"
        metrics_path = rdir / "eval_test_metrics.json"
        if not (cfg_path.exists() and ckpt_path.exists()):
            print(f"Skipping {rdir.name} (missing config or checkpoint)")
            continue

        cfg = yaml.safe_load(cfg_path.read_text())
        concept_mode = str(cfg.get("data", {}).get("concept_mode", "wave6"))
        concept_names = cfg.get("concept_names", get_dense_concept_names(concept_mode))
        seed = int(cfg.get("training", {}).get("seed", 42))
        set_seed(seed)

        loader = _get_test_loader(concept_mode)
        model = build_dense_model(cfg).to(device)
        state = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(state["model"], strict=True)

        det_m = evaluate_dense(
            model, loader, device, concept_names,
            use_bf16=use_bf16, intervene_all=False, compute_bootstrap=False,
        )
        unc_m = evaluate_dense_task_uncertainty(
            model, loader, device, n_mc_epistemic=16, use_bf16=use_bf16,
        )

        existing = json.loads(metrics_path.read_text()) if metrics_path.exists() else {}
        existing["pathology/macro_ece"] = det_m["pathology/macro_ece"]
        existing["pathology/macro_brier"] = det_m["pathology/macro_brier"]
        existing.update(unc_m)
        metrics_path.write_text(json.dumps(existing, indent=2))

        print(
            f"{rdir.name:<28s} | AUROC={existing.get('pathology/macro_auroc', det_m['pathology/macro_auroc']):.4f} | "
            f"ECE={existing['pathology/macro_ece']:.4f} | Brier={existing['pathology/macro_brier']:.4f} | "
            f"TaskErrDet={existing['uncertainty/task_error_detection_auroc']:.4f} | "
            f"MeanStd={existing['uncertainty/task_mean_std']:.4f}"
        )


if __name__ == "__main__":
    main()
