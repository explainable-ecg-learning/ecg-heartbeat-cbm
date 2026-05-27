#!/usr/bin/env bash
# run_eval.sh — Evaluate a trained heartbeat CBM checkpoint.
#
# Usage:
#   bash run_eval.sh --run-dir runs/20260422_120000
#   bash run_eval.sh --run-dir runs/20260422_120000 --split test
#   bash run_eval.sh --run-dir runs/20260422_120000 --checkpoint best_model.pt --gpu 0
#
# Outputs:
#   <run-dir>/eval_<split>_metrics.json   — flat metric dict
#   Printed table of task + per-concept metrics

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN_DIR=""
CHECKPOINT="best_model.pt"
SPLIT="test"
GPU=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --run-dir)    RUN_DIR="$2";     shift 2 ;;
        --checkpoint) CHECKPOINT="$2";  shift 2 ;;
        --split)      SPLIT="$2";       shift 2 ;;
        --gpu)        GPU="$2";         shift 2 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

if [[ -z "$RUN_DIR" ]]; then
    echo "Usage: bash run_eval.sh --run-dir <path/to/run>"
    exit 1
fi

if [[ -n "$GPU" ]]; then
    export CUDA_VISIBLE_DEVICES="$GPU"
fi

echo "============================================================"
echo " ECG Heartbeat CBM — Evaluation"
echo "============================================================"
echo " Run dir    : ${RUN_DIR}"
echo " Checkpoint : ${CHECKPOINT}"
echo " Split      : ${SPLIT}"
echo " GPU        : ${CUDA_VISIBLE_DEVICES:-<default>}"
echo "============================================================"

if [[ -f "${SCRIPT_DIR}/.venv_new/bin/activate" ]]; then
    # shellcheck disable=SC1091
    source "${SCRIPT_DIR}/.venv_new/bin/activate"
elif [[ -n "${VIRTUAL_ENV:-}" ]]; then
    echo "Using active venv: ${VIRTUAL_ENV}"
fi

cd "${SCRIPT_DIR}"

python - <<PYEOF
import json, sys, logging
from pathlib import Path

sys.path.insert(0, str(Path("${SCRIPT_DIR}") / "src"))

import torch
import yaml

from data import HeartbeatDataset, make_dataloaders, PTBXL_ROOT
from models import build_model, CONCEPT_NAMES
from train import evaluate, format_metrics, Checkpointer

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

run_dir    = Path("${RUN_DIR}")
ckpt_path  = run_dir / "checkpoints" / "${CHECKPOINT}"
split      = "${SPLIT}"

# Load config from run dir
cfg_path = run_dir / "config.yaml"
if not cfg_path.exists():
    cfg_path = Path("${SCRIPT_DIR}/config.yaml")
with open(cfg_path) as f:
    cfg = yaml.safe_load(f)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")

# Build model and load weights
model = build_model(cfg).to(device)
Checkpointer.load(ckpt_path, model, device=device)
print(f"Loaded checkpoint: {ckpt_path}")

# Load dataset
data_cfg  = cfg.get("data", {})
cache_dir = Path(cfg.get("training", {}).get("checkpoint_dir", "checkpoints")) / "dataset_cache"

if (cache_dir / f"{split}.pkl").exists():
    ds = HeartbeatDataset.from_cache(cache_dir, split)
else:
    print(f"No cache found for split '{split}'. Building from records (slow)...")
    train_ds, val_ds, test_ds = HeartbeatDataset.from_records(
        data_path     = data_cfg.get("data_path", str(PTBXL_ROOT)),
        sampling_rate = data_cfg.get("sampling_rate", 500),
        t_fixed       = data_cfg.get("t_fixed", 500),
        beat_window_pre  = data_cfg.get("beat_window_pre", 100),
        beat_window_post = data_cfg.get("beat_window_post", 400),
        val_fold  = data_cfg.get("val_fold", 9),
        test_fold = data_cfg.get("test_fold", 10),
        lead_idx  = data_cfg.get("lead", 1),
    )
    ds = {"train": train_ds, "val": val_ds, "test": test_ds}[split]

# Lead selection
model_leads = data_cfg.get("model_leads", None)
model_lead  = data_cfg.get("model_lead", None)
if model_leads is not None:
    ds.model_leads = model_leads
elif model_lead is not None:
    ds.model_lead = model_lead

from torch.utils.data import DataLoader
loader = DataLoader(
    ds, batch_size=data_cfg.get("batch_size", 256),
    shuffle=False, num_workers=data_cfg.get("num_workers", 4),
    pin_memory=True,
)

concept_names = cfg.get("concept_names", CONCEPT_NAMES)
metrics = evaluate(model, loader, device, concept_names)

print(f"\n{'='*50}")
print(f"  Results on [{split}] — {len(ds)} samples")
print(f"{'='*50}")
print(format_metrics(metrics))

out_path = run_dir / f"eval_{split}_metrics.json"
with open(out_path, "w") as f:
    json.dump(metrics, f, indent=2)
print(f"\nMetrics saved to: {out_path}")
PYEOF

echo ""
echo "Evaluation complete."
