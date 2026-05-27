#!/usr/bin/env bash
# run_train.sh — Train the ECG heartbeat CBM, then evaluate and visualise.
#
# Usage:
#   bash run_train.sh                          # default config + auto run-dir
#   bash run_train.sh --config config.yaml     # explicit config
#   bash run_train.sh --run-dir runs/my_run    # explicit output dir
#   bash run_train.sh --gpu 1                  # force a specific GPU
#   bash run_train.sh --skip-eval              # skip eval+vis after training
#
# All unknown flags are forwarded to train.py.

set -euo pipefail

# ── Defaults ──────────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG="${SCRIPT_DIR}/config.yaml"
GPU=""
RUN_DIR=""
SKIP_EVAL=false
EXTRA_ARGS=()

# ── Argument parsing ──────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --config)    CONFIG="$2";   shift 2 ;;
        --gpu)       GPU="$2";      shift 2 ;;
        --run-dir)   RUN_DIR="$2";  shift 2 ;;
        --skip-eval) SKIP_EVAL=true; shift ;;
        *)           EXTRA_ARGS+=("$1"); shift ;;
    esac
done

if [[ -n "$GPU" ]]; then
    export CUDA_VISIBLE_DEVICES="$GPU"
fi

# Auto-generate run dir so eval + vis can reuse the same path
if [[ -z "$RUN_DIR" ]]; then
    LOG_DIR=$(python3 -c "
import yaml, sys
cfg = yaml.safe_load(open('${CONFIG}'))
print(cfg.get('training', {}).get('log_dir', 'runs'))
" 2>/dev/null || echo "runs")
    RUN_DIR="${SCRIPT_DIR}/${LOG_DIR}/$(date +%Y%m%d_%H%M%S)"
fi

echo "============================================================"
echo " ECG Heartbeat CBM — Training"
echo "============================================================"
echo " Config  : ${CONFIG}"
echo " Run dir : ${RUN_DIR}"
echo " GPU     : ${CUDA_VISIBLE_DEVICES:-<default>}"
echo " Extra   : ${EXTRA_ARGS[*]:-<none>}"
echo "============================================================"

# ── Activate venv if present ──────────────────────────────────────────────────
if [[ -f "${SCRIPT_DIR}/.venv_new/bin/activate" ]]; then
    # shellcheck disable=SC1091
    source "${SCRIPT_DIR}/.venv_new/bin/activate"
    echo "Activated venv: ${SCRIPT_DIR}/.venv_new"
elif [[ -n "${VIRTUAL_ENV:-}" ]]; then
    echo "Using active venv: ${VIRTUAL_ENV}"
fi

# ── Train ─────────────────────────────────────────────────────────────────────
cd "${SCRIPT_DIR}"

python src/train.py \
    --config "${CONFIG}" \
    --run-dir "${RUN_DIR}" \
    "${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}"

if [[ "$SKIP_EVAL" == true ]]; then
    echo ""
    echo "Training done. Skipping eval+vis (--skip-eval). Run dir: ${RUN_DIR}"
    exit 0
fi

# ── Evaluate ──────────────────────────────────────────────────────────────────
echo ""
echo "============================================================"
echo " ECG Heartbeat CBM — Evaluation (test split)"
echo "============================================================"

python - <<PYEOF
import json, sys, logging
from pathlib import Path

sys.path.insert(0, "${SCRIPT_DIR}/src")

import torch
import yaml

from data import HeartbeatDataset, make_dataloaders, PTBXL_ROOT
from models import build_model, CONCEPT_NAMES
from train import evaluate, format_metrics, Checkpointer

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

run_dir   = Path("${RUN_DIR}")
ckpt_path = run_dir / "checkpoints" / "best_model.pt"
split     = "test"

cfg_path = run_dir / "config.yaml"
with open(cfg_path) as f:
    cfg = yaml.safe_load(f)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")

model = build_model(cfg).to(device)
Checkpointer.load(ckpt_path, model, device=device)
print(f"Loaded checkpoint: {ckpt_path}")

data_cfg  = cfg.get("data", {})
cache_dir = Path(cfg.get("training", {}).get("checkpoint_dir", "checkpoints")) / "dataset_cache"

if (cache_dir / f"{split}.pkl").exists():
    ds = HeartbeatDataset.from_cache(cache_dir, split)
else:
    print(f"No cache for split '{split}' — rebuilding from records…")
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

model_leads = data_cfg.get("model_leads", None)
model_lead  = data_cfg.get("model_lead", None)
if model_leads is not None:
    ds.model_leads = model_leads
elif model_lead is not None:
    ds.model_lead = model_lead

from torch.utils.data import DataLoader
loader = DataLoader(ds, batch_size=data_cfg.get("batch_size", 256), num_workers=4, pin_memory=True)

metrics = evaluate(model, loader, device=device)
print("\\nEvaluation results:")
print(format_metrics(metrics))

out_path = run_dir / f"eval_{split}_metrics.json"
with open(out_path, "w") as f:
    json.dump(metrics, f, indent=2)
print(f"\\nSaved metrics → {out_path}")
PYEOF

# ── Visualise ─────────────────────────────────────────────────────────────────
echo ""
echo "============================================================"
echo " ECG Heartbeat CBM — Visualisation"
echo "============================================================"

python src/visualize.py --run-dir "${RUN_DIR}"

echo ""
echo "All done. Artifacts in: ${RUN_DIR}"
