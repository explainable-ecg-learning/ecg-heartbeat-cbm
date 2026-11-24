#!/usr/bin/env bash
# experiments/run_eval.sh — Evaluate a trained heartbeat CBM / CEM / Black-Box checkpoint.
#
# Usage:
#   bash experiments/run_eval.sh --run-dir models/my_run
#   bash experiments/run_eval.sh --run-dir models/my_run --split val
#   bash experiments/run_eval.sh --run-dir models/my_run --checkpoint best_model.pt --gpu 0 --interventions --uncertainty

set -euo pipefail

EXP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${EXP_DIR}/.." && pwd)"
RUN_DIR=""
CHECKPOINT="best_model.pt"
SPLIT="test"
GPU=""
EXTRA_FLAGS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --run-dir)       RUN_DIR="$2";     shift 2 ;;
        --checkpoint)    CHECKPOINT="$2";  shift 2 ;;
        --split)         SPLIT="$2";       shift 2 ;;
        --gpu)           GPU="$2";         shift 2 ;;
        --interventions|--uncertainty|--faithfulness) EXTRA_FLAGS+=("$1"); shift ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

if [[ -z "$RUN_DIR" ]]; then
    echo "Usage: bash experiments/run_eval.sh --run-dir <path/to/run> [--split test] [--interventions] [--uncertainty] [--faithfulness]"
    exit 1
fi

if [[ -n "$GPU" ]]; then
    export CUDA_VISIBLE_DEVICES="$GPU"
fi

echo "============================================================"
echo " ECG Heartbeat CBM / CEM — Evaluation"
echo "============================================================"
echo " Run dir    : ${RUN_DIR}"
echo " Checkpoint : ${CHECKPOINT}"
echo " Split      : ${SPLIT}"
echo " GPU        : ${CUDA_VISIBLE_DEVICES:-<default>}"
echo "============================================================"

if [[ -f "${REPO_ROOT}/.venv_new/bin/activate" ]]; then
    # shellcheck disable=SC1091
    source "${REPO_ROOT}/.venv_new/bin/activate"
elif [[ -n "${VIRTUAL_ENV:-}" ]]; then
    echo "Using active venv: ${VIRTUAL_ENV}"
fi

cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}/src:${PYTHONPATH:-}"

python src/evaluate.py \
    --run-dir "${RUN_DIR}" \
    --checkpoint "${CHECKPOINT}" \
    --split "${SPLIT}" \
    "${EXTRA_FLAGS[@]+"${EXTRA_FLAGS[@]}"}"

echo ""
echo "Evaluation complete."
