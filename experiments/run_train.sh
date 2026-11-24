#!/usr/bin/env bash
# experiments/run_train.sh — Train the ECG heartbeat CBM / CEM, then evaluate and visualise.
#
# Usage:
#   bash experiments/run_train.sh                                 # default config + auto run-dir
#   bash experiments/run_train.sh --config config.yaml            # explicit config
#   bash experiments/run_train.sh --run-dir models/my_cem --set model.type=cem
#   bash experiments/run_train.sh --gpu 0                         # force a specific GPU
#   bash experiments/run_train.sh --skip-eval                     # skip eval+vis after training

set -euo pipefail

EXP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${EXP_DIR}/.." && pwd)"
CONFIG="${REPO_ROOT}/config.yaml"
GPU=""
RUN_DIR=""
SKIP_EVAL=false
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --config)    CONFIG="$2";    shift 2 ;;
        --gpu)       GPU="$2";       shift 2 ;;
        --run-dir)   RUN_DIR="$2";   shift 2 ;;
        --skip-eval) SKIP_EVAL=true; shift ;;
        *)           EXTRA_ARGS+=("$1"); shift ;;
    esac
done

if [[ -n "$GPU" ]]; then
    export CUDA_VISIBLE_DEVICES="$GPU"
fi

if [[ -z "$RUN_DIR" ]]; then
    LOG_DIR=$(python3 -c "
import yaml
cfg = yaml.safe_load(open('${CONFIG}'))
print(cfg.get('training', {}).get('log_dir', 'models'))
" 2>/dev/null || echo "models")
    RUN_DIR="${REPO_ROOT}/${LOG_DIR}/$(date +%Y%m%d_%H%M%S)"
fi

echo "============================================================"
echo " ECG Heartbeat CBM / CEM — Training"
echo "============================================================"
echo " Config  : ${CONFIG}"
echo " Run dir : ${RUN_DIR}"
echo " GPU     : ${CUDA_VISIBLE_DEVICES:-<default>}"
echo " Extra   : ${EXTRA_ARGS[*]:-<none>}"
echo "============================================================"

if [[ -f "${REPO_ROOT}/.venv_new/bin/activate" ]]; then
    # shellcheck disable=SC1091
    source "${REPO_ROOT}/.venv_new/bin/activate"
    echo "Activated venv: ${REPO_ROOT}/.venv_new"
elif [[ -n "${VIRTUAL_ENV:-}" ]]; then
    echo "Using active venv: ${VIRTUAL_ENV}"
fi

cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}/src:${PYTHONPATH:-}"

python src/train.py \
    --config "${CONFIG}" \
    --run-dir "${RUN_DIR}" \
    "${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}"

if [[ "$SKIP_EVAL" == true ]]; then
    echo ""
    echo "Training done. Skipping eval+vis (--skip-eval). Run dir: ${RUN_DIR}"
    exit 0
fi

echo ""
echo "============================================================"
echo " ECG Heartbeat CBM / CEM — Evaluation (test split)"
echo "============================================================"

python src/evaluate.py --run-dir "${RUN_DIR}" --split test --interventions --uncertainty --faithfulness

echo ""
echo "============================================================"
echo " ECG Heartbeat CBM / CEM — Visualisation"
echo "============================================================"

python src/visualize.py --run-dir "${RUN_DIR}"

echo ""
echo "All done. Artifacts in: ${RUN_DIR}"
