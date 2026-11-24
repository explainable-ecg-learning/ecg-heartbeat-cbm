#!/usr/bin/env bash
# experiments/run_window_neurokit.sh
# Re-run NeuroKit2 beat-centred CBM, CBM+RandInt, and CEM experiments.
# Models are saved under models/, and summary metrics are copied to experiments/results/.
#
# Usage:
#   bash experiments/run_window_neurokit.sh [--gpu 0] [--epochs 100] [--dry-run]

set -euo pipefail

EXP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${EXP_DIR}/.." && pwd)"
GPU="0"
EPOCHS="100"
DRY_RUN=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        --gpu)     GPU="$2";    shift 2 ;;
        --epochs)  EPOCHS="$2"; shift 2 ;;
        --dry-run) DRY_RUN=true; shift ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

MODELS_DIR="${REPO_ROOT}/models"
RESULTS_DIR="${REPO_ROOT}/experiments/results/window_neurokit"
mkdir -p "${MODELS_DIR}" "${RESULTS_DIR}"

run_cmd() {
    echo ">>> $*"
    if [[ "${DRY_RUN}" == false ]]; then
        "$@"
    fi
}

run_cmd bash "${EXP_DIR}/run_train.sh" --gpu "${GPU}" --run-dir "${MODELS_DIR}/cbm_v2" \
    --set model.type=cbm --set model.concept_layer.rand_int_prob=0.0 --set training.epochs="${EPOCHS}"

run_cmd bash "${EXP_DIR}/run_train.sh" --gpu "${GPU}" --run-dir "${MODELS_DIR}/cbm_randint_v2" \
    --set model.type=cbm --set model.concept_layer.rand_int_prob=0.25 --set training.epochs="${EPOCHS}"

run_cmd bash "${EXP_DIR}/run_train.sh" --gpu "${GPU}" --run-dir "${MODELS_DIR}/cem_v2" \
    --set model.type=cem --set model.concept_layer.rand_int_prob=0.25 --set training.epochs="${EPOCHS}"

if [[ "${DRY_RUN}" == false ]]; then
    for name in cbm_v2 cbm_randint_v2 cem_v2; do
        for f in eval_test_metrics.json intervention_test.json; do
            [[ -f "${MODELS_DIR}/${name}/${f}" ]] && cp "${MODELS_DIR}/${name}/${f}" "${RESULTS_DIR}/${name}_${f}"
        done
    done
    python3 "${REPO_ROOT}/experiments/collect_results.py"
fi
