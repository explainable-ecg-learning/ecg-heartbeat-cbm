#!/usr/bin/env bash
# experiments/run_pathology.sh
# Re-run downstream 5-class PTB-XL pathology CBM, CEM & Black-Box experiments.
#
# Usage:
#   bash experiments/run_pathology.sh [--gpu 0] [--epochs 25] [--dry-run]

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
GPU="0"
EPOCHS="25"
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
RESULTS_DIR="${REPO_ROOT}/experiments/results/pathology"
mkdir -p "${MODELS_DIR}" "${RESULTS_DIR}"

if [[ -f "${REPO_ROOT}/.venv_new/bin/activate" ]]; then
    # shellcheck disable=SC1091
    source "${REPO_ROOT}/.venv_new/bin/activate"
fi

for mtype in cbm cem blackbox; do
    CMD=(env CUDA_VISIBLE_DEVICES="${GPU}" PYTHONPATH="${REPO_ROOT}/src" python "${REPO_ROOT}/src/pathology.py"
        --model-type "${mtype}" --run-dir "${MODELS_DIR}/pathology_${mtype}" --epochs "${EPOCHS}")
    echo ">>> ${CMD[*]}"
    if [[ "${DRY_RUN}" == false ]]; then
        cd "${REPO_ROOT}"
        "${CMD[@]}"
        [[ -f "${MODELS_DIR}/pathology_${mtype}/test_metrics.json" ]] && \
            cp "${MODELS_DIR}/pathology_${mtype}/test_metrics.json" "${RESULTS_DIR}/pathology_${mtype}_test_metrics.json"
    fi
done

if [[ "${DRY_RUN}" == false ]]; then
    python3 "${REPO_ROOT}/experiments/collect_results.py"
fi
