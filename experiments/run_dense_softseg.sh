#!/usr/bin/env bash
# experiments/run_dense_softseg.sh
# Re-run Per-Timestamp 1D U-Net DenseSoftSegCBM & DenseSoftSegCEM (100 Hz, BF16 + TF32).
# Checkpoints are saved under models/, and test metrics are saved under experiments/results/dense_softseg/.
#
# Usage:
#   bash experiments/run_dense_softseg.sh [--gpu0 0] [--gpu1 1] [--concept-mode wave6|fiducial24] [--epochs 25] [--dry-run]

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
GPU0="0"
GPU1="1"
CONCEPT_MODE="wave6"
EPOCHS="25"
BATCH_SIZE="128"
DRY_RUN=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        --gpu0)         GPU0="$2";         shift 2 ;;
        --gpu1)         GPU1="$2";         shift 2 ;;
        --concept-mode) CONCEPT_MODE="$2"; shift 2 ;;
        --epochs)       EPOCHS="$2";       shift 2 ;;
        --batch-size)   BATCH_SIZE="$2";   shift 2 ;;
        --dry-run)      DRY_RUN=true;      shift ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

MODELS_DIR="${REPO_ROOT}/models"
RESULTS_DIR="${REPO_ROOT}/experiments/results/dense_softseg"
mkdir -p "${MODELS_DIR}" "${RESULTS_DIR}"

if [[ -f "${REPO_ROOT}/.venv_new/bin/activate" ]]; then
    # shellcheck disable=SC1091
    source "${REPO_ROOT}/.venv_new/bin/activate"
fi

SUFFIX=""
[[ "${CONCEPT_MODE}" != "wave6" ]] && SUFFIX="_${CONCEPT_MODE}"

echo "=== Step 1: Pre-build 100 Hz dense soft-seg RAM cache (${CONCEPT_MODE}) ==="
if [[ "${DRY_RUN}" == true ]]; then
    echo ">>> [dry-run] build dense 100 Hz cache (concept_mode=${CONCEPT_MODE})"
else
    cd "${REPO_ROOT}"
    PYTHONPATH=src python -c "
from pathlib import Path; import yaml
from data import resolve_data_paths
from dense_seg import load_or_build_dense_datasets
cfg = yaml.safe_load(open('config.yaml'))
dp, _ = resolve_data_paths(cfg)
ss = Path(cfg['data']['data_dir']) / cfg['data']['soft_seg_path']
load_or_build_dense_datasets(dp, ss, Path('checkpoints/dense_softseg_cache'), concept_mode='${CONCEPT_MODE}')
"
fi

CMD_CBM=(env CUDA_VISIBLE_DEVICES="${GPU0}" PYTHONPATH="${REPO_ROOT}/src" python "${REPO_ROOT}/src/dense_seg.py"
    --model-type dense_cbm --concept-mode "${CONCEPT_MODE}" --run-dir "${MODELS_DIR}/dense_cbm_softseg${SUFFIX}"
    --epochs "${EPOCHS}" --batch-size "${BATCH_SIZE}")
CMD_CEM=(env CUDA_VISIBLE_DEVICES="${GPU1}" PYTHONPATH="${REPO_ROOT}/src" python "${REPO_ROOT}/src/dense_seg.py"
    --model-type dense_cem --concept-mode "${CONCEPT_MODE}" --run-dir "${MODELS_DIR}/dense_cem_softseg${SUFFIX}"
    --epochs "${EPOCHS}" --batch-size "${BATCH_SIZE}")

echo "=== Step 2: Train DenseSoftSegCBM (GPU ${GPU0}) & DenseSoftSegCEM (GPU ${GPU1}) ==="
if [[ "${DRY_RUN}" == true ]]; then
    echo ">>> ${CMD_CBM[*]}"
    echo ">>> ${CMD_CEM[*]}"
elif [[ "${GPU0}" != "${GPU1}" ]]; then
    cd "${REPO_ROOT}"
    "${CMD_CBM[@]}" > "${RESULTS_DIR}/dense_cbm${SUFFIX}.log" 2>&1 &
    PID0=$!
    "${CMD_CEM[@]}" > "${RESULTS_DIR}/dense_cem${SUFFIX}.log" 2>&1 &
    PID1=$!
    wait "${PID0}" "${PID1}"
else
    cd "${REPO_ROOT}"
    "${CMD_CBM[@]}"
    "${CMD_CEM[@]}"
fi

if [[ "${DRY_RUN}" == false ]]; then
    for name in "dense_cbm_softseg${SUFFIX}" "dense_cem_softseg${SUFFIX}"; do
        [[ -f "${MODELS_DIR}/${name}/eval_test_metrics.json" ]] && \
            cp "${MODELS_DIR}/${name}/eval_test_metrics.json" "${RESULTS_DIR}/${name}_eval_test_metrics.json"
    done
    python3 "${REPO_ROOT}/experiments/collect_results.py"
fi
