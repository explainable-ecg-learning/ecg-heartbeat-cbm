#!/usr/bin/env bash
# experiments/run_window_softseg.sh
# Re-run Zenodo 7610236 (ECGdeli U-Net) dense sliding-window CBM+RandInt & CEM
# training (in parallel across 2 GPUs if --gpu1 is distinct from --gpu0) followed
# by the 4-way cross-delineator benchmark saved to experiments/results/soft_seg_eval_all/.
#
# Usage:
#   bash experiments/run_window_softseg.sh [--gpu0 0] [--gpu1 1] [--epochs 18] [--dry-run]

set -euo pipefail

EXP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${EXP_DIR}/.." && pwd)"
GPU0="0"
GPU1="1"
EPOCHS="18"
STRIDE="200"
BATCH_SIZE="512"
DRY_RUN=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        --gpu0)       GPU0="$2";       shift 2 ;;
        --gpu1)       GPU1="$2";       shift 2 ;;
        --epochs)     EPOCHS="$2";     shift 2 ;;
        --stride)     STRIDE="$2";     shift 2 ;;
        --batch-size) BATCH_SIZE="$2"; shift 2 ;;
        --dry-run)    DRY_RUN=true;    shift ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

MODELS_DIR="${REPO_ROOT}/models"
RESULTS_DIR="${REPO_ROOT}/experiments/results/window_softseg"
mkdir -p "${MODELS_DIR}" "${RESULTS_DIR}"

if [[ -f "${REPO_ROOT}/.venv_new/bin/activate" ]]; then
    # shellcheck disable=SC1091
    source "${REPO_ROOT}/.venv_new/bin/activate"
fi

COMMON_OVERRIDES=(
    --set data.delineator=soft_seg
    --set data.use_sliding_window=true
    --set data.stride="${STRIDE}"
    --set data.batch_size="${BATCH_SIZE}"
    --set model.concept_layer.rand_int_prob=0.25
    --set training.epochs="${EPOCHS}"
)

echo "=== Step 1: Pre-build SoftSeg sliding-window dataset cache ==="
if [[ "${DRY_RUN}" == true ]]; then
    echo ">>> [dry-run] build dataset cache (delineator=soft_seg, stride=${STRIDE})"
else
    cd "${REPO_ROOT}"
    PYTHONPATH=src python -c "
import yaml
from data import load_or_build_datasets
cfg = yaml.safe_load(open('config.yaml'))
cfg['data'].update({'delineator': 'soft_seg', 'use_sliding_window': True, 'stride': ${STRIDE}, 'batch_size': ${BATCH_SIZE}})
load_or_build_datasets(cfg, splits=('train', 'val', 'test'))
"
fi

echo "=== Step 2: Train cbm_softseg (GPU ${GPU0}) & cem_softseg (GPU ${GPU1}) ==="
CMD_CBM=(bash "${EXP_DIR}/run_train.sh" --gpu "${GPU0}" --run-dir "${MODELS_DIR}/cbm_softseg" --set model.type=cbm "${COMMON_OVERRIDES[@]}")
CMD_CEM=(bash "${EXP_DIR}/run_train.sh" --gpu "${GPU1}" --run-dir "${MODELS_DIR}/cem_softseg" --set model.type=cem "${COMMON_OVERRIDES[@]}")

if [[ "${DRY_RUN}" == true ]]; then
    echo ">>> ${CMD_CBM[*]}"
    echo ">>> ${CMD_CEM[*]}"
elif [[ "${GPU0}" != "${GPU1}" ]]; then
    "${CMD_CBM[@]}" > "${RESULTS_DIR}/cbm_softseg.log" 2>&1 &
    PID0=$!
    "${CMD_CEM[@]}" > "${RESULTS_DIR}/cem_softseg.log" 2>&1 &
    PID1=$!
    wait "${PID0}" "${PID1}"
else
    "${CMD_CBM[@]}"
    "${CMD_CEM[@]}"
fi

echo "=== Step 3: Cross-delineator benchmark (NeuroKit2 vs Zenodo/ECGdeli) ==="
SOFT_SEG_ZIP=$(python3 -c "
import yaml; from pathlib import Path
cfg = yaml.safe_load(open('${REPO_ROOT}/config.yaml'))
print(Path(cfg['data']['data_dir']) / cfg['data']['soft_seg_path'])
")
BENCH_RUNS=()
for name in cbm_randint_v2 cem_v2 cbm_softseg cem_softseg; do
    if [[ -f "${MODELS_DIR}/${name}/checkpoints/best_model.pt" ]] || [[ "${DRY_RUN}" == true ]]; then
        BENCH_RUNS+=("${MODELS_DIR}/${name}")
    fi
done

BENCH_CMD=(env CUDA_VISIBLE_DEVICES="${GPU0}" PYTHONPATH="${REPO_ROOT}/src" python "${REPO_ROOT}/src/soft_seg.py"
    --zip "${SOFT_SEG_ZIP}" --runs "${BENCH_RUNS[@]}" --n-records 200 --stride 25
    --out-dir "${REPO_ROOT}/experiments/results/soft_seg_eval_all")
echo ">>> ${BENCH_CMD[*]}"
if [[ "${DRY_RUN}" == false ]]; then
    cd "${REPO_ROOT}"
    "${BENCH_CMD[@]}"
    for name in cbm_softseg cem_softseg; do
        for f in eval_test_metrics.json intervention_test.json; do
            [[ -f "${MODELS_DIR}/${name}/${f}" ]] && cp "${MODELS_DIR}/${name}/${f}" "${RESULTS_DIR}/${name}_${f}"
        done
    done
    python3 "${REPO_ROOT}/experiments/collect_results.py"
fi
