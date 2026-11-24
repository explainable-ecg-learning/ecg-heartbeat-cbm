#!/usr/bin/env bash
# experiments/run_dense_paper_suite.sh
# Conference paper experiment suite (Checklist Items #1-#4):
#   1. Leakage & GT-intervention fixes:
#        - cbm_softseg_masked (U-wave masked, blocks >30% residual bypass weight)
#        - blackbox_softseg (matched sliding-window SoftSeg Black-Box baseline)
#        - pathology_cbm_detached (stop-grad + lambda_concept=2.0 fixes R² & GT intervention)
#   2. Multi-seed Dense 1D U-Net benchmark (3 seeds: 42, 43, 44) across 5 model families:
#        - dense_blackbox       (matched TemporalUNet1D, pathology only)
#        - dense_multitask      (matched TemporalUNet1D, seg + direct pathology, no bottleneck)
#        - dense_cbm_joint      (soft-mask bottleneck, joint gradients)
#        - dense_cbm_detached   (soft-mask bottleneck, stop-grad → zero soft-mask leakage)
#        - dense_cem            (per-timestamp concept embeddings + temporal RandInt)
#   3. DenseSoftSeg Ablations (seed 42):
#        - dense_cbm_norandint  (rand_int_prob=0.0)
#        - dense_cem_norandint  (rand_int_prob=0.0)
#        - dense_cem_nohetero   (--no-heteroscedastic → boundary uncertainty ablation)
#        - dense_cem_fiducial24 (--concept-mode fiducial24 → full 24-class fiducials)
#        - dense_cbm_fiducial24 (--concept-mode fiducial24 --detach-concepts)
#
# Runs pairs in parallel across GPU0 and GPU1 when --gpu0 != --gpu1.
#
# Usage:
#   bash experiments/run_dense_paper_suite.sh [--gpu0 0] [--gpu1 1] [--epochs 25] [--only fixes|multiseed|ablations|all] [--dry-run]

set -euo pipefail

EXP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${EXP_DIR}/.." && pwd)"
GPU0="0"
GPU1="1"
EPOCHS="25"
BATCH_SIZE="128"
ONLY="all"
DRY_RUN=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        --gpu0)       GPU0="$2";       shift 2 ;;
        --gpu1)       GPU1="$2";       shift 2 ;;
        --epochs)     EPOCHS="$2";     shift 2 ;;
        --batch-size) BATCH_SIZE="$2"; shift 2 ;;
        --only)       ONLY="$2";       shift 2 ;;
        --dry-run)    DRY_RUN=true;    shift ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

MODELS_DIR="${REPO_ROOT}/models"
LOG_DIR="${REPO_ROOT}/experiments/results/paper_suite_logs"
mkdir -p "${MODELS_DIR}" "${LOG_DIR}"

if [[ -f "${REPO_ROOT}/.venv_new/bin/activate" ]]; then
    # shellcheck disable=SC1091
    source "${REPO_ROOT}/.venv_new/bin/activate"
fi

should_run() { [[ "${ONLY}" == "all" || "${ONLY}" == "$1" ]]; }

run_pair() {
    local name0="$1" cmd0="$2" name1="${3:-}" cmd1="${4:-}"
    echo ">>> [GPU ${GPU0}] ${name0}: ${cmd0}"
    [[ -n "${name1}" ]] && echo ">>> [GPU ${GPU1}] ${name1}: ${cmd1}"
    if [[ "${DRY_RUN}" == true ]]; then
        return
    fi
    cd "${REPO_ROOT}"
    if [[ -n "${name1}" && "${GPU0}" != "${GPU1}" ]]; then
        eval "CUDA_VISIBLE_DEVICES=${GPU0} ${cmd0}" > "${LOG_DIR}/${name0}.log" 2>&1 &
        local pid0=$!
        eval "CUDA_VISIBLE_DEVICES=${GPU1} ${cmd1}" > "${LOG_DIR}/${name1}.log" 2>&1 &
        local pid1=$!
        wait "${pid0}" "${pid1}"
    else
        eval "CUDA_VISIBLE_DEVICES=${GPU0} ${cmd0}" > "${LOG_DIR}/${name0}.log" 2>&1
        if [[ -n "${name1}" ]]; then
            eval "CUDA_VISIBLE_DEVICES=${GPU1} ${cmd1}" > "${LOG_DIR}/${name1}.log" 2>&1
        fi
    fi
    echo "    ✓ finished: ${name0}${name1:+ & ${name1}}"
}

# ── 1. Multi-Seed Dense 1D U-Net Main Table (Seeds 42, 43, 44) ───────────────
if should_run "multiseed"; then
    echo "=== [1/3] Multi-Seed Dense 1D U-Net Suite (5 families × 3 seeds) ==="
    for s in 42 43 44; do
        run_pair \
            "dense_blackbox_s${s}" \
            "PYTHONPATH=src python src/dense_seg.py --model-type dense_blackbox --concept-mode wave6 --run-dir ${MODELS_DIR}/dense_blackbox_s${s} --epochs ${EPOCHS} --batch-size ${BATCH_SIZE} --seed ${s}" \
            "dense_multitask_s${s}" \
            "PYTHONPATH=src python src/dense_seg.py --model-type dense_multitask --concept-mode wave6 --run-dir ${MODELS_DIR}/dense_multitask_s${s} --epochs ${EPOCHS} --batch-size ${BATCH_SIZE} --seed ${s}"

        run_pair \
            "dense_cbm_joint_s${s}" \
            "PYTHONPATH=src python src/dense_seg.py --model-type dense_cbm --concept-mode wave6 --run-dir ${MODELS_DIR}/dense_cbm_joint_s${s} --epochs ${EPOCHS} --batch-size ${BATCH_SIZE} --rand-int-prob 0.25 --seed ${s}" \
            "dense_cbm_detached_s${s}" \
            "PYTHONPATH=src python src/dense_seg.py --model-type dense_cbm --concept-mode wave6 --detach-concepts --run-dir ${MODELS_DIR}/dense_cbm_detached_s${s} --epochs ${EPOCHS} --batch-size ${BATCH_SIZE} --rand-int-prob 0.25 --seed ${s}"

        run_pair \
            "dense_cem_s${s}" \
            "PYTHONPATH=src python src/dense_seg.py --model-type dense_cem --concept-mode wave6 --run-dir ${MODELS_DIR}/dense_cem_s${s} --epochs ${EPOCHS} --batch-size ${BATCH_SIZE} --rand-int-prob 0.25 --seed ${s}"
    done
fi

# ── 2. DenseSoftSeg Ablations (Seed 42) ──────────────────────────────────────
if should_run "ablations"; then
    echo "=== [2/3] DenseSoftSeg Ablations (RandInt, Heteroscedastic, fiducial24) ==="
    run_pair \
        "dense_cbm_norandint" \
        "PYTHONPATH=src python src/dense_seg.py --model-type dense_cbm --concept-mode wave6 --run-dir ${MODELS_DIR}/dense_cbm_norandint --epochs ${EPOCHS} --batch-size ${BATCH_SIZE} --rand-int-prob 0.0 --seed 42" \
        "dense_cem_norandint" \
        "PYTHONPATH=src python src/dense_seg.py --model-type dense_cem --concept-mode wave6 --run-dir ${MODELS_DIR}/dense_cem_norandint --epochs ${EPOCHS} --batch-size ${BATCH_SIZE} --rand-int-prob 0.0 --seed 42"

    run_pair \
        "dense_cem_nohetero" \
        "PYTHONPATH=src python src/dense_seg.py --model-type dense_cem --concept-mode wave6 --no-heteroscedastic --run-dir ${MODELS_DIR}/dense_cem_nohetero --epochs ${EPOCHS} --batch-size ${BATCH_SIZE} --rand-int-prob 0.25 --seed 42" \
        "dense_cbm_detached_norandint" \
        "PYTHONPATH=src python src/dense_seg.py --model-type dense_cbm --concept-mode wave6 --detach-concepts --run-dir ${MODELS_DIR}/dense_cbm_detached_norandint --epochs ${EPOCHS} --batch-size ${BATCH_SIZE} --rand-int-prob 0.0 --seed 42"

    # Pre-build fiducial24 cache once before running both GPUs in parallel
    if [[ "${DRY_RUN}" == false ]]; then
        cd "${REPO_ROOT}"
        PYTHONPATH=src python -c "
import yaml; from pathlib import Path
from data import resolve_data_paths
from dense_seg import load_or_build_dense_datasets
cfg = yaml.safe_load(open('config.yaml'))
dp, _ = resolve_data_paths(cfg)
ss = Path(cfg['data']['data_dir']) / cfg['data']['soft_seg_path']
load_or_build_dense_datasets(dp, ss, Path(cfg['training'].get('checkpoint_dir', 'checkpoints')) / 'dense_softseg_cache', concept_mode='fiducial24')
"
    fi

    run_pair \
        "dense_cem_fiducial24" \
        "PYTHONPATH=src python src/dense_seg.py --model-type dense_cem --concept-mode fiducial24 --run-dir ${MODELS_DIR}/dense_cem_fiducial24 --epochs ${EPOCHS} --batch-size ${BATCH_SIZE} --rand-int-prob 0.25 --seed 42" \
        "dense_cbm_fiducial24" \
        "PYTHONPATH=src python src/dense_seg.py --model-type dense_cbm --concept-mode fiducial24 --detach-concepts --run-dir ${MODELS_DIR}/dense_cbm_fiducial24 --epochs ${EPOCHS} --batch-size ${BATCH_SIZE} --rand-int-prob 0.25 --seed 42"
fi

# ── 3. Leakage & Pathology Fixes (U-wave masked SoftSeg CBM + Detached PathologyCBM) ──
if should_run "fixes"; then
    echo "=== [3/3] U-Wave Masking Fix, SoftSeg Black-Box & Detached PathologyCBM ==="
    run_pair \
        "cbm_softseg_masked" \
        "bash ${EXP_DIR}/run_train.sh --run-dir ${MODELS_DIR}/cbm_softseg_masked --set model.type=cbm --set data.delineator=soft_seg --set data.use_sliding_window=true --set data.stride=200 --set data.batch_size=512 --set model.concept_layer.rand_int_prob=0.25 --set model.concept_layer.mask_unsupervised_concepts=true --set training.epochs=18 --set training.seed=42" \
        "blackbox_softseg" \
        "bash ${EXP_DIR}/run_train.sh --run-dir ${MODELS_DIR}/blackbox_softseg --set model.type=blackbox --set data.delineator=soft_seg --set data.use_sliding_window=true --set data.stride=200 --set data.batch_size=512 --set training.epochs=18 --set training.seed=42"

    run_pair \
        "pathology_cbm_detached" \
        "PYTHONPATH=src python src/pathology.py --model-type cbm --detach-concepts --lambda-concept 2.0 --rand-int-prob 0.5 --run-dir ${MODELS_DIR}/pathology_cbm_detached --epochs 25 --seed 42" \
        "pathology_cem_s42" \
        "PYTHONPATH=src python src/pathology.py --model-type cem --run-dir ${MODELS_DIR}/pathology_cem_s42 --epochs 25 --seed 42"
fi

if [[ "${DRY_RUN}" == false ]]; then
    python3 "${EXP_DIR}/collect_results.py"
fi
