#!/usr/bin/env bash
# experiments/run_ablation.sh — Architecture & hyperparameter ablation study for ECG CBM / CEM.
#
# Usage:
#   bash experiments/run_ablation.sh                     # all variants, default GPU
#   bash experiments/run_ablation.sh --gpu 0             # force GPU 0
#   bash experiments/run_ablation.sh --only cem          # run a single named variant
#   bash experiments/run_ablation.sh --skip-eval         # skip eval+vis after each run
#   bash experiments/run_ablation.sh --dry-run           # print commands, do not execute

set -euo pipefail

EXP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${EXP_DIR}/.." && pwd)"
CONFIG="${REPO_ROOT}/config.yaml"
GPU=""
ONLY=""
SKIP_EVAL=false
DRY_RUN=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        --gpu)       GPU="$2";        shift 2 ;;
        --only)      ONLY="$2";       shift 2 ;;
        --skip-eval) SKIP_EVAL=true;  shift ;;
        --dry-run)   DRY_RUN=true;    shift ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

[[ -n "$GPU" ]] && export CUDA_VISIBLE_DEVICES="$GPU"

ABLATION_ROOT="${REPO_ROOT}/models/ablation"
mkdir -p "${ABLATION_ROOT}"

VARIANTS=(
    # Core architecture & paradigm comparisons
    "baseline|"
    "cbm_no_randint|model.concept_layer.rand_int_prob=0.0"
    "cem|model.type=cem"
    "cem_no_randint|model.type=cem model.concept_layer.rand_int_prob=0.0"
    "blackbox|model.type=blackbox"
    "cbm_sequential|training.mode=sequential"
    "cbm_independent|training.mode=independent"
    "cbm_probs_only|model.classifier.use_uncertainty=false"
    "cbm_plain_bce|training.use_heteroscedastic_loss=false model.classifier.use_uncertainty=false"

    # Learning rate
    "lr_1e-4|training.lr=1e-4"
    "lr_3e-4|training.lr=3e-4"
    "lr_3e-3|training.lr=3e-3"

    # Encoder dropout
    "dropout_0.0|model.encoder.dropout=0.0"
    "dropout_0.1|model.encoder.dropout=0.1"
    "dropout_0.4|model.encoder.dropout=0.4"

    # Loss weights
    "lambda_concept_0.3|training.lambda_concept_start=0.3"
    "lambda_concept_2.0|training.lambda_concept_start=2.0"
    "lambda_consist_0.0|training.lambda_consist=0.0"
    "lambda_consist_0.5|training.lambda_consist=0.5"
)

run_variant() {
    local name="$1"
    local overrides_str="$2"
    local run_dir="${ABLATION_ROOT}/${name}"

    echo ""
    echo "════════════════════════════════════════════════════════════"
    echo " Variant : ${name}"
    echo " Run dir : ${run_dir}"
    [[ -n "${overrides_str}" ]] && echo " Overrides: ${overrides_str}" || echo " Overrides: <none> (baseline)"
    echo "════════════════════════════════════════════════════════════"

    local set_flags=()
    if [[ -n "${overrides_str}" ]]; then
        # shellcheck disable=SC2206
        for kv in ${overrides_str}; do
            set_flags+=("--set" "${kv}")
        done
    fi

    local skip_flag=""
    [[ "${SKIP_EVAL}" == true ]] && skip_flag="--skip-eval"

    local cmd=(
        bash "${EXP_DIR}/run_train.sh"
        --config "${CONFIG}"
        --run-dir "${run_dir}"
        ${skip_flag:+$skip_flag}
        "${set_flags[@]+"${set_flags[@]}"}"
    )

    echo "CMD: ${cmd[*]}"
    if [[ "${DRY_RUN}" == true ]]; then
        echo "[dry-run] Skipping execution."
        return
    fi

    "${cmd[@]}"
}

echo "Ablation study — $(date)"
echo "Base config : ${CONFIG}"
echo "GPU         : ${CUDA_VISIBLE_DEVICES:-<default>}"
echo "Output root : ${ABLATION_ROOT}"

if [[ -f "${REPO_ROOT}/.venv_new/bin/activate" ]]; then
    # shellcheck disable=SC1091
    source "${REPO_ROOT}/.venv_new/bin/activate"
    echo "Activated venv: ${REPO_ROOT}/.venv_new"
fi

for entry in "${VARIANTS[@]}"; do
    name="${entry%%|*}"
    overrides="${entry#*|}"
    if [[ -n "${ONLY}" && "${name}" != "${ONLY}" ]]; then
        continue
    fi
    run_variant "${name}" "${overrides}"
done

echo ""
echo "════════════════════════════════════════════════════════════"
echo " All requested ablation variants finished."
echo " Results under: ${ABLATION_ROOT}"
echo "════════════════════════════════════════════════════════════"
