#!/usr/bin/env bash
# run_ablation.sh — Hyperparameter ablation study for the ECG heartbeat CBM.
#
# Runs a baseline (lead 0) plus one-at-a-time ablations over key
# hyperparameters.  Each variant is a separate run under runs/ablation/.
#
# Usage:
#   bash run_ablation.sh                     # all variants, default GPU
#   bash run_ablation.sh --gpu 0             # force GPU 0
#   bash run_ablation.sh --only baseline     # run a single named variant
#   bash run_ablation.sh --skip-eval         # skip eval after each run
#   bash run_ablation.sh --dry-run           # print commands, do not execute
#
# Variants (one-at-a-time, baseline = current config.yaml with lead 0):
#   Baseline
#   lr         : 1e-4 | 3e-4 | [1e-3] | 3e-3
#   dropout    : 0.0  | 0.1  | [0.2]  | 0.4
#   lambda_concept_start : 0.3 | [1.0] | 2.0
#   lambda_consist       : 0.0 | [0.1] | 0.5
#
# Values in [brackets] are the baseline; they appear in the baseline run only.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG="${SCRIPT_DIR}/config.yaml"
GPU=""
ONLY=""
SKIP_EVAL=false
DRY_RUN=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        --gpu)       GPU="$2";   shift 2 ;;
        --only)      ONLY="$2";  shift 2 ;;
        --skip-eval) SKIP_EVAL=true; shift ;;
        --dry-run)   DRY_RUN=true;   shift ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

[[ -n "$GPU" ]] && export CUDA_VISIBLE_DEVICES="$GPU"

ABLATION_ROOT="${SCRIPT_DIR}/runs/ablation"
mkdir -p "${ABLATION_ROOT}"

# ── Variant registry ──────────────────────────────────────────────────────────
# Each entry: "name|override1 override2 ..."  (space-separated --set flags)
# An empty override string means "use config.yaml as-is" (baseline).
VARIANTS=(
    "baseline|"

    # Learning rate
    "lr_1e-4|training.lr=1e-4"
    "lr_3e-4|training.lr=3e-4"
    "lr_3e-3|training.lr=3e-3"

    # Encoder dropout
    "dropout_0.0|model.encoder.dropout=0.0"
    "dropout_0.1|model.encoder.dropout=0.1"
    "dropout_0.4|model.encoder.dropout=0.4"

    # Concept loss start weight
    "lambda_concept_0.3|training.lambda_concept_start=0.3"
    "lambda_concept_2.0|training.lambda_concept_start=2.0"

    # Consistency loss weight
    "lambda_consist_0.0|training.lambda_consist=0.0"
    "lambda_consist_0.5|training.lambda_consist=0.5"
)

# ── Helper ────────────────────────────────────────────────────────────────────
run_variant() {
    local name="$1"
    local overrides_str="$2"   # space-separated list of KEY=VALUE strings

    local run_dir="${ABLATION_ROOT}/${name}"

    echo ""
    echo "════════════════════════════════════════════════════════════"
    echo " Variant : ${name}"
    echo " Run dir : ${run_dir}"
    [[ -n "${overrides_str}" ]] && echo " Overrides: ${overrides_str}" || echo " Overrides: <none> (baseline)"
    echo "════════════════════════════════════════════════════════════"

    # Build --set flags
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
        bash "${SCRIPT_DIR}/run_train.sh"
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

# ── Main loop ─────────────────────────────────────────────────────────────────
echo "Ablation study — $(date)"
echo "Base config : ${CONFIG}"
echo "GPU         : ${CUDA_VISIBLE_DEVICES:-<default>}"
echo "Output root : ${ABLATION_ROOT}"

# Activate venv if present
if [[ -f "${SCRIPT_DIR}/.venv_new/bin/activate" ]]; then
    source "${SCRIPT_DIR}/.venv_new/bin/activate"
    echo "Activated venv: ${SCRIPT_DIR}/.venv_new"
fi

for entry in "${VARIANTS[@]}"; do
    name="${entry%%|*}"
    overrides="${entry#*|}"

    # --only filter
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
