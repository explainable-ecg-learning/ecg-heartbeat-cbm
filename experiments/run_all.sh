#!/usr/bin/env bash
# experiments/run_all.sh
# Master orchestrator to re-run all (or selected) experiments and aggregate results
# into experiments/results/summary.json and experiments/results/summary.md.
#
# Usage:
#   bash experiments/run_all.sh --dry-run
#   bash experiments/run_all.sh --only window_softseg --gpu0 0 --gpu1 1
#   bash experiments/run_all.sh --only dense --gpu0 0 --gpu1 1
#   bash experiments/run_all.sh --only pathology --gpu0 0
#   bash experiments/run_all.sh --only neurokit --gpu0 0
#   bash experiments/run_all.sh --only ablation --gpu0 0

set -euo pipefail

EXP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${EXP_DIR}/.." && pwd)"
GPU0="0"
GPU1="1"
ONLY="all"
DRY_RUN=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        --gpu0|--gpu) GPU0="$2"; shift 2 ;;
        --gpu1)       GPU1="$2"; shift 2 ;;
        --only)       ONLY="$2"; shift 2 ;;
        --dry-run)    DRY_RUN=true; shift ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

DRY_FLAG=()
[[ "${DRY_RUN}" == true ]] && DRY_FLAG=(--dry-run)

should_run() { [[ "${ONLY}" == "all" || "${ONLY}" == "$1" ]]; }

if should_run "neurokit"; then
    bash "${EXP_DIR}/run_window_neurokit.sh" --gpu "${GPU0}" "${DRY_FLAG[@]}"
fi

if should_run "window_softseg"; then
    bash "${EXP_DIR}/run_window_softseg.sh" --gpu0 "${GPU0}" --gpu1 "${GPU1}" "${DRY_FLAG[@]}"
fi

if should_run "dense"; then
    bash "${EXP_DIR}/run_dense_softseg.sh" --gpu0 "${GPU0}" --gpu1 "${GPU1}" "${DRY_FLAG[@]}"
fi

if [[ "${ONLY}" == "paper" ]]; then
    bash "${EXP_DIR}/run_dense_paper_suite.sh" --gpu0 "${GPU0}" --gpu1 "${GPU1}" "${DRY_FLAG[@]}"
fi

if should_run "pathology"; then
    bash "${EXP_DIR}/run_pathology.sh" --gpu "${GPU0}" "${DRY_FLAG[@]}"
fi

if [[ "${ONLY}" == "ablation" ]]; then
    bash "${EXP_DIR}/run_ablation.sh" --gpu "${GPU0}" "${DRY_FLAG[@]}"
fi

if [[ "${DRY_RUN}" == false ]]; then
    python3 "${EXP_DIR}/collect_results.py"
fi
