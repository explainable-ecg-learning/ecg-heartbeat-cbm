# Experiments

Reproducible experiment runners for all model families in this repository.
Trained checkpoints are saved under `models/`, and evaluation summaries/artifacts are collected under `experiments/results/`.

## Runner Scripts

| Script | What It Runs | Saved Models (`models/`) | Saved Results (`experiments/results/`) |
|---|---|---|---|
| `run_window_neurokit.sh` | NeuroKit2 beat-centred CBM, CBM+RandInt, CEM | `cbm_v2`, `cbm_randint_v2`, `cem_v2` | `window_neurokit/` |
| `run_window_softseg.sh` | Zenodo 7610236 dense sliding-window CBM & CEM + 4-way cross-delineator benchmark | `cbm_softseg`, `cem_softseg` | `window_softseg/`, `soft_seg_eval_all/` |
| `run_dense_softseg.sh` | Per-timestamp 1D U-Net `DenseSoftSegCBM` & `DenseSoftSegCEM` (100 Hz, `BF16`+`TF32`) | `dense_cbm_softseg`, `dense_cem_softseg` | `dense_softseg/` |
| `run_dense_paper_suite.sh` | Full conference suite: 3-seed (`42, 43, 44`) `DenseBlackBox`/`DenseMultiTask`/`DenseCBM`/`DenseCEM` + ablations (`fiducial24`, RandInt, heteroscedastic) + leakage fixes | `dense_*_s{42,43,44}`, `cbm_softseg_masked`, `blackbox_softseg`, `pathology_cbm_detached` | `dense_softseg/`, `paper_suite_logs/`, `summary.{json,md}` |
| `run_pathology.sh` | Record-level 5-class PTB-XL pathology CBM, CEM & Black-Box | `pathology_cbm`, `pathology_cem`, `pathology_blackbox` | `pathology/` |
| `run_ablation.sh` | Architecture (CBM, CEM, Black-Box, Sequential, Independent) & hyperparameter ablations | `ablation/<variant>` | `ablation_log.txt` |
| `run_train.sh` | Single-run helper: train → evaluate (interventions, uncertainty, faithfulness) → visualise | `models/<run>` | — |
| `run_eval.sh` | Standalone evaluation wrapper for any trained window checkpoint | — | — |
| `run_all.sh` | Master orchestrator (`--only <suite>` or all) + result aggregation | All above | `summary.json`, `summary.md` |
| `eval_external_georgia.py` | Zero-shot OOD evaluation of PTB-XL Dense CBM/CEM/baselines & Window CBM/CEM on Georgia 12-Lead ECG (`8,770` clean records) | Evaluates `dense_*_s42` & window `*_softseg*` | `external_georgia_eval.json` |
| `plot_external_georgia.py` | Multi-panel zero-shot Georgia segmentation & clinical interval comparison figure (`assets/external_georgia_segmentation.png`) | — | — |
| `collect_results.py` | Aggregates all existing metrics from `models/` into `results/summary.{json,md}` | — | `summary.json`, `summary.md` |

## Usage

```bash
# Preview all commands without running
bash experiments/run_all.sh --dry-run

# Re-run a single suite (dual-GPU parallel where supported)
bash experiments/run_all.sh --only window_softseg --gpu0 0 --gpu1 1
bash experiments/run_all.sh --only dense --gpu0 0 --gpu1 1
bash experiments/run_all.sh --only paper --gpu0 0 --gpu1 1
bash experiments/run_all.sh --only pathology --gpu0 0
bash experiments/run_all.sh --only neurokit --gpu0 0
bash experiments/run_all.sh --only ablation --gpu0 0

# Re-aggregate metrics from existing checkpoints in models/
python3 experiments/collect_results.py
```
