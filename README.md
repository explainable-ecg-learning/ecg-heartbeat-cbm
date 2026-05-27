# ECG Heartbeat CBM

A **Concept Bottleneck Model (CBM)** for heartbeat completeness classification on 12-lead ECG signals from the [PTB-XL dataset](https://physionet.org/content/ptb-xl/). The model encodes each beat subsequence into interpretable ECG concepts (P, Q, R, S, T, U waves and intervals PR, QRS, ST, QT), then predicts whether a heartbeat is complete or incomplete — with per-concept uncertainty estimates.

---

## Architecture

```
Input: (B, 12, T)  — 12-lead ECG beat window
         │
    TemporalEncoder   (1D ResNet: stem + 4 residual blocks → GAP → 512-d)
         │
    ECGConceptLayer   (10 heads, each → μ_k, log σ²_k)
         │
    UncertaintyClassifier  ([concept_probs, concept_vars] → task_prob)
```

**Concepts** (indices 0–9): `P`, `Q`, `R`, `S`, `T`, `U`, `PR`, `QRS`, `ST`, `QT`

**MC-Dropout** wraps the full model at inference time to produce epistemic uncertainty alongside aleatoric uncertainty from the heteroscedastic concept heads.

---

## Setup

**Requirements:** Python ≥ 3.9, CUDA 12.x (recommended)

```bash
python3 -m venv .venv_new
source .venv_new/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

**Data:** PTB-XL must be available at the path configured in `config.yaml`:

```yaml
data:
  data_dir: "/path/to/ecg-cbm/data"   # base directory — only this needs changing
  data_path: "ptb-xl"                  # relative to data_dir (or absolute)
  concept_file: "concepts.parquet"     # relative to data_dir (or absolute)
```

---

## Training

```bash
# Default config
bash run_train.sh

# Explicit config + output directory
bash run_train.sh --config config.yaml --run-dir runs/my_run

# Force a specific GPU
bash run_train.sh --gpu 0
```

Or directly:

```bash
python src/train.py --config config.yaml --run-dir runs/my_run
```

Config values can be overridden at the command line without editing the YAML:

```bash
python src/train.py --set training.lr=3e-4 --set model.encoder.dropout=0.1
```

Training produces:
- `<run-dir>/checkpoints/best_model.pt` — best checkpoint by validation loss
- `<run-dir>/checkpoints/latest.pt` — most recent checkpoint
- `<run-dir>/test_metrics.json` — final test set metrics

---

## Evaluation

```bash
bash run_eval.sh --run-dir runs/my_run
bash run_eval.sh --run-dir runs/my_run --split val
bash run_eval.sh --run-dir runs/my_run --checkpoint best_model.pt --gpu 0
```

Outputs `<run-dir>/eval_<split>_metrics.json` with task-level (accuracy, F1, AUC, precision, recall) and per-concept metrics.

---

## Ablation Study

```bash
bash run_ablation.sh                     # all variants, default GPU
bash run_ablation.sh --gpu 0             # force GPU 0
bash run_ablation.sh --only baseline     # run a single named variant
bash run_ablation.sh --skip-eval         # skip eval after each run
bash run_ablation.sh --dry-run           # print commands, do not execute
```

Runs one-at-a-time ablations over `lr`, encoder `dropout`, `lambda_concept_start`, and `lambda_consist`. Results are written to `runs/ablation/<variant>/`.

---

## Visualisation

```bash
python src/visualize.py --run-dir runs/my_run
python src/visualize.py --run-dir runs/my_run --split test --n-samples 8
python src/visualize.py --run-dir runs/my_run --no-uncertainty
```

Saves figures to `<run-dir>/figures/`:

| File | Description |
|---|---|
| `training_curves.png` | Train/val loss curves |
| `metrics_bar.png` | Task + per-concept F1 bar chart |
| `concept_heatmap.png` | Predicted concept probabilities on test samples |
| `roc_curve.png` | Task-level ROC curve |
| `concept_confusion.png` | Per-concept confusion matrices |
| `uncertainty_scatter.png` | Task uncertainty vs. confidence scatter |
| `signal_examples.png` | ECG traces with concept annotations |

---

## Configuration

All hyperparameters are in `config.yaml`:

| Section | Key parameters |
|---|---|
| `data` | `data_dir`, `data_path`, `concept_file`, `sampling_rate`, `t_fixed`, `lead`, `beat_window_pre/post`, `val_fold`, `test_fold`, `neg_oversample_factor`, `use_sliding_window` |
| `model` | `encoder.in_channels`, `encoder.base_channels`, `encoder.dropout`, `concept_layer.num_concepts`, `mc_dropout_rate` |
| `training` | `epochs`, `lr`, `weight_decay`, `min_lr`, `lambda_concept_start/end`, `lambda_concept_anneal_epochs`, `lambda_consist`, `reliable_concepts`, `label_smooth_reliable`, `label_smooth_noisy`, `early_stopping_patience` |
| `augmentation` | `temporal_jitter`, `amplitude_scale_min/max`, `noise_std`, `lead_dropout_prob`, `lead_dropout_max` |

---

## Loss

$$\mathcal{L} = \mathcal{L}_\text{task} + \lambda_\text{concept}(t)\,\mathcal{L}_\text{concept} + \lambda_\text{consist}\,\mathcal{L}_\text{consistency}$$

- **$\mathcal{L}_\text{task}$**: BCE on complete/incomplete heartbeat prediction  
- **$\mathcal{L}_\text{concept}$**: Heteroscedastic BCE with MC sampling through concept uncertainty ($\mu_k \pm \sigma_k$); concepts with failed delineation are masked out  
- **$\mathcal{L}_\text{consistency}$**: Soft logical constraints (e.g. PR interval present → P wave and R wave both present); $\lambda_\text{concept}$ is annealed from 1.0 → 0.3 over the first 50 epochs

---

## Project Structure

```
config.yaml       — Unified training/data/model configuration
requirements.txt  — Python dependencies
run_train.sh      — Train, then evaluate and visualise
run_eval.sh       — Evaluate a saved checkpoint
run_ablation.sh   — One-at-a-time hyperparameter ablation study
src/
  data.py         — PTBXLDataset, HeartbeatDataset, signal utilities
  models.py       — TemporalEncoder, ECGConceptLayer, HeartbeatCBM, CBMWithUncertainty
  losses.py       — CombinedCBMLoss (task + heteroscedastic concept + consistency)
  train.py        — Training loop, early stopping, checkpointing, CLI
  visualize.py    — Post-hoc figures and concept heatmaps
explorer/
  app.py          — Interactive concept explorer (web app)
  run.sh          — Launcher for the explorer
```