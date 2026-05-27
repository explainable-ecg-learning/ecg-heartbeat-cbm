"""
train.py — Training, evaluation, and CLI entry point for the heartbeat CBM.

Usage:
    python train.py [--config config.yaml] [--run-dir runs/my_run]

The script:
  1. Loads config (YAML)
  2. Builds / loads dataset (HeartbeatDataset from PTB-XL via data.py)
  3. Builds model (CBMWithUncertainty via models.py)
  4. Trains with CombinedCBMLoss (losses.py)
  5. Evaluates on val set each epoch; early stops + checkpoints best model
  6. Final evaluation on test set; saves metrics to run_dir/test_metrics.json
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
import yaml
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from data import (
    PTBXL_ROOT, HeartbeatDataset, PTBXLDataset, augment, make_dataloaders
)
from losses import CombinedCBMLoss
from models import CBMWithUncertainty, CONCEPT_NAMES, build_model

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Callbacks
# ─────────────────────────────────────────────────────────────────────────────

class EarlyStopping:
    def __init__(self, patience: int = 15, min_delta: float = 1e-4) -> None:
        self.patience   = patience
        self.min_delta  = min_delta
        self.best       = float("inf")
        self.counter    = 0
        self.best_epoch = 0

    def __call__(self, val_loss: float, epoch: int) -> bool:
        if val_loss < self.best - self.min_delta:
            self.best, self.counter, self.best_epoch = val_loss, 0, epoch
        else:
            self.counter += 1
        return self.counter >= self.patience


class Checkpointer:
    def __init__(self, checkpoint_dir: Path) -> None:
        self.dir      = Path(checkpoint_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.best_val = float("inf")

    def save(self, model, optimizer, epoch: int, val_loss: float, metadata: dict = None) -> None:
        state = {"epoch": epoch, "val_loss": val_loss,
                 "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                 "metadata": metadata or {}}
        torch.save(state, self.dir / "latest.pt")
        if val_loss < self.best_val:
            self.best_val = val_loss
            torch.save(state, self.dir / "best_model.pt")
            logger.info(f"  ✓ New best checkpoint (val_loss={val_loss:.4f})")

    @staticmethod
    def load(path: Path, model, optimizer=None, device: str = "cpu") -> dict:
        state = torch.load(path, map_location=device, weights_only=False)
        model.load_state_dict(state["model"])
        if optimizer is not None and "optimizer" in state:
            optimizer.load_state_dict(state["optimizer"])
        return state


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate(
    model: CBMWithUncertainty,
    loader: DataLoader,
    device: str,
    concept_names: Optional[list] = None,
) -> Dict[str, float]:
    """Compute task + per-concept metrics on a loader. Returns flat metric dict."""
    model.eval()
    concept_names = concept_names or CONCEPT_NAMES

    task_probs_all, task_labels_all = [], []
    c_probs_all, c_labels_all = [], []

    for batch in tqdm(loader, desc="Eval", leave=False):
        x      = batch["signal"].to(device)
        y_task = batch["task_label"]
        y_conc = batch["concept_labels"]
        task_prob, c_probs, _ = model(x)
        task_probs_all.append(task_prob.squeeze(-1).cpu())
        task_labels_all.append(y_task.cpu())
        c_probs_all.append(c_probs.cpu())
        c_labels_all.append(y_conc.cpu())

    task_probs  = torch.cat(task_probs_all).numpy()
    task_labels = torch.cat(task_labels_all).numpy()
    c_probs_mat = torch.cat(c_probs_all).numpy()
    c_labels_mat = torch.cat(c_labels_all).numpy()

    metrics: Dict[str, float] = {}
    task_preds = (task_probs > 0.5).astype(int)
    metrics["task/accuracy"]  = accuracy_score(task_labels, task_preds)
    metrics["task/f1"]        = f1_score(task_labels, task_preds, zero_division=0)
    metrics["task/precision"] = precision_score(task_labels, task_preds, zero_division=0)
    metrics["task/recall"]    = recall_score(task_labels, task_preds, zero_division=0)
    try:
        metrics["task/auroc"] = roc_auc_score(task_labels, task_probs)
    except ValueError:
        metrics["task/auroc"] = float("nan")

    K = c_probs_mat.shape[1]
    for k in range(K):
        name = concept_names[k] if k < len(concept_names) else f"C{k}"
        preds_k = (c_probs_mat[:, k] > 0.5).astype(int)
        if k >= c_labels_mat.shape[1]:
            continue
        labels_k = c_labels_mat[:, k].astype(int)
        metrics[f"concept/{name}/f1"]      = f1_score(labels_k, preds_k, zero_division=0)
        metrics[f"concept/{name}/accuracy"] = accuracy_score(labels_k, preds_k)

    return metrics


def format_metrics(metrics: Dict[str, float]) -> str:
    return "\n".join(f"  {k}: {v:.4f}" for k, v in sorted(metrics.items()))


# ─────────────────────────────────────────────────────────────────────────────
# Augmentation helper
# ─────────────────────────────────────────────────────────────────────────────

def make_augment_fn(cfg: dict, rng: np.random.Generator):
    aug = cfg.get("augmentation", {})
    t   = cfg.get("data", {}).get("t_fixed", 500)
    return lambda sig: augment(
        sig, t_fixed=t,
        temporal_jitter=aug.get("temporal_jitter", 10),
        amp_min=aug.get("amplitude_scale_min", 0.8),
        amp_max=aug.get("amplitude_scale_max", 1.2),
        noise_std=aug.get("noise_std", 0.01),
        lead_dropout_prob=aug.get("lead_dropout_prob", 0.1),
        lead_dropout_max=aug.get("lead_dropout_max", 2),
        rng=rng,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Main training function
# ─────────────────────────────────────────────────────────────────────────────

def train(cfg: dict, run_dir: Optional[Path] = None) -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Device: {device}")

    data_cfg  = cfg.get("data", {})
    train_cfg = cfg.get("training", {})

    # Run directory
    if run_dir is None:
        run_dir = Path(train_cfg.get("log_dir", "runs")) / time.strftime("%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.yaml").write_text(yaml.dump(cfg))

    # File logging
    fh = logging.FileHandler(run_dir / "train.log", mode="a")
    fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logging.getLogger().addHandler(fh)
    logger.info(f"Logging to {run_dir / 'train.log'}")

    # ── Dataset ──────────────────────────────────────────────────────────────
    cache_dir = Path(train_cfg.get("checkpoint_dir", "checkpoints")) / "dataset_cache"
    rng = np.random.default_rng(42)

    if (cache_dir / "train.pkl").exists():
        logger.info("Loading dataset from cache…")
        train_ds = HeartbeatDataset.from_cache(cache_dir, "train")
        val_ds   = HeartbeatDataset.from_cache(cache_dir, "val")
        test_ds  = HeartbeatDataset.from_cache(cache_dir, "test")
    else:
        logger.info("Building dataset from raw PTB-XL (this will take a while)…")
        _data_dir    = data_cfg.get("data_dir", None)
        _data_path   = data_cfg.get("data_path", str(PTBXL_ROOT))
        if _data_dir and not Path(_data_path).is_absolute():
            _data_path = str(Path(_data_dir) / _data_path)
        train_ds, val_ds, test_ds = HeartbeatDataset.from_records(
            data_path              = _data_path,
            sampling_rate          = data_cfg.get("sampling_rate", 500),
            t_fixed                = data_cfg.get("t_fixed", 500),
            beat_window_pre        = data_cfg.get("beat_window_pre", 100),
            beat_window_post       = data_cfg.get("beat_window_post", 400),
            delineation_cache_dir  = cache_dir / "delineation",
            val_fold               = data_cfg.get("val_fold", 9),
            test_fold              = data_cfg.get("test_fold", 10),
            lead_idx               = data_cfg.get("lead", 1),
            neg_oversample_factor  = float(data_cfg.get("neg_oversample_factor", 1.0)),
            use_sliding_window     = bool(data_cfg.get("use_sliding_window", False)),
            stride                 = int(data_cfg.get("stride", 50)),
        )
        logger.info("Caching dataset splits…")
        train_ds.save_cache(cache_dir, "train")
        val_ds.save_cache(cache_dir, "val")
        test_ds.save_cache(cache_dir, "test")

    # Lead selection
    model_leads = data_cfg.get("model_leads", None)
    model_lead  = data_cfg.get("model_lead", None)
    for ds in (train_ds, val_ds, test_ds):
        if model_leads is not None:
            ds.model_leads = model_leads
        elif model_lead is not None:
            ds.model_lead = model_lead

    # Augmentation and active concepts
    train_ds.augment_fn = make_augment_fn(cfg, rng)
    active_concepts = cfg.get("active_concepts", None)
    if active_concepts is not None:
        for ds in (train_ds, val_ds, test_ds):
            ds.active_concepts = active_concepts

    logger.info(f"Train distribution: {train_ds.label_distribution()}")

    train_loader, val_loader, test_loader = make_dataloaders(
        train_ds, val_ds, test_ds,
        batch_size=data_cfg.get("batch_size", 256),
        num_workers=data_cfg.get("num_workers", 4),
    )

    # ── Model ────────────────────────────────────────────────────────────────
    model = build_model(cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model parameters: {n_params:,}")

    # ── Optimiser & Scheduler ────────────────────────────────────────────────
    optimizer = AdamW(
        model.parameters(),
        lr=float(train_cfg.get("lr", 1e-3)),
        weight_decay=float(train_cfg.get("weight_decay", 1e-4)),
    )
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=train_cfg.get("epochs", 100),
        eta_min=float(train_cfg.get("min_lr", 1e-5)),
    )

    # ── Loss ─────────────────────────────────────────────────────────────────
    criterion = CombinedCBMLoss(
        lambda_concept_start = float(train_cfg.get("lambda_concept_start", 1.0)),
        lambda_concept_end   = float(train_cfg.get("lambda_concept_end",   0.3)),
        lambda_consist       = float(train_cfg.get("lambda_consist",       0.1)),
        anneal_epochs        = int(train_cfg.get("lambda_concept_anneal_epochs", 50)),
        n_mc_samples         = int(train_cfg.get("n_mc_samples", 10)),
        reliable_concepts    = train_cfg.get("reliable_concepts", [2, 4, 7, 9]),
        smooth_reliable      = float(train_cfg.get("label_smooth_reliable", 0.02)),
        smooth_noisy         = float(train_cfg.get("label_smooth_noisy",    0.10)),
        use_heteroscedastic  = train_cfg.get("use_heteroscedastic_loss", True),
    )

    # ── Callbacks ────────────────────────────────────────────────────────────
    ckpt    = Checkpointer(run_dir / "checkpoints")
    stopper = EarlyStopping(patience=train_cfg.get("early_stopping_patience", 15))

    # ── Training loop ────────────────────────────────────────────────────────
    num_epochs = train_cfg.get("epochs", 100)
    concept_names = cfg.get("concept_names", CONCEPT_NAMES)

    for epoch in range(1, num_epochs + 1):
        model.train()
        epoch_losses = {"total": 0.0, "task": 0.0, "concept": 0.0, "consistency": 0.0}
        n_batches = 0

        for batch in tqdm(train_loader, desc=f"Epoch {epoch}/{num_epochs}", leave=False):
            x               = batch["signal"].to(device)
            y_task          = batch["task_label"].to(device)
            y_concepts      = batch["concept_labels"].to(device)
            concept_weights = batch["concept_weights"].to(device)

            task_prob, c_probs, c_log_vars, c_logits = model.forward_with_logits(x)
            losses = criterion(
                task_prob, c_probs, c_log_vars, c_logits,
                y_task, y_concepts, concept_weights, epoch=epoch,
            )

            optimizer.zero_grad()
            losses["total"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            for k in epoch_losses:
                v = losses[k]
                epoch_losses[k] += v.item() if torch.is_tensor(v) else v
            n_batches += 1

        scheduler.step()

        val_metrics = evaluate(model, val_loader, device, concept_names)
        val_loss_proxy = 1.0 - val_metrics.get("task/accuracy", 0.0)

        logger.info(
            f"Epoch {epoch:3d} | "
            f"Loss={epoch_losses['total'] / max(1, n_batches):.4f} | "
            f"ValAcc={val_metrics.get('task/accuracy', 0):.4f} | "
            f"ValAUROC={val_metrics.get('task/auroc', 0):.4f} | "
            f"λ_c={losses['lambda_concept']:.3f}"
        )

        ckpt.save(model, optimizer, epoch, val_loss_proxy, metadata={"val_metrics": val_metrics})

        if stopper(val_loss_proxy, epoch):
            logger.info(f"Early stopping at epoch {epoch} (best={stopper.best_epoch})")
            break

    # ── Final test evaluation ─────────────────────────────────────────────────
    logger.info("Loading best checkpoint for test evaluation…")
    Checkpointer.load(run_dir / "checkpoints" / "best_model.pt", model, device=device)
    test_metrics = evaluate(model, test_loader, device, concept_names)
    logger.info("Test results:\n" + format_metrics(test_metrics))

    with open(run_dir / "test_metrics.json", "w") as f:
        json.dump(test_metrics, f, indent=2)
    logger.info(f"Training complete. Artifacts in: {run_dir}")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def _apply_overrides(cfg: dict, overrides: list[str]) -> None:
    """Apply dot-notation key=value overrides to a nested config dict.

    Examples::

        training.lr=0.0001
        data.lead=0
        model.encoder.dropout=0.3
    """
    import ast

    for item in overrides:
        if "=" not in item:
            raise ValueError(f"Override must be key=value, got: {item!r}")
        key_path, _, raw_value = item.partition("=")
        keys = key_path.strip().split(".")
        # Try to parse as a Python literal (int, float, bool, list, …)
        try:
            value = ast.literal_eval(raw_value)
        except (ValueError, SyntaxError):
            value = raw_value  # fall back to plain string
        # Walk / create nested dicts
        node = cfg
        for k in keys[:-1]:
            node = node.setdefault(k, {})
        node[keys[-1]] = value


def main() -> None:
    parser = argparse.ArgumentParser(description="Train heartbeat CBM")
    parser.add_argument("--config",  default="config.yaml", help="Path to YAML config")
    parser.add_argument("--run-dir", default=None,          help="Output directory for this run")
    parser.add_argument(
        "--set", dest="overrides", action="append", default=[],
        metavar="KEY=VALUE",
        help="Override config values via dot notation, e.g. --set training.lr=0.0001",
    )
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    if args.overrides:
        _apply_overrides(cfg, args.overrides)

    run_dir = Path(args.run_dir) if args.run_dir else None
    train(cfg, run_dir)


if __name__ == "__main__":
    main()
