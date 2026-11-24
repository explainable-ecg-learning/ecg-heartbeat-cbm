"""
train.py — Training, staged CBM/CEM modes, checkpointing, and CLI entry point.

Supports:
  - Architectures (`model.type`): "cbm", "cem" (Concept Embedding Model), "blackbox"
  - Training modes (`training.mode`):
      * "joint"       : Simultaneous concept + task optimisation
      * "sequential"  : Stage 1 trains encoder + concepts; Stage 2 trains classifier on predicted concepts
      * "independent" : Stage 1 trains encoder + concepts; Stage 2 trains classifier with GT interventions
"""

from __future__ import annotations

import argparse
import ast
import json
import logging
import time
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
import yaml
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from data import augment, load_or_build_datasets, make_dataloaders
from evaluate import evaluate, evaluate_interventions, format_metrics
from losses import CombinedCBMLoss
from models import CONCEPT_NAMES, build_model
from utils import set_seed

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

    def reset(self) -> None:
        self.best       = float("inf")
        self.counter    = 0
        self.best_epoch = 0

    def __call__(self, val_metric: float, epoch: int) -> bool:
        if val_metric < self.best - self.min_delta:
            self.best, self.counter, self.best_epoch = val_metric, 0, epoch
        else:
            self.counter += 1
        return self.counter >= self.patience


class Checkpointer:
    def __init__(self, checkpoint_dir: Path) -> None:
        self.dir      = Path(checkpoint_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.best_val = float("inf")

    def reset_best(self) -> None:
        self.best_val = float("inf")

    def save(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        epoch: int,
        val_loss: float,
        metadata: Optional[dict] = None,
        best_name: str = "best_model.pt",
    ) -> None:
        state = {
            "epoch":     epoch,
            "val_loss":  val_loss,
            "model":     model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "metadata":  metadata or {},
        }
        torch.save(state, self.dir / "latest.pt")
        if val_loss < self.best_val:
            self.best_val = val_loss
            torch.save(state, self.dir / best_name)
            logger.info(f"  ✓ New best checkpoint [{best_name}] (monitor={val_loss:.4f})")

    @staticmethod
    def load(path: Path, model: torch.nn.Module, optimizer=None, device: str = "cpu") -> dict:
        state = torch.load(path, map_location=device, weights_only=False)
        model.load_state_dict(state["model"])
        if optimizer is not None and "optimizer" in state:
            optimizer.load_state_dict(state["optimizer"])
        return state


# ─────────────────────────────────────────────────────────────────────────────
# Augmentation helper
# ─────────────────────────────────────────────────────────────────────────────

def make_augment_fn(cfg: dict, rng: np.random.Generator):
    aug = cfg.get("augmentation", {})
    t   = int(cfg.get("data", {}).get("t_fixed", 500))
    return lambda sig: augment(
        sig,
        t_fixed=t,
        temporal_jitter=int(aug.get("temporal_jitter", 15)),
        amp_min=float(aug.get("amplitude_scale_min", 0.8)),
        amp_max=float(aug.get("amplitude_scale_max", 1.2)),
        noise_std=float(aug.get("noise_std", 0.01)),
        lead_dropout_prob=float(aug.get("lead_dropout_prob", 0.1)),
        lead_dropout_max=int(aug.get("lead_dropout_max", 2)),
        rng=rng,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Stage runner
# ─────────────────────────────────────────────────────────────────────────────

def _run_stage(
    *,
    stage_name: str,
    stage_loss_mode: str,   # "joint" | "concept_only" | "task_only"
    independent_gt: bool,
    model: torch.nn.Module,
    train_loader,
    val_loader,
    criterion: CombinedCBMLoss,
    train_cfg: dict,
    concept_names: list,
    ckpt: Checkpointer,
    best_ckpt_name: str,
    num_epochs: int,
    device: str,
) -> None:
    is_blackbox = getattr(model, "model_type", "cbm") == "blackbox"
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = AdamW(
        trainable_params,
        lr=float(train_cfg.get("lr", 1e-3)),
        weight_decay=float(train_cfg.get("weight_decay", 1e-4)),
    )
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=max(1, num_epochs),
        eta_min=float(train_cfg.get("min_lr", 1e-5)),
    )
    stopper = EarlyStopping(patience=int(train_cfg.get("early_stopping_patience", 15)))
    monitor_mode = str(train_cfg.get("monitor", "val_loss")).lower()

    for epoch in range(1, num_epochs + 1):
        model.train()
        if stage_loss_mode == "task_only" and not is_blackbox:
            # Keep frozen encoder BatchNorm statistics fixed during Stage 2
            model.encoder.eval()
            model.concept_layer.eval()

        epoch_losses = {"total": 0.0, "task": 0.0, "concept": 0.0, "consistency": 0.0}
        n_batches = 0
        last_lam_c = 0.0

        for batch in train_loader:
            x               = batch["signal"].to(device)
            y_task          = batch["task_label"].to(device)
            y_concepts      = batch["concept_labels"].to(device)
            concept_weights = batch["concept_weights"].to(device)

            if independent_gt:
                task_prob, c_probs, c_log_vars, c_logits = model.forward_with_logits(
                    x,
                    intervene_labels=y_concepts,
                    intervene_mask=concept_weights,
                )
            else:
                task_prob, c_probs, c_log_vars, c_logits = model.forward_with_logits(
                    x,
                    concept_labels=y_concepts,
                    concept_weights=concept_weights,
                )

            losses = criterion(
                task_prob, c_probs, c_log_vars, c_logits,
                y_task, y_concepts, concept_weights,
                epoch=epoch,
                stage=stage_loss_mode,
                compute_concepts=not is_blackbox,
            )

            optimizer.zero_grad()
            losses["total"].backward()
            torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
            optimizer.step()

            for k in epoch_losses:
                v = losses[k]
                epoch_losses[k] += float(v.item() if torch.is_tensor(v) else v)
            last_lam_c = float(losses["lambda_concept"])
            n_batches += 1

        scheduler.step()

        val_metrics = evaluate(
            model, val_loader, device,
            concept_names=concept_names,
            criterion=criterion,
            epoch=epoch,
            stage=stage_loss_mode,
        )

        if stage_loss_mode == "concept_only":
            monitor_val = val_metrics.get("val/loss_total", float("inf"))
        elif monitor_mode == "val_acc":
            monitor_val = 1.0 - val_metrics.get("task/accuracy", 0.0)
        else:
            monitor_val = val_metrics.get("val/loss_total", 1.0 - val_metrics.get("task/accuracy", 0.0))

        logger.info(
            f"[{stage_name}] Epoch {epoch:3d} | "
            f"Loss={epoch_losses['total'] / max(1, n_batches):.4f} | "
            f"ValLoss={val_metrics.get('val/loss_total', 0.0):.4f} | "
            f"ValAcc={val_metrics.get('task/accuracy', 0.0):.4f} | "
            f"ValAUROC={val_metrics.get('task/auroc', 0.0):.4f} | "
            f"ConceptF1={val_metrics.get('concept/macro_f1', 0.0):.4f} | "
            f"λ_c={last_lam_c:.3f}"
        )

        ckpt.save(
            model, optimizer, epoch, monitor_val,
            metadata={"val_metrics": val_metrics, "stage": stage_name},
            best_name=best_ckpt_name,
        )

        if stopper(monitor_val, epoch):
            logger.info(f"[{stage_name}] Early stopping at epoch {epoch} (best={stopper.best_epoch})")
            break


# ─────────────────────────────────────────────────────────────────────────────
# Main training function
# ─────────────────────────────────────────────────────────────────────────────

def train(cfg: dict, run_dir: Optional[Path] = None) -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Device: {device}")

    data_cfg  = cfg.get("data", {})
    train_cfg = cfg.get("training", {})
    seed      = int(train_cfg.get("seed", 42))
    set_seed(seed)

    if run_dir is None:
        run_dir = Path(train_cfg.get("log_dir", "models")) / time.strftime("%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.yaml").write_text(yaml.dump(cfg, sort_keys=False))

    root_logger = logging.getLogger()
    fh = logging.FileHandler(run_dir / "train.log", mode="a")
    fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    root_logger.addHandler(fh)
    try:
        logger.info(f"Logging to {run_dir / 'train.log'} | seed={seed}")

        # ── Dataset ──────────────────────────────────────────────────────────
        splits = load_or_build_datasets(cfg, splits=("train", "val", "test"))
        train_ds, val_ds, test_ds = splits["train"], splits["val"], splits["test"]
        rng = np.random.default_rng(seed)
        train_ds.augment_fn = make_augment_fn(cfg, rng)

        logger.info(f"Train distribution: {train_ds.label_distribution()}")

        train_loader, val_loader, test_loader = make_dataloaders(
            train_ds, val_ds, test_ds,
            batch_size=int(data_cfg.get("batch_size", 256)),
            num_workers=int(data_cfg.get("num_workers", 4)),
        )

        # ── Model & Loss ─────────────────────────────────────────────────────
        model = build_model(cfg).to(device)
        mtype = getattr(model, "model_type", "cbm")
        n_params = sum(p.numel() for p in model.parameters())
        logger.info(f"Model type: {mtype.upper()} | Parameters: {n_params:,}")

        criterion = CombinedCBMLoss(
            lambda_concept_start = float(train_cfg.get("lambda_concept_start", 1.0)),
            lambda_concept_end   = float(train_cfg.get("lambda_concept_end",   0.3)),
            lambda_consist       = float(train_cfg.get("lambda_consist",       0.1)),
            anneal_epochs        = int(train_cfg.get("lambda_concept_anneal_epochs", 50)),
            n_mc_samples         = int(train_cfg.get("n_mc_samples", 10)),
            reliable_concepts    = train_cfg.get("reliable_concepts", [2, 4, 7, 9]),
            smooth_reliable      = float(train_cfg.get("label_smooth_reliable", 0.02)),
            smooth_noisy         = float(train_cfg.get("label_smooth_noisy",    0.10)),
            use_heteroscedastic  = bool(train_cfg.get("use_heteroscedastic_loss", True)),
            var_reg_weight       = float(train_cfg.get("var_reg_weight", 1e-3)),
        )

        ckpt = Checkpointer(run_dir / "checkpoints")
        num_epochs = int(train_cfg.get("epochs", 100))
        concept_names = cfg.get("concept_names", CONCEPT_NAMES)
        active_concepts = cfg.get("active_concepts", None)
        if active_concepts is not None:
            concept_names = [concept_names[i] for i in active_concepts]

        train_mode = str(train_cfg.get("mode", "joint")).lower()
        if mtype == "blackbox":
            _run_stage(
                stage_name="blackbox",
                stage_loss_mode="task_only",
                independent_gt=False,
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                criterion=criterion,
                train_cfg=train_cfg,
                concept_names=concept_names,
                ckpt=ckpt,
                best_ckpt_name="best_model.pt",
                num_epochs=num_epochs,
                device=device,
            )
        elif train_mode in ("sequential", "independent"):
            stage1_epochs = int(train_cfg.get("stage1_epochs", max(1, num_epochs // 2)))
            stage2_epochs = max(1, num_epochs - stage1_epochs)
            logger.info(f"Running {train_mode} training: Stage 1 ({stage1_epochs} ep) → Stage 2 ({stage2_epochs} ep)")

            _run_stage(
                stage_name="stage1_concepts",
                stage_loss_mode="concept_only",
                independent_gt=False,
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                criterion=criterion,
                train_cfg=train_cfg,
                concept_names=concept_names,
                ckpt=ckpt,
                best_ckpt_name="best_stage1.pt",
                num_epochs=stage1_epochs,
                device=device,
            )
            Checkpointer.load(run_dir / "checkpoints" / "best_stage1.pt", model, device=device)
            model.freeze_concept_bottleneck()
            ckpt.reset_best()

            _run_stage(
                stage_name=f"stage2_{train_mode}",
                stage_loss_mode="task_only",
                independent_gt=(train_mode == "independent"),
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                criterion=criterion,
                train_cfg=train_cfg,
                concept_names=concept_names,
                ckpt=ckpt,
                best_ckpt_name="best_model.pt",
                num_epochs=stage2_epochs,
                device=device,
            )
            model.unfreeze_all()
        else:
            _run_stage(
                stage_name="joint",
                stage_loss_mode="joint",
                independent_gt=False,
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                criterion=criterion,
                train_cfg=train_cfg,
                concept_names=concept_names,
                ckpt=ckpt,
                best_ckpt_name="best_model.pt",
                num_epochs=num_epochs,
                device=device,
            )

        # ── Final test evaluation ────────────────────────────────────────────
        logger.info("Loading best checkpoint for test evaluation…")
        Checkpointer.load(run_dir / "checkpoints" / "best_model.pt", model, device=device)
        test_metrics = evaluate(
            model, test_loader, device,
            concept_names=concept_names,
            criterion=criterion,
            stage="joint" if mtype != "blackbox" else "task_only",
        )
        logger.info("Test results:\n" + format_metrics(test_metrics))

        with open(run_dir / "test_metrics.json", "w") as f:
            json.dump(test_metrics, f, indent=2)

        if mtype != "blackbox":
            int_res = evaluate_interventions(model, test_loader, device=device)
            with open(run_dir / "intervention_test.json", "w") as f:
                json.dump(int_res, f, indent=2)
            logger.info(
                f"Test interventions (0 → {model.num_concepts} concepts): "
                f"Acc {int_res['uncertainty_acc'][0]:.4f} → {int_res['uncertainty_acc'][-1]:.4f}"
            )

        logger.info(f"Training complete. Artifacts in: {run_dir}")
    finally:
        root_logger.removeHandler(fh)
        fh.close()


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def _apply_overrides(cfg: dict, overrides: list[str]) -> None:
    """Apply dot-notation key=value overrides to a nested config dict."""
    for item in overrides:
        if "=" not in item:
            raise ValueError(f"Override must be key=value, got: {item!r}")
        key_path, _, raw_value = item.partition("=")
        keys = key_path.strip().split(".")
        low = raw_value.strip().lower()
        if low in ("true", "false"):
            value = (low == "true")
        elif low in ("null", "none"):
            value = None
        else:
            try:
                value = ast.literal_eval(raw_value)
            except (ValueError, SyntaxError):
                value = raw_value
        node = cfg
        for k in keys[:-1]:
            node = node.setdefault(k, {})
        node[keys[-1]] = value


def main() -> None:
    parser = argparse.ArgumentParser(description="Train heartbeat CBM / CEM / Black-Box")
    parser.add_argument("--config",  default="config.yaml", help="Path to YAML config")
    parser.add_argument("--run-dir", default=None,          help="Output directory for this run")
    parser.add_argument("--seed",    type=int, default=None, help="Random seed (overrides training.seed)")
    parser.add_argument(
        "--set", dest="overrides", action="append", default=[],
        metavar="KEY=VALUE",
        help="Override config values via dot notation, e.g. --set model.type=cem",
    )
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    if args.overrides:
        _apply_overrides(cfg, args.overrides)
    if args.seed is not None:
        cfg.setdefault("training", {})["seed"] = args.seed

    run_dir = Path(args.run_dir) if args.run_dir else None
    train(cfg, run_dir)


if __name__ == "__main__":
    main()
