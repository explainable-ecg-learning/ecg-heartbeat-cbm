"""
pathology.py — Downstream 5-class PTB-XL Cardiac Pathology CBM, CEM & Black-Box.

Connects `PTBXLDataset` (10-second 12-lead ECGs + 5 diagnostic superclasses:
NORM, MI, STTC, CD, HYP) and `concepts.parquet` (20 clinical ECG concepts) to
evaluate the accuracy–interpretability trade-off and test-time clinical concept
interventions on a realistic, concept-incomplete diagnostic task.

Usage:
    python src/pathology.py --model-type cem --run-dir runs/pathology_cem --epochs 25
    python src/pathology.py --model-type cbm --run-dir runs/pathology_cbm --epochs 25
    python src/pathology.py --model-type blackbox --run-dir runs/pathology_bb --epochs 25
    python src/pathology.py --model-type cem --pretrained-run runs/cem_v2 --run-dir runs/pathology_cem_ft
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from sklearn.metrics import roc_auc_score
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from data import (
    CONCEPT_COLUMNS,
    NUM_RECORD_CONCEPTS,
    NUM_SUPERCLASSES,
    PTBXLDataset,
    SUPERCLASS_MAP,
    resolve_data_paths,
)
from models import TemporalEncoder
from utils import bootstrap_multilabel_auroc, set_seed

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

SUPERCLASS_NAMES: List[str] = [k for k, _ in sorted(SUPERCLASS_MAP.items(), key=lambda kv: kv[1])]


# ─────────────────────────────────────────────────────────────────────────────
# Robust Train-Fold Concept Normaliser (handles extreme delineation outliers)
# ─────────────────────────────────────────────────────────────────────────────

class ConceptNormalizer:
    """
    Fits [q_lo, q_hi] percentile clipping + z-score normalisation strictly on
    valid training-fold concepts (`concepts.parquet`), clamping to [-clip_z, clip_z].
    """

    def __init__(self, q_lo: float = 0.01, q_hi: float = 0.99, clip_z: float = 4.0) -> None:
        self.q_lo   = q_lo
        self.q_hi   = q_hi
        self.clip_z = clip_z
        self.lo_: Optional[np.ndarray] = None
        self.hi_: Optional[np.ndarray] = None
        self.mean_: Optional[np.ndarray] = None
        self.std_: Optional[np.ndarray] = None

    def fit(self, df_concepts, ecg_ids: List[int]) -> "ConceptNormalizer":
        if df_concepts is None:
            K = NUM_RECORD_CONCEPTS
            self.lo_   = np.full(K, -1e6, dtype=np.float32)
            self.hi_   = np.full(K,  1e6, dtype=np.float32)
            self.mean_ = np.zeros(K, dtype=np.float32)
            self.std_  = np.ones(K, dtype=np.float32)
            return self

        sub = df_concepts.reindex(index=ecg_ids, columns=CONCEPT_COLUMNS).values.astype(np.float64)
        K = sub.shape[1]
        lo, hi, mean, std = np.zeros(K), np.zeros(K), np.zeros(K), np.ones(K)
        for k in range(K):
            vals = sub[:, k]
            vals = vals[~np.isnan(vals)]
            if len(vals) < 10:
                lo[k], hi[k], mean[k], std[k] = -1e6, 1e6, 0.0, 1.0
                continue
            lo[k] = float(np.quantile(vals, self.q_lo))
            hi[k] = float(np.quantile(vals, self.q_hi))
            clipped = np.clip(vals, lo[k], hi[k])
            mean[k] = float(clipped.mean())
            std[k]  = float(max(clipped.std(), 1e-6))

        self.lo_   = lo.astype(np.float32)
        self.hi_   = hi.astype(np.float32)
        self.mean_ = mean.astype(np.float32)
        self.std_  = std.astype(np.float32)
        return self

    def transform_tensor(self, concepts: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        device = concepts.device
        lo   = torch.as_tensor(self.lo_,   device=device)
        hi   = torch.as_tensor(self.hi_,   device=device)
        mean = torch.as_tensor(self.mean_, device=device)
        std  = torch.as_tensor(self.std_,  device=device)
        clipped = torch.clamp(concepts, min=lo, max=hi)
        normed  = torch.clamp((clipped - mean) / std, -self.clip_z, self.clip_z)
        return torch.where(mask.bool(), normed, torch.zeros_like(normed))

    def to_dict(self) -> dict:
        return {
            "lo": self.lo_.tolist(),
            "hi": self.hi_.tolist(),
            "mean": self.mean_.tolist(),
            "std": self.std_.tolist(),
            "clip_z": self.clip_z,
        }


# ─────────────────────────────────────────────────────────────────────────────
# Attention-Pooled 10s ECG Encoder (compatible with pretrained TemporalEncoder)
# ─────────────────────────────────────────────────────────────────────────────

class AttentionTemporalEncoder(nn.Module):
    """
    Wraps `TemporalEncoder` conv blocks with gated temporal attention pooling so
    10-second ECG records focus on informative beats (PVCs, ST elevation, etc.).
    Directly accepts pretrained `TemporalEncoder` weights.
    """

    def __init__(self, in_channels: int = 12, base_channels: int = 64, dropout: float = 0.2) -> None:
        super().__init__()
        self.backbone = TemporalEncoder(in_channels, base_channels, dropout)
        d = self.backbone.out_dim
        self.attn = nn.Sequential(
            nn.Conv1d(d, d // 4, kernel_size=1),
            nn.Tanh(),
            nn.Conv1d(d // 4, 1, kernel_size=1),
        )
        self.out_dim = d

    def load_pretrained_encoder(self, ckpt_path: Path) -> None:
        state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        model_sd = state.get("model", state)
        enc_sd = {k.removeprefix("encoder."): v for k, v in model_sd.items() if k.startswith("encoder.")}
        missing, unexpected = self.backbone.load_state_dict(enc_sd, strict=False)
        logger.info(f"Loaded pretrained encoder from {ckpt_path} (missing={len(missing)}, unexpected={len(unexpected)})")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.backbone.stem(x)
        out = self.backbone.layer1(out)
        out = self.backbone.layer2(out)
        out = self.backbone.layer3(out)
        out = self.backbone.layer4(out)                      # (B, D, T')
        weights = torch.softmax(self.attn(out), dim=-1)      # (B, 1, T')
        pooled  = (out * weights).sum(dim=-1)                # (B, D)
        return self.backbone.dropout(pooled)


# ─────────────────────────────────────────────────────────────────────────────
# Pathology CBM, CEM & Black-Box Models (20 Continuous Concepts → 5 Superclasses)
# ─────────────────────────────────────────────────────────────────────────────

class PathologyCBM(nn.Module):
    """Continuous-concept CBM for 5-class PTB-XL pathology classification with RandInt."""

    model_type: str = "cbm"

    def __init__(
        self,
        in_channels: int = 12,
        base_channels: int = 64,
        num_concepts: int = NUM_RECORD_CONCEPTS,
        num_classes: int = NUM_SUPERCLASSES,
        dropout: float = 0.2,
        rand_int_prob: float = 0.25,
        detach_concepts: bool = False,
    ) -> None:
        super().__init__()
        self.encoder         = AttentionTemporalEncoder(in_channels, base_channels, dropout)
        self.num_concepts    = num_concepts
        self.num_classes     = num_classes
        self.rand_int_prob   = rand_int_prob
        self.detach_concepts = detach_concepts
        self.concept_head    = nn.Linear(self.encoder.out_dim, num_concepts)
        self.classifier      = nn.Linear(num_concepts, num_classes)

    def forward(
        self,
        x: torch.Tensor,
        concept_targets: Optional[torch.Tensor] = None,
        concept_mask: Optional[torch.Tensor] = None,
        intervene_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        z = self.encoder(x)
        c_pred = self.concept_head(z)
        c_base = c_pred.detach() if self.detach_concepts else c_pred
        c_in = c_base
        if intervene_mask is not None and concept_targets is not None:
            c_in = torch.where(intervene_mask.bool(), concept_targets, c_base)
        elif self.training and self.rand_int_prob > 0.0 and concept_targets is not None:
            bern = torch.rand_like(c_pred) < self.rand_int_prob
            if concept_mask is not None:
                bern = bern & concept_mask.bool()
            c_in = torch.where(bern, concept_targets, c_base)
        return self.classifier(c_in), c_pred


class PathologyCEM(nn.Module):
    """
    Continuous-concept CEM for 5-class PTB-XL pathology classification.
    Each clinical concept k learns high/low context embeddings (e⁺_k, e⁻_k) mixed
    via bounded concept activation σ(c_k) with RandInt during training.
    """

    model_type: str = "cem"

    def __init__(
        self,
        in_channels: int = 12,
        base_channels: int = 64,
        num_concepts: int = NUM_RECORD_CONCEPTS,
        num_classes: int = NUM_SUPERCLASSES,
        emb_size: int = 16,
        dropout: float = 0.2,
        rand_int_prob: float = 0.25,
    ) -> None:
        super().__init__()
        self.encoder       = AttentionTemporalEncoder(in_channels, base_channels, dropout)
        self.num_concepts  = num_concepts
        self.num_classes   = num_classes
        self.emb_size      = emb_size
        self.rand_int_prob = rand_int_prob

        d = self.encoder.out_dim
        self.pos_proj = nn.Sequential(nn.Linear(d, num_concepts * emb_size), nn.LeakyReLU(0.01))
        self.neg_proj = nn.Sequential(nn.Linear(d, num_concepts * emb_size), nn.LeakyReLU(0.01))
        self.scorer   = nn.Linear(emb_size * 2, 1)
        self.classifier = nn.Linear(num_concepts * emb_size, num_classes)

    def forward(
        self,
        x: torch.Tensor,
        concept_targets: Optional[torch.Tensor] = None,
        concept_mask: Optional[torch.Tensor] = None,
        intervene_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        z = self.encoder(x)
        B = z.size(0)
        c_pos = self.pos_proj(z).view(B, self.num_concepts, self.emb_size)
        c_neg = self.neg_proj(z).view(B, self.num_concepts, self.emb_size)
        c_pred = self.scorer(torch.cat([c_pos, c_neg], dim=-1)).squeeze(-1)   # (B, K)

        c_use = c_pred
        if intervene_mask is not None and concept_targets is not None:
            c_use = torch.where(intervene_mask.bool(), concept_targets, c_pred)
        elif self.training and self.rand_int_prob > 0.0 and concept_targets is not None:
            bern = torch.rand_like(c_pred) < self.rand_int_prob
            if concept_mask is not None:
                bern = bern & concept_mask.bool()
            c_use = torch.where(bern, concept_targets, c_pred)

        alpha = torch.sigmoid(c_use).unsqueeze(-1)
        mixed = alpha * c_pos + (1.0 - alpha) * c_neg
        return self.classifier(mixed.flatten(start_dim=1)), c_pred


class PathologyBlackBox(nn.Module):
    """End-to-end 10s ECG ResNet baseline without clinical concept bottleneck."""

    model_type: str = "blackbox"

    def __init__(
        self,
        in_channels: int = 12,
        base_channels: int = 64,
        num_concepts: int = NUM_RECORD_CONCEPTS,
        num_classes: int = NUM_SUPERCLASSES,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.encoder      = AttentionTemporalEncoder(in_channels, base_channels, dropout)
        self.num_concepts = num_concepts
        self.num_classes  = num_classes
        self.classifier   = nn.Linear(self.encoder.out_dim, num_classes)

    def forward(
        self,
        x: torch.Tensor,
        concept_targets: Optional[torch.Tensor] = None,
        concept_mask: Optional[torch.Tensor] = None,
        intervene_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        z = self.encoder(x)
        return self.classifier(z), torch.zeros(x.size(0), self.num_concepts, device=x.device)


def build_pathology_model(
    model_type: str = "cem",
    in_channels: int = 12,
    base_channels: int = 64,
    emb_size: int = 16,
    dropout: float = 0.2,
    rand_int_prob: float = 0.25,
    detach_concepts: bool = False,
) -> nn.Module:
    mt = model_type.lower()
    if mt == "cbm":
        return PathologyCBM(
            in_channels, base_channels, NUM_RECORD_CONCEPTS, NUM_SUPERCLASSES,
            dropout, rand_int_prob, detach_concepts=detach_concepts,
        )
    if mt == "cem":
        return PathologyCEM(in_channels, base_channels, NUM_RECORD_CONCEPTS, NUM_SUPERCLASSES, emb_size, dropout, rand_int_prob)
    if mt == "blackbox":
        return PathologyBlackBox(in_channels, base_channels, NUM_RECORD_CONCEPTS, NUM_SUPERCLASSES, dropout)
    raise ValueError(f"Unknown model_type={model_type!r}")


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation & Clinical Concept Interventions
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate_pathology(
    model: nn.Module,
    loader: DataLoader,
    normalizer: ConceptNormalizer,
    device: str,
    intervene_all: bool = False,
    compute_bootstrap: bool = False,
) -> Dict[str, float]:
    model.eval()
    probs_l, targets_l, c_preds_l, c_targs_l, c_masks_l = [], [], [], [], []
    for batch in tqdm(loader, desc="Pathology Eval", leave=False):
        x    = batch["signal"].to(device)
        y    = batch["pathology"]
        c_t  = normalizer.transform_tensor(batch["concepts"].to(device), batch["concept_mask"].to(device))
        c_m  = batch["concept_mask"].to(device)

        if intervene_all:
            logits, c_pred = model(x, concept_targets=c_t, intervene_mask=c_m)
        else:
            logits, c_pred = model(x)

        probs_l.append(torch.sigmoid(logits).cpu().numpy())
        targets_l.append(y.numpy())
        c_preds_l.append(c_pred.cpu().numpy())
        c_targs_l.append(c_t.cpu().numpy())
        c_masks_l.append(c_m.cpu().numpy())

    probs   = np.concatenate(probs_l, axis=0)
    targets = np.concatenate(targets_l, axis=0)
    c_preds = np.concatenate(c_preds_l, axis=0)
    c_targs = np.concatenate(c_targs_l, axis=0)
    c_masks = np.concatenate(c_masks_l, axis=0).astype(bool)

    metrics: Dict[str, float] = {}
    aurocs = []
    for idx, cls_name in enumerate(SUPERCLASS_NAMES):
        if len(np.unique(targets[:, idx])) >= 2:
            auc_v = float(roc_auc_score(targets[:, idx], probs[:, idx]))
            metrics[f"pathology/{cls_name}/auroc"] = auc_v
            aurocs.append(auc_v)
    metrics["pathology/macro_auroc"] = float(np.mean(aurocs)) if aurocs else float("nan")
    if compute_bootstrap:
        metrics.update(bootstrap_multilabel_auroc(targets, probs, SUPERCLASS_NAMES))

    if getattr(model, "model_type", "cbm") != "blackbox":
        maes, r2s = [], []
        for k in range(c_preds.shape[1]):
            m = c_masks[:, k]
            if m.sum() < 10:
                continue
            err = c_preds[m, k] - c_targs[m, k]
            mae = float(np.abs(err).mean())
            var = float(np.var(c_targs[m, k]))
            r2  = 1.0 - float(np.mean(err ** 2)) / max(var, 1e-6)
            maes.append(mae)
            r2s.append(r2)
        if maes:
            metrics["concept/mean_mae"] = float(np.mean(maes))
            metrics["concept/mean_r2"]  = float(np.mean(r2s))

    return metrics


# ─────────────────────────────────────────────────────────────────────────────
# CLI Training Loop
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Train PTB-XL 5-class Pathology CBM / CEM / Black-Box")
    parser.add_argument("--config",          default="config.yaml")
    parser.add_argument("--model-type",      default="cem", choices=["cbm", "cem", "blackbox"])
    parser.add_argument("--run-dir",         required=True)
    parser.add_argument("--sampling-rate",   type=int, default=100, choices=[100, 500])
    parser.add_argument("--epochs",          type=int, default=20)
    parser.add_argument("--batch-size",      type=int, default=128)
    parser.add_argument("--lr",              type=float, default=1e-3)
    parser.add_argument("--lambda-concept",  type=float, default=0.5)
    parser.add_argument("--emb-size",        type=int, default=16)
    parser.add_argument("--rand-int-prob",   type=float, default=0.25)
    parser.add_argument("--detach-concepts", action="store_true",
                        help="Stop task gradients into concept head (prevents CBM concept collapse)")
    parser.add_argument("--seed",            type=int, default=42)
    parser.add_argument("--pretrained-run",  default=None, help="Optional heartbeat run dir to warm-start encoder")
    args = parser.parse_args()

    set_seed(args.seed)

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    data_path, concept_path = resolve_data_paths(cfg)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    run_dir = Path(args.run_dir)
    (run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    (run_dir / "config.yaml").write_text(yaml.dump(vars(args), sort_keys=False))

    train_ds, val_ds, test_ds = PTBXLDataset.make_splits(
        data_path=data_path,
        sampling_rate=args.sampling_rate,
        concept_file=concept_path,
    )
    normalizer = ConceptNormalizer().fit(train_ds.concepts_df, train_ds.ecg_ids)
    (run_dir / "concept_normalizer.json").write_text(json.dumps(normalizer.to_dict(), indent=2))

    common = dict(batch_size=args.batch_size, num_workers=4, pin_memory=torch.cuda.is_available())
    train_loader = DataLoader(train_ds, shuffle=True, **common)
    val_loader   = DataLoader(val_ds,   shuffle=False, **common)
    test_loader  = DataLoader(test_ds,  shuffle=False, **common)

    model = build_pathology_model(
        model_type=args.model_type,
        emb_size=args.emb_size,
        rand_int_prob=args.rand_int_prob,
        detach_concepts=args.detach_concepts,
    ).to(device)

    if args.pretrained_run:
        ckpt = Path(args.pretrained_run) / "checkpoints" / "best_model.pt"
        if ckpt.exists():
            model.encoder.load_pretrained_encoder(ckpt)

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=max(1, args.epochs), eta_min=1e-5)

    best_auc = -1.0
    is_bb = (args.model_type == "blackbox")

    for epoch in range(1, args.epochs + 1):
        model.train()
        tot_loss = 0.0
        n_b = 0
        for batch in tqdm(train_loader, desc=f"Pathology {args.model_type.upper()} Ep {epoch}/{args.epochs}", leave=False):
            x   = batch["signal"].to(device)
            y   = batch["pathology"].to(device)
            c_m = batch["concept_mask"].to(device)
            c_t = normalizer.transform_tensor(batch["concepts"].to(device), c_m)

            logits, c_pred = model(x, concept_targets=c_t, concept_mask=c_m)
            loss_task = F.binary_cross_entropy_with_logits(logits, y)

            if is_bb or c_m.sum() < 1:
                loss_c = torch.zeros((), device=device)
            else:
                huber = F.smooth_l1_loss(c_pred, c_t, reduction="none")
                loss_c = (huber * c_m.float()).sum() / c_m.float().sum().clamp(min=1.0)

            loss = loss_task + args.lambda_concept * loss_c
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            tot_loss += float(loss.item())
            n_b += 1

        scheduler.step()
        val_m = evaluate_pathology(model, val_loader, normalizer, device)
        val_auc = val_m["pathology/macro_auroc"]
        logger.info(
            f"Epoch {epoch:2d} | Loss={tot_loss / max(1, n_b):.4f} | "
            f"Val Macro-AUROC={val_auc:.4f} | Concept R²={val_m.get('concept/mean_r2', 0.0):.3f}"
        )
        if val_auc > best_auc:
            best_auc = val_auc
            torch.save({"epoch": epoch, "val_macro_auroc": val_auc, "model": model.state_dict()},
                       run_dir / "checkpoints" / "best_model.pt")

    state = torch.load(run_dir / "checkpoints" / "best_model.pt", map_location=device, weights_only=False)
    model.load_state_dict(state["model"])
    test_m = evaluate_pathology(model, test_loader, normalizer, device, intervene_all=False, compute_bootstrap=True)
    if not is_bb:
        int_m = evaluate_pathology(model, test_loader, normalizer, device, intervene_all=True, compute_bootstrap=False)
        test_m["pathology/intervened_macro_auroc"] = int_m["pathology/macro_auroc"]
        test_m["pathology/intervention_gain"] = int_m["pathology/macro_auroc"] - test_m["pathology/macro_auroc"]

    (run_dir / "test_metrics.json").write_text(json.dumps(test_m, indent=2))
    logger.info("Test pathology results:\n" + json.dumps(test_m, indent=2))


if __name__ == "__main__":
    main()
