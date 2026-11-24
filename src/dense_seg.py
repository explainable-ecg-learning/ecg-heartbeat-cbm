"""
dense_seg.py — Per-Timestamp Soft-Segmentation CBM & CEM (1D U-Net).

Extends the window-level heartbeat CBM/CEM architecture to dense per-timestamp
concepts (100 Hz, 10 ms resolution) trained directly on Zenodo 7610236
(KIT-IBT/ECGdeli U-Net) soft segmentations jointly with 5-class PTB-XL cardiac
pathology classification (NORM, MI, STTC, CD, HYP).

Models:
  1. DenseSoftSegCBM (`dense_cbm`):
       1D U-Net → per-timestamp soft concept probabilities p(c_t | x) ∈ Δ^{K×T}
       + heteroscedastic uncertainty log σ²_{k,t} → mask-only temporal task head.
       Zero raw-signal leakage past the soft-segmentation bottleneck.
  2. DenseSoftSegCEM (`dense_cem`):
       1D U-Net → per-timestamp positive/negative concept embeddings (c⁺_{k,t}, c⁻_{k,t})
       + shared scorer → soft probabilities p_{k,t} & log σ²_{k,t}
       + temporal RandInt → mixed embeddings ĉ_{k,t} = p̃_{k,t} c⁺_{k,t} + (1 - p̃_{k,t}) c⁻_{k,t}
       → gated attention temporal task head.

Both models support:
  - Per-timestamp soft segmentation probabilities + epistemic & aleatoric uncertainty
  - Per-timestamp or span-level test-time concept interventions (e.g., overriding
    P / QRS / ST / T masks over any time span [t1, t2] or the full record)
  - Full GPU exploitation: BF16 AMP, TF32, cuDNN autotuning, fused AdamW,
    non-blocking pinned transfers, and batched MC-Dropout.

CLI:
    python src/dense_seg.py --model-type dense_cbm --run-dir runs/dense_cbm_softseg --epochs 25
    python src/dense_seg.py --model-type dense_cem --run-dir runs/dense_cem_softseg --epochs 25
"""

from __future__ import annotations

import argparse
import ast
import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import wfdb
import yaml
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from data import (
    NUM_SUPERCLASSES,
    SUPERCLASS_MAP,
    bandpass_filter,
    resolve_data_paths,
    z_normalize,
)
from evaluate import expected_calibration_error
from models import LOG_VAR_MAX, LOG_VAR_MIN, _mc_dropout_context, _ResBlock1D
from soft_seg import SOFT_SEG_CLASSES, SoftSegArchive
from utils import bootstrap_multilabel_auroc, set_seed

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

SUPERCLASS_NAMES: List[str] = [k for k, _ in sorted(SUPERCLASS_MAP.items(), key=lambda kv: kv[1])]

# Complete, disjoint partition of Zenodo 7610236's 24 classes into 6 canonical wave/segment concepts
WAVE6_NAMES: List[str] = ["P", "PQ", "QRS", "ST", "T", "TP"]
WAVE6_GROUPS: Dict[str, List[int]] = {
    "P":   [0, 1, 2, 3, 4],                     # P-Onset .. P-Offset
    "PQ":  [5],                                  # PR segment
    "QRS": [6, 7, 8, 9, 10, 11, 12, 13, 14],    # QRS-Onset .. QRS-Offset
    "ST":  [15, 16, 17],                         # qj, J-point, jt
    "T":   [18, 19, 20, 21, 22],                 # T-Onset .. T-Offset
    "TP":  [23],                                 # isoelectric baseline
}


def get_dense_concept_names(concept_mode: str = "wave6") -> List[str]:
    mode = concept_mode.lower()
    if mode == "wave6":
        return list(WAVE6_NAMES)
    if mode == "fiducial24":
        return list(SOFT_SEG_CLASSES)
    raise ValueError(f"Unknown concept_mode={concept_mode!r}; expected 'wave6' or 'fiducial24'")


def collapse_soft_seg_targets(
    soft_seg: np.ndarray,
    concept_mode: str = "wave6",
    lead_idx: Optional[int] = None,
) -> np.ndarray:
    """
    Convert raw Zenodo (T, 12, 24) soft probabilities in [0, 1] into a normalized
    (K, T) float32 probability simplex for per-timestamp training/evaluation.
    """
    seg = soft_seg[:, lead_idx, :] if lead_idx is not None else soft_seg.mean(axis=1)  # (T, 24)
    mode = concept_mode.lower()
    if mode == "wave6":
        out = np.stack([seg[:, WAVE6_GROUPS[name]].sum(axis=-1) for name in WAVE6_NAMES], axis=0)  # (6, T)
    elif mode == "fiducial24":
        out = seg.T.copy()  # (24, T)
    else:
        raise ValueError(f"Unknown concept_mode={concept_mode!r}")

    out = np.clip(out.astype(np.float32), 0.0, 1.0)
    denom = out.sum(axis=0, keepdims=True)
    # Fallback to uniform if an edge timestep has zero probability mass
    out = np.where(denom > 1e-6, out / np.maximum(denom, 1e-6), 1.0 / out.shape[0])
    return out.astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# 1D U-Net Dense Temporal Encoder-Decoder
# ─────────────────────────────────────────────────────────────────────────────

class TemporalUNet1D(nn.Module):
    """
    1D Residual U-Net preserving exact temporal resolution: (B, C, T) → (B, D, T).
    Reuses `_ResBlock1D` blocks with multi-scale skip connections and linear
    upsampling so any signal length T (e.g. 1000 @ 100 Hz or 500 @ 500 Hz) works.
    """

    def __init__(self, in_channels: int = 12, base_channels: int = 64, dropout: float = 0.15) -> None:
        super().__init__()
        c = base_channels
        self.stem = nn.Sequential(
            nn.Conv1d(in_channels, c, kernel_size=15, padding=7, bias=False),
            nn.BatchNorm1d(c),
            nn.ReLU(inplace=True),
        )
        # Encoder pyramid
        self.enc1 = _ResBlock1D(c,     c,     kernel_size=11, stride=1)
        self.enc2 = _ResBlock1D(c,     c * 2, kernel_size=11, stride=2)
        self.enc3 = _ResBlock1D(c * 2, c * 4, kernel_size=7,  stride=2)
        self.enc4 = _ResBlock1D(c * 4, c * 8, kernel_size=5,  stride=2)
        self.bottleneck = _ResBlock1D(c * 8, c * 8, kernel_size=5, stride=2)

        # Decoder pyramid
        self.dec4 = _ResBlock1D(c * 8 + c * 8, c * 4, kernel_size=5, stride=1)
        self.dec3 = _ResBlock1D(c * 4 + c * 4, c * 2, kernel_size=7, stride=1)
        self.dec2 = _ResBlock1D(c * 2 + c * 2, c,     kernel_size=9, stride=1)
        self.dec1 = _ResBlock1D(c + c,         c,     kernel_size=9, stride=1)

        self.dropout = nn.Dropout(p=dropout)
        self.out_dim = c

    @staticmethod
    def _up_concat(x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        up = F.interpolate(x, size=skip.shape[-1], mode="linear", align_corners=False)
        return torch.cat([up, skip], dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, C, T) → dense temporal features Z: (B, out_dim, T)."""
        s0 = self.stem(x)
        e1 = self.enc1(s0)          # (B, c,   T)
        e2 = self.enc2(e1)          # (B, 2c,  T/2)
        e3 = self.enc3(e2)          # (B, 4c,  T/4)
        e4 = self.enc4(e3)          # (B, 8c,  T/8)
        b  = self.bottleneck(e4)    # (B, 8c,  T/16)

        d4 = self.dec4(self._up_concat(b,  e4))
        d3 = self.dec3(self._up_concat(d4, e3))
        d2 = self.dec2(self._up_concat(d3, e2))
        d1 = self.dec1(self._up_concat(d2, e1))
        return self.dropout(d1)


# ─────────────────────────────────────────────────────────────────────────────
# Per-Timestamp Intervention & Temporal RandInt Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _apply_dense_intervention(
    c_probs: torch.Tensor,
    c_log_vars: torch.Tensor,
    intervene_labels: Optional[torch.Tensor] = None,
    intervene_mask: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply per-timestamp concept interventions on soft segmentation probabilities.

    Args:
        c_probs:          (B, K, T) predicted soft concept probabilities
        c_log_vars:       (B, K, T) predicted log-variances
        intervene_labels: (B, K, T) target soft/hard concept probabilities
        intervene_mask:   (B, 1, T) or (B, K, T) binary mask of intervened positions
    """
    if intervene_labels is None or intervene_mask is None:
        return c_probs, c_log_vars

    mask_bool = intervene_mask.bool()
    if mask_bool.ndim == 2:
        mask_bool = mask_bool.unsqueeze(1)  # (B, 1, T)
    if mask_bool.size(1) == 1 and c_probs.size(1) > 1:
        mask_bool = mask_bool.expand_as(c_probs)

    mask_f = mask_bool.to(c_probs.dtype)
    tgt = intervene_labels.to(dtype=c_probs.dtype)
    probs_out = (1.0 - mask_f) * c_probs + mask_f * tgt
    # Re-normalize simplex at timestamps where only a subset of classes was intervened
    denom = probs_out.sum(dim=1, keepdim=True).clamp(min=1e-6)
    probs_out = probs_out / denom
    lv_out = torch.where(mask_bool, torch.full_like(c_log_vars, -20.0), c_log_vars)
    return probs_out, lv_out


def _sample_temporal_randint_mask(
    shape: Tuple[int, int, int],
    prob: float,
    device: torch.device,
    span_len: int = 50,
) -> torch.Tensor:
    """
    Sample block-contiguous temporal RandInt masks of shape (B, 1, T) so the
    downstream head learns to respect coherent beat/segment interventions.
    """
    B, _, T = shape
    n_blocks = max(1, (T + span_len - 1) // span_len)
    block_mask = (torch.rand(B, 1, n_blocks, device=device) < prob).float()
    mask = F.interpolate(block_mask, size=T, mode="nearest")
    return mask.bool()


# ─────────────────────────────────────────────────────────────────────────────
# Dense Soft-Segmentation CBM (Pure Soft-Mask Bottleneck)
# ─────────────────────────────────────────────────────────────────────────────

class DenseMaskTaskHead(nn.Module):
    """
    Downstream 5-class pathology head operating STRICTLY on the per-timestamp
    soft segmentation probability timeline p(c_t | x) ∈ [0, 1]^{B × K × T}.
    Zero raw ECG features bypass the soft-segmentation bottleneck.
    """

    def __init__(self, num_concepts: int, num_classes: int = NUM_SUPERCLASSES, hidden: int = 64) -> None:
        super().__init__()
        self.temporal = nn.Sequential(
            nn.Conv1d(num_concepts, hidden, kernel_size=15, padding=7, bias=False),
            nn.BatchNorm1d(hidden),
            nn.ReLU(inplace=True),
            nn.Conv1d(hidden, hidden, kernel_size=15, stride=2, padding=7, bias=False),
            nn.BatchNorm1d(hidden),
            nn.ReLU(inplace=True),
            nn.Conv1d(hidden, hidden * 2, kernel_size=9, stride=2, padding=4, bias=False),
            nn.BatchNorm1d(hidden * 2),
            nn.ReLU(inplace=True),
        )
        # Combine global mask duty-cycles/variability + temporal pattern summary
        self.attn = nn.Sequential(
            nn.Conv1d(hidden * 2, hidden // 2, kernel_size=1),
            nn.Tanh(),
            nn.Conv1d(hidden // 2, 1, kernel_size=1),
        )
        self.fc = nn.Sequential(
            nn.Dropout(0.15),
            nn.Linear(hidden * 2 + num_concepts * 2, num_classes),
        )

    def forward(self, concept_probs: torch.Tensor) -> torch.Tensor:
        """concept_probs: (B, K, T) → task_logits: (B, num_classes)."""
        duty_mean = concept_probs.mean(dim=-1)                        # (B, K)
        duty_std  = concept_probs.std(dim=-1, unbiased=False)         # (B, K)
        h = self.temporal(concept_probs)                              # (B, 2H, T/4)
        w = torch.softmax(self.attn(h), dim=-1)                       # (B, 1, T/4)
        pooled = (h * w).sum(dim=-1)                                  # (B, 2H)
        return self.fc(torch.cat([pooled, duty_mean, duty_std], dim=-1))


class DenseSoftSegCBM(nn.Module):
    """
    Per-Timestamp Concept Bottleneck Model (1D U-Net + Soft-Seg Bottleneck).
    Produces per-timestamp soft wave probabilities & heteroscedastic uncertainty,
    and routes ONLY the soft segmentation timeline to the downstream task head.
    """

    model_type: str = "dense_cbm"

    def __init__(
        self,
        in_channels: int = 12,
        base_channels: int = 64,
        num_concepts: int = 6,
        num_classes: int = NUM_SUPERCLASSES,
        dropout: float = 0.15,
        mc_dropout_rate: float = 0.1,
        rand_int_prob: float = 0.25,
        detach_concepts: bool = False,
        span_len: int = 50,
    ) -> None:
        super().__init__()
        self.num_concepts    = num_concepts
        self.num_classes     = num_classes
        self.rand_int_prob   = rand_int_prob
        self.detach_concepts = detach_concepts
        self.span_len        = span_len

        self.unet         = TemporalUNet1D(in_channels, base_channels, dropout)
        self.mc_dropout   = nn.Dropout(p=mc_dropout_rate)
        self.concept_proj = nn.Conv1d(self.unet.out_dim, num_concepts * 2, kernel_size=1)
        self.task_head    = DenseMaskTaskHead(num_concepts, num_classes)

    def _concept_from_z(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        B, _, T = z.shape
        out = self.concept_proj(z).view(B, self.num_concepts, 2, T)
        logits   = out[:, :, 0, :]
        log_vars = out[:, :, 1, :].clamp(LOG_VAR_MIN, LOG_VAR_MAX)
        probs    = torch.softmax(logits, dim=1)
        return logits, probs, log_vars

    def forward(
        self,
        x: torch.Tensor,
        concept_targets: Optional[torch.Tensor] = None,
        intervene_labels: Optional[torch.Tensor] = None,
        intervene_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        z = self.mc_dropout(self.unet(x))
        c_logits, c_probs, c_log_vars = self._concept_from_z(z)

        c_probs_in = c_probs.detach() if self.detach_concepts else c_probs
        c_lv_in    = c_log_vars.detach() if self.detach_concepts else c_log_vars

        if intervene_labels is not None and intervene_mask is not None:
            c_probs_cls, c_lv_cls = _apply_dense_intervention(c_probs_in, c_lv_in, intervene_labels, intervene_mask)
        elif self.training and self.rand_int_prob > 0.0 and concept_targets is not None:
            rmask = _sample_temporal_randint_mask(
                (x.size(0), 1, x.size(-1)), self.rand_int_prob, x.device, span_len=self.span_len
            )
            c_probs_cls, c_lv_cls = _apply_dense_intervention(c_probs_in, c_lv_in, concept_targets, rmask)
        else:
            c_probs_cls, c_lv_cls = c_probs_in, c_lv_in

        task_logits = self.task_head(c_probs_cls)
        return {
            "task_logits":      task_logits,
            "concept_probs":    c_probs_cls if intervene_mask is not None else c_probs,
            "concept_logits":   c_logits,
            "concept_log_vars": c_lv_cls if intervene_mask is not None else c_log_vars,
        }

    @torch.no_grad()
    def get_bottleneck_summary(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns (concept_probs (B,K,T), pooled_bottleneck_repr (B,D)) for temporal leakage probing."""
        z = self.mc_dropout(self.unet(x))
        _, c_probs, _ = self._concept_from_z(z)
        return c_probs, _summarize_temporal_tensor(c_probs)

    @torch.no_grad()
    def intervene(
        self,
        x: torch.Tensor,
        intervene_labels: torch.Tensor,
        intervene_mask: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        was_training = self.training
        self.eval()
        out = self.forward(x, intervene_labels=intervene_labels, intervene_mask=intervene_mask)
        self.train(was_training)
        return out

    @torch.no_grad()
    def predict_with_uncertainty(
        self,
        x: torch.Tensor,
        n_mc_epistemic: int = 10,
        n_mc_aleatoric: int = 16,
        intervene_labels: Optional[torch.Tensor] = None,
        intervene_mask: Optional[torch.Tensor] = None,
        use_bf16: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """Batched GPU epistemic + aleatoric uncertainty per timestamp (B, K, T) and per class (B, C)."""
        B, C_in, T = x.shape
        M = max(1, n_mc_epistemic)
        x_rep = x.unsqueeze(0).expand(M, -1, -1, -1).reshape(M * B, C_in, T)
        amp_en = use_bf16 and x.is_cuda and torch.cuda.is_bf16_supported()

        with _mc_dropout_context(self), torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=amp_en):
            z = self.mc_dropout(self.unet(x_rep))
            c_logits, c_probs, c_lv = self._concept_from_z(z)

            if intervene_labels is not None and intervene_mask is not None:
                il_rep = intervene_labels.unsqueeze(0).expand(M, -1, -1, -1).reshape(M * B, self.num_concepts, T)
                im_rep = intervene_mask.unsqueeze(0).expand(M, -1, -1, -1).reshape(M * B, -1, T)
                c_probs, c_lv = _apply_dense_intervention(c_probs, c_lv, il_rep, im_rep)

            task_probs = torch.sigmoid(self.task_head(c_probs))  # (M*B, num_classes)

        c_logits_f = c_logits.float().view(M, B, self.num_concepts, T)
        c_probs_f  = c_probs.float().view(M, B, self.num_concepts, T)
        c_lv_f     = c_lv.float().view(M, B, self.num_concepts, T)
        t_probs_f  = task_probs.float().view(M, B, self.num_classes)

        # Aleatoric concept probability variance via reparameterized logit sampling on mean logits
        mean_logits = c_logits_f.mean(dim=0)                               # (B, K, T)
        mean_std    = torch.exp(0.5 * c_lv_f.mean(dim=0).clamp(-20.0, 6.0))
        eps = torch.randn(n_mc_aleatoric, B, self.num_concepts, T, device=x.device)
        al_samples = torch.softmax(mean_logits.unsqueeze(0) + eps * mean_std.unsqueeze(0), dim=2)
        c_aleatoric_var = al_samples.var(dim=0, unbiased=False)            # (B, K, T)
        if intervene_mask is not None:
            m_b = intervene_mask.bool()
            if m_b.size(1) == 1:
                m_b = m_b.expand_as(c_aleatoric_var)
            c_aleatoric_var = torch.where(m_b, torch.zeros_like(c_aleatoric_var), c_aleatoric_var)

        c_epistemic_var = c_probs_f.var(dim=0, unbiased=False)             # (B, K, T)
        return {
            "concept_probs_mean": c_probs_f.mean(dim=0),
            "concept_epistemic":  c_epistemic_var.sqrt(),
            "concept_aleatoric":  c_aleatoric_var.sqrt(),
            "concept_total_std":  (c_epistemic_var + c_aleatoric_var).clamp(min=0.0).sqrt(),
            "task_probs_mean":    t_probs_f.mean(dim=0),
            "task_probs_std":     t_probs_f.std(dim=0, unbiased=False),
        }


# ─────────────────────────────────────────────────────────────────────────────
# Dense Soft-Segmentation CEM (Per-Timestamp Concept Embeddings)
# ─────────────────────────────────────────────────────────────────────────────

class DenseConceptEmbeddingLayer(nn.Module):
    """
    Per-timestamp Concept Embedding Layer:
      For each timestamp t ∈ [0, T) and concept k ∈ {1..K}:
        c⁺_{k,t} = LeakyReLU(Conv1x1⁺(Z_t)) ∈ R^m
        c⁻_{k,t} = LeakyReLU(Conv1x1⁻(Z_t)) ∈ R^m
        (logit_{k,t}, log_var_{k,t}) = Scorer([c⁺_{k,t}, c⁻_{k,t}])
        p_{k,t}  = softmax_k(logit_{·, t})
        ĉ_{k,t}  = p̃_{k,t} c⁺_{k,t} + (1 - p̃_{k,t}) c⁻_{k,t} ∈ R^m
    """

    def __init__(
        self,
        in_dim: int = 64,
        num_concepts: int = 6,
        emb_size: int = 16,
        rand_int_prob: float = 0.25,
        span_len: int = 50,
    ) -> None:
        super().__init__()
        self.num_concepts  = num_concepts
        self.emb_size      = emb_size
        self.rand_int_prob = rand_int_prob
        self.span_len      = span_len

        self.pos_proj = nn.Sequential(
            nn.Conv1d(in_dim, num_concepts * emb_size, kernel_size=1),
            nn.LeakyReLU(0.01, inplace=True),
        )
        self.neg_proj = nn.Sequential(
            nn.Conv1d(in_dim, num_concepts * emb_size, kernel_size=1),
            nn.LeakyReLU(0.01, inplace=True),
        )
        # Shared per-concept scorer aligns active/inactive semantics across concepts & time
        self.scorer = nn.Linear(emb_size * 2, 2)

    def compute_embeddings_and_scores(
        self, z: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        B, _, T = z.shape
        K, m = self.num_concepts, self.emb_size
        c_pos = self.pos_proj(z).view(B, K, m, T)
        c_neg = self.neg_proj(z).view(B, K, m, T)

        # (B, K, 2m, T) → (B, K, T, 2m) → scorer → (B, K, T, 2)
        pair = torch.cat([c_pos, c_neg], dim=2).permute(0, 1, 3, 2)
        scores = self.scorer(pair)
        logits   = scores[..., 0]                                   # (B, K, T)
        log_vars = scores[..., 1].clamp(LOG_VAR_MIN, LOG_VAR_MAX)   # (B, K, T)
        return c_pos, c_neg, logits, log_vars

    @staticmethod
    def mix(c_pos: torch.Tensor, c_neg: torch.Tensor, probs: torch.Tensor) -> torch.Tensor:
        p = probs.unsqueeze(2)  # (B, K, 1, T)
        return p * c_pos + (1.0 - p) * c_neg

    def forward(
        self,
        z: torch.Tensor,
        concept_targets: Optional[torch.Tensor] = None,
        intervene_labels: Optional[torch.Tensor] = None,
        intervene_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        c_pos, c_neg, logits, log_vars = self.compute_embeddings_and_scores(z)
        c_probs = torch.softmax(logits, dim=1)
        mix_probs, lv_out = c_probs, log_vars

        if intervene_labels is not None and intervene_mask is not None:
            mix_probs, lv_out = _apply_dense_intervention(c_probs, log_vars, intervene_labels, intervene_mask)
        elif self.training and self.rand_int_prob > 0.0 and concept_targets is not None:
            rmask = _sample_temporal_randint_mask(
                (z.size(0), 1, z.size(-1)), self.rand_int_prob, z.device, span_len=self.span_len
            )
            mix_probs, _ = _apply_dense_intervention(c_probs, log_vars, concept_targets, rmask)

        mixed = self.mix(c_pos, c_neg, mix_probs)  # (B, K, m, T)
        return mixed, logits, c_probs, lv_out, mix_probs


class DenseCEMTaskHead(nn.Module):
    """
    Gated temporal attention + mean pooling over per-timestamp concept embeddings
    ĉ_{k,t} ∈ R^{B × K × m × T} → 5-class cardiac pathology logits.
    """

    def __init__(self, num_concepts: int, emb_size: int = 16, num_classes: int = NUM_SUPERCLASSES) -> None:
        super().__init__()
        d = num_concepts * emb_size
        self.proj = nn.Sequential(
            nn.Conv1d(d, d, kernel_size=5, padding=2, groups=num_concepts, bias=False),
            nn.BatchNorm1d(d),
            nn.ReLU(inplace=True),
        )
        self.attn = nn.Sequential(
            nn.Conv1d(d, max(16, d // 4), kernel_size=1),
            nn.Tanh(),
            nn.Conv1d(max(16, d // 4), 1, kernel_size=1),
        )
        self.fc = nn.Sequential(
            nn.Dropout(0.15),
            nn.Linear(d * 2, num_classes),
        )

    def forward(self, mixed_embs: torch.Tensor) -> torch.Tensor:
        """mixed_embs: (B, K, m, T) → task_logits: (B, num_classes)."""
        B, K, m, T = mixed_embs.shape
        seq = mixed_embs.reshape(B, K * m, T)
        h = self.proj(seq)
        w = torch.softmax(self.attn(h), dim=-1)      # (B, 1, T)
        attn_pool = (h * w).sum(dim=-1)              # (B, K*m)
        mean_pool = h.mean(dim=-1)                   # (B, K*m)
        return self.fc(torch.cat([attn_pool, mean_pool], dim=-1))


class DenseSoftSegCEM(nn.Module):
    """
    Per-Timestamp Concept Embedding Model (1D U-Net + Per-Timestamp CEM Bottleneck).
    Simultaneously predicts sharp per-timestamp soft segmentation probabilities,
    per-timestamp epistemic/aleatoric uncertainty, and rich wave-conditioned
    concept embeddings for downstream 5-class cardiac pathology classification.
    """

    model_type: str = "dense_cem"

    def __init__(
        self,
        in_channels: int = 12,
        base_channels: int = 64,
        num_concepts: int = 6,
        emb_size: int = 16,
        num_classes: int = NUM_SUPERCLASSES,
        dropout: float = 0.15,
        mc_dropout_rate: float = 0.1,
        rand_int_prob: float = 0.25,
        span_len: int = 50,
    ) -> None:
        super().__init__()
        self.num_concepts  = num_concepts
        self.emb_size      = emb_size
        self.num_classes   = num_classes
        self.rand_int_prob = rand_int_prob
        self.span_len      = span_len

        self.unet          = TemporalUNet1D(in_channels, base_channels, dropout)
        self.mc_dropout    = nn.Dropout(p=mc_dropout_rate)
        self.concept_layer = DenseConceptEmbeddingLayer(
            self.unet.out_dim, num_concepts, emb_size, rand_int_prob, span_len=span_len
        )
        self.task_head     = DenseCEMTaskHead(num_concepts, emb_size, num_classes)

    @torch.no_grad()
    def get_bottleneck_summary(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns (concept_probs (B,K,T), pooled_mixed_embs (B,D)) for temporal leakage probing."""
        z = self.mc_dropout(self.unet(x))
        mixed, _, c_probs, _, _ = self.concept_layer(z)
        B, K, m, T = mixed.shape
        seq = mixed.reshape(B, K * m, T)
        return c_probs, torch.cat([seq.mean(dim=-1), seq.std(dim=-1, unbiased=False)], dim=-1)

    def forward(
        self,
        x: torch.Tensor,
        concept_targets: Optional[torch.Tensor] = None,
        intervene_labels: Optional[torch.Tensor] = None,
        intervene_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        z = self.mc_dropout(self.unet(x))
        mixed, c_logits, c_probs, c_lv, mix_probs = self.concept_layer(
            z,
            concept_targets=concept_targets,
            intervene_labels=intervene_labels,
            intervene_mask=intervene_mask,
        )
        task_logits = self.task_head(mixed)
        return {
            "task_logits":      task_logits,
            "concept_probs":    mix_probs if intervene_mask is not None else c_probs,
            "concept_logits":   c_logits,
            "concept_log_vars": c_lv,
        }

    @torch.no_grad()
    def intervene(
        self,
        x: torch.Tensor,
        intervene_labels: torch.Tensor,
        intervene_mask: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        was_training = self.training
        self.eval()
        out = self.forward(x, intervene_labels=intervene_labels, intervene_mask=intervene_mask)
        self.train(was_training)
        return out

    @torch.no_grad()
    def predict_with_uncertainty(
        self,
        x: torch.Tensor,
        n_mc_epistemic: int = 10,
        n_mc_aleatoric: int = 16,
        intervene_labels: Optional[torch.Tensor] = None,
        intervene_mask: Optional[torch.Tensor] = None,
        use_bf16: bool = True,
    ) -> Dict[str, torch.Tensor]:
        B, C_in, T = x.shape
        M = max(1, n_mc_epistemic)
        x_rep = x.unsqueeze(0).expand(M, -1, -1, -1).reshape(M * B, C_in, T)
        amp_en = use_bf16 and x.is_cuda and torch.cuda.is_bf16_supported()

        with _mc_dropout_context(self), torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=amp_en):
            z = self.mc_dropout(self.unet(x_rep))
            il_rep = (
                intervene_labels.unsqueeze(0).expand(M, -1, -1, -1).reshape(M * B, self.num_concepts, T)
                if intervene_labels is not None else None
            )
            im_rep = (
                intervene_mask.unsqueeze(0).expand(M, -1, -1, -1).reshape(M * B, -1, T)
                if intervene_mask is not None else None
            )
            mixed, c_logits, c_probs, c_lv, mix_probs = self.concept_layer(
                z, intervene_labels=il_rep, intervene_mask=im_rep
            )
            eff_probs = mix_probs if im_rep is not None else c_probs
            task_probs = torch.sigmoid(self.task_head(mixed))

        c_logits_f = c_logits.float().view(M, B, self.num_concepts, T)
        c_probs_f  = eff_probs.float().view(M, B, self.num_concepts, T)
        c_lv_f     = c_lv.float().view(M, B, self.num_concepts, T)
        t_probs_f  = task_probs.float().view(M, B, self.num_classes)

        mean_logits = c_logits_f.mean(dim=0)
        mean_std    = torch.exp(0.5 * c_lv_f.mean(dim=0).clamp(-20.0, 6.0))
        eps = torch.randn(n_mc_aleatoric, B, self.num_concepts, T, device=x.device)
        al_samples = torch.softmax(mean_logits.unsqueeze(0) + eps * mean_std.unsqueeze(0), dim=2)
        c_aleatoric_var = al_samples.var(dim=0, unbiased=False)
        if intervene_mask is not None:
            m_b = intervene_mask.bool()
            if m_b.size(1) == 1:
                m_b = m_b.expand_as(c_aleatoric_var)
            c_aleatoric_var = torch.where(m_b, torch.zeros_like(c_aleatoric_var), c_aleatoric_var)

        c_epistemic_var = c_probs_f.var(dim=0, unbiased=False)
        return {
            "concept_probs_mean": c_probs_f.mean(dim=0),
            "concept_epistemic":  c_epistemic_var.sqrt(),
            "concept_aleatoric":  c_aleatoric_var.sqrt(),
            "concept_total_std":  (c_epistemic_var + c_aleatoric_var).clamp(min=0.0).sqrt(),
            "task_probs_mean":    t_probs_f.mean(dim=0),
            "task_probs_std":     t_probs_f.std(dim=0, unbiased=False),
        }


def _summarize_temporal_tensor(x: torch.Tensor) -> torch.Tensor:
    """
    Summarize a per-timestamp (B, K, T) probability timeline into fixed-length
    per-record temporal statistics: [mean, std, q90, mean |Δt|] → (B, 4*K).
    Used for temporal leakage / OIS probing on DenseCBM vs. DenseCEM.
    """
    xf = x.float()
    mean = xf.mean(dim=-1)
    std  = xf.std(dim=-1, unbiased=False)
    q90  = torch.quantile(xf, 0.90, dim=-1)
    diff = (xf[..., 1:] - xf[..., :-1]).abs().mean(dim=-1)
    return torch.cat([mean, std, q90, diff], dim=-1)


# ─────────────────────────────────────────────────────────────────────────────
# Matched-Backbone Baselines: DenseBlackBox & DenseMultiTask (No Bottleneck)
# ─────────────────────────────────────────────────────────────────────────────

class DenseLatentTaskHead(nn.Module):
    """
    Gated temporal attention + mean pooling directly on U-Net dense features
    Z ∈ R^{B × D × T} → 5-class cardiac pathology logits (no concept bottleneck).
    """

    def __init__(self, in_dim: int = 64, num_classes: int = NUM_SUPERCLASSES) -> None:
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv1d(in_dim, in_dim, kernel_size=5, padding=2, bias=False),
            nn.BatchNorm1d(in_dim),
            nn.ReLU(inplace=True),
        )
        self.attn = nn.Sequential(
            nn.Conv1d(in_dim, max(16, in_dim // 4), kernel_size=1),
            nn.Tanh(),
            nn.Conv1d(max(16, in_dim // 4), 1, kernel_size=1),
        )
        self.fc = nn.Sequential(
            nn.Dropout(0.15),
            nn.Linear(in_dim * 2, num_classes),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        h = self.proj(z)
        w = torch.softmax(self.attn(h), dim=-1)
        attn_pool = (h * w).sum(dim=-1)
        mean_pool = h.mean(dim=-1)
        return self.fc(torch.cat([attn_pool, mean_pool], dim=-1))


class DenseBlackBox(nn.Module):
    """
    Matched-backbone end-to-end 1D U-Net baseline trained ONLY on 5-class
    PTB-XL cardiac pathology (no soft-segmentation supervision or bottleneck).
    """

    model_type: str = "dense_blackbox"

    def __init__(
        self,
        in_channels: int = 12,
        base_channels: int = 64,
        num_concepts: int = 6,
        num_classes: int = NUM_SUPERCLASSES,
        dropout: float = 0.15,
        mc_dropout_rate: float = 0.1,
    ) -> None:
        super().__init__()
        self.num_concepts = num_concepts
        self.num_classes  = num_classes
        self.unet         = TemporalUNet1D(in_channels, base_channels, dropout)
        self.mc_dropout   = nn.Dropout(p=mc_dropout_rate)
        self.task_head    = DenseLatentTaskHead(self.unet.out_dim, num_classes)

    def forward(
        self,
        x: torch.Tensor,
        concept_targets: Optional[torch.Tensor] = None,
        intervene_labels: Optional[torch.Tensor] = None,
        intervene_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        z = self.mc_dropout(self.unet(x))
        task_logits = self.task_head(z)
        B, _, T = x.shape
        zeros = torch.zeros((B, self.num_concepts, T), device=x.device, dtype=z.dtype)
        return {
            "task_logits":      task_logits,
            "concept_probs":    zeros,
            "concept_logits":   zeros,
            "concept_log_vars": zeros,
        }

    @torch.no_grad()
    def predict_with_uncertainty(
        self,
        x: torch.Tensor,
        n_mc_epistemic: int = 10,
        n_mc_aleatoric: int = 16,
        intervene_labels: Optional[torch.Tensor] = None,
        intervene_mask: Optional[torch.Tensor] = None,
        use_bf16: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """Batched GPU MC-Dropout epistemic task uncertainty for DenseBlackBox."""
        B, C_in, T = x.shape
        M = max(1, n_mc_epistemic)
        x_rep = x.unsqueeze(0).expand(M, -1, -1, -1).reshape(M * B, C_in, T)
        amp_en = use_bf16 and x.is_cuda and torch.cuda.is_bf16_supported()

        with _mc_dropout_context(self), torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=amp_en):
            task_probs = torch.sigmoid(self.task_head(self.mc_dropout(self.unet(x_rep))))

        t_probs_f = task_probs.float().view(M, B, self.num_classes)
        zeros_c = torch.zeros((B, self.num_concepts, T), device=x.device, dtype=torch.float32)
        return {
            "concept_probs_mean": zeros_c,
            "concept_epistemic":  zeros_c,
            "concept_aleatoric":  zeros_c,
            "concept_total_std":  zeros_c,
            "task_probs_mean":    t_probs_f.mean(dim=0),
            "task_probs_std":     t_probs_f.std(dim=0, unbiased=False),
        }


class DenseMultiTask(nn.Module):
    """
    Matched-backbone multi-task baseline: shares `TemporalUNet1D` between
    per-timestamp soft segmentation AND a direct latent pathology head (no
    concept bottleneck). Isolates the cost of bottlenecking vs. auxiliary seg.
    """

    model_type: str = "dense_multitask"

    def __init__(
        self,
        in_channels: int = 12,
        base_channels: int = 64,
        num_concepts: int = 6,
        num_classes: int = NUM_SUPERCLASSES,
        dropout: float = 0.15,
        mc_dropout_rate: float = 0.1,
    ) -> None:
        super().__init__()
        self.num_concepts = num_concepts
        self.num_classes  = num_classes
        self.unet         = TemporalUNet1D(in_channels, base_channels, dropout)
        self.mc_dropout   = nn.Dropout(p=mc_dropout_rate)
        self.concept_proj = nn.Conv1d(self.unet.out_dim, num_concepts * 2, kernel_size=1)
        self.task_head    = DenseLatentTaskHead(self.unet.out_dim, num_classes)

    def _concept_from_z(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        B, _, T = z.shape
        out = self.concept_proj(z).view(B, self.num_concepts, 2, T)
        logits   = out[:, :, 0, :]
        log_vars = out[:, :, 1, :].clamp(LOG_VAR_MIN, LOG_VAR_MAX)
        probs    = torch.softmax(logits, dim=1)
        return logits, probs, log_vars

    def forward(
        self,
        x: torch.Tensor,
        concept_targets: Optional[torch.Tensor] = None,
        intervene_labels: Optional[torch.Tensor] = None,
        intervene_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        z = self.mc_dropout(self.unet(x))
        c_logits, c_probs, c_log_vars = self._concept_from_z(z)
        task_logits = self.task_head(z)
        return {
            "task_logits":      task_logits,
            "concept_probs":    c_probs,
            "concept_logits":   c_logits,
            "concept_log_vars": c_log_vars,
        }

    @torch.no_grad()
    def predict_with_uncertainty(
        self,
        x: torch.Tensor,
        n_mc_epistemic: int = 10,
        n_mc_aleatoric: int = 16,
        intervene_labels: Optional[torch.Tensor] = None,
        intervene_mask: Optional[torch.Tensor] = None,
        use_bf16: bool = True,
    ) -> Dict[str, torch.Tensor]:
        B, C_in, T = x.shape
        M = max(1, n_mc_epistemic)
        x_rep = x.unsqueeze(0).expand(M, -1, -1, -1).reshape(M * B, C_in, T)
        amp_en = use_bf16 and x.is_cuda and torch.cuda.is_bf16_supported()
        with _mc_dropout_context(self), torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=amp_en):
            z = self.mc_dropout(self.unet(x_rep))
            c_logits, c_probs, c_lv = self._concept_from_z(z)
            task_probs = torch.sigmoid(self.task_head(z))
        c_logits_f = c_logits.float().view(M, B, self.num_concepts, T)
        c_probs_f  = c_probs.float().view(M, B, self.num_concepts, T)
        c_lv_f     = c_lv.float().view(M, B, self.num_concepts, T)
        t_probs_f  = task_probs.float().view(M, B, self.num_classes)
        mean_logits = c_logits_f.mean(dim=0)
        mean_std    = torch.exp(0.5 * c_lv_f.mean(dim=0).clamp(-20.0, 6.0))
        eps = torch.randn(n_mc_aleatoric, B, self.num_concepts, T, device=x.device)
        al_samples = torch.softmax(mean_logits.unsqueeze(0) + eps * mean_std.unsqueeze(0), dim=2)
        c_aleatoric_var = al_samples.var(dim=0, unbiased=False)
        c_epistemic_var = c_probs_f.var(dim=0, unbiased=False)
        return {
            "concept_probs_mean": c_probs_f.mean(dim=0),
            "concept_epistemic":  c_epistemic_var.sqrt(),
            "concept_aleatoric":  c_aleatoric_var.sqrt(),
            "concept_total_std":  (c_epistemic_var + c_aleatoric_var).clamp(min=0.0).sqrt(),
            "task_probs_mean":    t_probs_f.mean(dim=0),
            "task_probs_std":     t_probs_f.std(dim=0, unbiased=False),
        }


def build_dense_model(cfg: dict) -> nn.Module:
    mcfg = cfg.get("model", cfg)
    mtype = str(mcfg.get("type", "dense_cem")).lower()
    in_ch = int(mcfg.get("in_channels", 12))
    base_ch = int(mcfg.get("base_channels", 64))
    num_c = int(mcfg.get("num_concepts", 6))
    emb_s = int(mcfg.get("emb_size", 16))
    num_cls = int(mcfg.get("num_classes", NUM_SUPERCLASSES))
    drop = float(mcfg.get("dropout", 0.15))
    mc_drop = float(mcfg.get("mc_dropout_rate", 0.1))
    ri_prob = float(mcfg.get("rand_int_prob", 0.25))
    detach_c = bool(mcfg.get("detach_concepts", False))
    span_len = int(mcfg.get("span_len", 50))

    if mtype == "dense_cbm":
        return DenseSoftSegCBM(
            in_ch, base_ch, num_c, num_cls, drop, mc_drop, ri_prob,
            detach_concepts=detach_c, span_len=span_len,
        )
    if mtype == "dense_cem":
        return DenseSoftSegCEM(
            in_ch, base_ch, num_c, emb_s, num_cls, drop, mc_drop, ri_prob, span_len=span_len
        )
    if mtype == "dense_blackbox":
        return DenseBlackBox(in_ch, base_ch, num_c, num_cls, drop, mc_drop)
    if mtype == "dense_multitask":
        return DenseMultiTask(in_ch, base_ch, num_c, num_cls, drop, mc_drop)
    raise ValueError(f"Unknown dense model type: {mtype!r}")


# ─────────────────────────────────────────────────────────────────────────────
# Heteroscedastic Soft-Segmentation + Downstream Pathology Loss (FP32)
# ─────────────────────────────────────────────────────────────────────────────

class DenseSoftSegLoss(nn.Module):
    """
    Combined loss for per-timestamp soft segmentation + 5-class pathology:
      - Heteroscedastic MC soft cross-entropy (Kendall & Gal 2017) against Zenodo soft targets
      - Soft Dice loss over (B, K, T) for sharp wave onset/offset boundaries
      - Variance regularization on log_vars
      - Multi-label BCEWithLogitsLoss on 5-class PTB-XL pathology superclasses
    Always computed in FP32 for numerical stability under BF16 AMP.
    """

    def __init__(
        self,
        lambda_seg: float = 2.0,
        lambda_dice: float = 0.5,
        lambda_task: float = 1.0,
        var_reg_weight: float = 1e-3,
        n_mc_samples: int = 6,
        use_heteroscedastic: bool = True,
        compute_seg: bool = True,
    ) -> None:
        super().__init__()
        self.lambda_seg          = lambda_seg
        self.lambda_dice         = lambda_dice
        self.lambda_task         = lambda_task
        self.var_reg_weight      = var_reg_weight
        self.n_mc_samples        = n_mc_samples
        self.use_heteroscedastic = use_heteroscedastic
        self.compute_seg         = compute_seg

    def forward(
        self,
        outputs: Dict[str, torch.Tensor],
        seg_targets: torch.Tensor,
        task_targets: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        t_log  = outputs["task_logits"].float()         # (B, C)
        y_task = task_targets.float()                   # (B, C)
        loss_task = F.binary_cross_entropy_with_logits(t_log, y_task)

        if not self.compute_seg:
            zero = torch.zeros((), device=t_log.device)
            return {
                "total":    self.lambda_task * loss_task,
                "seg_ce":   zero,
                "seg_dice": zero,
                "task":     loss_task,
            }

        logits = outputs["concept_logits"].float()      # (B, K, T)
        log_v  = outputs["concept_log_vars"].float()    # (B, K, T)
        y_seg  = seg_targets.float()                    # (B, K, T)

        # 1. Deterministic soft CE + optional heteroscedastic MC soft cross-entropy
        log_probs_det = F.log_softmax(logits, dim=1)
        loss_ce_det = -(y_seg * log_probs_det).sum(dim=1).mean()
        if self.use_heteroscedastic:
            stds = torch.exp(0.5 * log_v)
            eps  = torch.randn(self.n_mc_samples, *logits.shape, device=logits.device, dtype=torch.float32)
            sampled_logits = logits.unsqueeze(0) + eps * stds.unsqueeze(0)
            log_probs_mc = F.log_softmax(sampled_logits, dim=2)
            loss_ce_mc = -(y_seg.unsqueeze(0) * log_probs_mc).sum(dim=2).mean()
            loss_ce = 0.5 * loss_ce_mc + 0.5 * loss_ce_det
            loss_var = (log_v ** 2).mean() * self.var_reg_weight
        else:
            loss_ce = loss_ce_det
            loss_var = torch.zeros((), device=t_log.device)

        # 2. Soft Dice loss per concept averaged over batch and concepts
        probs = torch.softmax(logits, dim=1)
        inter = (probs * y_seg).sum(dim=(0, 2))
        denom = probs.sum(dim=(0, 2)) + y_seg.sum(dim=(0, 2))
        loss_dice = (1.0 - (2.0 * inter + 1e-4) / (denom + 1e-4)).mean()

        total = (
            self.lambda_seg * loss_ce
            + self.lambda_dice * loss_dice
            + self.lambda_task * loss_task
            + loss_var
        )
        return {
            "total":    total,
            "seg_ce":   loss_ce,
            "seg_dice": loss_dice,
            "task":     loss_task,
        }


# ─────────────────────────────────────────────────────────────────────────────
# Fast RAM-Cached 100 Hz PTB-XL + Zenodo Soft-Segmentation Dataset
# ─────────────────────────────────────────────────────────────────────────────

class DenseSoftSegDataset(Dataset):
    """
    Compact in-memory dataset of 10-second 100 Hz 12-lead ECGs paired with
    per-timestamp Zenodo soft-segmentation targets and 5-class PTB-XL labels.
    Zero disk I/O during training → saturates RTX A6000 Tensor Cores!
    """

    def __init__(
        self,
        signals_f16: np.ndarray,       # (N, 12, 1000) float16
        seg_u8: np.ndarray,            # (N, K, 1000)  uint8 in [0, 255]
        pathology_f32: np.ndarray,     # (N, 5)        float32
        ecg_ids: np.ndarray,           # (N,)          int32
        augment: bool = False,
    ) -> None:
        self.signals   = signals_f16
        self.seg_u8    = seg_u8
        self.pathology = pathology_f32
        self.ecg_ids   = ecg_ids
        self.augment   = augment

    def __len__(self) -> int:
        return len(self.ecg_ids)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sig = self.signals[idx].astype(np.float32)       # (12, 1000)
        seg = self.seg_u8[idx].astype(np.float32) / 255.0  # (K, 1000)
        # Re-normalize simplex after uint8 dequantization
        seg /= np.maximum(seg.sum(axis=0, keepdims=True), 1e-6)

        if self.augment:
            # Synchronized temporal shift (rolls both signal and soft mask equally!)
            shift = int(np.random.randint(-25, 26))
            if shift != 0:
                sig = np.roll(sig, shift, axis=-1)
                seg = np.roll(seg, shift, axis=-1)
            # Amplitude scaling + mild additive Gaussian noise (mask-invariant)
            sig = sig * float(np.random.uniform(0.85, 1.15))
            sig = sig + np.random.normal(0.0, 0.015, size=sig.shape).astype(np.float32)
            # Random lead dropout (up to 2 leads)
            if np.random.random() < 0.15:
                n_drop = int(np.random.randint(1, 3))
                drop_leads = np.random.choice(sig.shape[0], size=n_drop, replace=False)
                sig[drop_leads] = 0.0

        return {
            "signal":    torch.from_numpy(sig),
            "seg":       torch.from_numpy(seg),
            "pathology": torch.from_numpy(self.pathology[idx]),
            "record_id": int(self.ecg_ids[idx]),
        }


def load_or_build_dense_datasets(
    data_path: Path,
    soft_seg_path: Path,
    cache_root: Path,
    concept_mode: str = "wave6",
    lead_idx: Optional[int] = None,
    val_fold: int = 9,
    test_fold: int = 10,
    build_workers: int = 16,
) -> Dict[str, DenseSoftSegDataset]:
    """Load or build compact `.npz` split caches for 100 Hz PTB-XL + Zenodo soft segs."""
    lead_tag = "all" if lead_idx is None else f"lead{lead_idx}"
    cache_dir = cache_root / f"{concept_mode}_{lead_tag}_sr100"
    splits = ("train", "val", "test")

    if all((cache_dir / f"{s}.npz").exists() for s in splits):
        out: Dict[str, DenseSoftSegDataset] = {}
        for s in splits:
            z = np.load(cache_dir / f"{s}.npz")
            out[s] = DenseSoftSegDataset(
                z["signals"], z["seg_u8"], z["pathology"], z["ecg_ids"], augment=(s == "train")
            )
            logger.info(f"Loaded dense soft-seg {s} cache ({len(out[s])} records) from {cache_dir / f'{s}.npz'}")
        return out

    meta = pd.read_csv(data_path / "ptbxl_database.csv", index_col="ecg_id", dtype={"scp_codes": str})
    scp = pd.read_csv(data_path / "scp_statements.csv", index_col=0)
    scp = scp[scp["diagnostic"] == 1.0]
    code_to_sc = scp["diagnostic_class"].dropna().to_dict()

    def _parse_patho(scp_str: str) -> np.ndarray:
        lab = np.zeros(NUM_SUPERCLASSES, dtype=np.float32)
        try:
            codes = ast.literal_eval(scp_str)
        except Exception:
            return lab
        for code, likelihood in codes.items():
            sc = code_to_sc.get(code)
            if sc and sc in SUPERCLASS_MAP and likelihood > 0:
                lab[SUPERCLASS_MAP[sc]] = 1.0
        return lab

    meta["pathology"] = meta["scp_codes"].apply(_parse_patho)
    archive = SoftSegArchive(soft_seg_path)
    logger.info(f"Building dense 100 Hz soft-seg dataset ({concept_mode}, lead={lead_tag}) from {len(archive)} Zenodo records…")

    train_folds = [f for f in range(1, 11) if f not in (val_fold, test_fold)]
    split_dfs = {
        "train": meta[meta["strat_fold"].isin(train_folds)],
        "val":   meta[meta["strat_fold"] == val_fold],
        "test":  meta[meta["strat_fold"] == test_fold],
    }

    def _load_one(item):
        ecg_id, rel_fn, patho = item
        if ecg_id not in archive:
            return None
        try:
            raw, _ = wfdb.rdsamp(str(data_path / rel_fn))
            sig = z_normalize(bandpass_filter(raw.T.astype(np.float32), sampling_rate=100))
            if sig.shape != (12, 1000):
                return None
            ss = archive.load(ecg_id)                                   # (1000, 12, 24)
            seg = collapse_soft_seg_targets(ss, concept_mode, lead_idx) # (K, 1000)
            seg_u8 = np.clip(np.round(seg * 255.0), 0, 255).astype(np.uint8)
            return int(ecg_id), sig.astype(np.float16), seg_u8, patho.astype(np.float32)
        except Exception:
            return None

    cache_dir.mkdir(parents=True, exist_ok=True)
    datasets: Dict[str, DenseSoftSegDataset] = {}
    try:
        for sp_name, df in split_dfs.items():
            items = list(zip(df.index.tolist(), df["filename_lr"].tolist(), df["pathology"].tolist()))
            ids_l, sigs_l, segs_l, path_l = [], [], [], []
            with ThreadPoolExecutor(max_workers=max(1, build_workers)) as pool:
                for res in tqdm(pool.map(_load_one, items), total=len(items), desc=f"Build dense {sp_name}"):
                    if res is None:
                        continue
                    eid, s16, su8, pf = res
                    ids_l.append(eid); sigs_l.append(s16); segs_l.append(su8); path_l.append(pf)

            sigs_arr = np.stack(sigs_l, axis=0)
            segs_arr = np.stack(segs_l, axis=0)
            path_arr = np.stack(path_l, axis=0)
            ids_arr  = np.asarray(ids_l, dtype=np.int32)
            np.savez(cache_dir / f"{sp_name}.npz", signals=sigs_arr, seg_u8=segs_arr, pathology=path_arr, ecg_ids=ids_arr)
            logger.info(f"Saved dense {sp_name} cache ({len(ids_arr)} records) → {cache_dir / f'{sp_name}.npz'}")
            datasets[sp_name] = DenseSoftSegDataset(sigs_arr, segs_arr, path_arr, ids_arr, augment=(sp_name == "train"))
    finally:
        archive.close()

    return datasets


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation: Soft Segmentation, Uncertainty Quality, Pathology & Interventions
# ─────────────────────────────────────────────────────────────────────────────

def _multilabel_ece_brier(
    tp: np.ndarray,
    ty: np.ndarray,
    n_bins: int = 15,
) -> Tuple[float, float]:
    """Mean per-class 15-bin ECE and Brier score across multi-label predictions."""
    tp_f = np.asarray(tp, dtype=np.float64)
    ty_f = np.asarray(ty, dtype=np.float64)
    eces = [expected_calibration_error(tp_f[:, c], ty_f[:, c], n_bins=n_bins) for c in range(tp_f.shape[1])]
    macro_ece = float(np.mean(eces)) if eces else float("nan")
    macro_brier = float(np.mean((tp_f - ty_f) ** 2))
    return macro_ece, macro_brier


@torch.no_grad()
def evaluate_dense(
    model: nn.Module,
    loader: DataLoader,
    device: str,
    concept_names: List[str],
    use_bf16: bool = True,
    intervene_all: bool = False,
    compute_bootstrap: bool = False,
) -> Dict[str, float]:
    """Evaluate per-timestamp soft segmentation + 5-class cardiac pathology metrics."""
    model.eval()
    mtype = getattr(model, "model_type", "dense_cem")
    has_seg = (mtype != "dense_blackbox")
    K = len(concept_names)
    amp_en = use_bf16 and device.startswith("cuda") and torch.cuda.is_bf16_supported()

    soft_inter = np.zeros(K, dtype=np.float64)
    soft_denom = np.zeros(K, dtype=np.float64)
    hard_inter = np.zeros(K, dtype=np.float64)
    hard_denom = np.zeros(K, dtype=np.float64)
    mae_sum    = np.zeros(K, dtype=np.float64)
    kl_sum     = 0.0
    argmax_ok  = 0
    n_ts_total = 0

    task_probs_l, task_targs_l = [], []

    for batch in loader:
        x   = batch["signal"].to(device, non_blocking=True)
        seg = batch["seg"].to(device, non_blocking=True)
        y   = batch["pathology"]

        with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=amp_en):
            if intervene_all:
                mask = torch.ones((x.size(0), 1, x.size(-1)), device=device)
                out = model(x, intervene_labels=seg, intervene_mask=mask)
            else:
                out = model(x)

        task_probs_l.append(torch.sigmoid(out["task_logits"].float()).cpu().numpy())
        task_targs_l.append(y.numpy())

        if has_seg:
            p = out["concept_probs"].float().cpu().numpy()   # (B, K, T)
            t = seg.float().cpu().numpy()                    # (B, K, T)

            soft_inter += (p * t).sum(axis=(0, 2))
            soft_denom += p.sum(axis=(0, 2)) + t.sum(axis=(0, 2))
            mae_sum    += np.abs(p - t).sum(axis=(0, 2))

            p_cls = p.argmax(axis=1)                         # (B, T)
            t_cls = t.argmax(axis=1)                         # (B, T)
            argmax_ok += int((p_cls == t_cls).sum())
            n_ts_total += p_cls.size
            for k in range(K):
                pk = (p_cls == k)
                tk = (t_cls == k)
                hard_inter[k] += int((pk & tk).sum())
                hard_denom[k] += int(pk.sum() + tk.sum())

            p_safe = np.clip(p, 1e-6, 1.0)
            t_safe = np.clip(t, 1e-6, 1.0)
            kl_sum += float((t * (np.log(t_safe) - np.log(p_safe))).sum())

    metrics: Dict[str, float] = {}
    if has_seg:
        metrics["seg/argmax_acc"] = float(argmax_ok / max(1, n_ts_total))
        metrics["seg/mean_kl"]    = float(kl_sum / max(1, n_ts_total))
        metrics["seg/mean_mae"]   = float(mae_sum.sum() / max(1, n_ts_total * K))
        soft_dices, hard_dices = [], []
        for k, name in enumerate(concept_names):
            sd = float((2.0 * soft_inter[k] + 1e-6) / (soft_denom[k] + 1e-6))
            hd = float((2.0 * hard_inter[k] + 1e-6) / (hard_denom[k] + 1e-6))
            metrics[f"seg/{name}/soft_dice"] = sd
            metrics[f"seg/{name}/hard_dice"] = hd
            metrics[f"seg/{name}/mae"]       = float(mae_sum[k] / max(1, n_ts_total))
            soft_dices.append(sd)
            hard_dices.append(hd)
        metrics["seg/macro_soft_dice"] = float(np.mean(soft_dices))
        metrics["seg/macro_hard_dice"] = float(np.mean(hard_dices))

    tp = np.concatenate(task_probs_l, axis=0)
    ty = np.concatenate(task_targs_l, axis=0)
    aurocs = []
    for idx, cname in enumerate(SUPERCLASS_NAMES):
        if len(np.unique(ty[:, idx])) >= 2:
            v = float(roc_auc_score(ty[:, idx], tp[:, idx]))
            metrics[f"pathology/{cname}/auroc"] = v
            aurocs.append(v)
    metrics["pathology/macro_auroc"] = float(np.mean(aurocs)) if aurocs else float("nan")
    ece, brier = _multilabel_ece_brier(tp, ty)
    metrics["pathology/macro_ece"]   = ece
    metrics["pathology/macro_brier"] = brier
    if compute_bootstrap:
        metrics.update(bootstrap_multilabel_auroc(ty, tp, SUPERCLASS_NAMES))
    return metrics


@torch.no_grad()
def evaluate_dense_leakage_probe(
    model: nn.Module,
    val_loader: DataLoader,
    test_loader: DataLoader,
    device: str,
    use_bf16: bool = True,
) -> Dict[str, float]:
    """
    Temporal Concept-Leakage / Impurity Probe (analogous to OIS for dense models).

    Fits linear L2 probes on Fold-9 validation representations and evaluates on
    Fold-10 test records to separate:
      1. gt_mask_probe_auroc    : pathology signal genuinely contained in GT Zenodo soft masks
      2. pred_mask_probe_auroc  : pathology signal in predicted soft masks p(c_t | x)
      3. bottleneck_probe_auroc : pathology signal in the model's bottleneck representation
                                  (p(c_t | x) for DenseCBM; mixed embeddings ĉ_{k,t} for DenseCEM)
      4. soft_mask_excess_auroc = pred_mask - gt_mask  (soft-probability leakage)
      5. bottleneck_excess_auroc = bottleneck - gt_mask (total side-channel beyond GT masks)
    """
    if not hasattr(model, "get_bottleneck_summary"):
        return {}

    model.eval()
    amp_en = use_bf16 and device.startswith("cuda") and torch.cuda.is_bf16_supported()

    def _extract(loader: DataLoader):
        gt_f, pr_f, bn_f, ys = [], [], [], []
        for batch in loader:
            x   = batch["signal"].to(device, non_blocking=True)
            seg = batch["seg"].to(device, non_blocking=True)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=amp_en):
                c_probs, bn_repr = model.get_bottleneck_summary(x)
            gt_f.append(_summarize_temporal_tensor(seg).cpu().numpy())
            pr_f.append(_summarize_temporal_tensor(c_probs).cpu().numpy())
            bn_f.append(bn_repr.float().cpu().numpy())
            ys.append(batch["pathology"].numpy())
        return (
            np.concatenate(gt_f, axis=0),
            np.concatenate(pr_f, axis=0),
            np.concatenate(bn_f, axis=0),
            np.concatenate(ys, axis=0),
        )

    gt_v, pr_v, bn_v, y_v = _extract(val_loader)
    gt_t, pr_t, bn_t, y_t = _extract(test_loader)

    def _probe_macro_auroc(x_tr: np.ndarray, y_tr: np.ndarray, x_te: np.ndarray, y_te: np.ndarray) -> float:
        aucs = []
        for c in range(y_tr.shape[1]):
            if len(np.unique(y_tr[:, c])) < 2 or len(np.unique(y_te[:, c])) < 2:
                continue
            clf = make_pipeline(
                StandardScaler(),
                LogisticRegression(C=0.5, max_iter=500, solver="lbfgs"),
            )
            clf.fit(x_tr, y_tr[:, c].astype(int))
            scores = clf.predict_proba(x_te)[:, 1]
            aucs.append(float(roc_auc_score(y_te[:, c], scores)))
        return float(np.mean(aucs)) if aucs else float("nan")

    gt_auc = _probe_macro_auroc(gt_v, y_v, gt_t, y_t)
    pr_auc = _probe_macro_auroc(pr_v, y_v, pr_t, y_t)
    bn_auc = _probe_macro_auroc(bn_v, y_v, bn_t, y_t)
    return {
        "leakage/gt_mask_probe_auroc":     gt_auc,
        "leakage/pred_mask_probe_auroc":   pr_auc,
        "leakage/bottleneck_probe_auroc":  bn_auc,
        "leakage/soft_mask_excess_auroc":  pr_auc - gt_auc,
        "leakage/bottleneck_excess_auroc": bn_auc - gt_auc,
    }


@torch.no_grad()
def evaluate_dense_uncertainty_and_counterfactuals(
    model: nn.Module,
    loader: DataLoader,
    device: str,
    concept_names: List[str],
    max_records: int = 300,
    n_mc_epistemic: int = 8,
    n_mc_aleatoric: int = 12,
) -> Dict[str, float]:
    """
    Evaluate:
      1. Per-timestamp uncertainty error-detection AUROC (higher std → segmentation error?)
      2. Boundary vs. interior uncertainty ratio (are wave onsets/offsets more uncertain?)
      3. Counterfactual mask intervention shift (zeroing out QRS/ST/T/P wave masks → task shift)
    """
    model.eval()
    errs_l, unc_l, bnd_unc_l, int_unc_l = [], [], [], []
    cf_shifts: Dict[str, List[float]] = {n: [] for n in concept_names if n != "TP"}
    seen = 0

    for batch in loader:
        if seen >= max_records:
            break
        x   = batch["signal"].to(device, non_blocking=True)
        seg = batch["seg"].to(device, non_blocking=True)
        B   = x.size(0)
        if seen + B > max_records:
            take = max_records - seen
            x, seg = x[:take], seg[:take]
            B = take
        seen += B

        unc = model.predict_with_uncertainty(x, n_mc_epistemic=n_mc_epistemic, n_mc_aleatoric=n_mc_aleatoric)
        p_mean = unc["concept_probs_mean"].cpu().numpy()   # (B, K, T)
        p_std  = unc["concept_total_std"].mean(dim=1).cpu().numpy()  # (B, T)
        t_np   = seg.cpu().numpy()

        pred_cls = p_mean.argmax(axis=1)
        true_cls = t_np.argmax(axis=1)
        errs_l.append((pred_cls != true_cls).astype(np.int32).ravel())
        unc_l.append(p_std.ravel())

        # Boundary = timestamp where ground-truth argmax class changes from t-1 to t
        is_boundary = np.zeros_like(true_cls, dtype=bool)
        is_boundary[:, 1:] = (true_cls[:, 1:] != true_cls[:, :-1])
        is_boundary[:, :-1] |= (true_cls[:, :-1] != true_cls[:, 1:])
        if np.any(is_boundary) and np.any(~is_boundary):
            bnd_unc_l.append(float(p_std[is_boundary].mean()))
            int_unc_l.append(float(p_std[~is_boundary].mean()))

        # Counterfactual mask ablation: force concept k → 0 everywhere and measure pathology shift
        if hasattr(model, "intervene"):
            base_out = model(x)
            base_tp = torch.sigmoid(base_out["task_logits"].float())
            for k, cname in enumerate(concept_names):
                if cname == "TP":
                    continue
                ablate_labels = base_out["concept_probs"].clone()
                ablate_labels[:, k, :] = 0.0
                mask_k = torch.zeros_like(ablate_labels)
                mask_k[:, k, :] = 1.0
                cf_out = model.intervene(x, ablate_labels, mask_k)
                cf_tp = torch.sigmoid(cf_out["task_logits"].float())
                cf_shifts[cname].append(float((base_tp - cf_tp).abs().mean().item()))

    errs = np.concatenate(errs_l)
    uncs = np.concatenate(unc_l)
    out: Dict[str, float] = {
        "uncertainty/seg_error_detection_auroc": float(roc_auc_score(errs, uncs)) if len(np.unique(errs)) >= 2 else float("nan"),
        "uncertainty/mean_per_timestamp_std":    float(uncs.mean()),
    }
    if bnd_unc_l and int_unc_l:
        bnd_m, int_m = float(np.mean(bnd_unc_l)), float(np.mean(int_unc_l))
        out["uncertainty/boundary_std"]          = bnd_m
        out["uncertainty/interior_std"]          = int_m
        out["uncertainty/boundary_vs_interior"]  = bnd_m / max(int_m, 1e-8)
    for cname, vals in cf_shifts.items():
        if vals:
            out[f"counterfactual/ablate_{cname}_task_shift"] = float(np.mean(vals))
    return out


@torch.no_grad()
def evaluate_dense_task_uncertainty(
    model: nn.Module,
    loader: DataLoader,
    device: str,
    n_mc_epistemic: int = 16,
    use_bf16: bool = True,
) -> Dict[str, float]:
    """
    Evaluate downstream 5-class pathology calibration (MC ECE & Brier) and
    epistemic error-detection AUROC via batched MC-Dropout across the full loader.
    Works identically for DenseBlackBox, DenseMultiTask, DenseCBM, and DenseCEM.
    """
    model.eval()
    tp_mean_l, tp_std_l, ty_l = [], [], []
    max_forward = 512 if getattr(model, "num_concepts", 6) <= 6 else 192
    sub_bs = max(1, max_forward // max(1, n_mc_epistemic))

    for batch in loader:
        x = batch["signal"].to(device, non_blocking=True)
        y = batch["pathology"].numpy()
        for start in range(0, x.size(0), sub_bs):
            xb = x[start : start + sub_bs]
            unc = model.predict_with_uncertainty(
                xb, n_mc_epistemic=n_mc_epistemic, n_mc_aleatoric=1, use_bf16=use_bf16
            )
            tp_mean_l.append(unc["task_probs_mean"].float().cpu().numpy())
            tp_std_l.append(unc["task_probs_std"].float().cpu().numpy())
        ty_l.append(y)

    tp_mean = np.concatenate(tp_mean_l, axis=0)
    tp_std  = np.concatenate(tp_std_l, axis=0)
    ty      = np.concatenate(ty_l, axis=0)

    mc_ece, mc_brier = _multilabel_ece_brier(tp_mean, ty)
    aurocs = [
        float(roc_auc_score(ty[:, c], tp_mean[:, c]))
        for c in range(ty.shape[1])
        if len(np.unique(ty[:, c])) >= 2
    ]

    err_det_classes = []
    for c in range(ty.shape[1]):
        err_c = ((tp_mean[:, c] >= 0.5).astype(np.int32) != ty[:, c].astype(np.int32))
        if len(np.unique(err_c)) >= 2:
            err_det_classes.append(float(roc_auc_score(err_c, tp_std[:, c])))

    pooled_err = ((tp_mean >= 0.5).astype(np.int32) != ty.astype(np.int32)).ravel()
    pooled_std = tp_std.ravel()
    pooled_auc = (
        float(roc_auc_score(pooled_err, pooled_std))
        if len(np.unique(pooled_err)) >= 2
        else float("nan")
    )

    return {
        "pathology/mc_macro_auroc": float(np.mean(aurocs)) if aurocs else float("nan"),
        "pathology/mc_ece": mc_ece,
        "pathology/mc_brier": mc_brier,
        "uncertainty/task_error_detection_auroc": (
            float(np.mean(err_det_classes)) if err_det_classes else pooled_auc
        ),
        "uncertainty/task_error_detection_auroc_pooled": pooled_auc,
        "uncertainty/task_mean_std": float(tp_std.mean()),
    }


# ─────────────────────────────────────────────────────────────────────────────
# GPU-Optimized Training Loop
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Train Per-Timestamp Dense Soft-Seg CBM / CEM / Baselines")
    parser.add_argument("--config",             default="config.yaml")
    parser.add_argument(
        "--model-type", default="dense_cem",
        choices=["dense_cbm", "dense_cem", "dense_blackbox", "dense_multitask"],
    )
    parser.add_argument("--concept-mode",       default="wave6", choices=["wave6", "fiducial24"])
    parser.add_argument("--run-dir",            required=True)
    parser.add_argument("--epochs",             type=int,   default=25)
    parser.add_argument("--batch-size",         type=int,   default=128)
    parser.add_argument("--num-workers",        type=int,   default=4)
    parser.add_argument("--lr",                 type=float, default=1.5e-3)
    parser.add_argument("--base-channels",      type=int,   default=64)
    parser.add_argument("--emb-size",           type=int,   default=16)
    parser.add_argument("--rand-int-prob",      type=float, default=0.25)
    parser.add_argument("--span-len",           type=int,   default=50)
    parser.add_argument("--detach-concepts",    action="store_true",
                        help="Stop task gradients into soft masks (leakage-free DenseCBM)")
    parser.add_argument("--no-heteroscedastic", action="store_true",
                        help="Disable heteroscedastic MC soft-CE (ablation)")
    parser.add_argument("--lambda-seg",         type=float, default=2.0)
    parser.add_argument("--lambda-dice",        type=float, default=0.5)
    parser.add_argument("--lambda-task",        type=float, default=1.0)
    parser.add_argument("--seed",               type=int,   default=42)
    parser.add_argument("--no-bf16",            action="store_true")
    args = parser.parse_args()

    set_seed(args.seed)

    # Enable full Ampere / Ada Tensor Core & cuDNN acceleration
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True

    with open(args.config) as f:
        base_cfg = yaml.safe_load(f)
    data_path, _ = resolve_data_paths(base_cfg)
    raw_ss = base_cfg.get("data", {}).get("soft_seg_path", "ptbxl_segmentations_8bit.zip")
    data_dir = base_cfg.get("data", {}).get("data_dir", str(data_path.parent))
    soft_seg_path = Path(raw_ss) if Path(raw_ss).is_absolute() else (Path(data_dir) / raw_ss)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_bf16 = (not args.no_bf16) and torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    concept_names = get_dense_concept_names(args.concept_mode)
    has_seg = (args.model_type != "dense_blackbox")
    is_bottleneck = args.model_type in ("dense_cbm", "dense_cem")

    run_dir = Path(args.run_dir)
    (run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    run_cfg = {
        "data": {
            "data_dir": str(data_dir),
            "data_path": str(data_path),
            "soft_seg_path": str(soft_seg_path),
            "sampling_rate": 100,
            "t_fixed": 1000,
            "concept_mode": args.concept_mode,
        },
        "model": {
            "type": args.model_type,
            "in_channels": 12,
            "base_channels": args.base_channels,
            "num_concepts": len(concept_names),
            "emb_size": args.emb_size,
            "num_classes": NUM_SUPERCLASSES,
            "dropout": 0.15,
            "mc_dropout_rate": 0.1,
            "rand_int_prob": args.rand_int_prob,
            "span_len": args.span_len,
            "detach_concepts": args.detach_concepts,
        },
        "training": {
            "seed": args.seed,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "lambda_seg": args.lambda_seg,
            "lambda_dice": args.lambda_dice,
            "lambda_task": args.lambda_task,
            "use_heteroscedastic": not args.no_heteroscedastic,
            "use_bf16": use_bf16,
        },
        "concept_names": concept_names,
        "superclass_names": SUPERCLASS_NAMES,
    }
    (run_dir / "config.yaml").write_text(yaml.dump(run_cfg, sort_keys=False))

    cache_root = Path(base_cfg.get("training", {}).get("checkpoint_dir", "checkpoints")) / "dense_softseg_cache"
    splits = load_or_build_dense_datasets(
        data_path=data_path,
        soft_seg_path=soft_seg_path,
        cache_root=cache_root,
        concept_mode=args.concept_mode,
    )
    common = dict(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=args.num_workers > 0,
    )
    train_loader = DataLoader(splits["train"], shuffle=True,  drop_last=True, **common)
    val_loader   = DataLoader(splits["val"],   shuffle=False, **common)
    test_loader  = DataLoader(splits["test"],  shuffle=False, **common)

    model = build_dense_model(run_cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    logger.info(
        f"Training {args.model_type.upper()} ({args.concept_mode}, K={len(concept_names)}, seed={args.seed}) | "
        f"Params={n_params:,} | Device={device} | BF16={use_bf16} | BS={args.batch_size} | "
        f"Detach={args.detach_concepts} | Hetero={not args.no_heteroscedastic}"
    )

    criterion = DenseSoftSegLoss(
        lambda_seg=args.lambda_seg,
        lambda_dice=args.lambda_dice,
        lambda_task=args.lambda_task,
        use_heteroscedastic=not args.no_heteroscedastic,
        compute_seg=has_seg,
    )
    opt_kwargs = dict(lr=args.lr, weight_decay=1e-4)
    if device.startswith("cuda"):
        opt_kwargs["fused"] = True
    optimizer = AdamW(model.parameters(), **opt_kwargs)
    scheduler = CosineAnnealingLR(optimizer, T_max=max(1, args.epochs), eta_min=1e-5)

    best_score = -1.0
    history = []

    for epoch in range(1, args.epochs + 1):
        t0 = time.perf_counter()
        model.train()
        sums = {"total": 0.0, "seg_ce": 0.0, "seg_dice": 0.0, "task": 0.0}
        n_b = 0

        for batch in train_loader:
            x   = batch["signal"].to(device, non_blocking=True)
            seg = batch["seg"].to(device, non_blocking=True)
            y   = batch["pathology"].to(device, non_blocking=True)

            with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=use_bf16):
                out = model(x, concept_targets=seg)
            losses = criterion(out, seg, y)

            optimizer.zero_grad(set_to_none=True)
            losses["total"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            for k in sums:
                sums[k] += float(losses[k].item())
            n_b += 1

        scheduler.step()
        val_m = evaluate_dense(model, val_loader, device, concept_names, use_bf16=use_bf16)
        dt = time.perf_counter() - t0
        if has_seg:
            comp_score = 0.6 * val_m["seg/macro_soft_dice"] + 0.4 * val_m["pathology/macro_auroc"]
        else:
            comp_score = val_m["pathology/macro_auroc"]

        row = {
            "epoch": epoch,
            "train_loss": sums["total"] / max(1, n_b),
            "val_soft_dice": val_m.get("seg/macro_soft_dice", 0.0),
            "val_argmax_acc": val_m.get("seg/argmax_acc", 0.0),
            "val_pathology_auroc": val_m["pathology/macro_auroc"],
            "sec": dt,
        }
        history.append(row)
        logger.info(
            f"[{args.model_type.upper()}] Ep {epoch:2d}/{args.epochs} ({dt:.1f}s) | "
            f"Loss={row['train_loss']:.4f} | SoftDice={row['val_soft_dice']:.4f} | "
            f"Acc={row['val_argmax_acc']:.4f} | PathAUROC={row['val_pathology_auroc']:.4f}"
        )

        if comp_score > best_score:
            best_score = comp_score
            torch.save(
                {"epoch": epoch, "val_metrics": val_m, "model": model.state_dict()},
                run_dir / "checkpoints" / "best_model.pt",
            )

    (run_dir / "history.json").write_text(json.dumps(history, indent=2))

    # ── Final Test Evaluation (Soft Seg + Pathology + Bootstrap CI + Interventions + Uncertainty + Leakage) ──
    state = torch.load(run_dir / "checkpoints" / "best_model.pt", map_location=device, weights_only=False)
    model.load_state_dict(state["model"])
    test_m = evaluate_dense(
        model, test_loader, device, concept_names,
        use_bf16=use_bf16, intervene_all=False, compute_bootstrap=True,
    )
    if is_bottleneck:
        int_m = evaluate_dense(model, test_loader, device, concept_names, use_bf16=use_bf16, intervene_all=True)
        test_m["pathology/intervened_macro_auroc"] = int_m["pathology/macro_auroc"]
        test_m["pathology/intervention_gain"]      = int_m["pathology/macro_auroc"] - test_m["pathology/macro_auroc"]

        leak_m = evaluate_dense_leakage_probe(model, val_loader, test_loader, device, use_bf16=use_bf16)
        test_m.update(leak_m)

    if has_seg:
        unc_m = evaluate_dense_uncertainty_and_counterfactuals(model, test_loader, device, concept_names)
        test_m.update(unc_m)

    task_unc_m = evaluate_dense_task_uncertainty(
        model, test_loader, device, n_mc_epistemic=16, use_bf16=use_bf16
    )
    test_m.update(task_unc_m)

    (run_dir / "eval_test_metrics.json").write_text(json.dumps(test_m, indent=2))
    logger.info(f"Final Test Metrics ({args.model_type}):\n" + json.dumps(test_m, indent=2))


if __name__ == "__main__":
    main()
