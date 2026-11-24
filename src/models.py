"""
models.py — Model components for the ECG Heartbeat Concept Bottleneck & Embedding Models.

Supported architectures (via `model.type` in config.yaml):
  1. "cbm"      : TemporalEncoder → ECGConceptLayer (μ_k, log σ²_k) → UncertaintyClassifier
  2. "cem"      : TemporalEncoder → ConceptEmbeddingLayer (c⁺_k, c⁻_k, μ_k, log σ²_k) → CEMClassifier
                  (Concept Embedding Model, Zarlenga et al., NeurIPS 2022, with RandInt & uncertainty)
  3. "blackbox" : TemporalEncoder → Linear classifier (end-to-end baseline without concept bottleneck)

All architectures expose a unified interface:
  - forward(x, intervene_labels=None, intervene_mask=None) -> (task_prob, c_probs, c_log_vars)
  - forward_with_logits(x, concept_labels=None, concept_weights=None, ...) -> (task_prob, c_probs, c_log_vars, c_logits)
  - intervene(x, concept_labels, intervene_mask) -> task_prob
  - predict_with_uncertainty(x, n_mc_epistemic=20, n_mc_aleatoric=50) -> dict
  - get_concept_importance(concept_names=None) -> dict
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Dict, Iterator, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


CONCEPT_NAMES: List[str] = ["P", "Q", "R", "S", "T", "U", "PR", "QRS", "ST", "QT"]

LOG_VAR_MIN: float = -10.0
LOG_VAR_MAX: float = 6.0


# ─────────────────────────────────────────────────────────────────────────────
# MC-Dropout utility (keeps BatchNorm in eval mode!)
# ─────────────────────────────────────────────────────────────────────────────

@contextmanager
def _mc_dropout_context(model: nn.Module) -> Iterator[None]:
    """
    Enable Dropout modules for stochastic forward passes while keeping all
    other layers (critically BatchNorm1d) in eval mode so running statistics
    are neither mutated nor replaced by batch statistics.
    """
    prev_states = [(m, m.training) for m in model.modules()]
    try:
        model.eval()
        for m in model.modules():
            if isinstance(m, (nn.Dropout, nn.Dropout1d, nn.Dropout2d)):
                m.train(True)
        yield
    finally:
        for m, state in prev_states:
            m.train(state)


def apply_concept_intervention(
    c_probs: torch.Tensor,
    c_log_vars: torch.Tensor,
    intervene_labels: Optional[torch.Tensor] = None,
    intervene_mask: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Replace predicted concept probabilities with ground-truth values where
    `intervene_mask` is 1, and collapse corresponding concept uncertainty to ~0.
    """
    if intervene_labels is None or intervene_mask is None:
        return c_probs, c_log_vars
    mask_bool = intervene_mask.bool()
    mask = mask_bool.float()
    c_probs_int = (1.0 - mask) * c_probs + mask * intervene_labels.float()
    c_log_vars_int = torch.where(
        mask_bool, torch.full_like(c_log_vars, -20.0), c_log_vars
    )
    return c_probs_int, c_log_vars_int


# ─────────────────────────────────────────────────────────────────────────────
# Encoder
# ─────────────────────────────────────────────────────────────────────────────

class _ResBlock1D(nn.Module):
    """1D Residual block with optional stride-based downsampling."""

    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 11, stride: int = 1) -> None:
        super().__init__()
        pad = kernel_size // 2
        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel_size, stride=stride, padding=pad, bias=False)
        self.bn1   = nn.BatchNorm1d(out_ch)
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size, stride=1, padding=pad, bias=False)
        self.bn2   = nn.BatchNorm1d(out_ch)
        self.skip  = (
            nn.Sequential(nn.Conv1d(in_ch, out_ch, 1, stride=stride, bias=False), nn.BatchNorm1d(out_ch))
            if stride != 1 or in_ch != out_ch else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(self.conv1(x)))
        return F.relu(self.bn2(self.conv2(out)) + self.skip(x))


class TemporalEncoder(nn.Module):
    """
    1D ResNet encoder: (B, C, T) → (B, latent_dim).

    Default: in_channels=12, base_channels=64 → latent_dim=512.
    Architecture: stem(k=15) → 4 ResBlocks with progressive channel doubling.
    """

    def __init__(self, in_channels: int = 12, base_channels: int = 64, dropout: float = 0.2) -> None:
        super().__init__()
        c = base_channels
        self.stem   = nn.Sequential(
            nn.Conv1d(in_channels, c, kernel_size=15, padding=7, bias=False),
            nn.BatchNorm1d(c), nn.ReLU(inplace=True),
        )
        self.layer1 = _ResBlock1D(c,     c,     kernel_size=15, stride=1)
        self.layer2 = _ResBlock1D(c,     c * 2, kernel_size=11, stride=2)
        self.layer3 = _ResBlock1D(c * 2, c * 4, kernel_size=7,  stride=2)
        self.layer4 = _ResBlock1D(c * 4, c * 8, kernel_size=5,  stride=2)
        self.dropout = nn.Dropout(p=dropout)
        self.out_dim: int = c * 8

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, C, T) → z: (B, out_dim)"""
        out = self.stem(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        return self.dropout(out.mean(dim=-1))   # Global Average Pooling


# ─────────────────────────────────────────────────────────────────────────────
# Concept Bottleneck Layer & Classifier (CBM)
# ─────────────────────────────────────────────────────────────────────────────

class ECGConceptLayer(nn.Module):
    """
    K concept heads via a fused linear projection: z → (μ_k, log_σ²_k).
    Log-variances are clamped to [LOG_VAR_MIN, LOG_VAR_MAX] for stability.
    """

    def __init__(self, input_dim: int = 512, num_concepts: int = 10) -> None:
        super().__init__()
        self.num_concepts = num_concepts
        self.proj = nn.Linear(input_dim, num_concepts * 2)

    def _split(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """z: (B, D) → logits (B, K), clamped log_vars (B, K)"""
        out = self.proj(z).view(z.size(0), self.num_concepts, 2)
        logits   = out[:, :, 0]
        log_vars = out[:, :, 1].clamp(LOG_VAR_MIN, LOG_VAR_MAX)
        return logits, log_vars

    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns concept_probs (B, K) and concept_log_vars (B, K)."""
        logits, log_vars = self._split(z)
        return torch.sigmoid(logits), log_vars

    def forward_all(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Single-pass: returns (logits, concept_probs, log_vars) — all (B, K)."""
        logits, log_vars = self._split(z)
        return logits, torch.sigmoid(logits), log_vars

    def get_logits(self, z: torch.Tensor) -> torch.Tensor:
        return self._split(z)[0]

    def sample_concepts(self, z: torch.Tensor, n_samples: int = 10) -> torch.Tensor:
        """Reparameterisation sampling → (B, n_samples, K)."""
        logits, log_vars = self._split(z)
        stds = torch.exp(0.5 * log_vars)
        eps  = torch.randn(z.size(0), n_samples, self.num_concepts, device=z.device)
        return torch.sigmoid(logits.unsqueeze(1) + eps * stds.unsqueeze(1))


class UncertaintyClassifier(nn.Module):
    """
    Linear task head over concept probabilities (and optionally concept variances).

    When `detach_uncertainty=True` (default), `concept_log_vars` is detached before
    entering the task classifier so task BCE cannot use log-variances as an
    unsupervised side-channel bypassing the concept bottleneck.
    """

    def __init__(
        self,
        num_concepts: int = 10,
        use_uncertainty: bool = True,
        detach_uncertainty: bool = True,
    ) -> None:
        super().__init__()
        self.num_concepts       = num_concepts
        self.use_uncertainty    = use_uncertainty
        self.detach_uncertainty = detach_uncertainty
        in_dim = num_concepts * 2 if use_uncertainty else num_concepts
        self.linear = nn.Linear(in_dim, 1, bias=True)

    def _build_features(self, concept_probs: torch.Tensor, concept_log_vars: torch.Tensor) -> torch.Tensor:
        if not self.use_uncertainty:
            return concept_probs
        lv = concept_log_vars.detach() if self.detach_uncertainty else concept_log_vars
        return torch.cat([concept_probs, torch.exp(lv.clamp( -25.0, LOG_VAR_MAX))], dim=-1)

    def forward(self, concept_probs: torch.Tensor, concept_log_vars: torch.Tensor) -> torch.Tensor:
        """Returns task_prob (B, 1)."""
        return torch.sigmoid(self.linear(self._build_features(concept_probs, concept_log_vars)))

    @torch.no_grad()
    def mc_task_uncertainty(
        self,
        concept_probs: torch.Tensor,
        concept_log_vars: torch.Tensor,
        n_mc: int = 50,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Monte Carlo sampling through concept uncertainty. Returns (mean, std, all) per sample."""
        clamped_lv = concept_log_vars.clamp(-25.0, LOG_VAR_MAX)
        stds   = torch.exp(0.5 * clamped_lv)
        logits = torch.logit(concept_probs.clamp(1e-6, 1 - 1e-6))
        eps    = torch.randn(concept_probs.size(0), n_mc, self.num_concepts, device=concept_probs.device)
        sampled_probs = torch.sigmoid(logits.unsqueeze(1) + eps * stds.unsqueeze(1))
        if self.use_uncertainty:
            vars_exp = torch.exp(clamped_lv).unsqueeze(1).expand_as(sampled_probs)
            features = torch.cat([sampled_probs, vars_exp], dim=-1)
        else:
            features = sampled_probs
        task_probs = torch.sigmoid(self.linear(features)).squeeze(-1)   # (B, n_mc)
        return task_probs.mean(1), task_probs.std(1, unbiased=False), task_probs

    def get_weights_dict(self, concept_names: Optional[List[str]] = None) -> dict:
        names = concept_names or CONCEPT_NAMES[: self.num_concepts]
        w = self.linear.weight.data[0].cpu().tolist()
        K = self.num_concepts
        return {
            name: {
                "prob_weight":   w[i],
                "uncert_weight": w[i + K] if self.use_uncertainty else 0.0,
            }
            for i, name in enumerate(names)
        }


# ─────────────────────────────────────────────────────────────────────────────
# Concept Embedding Layer & Classifier (CEM — Zarlenga et al., NeurIPS 2022)
# ─────────────────────────────────────────────────────────────────────────────

class ConceptEmbeddingLayer(nn.Module):
    """
    Concept Embedding Model (CEM) bottleneck layer with heteroscedastic uncertainty
    and RandInt (random training-time concept interventions).

    For each concept k ∈ {1..K}:
      - Positive embedding: c⁺_k = LeakyReLU(W⁺_k z + b⁺_k) ∈ R^m
      - Negative embedding: c⁻_k = LeakyReLU(W⁻_k z + b⁻_k) ∈ R^m
      - Concept score:      (μ_k, log σ²_k) = Scorer([c⁺_k, c⁻_k]), p̂_k = σ(μ_k)
      - Bottleneck mixture: ĉ_k = p̃_k c⁺_k + (1 - p̃_k) c⁻_k ∈ R^m
        where p̃_k uses RandInt during training or explicit interventions at test time.
    """

    def __init__(
        self,
        input_dim: int = 512,
        num_concepts: int = 10,
        emb_size: int = 16,
        rand_int_prob: float = 0.25,
    ) -> None:
        super().__init__()
        self.num_concepts  = num_concepts
        self.emb_size      = emb_size
        self.rand_int_prob = rand_int_prob

        self.pos_proj = nn.Sequential(
            nn.Linear(input_dim, num_concepts * emb_size),
            nn.LeakyReLU(0.01),
        )
        self.neg_proj = nn.Sequential(
            nn.Linear(input_dim, num_concepts * emb_size),
            nn.LeakyReLU(0.01),
        )
        # Shared scoring head aligns active/inactive embedding semantics across concepts
        self.scorer = nn.Linear(emb_size * 2, 2)

    def compute_embeddings_and_scores(
        self, z: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns (c_pos, c_neg, logits, log_vars) with shapes (B,K,m), (B,K,m), (B,K), (B,K)."""
        B = z.size(0)
        c_pos = self.pos_proj(z).view(B, self.num_concepts, self.emb_size)
        c_neg = self.neg_proj(z).view(B, self.num_concepts, self.emb_size)
        scores = self.scorer(torch.cat([c_pos, c_neg], dim=-1))   # (B, K, 2)
        logits   = scores[:, :, 0]
        log_vars = scores[:, :, 1].clamp(LOG_VAR_MIN, LOG_VAR_MAX)
        return c_pos, c_neg, logits, log_vars

    @staticmethod
    def mix(c_pos: torch.Tensor, c_neg: torch.Tensor, probs: torch.Tensor) -> torch.Tensor:
        """Convex combination: probs (..., K) → (..., K, m)."""
        p = probs.unsqueeze(-1)
        return p * c_pos + (1.0 - p) * c_neg

    def forward_all(
        self,
        z: torch.Tensor,
        concept_labels: Optional[torch.Tensor] = None,
        concept_weights: Optional[torch.Tensor] = None,
        intervene_labels: Optional[torch.Tensor] = None,
        intervene_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns (mixed_embs (B,K,m), c_logits (B,K), c_probs (B,K), c_log_vars (B,K)).
        `c_logits` and `c_probs` are always the raw model predictions (for concept loss),
        whereas `mixed_embs` incorporates RandInt (in training) or test-time interventions.
        """
        c_pos, c_neg, logits, log_vars = self.compute_embeddings_and_scores(z)
        c_probs = torch.sigmoid(logits)
        mix_probs = c_probs

        # Explicit test-time / independent-mode intervention takes precedence
        if intervene_labels is not None and intervene_mask is not None:
            mask = intervene_mask.bool()
            mix_probs = torch.where(mask, intervene_labels.float(), mix_probs)
            log_vars = torch.where(mask, torch.full_like(log_vars, -20.0), log_vars)
        elif self.training and self.rand_int_prob > 0.0 and concept_labels is not None:
            bern = torch.rand_like(c_probs) < self.rand_int_prob
            if concept_weights is not None:
                bern = bern & (concept_weights > 0)
            mix_probs = torch.where(bern, concept_labels.float(), mix_probs)

        mixed_embs = self.mix(c_pos, c_neg, mix_probs)
        return mixed_embs, logits, c_probs, log_vars


class CEMClassifier(nn.Module):
    """Linear task head over concatenated concept embeddings (B, K*m) → (B, 1)."""

    def __init__(self, num_concepts: int = 10, emb_size: int = 16) -> None:
        super().__init__()
        self.num_concepts = num_concepts
        self.emb_size     = emb_size
        self.linear       = nn.Linear(num_concepts * emb_size, 1, bias=True)

    def forward(self, concept_embs: torch.Tensor) -> torch.Tensor:
        """concept_embs: (..., K, m) → task_prob (..., 1)."""
        flat = concept_embs.flatten(start_dim=-2)
        return torch.sigmoid(self.linear(flat))

    def get_weights_dict(self, concept_names: Optional[List[str]] = None) -> dict:
        """Per-concept L2 weight norm across each concept's m-dim embedding block."""
        names = concept_names or CONCEPT_NAMES[: self.num_concepts]
        w = self.linear.weight.data[0].view(self.num_concepts, self.emb_size).cpu()
        norms = w.norm(p=2, dim=-1).tolist()
        return {
            name: {"prob_weight": norms[i], "uncert_weight": 0.0}
            for i, name in enumerate(names)
        }


# ─────────────────────────────────────────────────────────────────────────────
# Full Models (Unified Interface)
# ─────────────────────────────────────────────────────────────────────────────

class HeartbeatCBM(nn.Module):
    """Standard CBM without MC-Dropout wrapper."""

    def __init__(
        self,
        in_channels: int = 12,
        base_channels: int = 64,
        encoder_dropout: float = 0.2,
        num_concepts: int = 10,
        use_uncertainty: bool = True,
        detach_uncertainty: bool = True,
    ) -> None:
        super().__init__()
        self.encoder       = TemporalEncoder(in_channels, base_channels, encoder_dropout)
        self.concept_layer = ECGConceptLayer(self.encoder.out_dim, num_concepts)
        self.classifier    = UncertaintyClassifier(num_concepts, use_uncertainty, detach_uncertainty)

    def forward(
        self,
        x: torch.Tensor,
        intervene_labels: Optional[torch.Tensor] = None,
        intervene_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        z = self.encoder(x)
        c_probs, c_log_vars = self.concept_layer(z)
        c_probs_cls, c_lv_cls = apply_concept_intervention(
            c_probs, c_log_vars, intervene_labels, intervene_mask
        )
        return self.classifier(c_probs_cls, c_lv_cls), c_probs_cls, c_lv_cls

    @property
    def num_concepts(self) -> int:
        return self.concept_layer.num_concepts


class CBMWithUncertainty(nn.Module):
    """
    Concept Bottleneck Model with heteroscedastic concept heads and MC-Dropout
    for full epistemic + aleatoric uncertainty decomposition.
    """

    model_type: str = "cbm"

    def __init__(
        self,
        encoder: TemporalEncoder,
        concept_layer: ECGConceptLayer,
        classifier: UncertaintyClassifier,
        mc_dropout_rate: float = 0.1,
        rand_int_prob: float = 0.0,
        inactive_concepts: Optional[List[int]] = None,
    ) -> None:
        super().__init__()
        self.encoder           = encoder
        self.concept_layer     = concept_layer
        self.classifier        = classifier
        self.mc_dropout        = nn.Dropout(mc_dropout_rate)
        self.rand_int_prob     = rand_int_prob
        self.inactive_concepts = tuple(sorted(set(int(i) for i in (inactive_concepts or []))))

    def _mask_inactive(
        self,
        c_probs: torch.Tensor,
        c_log_vars: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Zero out permanently unsupervised concepts (e.g., U-wave under Zenodo) to block residual leakage."""
        if not self.inactive_concepts:
            return c_probs, c_log_vars
        active = torch.ones(c_probs.size(-1), device=c_probs.device, dtype=c_probs.dtype)
        for idx in self.inactive_concepts:
            if 0 <= idx < active.numel():
                active[idx] = 0.0
        probs_m = c_probs * active
        # Set inactive log-var very negative so exp(log_var) ≈ 0 in UncertaintyClassifier
        inactive_bool = (active == 0.0).expand_as(c_log_vars)
        lv_m = torch.where(inactive_bool, torch.full_like(c_log_vars, -20.0), c_log_vars)
        return probs_m, lv_m

    def forward(
        self,
        x: torch.Tensor,
        intervene_labels: Optional[torch.Tensor] = None,
        intervene_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        z = self.mc_dropout(self.encoder(x))
        c_probs, c_log_vars = self.concept_layer(z)
        c_probs_cls, c_lv_cls = apply_concept_intervention(
            c_probs, c_log_vars, intervene_labels, intervene_mask
        )
        c_probs_cls, c_lv_cls = self._mask_inactive(c_probs_cls, c_lv_cls)
        return self.classifier(c_probs_cls, c_lv_cls), c_probs_cls, c_lv_cls

    def forward_with_logits(
        self,
        x: torch.Tensor,
        concept_labels: Optional[torch.Tensor] = None,
        concept_weights: Optional[torch.Tensor] = None,
        intervene_labels: Optional[torch.Tensor] = None,
        intervene_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Single-pass returning (task_prob, raw_c_probs, raw_c_log_vars, raw_c_logits)."""
        z = self.mc_dropout(self.encoder(x))
        c_logits, c_probs, c_log_vars = self.concept_layer.forward_all(z)
        if intervene_labels is not None and intervene_mask is not None:
            c_probs_cls, c_lv_cls = apply_concept_intervention(
                c_probs, c_log_vars, intervene_labels, intervene_mask
            )
        elif self.training and self.rand_int_prob > 0.0 and concept_labels is not None:
            bern = torch.rand_like(c_probs) < self.rand_int_prob
            if concept_weights is not None:
                bern = bern & (concept_weights > 0)
            c_probs_cls, c_lv_cls = apply_concept_intervention(
                c_probs, c_log_vars, concept_labels, bern
            )
        else:
            c_probs_cls, c_lv_cls = c_probs, c_log_vars
        c_probs_cls, c_lv_cls = self._mask_inactive(c_probs_cls, c_lv_cls)
        return self.classifier(c_probs_cls, c_lv_cls), c_probs, c_log_vars, c_logits

    @torch.no_grad()
    def mc_task_from_z(
        self,
        z: torch.Tensor,
        n_mc: int = 50,
        intervene_labels: Optional[torch.Tensor] = None,
        intervene_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Aleatoric MC sampling from latent z, supporting partial interventions."""
        c_probs, c_log_vars = self.concept_layer(z)
        c_probs_eff, c_lv_eff = apply_concept_intervention(
            c_probs, c_log_vars, intervene_labels, intervene_mask
        )
        c_probs_eff, c_lv_eff = self._mask_inactive(c_probs_eff, c_lv_eff)
        clamped_lv = c_lv_eff.clamp(-25.0, LOG_VAR_MAX)
        stds   = torch.exp(0.5 * clamped_lv)
        logits = torch.logit(c_probs_eff.clamp(1e-6, 1.0 - 1e-6))
        eps    = torch.randn(z.size(0), n_mc, self.num_concepts, device=z.device)
        sampled_probs = torch.sigmoid(logits.unsqueeze(1) + eps * stds.unsqueeze(1))
        if intervene_labels is not None and intervene_mask is not None:
            m_exp = intervene_mask.bool().unsqueeze(1)
            sampled_probs = torch.where(m_exp, intervene_labels.float().unsqueeze(1), sampled_probs)
        if self.inactive_concepts:
            for idx in self.inactive_concepts:
                if 0 <= idx < sampled_probs.size(-1):
                    sampled_probs[:, :, idx] = 0.0
        if self.classifier.use_uncertainty:
            vars_exp = torch.exp(clamped_lv).unsqueeze(1).expand_as(sampled_probs)
            features = torch.cat([sampled_probs, vars_exp], dim=-1)
        else:
            features = sampled_probs
        task_samples = torch.sigmoid(self.classifier.linear(features)).squeeze(-1)   # (B, n_mc)
        c_var_prob = sampled_probs.var(dim=1, unbiased=False)
        return task_samples.mean(1), task_samples.std(1, unbiased=False), task_samples, c_probs_eff, c_var_prob

    @torch.no_grad()
    def get_concept_representations(self, z: torch.Tensor) -> torch.Tensor:
        """Per-concept scalar representation (B, K, 1) for leakage / impurity analysis."""
        c_probs, _ = self.concept_layer(z)
        return c_probs.unsqueeze(-1)

    @torch.no_grad()
    def intervene(
        self,
        x: torch.Tensor,
        concept_labels: torch.Tensor,
        intervene_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Run test-time concept intervention; returns intervened task_prob (B, 1)."""
        was_training = self.training
        self.eval()
        task_prob, _, _ = self.forward(x, intervene_labels=concept_labels, intervene_mask=intervene_mask)
        self.train(was_training)
        return task_prob

    @torch.no_grad()
    def predict_with_uncertainty(
        self,
        x: torch.Tensor,
        n_mc_epistemic: int = 20,
        n_mc_aleatoric: int = 50,
        intervene_labels: Optional[torch.Tensor] = None,
        intervene_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Full uncertainty decomposition via the law of total variance.
        Keeps BatchNorm layers in eval mode while enabling Dropout layers.
        """
        c_probs_runs, c_vars_runs, task_probs_runs = [], [], []

        with _mc_dropout_context(self):
            for _ in range(n_mc_epistemic):
                z = self.mc_dropout(self.encoder(x))
                _, _, t_samples, c_probs, c_var_prob = self.mc_task_from_z(
                    z, n_mc_aleatoric, intervene_labels=intervene_labels, intervene_mask=intervene_mask
                )
                c_probs_runs.append(c_probs)
                c_vars_runs.append(c_var_prob)
                task_probs_runs.append(t_samples)

        c_probs_stack = torch.stack(c_probs_runs)                   # (M, B, K)
        c_vars_stack  = torch.stack(c_vars_runs)                    # (M, B, K)
        task_stack    = torch.cat(task_probs_runs, dim=1)           # (B, M * n_mc_aleatoric)

        concept_epistemic = c_probs_stack.var(dim=0, unbiased=False)
        concept_aleatoric = c_vars_stack.mean(dim=0)

        task_mean = task_stack.mean(1)
        task_std  = task_stack.std(1, unbiased=False)
        per_means = torch.stack([t.mean(1) for t in task_probs_runs])
        per_vars  = torch.stack([t.var(1, unbiased=False) for t in task_probs_runs])
        task_aleatoric = per_vars.mean(0).sqrt()
        task_epistemic = per_means.var(0, unbiased=False).sqrt()

        return {
            "concept_probs_mean": c_probs_stack.mean(0),
            "concept_probs_std":  (concept_epistemic + concept_aleatoric).clamp(min=0.0).sqrt(),
            "concept_aleatoric":  concept_aleatoric,
            "concept_epistemic":  concept_epistemic,
            "task_prob_mean":     task_mean,
            "task_prob_std":      task_std,
            "task_aleatoric":     task_aleatoric,
            "task_epistemic":     task_epistemic,
        }

    def freeze_concept_bottleneck(self) -> None:
        for p in self.encoder.parameters():
            p.requires_grad = False
        for p in self.concept_layer.parameters():
            p.requires_grad = False

    def unfreeze_all(self) -> None:
        for p in self.parameters():
            p.requires_grad = True

    def get_concept_importance(self, concept_names: Optional[List[str]] = None) -> dict:
        return self.classifier.get_weights_dict(concept_names)

    @property
    def num_concepts(self) -> int:
        return self.concept_layer.num_concepts


class CEMWithUncertainty(nn.Module):
    """
    Concept Embedding Model (CEM) with heteroscedastic concept scoring,
    RandInt training interventions, test-time concept interventions, and MC-Dropout.
    """

    model_type: str = "cem"

    def __init__(
        self,
        encoder: TemporalEncoder,
        concept_layer: ConceptEmbeddingLayer,
        classifier: CEMClassifier,
        mc_dropout_rate: float = 0.1,
        inactive_concepts: Optional[List[int]] = None,
    ) -> None:
        super().__init__()
        self.encoder           = encoder
        self.concept_layer     = concept_layer
        self.classifier        = classifier
        self.mc_dropout        = nn.Dropout(mc_dropout_rate)
        self.inactive_concepts = tuple(sorted(set(int(i) for i in (inactive_concepts or []))))

    def _mask_inactive_embs(self, mixed_embs: torch.Tensor) -> torch.Tensor:
        if not self.inactive_concepts:
            return mixed_embs
        mask = torch.ones(mixed_embs.size(1), device=mixed_embs.device, dtype=mixed_embs.dtype)
        for idx in self.inactive_concepts:
            if 0 <= idx < mask.numel():
                mask[idx] = 0.0
        return mixed_embs * mask.view(1, -1, 1)

    def forward(
        self,
        x: torch.Tensor,
        intervene_labels: Optional[torch.Tensor] = None,
        intervene_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        z = self.mc_dropout(self.encoder(x))
        mixed_embs, _, c_probs, c_log_vars = self.concept_layer.forward_all(
            z, intervene_labels=intervene_labels, intervene_mask=intervene_mask
        )
        mixed_embs = self._mask_inactive_embs(mixed_embs)
        c_probs_out, c_lv_out = apply_concept_intervention(
            c_probs, c_log_vars, intervene_labels, intervene_mask
        )
        return self.classifier(mixed_embs), c_probs_out, c_lv_out

    def forward_with_logits(
        self,
        x: torch.Tensor,
        concept_labels: Optional[torch.Tensor] = None,
        concept_weights: Optional[torch.Tensor] = None,
        intervene_labels: Optional[torch.Tensor] = None,
        intervene_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Single-pass returning (task_prob, raw_c_probs, raw_c_log_vars, raw_c_logits)."""
        z = self.mc_dropout(self.encoder(x))
        mixed_embs, c_logits, c_probs, c_log_vars = self.concept_layer.forward_all(
            z,
            concept_labels=concept_labels,
            concept_weights=concept_weights,
            intervene_labels=intervene_labels,
            intervene_mask=intervene_mask,
        )
        mixed_embs = self._mask_inactive_embs(mixed_embs)
        return self.classifier(mixed_embs), c_probs, c_log_vars, c_logits

    @torch.no_grad()
    def intervene(
        self,
        x: torch.Tensor,
        concept_labels: torch.Tensor,
        intervene_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Swap positive/negative concept embeddings according to ground-truth intervention."""
        was_training = self.training
        self.eval()
        task_prob, _, _ = self.forward(x, intervene_labels=concept_labels, intervene_mask=intervene_mask)
        self.train(was_training)
        return task_prob

    @torch.no_grad()
    def mc_task_from_z(
        self,
        z: torch.Tensor,
        n_mc: int = 50,
        intervene_labels: Optional[torch.Tensor] = None,
        intervene_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Aleatoric MC sampling through CEM concept mixtures, supporting partial interventions."""
        c_pos, c_neg, logits, log_vars = self.concept_layer.compute_embeddings_and_scores(z)
        c_probs = torch.sigmoid(logits)
        stds    = torch.exp(0.5 * log_vars)
        eps     = torch.randn(z.size(0), n_mc, self.num_concepts, device=z.device)
        sampled_probs = torch.sigmoid(logits.unsqueeze(1) + eps * stds.unsqueeze(1))   # (B, n_mc, K)
        if intervene_labels is not None and intervene_mask is not None:
            mask_bool = intervene_mask.bool()
            c_probs = torch.where(mask_bool, intervene_labels.float(), c_probs)
            sampled_probs = torch.where(mask_bool.unsqueeze(1), intervene_labels.float().unsqueeze(1), sampled_probs)
        # Mix embeddings across MC samples: (B, n_mc, K, m)
        p_exp = sampled_probs.unsqueeze(-1)
        mixed = p_exp * c_pos.unsqueeze(1) + (1.0 - p_exp) * c_neg.unsqueeze(1)
        if self.inactive_concepts:
            mask = torch.ones(self.num_concepts, device=mixed.device, dtype=mixed.dtype)
            for idx in self.inactive_concepts:
                if 0 <= idx < mask.numel():
                    mask[idx] = 0.0
            mixed = mixed * mask.view(1, 1, -1, 1)
        task_samples = self.classifier(mixed).squeeze(-1)   # (B, n_mc)
        c_var_prob = sampled_probs.var(dim=1, unbiased=False)
        return task_samples.mean(1), task_samples.std(1, unbiased=False), task_samples, c_probs, c_var_prob

    @torch.no_grad()
    def get_concept_representations(self, z: torch.Tensor) -> torch.Tensor:
        """Per-concept embedding representation (B, K, m) for OIS leakage analysis."""
        mixed_embs, _, _, _ = self.concept_layer.forward_all(z)
        return mixed_embs

    @torch.no_grad()
    def predict_with_uncertainty(
        self,
        x: torch.Tensor,
        n_mc_epistemic: int = 20,
        n_mc_aleatoric: int = 50,
        intervene_labels: Optional[torch.Tensor] = None,
        intervene_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Full epistemic + aleatoric decomposition for CEM (supports partial interventions)."""
        c_probs_runs, c_vars_runs, task_probs_runs = [], [], []

        with _mc_dropout_context(self):
            for _ in range(n_mc_epistemic):
                z = self.mc_dropout(self.encoder(x))
                _, _, t_samples, c_probs, c_var_prob = self.mc_task_from_z(
                    z, n_mc_aleatoric, intervene_labels=intervene_labels, intervene_mask=intervene_mask
                )
                c_probs_runs.append(c_probs)
                c_vars_runs.append(c_var_prob)
                task_probs_runs.append(t_samples)

        c_probs_stack = torch.stack(c_probs_runs)
        c_vars_stack  = torch.stack(c_vars_runs)
        task_stack    = torch.cat(task_probs_runs, dim=1)

        concept_epistemic = c_probs_stack.var(dim=0, unbiased=False)
        concept_aleatoric = c_vars_stack.mean(dim=0)

        task_mean = task_stack.mean(1)
        task_std  = task_stack.std(1, unbiased=False)
        per_means = torch.stack([t.mean(1) for t in task_probs_runs])
        per_vars  = torch.stack([t.var(1, unbiased=False) for t in task_probs_runs])
        task_aleatoric = per_vars.mean(0).sqrt()
        task_epistemic = per_means.var(0, unbiased=False).sqrt()

        return {
            "concept_probs_mean": c_probs_stack.mean(0),
            "concept_probs_std":  (concept_epistemic + concept_aleatoric).clamp(min=0.0).sqrt(),
            "concept_aleatoric":  concept_aleatoric,
            "concept_epistemic":  concept_epistemic,
            "task_prob_mean":     task_mean,
            "task_prob_std":      task_std,
            "task_aleatoric":     task_aleatoric,
            "task_epistemic":     task_epistemic,
        }

    def freeze_concept_bottleneck(self) -> None:
        for p in self.encoder.parameters():
            p.requires_grad = False
        for p in self.concept_layer.parameters():
            p.requires_grad = False

    def unfreeze_all(self) -> None:
        for p in self.parameters():
            p.requires_grad = True

    def get_concept_importance(self, concept_names: Optional[List[str]] = None) -> dict:
        return self.classifier.get_weights_dict(concept_names)

    @property
    def num_concepts(self) -> int:
        return self.concept_layer.num_concepts


class BlackBoxWithUncertainty(nn.Module):
    """
    End-to-end Black-Box 1D ResNet baseline (no concept bottleneck).
    Conforms to the CBM/CEM interface so training, evaluation, and ablations work seamlessly.
    """

    model_type: str = "blackbox"

    def __init__(
        self,
        encoder: TemporalEncoder,
        num_concepts: int = 10,
        mc_dropout_rate: float = 0.1,
    ) -> None:
        super().__init__()
        self.encoder      = encoder
        self._num_concepts = num_concepts
        self.mc_dropout   = nn.Dropout(mc_dropout_rate)
        self.classifier   = nn.Linear(encoder.out_dim, 1)

    def _dummy_concepts(self, batch_size: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        zeros = torch.zeros(batch_size, self._num_concepts, device=device)
        return zeros, zeros, zeros

    def forward(
        self,
        x: torch.Tensor,
        intervene_labels: Optional[torch.Tensor] = None,
        intervene_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        z = self.mc_dropout(self.encoder(x))
        task_prob = torch.sigmoid(self.classifier(z))
        c_probs, c_log_vars, _ = self._dummy_concepts(x.size(0), x.device)
        return task_prob, c_probs, c_log_vars

    def forward_with_logits(
        self,
        x: torch.Tensor,
        concept_labels: Optional[torch.Tensor] = None,
        concept_weights: Optional[torch.Tensor] = None,
        intervene_labels: Optional[torch.Tensor] = None,
        intervene_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        z = self.mc_dropout(self.encoder(x))
        task_prob = torch.sigmoid(self.classifier(z))
        c_probs, c_log_vars, c_logits = self._dummy_concepts(x.size(0), x.device)
        return task_prob, c_probs, c_log_vars, c_logits

    @torch.no_grad()
    def intervene(
        self,
        x: torch.Tensor,
        concept_labels: torch.Tensor,
        intervene_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Black-box models cannot be intervened upon; returns unchanged prediction."""
        was_training = self.training
        self.eval()
        task_prob, _, _ = self.forward(x)
        self.train(was_training)
        return task_prob

    @torch.no_grad()
    def predict_with_uncertainty(
        self,
        x: torch.Tensor,
        n_mc_epistemic: int = 20,
        n_mc_aleatoric: int = 50,
    ) -> Dict[str, torch.Tensor]:
        task_runs = []
        with _mc_dropout_context(self):
            for _ in range(n_mc_epistemic):
                z = self.mc_dropout(self.encoder(x))
                task_runs.append(torch.sigmoid(self.classifier(z)).squeeze(-1))
        task_stack = torch.stack(task_runs, dim=0)   # (M, B)
        B = x.size(0)
        zeros_c = torch.zeros(B, self._num_concepts, device=x.device)
        return {
            "concept_probs_mean": zeros_c,
            "concept_probs_std":  zeros_c,
            "concept_aleatoric":  zeros_c,
            "concept_epistemic":  zeros_c,
            "task_prob_mean":     task_stack.mean(0),
            "task_prob_std":      task_stack.std(0, unbiased=False),
            "task_aleatoric":     torch.zeros(B, device=x.device),
            "task_epistemic":     task_stack.std(0, unbiased=False),
        }

    def freeze_concept_bottleneck(self) -> None:
        pass

    def unfreeze_all(self) -> None:
        for p in self.parameters():
            p.requires_grad = True

    def get_concept_importance(self, concept_names: Optional[List[str]] = None) -> dict:
        names = concept_names or CONCEPT_NAMES[: self._num_concepts]
        return {name: {"prob_weight": 0.0, "uncert_weight": 0.0} for name in names}

    @property
    def num_concepts(self) -> int:
        return self._num_concepts


# ─────────────────────────────────────────────────────────────────────────────
# Factory
# ─────────────────────────────────────────────────────────────────────────────

def build_model(cfg: dict) -> nn.Module:
    """Build CBMWithUncertainty, CEMWithUncertainty, or BlackBoxWithUncertainty from config."""
    model_cfg = cfg.get("model", {})
    model_type = str(model_cfg.get("type", "cbm")).lower()
    enc_cfg = model_cfg.get("encoder", {})
    cpt_cfg = model_cfg.get("concept_layer", {})
    cls_cfg = model_cfg.get("classifier", {})

    model_leads = cfg.get("data", {}).get("model_leads", None)
    model_lead  = cfg.get("data", {}).get("model_lead",  None)
    if model_leads is not None:
        in_channels = len(model_leads)
    elif model_lead is not None:
        in_channels = 1
    else:
        in_channels = enc_cfg.get("in_channels", 12)

    active_concepts = cfg.get("active_concepts", None)
    num_concepts = len(active_concepts) if active_concepts is not None else cpt_cfg.get("num_concepts", 10)
    mc_dropout_rate = float(model_cfg.get("mc_dropout_rate", 0.1))

    # Block unsupervised concepts (e.g. U-wave under Zenodo soft_seg) from acting as
    # residual task bypass channels when mask_unsupervised_concepts is enabled.
    inactive_concepts = cpt_cfg.get("inactive_concepts", None)
    if inactive_concepts is None and bool(cpt_cfg.get("mask_unsupervised_concepts", False)):
        if str(cfg.get("data", {}).get("delineator", "neurokit2")).lower() == "soft_seg":
            c_names = cfg.get("concept_names", CONCEPT_NAMES)
            if "U" in c_names:
                inactive_concepts = [c_names.index("U")]

    encoder = TemporalEncoder(
        in_channels=in_channels,
        base_channels=enc_cfg.get("base_channels", 64),
        dropout=float(enc_cfg.get("dropout", 0.2)),
    )

    if model_type == "cbm":
        concept_layer = ECGConceptLayer(encoder.out_dim, num_concepts)
        classifier = UncertaintyClassifier(
            num_concepts=num_concepts,
            use_uncertainty=bool(cls_cfg.get("use_uncertainty", True)),
            detach_uncertainty=bool(cls_cfg.get("detach_uncertainty", True)),
        )
        rand_int_prob = float(cpt_cfg.get("rand_int_prob", 0.0))
        return CBMWithUncertainty(
            encoder, concept_layer, classifier,
            mc_dropout_rate=mc_dropout_rate,
            rand_int_prob=rand_int_prob,
            inactive_concepts=inactive_concepts,
        )

    if model_type == "cem":
        emb_size = int(cpt_cfg.get("emb_size", 16))
        rand_int_prob = float(cpt_cfg.get("rand_int_prob", 0.25))
        concept_layer_cem = ConceptEmbeddingLayer(
            input_dim=encoder.out_dim,
            num_concepts=num_concepts,
            emb_size=emb_size,
            rand_int_prob=rand_int_prob,
        )
        classifier_cem = CEMClassifier(num_concepts=num_concepts, emb_size=emb_size)
        return CEMWithUncertainty(
            encoder, concept_layer_cem, classifier_cem,
            mc_dropout_rate=mc_dropout_rate,
            inactive_concepts=inactive_concepts,
        )

    if model_type == "blackbox":
        return BlackBoxWithUncertainty(encoder, num_concepts=num_concepts, mc_dropout_rate=mc_dropout_rate)

    raise ValueError(f"Unknown model.type={model_type!r}. Expected 'cbm', 'cem', or 'blackbox'.")
