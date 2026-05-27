"""
models.py — All model components for the ECG heartbeat CBM.

Architecture:
    TemporalEncoder  (1D ResNet: (B, C, T) → (B, 512))
        ↓
    ECGConceptLayer  (10 heads, each → (μ_k, log_σ²_k))
        ↓
    UncertaintyClassifier  ([concept_probs, concept_vars] → task_prob)

Exported:
    HeartbeatCBM           — standard forward pass only (no MC-Dropout wrapper)
    CBMWithUncertainty     — MC-Dropout wrapper for epistemic uncertainty
    build_model(cfg)       — factory from config dict
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


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
        self.dropout  = nn.Dropout(p=dropout)
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
# Concept layer
# ─────────────────────────────────────────────────────────────────────────────

CONCEPT_NAMES: List[str] = ["P", "Q", "R", "S", "T", "U", "PR", "QRS", "ST", "QT"]


class ECGConceptLayer(nn.Module):
    """
    K independent concept heads.  Each head outputs (μ_k, log_σ²_k).

    Single Linear(input_dim, 2) per concept — one forward pass covers all heads
    via a stacked weight matrix for efficiency.
    """

    def __init__(self, input_dim: int = 512, num_concepts: int = 10) -> None:
        super().__init__()
        self.num_concepts = num_concepts
        # Fused: map z → (K, 2) in one matrix multiply instead of K separate ones
        self.proj = nn.Linear(input_dim, num_concepts * 2)

    def _split(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """z: (B, D) → logits (B, K), log_vars (B, K)"""
        out = self.proj(z).view(z.size(0), self.num_concepts, 2)   # (B, K, 2)
        return out[:, :, 0], out[:, :, 1]

    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns concept_probs (B,K) and concept_log_vars (B,K)."""
        logits, log_vars = self._split(z)
        return torch.sigmoid(logits), log_vars

    def forward_all(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Single-pass: returns (logits, concept_probs, log_vars) — all (B,K)."""
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


# ─────────────────────────────────────────────────────────────────────────────
# Classifier head
# ─────────────────────────────────────────────────────────────────────────────

class UncertaintyClassifier(nn.Module):
    """
    Linear head: [concept_probs (K), concept_vars (K)] → task_prob (1).

    Taking variances as inputs lets the model learn to down-weight uncertain concepts.
    """

    def __init__(self, num_concepts: int = 10) -> None:
        super().__init__()
        self.num_concepts = num_concepts
        self.linear = nn.Linear(num_concepts * 2, 1, bias=True)

    def forward(self, concept_probs: torch.Tensor, concept_log_vars: torch.Tensor) -> torch.Tensor:
        """Returns task_prob (B, 1)."""
        features = torch.cat([concept_probs, torch.exp(concept_log_vars)], dim=-1)
        return torch.sigmoid(self.linear(features))

    @torch.no_grad()
    def mc_task_uncertainty(
        self,
        concept_probs: torch.Tensor,
        concept_log_vars: torch.Tensor,
        n_mc: int = 50,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Monte Carlo sampling through concept uncertainty. Returns (mean, std, all) per sample."""
        stds   = torch.exp(0.5 * concept_log_vars)
        logits = torch.logit(concept_probs.clamp(1e-6, 1 - 1e-6))
        eps    = torch.randn(concept_probs.size(0), n_mc, self.num_concepts, device=concept_probs.device)
        sampled_probs = torch.sigmoid(logits.unsqueeze(1) + eps * stds.unsqueeze(1))
        vars_exp  = torch.exp(concept_log_vars).unsqueeze(1).expand_as(sampled_probs)
        features  = torch.cat([sampled_probs, vars_exp], dim=-1)
        task_probs = torch.sigmoid(self.linear(features)).squeeze(-1)   # (B, n_mc)
        return task_probs.mean(1), task_probs.std(1), task_probs

    def get_weights_dict(self, concept_names: Optional[List[str]] = None) -> dict:
        names = concept_names or CONCEPT_NAMES[: self.num_concepts]
        w = self.linear.weight.data[0].cpu().tolist()
        K = self.num_concepts
        return {name: {"prob_weight": w[i], "uncert_weight": w[i + K]} for i, name in enumerate(names)}


# ─────────────────────────────────────────────────────────────────────────────
# Full CBM
# ─────────────────────────────────────────────────────────────────────────────

class HeartbeatCBM(nn.Module):
    """Standard CBM without MC-Dropout. Use for inference when epistemic uncertainty is not needed."""

    def __init__(
        self,
        in_channels: int = 12,
        base_channels: int = 64,
        encoder_dropout: float = 0.2,
        num_concepts: int = 10,
    ) -> None:
        super().__init__()
        self.encoder       = TemporalEncoder(in_channels, base_channels, encoder_dropout)
        self.concept_layer = ECGConceptLayer(self.encoder.out_dim, num_concepts)
        self.classifier    = UncertaintyClassifier(num_concepts)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """x: (B,C,T) → (task_prob, concept_probs, concept_log_vars)"""
        z = self.encoder(x)
        c_probs, c_log_vars = self.concept_layer(z)
        return self.classifier(c_probs, c_log_vars), c_probs, c_log_vars

    @property
    def num_concepts(self) -> int:
        return self.concept_layer.num_concepts


class CBMWithUncertainty(nn.Module):
    """
    CBM with MC-Dropout for epistemic uncertainty decomposition.

    An additional dropout is applied to z (latent vector) before the concept
    layer, so repeated stochastic forward passes yield epistemic uncertainty.
    """

    def __init__(
        self,
        encoder: TemporalEncoder,
        concept_layer: ECGConceptLayer,
        classifier: UncertaintyClassifier,
        mc_dropout_rate: float = 0.1,
    ) -> None:
        super().__init__()
        self.encoder       = encoder
        self.concept_layer = concept_layer
        self.classifier    = classifier
        self.mc_dropout    = nn.Dropout(mc_dropout_rate)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        z = self.mc_dropout(self.encoder(x))
        c_probs, c_log_vars = self.concept_layer(z)
        return self.classifier(c_probs, c_log_vars), c_probs, c_log_vars

    def forward_with_logits(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Efficient single-pass: returns (task_prob, c_probs, c_log_vars, c_logits)."""
        z = self.mc_dropout(self.encoder(x))
        c_logits, c_probs, c_log_vars = self.concept_layer.forward_all(z)
        return self.classifier(c_probs, c_log_vars), c_probs, c_log_vars, c_logits

    @torch.no_grad()
    def predict_with_uncertainty(
        self,
        x: torch.Tensor,
        n_mc_epistemic: int = 20,
        n_mc_aleatoric: int = 50,
    ) -> Dict[str, torch.Tensor]:
        """
        Full uncertainty decomposition via law of total variance.

        Returns dict with keys:
            concept_probs_mean, concept_probs_std, concept_aleatoric, concept_epistemic,
            task_prob_mean, task_prob_std, task_aleatoric, task_epistemic
        """
        self.train()   # keep dropout active
        c_probs_runs, c_vars_runs, task_probs_runs = [], [], []

        for _ in range(n_mc_epistemic):
            z = self.mc_dropout(self.encoder(x))
            c_probs, c_log_vars = self.concept_layer(z)
            _, _, t_samples = self.classifier.mc_task_uncertainty(c_probs, c_log_vars, n_mc_aleatoric)
            c_probs_runs.append(c_probs)
            c_vars_runs.append(torch.exp(c_log_vars))
            task_probs_runs.append(t_samples)

        c_probs_stack = torch.stack(c_probs_runs)                   # (M, B, K)
        c_vars_stack  = torch.stack(c_vars_runs)                    # (M, B, K)
        task_stack    = torch.cat(task_probs_runs, dim=1)           # (B, M*n_mc_aleatoric)

        concept_epistemic = c_probs_stack.var(dim=0)
        concept_aleatoric = c_vars_stack.mean(dim=0)

        task_mean = task_stack.mean(1)
        task_std  = task_stack.std(1)
        per_means = torch.stack([task_probs_runs[i].mean(1) for i in range(n_mc_epistemic)])
        per_vars  = torch.stack([task_probs_runs[i].var(1)  for i in range(n_mc_epistemic)])
        task_aleatoric = per_vars.mean(0).sqrt()
        task_epistemic = per_means.var(0).sqrt()

        self.eval()
        return {
            "concept_probs_mean": c_probs_stack.mean(0),
            "concept_probs_std":  (concept_epistemic + concept_aleatoric).sqrt(),
            "concept_aleatoric":  concept_aleatoric,
            "concept_epistemic":  concept_epistemic,
            "task_prob_mean":     task_mean,
            "task_prob_std":      task_std,
            "task_aleatoric":     task_aleatoric,
            "task_epistemic":     task_epistemic,
        }

    @property
    def num_concepts(self) -> int:
        return self.concept_layer.num_concepts


# ─────────────────────────────────────────────────────────────────────────────
# Factory
# ─────────────────────────────────────────────────────────────────────────────

def build_model(cfg: dict) -> CBMWithUncertainty:
    """Build CBMWithUncertainty from a config dict (see config.yaml)."""
    enc_cfg = cfg.get("model", {}).get("encoder", {})
    cpt_cfg = cfg.get("model", {}).get("concept_layer", {})

    model_leads = cfg.get("data", {}).get("model_leads", None)
    model_lead  = cfg.get("data", {}).get("model_lead",  None)
    if model_leads is not None:
        in_channels = len(model_leads)
    elif model_lead is not None:
        in_channels = 1
    else:
        in_channels = enc_cfg.get("in_channels", 12)

    num_concepts = cpt_cfg.get("num_concepts", 10)

    encoder       = TemporalEncoder(in_channels, enc_cfg.get("base_channels", 64), enc_cfg.get("dropout", 0.2))
    concept_layer = ECGConceptLayer(encoder.out_dim, num_concepts)
    classifier    = UncertaintyClassifier(num_concepts)

    return CBMWithUncertainty(
        encoder, concept_layer, classifier,
        mc_dropout_rate=cfg.get("model", {}).get("mc_dropout_rate", 0.1),
    )
