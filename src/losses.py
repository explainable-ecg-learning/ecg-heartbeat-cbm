"""
losses.py — Combined CBM / CEM loss functions.

L_total = w_task · L_task + λ_concept(epoch) · L_concept + λ_consist · L_consistency

  L_task     : BCE on task probability (complete vs incomplete heartbeat)
  L_concept  : Heteroscedastic BCE — MC sampling through concept uncertainty
               with L2 variance regularisation to prevent log_var drift on masked concepts
  L_consist  : Soft logical constraints between composite and constituent concepts
               (e.g. PR → P AND R)
"""

from __future__ import annotations

from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────────────────────────
# Heteroscedastic & masked plain concept losses
# ─────────────────────────────────────────────────────────────────────────────

def _heteroscedastic_bce(
    logits: torch.Tensor,
    log_vars: torch.Tensor,
    targets: torch.Tensor,
    weights: torch.Tensor,
    n_samples: int = 10,
    var_reg_weight: float = 1e-3,
) -> torch.Tensor:
    """
    Confidence-weighted heteroscedastic BCE with log-variance regularisation.

    Approximates E_{ε~N(0,1)}[BCE(σ(μ + σ·ε), y)] via MC sampling, plus a small
    L2 prior `var_reg_weight * mean(log_vars²)` so unconstrained/masked concepts
    remain anchored near unit variance rather than drifting to ±∞.
    """
    stds = torch.exp(0.5 * log_vars)                                    # (B, K)
    eps  = torch.randn(logits.size(0), n_samples, logits.size(1), device=logits.device)
    sampled_logits = logits.unsqueeze(1) + eps * stds.unsqueeze(1)      # (B, n, K)

    targets_exp = targets.unsqueeze(1).expand_as(sampled_logits)
    bce = F.binary_cross_entropy_with_logits(sampled_logits, targets_exp, reduction="none")
    bce_mean = bce.mean(dim=1)                                          # (B, K)

    var_reg = var_reg_weight * (log_vars ** 2).mean() if var_reg_weight > 0 else 0.0

    w_total = weights.sum()
    if w_total < 1:
        return torch.zeros((), device=logits.device) + var_reg
    return (bce_mean * weights).sum() / w_total + var_reg


def _masked_bce_with_logits(
    logits: torch.Tensor,
    targets: torch.Tensor,
    weights: torch.Tensor,
) -> torch.Tensor:
    """Plain confidence-masked BCE (normalised by valid weight sum, not B*K)."""
    w_total = weights.sum()
    if w_total < 1:
        return torch.zeros((), device=logits.device)
    bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    return (bce * weights).sum() / w_total


def _smooth_targets(
    targets: torch.Tensor,
    reliable_concepts: List[int],
    smooth_reliable: float = 0.02,
    smooth_noisy: float = 0.10,
) -> torch.Tensor:
    """Apply per-concept label smoothing."""
    K = targets.size(1)
    smoothed = targets.clone().float()
    for k in range(K):
        eps = smooth_reliable if k in reliable_concepts else smooth_noisy
        smoothed[:, k] = smoothed[:, k] * (1.0 - eps) + (1.0 - smoothed[:, k]) * eps
    return smoothed


# ─────────────────────────────────────────────────────────────────────────────
# Consistency loss
# ─────────────────────────────────────────────────────────────────────────────

def _consistency_loss(c: torch.Tensor) -> torch.Tensor:
    """
    Soft penalty for logical violations among 10 beat concepts.

    Concept indices: P=0, Q=1, R=2, S=3, T=4, U=5, PR=6, QRS=7, ST=8, QT=9
    Constraints: PR→P, PR→R, QRS→R, ST→S, ST→T, QT→QRS, QT→T
    """
    loss  = F.relu(c[:, 6] - c[:, 0]).mean()   # PR → P
    loss += F.relu(c[:, 6] - c[:, 2]).mean()   # PR → R
    loss += F.relu(c[:, 7] - c[:, 2]).mean()   # QRS → R
    loss += F.relu(c[:, 8] - c[:, 3]).mean()   # ST → S
    loss += F.relu(c[:, 8] - c[:, 4]).mean()   # ST → T
    loss += F.relu(c[:, 9] - c[:, 7]).mean()   # QT → QRS
    loss += F.relu(c[:, 9] - c[:, 4]).mean()   # QT → T
    return loss / 7.0


# ─────────────────────────────────────────────────────────────────────────────
# Combined loss module
# ─────────────────────────────────────────────────────────────────────────────

class CombinedCBMLoss(nn.Module):
    """
    Combined loss for CBM, CEM, and Black-Box models across joint and staged modes.
    """

    def __init__(
        self,
        lambda_concept_start: float = 1.0,
        lambda_concept_end:   float = 0.3,
        lambda_consist:       float = 0.1,
        anneal_epochs:        int   = 50,
        n_mc_samples:         int   = 10,
        reliable_concepts:    Optional[List[int]] = None,
        smooth_reliable:      float = 0.02,
        smooth_noisy:         float = 0.10,
        use_heteroscedastic:  bool  = True,
        var_reg_weight:       float = 1e-3,
    ) -> None:
        super().__init__()
        self.lam_start           = lambda_concept_start
        self.lam_end             = lambda_concept_end
        self.lambda_consist      = lambda_consist
        self.anneal_epochs       = anneal_epochs
        self.n_mc_samples        = n_mc_samples
        self.reliable_concepts   = reliable_concepts if reliable_concepts is not None else [2, 4, 7, 9]
        self.smooth_reliable     = smooth_reliable
        self.smooth_noisy        = smooth_noisy
        self.use_heteroscedastic = use_heteroscedastic
        self.var_reg_weight      = var_reg_weight

    def lambda_concept(self, epoch: int) -> float:
        frac = min(1.0, epoch / max(1, self.anneal_epochs))
        return self.lam_start + frac * (self.lam_end - self.lam_start)

    def forward(
        self,
        task_prob:        torch.Tensor,   # (B, 1)
        concept_probs:    torch.Tensor,   # (B, K)
        concept_log_vars: torch.Tensor,   # (B, K)
        concept_logits:   torch.Tensor,   # (B, K)
        task_labels:      torch.Tensor,   # (B,)
        concept_labels:   torch.Tensor,   # (B, K)
        concept_weights:  torch.Tensor,   # (B, K)
        epoch:            int = 0,
        stage:            str = "joint",  # "joint" | "concept_only" | "task_only"
        compute_concepts: bool = True,
    ) -> dict:
        """Returns dict with keys: total, task, concept, consistency, lambda_concept."""
        device = task_prob.device
        pred_prob = task_prob.squeeze(-1).clamp(1e-6, 1.0 - 1e-6)
        loss_task = F.binary_cross_entropy(pred_prob, task_labels.float())

        if not compute_concepts or concept_logits.size(1) != concept_labels.size(1):
            loss_concept = torch.zeros((), device=device)
            loss_consist = torch.zeros((), device=device)
            lam_c = 0.0
        else:
            smoothed = _smooth_targets(
                concept_labels, self.reliable_concepts,
                self.smooth_reliable, self.smooth_noisy,
            )
            if self.use_heteroscedastic:
                loss_concept = _heteroscedastic_bce(
                    concept_logits, concept_log_vars, smoothed, concept_weights,
                    n_samples=self.n_mc_samples, var_reg_weight=self.var_reg_weight,
                )
            else:
                loss_concept = _masked_bce_with_logits(concept_logits, smoothed, concept_weights)

            if concept_probs.size(1) == 10 and self.lambda_consist > 0:
                loss_consist = _consistency_loss(concept_probs)
            else:
                loss_consist = torch.zeros((), device=device)
            lam_c = self.lambda_concept(epoch)

        if stage == "concept_only":
            total = loss_concept + self.lambda_consist * loss_consist
        elif stage == "task_only":
            total = loss_task
            lam_c = 0.0
        else:
            total = loss_task + lam_c * loss_concept + self.lambda_consist * loss_consist

        return {
            "total":          total,
            "task":           loss_task.detach(),
            "concept":        loss_concept.detach(),
            "consistency":    loss_consist.detach(),
            "lambda_concept": lam_c,
        }
