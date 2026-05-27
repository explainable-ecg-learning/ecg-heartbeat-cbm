"""
losses.py — Combined CBM loss functions.

L_total = L_task + λ_concept(epoch) · L_concept + λ_consist · L_consistency

  L_task     : BCE on task probability (complete vs incomplete heartbeat)
  L_concept  : Heteroscedastic BCE — MC sampling through concept uncertainty
               encourages the model to capture label noise via σ²_k
  L_consist  : Soft logical constraints between composite and constituent concepts
               (e.g. PR → P AND R)

Exported:
    CombinedCBMLoss   — main nn.Module used in train.py
"""

from __future__ import annotations

from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────────────────────────
# Heteroscedastic concept loss
# ─────────────────────────────────────────────────────────────────────────────

def _heteroscedastic_bce(
    logits: torch.Tensor,
    log_vars: torch.Tensor,
    targets: torch.Tensor,
    weights: torch.Tensor,
    n_samples: int = 10,
) -> torch.Tensor:
    """
    Confidence-weighted heteroscedastic BCE.

    Approximates E_{ε~N(0,1)}[BCE(σ(μ + σ·ε), y)] via MC sampling.
    Concepts with weight=0 (failed delineation) are excluded from the loss.

    logits   : (B, K)
    log_vars : (B, K)
    targets  : (B, K)  — may be label-smoothed
    weights  : (B, K)  — binary {0, 1} confidence mask
    """
    stds = torch.exp(0.5 * log_vars)                                    # (B, K)
    eps  = torch.randn(logits.size(0), n_samples, logits.size(1), device=logits.device)
    sampled_logits = logits.unsqueeze(1) + eps * stds.unsqueeze(1)      # (B, n, K)
    sampled_probs  = torch.sigmoid(sampled_logits)

    targets_exp = targets.unsqueeze(1).expand_as(sampled_probs)
    bce = F.binary_cross_entropy(sampled_probs, targets_exp, reduction="none")   # (B, n, K)
    bce_mean = bce.mean(dim=1)                                          # (B, K) — average MC samples

    w_total = weights.sum()
    if w_total < 1:
        return torch.zeros(1, device=logits.device).squeeze()
    return (bce_mean * weights).sum() / w_total


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
        smoothed[:, k] = smoothed[:, k] * (1 - eps) + (1 - smoothed[:, k]) * eps
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
    Combined loss for the heartbeat CBM.

    Parameters
    ----------
    lambda_concept_start : Initial λ for concept loss (default 1.0).
    lambda_concept_end   : Final λ after annealing (default 0.3).
    lambda_consist       : Weight for soft consistency penalty (default 0.1).
    anneal_epochs        : Epochs over which to linearly anneal λ_concept.
    n_mc_samples         : MC samples in heteroscedastic loss.
    reliable_concepts    : Concept indices with tight label smoothing.
    smooth_reliable      : ε for reliable concepts (default 0.02).
    smooth_noisy         : ε for noisy concepts (default 0.10).
    use_heteroscedastic  : Use heteroscedastic BCE; set False for plain BCE ablation.
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
    ) -> None:
        super().__init__()
        self.lam_start          = lambda_concept_start
        self.lam_end            = lambda_concept_end
        self.lambda_consist     = lambda_consist
        self.anneal_epochs      = anneal_epochs
        self.n_mc_samples       = n_mc_samples
        self.reliable_concepts  = reliable_concepts if reliable_concepts is not None else [2, 4, 7, 9]
        self.smooth_reliable    = smooth_reliable
        self.smooth_noisy       = smooth_noisy
        self.use_heteroscedastic = use_heteroscedastic

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
    ) -> dict:
        """Returns dict with keys: total, task, concept, consistency, lambda_concept."""
        # Task loss
        loss_task = F.binary_cross_entropy(task_prob.squeeze(-1), task_labels.float())

        # Concept loss
        K_pred   = concept_logits.size(1)
        K_labels = concept_labels.size(1)
        if K_pred != K_labels:
            loss_concept = torch.zeros(1, device=task_prob.device).squeeze()
        else:
            smoothed = _smooth_targets(concept_labels, self.reliable_concepts,
                                       self.smooth_reliable, self.smooth_noisy)
            if self.use_heteroscedastic:
                loss_concept = _heteroscedastic_bce(
                    concept_logits, concept_log_vars, smoothed, concept_weights, self.n_mc_samples
                )
            else:
                loss_concept = F.binary_cross_entropy(
                    concept_probs, smoothed, weight=concept_weights, reduction="mean"
                )

        # Consistency loss (only for 10-concept standard model)
        if concept_probs.size(1) == 10 and self.lambda_consist > 0:
            loss_consist = _consistency_loss(concept_probs)
        else:
            loss_consist = torch.zeros(1, device=task_prob.device).squeeze()

        lam_c = self.lambda_concept(epoch)
        total = loss_task + lam_c * loss_concept + self.lambda_consist * loss_consist

        return {
            "total":         total,
            "task":          loss_task.detach(),
            "concept":       loss_concept.detach(),
            "consistency":   loss_consist.detach(),
            "lambda_concept": lam_c,
        }
