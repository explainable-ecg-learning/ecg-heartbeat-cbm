"""
utils.py — Shared reproducibility seeding and paired bootstrap confidence intervals.
"""

from __future__ import annotations

import random
from typing import Dict, List

import numpy as np
import torch
from sklearn.metrics import roc_auc_score


def set_seed(seed: int, deterministic: bool = False) -> None:
    """Seed Python, NumPy, and PyTorch (CPU + CUDA) RNGs."""
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def bootstrap_multilabel_auroc(
    targets: np.ndarray,
    probs: np.ndarray,
    class_names: List[str],
    n_boot: int = 1000,
    seed: int = 42,
) -> Dict[str, float]:
    """
    Record-level non-parametric bootstrap 95% CI for multi-label Macro-AUROC
    and per-class AUROC (e.g., PTB-XL 5 diagnostic superclasses).
    """
    rng = np.random.default_rng(seed)
    n, n_cls = targets.shape
    boot_macro: List[float] = []
    boot_cls: Dict[int, List[float]] = {c: [] for c in range(n_cls)}

    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        yt = targets[idx]
        yp = probs[idx]
        cls_aucs = []
        for c in range(n_cls):
            if len(np.unique(yt[:, c])) < 2:
                continue
            v = float(roc_auc_score(yt[:, c], yp[:, c]))
            boot_cls[c].append(v)
            cls_aucs.append(v)
        if len(cls_aucs) == n_cls:
            boot_macro.append(float(np.mean(cls_aucs)))

    out: Dict[str, float] = {}
    if boot_macro:
        arr = np.asarray(boot_macro, dtype=np.float64)
        out["pathology/macro_auroc_ci95_lo"] = float(np.percentile(arr, 2.5))
        out["pathology/macro_auroc_ci95_hi"] = float(np.percentile(arr, 97.5))
        out["pathology/macro_auroc_boot_std"] = float(arr.std(ddof=1))
    for c, cname in enumerate(class_names):
        if boot_cls[c]:
            arr = np.asarray(boot_cls[c], dtype=np.float64)
            out[f"pathology/{cname}/auroc_ci95_lo"] = float(np.percentile(arr, 2.5))
            out[f"pathology/{cname}/auroc_ci95_hi"] = float(np.percentile(arr, 97.5))
    return out


def bootstrap_binary_metrics(
    targets: np.ndarray,
    probs: np.ndarray,
    n_boot: int = 1000,
    seed: int = 42,
) -> Dict[str, float]:
    """Bootstrap 95% CIs for binary accuracy and AUROC."""
    rng = np.random.default_rng(seed)
    n = len(targets)
    accs, aucs = [], []
    preds = (probs >= 0.5).astype(np.int64)
    t_int = targets.astype(np.int64)

    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        accs.append(float((preds[idx] == t_int[idx]).mean()))
        if len(np.unique(t_int[idx])) >= 2:
            aucs.append(float(roc_auc_score(t_int[idx], probs[idx])))

    out: Dict[str, float] = {}
    if accs:
        a = np.asarray(accs, dtype=np.float64)
        out["task/accuracy_ci95_lo"] = float(np.percentile(a, 2.5))
        out["task/accuracy_ci95_hi"] = float(np.percentile(a, 97.5))
    if aucs:
        u = np.asarray(aucs, dtype=np.float64)
        out["task/auroc_ci95_lo"] = float(np.percentile(u, 2.5))
        out["task/auroc_ci95_hi"] = float(np.percentile(u, 97.5))
    return out
