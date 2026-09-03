"""
Sensor Feature Importance via Model-Agnostic Permutation Importance.
"""

from typing import Dict, List, Any
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from src.utils.device import resolve_device


def compute_permutation_feature_importance(
    model: nn.Module,
    X_val: np.ndarray,
    y_val: np.ndarray,
    feature_names: List[str],
    device: str = "auto",
    n_repeats: int = 5,
    seed: int = 42,
) -> Dict[str, float]:
    """
    Calculate permutation feature importance on validation telemetry.
    """
    rng = np.random.default_rng(seed)
    model.eval()
    dev = resolve_device(device)
    model.to(dev)

    with torch.no_grad():
        inputs = torch.tensor(X_val, dtype=torch.float32).to(dev)
        base_logits = model(inputs)
        base_preds = torch.argmax(base_logits, dim=1).cpu().numpy()
        base_f1 = f1_score(y_val, base_preds, average="macro", zero_division=0)

    importances = {}

    for i, name in enumerate(feature_names):
        drop_scores = []
        for _ in range(n_repeats):
            X_perm = X_val.copy()
            perm_idx = rng.permutation(len(X_perm))
            X_perm[:, i] = X_perm[perm_idx, i]

            with torch.no_grad():
                perm_inputs = torch.tensor(X_perm, dtype=torch.float32).to(dev)
                perm_logits = model(perm_inputs)
                perm_preds = torch.argmax(perm_logits, dim=1).cpu().numpy()
                perm_f1 = f1_score(y_val, perm_preds, average="macro", zero_division=0)
                drop_scores.append(float(base_f1 - perm_f1))

        importances[name] = float(np.mean(drop_scores))

    total = sum(max(0.0, v) for v in importances.values())
    if total > 1e-6:
        norm_importances = {
            k: round(max(0.0, v) / total * 100.0, 2)
            for k, v in sorted(importances.items(), key=lambda x: x[1], reverse=True)
        }
    else:
        norm_importances = {name: round(100.0 / len(feature_names), 2) for name in feature_names}

    return norm_importances
