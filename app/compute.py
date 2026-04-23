"""Suitability computation functions."""

from __future__ import annotations

import numpy as np


def weighted_sum(layers: dict[str, np.ndarray], weights: dict[str, float]) -> np.ndarray:
    """Compute weighted sum using percentage-style weights."""

    total = float(sum(weights.values()))
    if total <= 0:
        raise ValueError("Weights must sum to a positive number")

    out = np.zeros_like(next(iter(layers.values())), dtype=np.float32)
    for name, arr in layers.items():
        out += arr * np.float32(weights[name] / total)
    return out


def compute_indices(
    fhi_layers: dict[str, np.ndarray],
    svi_layers: dict[str, np.ndarray],
    fhi_weights: dict[str, float],
    svi_weights: dict[str, float],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute FHI, SVI, and final Suitability grids."""

    fhi = weighted_sum(fhi_layers, fhi_weights)
    svi = weighted_sum(svi_layers, svi_weights)
    suitability = np.float32(0.6) * (np.float32(6.0) - fhi) + np.float32(0.4) * (np.float32(6.0) - svi)
    return fhi, svi, suitability
