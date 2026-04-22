"""Small raster helpers used by the interactive prototype."""

from __future__ import annotations

import numpy as np


def weighted_sum(layers: list[np.ndarray], weights: list[float]) -> np.ndarray:
    """Combine aligned raster layers with normalized weights."""

    if len(layers) != len(weights):
        raise ValueError("layers and weights must have the same length")

    total = float(sum(weights))
    if total <= 0:
        raise ValueError("weights must sum to a value greater than zero")

    normalized = [w / total for w in weights]
    return np.sum([layer * weight for layer, weight in zip(layers, normalized)], axis=0)


def minmax_scale(layer: np.ndarray) -> np.ndarray:
    """Scale a raster to [0, 1] while handling constant layers."""

    lo = np.nanmin(layer)
    hi = np.nanmax(layer)
    if np.isclose(hi, lo):
        return np.zeros_like(layer, dtype=float)
    return (layer - lo) / (hi - lo)
