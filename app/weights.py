"""Weight helpers for FHI/SVI factor groups."""

from __future__ import annotations

from dataclasses import dataclass

try:
    from .config import FHI_DEFAULT_WEIGHTS, SVI_DEFAULT_WEIGHTS
except ImportError:  # script execution fallback
    from config import FHI_DEFAULT_WEIGHTS, SVI_DEFAULT_WEIGHTS


@dataclass(frozen=True)
class WeightSet:
    values: dict[str, float]

    def normalized(self) -> dict[str, float]:
        total = float(sum(self.values.values()))
        if total <= 0:
            equal = 1.0 / max(len(self.values), 1)
            return {name: equal for name in self.values}
        return {name: value / total for name, value in self.values.items()}


DEFAULT_FHI_WEIGHTS = WeightSet(FHI_DEFAULT_WEIGHTS)
DEFAULT_SVI_WEIGHTS = WeightSet(SVI_DEFAULT_WEIGHTS)
