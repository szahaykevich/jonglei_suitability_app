"""Weight configuration and helpers for suitability scoring."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict


@dataclass(frozen=True)
class IndexWeights:
    """Container for a group's feature weights."""

    values: Dict[str, float]

    def normalized(self) -> Dict[str, float]:
        total = sum(self.values.values())
        if total <= 0:
            equal = 1 / max(len(self.values), 1)
            return {name: equal for name in self.values}
        return {name: value / total for name, value in self.values.items()}


DEFAULT_FHI_WEIGHTS = IndexWeights(
    {
        "flood_risk": 0.35,
        "water_access": 0.25,
        "terrain_stability": 0.20,
        "distance_to_settlements": 0.20,
    }
)

DEFAULT_SVI_WEIGHTS = IndexWeights(
    {
        "road_access": 0.30,
        "health_facility_access": 0.30,
        "population_pressure": 0.25,
        "security_incidents": 0.15,
    }
)


def combined_index_weight(fhi_weight: float, svi_weight: float) -> tuple[float, float]:
    """Normalize top-level FHI/SVI blend weights."""

    total = fhi_weight + svi_weight
    if total <= 0:
        return 0.5, 0.5
    return fhi_weight / total, svi_weight / total
