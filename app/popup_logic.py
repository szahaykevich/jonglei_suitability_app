"""Hover/popup text formatting for map cells."""

from __future__ import annotations


def suitability_hover_text(county_name: str, suitability_value: float, fhi_score: float, svi_score: float) -> str:
    """Build a concise hover label for map interaction."""

    return (
        f"Area: {county_name}<br>"
        f"Suitability: {suitability_value:.3f}<br>"
        f"FHI: {fhi_score:.3f}<br>"
        f"SVI: {svi_score:.3f}"
    )
