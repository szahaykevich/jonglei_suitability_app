"""Plotly visualization helpers for suitability map."""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go


def build_suitability_figure(suitability: np.ndarray, fhi: np.ndarray, svi: np.ndarray) -> go.Figure:
    """Build map-like heatmap figure (no basemap)."""

    fig = go.Figure(
        go.Heatmap(
            z=suitability,
            customdata=np.dstack([fhi, svi]),
            colorscale="RdYlGn",
            zmin=1,
            zmax=5,
            zsmooth=False,
            hovertemplate=(
                "<b>Suitability: %{z:.2f}</b><br>"
                "FHI: %{customdata[0]:.2f}<br>"
                "SVI: %{customdata[1]:.2f}<extra></extra>"
            ),
            colorbar={"title": "Suitability"},
        )
    )
    fig.update_layout(
        title="Jonglei Suitability (FHI / SVI)",
        margin={"l": 10, "r": 10, "t": 48, "b": 10},
        xaxis={"showticklabels": False, "showgrid": False, "zeroline": False},
        yaxis={"showticklabels": False, "showgrid": False, "zeroline": False, "autorange": "reversed"},
        paper_bgcolor="white",
        plot_bgcolor="white",
    )
    return fig


def extract_point_summary(event_data: dict | None, label: str) -> str:
    """Format hover/click placeholder details from plotly event payload."""

    if not event_data or not event_data.get("points"):
        return f"{label}: hover or click a cell to view Suitability, FHI, and SVI."

    point = event_data["points"][0]
    fhi, svi = point.get("customdata", [np.nan, np.nan])
    return (
        f"{label}: row={point.get('y')}, col={point.get('x')} | "
        f"Suitability={float(point.get('z', np.nan)):.2f}, "
        f"FHI={float(fhi):.2f}, SVI={float(svi):.2f}"
    )
