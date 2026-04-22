"""Dash app entrypoint for Jonglei suitability exploration."""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
from dash import Dash, Input, Output, dcc, html

from popup_logic import suitability_hover_text
from raster_utils import minmax_scale
from weights import DEFAULT_FHI_WEIGHTS, DEFAULT_SVI_WEIGHTS, combined_index_weight

# Demo grid (replace with raster data in production)
np.random.seed(7)
GRID_SHAPE = (25, 25)
FHI_GRID = minmax_scale(np.random.rand(*GRID_SHAPE))
SVI_GRID = minmax_scale(np.random.rand(*GRID_SHAPE))


def suitability_grid(fhi_weight: float, svi_weight: float) -> np.ndarray:
    fhi_w, svi_w = combined_index_weight(fhi_weight, svi_weight)
    return (FHI_GRID * fhi_w) + (SVI_GRID * svi_w)


def make_figure(fhi_weight: float, svi_weight: float) -> go.Figure:
    surface = suitability_grid(fhi_weight, svi_weight)
    hover = np.empty(surface.shape, dtype=object)
    for row in range(surface.shape[0]):
        for col in range(surface.shape[1]):
            hover[row, col] = suitability_hover_text(
                county_name=f"Cell {row},{col}",
                suitability_value=surface[row, col],
                fhi_score=FHI_GRID[row, col],
                svi_score=SVI_GRID[row, col],
            )

    fig = go.Figure(
        data=go.Heatmap(
            z=surface,
            text=hover,
            hoverinfo="text",
            colorscale="Viridis",
            colorbar={"title": "Suitability"},
        )
    )
    fig.update_layout(
        title="Jonglei Suitability Map (Prototype)",
        margin={"l": 10, "r": 10, "t": 40, "b": 10},
    )
    return fig


app = Dash(__name__)
app.title = "Jonglei Suitability"

app.layout = html.Div(
    [
        html.H2("UNMISS Jonglei Suitability Explorer"),
        html.Label("FHI contribution"),
        dcc.Slider(
            id="fhi-weight",
            min=0,
            max=1,
            step=0.05,
            value=0.6,
            marks={0: "0", 0.5: "0.5", 1: "1"},
        ),
        html.Label("SVI contribution"),
        dcc.Slider(
            id="svi-weight",
            min=0,
            max=1,
            step=0.05,
            value=0.4,
            marks={0: "0", 0.5: "0.5", 1: "1"},
        ),
        dcc.Graph(id="suitability-plot", figure=make_figure(0.6, 0.4)),
        html.Div(
            f"Default FHI weights: {DEFAULT_FHI_WEIGHTS.normalized()} | "
            f"Default SVI weights: {DEFAULT_SVI_WEIGHTS.normalized()}"
        ),
    ],
    style={"maxWidth": "980px", "margin": "0 auto", "padding": "1rem"},
)


@app.callback(Output("suitability-plot", "figure"), Input("fhi-weight", "value"), Input("svi-weight", "value"))
def refresh_map(fhi_weight: float, svi_weight: float) -> go.Figure:
    return make_figure(fhi_weight, svi_weight)


if __name__ == "__main__":
    app.run(debug=True)
