"""Dash app entrypoint for Jonglei suitability exploration."""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
from dash import ALL, Dash, Input, Output, State, dcc, html, no_update

from app.compute import compute_indices
from app.config import FHI_DEFAULT_WEIGHTS, SVI_DEFAULT_WEIGHTS
from app.data_loader import load_rasters
from app.popup_logic import suitability_hover_text

RASTER_BUNDLE = load_rasters()
FHI_LAYERS = RASTER_BUNDLE.fhi_layers
SVI_LAYERS = RASTER_BUNDLE.svi_layers


def compute_maps(fhi_weights_pct: dict[str, float], svi_weights_pct: dict[str, float]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute FHI, SVI and suitability maps from loaded rasters."""

    return compute_indices(FHI_LAYERS, SVI_LAYERS, fhi_weights_pct, svi_weights_pct)


def heatmap_figure(z: np.ndarray, title: str, colorbar_title: str, with_hover: bool = False, fhi: np.ndarray | None = None, svi: np.ndarray | None = None) -> go.Figure:
    """Build a heatmap figure."""

    if with_hover and fhi is not None and svi is not None:
        hover = np.empty(z.shape, dtype=object)
        for row in range(z.shape[0]):
            for col in range(z.shape[1]):
                hover[row, col] = suitability_hover_text(
                    county_name=f"Cell {row},{col}",
                    suitability_value=float(z[row, col]),
                    fhi_score=float(fhi[row, col]),
                    svi_score=float(svi[row, col]),
                )
        trace = go.Heatmap(z=z, text=hover, hoverinfo="text", colorscale="Viridis", colorbar={"title": colorbar_title})
    else:
        trace = go.Heatmap(z=z, colorscale="YlOrRd", hoverinfo="skip", colorbar={"title": colorbar_title})

    fig = go.Figure(data=trace)
    fig.update_layout(title=title, margin={"l": 10, "r": 10, "t": 40, "b": 10})
    return fig


def make_slider(family: str, name: str, value: float) -> html.Div:
    """Create a slider block with a visible percent label."""

    slider_id = {"type": "weight-slider", "family": family, "name": name}
    return html.Div(
        [
            html.Div([html.Span(name), html.Span(f"{value:.2f}%", id={"type": "weight-label", "family": family, "name": name}, style={"fontWeight": "600"})], style={"display": "flex", "justifyContent": "space-between"}),
            dcc.Slider(id=slider_id, min=0, max=100, step=0.01, value=value, tooltip={"always_visible": False, "placement": "bottom"}),
        ],
        style={"marginBottom": "0.6rem"},
    )


def pct_total(values: list[float]) -> float:
    return float(sum(values))


def total_ok(total: float) -> bool:
    return abs(total - 100.0) <= 0.05


initial_fhi, initial_svi, initial_si = compute_maps(FHI_DEFAULT_WEIGHTS, SVI_DEFAULT_WEIGHTS)

app = Dash(__name__)
app.title = "Jonglei Suitability"

warnings_text = [html.Li(message) for message in RASTER_BUNDLE.warnings]

app.layout = html.Div(
    [
        html.H2("UNMISS Jonglei Suitability Explorer"),
        html.P("Adjust FHI and SVI factor weights. Maps update only when each group totals exactly 100% and you press Update Maps."),
        html.P(f"Raster source: {RASTER_BUNDLE.raster_dir}"),
        html.Ul(warnings_text, style={"color": "#b45309"}) if warnings_text else html.Div(),
        html.Div(
            [
                html.Div(
                    [
                        html.H3("FHI factor weights (%)"),
                        *[make_slider("fhi", k, v) for k, v in FHI_DEFAULT_WEIGHTS.items()],
                        html.Div(id="fhi-total", style={"fontWeight": "700"}),
                    ],
                    style={"padding": "0.75rem", "border": "1px solid #ddd", "borderRadius": "8px"},
                ),
                html.Div(
                    [
                        html.H3("SVI factor weights (%)"),
                        *[make_slider("svi", k, v) for k, v in SVI_DEFAULT_WEIGHTS.items()],
                        html.Div(id="svi-total", style={"fontWeight": "700"}),
                    ],
                    style={"padding": "0.75rem", "border": "1px solid #ddd", "borderRadius": "8px"},
                ),
            ],
            style={"display": "grid", "gridTemplateColumns": "1fr 1fr", "gap": "1rem"},
        ),
        html.Button("Update Maps", id="update-btn", n_clicks=0, style={"marginTop": "1rem", "padding": "0.5rem 1rem", "fontWeight": "700"}),
        html.Div(id="status-msg", style={"marginTop": "0.5rem"}),
        dcc.Graph(id="suitability-plot", figure=heatmap_figure(initial_si, "Suitability Index (Interactive)", "Suitability", with_hover=True, fhi=initial_fhi, svi=initial_svi)),
        html.Div(
            [
                dcc.Graph(id="fhi-plot", figure=heatmap_figure(initial_fhi, "Flood Hazard Index (Static)", "FHI"), config={"staticPlot": True}),
                dcc.Graph(id="svi-plot", figure=heatmap_figure(initial_svi, "Social Vulnerability Index (Static)", "SVI"), config={"staticPlot": True}),
            ],
            style={"display": "grid", "gridTemplateColumns": "1fr 1fr", "gap": "1rem"},
        ),
    ],
    style={"maxWidth": "1200px", "margin": "0 auto", "padding": "1rem"},
)


@app.callback(
    Output({"type": "weight-label", "family": "fhi", "name": ALL}, "children"),
    Output({"type": "weight-label", "family": "svi", "name": ALL}, "children"),
    Output("fhi-total", "children"),
    Output("svi-total", "children"),
    Input({"type": "weight-slider", "family": "fhi", "name": ALL}, "value"),
    Input({"type": "weight-slider", "family": "svi", "name": ALL}, "value"),
)
def refresh_weight_labels(fhi_vals: list[float], svi_vals: list[float]):
    fhi_labels = [f"{v:.2f}%" for v in fhi_vals]
    svi_labels = [f"{v:.2f}%" for v in svi_vals]

    fhi_total = pct_total(fhi_vals)
    svi_total = pct_total(svi_vals)

    fhi_text = f"FHI total: {fhi_total:.2f}% {'✅' if total_ok(fhi_total) else '❌'}"
    svi_text = f"SVI total: {svi_total:.2f}% {'✅' if total_ok(svi_total) else '❌'}"
    return fhi_labels, svi_labels, fhi_text, svi_text


@app.callback(
    Output("suitability-plot", "figure"),
    Output("fhi-plot", "figure"),
    Output("svi-plot", "figure"),
    Output("status-msg", "children"),
    Input("update-btn", "n_clicks"),
    State({"type": "weight-slider", "family": "fhi", "name": ALL}, "value"),
    State({"type": "weight-slider", "family": "svi", "name": ALL}, "value"),
    prevent_initial_call=True,
)
def update_maps(n_clicks: int, fhi_vals: list[float], svi_vals: list[float]):
    del n_clicks
    fhi_total = pct_total(fhi_vals)
    svi_total = pct_total(svi_vals)

    if not total_ok(fhi_total) or not total_ok(svi_total):
        return no_update, no_update, no_update, html.Span(
            f"Cannot update maps yet. FHI total = {fhi_total:.2f}% and SVI total = {svi_total:.2f}%. Both must equal 100.00%.",
            style={"color": "crimson", "fontWeight": "700"},
        )

    fhi_weights = dict(zip(FHI_DEFAULT_WEIGHTS.keys(), fhi_vals))
    svi_weights = dict(zip(SVI_DEFAULT_WEIGHTS.keys(), svi_vals))
    fhi, svi, si = compute_maps(fhi_weights, svi_weights)

    return (
        heatmap_figure(si, "Suitability Index (Interactive)", "Suitability", with_hover=True, fhi=fhi, svi=svi),
        heatmap_figure(fhi, "Flood Hazard Index (Static)", "FHI"),
        heatmap_figure(svi, "Social Vulnerability Index (Static)", "SVI"),
        html.Span("Maps updated using revised FHI and SVI weights.", style={"color": "green", "fontWeight": "700"}),
    )


if __name__ == "__main__":
    app.run(debug=True)
