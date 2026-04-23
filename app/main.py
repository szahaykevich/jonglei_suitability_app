"""Dash app entrypoint for Jonglei suitability exploration."""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
from dash import ALL, Dash, Input, Output, State, ctx, dcc, html, no_update

from popup_logic import suitability_hover_text

# ---------------------------------------------------------------------------
# Methodology defaults from interactive notebook (percent values)
# ---------------------------------------------------------------------------
FHI_DEFAULTS: dict[str, float] = {
    "Elevation": 18.09,
    "Slope": 18.29,
    "TWI": 17.48,
    "DistRiver": 14.23,
    "Rainfall": 12.73,
    "LULC": 11.95,
    "SoilTexture": 7.24,
}

SVI_DEFAULTS: dict[str, float] = {
    "DistRoad": 25.0,
    "DistHealth": 25.0,
    "PopDensity": 25.0,
    "ConflictIndex": 25.0,
}

# ---------------------------------------------------------------------------
# Demo data grids (replace with raster data in production)
# ---------------------------------------------------------------------------
np.random.seed(7)
GRID_SHAPE = (25, 25)


def make_demo_layers(names: list[str], seed_start: int) -> dict[str, np.ndarray]:
    """Create deterministic 1..5 demo layers per factor."""

    out: dict[str, np.ndarray] = {}
    for i, name in enumerate(names):
        rng = np.random.default_rng(seed_start + i)
        out[name] = rng.integers(1, 6, size=GRID_SHAPE).astype(np.float32)
    return out


FHI_LAYERS = make_demo_layers(list(FHI_DEFAULTS.keys()), seed_start=100)
SVI_LAYERS = make_demo_layers(list(SVI_DEFAULTS.keys()), seed_start=200)


# ---------------------------------------------------------------------------
# Computation helpers
# ---------------------------------------------------------------------------
def weighted_index(layers: dict[str, np.ndarray], weights_pct: dict[str, float]) -> np.ndarray:
    """Compute weighted index where weights are percentages summing to 100."""

    return sum((weights_pct[name] / 100.0) * layer for name, layer in layers.items())


def compute_maps(fhi_weights_pct: dict[str, float], svi_weights_pct: dict[str, float]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute FHI, SVI and suitability maps.

    Suitability uses notebook inversion logic:
    SI = 0.6 * (6 - FHI) + 0.4 * (6 - SVI)
    """

    fhi = weighted_index(FHI_LAYERS, fhi_weights_pct)
    svi = weighted_index(SVI_LAYERS, svi_weights_pct)
    suitability = (0.6 * (6.0 - fhi)) + (0.4 * (6.0 - svi))
    return fhi, svi, suitability


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------
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
    """Create a slider + numeric input block with a visible percent label."""

    slider_id = {"type": "weight-slider", "family": family, "name": name}
    input_id = {"type": "weight-input", "family": family, "name": name}
    return html.Div(
        [
            html.Div(
                [
                    html.Span(name),
                    html.Span(
                        f"{value:.2f}%",
                        id={"type": "weight-label", "family": family, "name": name},
                        style={"fontWeight": "600"},
                    ),
                ],
                style={"display": "flex", "justifyContent": "space-between"},
            ),
            html.Div(
                [
                    dcc.Slider(
                        id=slider_id,
                        min=0,
                        max=100,
                        step=0.01,
                        value=value,
                        tooltip={"always_visible": False, "placement": "bottom"},
                    ),
                    dcc.Input(
                        id=input_id,
                        type="number",
                        value=value,
                        min=0,
                        max=100,
                        step=0.01,
                        debounce=True,
                        style={"width": "88px", "marginLeft": "0.5rem"},
                    ),
                ],
                style={"display": "flex", "alignItems": "center"},
            ),
        ],
        style={"marginBottom": "0.6rem"},
    )


def pct_total(values: list[float]) -> float:
    return float(sum(values))


def total_ok(total: float) -> bool:
    return abs(total - 100.0) <= 0.05


initial_fhi, initial_svi, initial_si = compute_maps(FHI_DEFAULTS, SVI_DEFAULTS)

app = Dash(__name__)
app.title = "Jonglei Suitability"

app.layout = html.Div(
    [
        html.H2("UNMISS Jonglei Suitability Explorer"),
        html.P("Adjust FHI and SVI factor weights. Maps update only when each group totals exactly 100% and you press Update Maps."),
        html.Div(
            [
                html.Div(
                    [
                        html.H3("FHI factor weights (%)"),
                        *[make_slider("fhi", k, v) for k, v in FHI_DEFAULTS.items()],
                        html.Div(id="fhi-total", style={"fontWeight": "700"}),
                    ],
                    style={"padding": "0.75rem", "border": "1px solid #ddd", "borderRadius": "8px"},
                ),
                html.Div(
                    [
                        html.H3("SVI factor weights (%)"),
                        *[make_slider("svi", k, v) for k, v in SVI_DEFAULTS.items()],
                        html.Div(id="svi-total", style={"fontWeight": "700"}),
                    ],
                    style={"padding": "0.75rem", "border": "1px solid #ddd", "borderRadius": "8px"},
                ),
            ],
            style={"display": "grid", "gridTemplateColumns": "1fr 1fr", "gap": "1rem"},
        ),
        html.Button(
            "Update Maps",
            id="update-btn",
            n_clicks=0,
            disabled=False,
            style={"marginTop": "1rem", "padding": "0.5rem 1rem", "fontWeight": "700"},
        ),
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
    Output({"type": "weight-slider", "family": "fhi", "name": ALL}, "value"),
    Output({"type": "weight-slider", "family": "svi", "name": ALL}, "value"),
    Output({"type": "weight-input", "family": "fhi", "name": ALL}, "value"),
    Output({"type": "weight-input", "family": "svi", "name": ALL}, "value"),
    Output({"type": "weight-label", "family": "fhi", "name": ALL}, "children"),
    Output({"type": "weight-label", "family": "svi", "name": ALL}, "children"),
    Output("fhi-total", "children"),
    Output("svi-total", "children"),
    Output("update-btn", "disabled"),
    Input({"type": "weight-slider", "family": "fhi", "name": ALL}, "value"),
    Input({"type": "weight-slider", "family": "svi", "name": ALL}, "value"),
    Input({"type": "weight-input", "family": "fhi", "name": ALL}, "value"),
    Input({"type": "weight-input", "family": "svi", "name": ALL}, "value"),
)
def refresh_weight_controls(
    fhi_slider_vals: list[float],
    svi_slider_vals: list[float],
    fhi_input_vals: list[float],
    svi_input_vals: list[float],
):
    trig = ctx.triggered_id
    fhi_vals = list(fhi_slider_vals)
    svi_vals = list(svi_slider_vals)

    if isinstance(trig, dict) and trig.get("type") == "weight-input":
        if trig.get("family") == "fhi":
            fhi_vals = [float(v or 0.0) for v in fhi_input_vals]
        elif trig.get("family") == "svi":
            svi_vals = [float(v or 0.0) for v in svi_input_vals]
    else:
        fhi_vals = [float(v or 0.0) for v in fhi_slider_vals]
        svi_vals = [float(v or 0.0) for v in svi_slider_vals]

    fhi_labels = [f"{v:.2f}%" for v in fhi_vals]
    svi_labels = [f"{v:.2f}%" for v in svi_vals]

    fhi_total = pct_total(fhi_vals)
    svi_total = pct_total(svi_vals)

    fhi_text = f"FHI total: {fhi_total:.2f}% {'✅' if total_ok(fhi_total) else '❌'}"
    svi_text = f"SVI total: {svi_total:.2f}% {'✅' if total_ok(svi_total) else '❌'}"
    disabled = not (total_ok(fhi_total) and total_ok(svi_total))
    return fhi_vals, svi_vals, fhi_vals, svi_vals, fhi_labels, svi_labels, fhi_text, svi_text, disabled


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

    fhi_weights = dict(zip(FHI_DEFAULTS.keys(), fhi_vals))
    svi_weights = dict(zip(SVI_DEFAULTS.keys(), svi_vals))
    fhi, svi, si = compute_maps(fhi_weights, svi_weights)

    return (
        heatmap_figure(si, "Suitability Index (Interactive)", "Suitability", with_hover=True, fhi=fhi, svi=svi),
        heatmap_figure(fhi, "Flood Hazard Index (Static)", "FHI"),
        heatmap_figure(svi, "Social Vulnerability Index (Static)", "SVI"),
        html.Span("Maps updated using revised FHI and SVI weights.", style={"color": "green", "fontWeight": "700"}),
    )


if __name__ == "__main__":
    app.run(debug=True)
