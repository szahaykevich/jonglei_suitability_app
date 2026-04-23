"""Dash app entrypoint for Jonglei suitability exploration."""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
from dash import ALL, Dash, Input, Output, State, ctx, dcc, html, no_update

from app.compute import compute_indices
from app.config import FHI_DEFAULT_WEIGHTS, SVI_DEFAULT_WEIGHTS
from app.data_loader import load_rasters
from app.popup_logic import suitability_hover_text

RASTER_BUNDLE = load_rasters()
FHI_LAYERS = RASTER_BUNDLE.fhi_layers
SVI_LAYERS = RASTER_BUNDLE.svi_layers
FHI_NAMES = list(FHI_DEFAULT_WEIGHTS.keys())
SVI_NAMES = list(SVI_DEFAULT_WEIGHTS.keys())


def compute_maps(fhi_weights_pct: dict[str, float], svi_weights_pct: dict[str, float]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute FHI, SVI and suitability maps from loaded rasters."""

    return compute_indices(FHI_LAYERS, SVI_LAYERS, fhi_weights_pct, svi_weights_pct)


def heatmap_figure(z: np.ndarray, title: str, colorbar_title: str, with_hover: bool = False, fhi: np.ndarray | None = None, svi: np.ndarray | None = None) -> go.Figure:
    """Build a north-up heatmap figure."""

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
    fig.update_layout(title=title, margin={"l": 10, "r": 10, "t": 40, "b": 10}, yaxis={"autorange": "reversed"})
    return fig


def make_weight_control(family: str, name: str, value: float) -> html.Div:
    """Create synced slider + numeric input controls."""

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
            dcc.Slider(
                id={"type": "weight-slider", "family": family, "name": name},
                min=0,
                max=100,
                step=0.01,
                value=value,
                tooltip={"always_visible": False, "placement": "bottom"},
            ),
            dcc.Input(
                id={"type": "weight-input", "family": family, "name": name},
                type="number",
                min=0,
                max=100,
                step=0.01,
                value=value,
                style={"width": "100%", "marginTop": "0.35rem"},
            ),
        ],
        style={"marginBottom": "0.8rem"},
    )


def pct_total(values: list[float]) -> float:
    return float(sum(values))


def _coerce_values(values: list[float]) -> list[float]:
    return [max(0.0, min(100.0, float(v or 0.0))) for v in values]


def _rounded_to_100(values: list[float]) -> list[float]:
    rounded = [round(float(v), 2) for v in values]
    if not rounded:
        return rounded
    delta = round(100.0 - sum(rounded), 2)
    rounded[-1] = round(rounded[-1] + delta, 2)
    return rounded


def normalize_weights(values: list[float], changed_idx: int | None) -> list[float]:
    """Keep total at 100 while preserving other-factor proportions."""

    if not values:
        return []

    clean = _coerce_values(values)

    if changed_idx is None or len(clean) == 1:
        total = sum(clean)
        if total <= 0:
            clean = [100.0 / len(clean)] * len(clean)
        else:
            clean = [v * (100.0 / total) for v in clean]
        return _rounded_to_100(clean)

    changed_val = clean[changed_idx]
    remainder_target = max(0.0, 100.0 - changed_val)

    other_idxs = [i for i in range(len(clean)) if i != changed_idx]
    others_sum = sum(clean[i] for i in other_idxs)

    if others_sum <= 0:
        share = remainder_target / len(other_idxs) if other_idxs else 0.0
        for i in other_idxs:
            clean[i] = share
    else:
        scale = remainder_target / others_sum
        for i in other_idxs:
            clean[i] = clean[i] * scale

    clean[changed_idx] = changed_val
    return _rounded_to_100(clean)


initial_fhi, initial_svi, initial_si = compute_maps(FHI_DEFAULT_WEIGHTS, SVI_DEFAULT_WEIGHTS)

app = Dash(__name__)
app.title = "Jonglei Suitability"

warnings_text = [html.Li(message) for message in RASTER_BUNDLE.warnings]

app.layout = html.Div(
    [
        html.H2("UNMISS Jonglei Suitability Explorer"),
        html.P("Adjust one weight and the rest auto-rescale in that group to keep totals at 100%. Click Update Maps to recompute outputs."),
        html.P(f"Raster source: {RASTER_BUNDLE.raster_dir}"),
        html.Ul(warnings_text, style={"color": "#b45309"}) if warnings_text else html.Div(),
        html.Div(
            [
                html.Div(
                    [
                        html.H3("FHI factor weights (%)"),
                        *[make_weight_control("fhi", k, v) for k, v in FHI_DEFAULT_WEIGHTS.items()],
                        html.Div(id="fhi-total", style={"fontWeight": "700"}),
                    ],
                    style={"padding": "0.75rem", "border": "1px solid #ddd", "borderRadius": "8px"},
                ),
                html.Div(
                    [
                        html.H3("SVI factor weights (%)"),
                        *[make_weight_control("svi", k, v) for k, v in SVI_DEFAULT_WEIGHTS.items()],
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
    Output({"type": "weight-slider", "family": "fhi", "name": ALL}, "value"),
    Output({"type": "weight-slider", "family": "svi", "name": ALL}, "value"),
    Output({"type": "weight-input", "family": "fhi", "name": ALL}, "value"),
    Output({"type": "weight-input", "family": "svi", "name": ALL}, "value"),
    Output({"type": "weight-label", "family": "fhi", "name": ALL}, "children"),
    Output({"type": "weight-label", "family": "svi", "name": ALL}, "children"),
    Output("fhi-total", "children"),
    Output("svi-total", "children"),
    Input({"type": "weight-slider", "family": "fhi", "name": ALL}, "value"),
    Input({"type": "weight-slider", "family": "svi", "name": ALL}, "value"),
    Input({"type": "weight-input", "family": "fhi", "name": ALL}, "value"),
    Input({"type": "weight-input", "family": "svi", "name": ALL}, "value"),
)
def sync_and_normalize_weights(
    fhi_slider_vals: list[float],
    svi_slider_vals: list[float],
    fhi_input_vals: list[float],
    svi_input_vals: list[float],
):
    trigger = ctx.triggered_id
    norm_fhi = _coerce_values(fhi_slider_vals)
    norm_svi = _coerce_values(svi_slider_vals)

    if isinstance(trigger, dict):
        name = trigger.get("name")
        family = trigger.get("family")
        control_type = trigger.get("type")

        if family == "fhi":
            source = fhi_input_vals if control_type == "weight-input" else fhi_slider_vals
            idx = FHI_NAMES.index(name)
            norm_fhi = normalize_weights(source, idx)
        elif family == "svi":
            source = svi_input_vals if control_type == "weight-input" else svi_slider_vals
            idx = SVI_NAMES.index(name)
            norm_svi = normalize_weights(source, idx)
    else:
        norm_fhi = normalize_weights(norm_fhi, None)
        norm_svi = normalize_weights(norm_svi, None)

    fhi_labels = [f"{v:.2f}%" for v in norm_fhi]
    svi_labels = [f"{v:.2f}%" for v in norm_svi]

    return (
        norm_fhi,
        norm_svi,
        norm_fhi,
        norm_svi,
        fhi_labels,
        svi_labels,
        f"FHI total: {pct_total(norm_fhi):.2f}% ✅",
        f"SVI total: {pct_total(norm_svi):.2f}% ✅",
    )


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
    if n_clicks is None:
        return no_update, no_update, no_update, no_update

    fhi_weights = dict(zip(FHI_NAMES, fhi_vals))
    svi_weights = dict(zip(SVI_NAMES, svi_vals))
    fhi, svi, si = compute_maps(fhi_weights, svi_weights)

    return (
        heatmap_figure(si, "Suitability Index (Interactive)", "Suitability", with_hover=True, fhi=fhi, svi=svi),
        heatmap_figure(fhi, "Flood Hazard Index (Static)", "FHI"),
        heatmap_figure(svi, "Social Vulnerability Index (Static)", "SVI"),
        html.Span(f"Maps refreshed for update #{n_clicks}.", style={"color": "green", "fontWeight": "700"}),
    )


if __name__ == "__main__":
    app.run(debug=True)
