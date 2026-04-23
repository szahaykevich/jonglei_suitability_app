"""Dash app entrypoint for Jonglei suitability exploration."""

from __future__ import annotations

from dash import Dash, Input, Output, dcc, html

try:
    from .compute import compute_indices
    from .config import FHI_DEFAULT_WEIGHTS, SVI_DEFAULT_WEIGHTS
    from .data_loader import load_rasters
    from .plotting import build_suitability_figure, extract_point_summary
except ImportError:  # script execution fallback
    from compute import compute_indices
    from config import FHI_DEFAULT_WEIGHTS, SVI_DEFAULT_WEIGHTS
    from data_loader import load_rasters
    from plotting import build_suitability_figure, extract_point_summary

RASTERS = load_rasters()


def weight_slider(slider_id: str, label: str, value: float) -> html.Div:
    return html.Div(
        [
            html.Label(f"{label} ({value:.2f}%)", id=f"{slider_id}-label"),
            dcc.Slider(
                id=slider_id,
                min=1,
                max=60,
                step=0.01,
                value=value,
                tooltip={"placement": "bottom", "always_visible": False},
            ),
        ],
        style={"marginBottom": "0.65rem"},
    )


def compute_figure(fhi_weights: dict[str, float], svi_weights: dict[str, float]):
    fhi, svi, suitability = compute_indices(RASTERS.fhi_layers, RASTERS.svi_layers, fhi_weights, svi_weights)
    fig = build_suitability_figure(suitability, fhi, svi)
    return fig


app = Dash(__name__)
app.title = "Jonglei Suitability"

app.layout = html.Div(
    [
        html.H2("UNMISS Jonglei Suitability Explorer"),
        html.Div(
            [
                html.Strong("Raster directory"),
                html.Div(str(RASTERS.raster_dir)),
                html.Div(
                    RASTERS.warnings[0] if RASTERS.warnings else "All configured rasters loaded.",
                    style={"color": "#b45309" if RASTERS.warnings else "#065f46", "marginTop": "0.25rem"},
                    id="raster-status",
                ),
            ],
            style={"padding": "0.75rem", "border": "1px solid #ddd", "marginBottom": "1rem"},
        ),
        html.Div(
            [
                html.Div(
                    [
                        html.H4("FHI weights"),
                        *[
                            weight_slider(f"fhi-{name}", name, value)
                            for name, value in FHI_DEFAULT_WEIGHTS.items()
                        ],
                    ],
                    style={"flex": "1", "minWidth": "320px"},
                ),
                html.Div(
                    [
                        html.H4("SVI weights"),
                        *[
                            weight_slider(f"svi-{name}", name, value)
                            for name, value in SVI_DEFAULT_WEIGHTS.items()
                        ],
                    ],
                    style={"flex": "1", "minWidth": "320px"},
                ),
            ],
            style={"display": "flex", "gap": "1rem", "flexWrap": "wrap"},
        ),
        dcc.Graph(id="suitability-map", figure=compute_figure(FHI_DEFAULT_WEIGHTS, SVI_DEFAULT_WEIGHTS)),
        html.Div(
            [
                html.H4("Hover details"),
                html.Div(
                    "Hover: hover or click a cell to view Suitability, FHI, and SVI.",
                    id="hover-details",
                    style={"fontFamily": "monospace"},
                ),
                html.H4("Click details", style={"marginTop": "0.75rem"}),
                html.Div(
                    "Click: hover or click a cell to view Suitability, FHI, and SVI.",
                    id="click-details",
                    style={"fontFamily": "monospace"},
                ),
            ],
            style={"padding": "0.75rem", "border": "1px solid #ddd", "marginTop": "0.75rem"},
        ),
    ],
    style={"maxWidth": "1200px", "margin": "0 auto", "padding": "1rem"},
)


@app.callback(
    Output("suitability-map", "figure"),
    Output("hover-details", "children"),
    Output("click-details", "children"),
    *([Input(f"fhi-{name}", "value") for name in FHI_DEFAULT_WEIGHTS]),
    *([Input(f"svi-{name}", "value") for name in SVI_DEFAULT_WEIGHTS]),
    Input("suitability-map", "hoverData"),
    Input("suitability-map", "clickData"),
)
def refresh_map(*args):
    fhi_count = len(FHI_DEFAULT_WEIGHTS)
    svi_count = len(SVI_DEFAULT_WEIGHTS)

    fhi_vals = args[:fhi_count]
    svi_vals = args[fhi_count : fhi_count + svi_count]
    hover_data = args[-2]
    click_data = args[-1]

    fhi_weights = {name: float(val) for name, val in zip(FHI_DEFAULT_WEIGHTS, fhi_vals)}
    svi_weights = {name: float(val) for name, val in zip(SVI_DEFAULT_WEIGHTS, svi_vals)}

    fig = compute_figure(fhi_weights, svi_weights)
    hover_summary = extract_point_summary(hover_data, "Hover")
    click_summary = extract_point_summary(click_data, "Click")
    return fig, hover_summary, click_summary


if __name__ == "__main__":
    app.run(debug=True)
