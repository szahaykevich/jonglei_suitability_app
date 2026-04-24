"""Application configuration for raster factor bindings and defaults."""

from __future__ import annotations

import os
from pathlib import Path

APP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = APP_DIR.parent
DEFAULT_RASTER_DIR = PROJECT_ROOT / "data" / "processed"
RASTER_DIR_ENV_VAR = "JONGLEI_RASTER_DIR"
DISPLAY_SCALE = 0.10
PLACEHOLDER_SHAPE = (80, 80)
ROADS_VECTOR_CANDIDATES = [
    PROJECT_ROOT / "public" / "data" / "jonglei_roads.shp",
    PROJECT_ROOT / "data" / "processed" / "jonglei_roads.shp",
    PROJECT_ROOT / "public" / "data" / "jonglei_roads.geojson",
    PROJECT_ROOT / "data" / "processed" / "jonglei_roads.geojson",
]

FHI_FACTOR_FILES = {
    "Elevation": "Reclass_DEM1_2.tif",
    "Slope": "Reclass_Slop1.tif",
    "TWI": "Reclass_TWI1.tif",
    "LULC": "Reclass_LULC_f1.tif",
    "SoilTexture": "Reclass_soil1.tif",
    "Rainfall": "Reclass_Rain1.tif",
    "DistRiver": "Reclass_EucD1.tif",
}

SVI_FACTOR_FILES = {
    "DistRoad": "Reclass_road1.tif",
    "PopDensity": "Reclass_population1.tif",
    "DistHealth": "Reclass_Dist_to_health1.tif",
    "ConflictIndex": "Reclass_Conflict1.tif",
}

FHI_DEFAULT_WEIGHTS = {
    "Elevation": 18.09,
    "Slope": 18.29,
    "TWI": 17.48,
    "DistRiver": 14.23,
    "Rainfall": 12.73,
    "LULC": 11.95,
    "SoilTexture": 7.24,
}

SVI_DEFAULT_WEIGHTS = {
    "DistRoad": 25.0,
    "DistHealth": 25.0,
    "PopDensity": 25.0,
    "ConflictIndex": 25.0,
}


def get_raster_dir() -> Path:
    """Return the user-selected raster directory or the repo default."""

    custom = os.getenv(RASTER_DIR_ENV_VAR)
    if custom:
        return Path(custom).expanduser().resolve()
    return DEFAULT_RASTER_DIR


def factor_paths(raster_dir: Path | None = None) -> tuple[dict[str, Path], dict[str, Path]]:
    """Build absolute paths for all configured FHI and SVI factors."""

    root = raster_dir or get_raster_dir()
    return (
        {name: root / fname for name, fname in FHI_FACTOR_FILES.items()},
        {name: root / fname for name, fname in SVI_FACTOR_FILES.items()},
    )
