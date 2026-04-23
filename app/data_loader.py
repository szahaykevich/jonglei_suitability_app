"""Load FHI/SVI raster factors with graceful fallback behavior."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import rasterio

try:
    from .config import DISPLAY_SCALE, PLACEHOLDER_SHAPE, factor_paths, get_raster_dir
    from .raster_utils import is_lfs_pointer, placeholder_layer, read_reclass_layer
except ImportError:  # script execution fallback
    from config import DISPLAY_SCALE, PLACEHOLDER_SHAPE, factor_paths, get_raster_dir
    from raster_utils import is_lfs_pointer, placeholder_layer, read_reclass_layer


@dataclass
class RasterBundle:
    fhi_layers: dict[str, np.ndarray]
    svi_layers: dict[str, np.ndarray]
    raster_dir: Path
    warnings: list[str]


def _reference_grid(ref_path: Path) -> tuple[tuple[int, int], object]:
    with rasterio.open(ref_path) as ref:
        out_shape = (
            max(1, int(ref.height * DISPLAY_SCALE)),
            max(1, int(ref.width * DISPLAY_SCALE)),
        )
        return out_shape, ref.bounds


def load_rasters(raster_dir: Path | None = None) -> RasterBundle:
    """Load all configured factors; fallback to placeholders when needed."""

    root = raster_dir or get_raster_dir()
    fhi_paths, svi_paths = factor_paths(root)
    warnings: list[str] = []

    dem = fhi_paths["Elevation"]
    if not dem.exists() or is_lfs_pointer(dem):
        warnings.append(
            "Could not open full TIFF rasters (missing files or Git LFS pointers found). "
            "Showing placeholder grids. Set JONGLEI_RASTER_DIR to local full TIFFs for live data."
        )
        return _placeholder_bundle(root, warnings)

    try:
        out_shape, bounds = _reference_grid(dem)
        fhi = {name: read_reclass_layer(path, out_shape=out_shape, bounds=bounds) for name, path in fhi_paths.items()}
        svi = {name: read_reclass_layer(path, out_shape=out_shape, bounds=bounds) for name, path in svi_paths.items()}
        return RasterBundle(fhi_layers=fhi, svi_layers=svi, raster_dir=root, warnings=warnings)
    except Exception as exc:  # graceful runtime behavior for local environments
        warnings.append(
            f"Raster load failed ({exc.__class__.__name__}: {exc}). "
            "Showing placeholder grids."
        )
        return _placeholder_bundle(root, warnings)


def _placeholder_bundle(root: Path, warnings: list[str]) -> RasterBundle:
    fhi_paths, svi_paths = factor_paths(root)
    fhi = {name: placeholder_layer(PLACEHOLDER_SHAPE) for name in fhi_paths}
    svi = {name: placeholder_layer(PLACEHOLDER_SHAPE) for name in svi_paths}
    return RasterBundle(fhi_layers=fhi, svi_layers=svi, raster_dir=root, warnings=warnings)
