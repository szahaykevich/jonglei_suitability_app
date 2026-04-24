"""Load FHI/SVI raster factors with graceful fallback behavior."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import from_bounds

from app.config import DISPLAY_SCALE, PLACEHOLDER_SHAPE, ROADS_VECTOR_CANDIDATES, factor_paths, get_raster_dir
from app.raster_utils import is_lfs_pointer, placeholder_layer, read_reclass_layer


@dataclass
class RasterBundle:
    fhi_layers: dict[str, np.ndarray]
    svi_layers: dict[str, np.ndarray]
    raster_dir: Path
    display_transform: object | None
    display_crs: object | None
    roads_path: Path
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
    roads_path, roads_note = resolve_roads_path(root)
    if roads_note:
        warnings.append(roads_note)

    if not dem.exists() or is_lfs_pointer(dem):
        warnings.append(
            "Could not open full TIFF rasters (missing files or Git LFS pointers found). "
            "Showing placeholder grids. Set JONGLEI_RASTER_DIR to local full TIFFs for live data."
        )
        return _placeholder_bundle(root, roads_path, warnings)

    try:
        out_shape, bounds = _reference_grid(dem)
        with rasterio.open(dem) as ref:
            display_crs = ref.crs
        fhi = {name: read_reclass_layer(path, out_shape=out_shape, bounds=bounds) for name, path in fhi_paths.items()}
        svi = {name: read_reclass_layer(path, out_shape=out_shape, bounds=bounds) for name, path in svi_paths.items()}
        display_transform = from_bounds(bounds.left, bounds.bottom, bounds.right, bounds.top, out_shape[1], out_shape[0])
        return RasterBundle(
            fhi_layers=fhi,
            svi_layers=svi,
            raster_dir=root,
            display_transform=display_transform,
            display_crs=display_crs,
            roads_path=roads_path,
            warnings=warnings,
        )
    except Exception as exc:  # graceful runtime behavior for local environments
        warnings.append(
            f"Raster load failed ({exc.__class__.__name__}: {exc}). "
            "Showing placeholder grids."
        )
        return _placeholder_bundle(root, roads_path, warnings)


def _placeholder_bundle(root: Path, roads_path: Path, warnings: list[str]) -> RasterBundle:
    fhi_paths, svi_paths = factor_paths(root)
    fhi = {name: placeholder_layer(PLACEHOLDER_SHAPE) for name in fhi_paths}
    svi = {name: placeholder_layer(PLACEHOLDER_SHAPE) for name in svi_paths}
    return RasterBundle(
        fhi_layers=fhi,
        svi_layers=svi,
        raster_dir=root,
        display_transform=None,
        display_crs=None,
        roads_path=roads_path,
        warnings=warnings,
    )


def resolve_roads_path(root: Path) -> tuple[Path, str | None]:
    """Find the best roads overlay file path inside the repository."""

    root_candidates = [root / "roads.geojson", root / "jonglei_roads.geojson"]
    for candidate in [*ROADS_VECTOR_CANDIDATES, *root_candidates]:
        if candidate.exists():
            return candidate, None

    repo_root = Path(__file__).resolve().parents[1]
    roads_vectors = sorted(
        path
        for path in repo_root.rglob("*")
        if path.is_file() and path.suffix.lower() in {".geojson", ".json", ".shp", ".gpkg", ".zip"} and "road" in path.name.lower()
    )
    if roads_vectors:
        return roads_vectors[0], f"Roads overlay path auto-detected at {roads_vectors[0]}."

    fallback = ROADS_VECTOR_CANDIDATES[0]
    return (
        fallback,
        f"Roads overlay file not found. Expected one of: {', '.join(str(path) for path in ROADS_VECTOR_CANDIDATES)}.",
    )
