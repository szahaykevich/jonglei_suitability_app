"""Raster IO and validation helpers."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.windows import from_bounds as bounds_window

LFS_POINTER_PREFIX = b"version https://git-lfs.github.com/spec/v1"


def is_lfs_pointer(path: Path) -> bool:
    """True when file content is a Git LFS pointer, not full binary raster."""

    if not path.exists() or path.stat().st_size > 512:
        return False
    with path.open("rb") as fh:
        head = fh.read(128)
    return head.startswith(LFS_POINTER_PREFIX)


def read_reclass_layer(path: Path, *, out_shape: tuple[int, int], bounds) -> np.ndarray:
    """Load one reclassified layer into the common display grid."""

    with rasterio.open(path) as src:
        win = bounds_window(bounds.left, bounds.bottom, bounds.right, bounds.top, src.transform)
        data = src.read(
            1,
            window=win,
            out_shape=out_shape,
            resampling=Resampling.nearest,
        ).astype(np.float32)
        nodata = src.nodata

    if nodata is not None:
        data[data == nodata] = np.nan

    data = np.round(data)
    data[(data < 1) | (data > 5)] = np.nan
    return data


def placeholder_layer(shape: tuple[int, int]) -> np.ndarray:
    """Create deterministic placeholder values in the 1..5 range."""

    y = np.linspace(1.5, 4.5, shape[0], dtype=np.float32)[:, None]
    x = np.linspace(1.0, 5.0, shape[1], dtype=np.float32)[None, :]
    return np.clip((x + y) / 2.0, 1.0, 5.0)
