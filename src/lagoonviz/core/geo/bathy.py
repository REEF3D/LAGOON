# SPDX-License-Identifier: GPL-3.0-or-later
# Architect: Hans Bihs
"""Bathymetry for DIVEMesh: from a survey or terrain model to ``geo.dat``.

A bathymetry source is a raster (GeoTIFF, Esri ASCII grid) or scattered soundings
(x y z), in a CRS. :func:`grid_heights` brings it onto a grid of model
coordinates, through the case's georeference:

- a raster is sampled bilinearly; holes are filled from the nearest data;
- scattered points are interpolated: ``"tin"`` (linear on their Delaunay
  triangulation, the default), ``"idw"`` (inverse distance, the 12 nearest points)
  or ``"nearest"``; outside the points' hull the nearest point's height is taken.

:func:`write_geo_dat` writes the grid as ``geo.dat`` in model coordinates, the
points DIVEMesh reads with ``G 10 1``; DIVEMesh's own interpolation then only
carries them onto its cells.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from lagoon_format.geo import Georef

from lagoonviz.core.geo.basemap import warp
from lagoonviz.core.geo.crs import transformer
from lagoonviz.core.geo.raster import (
    POINT_SUFFIXES,
    RASTER_SUFFIXES,
    GeoImage,
    RasterError,
    crs_of,
    read_points,
    read_raster,
)

METHODS = ("tin", "idw", "nearest")


class Cancelled(Exception):
    """Making geo.dat was cancelled (by the progress callback)."""


#: Progress signature: ``progress(fraction 0..1, what is being done)``; it may raise
#: :class:`Cancelled` to stop
Progress = Callable[[float, str], None]


def _no_progress(_fraction: float, _text: str) -> None:
    pass


#: at most this many geo.dat points (the spacing grows to stay under it)
MOST_POINTS = 4_000_000


@dataclass
class Bathymetry:
    """A source of bed heights in a CRS: a raster or scattered points."""

    name: str
    crs: str | None
    raster: GeoImage | None = None
    points: np.ndarray | None = None  # (n, 3) in the CRS
    #: the values are depths (positive down), not heights
    depths: bool = False

    def describe(self) -> str:
        if self.raster is not None:
            rows, cols = self.raster.shape
            return f"{self.name}: raster {cols} x {rows}, {self.raster.cell_size():g} m cells"
        n = 0 if self.points is None else len(self.points)
        return f"{self.name}: {n} points"


def load(path: str | Path, crs: str | None = None, depths: bool = False) -> Bathymetry:
    """A bathymetry file: GeoTIFF or .asc (a raster), else x y z points. ``crs``: the
    CRS when the file does not name one (a raster's own is kept unless given)."""
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix in RASTER_SUFFIXES:
        raster = read_raster(path, crs_of(crs) if crs else None)
        if raster.is_image:
            raise RasterError(f"{path.name} is a picture, not heights")
        return Bathymetry(path.name, raster.crs, raster=raster, depths=depths)
    if suffix in POINT_SUFFIXES or suffix == "":
        return Bathymetry(path.name, crs_of(crs), points=read_points(path), depths=depths)
    raise RasterError(f"{path.name}: LAGOON reads GeoTIFF, .asc and x y z text files")


def model_points(bathy: Bathymetry, georef: Georef) -> np.ndarray:
    """Scattered points of a source in model coordinates (n, 3)."""
    if bathy.points is None:
        raise RasterError(f"{bathy.name} is a raster")
    points = np.asarray(bathy.points, float)
    big_x, big_y = points[:, 0], points[:, 1]
    crs = bathy.crs or georef.crs
    if crs != georef.crs:
        big_x, big_y = transformer(crs, georef.crs)(big_x, big_y)
    x, y = georef.to_model(big_x, big_y)
    z = points[:, 2] * (-1.0 if bathy.depths else 1.0) - georef.z_offset
    return np.column_stack([x, y, z])


def interpolate(points: np.ndarray, xs: np.ndarray, ys: np.ndarray, method: str = "tin"):
    """Heights of scattered model points (n, 3) on a grid: (len(ys), len(xs))."""
    from scipy.spatial import cKDTree

    if method not in METHODS:
        raise ValueError(f"method must be one of {', '.join(METHODS)}, got {method!r}")
    points = np.asarray(points, float)
    points = points[np.isfinite(points).all(axis=1)]
    if len(points) == 0:
        raise RasterError("no points with heights")
    gx, gy = np.meshgrid(np.asarray(xs, float), np.asarray(ys, float))
    targets = np.column_stack([gx.ravel(), gy.ravel()])
    tree = cKDTree(points[:, :2])
    if method == "tin" and len(points) >= 3:
        from scipy.interpolate import LinearNDInterpolator
        from scipy.spatial import QhullError

        try:
            values = LinearNDInterpolator(points[:, :2], points[:, 2])(targets)
        except QhullError:  # all on a line
            values = np.full(len(targets), np.nan)
    elif method == "idw" and len(points) >= 2:
        k = min(12, len(points))
        distance, index = tree.query(targets, k=k)
        weight = 1.0 / np.maximum(distance, 1e-9) ** 2
        values = (weight * points[index, 2]).sum(axis=1) / weight.sum(axis=1)
    else:
        values = np.full(len(targets), np.nan)
    holes = ~np.isfinite(values)
    if holes.any():
        values[holes] = points[tree.query(targets[holes])[1], 2]
    return values.reshape(gx.shape)


def fill_holes(heights: np.ndarray, xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
    """NaN cells of a grid given the nearest cell's height (all NaN: an error)."""
    good = np.isfinite(heights)
    if good.all():
        return heights
    if not good.any():
        raise RasterError("the bathymetry does not reach the domain")
    from scipy.spatial import cKDTree

    gx, gy = np.meshgrid(xs, ys)
    tree = cKDTree(np.column_stack([gx[good], gy[good]]))
    _, index = tree.query(np.column_stack([gx[~good], gy[~good]]))
    out = heights.copy()
    out[~good] = heights[good][index]
    return out


def grid_heights(
    bathy: Bathymetry, georef: Georef, xs: np.ndarray, ys: np.ndarray, method: str = "tin"
) -> np.ndarray:
    """Bed heights of a source on a model grid (len(ys), len(xs)), every cell with a
    height (the nearest data where the source has none)."""
    xs = np.asarray(xs, float)
    ys = np.asarray(ys, float)
    if bathy.raster is not None:
        raster = bathy.raster
        if raster.crs is None:
            raster.crs = georef.crs
        heights = warp(raster, georef, xs, ys) + georef.z_offset  # the datum's heights
        heights = heights * (-1.0 if bathy.depths else 1.0) - georef.z_offset
        return fill_holes(heights, xs, ys)
    return interpolate(model_points(bathy, georef), xs, ys, method)


#: bathymetry from the web: the choice -> the sources asked, the finest first
WEB = {
    "best": ("kartverket", "emodnet", "copernicus", "etopo", "gebco"),
    "kartverket": ("kartverket",),
    "emodnet": ("emodnet",),
    "copernicus": ("copernicus",),
    "etopo": ("etopo",),
    "gebco": ("gebco",),
}


def web_heights(
    choice: str,
    georef: Georef,
    xs: np.ndarray,
    ys: np.ndarray,
    spacing: float,
    fetch=None,
    progress: Progress | None = None,
) -> tuple[np.ndarray, list[str]]:
    """Bed heights on a model grid from the web (lagoonviz.core.geo.download): the
    sources of a choice in turn, each filling what the ones before had no data for,
    the rest from the nearest data. Returns the heights (model z) and the labels of
    the sources used."""
    from lagoonviz.core.geo.download import SOURCES, covers, download_all

    if choice not in WEB:
        raise ValueError(f"choose one of {', '.join(WEB)}, got {choice!r}")
    xs = np.asarray(xs, float)
    ys = np.asarray(ys, float)
    box = (float(xs[0]), float(xs[-1]), float(ys[0]), float(ys[-1]))
    heights = np.full((len(ys), len(xs)), np.nan)
    used: list[str] = []
    errors: list[str] = []
    progress = progress or _no_progress
    keys = WEB[choice]
    for n, key in enumerate(keys):
        if np.isfinite(heights).all():
            break
        progress(n / len(keys), f"Downloading from {SOURCES[key].label}…")
        if not covers(SOURCES[key], georef, box):
            errors.append(f"{SOURCES[key].label}: not here")
            continue
        try:
            rasters = download_all(key, georef, box, spacing, fetch)
        except (OSError, RasterError) as error:
            errors.append(str(error))
            continue
        progress((n + 0.8) / len(keys), f"Onto the grid: {SOURCES[key].label}")
        for raster in rasters:  # (Copernicus: one per 1° tile)
            found = warp(raster, georef, xs, ys)  # model z (z_offset taken away)
            missing = ~np.isfinite(heights) & np.isfinite(found)
            if missing.any():
                heights[missing] = found[missing]
                if SOURCES[key].label not in used:
                    used.append(SOURCES[key].label)
    if not np.isfinite(heights).any():
        raise RasterError("no bathymetry for the domain: " + "; ".join(errors or ["no data"]))
    return fill_holes(heights, xs, ys), used


def geo_grid(
    domain: tuple[float, float, float, float], spacing: float, margin: float | None = None
):
    """The points geo.dat is written at: the domain (x0, x1, y0, y1) and a margin
    (default two spacings) round it, every ``spacing`` metres (more if it would be
    more than MOST_POINTS)."""
    x0, x1, y0, y1 = domain
    if not spacing > 0:
        spacing = max(x1 - x0, y1 - y0) / 200.0
    margin = 2.0 * spacing if margin is None else margin
    while True:
        nx = int(np.floor((x1 - x0 + 2 * margin) / spacing)) + 1
        ny = int(np.floor((y1 - y0 + 2 * margin) / spacing)) + 1
        if nx * ny <= MOST_POINTS:
            break
        spacing *= 1.25
    xs = x0 - margin + spacing * np.arange(nx)
    ys = y0 - margin + spacing * np.arange(ny)
    return xs, ys


def write_geo_dat(
    path: str | Path,
    xs: np.ndarray,
    ys: np.ndarray,
    heights: np.ndarray,
    progress: Progress | None = None,
) -> int:
    """Write a grid of heights as geo.dat (x y z per line, model coordinates); NaN
    cells are left out. Returns the number of points. A cancel (``progress``
    raising Cancelled) leaves the geo.dat there was."""
    progress = progress or _no_progress
    gx, gy = np.meshgrid(xs, ys)
    points = np.column_stack([gx.ravel(), gy.ravel(), np.asarray(heights, float).ravel()])
    points = points[np.isfinite(points).all(axis=1)]
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".part")
    try:
        with temporary.open("w", encoding="ascii") as file:
            for start in range(0, len(points), 100_000):  # C formatting, in parts
                progress(start / max(len(points), 1), f"Writing geo.dat: {start:,} points")
                part = points[start : start + 100_000]
                file.write(("%.4f %.4f %.4f\n" * len(part)) % tuple(part.ravel()))
    except Cancelled:
        temporary.unlink(missing_ok=True)
        raise
    temporary.replace(path)
    progress(1.0, f"geo.dat: {len(points):,} points")
    return len(points)
