# SPDX-License-Identifier: GPL-3.0-or-later
# Architect: Hans Bihs
"""Background maps in the model's frame.

A map tile server (or a GeoTIFF of the user's, e.g. an orthophoto) is in its own
CRS, the model in its georeference's, turned by the model's rotation. :func:`warp`
samples any :class:`~lagoonviz.core.geo.raster.GeoImage` on a grid of model
coordinates; :func:`background` puts tiles together over an area of the model and
warps them, so the renderer draws a plain image on a rectangle in model
coordinates, under the results.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from lagoon_format.geo import Georef

from lagoonviz.core.geo.crs import transformer
from lagoonviz.core.geo.raster import GeoImage, bilinear
from lagoonviz.core.geo.tiles import TileCache, TileSource, mosaic, resolution, zoom_for

#: the coarse grid the CRS transform is computed on, then interpolated (pixels)
STEP = 16


@dataclass
class ModelImage:
    """An image on a rectangle of model coordinates (x0, x1, y0, y1): row 0 at y0
    (the south of a model that is not turned), column 0 at x0."""

    image: np.ndarray  # (rows, cols, 4) uint8 or (rows, cols) float
    extent: tuple[float, float, float, float]
    attribution: str = ""
    #: the zoom level of the tiles (tile sources)
    zoom: int | None = None
    #: tiles that could not be had
    missing: int = 0


def model_grid(extent: tuple[float, float, float, float], pixels: int):
    """Cell centres of a grid over a model rectangle, at most ``pixels`` along its
    longer side: (xs, ys, cell size)."""
    x0, x1, y0, y1 = extent
    size = max(x1 - x0, y1 - y0) / max(int(pixels), 2)
    nx = max(2, int(math.ceil((x1 - x0) / size)))
    ny = max(2, int(math.ceil((y1 - y0) / size)))
    xs = x0 + (np.arange(nx) + 0.5) * (x1 - x0) / nx
    ys = y0 + (np.arange(ny) + 0.5) * (y1 - y0) / ny
    return xs, ys, size


def _pixels_of(source: GeoImage, georef: Georef, xs: np.ndarray, ys: np.ndarray):
    """Fractional (col, row) in the source of every model grid point: computed on a
    coarse grid (every STEP points) and interpolated, the CRS transforms being smooth
    over a few pixels."""
    nx, ny = len(xs), len(ys)
    ix = np.unique(np.r_[np.arange(0, nx, STEP), nx - 1])
    iy = np.unique(np.r_[np.arange(0, ny, STEP), ny - 1])
    gx, gy = np.meshgrid(xs[ix], ys[iy])
    big_x, big_y = georef.to_world(gx, gy)
    if source.crs is not None and source.crs != georef.crs:
        big_x, big_y = transformer(georef.crs, source.crs)(big_x, big_y)
    col, row = source.pixel(big_x, big_y)
    # back onto the full grid: separable linear interpolation
    fx = np.interp(np.arange(nx), ix, np.arange(len(ix)))
    fy = np.interp(np.arange(ny), iy, np.arange(len(iy)))
    coarse = np.stack([col, row], axis=-1)  # (len(iy), len(ix), 2)
    full = bilinear(coarse, fx[None, :].repeat(ny, 0), fy[:, None].repeat(nx, 1))
    return full[..., 0], full[..., 1]


def warp(source: GeoImage, georef: Georef, xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
    """A raster sampled at model grid points (xs along x, ys along y): (len(ys),
    len(xs)) heights in the model's vertical frame (NaN outside), or (len(ys),
    len(xs), 4) uint8 colours (transparent outside)."""
    col, row = _pixels_of(source, georef, np.asarray(xs, float), np.asarray(ys, float))
    if source.is_image:
        data = source.data
        if data.shape[-1] == 3:
            data = np.concatenate([data, np.full(data.shape[:2] + (1,), 255, np.uint8)], -1)
        out = np.empty(col.shape + (4,), np.uint8)
        for start in range(0, col.shape[0], 256):  # in bands: a large map stays small
            part = bilinear(data, col[start : start + 256] - 0.5, row[start : start + 256] - 0.5)
            out[start : start + 256] = np.clip(np.rint(part), 0, 255).astype(np.uint8)
        return out
    values = bilinear(source.values(), col - 0.5, row - 0.5)
    return values - georef.z_offset


def background(
    georef: Georef,
    extent: tuple[float, float, float, float],
    source: TileSource | GeoImage,
    pixels: int = 2048,
    cache: TileCache | None = None,
    attribution: str = "",
    progress=None,
) -> ModelImage:
    """A background map over a model rectangle (x0, x1, y0, y1): tiles of a tile
    source (fetched or from the cache), or a GeoTIFF image, warped into the model's
    frame, at most ``pixels`` along the longer side. ``progress(done, total)``
    hears of the tiles had."""
    xs, ys, size = model_grid(extent, pixels)
    zoom = None
    missing = 0
    if isinstance(source, TileSource):
        # the model rectangle's corners and edges on the Web Mercator plane
        edge = np.linspace(0.0, 1.0, 9)
        x0, x1, y0, y1 = extent
        mx = np.r_[x0 + (x1 - x0) * edge, np.full(9, x1), x1 - (x1 - x0) * edge, np.full(9, x0)]
        my = np.r_[np.full(9, y0), y0 + (y1 - y0) * edge, np.full(9, y1), y1 - (y1 - y0) * edge]
        big_x, big_y = georef.to_world(mx, my)
        to_mercator = transformer(georef.crs, "EPSG:3857")
        merc_x, merc_y = to_mercator(big_x, big_y)
        bounds = (
            float(merc_x.min()),
            float(merc_y.min()),
            float(merc_x.max()),
            float(merc_y.max()),
        )
        # a model metre is 1/cos(latitude) Web Mercator metres
        _, lat = transformer("EPSG:3857", "EPSG:4326")(np.mean(merc_x), np.mean(merc_y))
        scale = 1.0 / max(math.cos(math.radians(float(lat))), 1e-6)
        zoom = zoom_for(size * scale, source.max_zoom)
        tiles = mosaic(source, bounds, zoom, cache, progress=progress)
        zoom, missing = tiles.zoom, tiles.missing
        res = resolution(tiles.zoom)
        b = tiles.bounds
        image = GeoImage(tiles.image, (b[0], res, 0.0, b[3], 0.0, -res), "EPSG:3857")
        attribution = attribution or source.attribution
    else:
        image = source
        if image.crs is None:
            image.crs = georef.crs
    warped = warp(image, georef, xs, ys)
    if warped.ndim == 2:
        raise ValueError(f"{image.name or 'the raster'} holds heights, not a picture")
    return ModelImage(warped, tuple(float(v) for v in extent), attribution, zoom, missing)


def expanded(
    box: tuple[float, float, float, float], margin: float
) -> tuple[float, float, float, float]:
    """A model rectangle (x0, x1, y0, y1) grown by ``margin`` times its larger side
    on every side."""
    x0, x1, y0, y1 = box
    grow = margin * max(x1 - x0, y1 - y0, 1e-9)
    return x0 - grow, x1 + grow, y0 - grow, y1 + grow
