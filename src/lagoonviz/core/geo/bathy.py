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


#: the sources that give the sea (the bed under water), in no particular order
SEA_SOURCES = ("emodnet", "etopo", "gebco")
#: the sea next to the coast at least this deep [m], and deeper by SHORE_SLOPE per
#: metre from the coast, up to SHORE_DEEPEST: the shoreline where the land mask has
#: it, whatever a coarse sea source says there
SHORE_DEPTH = 0.1
SHORE_SLOPE = 0.02
SHORE_DEEPEST = 1.0
#: open sea for certain: a sea source this deep [m] (EMODnet: 1 m; it has no land)
OPEN_SEA = 10.0


def web_heights(
    choice: str,
    georef: Georef,
    xs: np.ndarray,
    ys: np.ndarray,
    spacing: float,
    fetch=None,
    progress: Progress | None = None,
    lat_below_msl: float = 0.0,
    coastline: bool = True,
    nearshore: str = "dean",
    dean_a: float | None = None,
    overlays=(),
    method: str = "tin",
    trust_depth: float | None = None,
) -> tuple[np.ndarray, list[str]]:
    """Bed heights on a model grid from the web (lagoonviz.core.geo.download): the
    sources of a choice in turn, each filling what the ones before had no data for,
    the rest from the nearest data. Returns the heights (model z) and the labels of
    the sources used.

    ``coastline`` (with land and sea sources, as "best"): Copernicus' 30 m heights
    decide where the coast is (:func:`coastal`), not the coarser sea sources;
    ``nearshore``: how the bed between that coast and the sea source's trusted
    depths is made ("dean", "smooth" or "flat"; :mod:`~lagoonviz.core.geo.nearshore`),
    ``dean_a`` Dean's A for "dean"; ``trust_depth``: the sea source's depths
    shallower than this [m] are not trusted either (ETOPO's and GEBCO's flat 1 or 2
    m over tidal flats, harbours and creeks).
    ``lat_below_msl``: how far EMODnet's datum, the lowest astronomical tide, lies
    below mean sea level [m], to put its depths on the land's zero.
    ``overlays``: data of one's own (:class:`~lagoonviz.core.geo.nearshore.Overlay`)
    laid over the download, the later over the earlier, blended in at their edges
    (``method``: how their soundings are interpolated); control soundings
    (ICESat-2) guide the nearshore profile only."""
    from lagoonviz.core.geo import nearshore as near
    from lagoonviz.core.geo.download import SOURCES, covers, download_all

    if choice not in WEB:
        raise ValueError(f"choose one of {', '.join(WEB)}, got {choice!r}")
    if nearshore not in near.NEARSHORE:
        raise ValueError(f"nearshore must be one of {', '.join(near.NEARSHORE)}, got {nearshore!r}")
    xs = np.asarray(xs, float)
    ys = np.asarray(ys, float)
    box = (float(xs[0]), float(xs[-1]), float(ys[0]), float(ys[-1]))
    keys = WEB[choice]
    coastline = coastline and "copernicus" in keys and any(k in SEA_SOURCES for k in keys)
    layers: dict[str, np.ndarray] = {}
    errors: list[str] = []
    progress = progress or _no_progress
    grid_spacing = (
        abs(float(ys[1] - ys[0])) if len(ys) > 1 else 1.0,
        abs(float(xs[1] - xs[0])) if len(xs) > 1 else 1.0,
    )
    known = np.full((len(ys), len(xs)), np.nan)  # one's own data, the later over the earlier
    control = np.full((len(ys), len(xs)), np.nan)
    dense = []
    for overlay in overlays:
        progress(0.0, f"Onto the grid: {overlay.name}")
        values, covered = near.on_grid(overlay, georef, xs, ys, method)
        if overlay.control:
            control = np.where(covered, values, control)
        else:
            known = np.where(covered, values, known)
            dense.append((values, covered))
    for n, key in enumerate(keys):
        if _settled(layers, coastline):
            break
        progress(n / len(keys), f"Downloading from {SOURCES[key].label}…")
        if not covers(SOURCES[key], georef, box):
            errors.append(f"{SOURCES[key].label}: not here")
            continue
        try:
            rasters = download_all(
                key, georef, box, spacing, fetch, land_only=not (coastline and key == "copernicus")
            )
        except (OSError, RasterError) as error:
            errors.append(str(error))
            continue
        progress((n + 0.8) / len(keys), f"Onto the grid: {SOURCES[key].label}")
        layer = np.full((len(ys), len(xs)), np.nan)
        for raster in rasters:  # (Copernicus: one per 1° tile)
            found = warp(raster, georef, xs, ys)  # model z (z_offset taken away)
            missing = ~np.isfinite(layer) & np.isfinite(found)
            layer[missing] = found[missing]
        if key == "emodnet" and lat_below_msl:
            layer = layer - float(lat_below_msl)  # depths below LAT: heights over MSL
        layers[key] = layer
    if coastline and "copernicus" in layers:
        heights, from_keys = coastal(
            layers,
            xs,
            ys,
            keys,
            nearshore=nearshore,
            dean_a=dean_a,
            known=known if dense else None,
            control=control if np.isfinite(control).any() else None,
            trust_depth=trust_depth,
        )
    else:
        heights = np.full((len(ys), len(xs)), np.nan)
        from_keys = []
        for key in keys:
            layer = layers.get(key)
            if layer is None:
                continue
            if key == "copernicus":  # land only: its 0 is the sea, left to the others
                layer = np.where(layer > 0.0, layer, np.nan)
            missing = ~np.isfinite(heights) & np.isfinite(layer)
            if missing.any():
                heights[missing] = layer[missing]
                from_keys.append(key)
    for values, covered in dense:  # one's own data over the download
        heights = near.lay(heights, values, covered, grid_spacing)
    if not np.isfinite(heights).any():
        raise RasterError("no bathymetry for the domain: " + "; ".join(errors or ["no data"]))
    used = [SOURCES[k].label for k in keys if k in from_keys]
    used += [o.name for o in overlays]
    return fill_holes(heights, xs, ys), used


def _settled(layers: dict[str, np.ndarray], coastline: bool) -> bool:
    """Whether the sources so far give every point: a height (with the coastline:
    Kartverket's, Copernicus' land, or a sea source's depth; Copernicus first)."""
    if not layers:
        return False
    shape = next(iter(layers.values())).shape
    fine = layers.get("kartverket")
    done = np.isfinite(fine) if fine is not None else np.zeros(shape, bool)
    if not coastline:
        for layer in layers.values():
            done |= np.isfinite(layer)
        return bool(done.all())
    for key in SEA_SOURCES:  # the open sea, far from any coast: no mask needed
        if key in layers:
            with np.errstate(invalid="ignore"):
                done |= layers[key] < -OPEN_SEA
    if "copernicus" in layers:
        cop = layers["copernicus"]
        done |= np.isfinite(cop) & (cop > 0.0)
        for key in SEA_SOURCES:
            if key in layers:
                done |= np.isfinite(layers[key])
    return bool(done.all())


def coastal(
    layers: dict[str, np.ndarray],
    xs,
    ys,
    order=(),
    nearshore: str = "flat",
    dean_a: float | None = None,
    trust: float | None = None,
    known: np.ndarray | None = None,
    control: np.ndarray | None = None,
    trust_depth: float | None = None,
) -> tuple[np.ndarray, list[str]]:
    """Heights where a fine land mask sets the coast (Copernicus GLO-30: land is above
    0) and coarse sources give the sea's depths:

    - land: Copernicus' heights;
    - the sea (low ground joined to the open sea): the sea sources' depths, at least
      SHORE_DEPTH deep at the coast, deeper by SHORE_SLOPE per metre out to
      SHORE_DEEPEST, so the shoreline lies where Copernicus has it (a coarse cell
      over the coast no longer puts land into the sea, or sea onto the land);
    - next to the coast, with ``nearshore`` "dean" or "smooth": the nearshore band,
      where the coarse sea cannot be trusted (closer to the coast than TRUST_CELLS
      of its cells, ``trust`` metres if given; or shallower than ``trust_depth``,
      default SHORE_DEPTH: a coarse source's flat shallows; or no sea data), made
      again from the coastline out to the trusted depths
      (:func:`~lagoonviz.core.geo.nearshore.profile`; ``dean_a``: Dean's A), meeting
      ``known`` (one's own data, model z, NaN where none) and ``control`` (sparse
      soundings) on its way;
    - low ground cut off from the sea (polders behind dikes, not joined to the open
      sea: no Copernicus tile, a sea source OPEN_SEA deep, or Copernicus' water at
      0 m where the sea source has the sea): Copernicus' heights;
    - Kartverket (land and sea at up to 1 m) over all of them where it has data.

    ``layers``: key -> heights on the grid (NaN: no data; Copernicus NaN: no tile,
    the open sea). Returns the heights and the keys used."""
    from scipy import ndimage

    from lagoonviz.core.geo import nearshore as near

    cop = layers["copernicus"]
    shape = cop.shape
    sea_keys = [k for k in (order or SEA_SOURCES) if k in SEA_SOURCES and k in layers]
    sea = np.full(shape, np.nan)
    source = np.full(shape, -1)  # which sea source gave each point
    for index, key in enumerate(sea_keys):
        missing = ~np.isfinite(sea) & np.isfinite(layers[key])
        sea[missing] = layers[key][missing]
        source[missing] = index
    land = np.isfinite(cop) & (cop > 0.0)
    low = ~land
    emodnet = layers.get("emodnet")
    open_sea = ~np.isfinite(cop) | (np.isfinite(sea) & (sea < -OPEN_SEA))
    # Copernicus' sea is its water mask's 0 m (polders lie below it): a shallow
    # coast, nowhere OPEN_SEA deep in the domain, is still the sea
    with np.errstate(invalid="ignore"):
        open_sea |= (cop == 0.0) & np.isfinite(sea) & (sea < 0.0)
    if emodnet is not None:
        open_sea |= np.isfinite(emodnet) & (emodnet < -1.0)
    labels, _ = ndimage.label(low)  # joined along the grid's rows and columns
    joined = np.unique(labels[low & open_sea])
    water = np.isin(labels, joined[joined > 0])
    dx = abs(float(xs[1] - xs[0])) if len(xs) > 1 else 1.0
    dy = abs(float(ys[1] - ys[0])) if len(ys) > 1 else 1.0
    if land.any():
        distance = ndimage.distance_transform_edt(~land, sampling=(dy, dx))
    else:
        distance = np.full(shape, np.inf)
    depths = sea
    if water.any() and not np.isfinite(sea[water]).all():
        if np.isfinite(sea).any():  # no sea data there: the nearest sea data's
            _, (r, c) = ndimage.distance_transform_edt(~np.isfinite(sea), return_indices=True)
            depths = sea[r, c]
        else:
            depths = np.zeros(shape)
    shore = np.minimum(SHORE_DEPTH + SHORE_SLOPE * distance, SHORE_DEEPEST)
    out = np.full(shape, np.nan)
    out[land] = cop[land]
    out[water] = np.minimum(depths, -shore)[water]
    if nearshore != "flat" and water.any() and land.any():
        out = _nearshore(
            out,
            water,
            land,
            depths,
            source,
            sea_keys,
            distance,
            (dy, dx),
            mode=nearshore,
            a=near.DEAN_A if dean_a is None else dean_a,
            trust=trust,
            trust_depth=trust_depth,
            known=known,
            control=control,
        )
    cut_off = low & ~water  # polders: their own heights, else the sea sources'
    out[cut_off] = np.where(np.isfinite(cop), cop, sea)[cut_off]
    used = ["copernicus"] if land.any() or (cut_off & np.isfinite(cop)).any() else []
    taken = source[water | (cut_off & ~np.isfinite(cop))]
    used += [k for i, k in enumerate(sea_keys) if (taken == i).any()]
    fine = layers.get("kartverket")
    if fine is not None and np.isfinite(fine).any():
        out = np.where(np.isfinite(fine), fine, out)
        used.append("kartverket")
    return out, used


def _nearshore(
    out,
    water,
    land,
    depths,
    source,
    sea_keys,
    distance,
    spacing,
    mode,
    a,
    trust=None,
    trust_depth=None,
    known=None,
    control=None,
):
    """The nearshore band of :func:`coastal` made again (see there)."""
    from lagoonviz.core.geo import nearshore as near
    from lagoonviz.core.geo.download import SOURCES

    if trust is None:  # each sea source trusted TRUST_CELLS of its own cells out
        reach = np.zeros(out.shape)
        for index, key in enumerate(sea_keys):
            reach[source == index] = near.TRUST_CELLS * SOURCES[key].finest
    else:
        reach = np.full(out.shape, float(trust))
    shallow = SHORE_DEPTH if trust_depth is None else max(float(trust_depth), SHORE_DEPTH)
    with np.errstate(invalid="ignore"):
        # untrusted: near the coast, no sea data of its own, or shallower than trusted
        band = water & ((distance < reach) | (source < 0) | ~(depths < -shallow))
        # trusted: the coarse depths as they are, only kept wet (no flat 1 m)
        trusted = water & ~band
        out[trusted] = np.minimum(depths, -SHORE_DEPTH)[trusted]
    # the beach profile reaches as far as the coarse source is not trusted; beyond
    # (wide shallows the source has flat or dry) the bed is the smoothest between data
    reach = np.where(reach > 0, reach, np.max(reach, initial=0.0))
    return near.profile(
        out, band, land, np.minimum(distance, reach), spacing, mode, a, known, control
    )


#: geo.dat's spacing asked for as the finest the data has
FINEST = "finest"
#: metres in a degree of latitude
DEGREE = 111_320.0


def finest_web(choice: str, georef: Georef, box) -> tuple[float, str]:
    """The finest cell of the sources of a download choice that have data over a
    model rectangle (x0, x1, y0, y1): (metres, the source's label)."""
    from lagoonviz.core.geo.download import SOURCES, covers

    found = [
        (SOURCES[key].finest, SOURCES[key].label)
        for key in WEB[choice]
        if covers(SOURCES[key], georef, box)
    ]
    if not found:
        raise RasterError("no bathymetry source has data for the domain")
    return min(found)


def finest_file(bathy: Bathymetry) -> float:
    """The finest spacing a bathymetry file has [m]: a raster's cell, or the median
    distance between scattered points (degrees taken as metres at their latitude)."""
    from lagoonviz.core.geo.crs import is_geographic

    geographic = bathy.crs not in (None, "model") and is_geographic(str(bathy.crs))
    if bathy.raster is not None:
        t = bathy.raster.transform
        if geographic:
            lat = float(np.mean(bathy.raster.bounds()[1::2]))
            dx = abs(t[1]) * DEGREE * np.cos(np.radians(lat))
            return float(np.sqrt(dx * abs(t[5]) * DEGREE))
        return bathy.raster.cell_size()
    points = np.asarray(bathy.points, float)[:, :2]
    if len(points) < 2:
        raise RasterError(f"{bathy.name}: too few points")
    if len(points) > 200_000:  # a fair sample, round the middle
        pick = np.random.default_rng(0).choice(len(points), 200_000, replace=False)
        points = points[pick]
    if geographic:
        lat = float(np.median(points[:, 1]))
        points = points * [DEGREE * np.cos(np.radians(lat)), DEGREE]
    from scipy.spatial import cKDTree

    distance, _ = cKDTree(points).query(points, k=2)
    near = distance[:, 1][distance[:, 1] > 0]
    if not len(near):
        raise RasterError(f"{bathy.name}: its points all lie on one spot")
    return float(np.median(near))


def geo_grid(
    domain: tuple[float, float, float, float],
    spacing: float,
    margin: float | None = None,
    most: int | None = None,
):
    """The points geo.dat is written at: the domain (x0, x1, y0, y1) and a margin
    (default two spacings) round it, every ``spacing`` metres (more if it would be
    more than ``most`` points, default MOST_POINTS; 0: no limit)."""
    most = MOST_POINTS if most is None else int(most)
    x0, x1, y0, y1 = domain
    if not spacing > 0:
        spacing = max(x1 - x0, y1 - y0) / 200.0
    margin = 2.0 * spacing if margin is None else margin
    while True:
        nx = int(np.floor((x1 - x0 + 2 * margin) / spacing)) + 1
        ny = int(np.floor((y1 - y0 + 2 * margin) / spacing)) + 1
        if most <= 0 or nx * ny <= most:
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
