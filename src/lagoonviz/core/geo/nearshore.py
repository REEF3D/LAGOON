# SPDX-License-Identifier: GPL-3.0-or-later
# Architect: Hans Bihs
"""The bed next to the coast, where a downloaded sea source is too coarse to know it.

ETOPO's and GEBCO's 450 m cells (EMODnet's 115 m) over a coast mix land and sea:
their first few hundred metres out are a guess, often land or a few decimetres.
:func:`~lagoonviz.core.geo.bathy.coastal` puts the coastline where Copernicus' 30 m
land has it; :func:`profile` makes the bed between that coastline and the first
depths of the sea source that can be trusted (TRUST_CELLS of its cells out), the
*nearshore band*:

- ``"dean"``: an equilibrium beach profile, A x^(2/3) deep at x metres from the
  coast (Bruun 1954, Dean 1977), bent to meet the sea source where it is trusted
  and one's own data where there is some (the difference is carried across the
  band by Laplace's equation);
- ``"smooth"``: the smoothest surface (Laplace's equation) from the coastline (0 m)
  to the trusted sea and one's own data: for creeks, estuaries and mudflats, whose
  bed is no beach;
- ``"flat"``: as before: the coarse depths, held at least 0.1 m deep at the coast
  and 2 cm deeper per metre out, to 1 m (a flat 1 m where the source has the sea
  shallower).

Data of one's own (:class:`Overlay`) are laid over a download: surveys, chart
soundings, an intertidal terrain model from satellite waterlines, a coastal terrain
model such as DeltaDTM. ICESat-2's nearshore soundings (ATL24,
:mod:`~lagoonviz.core.geo.icesat2`) only guide the profile in the nearshore band:
their tracks are too sparse to lie over the sea as they are.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

#: how the bed next to the coast is made: key -> what it is
NEARSHORE = {
    "dean": "a beach profile (Dean), bent to meet the sea's data",
    "smooth": "the smoothest bed from the coast to the sea's data",
    "flat": "the coarse depths, at least 0.1 m to 1 m deep (as before)",
}
#: the default for downloads
DEFAULT = "dean"
#: Dean's A [m^(1/3)]: about 0.21 D^0.48 for a grain size D in mm (Moore 1982,
#: Dean 1987): 0.07 for 0.1 mm, 0.1 for 0.2 mm, 0.14 for 0.4 mm sand
DEAN_A = 0.1
#: a coarse sea source is trusted this many of its cells out from the coast
TRUST_CELLS = 1.5
#: data of one's own blend into the download over this many grid cells at its edge
FEATHER_CELLS = 5
#: the sea next to the coast at least this deep [m] (the same as bathy.SHORE_DEPTH)
SHORE_DEPTH = 0.1
#: at most this many unknowns solved directly (sparse LU); more by conjugate gradients
DIRECT = 300_000


def dean(distance: np.ndarray, a: float = DEAN_A) -> np.ndarray:
    """Dean's equilibrium profile: the depth [m] at ``distance`` metres from the
    shoreline, A x^(2/3)."""
    return float(a) * np.cbrt(np.square(np.maximum(np.asarray(distance, float), 0.0)))


def harmonic(target: np.ndarray, unknown: np.ndarray, spacing=(1.0, 1.0)) -> np.ndarray:
    """``unknown`` cells of a grid given the solution of Laplace's equation, the
    other cells with a value in ``target`` (not NaN) its boundary; cells without a
    value, and the grid's edges, are closed (no flux). ``spacing``: (dy, dx), the
    rows' and columns' spacing. Unknown cells cut off from every boundary value get
    0. Returns ``target`` with the unknown cells filled (NaN elsewhere stays)."""
    from scipy import ndimage, sparse

    target = np.asarray(target, float)
    unknown = np.asarray(unknown, bool)
    rows, cols = target.shape
    fixed = np.isfinite(target) & ~unknown
    out = np.where(fixed, target, np.nan)
    if not unknown.any():
        return out
    labels, _ = ndimage.label(unknown)  # joined along rows and columns
    touch = ndimage.binary_dilation(fixed) & unknown
    reached = np.unique(labels[touch])
    solvable = np.isin(labels, reached[reached > 0])
    out[unknown & ~solvable] = 0.0
    cells = np.flatnonzero(solvable)
    if not len(cells):
        return out
    index = np.full(rows * cols, -1, np.int64)
    index[cells] = np.arange(len(cells))
    r, c = np.divmod(cells, cols)
    flat_target = np.where(fixed, target, 0.0).ravel()
    flat_fixed = fixed.ravel()
    dy, dx = (abs(float(s)) or 1.0 for s in spacing)
    diag = np.zeros(len(cells))
    rhs = np.zeros(len(cells))
    i_parts, j_parts, v_parts = [], [], []
    for dr, dc, weight in ((0, 1, dx**-2), (0, -1, dx**-2), (1, 0, dy**-2), (-1, 0, dy**-2)):
        rr, cc = r + dr, c + dc
        inside = (rr >= 0) & (rr < rows) & (cc >= 0) & (cc < cols)
        neighbour = np.where(inside, rr * cols + cc, 0)
        other = np.where(inside, index[neighbour], -1)
        boundary = inside & flat_fixed[neighbour]
        joined = other >= 0
        diag += weight * (joined | boundary)
        rhs += weight * np.where(boundary, flat_target[neighbour], 0.0)
        i_parts.append(np.flatnonzero(joined))
        j_parts.append(other[joined])
        v_parts.append(np.full(int(joined.sum()), -weight))
    n = len(cells)
    matrix = sparse.csr_matrix(
        (
            np.concatenate([diag, *v_parts]),
            (np.concatenate([np.arange(n), *i_parts]), np.concatenate([np.arange(n), *j_parts])),
        ),
        shape=(n, n),
    )
    if n <= DIRECT:
        from scipy.sparse.linalg import spsolve

        values = spsolve(matrix.tocsc(), rhs)
    else:
        values = _conjugate_gradients(matrix, rhs, diag, out, fixed, solvable)
    filled = out.ravel()
    filled[cells] = values
    return filled.reshape(rows, cols)


def _conjugate_gradients(matrix, rhs, diag, out, fixed, solvable):
    """A large band: conjugate gradients (Jacobi), from the nearest boundary value."""
    from scipy import ndimage
    from scipy.sparse.linalg import LinearOperator, cg

    _, (nr, nc) = ndimage.distance_transform_edt(~fixed, return_indices=True)
    start = out[nr, nc][solvable]
    precondition = LinearOperator(matrix.shape, matvec=lambda v: v / diag, dtype=float)
    try:
        values, _ = cg(matrix, rhs, x0=start, rtol=1e-9, maxiter=20_000, M=precondition)
    except TypeError:  # SciPy before 1.12: tol
        values, _ = cg(matrix, rhs, x0=start, tol=1e-9, maxiter=20_000, M=precondition)
    return values


def profile(
    heights: np.ndarray,
    band: np.ndarray,
    land: np.ndarray,
    distance: np.ndarray,
    spacing,
    mode: str = DEFAULT,
    a: float = DEAN_A,
    known: np.ndarray | None = None,
    control: np.ndarray | None = None,
) -> np.ndarray:
    """The bed in the nearshore ``band`` (cells of the sea next to the coast whose
    coarse depths are not trusted) made again (``mode`` "dean" or "smooth"), between
    the coastline (0 m next to ``land``) and the heights round the band (the
    trusted sea). ``distance``: each cell's distance to the land [m]; ``spacing``:
    (dy, dx). ``known``: heights of one's own data (NaN where none), kept wherever
    they lie in the sea and met by the profile; ``control``: sparse soundings
    (ICESat-2), kept and met in the band only. Only the band's cells change (one's
    data elsewhere is laid on afterwards, :func:`lay`). The band stays wet (at least
    SHORE_DEPTH deep) away from one's own data; near it, the bed follows the data,
    dry flats included. Returns the heights (a copy)."""
    if mode not in ("dean", "smooth"):
        raise ValueError(f"nearshore must be 'dean' or 'smooth' here, got {mode!r}")
    out = np.array(heights, float)
    band = np.asarray(band, bool)
    if not band.any():
        return out
    sea = ~np.asarray(land, bool) & np.isfinite(out)
    base = -dean(distance, a) if mode == "dean" else np.zeros(out.shape)
    data = np.full(out.shape, np.nan)
    if known is not None:  # on land too: there the coastline is the data's, not 0 m
        data = np.where(np.isfinite(known), known, data)
    if control is not None:
        data = np.where(np.isfinite(control) & band & ~np.isfinite(data), control, data)
    target = np.full(out.shape, np.nan)  # the departure from the base profile
    target[land] = 0.0  # the coastline at 0 m
    trusted = sea & ~band
    target[trusted] = out[trusted] - base[trusted]
    given = np.isfinite(data)
    target[given] = data[given] - base[given]
    unknown = band & ~given
    filled = harmonic(target, unknown, spacing) + base
    made = unknown & np.isfinite(filled)
    if given.any():  # wet away from one's data; the data's own near it
        from scipy import ndimage

        dy, dx = (abs(float(s)) or 1.0 for s in spacing)
        near = ndimage.distance_transform_edt(~given, sampling=(dy, dx))
        wet = near > FEATHER_CELLS * max(dx, dy)
    else:
        wet = np.ones(out.shape, bool)
    filled = np.where(wet, np.minimum(filled, -SHORE_DEPTH), filled)
    out[made] = filled[made]
    out[given & band] = data[given & band]
    return out


def lay(heights: np.ndarray, values: np.ndarray, covered: np.ndarray, spacing) -> np.ndarray:
    """``values`` laid over ``heights`` where ``covered``, blended in over
    FEATHER_CELLS grid cells from the edge of what they cover (a seam without a
    step). Returns the heights (a copy)."""
    from scipy import ndimage

    out = np.array(heights, float)
    covered = np.asarray(covered, bool) & np.isfinite(values)
    if not covered.any():
        return out
    dy, dx = (abs(float(s)) or 1.0 for s in spacing)
    inside = ndimage.distance_transform_edt(covered, sampling=(dy, dx))
    weight = np.clip(inside / (FEATHER_CELLS * max(dx, dy)), 0.0, 1.0)
    weight = np.where(np.isfinite(out), weight, 1.0)
    blended = weight * np.where(covered, values, 0.0) + (1.0 - weight) * np.nan_to_num(out)
    out[covered] = blended[covered]
    return out


# ------------------------------------------------------------------- one's own data
@dataclass
class Overlay:
    """Bathymetry of one's own laid over a download: a raster or soundings
    (:class:`~lagoonviz.core.geo.bathy.Bathymetry`), its zero ``below_msl`` metres
    below mean sea level (a chart's datum; 0: on mean sea level, as the downloaded
    land), and whether it is ``control`` only: sparse soundings that guide the
    nearshore profile (ICESat-2) instead of lying over the sea."""

    bathy: object
    below_msl: float = 0.0
    control: bool = False
    #: how it was given (the recipe): path, crs, depths, below_msl
    spec: dict = field(default_factory=dict)

    @property
    def name(self) -> str:
        return str(getattr(self.bathy, "name", "") or Path(self.spec.get("path", "")).name)


def spec_of(entry) -> dict:
    """An overlay as given (a path, or a dict with ``path`` and optionally ``crs``,
    ``depths``, ``below_msl``, ``columns``) in its full form."""
    if isinstance(entry, str | Path):
        entry = {"path": str(entry)}
    if not isinstance(entry, dict) or not entry.get("path"):
        raise ValueError(f"an overlay needs a path, got {entry!r}")
    spec = {
        "path": str(Path(str(entry["path"])).expanduser()),
        "crs": entry.get("crs") or None,
        "depths": bool(entry.get("depths", False)),
        "below_msl": float(entry.get("below_msl", 0.0) or 0.0),
    }
    if entry.get("columns"):  # x, y, z of a points file (indices or the header's names)
        spec["columns"] = list(entry["columns"])
    return spec


def load_overlay(entry, bounds=None) -> Overlay:
    """An overlay from a file: a raster or soundings as for geo.dat
    (:func:`~lagoonviz.core.geo.bathy.load`); an ICESat-2 ATL24 file (``.h5``) or a
    folder of them as control soundings. ``bounds`` (lon0, lat0, lon1, lat1): only
    the ATL24 photons there."""
    from lagoonviz.core.geo import icesat2
    from lagoonviz.core.geo.bathy import load

    spec = spec_of(entry)
    path = Path(spec["path"])
    if icesat2.is_atl24(path):
        bathy = icesat2.soundings(icesat2.files_in(path), bounds)
        return Overlay(bathy, spec["below_msl"], True, spec)
    columns = spec.get("columns")
    if spec["crs"] == "model":  # already in the model's coordinates
        bathy = load(path, None, spec["depths"], columns)
        bathy.crs = "model"
        if bathy.raster is not None:
            bathy.raster.crs = "model"
    else:
        bathy = load(path, spec["crs"], spec["depths"], columns)
    return Overlay(bathy, spec["below_msl"], False, spec)


def on_grid(overlay: Overlay, georef, xs, ys, method: str = "tin"):
    """An overlay's heights on a model grid (model z: the georeference's z_offset
    taken away) and the cells it covers: (heights, covered), each (len(ys),
    len(xs)). A raster covers where it has data; soundings the cells within reach
    of a sounding (twice their typical spacing, at least 1.5 grid cells); control
    soundings only the cells they lie in (their mean)."""
    from scipy.spatial import cKDTree

    from lagoonviz.core.geo.basemap import warp
    from lagoonviz.core.geo.bathy import interpolate, model_points

    xs = np.asarray(xs, float)
    ys = np.asarray(ys, float)
    bathy = overlay.bathy
    shape = (len(ys), len(xs))
    if bathy.crs == "model":  # in model coordinates: no georeference between
        from lagoon_format.geo import Georef

        georef = Georef("model")
    if bathy.raster is not None:
        raster = bathy.raster
        if raster.crs is None:
            raster.crs = georef.crs
        values = warp(raster, georef, xs, ys) + georef.z_offset  # the file's own heights
        values = values * (-1.0 if bathy.depths else 1.0) - overlay.below_msl - georef.z_offset
        return values, np.isfinite(values)
    points = model_points(bathy, georef)
    points = points[np.isfinite(points).all(axis=1)]
    points[:, 2] -= overlay.below_msl
    if not len(points):
        return np.full(shape, np.nan), np.zeros(shape, bool)
    dx = abs(float(xs[1] - xs[0])) if len(xs) > 1 else 1.0
    dy = abs(float(ys[1] - ys[0])) if len(ys) > 1 else 1.0
    if overlay.control:
        col = np.rint((points[:, 0] - xs[0]) / (xs[1] - xs[0] if len(xs) > 1 else 1.0))
        row = np.rint((points[:, 1] - ys[0]) / (ys[1] - ys[0] if len(ys) > 1 else 1.0))
        inside = (col >= 0) & (col < len(xs)) & (row >= 0) & (row < len(ys))
        cell = (row[inside] * len(xs) + col[inside]).astype(np.int64)
        total = np.bincount(cell, points[inside, 2], minlength=len(xs) * len(ys))
        count = np.bincount(cell, minlength=len(xs) * len(ys))
        with np.errstate(invalid="ignore", divide="ignore"):
            values = (total / count).reshape(shape)
        return values, (count > 0).reshape(shape)
    inside = (
        (points[:, 0] >= xs.min() - 50 * dx)
        & (points[:, 0] <= xs.max() + 50 * dx)
        & (points[:, 1] >= ys.min() - 50 * dy)
        & (points[:, 1] <= ys.max() + 50 * dy)
    )
    points = points[inside]
    if len(points) == 0:
        return np.full(shape, np.nan), np.zeros(shape, bool)
    tree = cKDTree(points[:, :2])
    typical = 0.0
    if len(points) > 1:
        sample = points[:: max(1, len(points) // 100_000), :2]
        gap, _ = tree.query(sample, k=2)
        near = gap[:, 1][gap[:, 1] > 0]
        typical = float(np.median(near)) if len(near) else 0.0
    reach = max(2.0 * typical, 1.5 * max(dx, dy))
    gx, gy = np.meshgrid(xs, ys)
    gap, _ = tree.query(np.column_stack([gx.ravel(), gy.ravel()]), distance_upper_bound=reach)
    covered = np.isfinite(gap).reshape(shape)
    values = interpolate(points, xs, ys, method)
    return np.where(covered, values, np.nan), covered
