# SPDX-License-Identifier: GPL-3.0-or-later
# Architect: Hans Bihs
"""A river's channel carved into geo.dat where the terrain model has only its water.

Terrain models made from above (Copernicus GLO-30, the land of most national
models) see a river's water surface, not its bed: in geo.dat the river is a flat
strip at the water level, and a run has no channel to carry the flow. Without a
survey of the bed, the depth a river needs to carry its discharge follows from
Manning's formula for a wide channel (hydraulic radius about the depth):

    Q = W h^(5/3) S^(1/2) / n   =>   h = (n Q / (W S^(1/2)))^(3/5)

with the discharge Q [m³/s] (GloFAS's, or one's own), the water's width W [m] at
each cross-section, the water surface's slope S and Manning's n (0.025 for a
smooth sandy river to 0.05 for a rough or weedy one). :func:`carve` cuts a
channel of that mean depth under the water surface of the terrain model, between
the edges of its water at each cross-section along the river corridor: parabolic
(deepest in the middle, 1.5 h; the default) or trapezoidal (1.25 h with banks
over a fifth of the width each side). It only lowers the bed, and only where the
terrain is at the water level (within ``wet`` metres): the land of the corridor's
margin stays as it is.

The water level of each cross-section is the terrain model's in the middle half
of the water, made to fall (never rise) downstream; the slope is the straight line
through them over the whole reach, unless given.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

#: the channel's shapes
SHAPES = ("parabolic", "trapezoid")
#: the gentlest slope the depth is found with (a flat terrain model: no slope)
MIN_SLOPE = 1e-5
#: points across a cross-section
ACROSS = 65


def manning_depth(discharge, width, slope: float, manning: float):
    """The mean depth [m] a wide channel needs to carry ``discharge`` [m³/s] over
    ``width`` [m] at ``slope`` with Manning's n."""
    width = np.maximum(np.asarray(width, float), 1e-6)
    slope = max(float(slope), MIN_SLOPE)
    return (float(manning) * float(discharge) / (width * np.sqrt(slope))) ** 0.6


@dataclass
class Channel:
    """The channel carved: per cross-section along the river (upstream first) its
    distance [m], the water's width [m], the water level [m] and the mean depth
    [m] (NaN where the terrain model has no water); and how it was found."""

    along: np.ndarray
    width: np.ndarray
    level: np.ndarray
    depth: np.ndarray
    slope: float
    discharge: float
    manning: float
    shape: str
    lowered: int = 0  # geo.dat points lowered
    spacing: float = 0.0  # geo.dat's spacing [m]

    def text(self) -> str:
        """One line about it, for the panel and the log."""
        ok = np.isfinite(self.depth)
        if not ok.any():
            return "no water found along the river in geo.dat: nothing carved"
        d, w = self.depth[ok], self.width[ok]
        text = (
            f"Q {self.discharge:g} m³/s, n {self.manning:g}, slope {self.slope:.2g}: mean "
            f"depth {d.min():.1f}–{d.max():.1f} m over {self.along[-1] / 1000:.1f} km, "
            f"water {w.min():.0f}–{w.max():.0f} m wide; {self.lowered:,} points lowered"
        )
        if self.spacing and np.median(w) < 3 * self.spacing:
            text += f" (geo.dat every {self.spacing:g} m: too coarse for this river)"
        return text


def _sections(corridor, count: int) -> tuple[np.ndarray, np.ndarray]:
    """Nodes of ``count`` straight cross-sections from the right bank to the left:
    x and y, each (count, ACROSS)."""
    from lagoonviz.core.corridor import resample

    right = resample(corridor.bank_line("right"), count)
    left = resample(corridor.bank_line("left"), count)
    t = np.linspace(0.0, 1.0, ACROSS)[None, :, None]
    nodes = (1.0 - t) * right[:, None, :] + t * left[:, None, :]
    return nodes[..., 0], nodes[..., 1]


def _wet_span(heights: np.ndarray, level: float, wet: float) -> tuple[int, int] | None:
    """The run of nodes across at the water level (within ``wet`` m) round the
    lowest node of the middle half: (first, last) index, or None."""
    n = len(heights)
    middle = slice(n // 4, n - n // 4)
    if not np.isfinite(heights[middle]).any():
        return None
    start = n // 4 + int(np.nanargmin(heights[middle]))
    low = np.nan_to_num(heights, nan=np.inf) <= level + wet
    if not low[start]:
        return None
    first = start
    while first > 0 and low[first - 1]:
        first -= 1
    last = start
    while last < n - 1 and low[last + 1]:
        last += 1
    return (first, last) if last > first else None


def channel(
    corridor,
    points: np.ndarray,
    discharge: float,
    manning: float = 0.035,
    slope: float | None = None,
    shape: str = "parabolic",
    wet: float = 1.0,
    count: int | None = None,
) -> tuple[Channel, np.ndarray, np.ndarray, np.ndarray, list]:
    """The channel along a corridor from geo.dat's points (n, 3), and the
    cross-sections it was found on: (channel, x, y, heights at the nodes, the wet
    span of each section)."""
    from lagoonviz.core.corridor import interpolate_bed, lengths

    if shape not in SHAPES:
        raise ValueError(f"shape must be one of {', '.join(SHAPES)}, got {shape!r}")
    if not float(discharge) > 0:
        raise ValueError(f"the discharge must be above 0, got {discharge}")
    if not float(manning) > 0:
        raise ValueError(f"Manning's n must be above 0, got {manning}")
    centre = corridor.centreline(400)
    length = float(lengths(centre)[-1])
    width0 = float(np.median(np.hypot(*(corridor.bank_line("left")[[0, -1]] -
                                        corridor.bank_line("right")[[0, -1]]).T)))  # fmt: skip
    if count is None:  # a section every half width or so
        count = int(np.clip(length / max(0.5 * width0, 1e-6), 20, 400))
    x, y = _sections(corridor, int(count))
    z = interpolate_bed(np.asarray(points, float), x, y)
    middle = (x[:, ACROSS // 2], y[:, ACROSS // 2])
    along = lengths(np.column_stack(middle))
    # the water level: the middle half's lowest quarter, falling downstream
    quarter = z[:, ACROSS // 4 : ACROSS - ACROSS // 4]
    level = np.nanpercentile(quarter, 25, axis=1)
    level = np.minimum.accumulate(np.where(np.isfinite(level), level, np.inf))
    level[~np.isfinite(level)] = np.nan
    spans = [_wet_span(z[i], level[i], wet) for i in range(len(level))]
    width = np.full(len(level), np.nan)
    for i, span in enumerate(spans):
        if span is not None:
            a, b = span
            width[i] = float(np.hypot(x[i, b] - x[i, a], y[i, b] - y[i, a]))
    if slope is None:
        ok = np.isfinite(level)
        slope = -np.polyfit(along[ok], level[ok], 1)[0] if ok.sum() >= 2 else MIN_SLOPE
        slope = max(float(slope), MIN_SLOPE)
    depth = np.where(np.isfinite(width), manning_depth(discharge, width, slope, manning), np.nan)
    found = Channel(
        along, width, level, depth, float(slope), float(discharge), float(manning), shape
    )
    return found, x, y, z, spans


def _profile(n: np.ndarray, shape: str) -> np.ndarray:
    """The depth across, of the mean depth, at n from -1 (one edge) to 1 (the
    other): its mean over the width is 1."""
    n = np.abs(n)
    if shape == "trapezoid":  # banks over a fifth of the width each side
        return 1.25 * np.clip((1.0 - n) / 0.2, 0.0, 1.0)
    return 1.5 * np.clip(1.0 - n * n, 0.0, None)


def carve(
    points: np.ndarray,
    corridor,
    discharge: float,
    manning: float = 0.035,
    slope: float | None = None,
    shape: str = "parabolic",
    wet: float = 1.0,
) -> tuple[np.ndarray, Channel]:
    """geo.dat's points (n, 3) with the river's channel cut into them (module doc),
    and the channel. Points outside the water, or already deeper, stay."""
    from scipy.interpolate import LinearNDInterpolator
    from scipy.spatial import QhullError

    from lagoonviz.core.corridor import _spacing

    points = np.array(points, float)
    found, x, y, _z, spans = channel(corridor, points, discharge, manning, slope, shape, wet)
    found.spacing = float(_spacing(points) or 0.0)
    rows = [i for i, s in enumerate(spans) if s is not None and np.isfinite(found.depth[i])]
    if len(rows) < 2:
        return points, found
    # each section's own across coordinate n: -1 at one edge of the water, 1 at the other
    nodes, values = [], []
    for i in rows:
        a, b = spans[i]
        j = np.arange(ACROSS)
        n = np.where((j >= a) & (j <= b), 2.0 * (j - a) / max(b - a, 1) - 1.0, np.nan)
        # one node beyond each edge (n beyond ±1): points just outside stay as they are
        keep = (j >= max(a - 1, 0)) & (j <= min(b + 1, ACROSS - 1))
        n = np.where(j < a, -1.0 - 2.0 * (a - j) / max(b - a, 1), n)
        n = np.where(j > b, 1.0 + 2.0 * (j - b) / max(b - a, 1), n)
        nodes.append(np.column_stack([x[i, keep], y[i, keep]]))
        target_level = found.level[i] - found.depth[i] * _profile(n[keep], shape)
        values.append(np.column_stack([n[keep], np.full(keep.sum(), found.level[i]), target_level]))
    nodes = np.vstack(nodes)
    values = np.vstack(values)
    box_low, box_high = nodes.min(axis=0), nodes.max(axis=0)
    near = ((points[:, :2] >= box_low) & (points[:, :2] <= box_high)).all(axis=1)
    try:
        interpolate = LinearNDInterpolator(nodes, values)
    except QhullError:
        return points, found
    at = interpolate(points[near, :2])
    n, level, target = at[:, 0], at[:, 1], at[:, 2]
    old = points[near, 2]
    ok = np.isfinite(n) & (np.abs(n) < 1.0) & (old <= level + wet) & (target < old)
    index = np.flatnonzero(near)[ok]
    points[index, 2] = target[ok]
    found.lowered = int(ok.sum())
    return points, found
