# SPDX-License-Identifier: GPL-3.0-or-later
"""Vector arrows on surfaces (NumPy only): which points get one, and the arrow lines.

Arrows are thinned to about a wanted number on an even grid in the plane of the
surface (the two directions it spreads most in), so a fine grid does not turn
into a black mat and a vertical slice gets arrows over its whole height. The grid's
spacing comes from the area the surface covers (its triangles), not from its
bounding box, so a river winding through a wide domain gets the arrows wanted on
its water, close enough to show how the flow turns.

"Vectors of" names the variable and the arrows' colour: ``velocity (white)``,
``velocity (black)``, ``velocity (grey)``; the bare name (sessions from before,
SEASTATE's wave directions) takes the look's contour colour.
"""

from __future__ import annotations

import numpy as np

#: the arrows' colours offered with every vector variable
VECTOR_COLOURS = {"white": "#ffffff", "black": "#000000", "grey": "#808080"}


def vector_choices(names) -> list[str]:
    """The "Vectors of" choices of a dataset's vector variables."""
    return ["none", *[f"{name} ({colour})" for name in names for colour in VECTOR_COLOURS]]


def split_vectors(value: str) -> tuple[str, str | None]:
    """A "Vectors of" value as (variable, colour #rrggbb or None: the look's)."""
    value = str(value)
    name, _, rest = value.rpartition(" (")
    if name and rest.endswith(")") and rest[:-1] in VECTOR_COLOURS:
        return name, VECTOR_COLOURS[rest[:-1]]
    return value, None


def pick_points(
    points: np.ndarray, candidates: np.ndarray, count: int, spacing: float | None = None
) -> np.ndarray:
    """Indices (from ``candidates``) of about ``count`` points spread evenly: one per
    cell of a square grid over the surface's two widest directions, ``spacing`` apart
    (None: from the candidates' bounding box)."""
    candidates = np.asarray(candidates, dtype=np.int64)
    if candidates.size == 0 or count <= 0:
        return np.zeros(0, np.int64)
    chosen = np.asarray(points, dtype=np.float64)[candidates]
    extent = np.ptp(chosen, axis=0)
    axes = np.argsort(extent)[::-1][:2]
    if spacing is None or not spacing > 0:
        area = float(max(extent[axes[0]], 1e-12) * max(extent[axes[1]], 1e-12))
        if extent[axes[1]] <= 1e-9 * max(extent[axes[0]], 1e-12):  # a line of points
            area = float(extent[axes[0]] ** 2) / max(count, 1)
        spacing = np.sqrt(area / max(count, 1))
    low = chosen[:, axes].min(axis=0)
    cells = np.floor((chosen[:, axes] - low) / spacing).astype(np.int64)
    keys = cells[:, 0] * (int(cells[:, 1].max()) + 1) + cells[:, 1]
    _, first = np.unique(keys, return_index=True)  # the first point of every cell
    return candidates[np.sort(first)]


def grid_spacing(points: np.ndarray, count: int, triangles: np.ndarray | None = None) -> float:
    """The spacing of about ``count`` arrows over a surface: from the area of its
    ``triangles`` (indices into ``points``) seen across its two widest directions, or
    without them (or none with an area) from the bounding box of ``points``."""
    points = np.asarray(points, dtype=np.float64)
    if triangles is not None and len(triangles):
        corners = points[np.asarray(triangles, dtype=np.int64)]
        # the area seen across the grid :func:`pick_points` lays (the surface's two
        # widest directions): a free surface's plan area, not that of its slopes
        extent = np.ptp(corners.reshape(-1, 3), axis=0)
        a, b = np.argsort(extent)[::-1][:2]
        u = corners[:, 1] - corners[:, 0]
        w = corners[:, 2] - corners[:, 0]
        area = 0.5 * float(np.abs(u[:, a] * w[:, b] - u[:, b] * w[:, a]).sum())
        if area > 0:
            return float(np.sqrt(area / max(count, 1)))
        points = corners.reshape(-1, 3)
    extent = np.sort(np.ptp(points, axis=0))[::-1]
    area = float(max(extent[0], 1e-12) * max(extent[1], 1e-12))
    return float(np.sqrt(area / max(count, 1)))


def spread_points(
    points: np.ndarray, candidates: np.ndarray, triangles: np.ndarray, count: int
) -> tuple[np.ndarray, float]:
    """About ``count`` of the ``candidates`` spread evenly over the area of
    ``triangles``, and their spacing. On narrow water the grid cells the banks cut
    each add an arrow: with many more than asked the spacing is widened, and set
    again a few times until the number is within a quarter of the one asked."""
    spacing = grid_spacing(points, count, triangles)
    picked = pick_points(points, candidates, count, spacing)
    widened = False
    for _ in range(4):
        n = picked.size
        if count <= 0 or not n or (0.8 * count <= n <= 1.25 * count):
            break
        if n < 0.8 * count and not widened:
            break  # fewer points than asked for: they all have one
        widened = True
        spacing *= float(np.sqrt(n / count))
        picked = pick_points(points, candidates, count, spacing)
    return picked, spacing


def arrow_segments(base: np.ndarray, vectors: np.ndarray, length: float) -> np.ndarray:
    """Line segments ``(3k, 2, 3)`` of arrows: a shaft and two head strokes each.

    ``length`` is the drawn length per unit of the vector.
    """
    base = np.asarray(base, dtype=np.float64)
    tip = base + np.asarray(vectors, dtype=np.float64) * length
    shaft = tip - base
    size = np.linalg.norm(shaft, axis=1)
    keep = size > 1e-12
    base, tip, shaft, size = base[keep], tip[keep], shaft[keep], size[keep]
    if not len(base):
        return np.zeros((0, 2, 3), np.float32)
    direction = shaft / size[:, None]
    # the head opens in the plane of the arrow and the vertical (or x, for vertical arrows)
    side = np.cross(direction, np.array([0.0, 0.0, 1.0]))
    flat = np.linalg.norm(side, axis=1) < 1e-6
    side[flat] = np.cross(direction[flat], np.array([1.0, 0.0, 0.0]))
    side /= np.linalg.norm(side, axis=1)[:, None]
    back = tip - 0.3 * shaft
    spread = 0.15 * size[:, None] * side
    segments = np.concatenate(
        [
            np.stack([base, tip], axis=1),
            np.stack([tip, back + spread], axis=1),
            np.stack([tip, back - spread], axis=1),
        ]
    )
    return segments.astype(np.float32)
