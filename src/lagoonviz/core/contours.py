# SPDX-License-Identifier: GPL-3.0-or-later
"""Contour lines on triangle surfaces (marching triangles, NumPy only).

A contour level crosses a triangle when one corner lies on the other side of the
level than the two others; the line through the triangle joins the two crossing
points on the edges from that corner. Triangles with a missing value (NaN) and
degenerate triangles (hidden dry land) give no line.
"""

from __future__ import annotations

import numpy as np

from lagoonviz.core.plotting import nice_ticks


def contour_levels(low: float, high: float, count: int) -> list[float]:
    """About ``count`` round levels strictly inside (low, high)."""
    if not (np.isfinite(low) and np.isfinite(high)) or high <= low:
        return []
    levels = [v for v in nice_ticks(low, high, max(int(count), 2)) if low < v < high]
    return [float(v) for v in levels]


def contour_segments(
    points: np.ndarray, triangles: np.ndarray, values: np.ndarray, levels: list[float]
) -> np.ndarray:
    """Line segments ``(k, 2, 3)`` of the contours of ``values`` at ``levels``."""
    triangles = np.asarray(triangles)
    if triangles.size == 0 or not levels:
        return np.zeros((0, 2, 3), np.float32)
    corner_values = np.asarray(values, dtype=np.float64)[triangles]  # (m, 3)
    corner_points = np.asarray(points, dtype=np.float64)[triangles]  # (m, 3, 3)
    usable = np.isfinite(corner_values).all(axis=1)
    usable &= ~(
        (triangles[:, 0] == triangles[:, 1]) & (triangles[:, 1] == triangles[:, 2])
    )  # hidden (degenerate) triangles
    corner_values, corner_points = corner_values[usable], corner_points[usable]
    pieces = []
    for level in levels:
        above = corner_values >= level
        count = above.sum(axis=1)
        crossing = (count == 1) | (count == 2)
        if not crossing.any():
            continue
        v, p, a = corner_values[crossing], corner_points[crossing], above[crossing]
        # the corner on its own side of the level
        lone = np.where(a.sum(axis=1) == 1, np.argmax(a, axis=1), np.argmin(a, axis=1))
        rows = np.arange(len(v))
        i0, i1, i2 = lone, (lone + 1) % 3, (lone + 2) % 3
        v0, v1, v2 = v[rows, i0], v[rows, i1], v[rows, i2]
        p0, p1, p2 = p[rows, i0], p[rows, i1], p[rows, i2]
        with np.errstate(divide="ignore", invalid="ignore"):
            t1 = np.clip((level - v0) / (v1 - v0), 0.0, 1.0)
            t2 = np.clip((level - v0) / (v2 - v0), 0.0, 1.0)
        start = p0 + np.nan_to_num(t1)[:, None] * (p1 - p0)
        end = p0 + np.nan_to_num(t2)[:, None] * (p2 - p0)
        pieces.append(np.stack([start, end], axis=1))
    if not pieces:
        return np.zeros((0, 2, 3), np.float32)
    return np.concatenate(pieces).astype(np.float32)
