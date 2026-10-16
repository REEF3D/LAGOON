# SPDX-License-Identifier: GPL-3.0-or-later
# Architect: Hans Bihs
"""A line traced point by point in the 2D top view (a river drawn on the map).

- A click adds a point at the downstream end; a click on a segment puts a point
  into it there; a click on the last point again (a double click) or Enter ends the
  line.
- A point is dragged to move it; a right click (or Alt + click) on it removes it;
  Backspace takes the last point back.
- A line ended can be taken up again: its points are given back to a new trace.

The 3D view feeds it; it knows nothing of Qt, so it is tested without a window."""

from __future__ import annotations

import math

import numpy as np


class LineTrace:
    """The points clicked so far (model x, y), and what to draw while tracing."""

    def __init__(self, points=None, fewest: int = 2) -> None:
        self.points: list[tuple[float, float]] = [
            (float(x), float(y)) for x, y in (points if points is not None else [])
        ]
        self.fewest = int(fewest)

    # ------------------------------------------------------------- what is where
    def point_at(self, point, reach: float) -> int | None:
        """The point within ``reach`` of ``point`` (the nearest), or None."""
        if not self.points:
            return None
        p = np.asarray(self.points, float)
        distance = np.hypot(p[:, 0] - float(point[0]), p[:, 1] - float(point[1]))
        k = int(np.argmin(distance))
        return k if distance[k] <= reach else None

    def segment_at(self, point, reach: float) -> int | None:
        """The segment (k: from point k to k + 1) within ``reach`` of ``point`` away
        from its ends, or None."""
        x, y = float(point[0]), float(point[1])
        best = None
        for k in range(len(self.points) - 1):
            (ax, ay), (bx, by) = self.points[k], self.points[k + 1]
            dx, dy = bx - ax, by - ay
            length2 = dx * dx + dy * dy
            if length2 <= 0:
                continue
            t = ((x - ax) * dx + (y - ay) * dy) / length2
            if not 0.0 < t < 1.0:
                continue
            d = math.hypot(x - (ax + t * dx), y - (ay + t * dy))
            if d <= reach and (best is None or d < best[0]):
                best = (d, k)
        return None if best is None else best[1]

    # ------------------------------------------------------------- changes
    def click(self, point, reach: float) -> bool:
        """A click at ``point``; ``reach``: how far (model m) a click may lie from a
        point or a segment to be on it. On the last point (with enough points): the
        line is done (True). On another point: nothing. On a segment: a point put
        into it. Elsewhere: a point added at the end."""
        x, y = float(point[0]), float(point[1])
        on = self.point_at((x, y), reach)
        if on is not None:
            return on == len(self.points) - 1 and self.done
        segment = self.segment_at((x, y), reach)
        if segment is not None:
            self.points.insert(segment + 1, (x, y))
        else:
            self.points.append((x, y))
        return False

    def move(self, index: int, point) -> None:
        self.points[int(index)] = (float(point[0]), float(point[1]))

    def remove(self, index: int) -> None:
        del self.points[int(index)]

    def back(self) -> bool:
        """The last point taken back; False if there was none."""
        if not self.points:
            return False
        self.points.pop()
        return True

    # ------------------------------------------------------------- what it is
    @property
    def done(self) -> bool:
        """Whether there are enough points to end the line."""
        return len(self.points) >= self.fewest

    def length(self, pointer=None) -> float:
        """The line's length [m], to the pointer if given."""
        line = self.points + ([tuple(pointer)] if pointer is not None and self.points else [])
        if len(line) < 2:
            return 0.0
        return float(np.hypot(*np.diff(np.asarray(line, float), axis=0).T).sum())

    def sketch(self, pointer=None, z: float = 0.0):
        """(segments (n, 2, 3) or None, points (m, 3) or None) to draw: the line so
        far, and on to the pointer (None: not)."""
        line = list(self.points)
        if pointer is not None and line:
            line.append((float(pointer[0]), float(pointer[1])))
        dots = np.array([[x, y, z] for x, y in self.points], float) if self.points else None
        if len(line) < 2:
            return None, dots
        p = np.column_stack([np.asarray(line, float), np.full(len(line), z)])
        return np.stack([p[:-1], p[1:]], axis=1), dots
