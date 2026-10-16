# SPDX-License-Identifier: GPL-3.0-or-later
# Architect: Hans Bihs
"""A line traced point by point in the 2D top view (a river drawn on the map): each
click adds a point, a click on the last point again (a double click) or Enter ends
it, Backspace takes the last point back. The 3D view feeds it; it knows nothing of
Qt, so it is tested without a window."""

from __future__ import annotations

import math

import numpy as np


class LineTrace:
    """The points clicked so far (model x, y), and what to draw while tracing."""

    def __init__(self, fewest: int = 2) -> None:
        self.points: list[tuple[float, float]] = []
        self.fewest = int(fewest)

    def click(self, point, reach: float) -> bool:
        """A click at ``point``; ``reach``: how far (model m) a click may lie from the
        last point to be on it. True when the line is done (a click on its last
        point, with enough points)."""
        x, y = float(point[0]), float(point[1])
        if self.points:
            lx, ly = self.points[-1]
            if math.hypot(x - lx, y - ly) <= reach:
                return len(self.points) >= self.fewest
        self.points.append((x, y))
        return False

    def back(self) -> bool:
        """The last point taken back; False if there was none."""
        if not self.points:
            return False
        self.points.pop()
        return True

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
        far, and on to the pointer."""
        line = list(self.points)
        if pointer is not None and line:
            line.append((float(pointer[0]), float(pointer[1])))
        dots = np.array([[x, y, z] for x, y in self.points], float) if self.points else None
        if len(line) < 2:
            return None, dots
        p = np.column_stack([np.asarray(line, float), np.full(len(line), z)])
        return np.stack([p[:-1], p[1:]], axis=1), dots
