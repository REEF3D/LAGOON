# SPDX-License-Identifier: GPL-3.0-or-later
# Architect: Hans Bihs
"""Pacing the playback to what the frames can be read at.

Frames cut from a volume (CFD's free surface, an iso-surface of the level set; slices)
take a good part of a second each. Read ahead on several threads, they came in
batches: playing raced through the few that were there at the frame rate asked for
and then stood still until the next batch, a few frames and a long pause. The
:class:`Pacer` watches how fast the frames ahead become ready and, while few are
ready, plays no faster than that: steadily, at the rate the computer reads them.
"""

from __future__ import annotations

import time
from collections import deque


class Pacer:
    """The interval to wait before the next frame: the frame rate's while enough
    frames are ready ahead, else no shorter than the time a frame took to become
    ready of late (the furthest frame read ahead, over ``window`` seconds)."""

    def __init__(self, window: float = 3.0, enough: int = 6, clock=time.monotonic):
        self.window = float(window)
        self.enough = int(enough)
        self.clock = clock
        self._ready: deque[tuple[float, int]] = deque()  # (when, frames ready ahead + shown)

    def reset(self) -> None:
        """Playing starts (again): what was measured before no longer counts."""
        self._ready.clear()

    def note(self, shown: int, ready_ahead: int) -> None:
        """At a tick: the frame shown and how many frames after it are ready."""
        now = self.clock()
        self._ready.append((now, int(shown) + int(ready_ahead)))
        while self._ready and self._ready[0][0] < now - self.window:
            self._ready.popleft()

    def read_interval(self) -> float | None:
        """The seconds a frame took to become ready of late; None when unknown."""
        if len(self._ready) < 2:
            return None
        (t0, f0), (t1, f1) = self._ready[0], self._ready[-1]
        if t1 - t0 < 0.5:
            return None  # too short to tell
        if f1 <= f0:
            return float("inf")  # nothing came meanwhile
        return (t1 - t0) / (f1 - f0)

    def interval(self, frame_interval: float, ready_ahead: int) -> float:
        """The interval to the next frame: ``frame_interval`` (1 / fps) while at least
        ``enough`` frames are ready ahead, else the read interval if that is longer
        (at most 2 s: the frame shown waits for its data anyway)."""
        if ready_ahead >= self.enough:
            return frame_interval
        reading = self.read_interval()
        if reading is None:
            return frame_interval
        return max(frame_interval, min(reading, 2.0))
