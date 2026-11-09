# SPDX-License-Identifier: GPL-3.0-or-later
# Architect: Hans Bihs
"""Pacing the playback to what the frames can be read at.

Frames cut from a volume (CFD's free surface, an iso-surface of the level set; slices)
take a good part of a second each. Read ahead on several threads, they come in
batches: playing at the frame rate asked for raced through each batch and then stood
still until the next one, a few frames and a long pause. The :class:`Pacer` watches
how fast frames become ready and, while frames are being read, plays no faster than
that: steadily, at the rate the computer reads them. Once nothing is being read (every
frame of the range is there), at the frame rate.
"""

from __future__ import annotations

import time
from collections import deque


class Pacer:
    """The interval to wait before the next frame: no shorter than the time a frame
    took to become ready over the last ``arrivals`` frames that did, while frames are
    being read."""

    def __init__(self, arrivals: int = 12, enough: int = 48, clock=time.monotonic):
        #: frames ahead the timeline counts (more than are read ahead at a time: a
        #: count stopped short grows as playing moves on, and paced itself)
        self.enough = int(enough)
        self.clock = clock
        self._arrivals: deque[tuple[float, int]] = deque(maxlen=max(int(arrivals), 2))
        self._furthest: int | None = None

    def reset(self) -> None:
        """Playing starts (again): what was measured before no longer counts."""
        self._arrivals.clear()
        self._furthest = None

    def note(self, shown: int, ready_ahead: int) -> None:
        """At a tick: the frame shown and how many frames after it are ready."""
        furthest = int(shown) + int(ready_ahead)
        if self._furthest is None or furthest < self._furthest:  # started, or wrapped
            self._furthest = furthest
            self._arrivals.clear()
            self._arrivals.append((self.clock(), furthest))
            return
        if furthest > self._furthest:
            self._furthest = furthest
            self._arrivals.append((self.clock(), furthest))

    def read_interval(self) -> float | None:
        """The seconds a frame took to become ready, over the last arrivals; None when
        fewer than two were seen. It holds while nothing comes (a batch read on several
        threads arrives at once, then nothing for a while: measured over a few seconds,
        that was "nothing comes", and playing raced through the next batch)."""
        if len(self._arrivals) < 2:
            return None
        (t0, f0), (t1, f1) = self._arrivals[0], self._arrivals[-1]
        if f1 <= f0 or t1 - t0 <= 0:
            return None
        return (t1 - t0) / (f1 - f0)

    def interval(
        self, frame_interval: float, ready_ahead: int, reading: bool = True, left: int | None = None
    ) -> float:
        """The interval to the next frame: ``frame_interval`` (1 / fps), or, while
        frames are being read (``reading``), the read interval if that is longer (at
        most 5 s). Nothing being read (every frame there), or the last frames of the
        range (``left`` <= ``ready_ahead``): the frame rate."""
        if not reading or (left is not None and ready_ahead >= left):
            return frame_interval
        measured = self.read_interval()
        if measured is None:
            return frame_interval
        return max(frame_interval, min(measured, 5.0))
