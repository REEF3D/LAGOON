# SPDX-License-Identifier: GPL-3.0-or-later
# Architect: Hans Bihs
"""Speed or memory: how much LAGOON keeps in memory to be fast.

Four modes, for the whole program (View → Memory use in the window; the
``LAGOON_MEMORY`` environment variable for ``lagoon-render`` and scripts):

- ``"performance"``: in memory. Frames kept up to a quarter of the computer's
  memory (at least 2 GB, at most 16 GB); playing reads the whole play range ahead
  when it fits, so it runs from memory from the second time round; more threads
  reading; a volume's cells kept for slices and iso-surfaces;
- ``"balanced"`` (the default): 1 GB of frames, as LAGOON always did;
- ``"low memory"``: 256 MB of frames, few read ahead, two threads, the volume's
  cells read again for every slice instead of kept (large volumes fit, playing is
  slower);
- ``"streaming"``: for very large cases (100 million cells and more). Frames are
  read as they are shown and only a few kept; while playing or dragging the
  timeline, surfaces of more than :attr:`MemoryProfile.detail_points` points are
  shown coarser (every 2nd, 3rd, ... grid line), and at full detail as soon as the
  time stands still. Films and stills are always made at full detail.

Whatever the mode, frames are never read further ahead than the frame budget
holds (they would push out each other before being shown).
"""

from __future__ import annotations

import os
from collections.abc import Callable
from dataclasses import dataclass

MODES = ("performance", "balanced", "low memory", "streaming")
#: how the modes are called in the window's menu
LABELS = {
    "performance": "Performance (in memory)",
    "balanced": "Balanced",
    "low memory": "Low memory",
    "streaming": "Streaming (very large cases)",
}
#: points per frame from which a case counts as very large (streaming suggested)
LARGE_POINTS = 5_000_000
DEFAULT_MODE = "balanced"

GB = 1 << 30
#: streaming: surfaces are shown with at most about this many points while playing
STREAM_POINTS = 2_500_000


@dataclass(frozen=True)
class MemoryProfile:
    mode: str
    #: bytes of frames the cache keeps (the least recently used go first)
    cache_bytes: int
    #: threads reading frames in the background
    loader_workers: int
    #: frames read ahead while playing (at least; more at a high frame rate)
    play_ahead: int
    #: frames read ahead while filming, and the threads reading them
    film_read_ahead: int
    film_readers: int
    #: keep a volume's cells after a slice or iso-surface (else read them again)
    keep_cells: bool
    #: while the time moves, surfaces of more points than this are shown coarser
    #: (None: always at full detail)
    detail_points: int | None = None
    #: read the whole play range ahead while playing, when it fits the budget
    play_all: bool = False

    def describe(self) -> str:
        ahead = "the play range" if self.play_all else str(self.play_ahead)
        return (
            f"{self.mode}: up to {self.cache_bytes / GB:.2g} GB of frames, "
            f"{ahead} read ahead while playing, {self.loader_workers} threads"
            + ("" if self.keep_cells else ", volume cells not kept")
            + (
                f", surfaces over {self.detail_points / 1e6:.3g} million points "
                "coarser while playing"
                if self.detail_points
                else ""
            )
        )


def physical_memory() -> int | None:
    """The computer's memory in bytes (None if it cannot be found)."""
    try:
        return int(os.sysconf("SC_PAGE_SIZE")) * int(os.sysconf("SC_PHYS_PAGES"))
    except (ValueError, OSError, AttributeError):
        return None


def profile(mode: str) -> MemoryProfile:
    """The settings of a mode (see the module docstring)."""
    if mode not in MODES:
        raise ValueError(f"memory use: one of {', '.join(MODES)}, got {mode!r}")
    if mode == "performance":
        total = physical_memory() or 8 * GB
        cache = min(max(total // 4, 2 * GB), 16 * GB)
        return MemoryProfile(mode, cache, 6, 16, 10, 4, True, play_all=True)
    if mode == "low memory":
        return MemoryProfile(mode, 256 << 20, 2, 2, 2, 1, False)
    if mode == "streaming":
        return MemoryProfile(mode, 768 << 20, 4, 6, 3, 2, False, detail_points=STREAM_POINTS)
    return MemoryProfile(mode, GB, 4, 8, 6, 3, True)


_current = DEFAULT_MODE
_listeners: list[Callable[[MemoryProfile], None]] = []


def _from_environment() -> None:
    global _current
    wanted = os.environ.get("LAGOON_MEMORY", "").strip().lower().replace("_", " ")
    if wanted in ("low", "low-memory", "low memory", "memory"):
        _current = "low memory"
    elif wanted in ("stream", "streamed"):
        _current = "streaming"
    elif wanted in MODES:
        _current = wanted


_from_environment()


def current() -> MemoryProfile:
    """The profile of the mode in use."""
    return profile(_current)


def mode() -> str:
    return _current


def set_mode(name: str) -> MemoryProfile:
    """Use a mode from now on; those listening (the frame cache, the loader) adjust."""
    global _current
    found = profile(name)
    _current = name
    for listener in list(_listeners):
        listener(found)
    return found


def subscribe(listener: Callable[[MemoryProfile], None]) -> None:
    """``listener(profile)`` is called whenever the mode changes."""
    _listeners.append(listener)


def unsubscribe(listener: Callable[[MemoryProfile], None]) -> None:
    if listener in _listeners:
        _listeners.remove(listener)
