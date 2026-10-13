# SPDX-License-Identifier: GPL-3.0-or-later
# Architect: Hans Bihs
"""Speed or memory: how much LAGOON keeps in memory to be fast.

Three modes, for the whole program (View → Memory use in the window; the
``LAGOON_MEMORY`` environment variable for ``lagoon-render`` and scripts):

- ``"performance"``: frames kept up to a quarter of the computer's memory (at least
  2 GB, at most 16 GB), more frames read ahead while playing and filming, more
  threads reading, a volume's cells kept for slices and iso-surfaces;
- ``"balanced"`` (the default): 1 GB of frames, as LAGOON always did;
- ``"low memory"``: 256 MB of frames, few read ahead, two threads, the volume's
  cells read again for every slice instead of kept (large volumes fit, playing is
  slower).
"""

from __future__ import annotations

import os
from collections.abc import Callable
from dataclasses import dataclass

MODES = ("performance", "balanced", "low memory")
DEFAULT_MODE = "balanced"

GB = 1 << 30


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

    def describe(self) -> str:
        return (
            f"{self.mode}: up to {self.cache_bytes / GB:.2g} GB of frames, "
            f"{self.play_ahead} read ahead while playing, {self.loader_workers} threads"
            + ("" if self.keep_cells else ", volume cells not kept")
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
        return MemoryProfile(mode, cache, 6, 16, 10, 4, True)
    if mode == "low memory":
        return MemoryProfile(mode, 256 << 20, 2, 2, 2, 1, False)
    return MemoryProfile(mode, GB, 4, 8, 6, 3, True)


_current = DEFAULT_MODE
_listeners: list[Callable[[MemoryProfile], None]] = []


def _from_environment() -> None:
    global _current
    wanted = os.environ.get("LAGOON_MEMORY", "").strip().lower().replace("_", " ")
    if wanted in ("low", "low-memory", "low memory", "memory"):
        _current = "low memory"
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
