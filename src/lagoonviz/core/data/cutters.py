# SPDX-License-Identifier: GPL-3.0-or-later
# Architect: Hans Bihs
"""Cutting the iso-surfaces of large volumes in processes of their own.

A CFD free surface (the level set phi at 0) is cut from the volume of every frame:
reading the level set, finding the cells it passes through, cutting them, with AMR
the blocks' seams and the water-side colours. Much of that is Python, and Python
runs one thread at a time: the loader's threads read ahead side by side but cut one
after another, and a 10-core computer used about one and a half of its cores.

Cut in processes of their own (as many as the loader has threads, at most the
cores but one), the frames are cut side by side. Each process opens the case once
and keeps it with what it learnt about its cells; the surface comes back to the
loader thread that asked for it. Volumes of fewer than :data:`MIN_POINTS` points per
frame, the low memory and streaming modes (every process holds its own copy of the
cells) and ``LAGOON_CUT_PROCESSES=0`` cut in the loader's threads, as before; so does
anything a process could not cut (it is then cut there, and a pool that broke,
e.g. a process killed for its memory, is not used again).
"""

from __future__ import annotations

import logging
import multiprocessing
import os
import threading
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool

from lagoonviz.core import memory

log = logging.getLogger(__name__)

#: volumes of fewer points per frame are cut in the loader's threads (a process
#: takes a moment to start, and a small surface is cut in less)
MIN_POINTS = 200_000
#: the cases a process keeps open (the most recently used)
KEPT_CASES = 4

_POOL: ProcessPoolExecutor | None = None
_POOL_SIZE = 0
_BROKEN = False
_LOCK = threading.Lock()


def processes() -> int:
    """How many processes cut: ``LAGOON_CUT_PROCESSES`` if set (0: none), else the
    loader's threads, at most the cores but one; none with fewer than two (the
    loader's threads do as well) and in the low memory and streaming modes."""
    raw = os.environ.get("LAGOON_CUT_PROCESSES", "").strip()
    if raw:
        try:
            return max(0, int(raw))
        except ValueError:
            log.warning("LAGOON_CUT_PROCESSES=%r is no number: ignored", raw)
    profile = memory.current()
    if profile.mode in ("low memory", "streaming"):
        return 0
    count = min(profile.loader_workers, (os.cpu_count() or 1) - 1)
    return count if count >= 2 else 0


def _pool(size: int) -> ProcessPoolExecutor:
    global _POOL, _POOL_SIZE
    with _LOCK:
        if _POOL is None or _POOL_SIZE != size:
            if _POOL is not None:
                _POOL.shutdown(wait=False, cancel_futures=True)
            # spawned, not forked: the window's threads (Qt, the GPU) must not be
            # copied into a child half way through
            _POOL = ProcessPoolExecutor(
                size,
                mp_context=multiprocessing.get_context("spawn"),
                initializer=_started,
                initargs=(memory.mode(),),
            )
            _POOL_SIZE = size
        return _POOL


def shutdown() -> None:
    """Stop the processes (the window closes; tests)."""
    global _POOL, _POOL_SIZE, _BROKEN
    with _LOCK:
        pool, _POOL, _POOL_SIZE, _BROKEN = _POOL, None, 0, False
    if pool is not None:
        pool.shutdown(wait=True, cancel_futures=True)


def cut(case: str, dataset: str, method: str, *args, **kwargs):
    """``case.datasets[dataset].method(*args, **kwargs)`` (``args[0]``: the frame) in
    a process of its own; None when it is not cut there (cut it here then)."""
    global _BROKEN
    size = processes()
    if _BROKEN or size < 1:
        return None
    try:
        status, result = _pool(size).submit(_cut, case, dataset, method, args, kwargs).result()
    except BrokenProcessPool as error:
        with _LOCK:
            first = not _BROKEN
            _BROKEN = True
        if first:
            log.warning("iso-surfaces cut in the loader threads from now on: %s", error)
        return None
    except Exception as error:  # e.g. what is asked for cannot be sent: cut it here
        log.debug("not cut in a process: %s", error)
        return None
    if status != "ok":
        log.debug("not cut in a process: %s", result)
        return None
    return result


# ------------------------------------------------------------------ in the process
_CASES: dict[str, object] = {}


def _started(mode: str) -> None:
    """A cutting process starts: the memory use of the window's."""
    try:
        memory.set_mode(mode)
    except ValueError:
        pass


def _cut(case: str, dataset: str, method: str, args: tuple, kwargs: dict):
    """In a process: ("ok", the surface), or ("not here", why). Errors are not
    raised (one that cannot be sent back would break the pool): the loader's thread
    cuts the frame again and raises them."""
    try:
        from lagoonviz.core.data.reef3d import open_case

        index = int(args[0])
        for again in (False, True):
            opened = None if again else _CASES.pop(case, None)
            if opened is None:
                opened = open_case(case)  # new, or grown since (a run still going)
            _CASES[case] = opened  # the most recently used last
            while len(_CASES) > KEPT_CASES:
                del _CASES[next(iter(_CASES))]
            found = opened.datasets.get(dataset)
            if found is not None and 0 <= index < len(found):
                break
        else:
            return "not here", f"{dataset} has no frame {index} in {case}"
        mesh = getattr(found, method)(*args, **kwargs)
        mesh.with_normals()  # here, not in the loader's thread
        return "ok", mesh
    except Exception as error:
        return "not here", f"{type(error).__name__}: {error}"
