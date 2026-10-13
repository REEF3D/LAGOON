# SPDX-License-Identifier: GPL-3.0-or-later
"""Python's cyclic garbage collector, run on the GUI thread only.

The collector runs in whichever thread allocates when a threshold is reached. In
LAGOON that is often a loader thread (frames, stores, maps). If the GUI thread is
just then deleting a widget, Shiboken releases the GIL halfway through the
deletion; a collection in the other thread then walks the half-deleted wrapper
(``SbkObject_tp_traverse``) and the process dies with a segmentation fault. Large
cases (many and large frames loaded in the background) made it likely.

The automatic collector is switched off and a timer on the GUI thread collects the
generations whose thresholds are reached, as the collector would have.
Reference counting frees everything else at once as before.
"""

from __future__ import annotations

import gc
import logging

from PySide6.QtCore import QObject, QTimer

log = logging.getLogger("lagoon")


class MainThreadGarbageCollector(QObject):
    """Collects cyclic garbage on the thread it lives on (the GUI thread)."""

    def __init__(self, parent: QObject | None = None, interval_ms: int = 250):
        super().__init__(parent)
        self.thresholds = gc.get_threshold()
        gc.disable()
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.check)
        self.timer.start(interval_ms)

    def check(self) -> None:
        counts = gc.get_count()
        for generation in (2, 1, 0):
            if counts[generation] > self.thresholds[generation]:
                gc.collect(generation)
                break

    def stop(self) -> None:
        """Back to Python's own collector (tests, shutting down)."""
        self.timer.stop()
        gc.enable()


def install(app: QObject) -> MainThreadGarbageCollector:
    """Collect garbage on the GUI thread only, for as long as ``app`` lives."""
    collector = MainThreadGarbageCollector(app)
    app._gc_guard = collector  # kept on the app so it lives as long
    log.info("garbage collection on the GUI thread")
    return collector
