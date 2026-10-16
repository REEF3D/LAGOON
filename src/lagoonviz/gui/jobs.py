# SPDX-License-Identifier: GPL-3.0-or-later
# Architect: Hans Bihs
"""A long job of the window (reading a whole run for a statistics map): a worker
thread with a progress dialog that can cancel it, the result handed back to the
window's thread."""

from __future__ import annotations

import logging
import threading

from PySide6.QtCore import QObject, Qt, Signal
from PySide6.QtWidgets import QProgressDialog, QWidget

log = logging.getLogger(__name__)


class Cancelled(Exception):
    """The job was cancelled from its dialog."""


class _Runner(QObject):
    progressed = Signal(int, str)
    done = Signal(object)


#: jobs running, kept until they end
_RUNNING: set = set()


def run_job(parent: QWidget, title: str, work, finish, failed=None) -> None:
    """``work(progress)`` in a thread (``progress(fraction, text)``; it raises
    :class:`Cancelled` once cancelled), then ``finish(result)`` in the window's
    thread; ``failed(message)`` there when it raised (default: logged)."""
    dialog = QProgressDialog(title + "…", "Cancel", 0, 1000, parent)
    dialog.setWindowTitle(title)
    dialog.setWindowModality(Qt.WindowModal)
    dialog.setMinimumDuration(300)
    dialog.setAutoClose(False)
    dialog.setAutoReset(False)
    dialog.setValue(0)
    stop = threading.Event()
    runner = _Runner()
    runner.progressed.connect(lambda v, text: (dialog.setValue(v), dialog.setLabelText(text)))
    dialog.canceled.connect(stop.set)
    box: dict = {}

    def ended(error) -> None:
        dialog.close()
        _RUNNING.discard(runner)
        if isinstance(error, Cancelled):
            return
        if error is not None:
            message = str(error) or type(error).__name__
            log.warning("%s: %s", title, message)
            if failed is not None:
                failed(message)
            return
        try:
            finish(box["result"])
        except (ValueError, OSError) as problem:
            if failed is not None:
                failed(str(problem))

    runner.done.connect(ended)

    def progress(fraction: float, text: str) -> None:
        if stop.is_set():
            raise Cancelled()
        runner.progressed.emit(int(round(1000 * min(max(fraction, 0.0), 1.0))), text)

    def run() -> None:
        try:
            box["result"] = work(progress)
        except BaseException as error:  # noqa: BLE001 (reported in the window)
            runner.done.emit(error)
            return
        runner.done.emit(None)

    _RUNNING.add(runner)
    threading.Thread(target=run, name="lagoon-job", daemon=True).start()
