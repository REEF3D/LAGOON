# SPDX-License-Identifier: GPL-3.0-or-later
# Architect: Hans Bihs
"""File > Convert to LAGOON format: the case's VTK output to a LAGOON store, in a
background thread with a progress dialog; the case is reloaded from the store."""

from __future__ import annotations

import logging
import threading
import traceback
from pathlib import Path

from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import QMessageBox, QProgressDialog

from lagoonviz.core.convert import ConvertError, Report, convert

log = logging.getLogger(__name__)


class Cancelled(Exception):
    """The user stopped the conversion (what is written so far stays usable)."""


class ConvertJob:
    """One conversion in a thread; the GUI polls ``state``."""

    def __init__(self, case_path: Path):
        self.case_path = case_path
        self.state = {"role": "", "done": 0, "total": 0}
        self.report: Report | None = None
        self.error: str | None = None
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self._run, name="lagoon-convert", daemon=True)

    def start(self) -> ConvertJob:
        self.thread.start()
        return self

    @property
    def finished(self) -> bool:
        return not self.thread.is_alive()

    def _progress(self, role: str, done: int, total: int) -> None:
        self.state = {"role": role, "done": done, "total": total}
        if self.stop.is_set():
            raise Cancelled

    def _run(self) -> None:
        try:
            self.report = convert(self.case_path, progress=self._progress)
        except Cancelled:
            self.error = "stopped"
        except (ConvertError, OSError, ValueError) as error:
            self.error = str(error)
        except Exception as error:  # noqa: BLE001 (reported in the window)
            log.error("conversion failed:\n%s", traceback.format_exc())
            self.error = f"{type(error).__name__}: {error}"


ROLE_NAMES = {
    "volume": "volume",
    "free_surface": "free surface",
    "bed": "bed",
    "amr": "free surface AMR",
}


def case_folder(window) -> Path | None:
    """The folder of the selected case (or of the first case)."""
    session = window.session
    selected = window.scene_tree.selected_id()
    case_item = session.data.case_item(selected) if selected else None
    if case_item is None:
        case_item = next((i for i in session.scene if i.kind == "case"), None)
    if case_item is None:
        return None
    case = session.data.case(case_item.id)
    return case.path if case is not None else None


def convert_case(window, ask: bool = True) -> ConvertJob | None:
    """Convert the selected case; returns the running job (tests wait for it)."""
    folder = case_folder(window)
    if folder is None:
        QMessageBox.information(window, "Convert to LAGOON format", "Open a REEF3D case first.")
        return None
    if ask:
        answer = QMessageBox.question(
            window,
            "Convert to LAGOON format",
            f"Write the output of {folder.name} as a LAGOON store in the case folder?\n\n"
            "The store is several times smaller than the VTU/VTP files and faster to read; "
            "LAGOON opens it instead of them. The VTU/VTP files are not changed. "
            "Converting again adds the outputs written since.",
        )
        if answer != QMessageBox.Yes:
            return None
    job = ConvertJob(folder).start()
    progress = QProgressDialog("Converting…", "Stop", 0, 0, window)
    progress.setWindowTitle("Convert to LAGOON format")
    progress.setWindowModality(Qt.WindowModal)
    progress.setMinimumDuration(300)
    progress.canceled.connect(job.stop.set)
    timer = QTimer(window)
    timer.setInterval(150)

    def poll() -> None:
        state = job.state
        if state["total"]:
            progress.setMaximum(state["total"])
            progress.setValue(state["done"])
            name = ROLE_NAMES.get(state["role"], state["role"])
            progress.setLabelText(
                f"Converting the {name}: output {state['done']} of {state['total']}"
            )
        if not job.finished:
            return
        timer.stop()
        progress.close()
        _finished(window, job)

    timer.timeout.connect(poll)
    timer.start()
    window._convert_job = job  # kept alive while it runs
    return job


def _finished(window, job: ConvertJob) -> None:
    if job.report is not None:
        window.reload_cases()  # the case now reads the store
        window.statusBar().showMessage(job.report.summary(), 20000)
        log.info("converted: %s", job.report.summary())
    elif job.error == "stopped":
        window.statusBar().showMessage("Conversion stopped; convert again to continue", 10000)
        window.reload_cases()
    else:
        QMessageBox.warning(
            window, "Convert to LAGOON format", f"The conversion failed:\n{job.error}"
        )
