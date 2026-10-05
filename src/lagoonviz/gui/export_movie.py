# SPDX-License-Identifier: GPL-3.0-or-later
"""File → Export animation: film the current view, off screen, into an MP4.

Two renderers: LAGOON's own (fast, the frames as the 3D view draws them) and
Blender's Cycles (path traced, minutes per frame; ``lagoonviz.render.pathtrace``),
which runs in the background while the dialog shows how far it is.
"""

from __future__ import annotations

import logging
import os
import threading
import time as time_module
import traceback
from pathlib import Path

import numpy as np
from PySide6.QtCore import QRectF, Qt
from PySide6.QtGui import QColor, QFont, QImage, QLinearGradient, QPainter
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QProgressDialog,
    QPushButton,
    QSpinBox,
    QWidget,
)

from lagoonviz.core import Session
from lagoonviz.core.colormaps import lut
from lagoonviz.core.plotting import tick_label

log = logging.getLogger("lagoon.gui")

SIZES = {
    "HD 1280 × 720": (1280, 720),
    "Full HD 1920 × 1080": (1920, 1080),
    "QHD 2560 × 1440": (2560, 1440),
    "4K 3840 × 2160": (3840, 2160),
}


def view_sizes(viewport_size) -> list[tuple[str, tuple[int, int]]]:
    """Film sizes with the 3D view's shape, so the film frames exactly what it shows:
    Full HD, HD and 4K widths (even sizes, as H.264 needs)."""
    width, height = (max(int(v), 1) for v in viewport_size)
    sizes = []
    for name, target in (("Full HD", 1920), ("HD", 1280), ("4K", 3840)):
        w = target
        h = max(2, round(target * height / width / 2) * 2)
        sizes.append((f"As the 3D view, {name} width: {w} × {h}", (w, h)))
    return sizes


class ExportAnimationDialog(QDialog):
    """Where to, how big, which outputs, how fast."""

    def __init__(self, parent: QWidget, session: Session, folder: str, viewport_size) -> None:
        super().__init__(parent)
        self.setWindowTitle("Export animation")
        data = session.data
        n = data.n_frames()
        first = data.first_output()
        form = QFormLayout(self)

        self.path = QLineEdit(str(Path(folder) / "lagoon_animation.mp4"))
        browse = QPushButton("Choose…")
        browse.clicked.connect(self._browse)
        row = QHBoxLayout()
        row.addWidget(self.path, 1)
        row.addWidget(browse)
        holder = QWidget()
        holder.setLayout(row)
        form.addRow("File", holder)

        self.size = QComboBox()
        # the film shows what the 3D view shows: same shape (a different shape shows
        # more at the sides or above and below), Full HD width by default
        for label, value in view_sizes(viewport_size):
            self.size.addItem(label, value)
        for label, value in SIZES.items():
            self.size.addItem(f"{label} (16:9: other framing unless the view is 16:9)", value)
        self.size.setCurrentIndex(0)
        form.addRow("Size", self.size)

        self.start = QSpinBox()
        self.end = QSpinBox()
        for spin in (self.start, self.end):
            spin.setRange(first, max(n - 1, first))
        self.start.setValue(first)
        self.end.setValue(max(n - 1, first))
        self.every = QSpinBox()
        self.every.setRange(1, 1000)
        outputs = QHBoxLayout()
        for text, widget in (("from", self.start), ("to", self.end), ("every", self.every)):
            outputs.addWidget(QLabel(text))
            outputs.addWidget(widget)
        holder = QWidget()
        holder.setLayout(outputs)
        form.addRow("Outputs", holder)

        self.fps = QDoubleSpinBox()
        self.fps.setRange(1, 120)
        self.fps.setDecimals(0)
        self.fps.setValue(25)
        form.addRow("Frames per second", self.fps)

        self.overlay = QCheckBox("Time and colour bars")
        self.overlay.setChecked(True)
        form.addRow("Show", self.overlay)

        from lagoonviz.render.pathtrace import find_blender

        self.blender = find_blender()
        self.engine = QComboBox()
        self.engine.addItem("LAGOON (fast, as the 3D view draws it)", "pygfx")
        self.engine.addItem("Blender Cycles (path traced, slow)", "cycles")
        self.engine.setToolTip(
            "Cycles traces the light: reflections, refraction through the water, soft "
            "shadows. In the look of the 3D view if it is a cinematic one, else open ocean."
        )
        form.addRow("Renderer", self.engine)
        self.samples = QSpinBox()
        self.samples.setRange(4, 4096)
        self.samples.setValue(128)
        self.samples.setToolTip("Samples per pixel: more is less noisy and slower")
        self.blender_label = QLabel(
            f"Blender: {self.blender}"
            if self.blender
            else "Blender not found: install it from blender.org (4.2 or newer), "
            "or set LAGOON_BLENDER"
        )
        self.blender_label.setWordWrap(True)
        cycles = QHBoxLayout()
        cycles.addWidget(QLabel("samples"))
        cycles.addWidget(self.samples)
        cycles.addWidget(self.blender_label, 1)
        self.cycles_row = QWidget()
        self.cycles_row.setLayout(cycles)
        form.addRow("", self.cycles_row)
        self.engine.currentIndexChanged.connect(lambda _: self._on_engine())
        self._on_engine()

        self.summary = QLabel()
        form.addRow("", self.summary)
        for spin in (self.start, self.end, self.every):
            spin.valueChanged.connect(self._update_summary)
        self.fps.valueChanged.connect(self._update_summary)
        self._update_summary()

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.button(QDialogButtonBox.Ok).setText("Export")
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        form.addRow(buttons)
        self._on_engine()

    def _on_engine(self) -> None:
        cycles = self.engine.currentData() == "cycles"
        self.cycles_row.setVisible(cycles)
        self.overlay.setEnabled(not cycles)  # Cycles films show the scene alone
        buttons = self.findChild(QDialogButtonBox)
        if buttons is not None:
            buttons.button(QDialogButtonBox.Ok).setEnabled(not cycles or bool(self.blender))

    def _browse(self) -> None:
        path, _ = QFileDialog.getSaveFileName(
            self, "Export animation", self.path.text(), "MP4 video (*.mp4)"
        )
        if path:
            self.path.setText(path if Path(path).suffix else path + ".mp4")

    def _update_summary(self) -> None:
        frames = len(range(self.start.value(), self.end.value() + 1, self.every.value()))
        seconds = frames / max(self.fps.value(), 1)
        self.summary.setText(f"{frames} frames, {seconds:.1f} s of video")

    def settings(self) -> dict:
        return {
            "path": self.path.text(),
            "size": self.size.currentData(),
            "start": self.start.value(),
            "end": self.end.value(),
            "every": self.every.value(),
            "fps": self.fps.value(),
            "overlay": self.overlay.isChecked() and self.overlay.isEnabled(),
            "engine": self.engine.currentData(),
            "samples": self.samples.value(),
        }


def make_overlay(session: Session):
    """Draw the time and the colour bars of the coloured items onto a frame."""

    def overlay(image: np.ndarray, time: float) -> np.ndarray:
        height, width = image.shape[:2]
        pixels = np.ascontiguousarray(image[..., :4], dtype=np.uint8)
        qimage = QImage(pixels.data, width, height, 4 * width, QImage.Format_RGBA8888)
        painter = QPainter(qimage)
        painter.setRenderHint(QPainter.Antialiasing)
        unit = max(height / 1080, 0.4)
        font = QFont(QApplication.font())
        font.setPixelSize(round(26 * unit))
        painter.setFont(font)
        painter.setPen(QColor("#1e2833"))
        data = session.data
        label = f"t = {time:.2f} s" if data.time_in_seconds() else f"output {round(time)}"
        margin = 28 * unit
        painter.drawText(QRectF(margin, margin, width / 2, 40 * unit), Qt.AlignLeft, label)

        bars = []
        for item_id in data.data_items():
            item = session.scene.get(item_id)
            variable = str(item.properties.get("variable", "none"))
            if not item.visible or variable == "none":
                continue
            mesh = data.load(item_id)
            if mesh is None:
                continue
            try:
                vmin, vmax = data.value_range(item_id, mesh)
            except KeyError:
                continue
            bars.append((f"{item.name} · {variable}", str(item.properties["colormap"]), vmin, vmax))
        font.setPixelSize(round(20 * unit))
        painter.setFont(font)
        bar_w, bar_h = width * 0.28, 14 * unit
        y = height - margin - len(bars) * 46 * unit
        for name, colormap, vmin, vmax in bars:
            x = margin
            painter.setPen(QColor("#1e2833"))
            painter.drawText(QRectF(x, y, bar_w, 22 * unit), Qt.AlignLeft, name)
            gradient = QLinearGradient(x, 0, x + bar_w, 0)
            table = lut(colormap, 16)
            for i, rgba in enumerate(table):
                gradient.setColorAt(i / (len(table) - 1), QColor.fromRgbF(*map(float, rgba)))
            top = y + 24 * unit
            painter.fillRect(QRectF(x, top, bar_w, bar_h), gradient)
            step = abs(vmax - vmin) / 10 or 1.0
            painter.drawText(
                QRectF(x + bar_w + 8 * unit, top - 5 * unit, 300 * unit, 24 * unit),
                Qt.AlignLeft,
                f"{tick_label(vmin, step)} … {tick_label(vmax, step)}",
            )
            y += 46 * unit
        painter.end()
        return np.frombuffer(qimage.constBits(), np.uint8).reshape(height, width, 4).copy()

    return overlay


def movie_folder(session: Session | None) -> Path:
    """Where films go by default: next to the case if that folder is writable,
    otherwise ~/Movies (or home)."""
    if session is not None:
        for item in session.scene:
            if item.kind == "case" and item.properties.get("path"):
                case = Path(str(item.properties["path"])).expanduser()
                folder = case if case.is_dir() else case.parent
                if os.access(folder, os.W_OK):
                    return folder
    movies = Path.home() / "Movies"
    return movies if movies.is_dir() else Path.home()


def export_animation(window) -> str | None:
    """Ask for the settings, film, and report. Returns the written path."""
    from lagoonviz.render.movie import frame_times, movie_available, write_movie

    session = window.session
    if session.data.n_frames() == 0:
        QMessageBox.information(window, "Export animation", "Open a case with output first.")
        return None
    viewport = window.viewport
    canvas = getattr(viewport, "canvas", viewport)
    size = (canvas.width(), canvas.height())
    dialog = ExportAnimationDialog(window, session, str(movie_folder(session)), size)
    if dialog.exec() != QDialog.Accepted:
        return None
    settings = dialog.settings()
    path = Path(settings["path"]).expanduser()
    if path.suffix.lower() not in (".mp4", ".mov", ".m4v"):
        path = path.with_suffix(".mp4")
    folder = path.parent
    if not folder.is_dir() or not os.access(folder, os.W_OK):
        QMessageBox.warning(
            window,
            "Export animation",
            f"Cannot write to {folder}.\nChoose a folder of yours, e.g. {movie_folder(None)}.",
        )
        return None
    if not movie_available():
        QMessageBox.warning(
            window,
            "Export animation",
            "MP4 needs the imageio-ffmpeg package; run setup_lagoon.command again.\n"
            f"Writing PNG frames to {path.with_suffix('')} instead.",
        )
    # film what the 3D view shows now: its camera and its look
    if hasattr(viewport, "view"):
        session.set_camera(viewport.view.camera_state())
    style = getattr(getattr(viewport, "view", None), "style_name", "studio")
    times = frame_times(session, settings["start"], settings["end"], settings["every"])
    keep_time = session.scene.time
    progress = QProgressDialog("Rendering frames…", "Stop", 0, len(times), window)
    progress.setWindowTitle("Export animation")
    progress.setWindowModality(Qt.WindowModal)
    progress.setMinimumDuration(0)

    def report(done: int, total: int) -> bool:
        progress.setValue(done)
        progress.setLabelText(f"Rendering frame {done} of {total}…")
        QApplication.processEvents()
        return not progress.wasCanceled()

    window.timeline.play_button.setChecked(False)
    if settings["engine"] == "cycles":
        progress.setLabelText("Handing the scene to Blender…")
        written = _path_trace(session, path, times, settings, style, progress, dialog.blender)
        session.set_time(keep_time)
        progress.close()
        if written is None:
            return None
        _record_film(session, path, settings, style)
        window.statusBar().showMessage(f"Path traced animation saved to {written}", 30000)
        return str(written)
    try:
        written = write_movie(
            session,
            path,
            times,
            settings["size"],
            settings["fps"],
            style,
            overlay=make_overlay(session) if settings["overlay"] else None,
            progress=report,
        )
    except Exception as error:  # noqa: BLE001 (report, keep the window)
        log.error("export failed:\n%s", traceback.format_exc())
        progress.close()
        QMessageBox.warning(window, "Export animation", f"The export failed:\n{error}")
        return None
    finally:
        session.set_time(keep_time)
    progress.close()
    _record_film(session, path, settings, style)
    from lagoonviz.render.movie import film_summary

    summary = film_summary()
    log.info("animation saved to %s: %s", written, summary)
    window.statusBar().showMessage(f"Animation saved to {written}: {summary}", 30000)
    return str(written)


def _record_film(session: Session, path: Path, settings: dict, style: str) -> None:
    """The film in the session's script."""
    try:
        from lagoonviz.render.script import film_code

        session.record_output(
            film_code(
                path,
                start=settings["start"],
                end=settings["end"],
                every=settings["every"],
                size=settings["size"],
                fps=settings["fps"],
                style=style,
                camera=session.scene.camera,
                plan=(session.scene.plan_view, session.scene.plan_side),
                overlay=bool(settings["overlay"]),
                engine=settings.get("engine", "pygfx"),
                samples=settings.get("samples", 128),
            )
        )
    except Exception:  # noqa: BLE001 (the film is written; only the script misses it)
        log.warning("film not recorded for the script:\n%s", traceback.format_exc())


def _path_trace(session, path, times, settings, style, progress, blender) -> Path | None:
    """Path trace with Blender: the scene is written here (it reads the session),
    Blender renders in a background thread while the dialog shows the frames done
    and the time left. Stop ends Blender; the frames done are kept, and exporting
    again goes on from them (frames whose data, camera and settings are unchanged)."""
    from lagoonviz.render.pathtrace import export_scene, finish, work_folder

    work = work_folder(path)

    def prepared(done: int, total: int) -> bool:
        progress.setMaximum(max(total, 1))
        progress.setValue(done)
        progress.setLabelText(f"Handing the scene to Blender: frame {done} of {total}…")
        QApplication.processEvents()
        return not progress.wasCanceled()

    try:
        export_scene(session, work, times, settings["size"], style, settings["samples"], prepared)
    except Exception as error:  # noqa: BLE001 (report, keep the window)
        log.error("path tracing export failed:\n%s", traceback.format_exc())
        QMessageBox.warning(progress.parentWidget(), "Export animation", f"Failed:\n{error}")
        return None
    if progress.wasCanceled():
        return None
    progress.setValue(0)
    state: dict = {"done": 0, "total": len(times), "result": None, "error": None}
    stop = threading.Event()

    def report(done: int, total: int) -> bool:
        state["done"], state["total"] = done, total
        return not stop.is_set()

    def run() -> None:
        try:
            state["result"] = finish(
                work, path, settings["fps"], blender, report, resume=True, cancel=stop.is_set
            )
        except Exception as error:  # noqa: BLE001 (reported below)
            log.error("path tracing failed:\n%s", traceback.format_exc())
            state["error"] = error

    worker = threading.Thread(target=run, daemon=True, name="lagoon-path-trace")
    started = time_module.monotonic()
    worker.start()
    while worker.is_alive():
        done, total = state["done"], state["total"]
        progress.setMaximum(max(total, 1))
        progress.setValue(done)
        elapsed = time_module.monotonic() - started
        left = f", about {_duration(elapsed / done * (total - done))} left" if done else ""
        progress.setLabelText(f"Blender Cycles: frame {done} of {total} done{left}")
        QApplication.processEvents()
        if progress.wasCanceled():
            stop.set()
        worker.join(0.1)
    if state["error"] is not None and not stop.is_set():
        QMessageBox.warning(
            progress.parentWidget(), "Export animation", f"Path tracing failed:\n{state['error']}"
        )
        return None
    return state["result"]


def _duration(seconds: float) -> str:
    seconds = max(int(seconds), 0)
    if seconds < 90:
        return f"{seconds} s"
    if seconds < 5400:
        return f"{round(seconds / 60)} min"
    return f"{seconds / 3600:.1f} h"


def render_still(window, samples: int = 128) -> str | None:
    """File → Render still with Cycles: the 3D view as it is now, path traced by
    Blender at the view's shape (Full HD width), in the view's look (open ocean for
    the engineering looks). Returns the written path."""
    from PySide6.QtWidgets import QFileDialog

    from lagoonviz.render.pathtrace import find_blender

    session = window.session
    blender = find_blender()
    if blender is None:
        QMessageBox.information(
            window,
            "Render still with Cycles",
            "Path tracing needs Blender 4.2 or newer: install it from blender.org "
            "(LAGOON finds it in Applications), or set LAGOON_BLENDER to its executable.",
        )
        return None
    path, _ = QFileDialog.getSaveFileName(
        window,
        "Render still with Cycles",
        str(movie_folder(session) / "lagoon_cycles.png"),
        "PNG image (*.png)",
    )
    if not path:
        return None
    path = Path(path).with_suffix(".png")
    viewport = window.viewport
    if hasattr(viewport, "view"):
        session.set_camera(viewport.view.camera_state())
    style = getattr(getattr(viewport, "view", None), "style_name", "studio")
    canvas = getattr(viewport, "canvas", viewport)
    size = view_sizes((canvas.width(), canvas.height()))[0][1]
    settings = {"size": size, "samples": samples, "fps": 25.0}
    progress = QProgressDialog("Handing the scene to Blender…", "Stop", 0, 1, window)
    progress.setWindowTitle("Render still with Cycles")
    progress.setWindowModality(Qt.WindowModal)
    progress.setMinimumDuration(0)
    keep_time = session.scene.time
    written = _path_trace(session, path, [keep_time], settings, style, progress, blender)
    session.set_time(keep_time)
    progress.close()
    if written is None:
        return None
    import shutil

    from lagoonviz.render.pathtrace import work_folder

    shutil.rmtree(work_folder(path), ignore_errors=True)  # a still keeps no frames
    try:
        from lagoonviz.render.script import cycles_still_code

        session.record_output(
            cycles_still_code(
                path,
                size=size,
                time=keep_time,
                style=style,
                samples=samples,
                camera=session.scene.camera,
                plan=(session.scene.plan_view, session.scene.plan_side),
            )
        )
    except Exception:  # noqa: BLE001 (the image is written; only the script misses it)
        log.warning("still not recorded for the script:\n%s", traceback.format_exc())
    window.statusBar().showMessage(f"Path traced still saved to {written}", 30000)
    return str(written)
