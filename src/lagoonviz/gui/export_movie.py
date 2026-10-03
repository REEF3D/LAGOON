# SPDX-License-Identifier: GPL-3.0-or-later
"""File → Export animation: film the current view, off screen, into an MP4."""

from __future__ import annotations

import logging
import os
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
        width, height = viewport_size
        self.size.addItem(f"As the 3D view {width} × {height}", (width, height))
        for label, value in SIZES.items():
            self.size.addItem(label, value)
        self.size.setCurrentIndex(2)  # Full HD
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
            "overlay": self.overlay.isChecked(),
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
    window.statusBar().showMessage(f"Animation saved to {written}", 15000)
    return str(written)
