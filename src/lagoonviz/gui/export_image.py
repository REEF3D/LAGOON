# SPDX-License-Identifier: GPL-3.0-or-later
# Architect: Hans Bihs
"""File → Export image: the 3D view alone, drawn again off screen at a size of
one's choice (as on the screen, twice that, Full HD to 8K wide, a width in pixels,
or a width in centimetres at a print resolution), as PNG, JPEG or TIFF.

The image frames what the 3D view shows (its shape is the view's), and lines,
markers and text keep their share of the image: a 4K image of the view is the view
drawn sharper, not smaller. Save screenshot stays the quick picture of the window.

Two renderers, as for Export animation: LAGOON's own (at once, as the 3D view draws
it) and Blender's Cycles (path traced, minutes; the same scene, look, samples and
white water as a film's frame at that time).
"""

from __future__ import annotations

import logging
import os
import traceback
from pathlib import Path

from PySide6.QtCore import Qt
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
    QPushButton,
    QSpinBox,
    QWidget,
)

from lagoonviz.core import Session

log = logging.getLogger("lagoon.gui")

#: the window settings the dialog remembers
SETTINGS = "export_image/"
CM_PER_INCH = 2.54


def size_choices(view_size, pixel_ratio: float = 1.0) -> list[tuple[str, tuple]]:
    """The sizes offered for a 3D view ``view_size`` (logical pixels) shown at
    ``pixel_ratio`` (2 on a Retina screen): (label, (kind, value)); kind "width"
    (pixels), "custom" (a width of one's own) or "print" (centimetres at a dpi)."""
    width, height = (max(int(v), 1) for v in view_size)
    screen = max(int(round(width * max(pixel_ratio, 1.0))), 1)
    choices = [
        (f"As on the screen: {screen} × {_height(screen, view_size)}", ("width", screen)),
        (f"Twice as sharp: {2 * screen} × {_height(2 * screen, view_size)}", ("width", 2 * screen)),
    ]
    for name, target in (("Full HD", 1920), ("4K", 3840), ("8K", 7680)):
        choices.append(
            (f"{name} width: {target} × {_height(target, view_size)}", ("width", target))
        )
    choices.append(("Width in pixels of my choice", ("custom", None)))
    choices.append(("For print: width in cm at a resolution", ("print", None)))
    return choices


def _height(width: int, view_size) -> int:
    """The height of an image ``width`` wide with the view's shape."""
    w, h = (max(float(v), 1.0) for v in view_size)
    return max(int(round(width * h / w)), 1)


def pixels_for(choice: tuple, view_size, custom: int, cm: float, dpi: float):
    """An image's pixels (width, height), its scale to the view (image pixels to a
    view pixel) and its dpi (print only, else None)."""
    kind, value = choice
    if kind == "custom":
        width = int(custom)
    elif kind == "print":
        width = int(round(float(cm) / CM_PER_INCH * float(dpi)))
    else:
        width = int(value)
    width = max(width, 1)
    size = (width, _height(width, view_size))
    scale = width / max(float(view_size[0]), 1.0)
    return size, scale, (float(dpi) if kind == "print" else None)


class ExportImageDialog(QDialog):
    """Where to, how large, what on it."""

    def __init__(
        self, parent: QWidget, session: Session, path: str, view_size, pixel_ratio=1.0
    ) -> None:
        from lagoonviz.render.movie import texture_limit

        super().__init__(parent)
        self.setWindowTitle("Export image")
        self.session = session
        self.view_size = tuple(int(v) for v in view_size)
        self.limit = texture_limit()
        settings = getattr(parent, "settings", None)
        self._settings = settings
        form = QFormLayout(self)

        self.path = QLineEdit(path)
        browse = QPushButton("Choose…")
        browse.clicked.connect(self._browse)
        row = QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        row.addWidget(self.path, 1)
        row.addWidget(browse)
        holder = QWidget()
        holder.setLayout(row)
        form.addRow("File", holder)

        self.size = QComboBox()
        for label, value in size_choices(self.view_size, pixel_ratio):
            self.size.addItem(label, value)
        self.size.setToolTip(
            "The image frames what the 3D view shows; lines, markers and text keep "
            "their share of the image (a larger image is the view drawn sharper)"
        )
        form.addRow("Size", self.size)

        self.custom = QSpinBox()
        self.custom.setRange(16, 100_000)
        self.custom.setSingleStep(100)
        self.custom.setSuffix(" px wide")
        self.custom.setValue(3000)
        self.custom_label = QLabel("Width")
        form.addRow(self.custom_label, self.custom)

        self.cm = QDoubleSpinBox()
        self.cm.setRange(1.0, 500.0)
        self.cm.setDecimals(1)
        self.cm.setSuffix(" cm")
        self.cm.setValue(17.0)
        self.cm.setToolTip("17 cm: a journal's full page width; 8.5 cm: one column")
        self.dpi = QSpinBox()
        self.dpi.setRange(36, 2400)
        self.dpi.setSingleStep(50)
        self.dpi.setSuffix(" dpi")
        self.dpi.setValue(300)
        printed = QHBoxLayout()
        printed.setContentsMargins(0, 0, 0, 0)
        printed.addWidget(self.cm)
        printed.addWidget(QLabel("at"))
        printed.addWidget(self.dpi)
        self.print_row = QWidget()
        self.print_row.setLayout(printed)
        self.print_label = QLabel("Width")
        form.addRow(self.print_label, self.print_row)

        self.summary = QLabel()
        self.summary.setEnabled(False)
        form.addRow("", self.summary)

        self.legend = QCheckBox("Time and colour bars")
        self.legend.setChecked(True)
        self.legend.setToolTip("Where the legend layout puts them (View → Legend…)")
        legend_button = QPushButton("Legend…")
        legend_button.setToolTip("Where the time and the colour bars go, and their look")
        legend_button.clicked.connect(self._legend)
        shown = QHBoxLayout()
        shown.setContentsMargins(0, 0, 0, 0)
        shown.addWidget(self.legend)
        shown.addWidget(legend_button)
        shown.addStretch(1)
        holder = QWidget()
        holder.setLayout(shown)
        form.addRow("Show", holder)

        self._cycles_controls(form, parent)

        self.buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        self.buttons.button(QDialogButtonBox.Ok).setText("Export")
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)
        form.addRow(self.buttons)
        self.form = form

        self._restore()
        for signal in (
            self.size.currentIndexChanged,
            self.custom.valueChanged,
            self.cm.valueChanged,
            self.dpi.valueChanged,
            self.engine.currentIndexChanged,
        ):
            signal.connect(lambda *_: self._update())
        self._update()

    def _cycles_controls(self, form: QFormLayout, parent) -> None:
        """The renderer, and for Cycles its samples and white water (as Export
        animation has them)."""
        from lagoonviz.core.foam import WHITE_WATER_AMOUNTS
        from lagoonviz.gui.export_movie import white_water_setting
        from lagoonviz.render.pathtrace import find_blender

        self.blender = find_blender()
        self.engine = QComboBox()
        self.engine.addItem("LAGOON (at once, as the 3D view draws it)", "pygfx")
        self.engine.addItem("Blender Cycles (path traced, slow)", "cycles")
        self.engine.setToolTip(
            "Cycles traces the light: reflections, refraction through the water, soft "
            "shadows. In the look of the 3D view if it is a cinematic one, else open ocean; "
            "the same picture as a path traced film's frame at this time."
        )
        form.addRow("Renderer", self.engine)
        self.samples = QSpinBox()
        self.samples.setRange(4, 4096)
        self.samples.setValue(64)
        self.samples.setToolTip("Samples per pixel: more is less noisy and slower")
        self.foam = QComboBox()
        for name in WHITE_WATER_AMOUNTS:
            self.foam.addItem(name, name)
        self.foam.setCurrentIndex(max(self.foam.findData(white_water_setting(parent)), 0))
        self.foam.setToolTip(
            "White water where the surface breaks: spray off the crests, foam drifting "
            "on the surface, bubbles under it (from the free surface's velocity); also "
            "View → White water in Cycles"
        )
        self.blender_label = QLabel(
            f"Blender: {self.blender}"
            if self.blender
            else "Blender not found: install it from blender.org (4.2 or newer), "
            "or set LAGOON_BLENDER"
        )
        self.blender_label.setWordWrap(True)
        cycles = QHBoxLayout()
        cycles.setContentsMargins(0, 0, 0, 0)
        cycles.addWidget(QLabel("samples"))
        cycles.addWidget(self.samples)
        cycles.addWidget(QLabel("white water"))
        cycles.addWidget(self.foam)
        cycles.addWidget(self.blender_label, 1)
        self.cycles_row = QWidget()
        self.cycles_row.setLayout(cycles)
        form.addRow("", self.cycles_row)

    # ------------------------------------------------------------- state
    def _restore(self) -> None:
        settings = self._settings
        if settings is None:
            return
        try:
            index = int(settings.value(SETTINGS + "size", 0))
            self.custom.setValue(int(settings.value(SETTINGS + "custom", self.custom.value())))
            self.cm.setValue(float(settings.value(SETTINGS + "cm", self.cm.value())))
            self.dpi.setValue(int(settings.value(SETTINGS + "dpi", self.dpi.value())))
            legend = settings.value(SETTINGS + "legend", True)
            self.legend.setChecked(str(legend).lower() not in ("false", "0"))
            self.samples.setValue(int(settings.value(SETTINGS + "samples", self.samples.value())))
            engine = self.engine.findData(str(settings.value(SETTINGS + "engine", "pygfx")))
            self.engine.setCurrentIndex(max(engine, 0))
        except (TypeError, ValueError):
            return
        if 0 <= index < self.size.count():
            self.size.setCurrentIndex(index)

    def remember(self) -> None:
        settings = self._settings
        if settings is None:
            return
        settings.setValue(SETTINGS + "size", self.size.currentIndex())
        settings.setValue(SETTINGS + "custom", self.custom.value())
        settings.setValue(SETTINGS + "cm", self.cm.value())
        settings.setValue(SETTINGS + "dpi", self.dpi.value())
        settings.setValue(SETTINGS + "legend", self.legend.isChecked())
        settings.setValue(SETTINGS + "engine", self.engine.currentData())
        settings.setValue(SETTINGS + "samples", self.samples.value())

    def chosen(self):
        """(width, height), scale to the view, dpi (or None)."""
        return pixels_for(
            self.size.currentData(),
            self.view_size,
            self.custom.value(),
            self.cm.value(),
            self.dpi.value(),
        )

    @property
    def cycles(self) -> bool:
        return self.engine.currentData() == "cycles"

    def _update(self) -> None:
        kind = self.size.currentData()[0]
        self.form.setRowVisible(self.custom, kind == "custom")
        self.form.setRowVisible(self.print_row, kind == "print")
        cycles = self.cycles
        self.form.setRowVisible(self.cycles_row, cycles)
        self.legend.setEnabled(not cycles)  # Cycles shows the scene alone, as in its films
        (width, height), _, dpi = self.chosen()
        megabytes = width * height * 4 / 1e6
        text = f"{width} × {height} pixels"
        if dpi:
            text += f" ({width / dpi * CM_PER_INCH:.1f} × {height / dpi * CM_PER_INCH:.1f} cm)"
        too_large = not cycles and max(width, height) > self.limit  # (the GPU's)
        if cycles and not self.blender:
            text += ": Cycles needs Blender"
        if too_large:
            text += f": too large, the GPU draws {self.limit} pixels a side at most"
        elif megabytes > 100:
            text += f", about {megabytes:.0f} MB in memory"
        self.summary.setText(text)
        blocked = too_large or (cycles and not self.blender)
        self.summary.setEnabled(blocked)
        self.summary.setStyleSheet("color: #c0392b;" if blocked else "")
        self.buttons.button(QDialogButtonBox.Ok).setEnabled(not blocked)

    def _legend(self) -> None:
        from lagoonviz.gui.legend_dialog import LegendDialog

        LegendDialog(self, self.session).exec()

    def _browse(self) -> None:
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Export image",
            self.path.text(),
            "PNG image (*.png);;JPEG image (*.jpg *.jpeg);;TIFF image (*.tif *.tiff)",
        )
        if path:
            self.path.setText(path)

    def settings(self) -> dict:
        size, scale, dpi = self.chosen()
        return {
            "path": self.path.text().strip(),
            "size": size,
            "scale": scale,
            "dpi": dpi,
            "legend": self.legend.isChecked() and not self.cycles,
            "engine": self.engine.currentData(),
            "samples": self.samples.value(),
            "foam": self.foam.currentData(),
        }


def image_name(session: Session) -> str:
    """A file name for the view at the current time."""
    time = session.scene.time
    seconds = session.data.time_in_seconds() if session.data.n_frames() else False
    stamp = f"_t{time:.2f}s".replace(".", "p") if seconds else ""
    return f"lagoon_view{stamp}.png"


def export_image(window, path: str | Path | None = None, **chosen) -> str | None:
    """Ask for the file and size (or take ``path`` and the dialog's settings as
    keywords: size, scale, dpi, legend; engine "cycles" with samples and foam), draw
    the 3D view off screen (or path trace it with Blender) and write it. Returns the
    path written."""
    from lagoonviz.gui.export_movie import movie_folder
    from lagoonviz.render.offscreen import IMAGE_SUFFIXES, render_image

    if isinstance(path, bool):  # a menu's triggered(checked) passed on: ask
        path = None
    session = window.session
    viewport = window.viewport
    canvas = getattr(viewport, "canvas", viewport)
    view_size = (max(canvas.width(), 1), max(canvas.height(), 1))
    if path is None:
        folder = movie_folder(session)
        dialog = ExportImageDialog(
            window,
            session,
            str(folder / image_name(session)),
            view_size,
            canvas.devicePixelRatioF() if hasattr(canvas, "devicePixelRatioF") else 1.0,
        )
        if dialog.exec() != QDialog.Accepted:
            return None
        dialog.remember()
        settings = dialog.settings()
    else:
        size = chosen.get("size") or view_size
        settings = {
            "path": str(path),
            "size": tuple(int(v) for v in size),
            "scale": float(chosen.get("scale", size[0] / view_size[0])),
            "dpi": chosen.get("dpi"),
            "legend": bool(chosen.get("legend", False)),
            "engine": str(chosen.get("engine", "pygfx")),
            "samples": int(chosen.get("samples", 64)),
            "foam": chosen.get("foam", None),
        }
    target = Path(settings["path"]).expanduser()
    if not target.suffix:
        target = target.with_suffix(".png")
    if target.suffix.lower() not in IMAGE_SUFFIXES:
        QMessageBox.warning(
            window, "Export image", f"{target.name}: choose .png, .jpg or .tif as the file's end"
        )
        return None
    if not target.parent.is_dir() or not os.access(target.parent, os.W_OK):
        QMessageBox.warning(
            window,
            "Export image",
            f"Cannot write to {target.parent}.\n"
            f"Choose a folder of yours, e.g. {movie_folder(None)}.",
        )
        return None
    # draw what the 3D view shows now: its camera and its look
    view = getattr(viewport, "view", None)
    if view is not None:
        session.set_camera(view.camera_state())
    style = getattr(view, "style_name", "studio")
    window.timeline.play_button.setChecked(False)
    if settings["engine"] == "cycles":
        return _path_trace_still(window, target, settings, style)
    QApplication.setOverrideCursor(Qt.WaitCursor)
    try:
        written = render_image(
            session,
            target,
            settings["size"],
            scale=settings["scale"],
            style=style,
            legend=settings["legend"],
            dpi=settings["dpi"],
        )
    except Exception as error:  # noqa: BLE001 (report, keep the window)
        log.error("image export failed:\n%s", traceback.format_exc())
        QApplication.restoreOverrideCursor()
        QMessageBox.warning(window, "Export image", f"The export failed:\n{error}")
        return None
    QApplication.restoreOverrideCursor()
    _record(session, written, settings, style)
    width, height = settings["size"]
    window.statusBar().showMessage(f"Image saved to {written} ({width} × {height})", 15000)
    return str(written)


def _path_trace_still(window, target: Path, settings: dict, style: str) -> str | None:
    """The still path traced by Blender: the scene of a path traced film's frame at
    the session's time (the same look, samples, white water and size), written as
    the image asked for."""
    import shutil

    from PySide6.QtWidgets import QProgressDialog

    from lagoonviz.gui.export_movie import _path_trace, set_white_water, white_water_setting
    from lagoonviz.render.pathtrace import find_blender, work_folder

    session = window.session
    blender = find_blender()
    if blender is None:
        QMessageBox.information(
            window,
            "Export image",
            "Path tracing needs Blender 4.2 or newer: install it from blender.org "
            "(LAGOON finds it in Applications), or set LAGOON_BLENDER to its executable.",
        )
        return None
    foam = settings.get("foam") or white_water_setting(window)
    set_white_water(window, foam)  # remembered for the next film or still
    traced = {**settings, "fps": 25.0, "foam": foam}
    progress = QProgressDialog("Handing the scene to Blender…", "Stop", 0, 1, window)
    progress.setWindowTitle("Export image")
    progress.setWindowModality(Qt.WindowModal)
    progress.setMinimumDuration(0)
    keep_time = session.scene.time
    written = _path_trace(
        session, target, [keep_time], traced, style, progress, blender, "Export image"
    )
    session.set_time(keep_time)
    progress.close()
    if written is None:
        return None
    shutil.rmtree(work_folder(target), ignore_errors=True)  # a still keeps no frames
    try:
        from lagoonviz.render.script import cycles_still_code

        session.record_output(
            cycles_still_code(
                target,
                size=settings["size"],
                time=keep_time,
                style=style,
                samples=settings["samples"],
                camera=session.scene.camera,
                plan=(session.scene.plan_view, session.scene.plan_side),
                foam=foam,
                dpi=settings["dpi"],
            )
        )
    except Exception:  # noqa: BLE001 (the image is written; only the script misses it)
        log.warning("still not recorded for the script:\n%s", traceback.format_exc())
    width, height = settings["size"]
    window.statusBar().showMessage(
        f"Path traced image saved to {written} ({width} × {height})", 30000
    )
    return str(written)


def _record(session: Session, path: Path, settings: dict, style: str) -> None:
    """The image in the session's script."""
    try:
        from lagoonviz.render.script import image_code

        session.record_output(
            image_code(
                path,
                size=settings["size"],
                scale=round(float(settings["scale"]), 4),
                time=session.scene.time,
                style=style,
                camera=session.scene.camera,
                plan=(session.scene.plan_view, session.scene.plan_side),
                legend=bool(settings["legend"]),
                dpi=settings["dpi"],
            )
        )
    except Exception:  # noqa: BLE001 (the image is written; only the script misses it)
        log.warning("image not recorded for the script:\n%s", traceback.format_exc())
