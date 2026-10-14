# SPDX-License-Identifier: GPL-3.0-or-later
# Architect: Hans Bihs
"""File → Export figure: the plots, wave spectra, a comparison with measurements,
the free surface lines or the 3D view as a paper figure (PDF, SVG, EPS, PNG, TIFF),
drawn again with matplotlib at a journal width in a house style."""

from __future__ import annotations

import logging
import os
import shutil
import traceback
from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtGui import QImage, QPixmap
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
    QVBoxLayout,
    QWidget,
)

from lagoonviz.core import Session

log = logging.getLogger("lagoon.gui")

#: what can be exported: key, label
CONTENTS = (
    ("series", "Plots (time series)"),
    ("spectra", "Wave spectra of the plots"),
    ("validation", "Compare with measurements"),
    ("profiles", "Free surface lines"),
    ("view", "3D view"),
    ("setup", "Case set-up (plan and side view)"),
)
#: contents drawn from the plotted time series
FROM_SERIES = ("series", "spectra", "validation")
CUSTOM = "Custom"


def available_contents(session: Session, has_rows: bool) -> list[tuple[str, str]]:
    """The contents that have something to draw."""
    data = session.data
    found = []
    for key, label in CONTENTS:
        if key in FROM_SERIES and not has_rows:
            continue
        if key == "profiles" and not any(
            session.scene.get(i).visible and data.profile_lines(i) for i in data.profile_items()
        ):
            continue
        if key == "view" and not data.data_items():
            continue
        if key == "setup" and not any(
            data.case_setup(i.id) is not None for i in session.scene if i.kind == "case"
        ):
            continue
        found.append((key, label))
    return found


class ExportFigureDialog(QDialog):
    def __init__(
        self,
        parent: QWidget,
        contents: list[tuple[str, str]],
        folder: str,
        lines: list[tuple[str, int, str]] | None = None,
    ) -> None:
        """``lines``: (label, row, column) of the plotted lines, for comparisons."""
        super().__init__(parent)
        from lagoonviz.render.figure import STYLES, WIDTHS

        self.setWindowTitle("Export figure")
        self._folder = folder
        form = QFormLayout()
        self.form = form
        self.content = QComboBox()
        for key, label in contents:
            self.content.addItem(label, key)
        form.addRow("Figure of", self.content)

        self.style = QComboBox()
        for name in STYLES:
            self.style.addItem(name)
        self.style.setToolTip("House style: fonts, line widths and line colours")
        form.addRow("Style", self.style)

        # comparisons: one plotted line against a measured file
        self.line = QComboBox()
        for label, row, column in lines or []:
            self.line.addItem(label, (row, column))
        self.measured = QLineEdit()
        self.measured.setPlaceholderText("time and values in columns (text, CSV)")
        self.measured.editingFinished.connect(self._read_columns)
        measured_browse = QPushButton("Browse…")
        measured_browse.clicked.connect(self._browse_measured)
        row = QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        row.addWidget(self.measured, 1)
        row.addWidget(measured_browse)
        self.measured_row = QWidget()
        self.measured_row.setLayout(row)
        self.measured_column = QComboBox()
        form.addRow("Computed", self.line)
        form.addRow("Measured file", self.measured_row)
        form.addRow("Measured column", self.measured_column)

        # spectra and comparisons: the part of the run to use
        self.start = QDoubleSpinBox()
        self.end = QDoubleSpinBox()
        for box, text in ((self.start, "from the start"), (self.end, "to the end")):
            box.setRange(0, 1e7)
            box.setDecimals(2)
            box.setSuffix(" s")
            box.setSpecialValueText(text)
        row = QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        row.addWidget(self.start)
        row.addWidget(QLabel("to"))
        row.addWidget(self.end)
        self.window_row = QWidget()
        self.window_row.setLayout(row)
        form.addRow("Time window", self.window_row)
        self.content.currentIndexChanged.connect(self._content_chosen)

        self.width_choice = QComboBox()
        for name, width in WIDTHS.items():
            self.width_choice.addItem(f"{name[0].upper()}{name[1:]} ({width:g} mm)", width)
        self.width_choice.addItem(CUSTOM, None)
        self.width = QDoubleSpinBox()
        self.width.setRange(20, 1000)
        self.width.setDecimals(0)
        self.width.setSuffix(" mm")
        self.width.setValue(90)
        self.width.setEnabled(False)
        self.width_choice.currentIndexChanged.connect(self._width_chosen)
        row = QHBoxLayout()
        row.addWidget(self.width_choice, 1)
        row.addWidget(self.width)
        holder = QWidget()
        holder.setLayout(row)
        row.setContentsMargins(0, 0, 0, 0)
        form.addRow("Width", holder)

        self.height = QDoubleSpinBox()
        self.height.setRange(0, 1000)
        self.height.setDecimals(0)
        self.height.setSuffix(" mm")
        self.height.setSpecialValueText("automatic")
        self.height.setValue(0)
        form.addRow("Height", self.height)

        self.font_size = QDoubleSpinBox()
        self.font_size.setRange(4, 24)
        self.font_size.setDecimals(1)
        self.font_size.setSuffix(" pt")
        self.font_size.setValue(8)
        form.addRow("Font size", self.font_size)

        self.font_family = QComboBox()
        self.font_family.addItem("Sans serif", "sans-serif")
        self.font_family.addItem("Serif", "serif")
        form.addRow("Font", self.font_family)

        self.dpi = QSpinBox()
        self.dpi.setRange(72, 2400)
        self.dpi.setValue(300)
        self.dpi.setSuffix(" dpi")
        self.dpi.setToolTip("Resolution of PNG/TIFF files and of the 3D view's image")
        form.addRow("Resolution", self.dpi)

        self.usetex = QCheckBox("Labels through LaTeX")
        latex = shutil.which("latex") is not None
        self.usetex.setEnabled(latex)
        self.usetex.setToolTip(
            "Typeset all text with your LaTeX installation"
            if latex
            else "No LaTeX found; $...$ in labels still works through matplotlib's mathtext"
        )
        form.addRow("", self.usetex)
        self.style.currentTextChanged.connect(self._style_chosen)
        self._content_chosen()

        self.title = QLineEdit()
        self.title.setPlaceholderText("automatic (time) or none")
        form.addRow("Title", self.title)

        self.path = QLineEdit(str(Path(folder) / "figure.pdf"))
        browse = QPushButton("Browse…")
        browse.clicked.connect(self._browse)
        row = QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        row.addWidget(self.path, 1)
        row.addWidget(browse)
        holder = QWidget()
        holder.setLayout(row)
        form.addRow("File", holder)

        self.preview = QLabel()
        self.preview.setAlignment(Qt.AlignCenter)
        self.preview.setMinimumSize(360, 200)
        self.preview.setText("Preview")
        preview_button = QPushButton("Preview")
        preview_button.clicked.connect(self.show_preview)
        self.render_preview = None  # set by export_figure: settings -> QImage

        buttons = QDialogButtonBox(QDialogButtonBox.Save | QDialogButtonBox.Cancel)
        buttons.addButton(preview_button, QDialogButtonBox.ActionRole)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout = QVBoxLayout(self)
        layout.addLayout(form)
        layout.addWidget(self.preview, 1)
        layout.addWidget(buttons)

    def _content_chosen(self) -> None:
        """Only the settings the chosen figure uses."""
        content = self.content.currentData()
        comparing = content == "validation"
        for widget in (self.line, self.measured_row, self.measured_column):
            self.form.setRowVisible(widget, comparing)
        self.form.setRowVisible(self.window_row, content in ("spectra", "validation"))
        if content == "setup" and self.width.value() < 140:
            # plan, side view, tables and legend: a column and a half at least
            index = self.width_choice.findData(140.0)
            if index >= 0:
                self.width_choice.setCurrentIndex(index)

    def _browse_measured(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Measured time series", self._folder, "Time series (*.dat *.txt *.csv);;All (*)"
        )
        if path:
            self.measured.setText(path)
            self._read_columns()

    def _read_columns(self) -> None:
        from lagoonviz.core.analysis import read_measurements

        self.measured_column.clear()
        text = self.measured.text().strip()
        if not text:
            return
        try:
            _time, columns = read_measurements(Path(text).expanduser())
        except (OSError, ValueError) as error:
            self.measured_column.addItem(f"cannot read: {error}", None)
            return
        for name in columns:
            self.measured_column.addItem(name, name)

    def _width_chosen(self) -> None:
        width = self.width_choice.currentData()
        self.width.setEnabled(width is None)
        if width is not None:
            self.width.setValue(width)

    def _browse(self) -> None:
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Export figure",
            self.path.text(),
            "PDF (*.pdf);;SVG (*.svg);;EPS (*.eps);;PNG (*.png);;TIFF (*.tif)",
        )
        if path:
            self.path.setText(path)

    def _style_chosen(self, name: str) -> None:
        """A style sets the font; the boxes can still change it."""
        from lagoonviz.render.figure import STYLES

        style = STYLES.get(name)
        if style is None:
            return
        self.font_size.setValue(style["font_size"])
        self.font_family.setCurrentIndex(max(self.font_family.findData(style["font_family"]), 0))

    def settings(self) -> dict:
        return {
            "content": self.content.currentData(),
            "style": self.style.currentText(),
            "line": self.line.currentData(),
            "measured": self.measured.text().strip(),
            "measured_column": self.measured_column.currentData(),
            "start": self.start.value() or None,
            "end": self.end.value() or None,
            "width_mm": self.width.value(),
            "height_mm": self.height.value() or None,
            "font_size": self.font_size.value(),
            "font_family": self.font_family.currentData(),
            "dpi": self.dpi.value(),
            "usetex": self.usetex.isChecked(),
            "title": self.title.text().strip(),
            "path": self.path.text().strip(),
        }

    def show_preview(self) -> None:
        if self.render_preview is None:
            return
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            image = self.render_preview(self.settings())
        except Exception as error:  # noqa: BLE001 (shown, the dialog stays)
            self.preview.setText(f"No preview: {error}")
            return
        finally:
            QApplication.restoreOverrideCursor()
        pixmap = QPixmap.fromImage(image)
        self.preview.setPixmap(
            pixmap.scaled(self.preview.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        )


def make_figure(window, settings: dict, dpi: int | None = None):
    """The matplotlib figure for the dialog's settings (``dpi`` overrides, for previews)."""
    from lagoonviz.core.analysis import read_measurements
    from lagoonviz.render.figure import (
        FigureSpec,
        profile_figure,
        series_figure,
        spectrum_figure,
        validation_figure,
        view_figure,
    )

    spec = FigureSpec.from_style(
        settings.get("style", "REEF3D"),
        width_mm=settings["width_mm"],
        height_mm=settings["height_mm"],
        dpi=dpi or settings["dpi"],
        font_size=settings["font_size"],
        font_family=settings["font_family"],
        usetex=settings["usetex"],
        title=settings["title"],
    )
    session = window.session
    content = settings["content"]
    rows = window.plots.shown_rows()
    if content == "series":
        return series_figure(session, spec, rows=rows, window=window.plots.time_window()), spec
    if content == "spectra":
        return spectrum_figure(
            session, spec, rows=rows, start=settings.get("start"), end=settings.get("end")
        ), spec
    if content == "validation":
        if not settings.get("measured") or not settings.get("measured_column"):
            raise ValueError("choose a measured file and its column")
        row, column = settings["line"]
        trace = rows[row][1][column]
        time, columns = read_measurements(Path(settings["measured"]).expanduser())
        name = f"{Path(settings['measured']).stem} {settings['measured_column']}"
        return validation_figure(
            trace,
            time,
            columns[settings["measured_column"]],
            spec,
            name,
            settings.get("start"),
            settings.get("end"),
        ), spec
    if content == "profiles":
        return profile_figure(session, spec), spec
    if content == "setup":
        from lagoonviz.render.setup_sketch import setup_figure

        return setup_figure(session, spec), spec
    viewport = window.viewport
    view = getattr(viewport, "view", None)
    if view is not None:
        session.set_camera(view.camera_state())  # what the 3D view shows now
    style = getattr(view, "style_name", "studio")
    return view_figure(session, spec, style=style), spec


def _preview_image(window, settings: dict) -> QImage:
    import io

    figure, spec = make_figure(window, settings, dpi=110)
    buffer = io.BytesIO()
    import matplotlib

    with matplotlib.rc_context(spec.rc()):
        figure.savefig(buffer, format="png", dpi=110)
    image = QImage()
    image.loadFromData(buffer.getvalue(), "PNG")
    return image


def _row_label(session: Session, item_id: str) -> str:
    if item_id in session.scene:
        return session.scene.get(item_id).name
    return item_id


def export_figure(window) -> str | None:
    """Ask for the settings, draw the figure and save it. Returns the written path."""
    from lagoonviz.gui.export_movie import movie_folder

    try:
        import matplotlib  # noqa: F401
    except ImportError:
        QMessageBox.warning(
            window, "Export figure", "Figures need matplotlib; run setup_lagoon.command again."
        )
        return None
    session = window.session
    contents = available_contents(session, bool(window.plots.shown_rows()))
    if not contents:
        QMessageBox.information(
            window, "Export figure", "Open a case first: there is nothing to draw."
        )
        return None
    lines = [
        (f"{trace.name} ({_row_label(session, item_id)})", r, c)
        for r, (item_id, traces) in enumerate(window.plots.shown_rows())
        for c, trace in enumerate(traces)
    ]
    dialog = ExportFigureDialog(window, contents, str(movie_folder(session)), lines)
    dialog.render_preview = lambda settings: _preview_image(window, settings)
    if dialog.exec() != QDialog.Accepted:
        return None
    settings = dialog.settings()
    path = Path(settings["path"]).expanduser()
    if not os.access(path.parent if path.parent.exists() else Path.home(), os.W_OK):
        QMessageBox.warning(window, "Export figure", f"Cannot write to {path.parent}.")
        return None
    from lagoonviz.render.figure import save_figure

    QApplication.setOverrideCursor(Qt.WaitCursor)
    try:
        figure, spec = make_figure(window, settings)
        written = save_figure(figure, path, spec)
    except Exception as error:  # noqa: BLE001 (report, keep the window)
        log.error("figure export failed:\n%s", traceback.format_exc())
        QMessageBox.warning(window, "Export figure", f"The figure could not be made:\n{error}")
        return None
    finally:
        QApplication.restoreOverrideCursor()
    record_figure(window, settings, written)
    window.statusBar().showMessage(f"Figure saved to {written}", 15000)
    return str(written)


def record_figure(window, settings: dict, path) -> None:
    """The figure into the session's recorded script (File → Export script)."""
    from lagoonviz.render.script import figure_code

    session = window.session
    try:
        rows = window.plots.shown_rows()
        line = None
        if settings["content"] == "validation" and settings.get("line") is not None:
            row, column = settings["line"]
            item_id, traces = rows[row]
            line = (item_id, traces[column].name)
        view = getattr(window.viewport, "view", None)
        code = figure_code(
            settings["content"],
            path,
            settings,
            time=session.scene.time,
            items=[item_id for item_id, _ in rows],
            window=window.plots.time_window() if rows else None,
            line=line,
            camera=session.scene.camera,
            plan=(session.scene.plan_view, session.scene.plan_side),
            style=getattr(view, "style_name", "studio"),
        )
    except Exception:  # noqa: BLE001 (the figure is written; only the script misses it)
        log.warning("figure not recorded for the script:\n%s", traceback.format_exc())
        return
    session.record_output(code)
