# SPDX-License-Identifier: GPL-3.0-or-later
# Architect: Hans Bihs
"""View → Wave analysis: the gauges' spectra and wave parameters, reflection, and the
response of a floating body to the waves, over a time window
(lagoonviz.core.analysis, lagoonviz.core.waves; figures: lagoonviz.render.wave_figures).

- *Spectra*: the lines ticked, their spectra and a table: Hm0, Tp, Tm01, Tm02 from
  the spectrum, H1/3, Hmax, T1/3 and the number of waves from zero down-crossings.
- *Reflection*: two or three gauges along the wave direction and the water depth:
  incident and reflected spectra (Mansard and Funke; two gauges: Goda and Suzuki),
  Kr, incident and reflected Hm0.
- *Response*: a wave gauge in, a motion out: gain (RAO), phase, coherence.

The window is the Plots panel's time window when the window opens; *Copy table*
puts the numbers on the clipboard (tab separated), *Save figure* writes the figure
of the tab (PDF, SVG, PNG).
"""

from __future__ import annotations

import io
import logging
import math

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtGui import QGuiApplication, QImage, QPixmap
from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QSpinBox,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from lagoonviz.core.analysis import spectrum, wave_parameters
from lagoonviz.core.legend import line_name
from lagoonviz.core.waves import reflection, transfer, zero_crossing

log = logging.getLogger(__name__)

PARAMETERS = ("Hm0", "Tp", "Tm01", "Tm02", "H1/3", "Hmax", "T1/3", "waves")


def analysis_lines(session) -> list[tuple[str, str, object]]:
    """(key, label, Trace) of every time series line of the scene, measured ones too."""
    data = session.data
    found = []
    for item_id in data.series_items():
        item = session.scene.get(item_id)
        for trace in data.traces(item_id) + data.measured_traces(item_id):
            name = (
                f"{line_name(item, trace.measured_for)} measured"
                if trace.measured_for
                else line_name(item, trace.name)
            )
            found.append((f"{item_id}/{trace.name}", f"{item.name}: {name}", trace))
    return found


def _number(spin_range=(-1e9, 1e9), decimals=3, value=0.0) -> QDoubleSpinBox:
    box = QDoubleSpinBox()
    box.setRange(*spin_range)
    box.setDecimals(decimals)
    box.setValue(value)
    box.setKeyboardTracking(False)
    return box


def _pixmap(figure, width: int) -> QPixmap:
    import matplotlib

    buffer = io.BytesIO()
    with matplotlib.rc_context({"font.size": 8}):
        figure.savefig(buffer, format="png", dpi=110)
    image = QImage()
    image.loadFromData(buffer.getvalue(), "PNG")
    pixmap = QPixmap.fromImage(image)
    if width > 50 and pixmap.width() > width:
        pixmap = pixmap.scaledToWidth(width, Qt.SmoothTransformation)
    return pixmap


def _fmt(value: float, digits: int = 4) -> str:
    return f"{value:.{digits}g}" if value is not None and math.isfinite(value) else "–"


class WaveAnalysisDialog(QDialog):
    def __init__(self, session, window: tuple[float, float] | None = None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Wave analysis")
        self.resize(980, 620)
        self.session = session
        self.lines = analysis_lines(session)
        self.by_key = {key: trace for key, _label, trace in self.lines}
        self.labels = {key: label for key, label, _trace in self.lines}
        self.figures: dict[str, object] = {}
        self.tables: dict[str, list[list[str]]] = {}

        low, high = window or self._full_window()
        self.start = _number(value=low)
        self.end = _number(value=high)
        self.segments = QSpinBox()
        self.segments.setRange(1, 64)
        self.segments.setValue(8)
        self.segments.setToolTip(
            "Welch segments, half overlapping: more give smoother spectra, fewer finer frequencies"
        )
        top = QHBoxLayout()
        top.addWidget(QLabel("From"))
        top.addWidget(self.start)
        top.addWidget(QLabel("to"))
        top.addWidget(self.end)
        top.addWidget(QLabel("s, segments"))
        top.addWidget(self.segments)
        top.addStretch(1)
        self.compute_button = QPushButton("Compute")
        self.compute_button.clicked.connect(self.compute)
        top.addWidget(self.compute_button)

        self.tabs = QTabWidget()
        self.tabs.addTab(self._spectra_tab(), "Spectra")
        self.tabs.addTab(self._reflection_tab(), "Reflection")
        self.tabs.addTab(self._response_tab(), "Response (RAO)")
        self.tabs.currentChanged.connect(lambda _i: self.compute())

        self.message = QLabel()
        self.message.setWordWrap(True)
        copy = QPushButton("Copy table")
        copy.clicked.connect(self.copy_table)
        save = QPushButton("Save figure…")
        save.clicked.connect(self.save_figure)
        close = QPushButton("Close")
        close.clicked.connect(self.close)
        bottom = QHBoxLayout()
        bottom.addWidget(self.message, 1)
        bottom.addWidget(copy)
        bottom.addWidget(save)
        bottom.addWidget(close)

        layout = QVBoxLayout(self)
        layout.addLayout(top)
        layout.addWidget(self.tabs, 1)
        layout.addLayout(bottom)
        if not self.lines:
            self.message.setText("No time series in the scene: add a gauge file first.")
        self.compute()

    def _full_window(self) -> tuple[float, float]:
        starts = [float(t.time[0]) for _k, _l, t in self.lines if t.time.size]
        ends = [float(t.time[-1]) for _k, _l, t in self.lines if t.time.size]
        return (min(starts), max(ends)) if starts else (0.0, 1.0)

    # ------------------------------------------------------------- tabs
    def _figure_label(self) -> QLabel:
        label = QLabel()
        label.setAlignment(Qt.AlignCenter)
        label.setMinimumWidth(380)
        return label

    def _spectra_tab(self) -> QWidget:
        self.line_list = QListWidget()
        gauges = [k for k, _l, t in self.lines if not t.unit or t.unit == "m"]
        default = set(gauges[:4])
        for key, label, _trace in self.lines:
            row = QListWidgetItem(label)
            row.setData(Qt.UserRole, key)
            row.setFlags(row.flags() | Qt.ItemIsUserCheckable)
            row.setCheckState(Qt.Checked if key in default else Qt.Unchecked)
            self.line_list.addItem(row)
        self.line_list.itemChanged.connect(lambda _i: self.compute())
        self.spectra_table = QTableWidget(0, len(PARAMETERS))
        self.spectra_table.setHorizontalHeaderLabels(PARAMETERS)
        self.spectra_table.setToolTip(
            "Hm0 = 4 sqrt(m0), Tp, Tm01 = m0/m1, Tm02 = sqrt(m0/m2) from the spectrum; "
            "H1/3, Hmax, T1/3 and the number of waves from zero down-crossings"
        )
        self.spectra_figure = self._figure_label()
        left = QSplitter(Qt.Vertical)
        left.addWidget(self.line_list)
        left.addWidget(self.spectra_table)
        split = QSplitter(Qt.Horizontal)
        split.addWidget(left)
        split.addWidget(self.spectra_figure)
        split.setSizes([420, 560])
        return split

    def _combo(self, keys: list[str], none: bool = False) -> QComboBox:
        box = QComboBox()
        if none:
            box.addItem("(none)", "")
        for key in keys:
            box.addItem(self.labels[key], key)
        box.currentIndexChanged.connect(lambda _i: self._positions_from_gauges())
        return box

    def _reflection_tab(self) -> QWidget:
        gauges = [
            k for k, _l, t in self.lines if not t.measured_for and (not t.unit or t.unit == "m")
        ]
        self.gauge_boxes = [
            self._combo(gauges),
            self._combo(gauges),
            self._combo(gauges, none=True),
        ]
        for index, box in enumerate(self.gauge_boxes):
            box.setCurrentIndex(min(index + (1 if index == 2 else 0), box.count() - 1))
        self.position_boxes = [_number(decimals=4) for _ in range(3)]
        self.depth = _number((0.001, 1e5), 4, self._still_water())
        self.depth.setToolTip("Water depth at the gauges (default: the case's still water level)")
        form = QFormLayout()
        for k in range(3):
            row = QHBoxLayout()
            row.addWidget(self.gauge_boxes[k], 1)
            row.addWidget(QLabel("x"))
            row.addWidget(self.position_boxes[k])
            form.addRow(f"Gauge {k + 1}", row)
        form.addRow("Depth [m]", self.depth)
        self.reflection_result = QLabel()
        self.reflection_result.setWordWrap(True)
        self.reflection_result.setTextFormat(Qt.RichText)
        left = QWidget()
        column = QVBoxLayout(left)
        column.addLayout(form)
        column.addWidget(self.reflection_result)
        column.addStretch(1)
        self.reflection_figure = self._figure_label()
        split = QSplitter(Qt.Horizontal)
        split.addWidget(left)
        split.addWidget(self.reflection_figure)
        split.setSizes([420, 560])
        self._positions_from_gauges()
        return split

    def _response_tab(self) -> QWidget:
        waves = [k for k, _l, t in self.lines if not t.unit or t.unit == "m"]
        motions = [k for k, _l, t in self.lines if k not in waves or "6dof" in k.lower()]
        self.input_box = self._combo(waves)
        self.output_box = self._combo(motions or [k for k, _l, _t in self.lines])
        self.coherence = _number((0.0, 1.0), 2, 0.5)
        self.coherence.setToolTip("The gain is drawn strong where the coherence reaches this")
        form = QFormLayout()
        form.addRow("Waves (in)", self.input_box)
        form.addRow("Motion (out)", self.output_box)
        form.addRow("Coherence at least", self.coherence)
        self.response_result = QLabel()
        self.response_result.setWordWrap(True)
        self.response_result.setTextFormat(Qt.RichText)
        left = QWidget()
        column = QVBoxLayout(left)
        column.addLayout(form)
        column.addWidget(self.response_result)
        column.addStretch(1)
        self.response_figure = self._figure_label()
        split = QSplitter(Qt.Horizontal)
        split.addWidget(left)
        split.addWidget(self.response_figure)
        split.setSizes([420, 560])
        return split

    # ------------------------------------------------------------- helpers
    def _still_water(self) -> float:
        for item in self.session.scene:
            if item.kind == "series":
                try:
                    level = float(self.session.data.case_water_level(item.id))
                except Exception:  # noqa: BLE001 (no level known: a plain default)
                    continue
                if math.isfinite(level) and level > 0:
                    return level
        return 1.0

    def _positions_from_gauges(self) -> None:
        """The gauges' x from their files (the gauge locations REEF3D writes)."""
        if not hasattr(self, "position_boxes"):
            return
        for box, position in zip(self.gauge_boxes, self.position_boxes, strict=True):
            key = box.currentData()
            if not key:
                continue
            item_id, _, name = key.partition("/")
            table = self.session.data.series(item_id) if item_id in self.session.scene else None
            where = table.locations.get(name) if table is not None and table.locations else None
            if where:
                position.blockSignals(True)
                position.setValue(float(where[0]))
                position.blockSignals(False)

    def _window(self) -> tuple[float, float, int]:
        return self.start.value(), self.end.value(), self.segments.value()

    def _spec(self):
        from lagoonviz.render.figure import FigureSpec

        return FigureSpec(width_mm=120.0, dpi=110)

    # ------------------------------------------------------------- computing
    def compute(self) -> None:
        if not self.lines:
            return
        tab = self.tabs.currentIndex()
        try:
            [self._compute_spectra, self._compute_reflection, self._compute_response][tab]()
            self.message.setText("")
        except ValueError as error:
            self.message.setText(str(error))
        except Exception as error:  # noqa: BLE001 (shown in the window)
            log.exception("wave analysis failed")
            self.message.setText(f"Could not compute: {error}")

    def checked(self) -> list[str]:
        return [
            self.line_list.item(i).data(Qt.UserRole)
            for i in range(self.line_list.count())
            if self.line_list.item(i).checkState() == Qt.Checked
        ]

    def _compute_spectra(self) -> None:
        from lagoonviz.render.wave_figures import spectra_figure

        start, end, segments = self._window()
        lines, rows, styles = [], [], []
        for key in self.checked():
            trace = self.by_key[key]
            f, s = spectrum(trace.time, trace.values, segments, start, end)
            p = wave_parameters(f, s)
            z = zero_crossing(trace.time, trace.values, start, end)
            lines.append((self.labels[key], f, s))
            styles.append("--" if trace.measured_for else "-")
            rows.append(
                [self.labels[key]]
                + [_fmt(v) for v in (p.hm0, p.tp, p.tm01, p.tm02, z.h13, z.hmax, z.t13)]
                + [str(z.n)]
            )
        table = self.spectra_table
        table.setRowCount(len(rows))
        table.setVerticalHeaderLabels([r[0] for r in rows])
        for r, row in enumerate(rows):
            for c, text in enumerate(row[1:]):
                cell = QTableWidgetItem(text)
                cell.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
                table.setItem(r, c, cell)
        table.resizeColumnsToContents()
        self.tables["Spectra"] = [["line", *PARAMETERS], *rows]
        if lines:
            figure = spectra_figure(lines, self._spec(), styles=styles)
            self.figures["Spectra"] = figure
            self.spectra_figure.setPixmap(_pixmap(figure, self.spectra_figure.width()))
        else:
            self.spectra_figure.setText("Tick lines on the left")

    def _compute_reflection(self) -> None:
        from lagoonviz.render.wave_figures import reflection_figure

        start, end, segments = self._window()
        keys = [box.currentData() for box in self.gauge_boxes if box.currentData()]
        if len(set(keys)) < 2:
            raise ValueError("choose two or three different gauges")
        positions = [
            box.value()
            for box, gauge in zip(self.position_boxes, self.gauge_boxes, strict=True)
            if gauge.currentData()
        ]
        traces = [self.by_key[k] for k in keys]
        result = reflection(
            [t.time for t in traces],
            [t.values for t in traces],
            positions,
            self.depth.value(),
            start,
            end,
            segments,
        )
        self.reflection_result.setText(
            f"<b>K<sub>r</sub> = {_fmt(result.kr, 3)}</b><br>"
            f"H<sub>m0</sub> incident {_fmt(result.hm0_incident)} m, reflected "
            f"{_fmt(result.hm0_reflected)} m<br><i>{result.method}</i>"
        )
        self.tables["Reflection"] = [
            ["Kr", "Hm0 incident", "Hm0 reflected", "method"],
            [
                _fmt(result.kr, 4),
                _fmt(result.hm0_incident),
                _fmt(result.hm0_reflected),
                result.method,
            ],
        ]
        figure = reflection_figure(result, self._spec())
        self.figures["Reflection"] = figure
        self.reflection_figure.setPixmap(_pixmap(figure, self.reflection_figure.width()))

    def _compute_response(self) -> None:
        from lagoonviz.render.wave_figures import response_figure

        start, end, segments = self._window()
        x = self.by_key[self.input_box.currentData()]
        y = self.by_key[self.output_box.currentData()]
        result = transfer(x.time, x.values, y.values, y.time, start, end, segments)
        good = (result.coherence >= self.coherence.value()) & (result.frequency > 0)
        unit = f"{y.unit}/{x.unit or 'm'}" if y.unit else ""
        if good.any():
            k = int(np.nanargmax(np.where(good, result.gain, -np.inf)))
            peak = (
                f"largest RAO {_fmt(result.gain[k])} {unit} at f = {_fmt(result.frequency[k])} Hz "
                f"(T = {_fmt(1 / result.frequency[k])} s), phase {_fmt(result.phase[k], 3)}°"
            )
        else:
            peak = "no frequency with the coherence asked for"
        self.response_result.setText(peak)
        rows = [["f [Hz]", "RAO", "phase [deg]", "coherence"]]
        for k in np.flatnonzero(good):
            rows.append(
                [
                    _fmt(result.frequency[k]),
                    _fmt(result.gain[k]),
                    _fmt(result.phase[k], 3),
                    _fmt(result.coherence[k], 3),
                ]
            )
        self.tables["Response (RAO)"] = rows
        figure = response_figure(result, self._spec(), unit, self.coherence.value())
        self.figures["Response (RAO)"] = figure
        self.response_figure.setPixmap(_pixmap(figure, self.response_figure.width()))

    # ------------------------------------------------------------- output
    def _tab_name(self) -> str:
        return self.tabs.tabText(self.tabs.currentIndex())

    def copy_table(self) -> str:
        rows = self.tables.get(self._tab_name(), [])
        text = "\n".join("\t".join(row) for row in rows)
        QGuiApplication.clipboard().setText(text)
        return text

    def save_figure(self, path: str | None = None) -> str | None:
        from lagoonviz.render.figure import save_figure

        figure = self.figures.get(self._tab_name())
        if figure is None:
            return None
        if not path:
            path, _ = QFileDialog.getSaveFileName(
                self, "Save figure", "", "PDF (*.pdf);;SVG (*.svg);;PNG (*.png)"
            )
        if not path:
            return None
        save_figure(figure, path)
        return path
