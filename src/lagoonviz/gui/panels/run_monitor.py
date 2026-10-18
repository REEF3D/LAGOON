# SPDX-License-Identifier: GPL-3.0-or-later
# Architect: Hans Bihs
"""The Run monitor: how the latest run of a case is going, read every few seconds
from its run log and REEF3D_Log (lagoonviz.core.run_monitor).

Iteration, simulation time and time step; the speed (simulation seconds per wall
hour, wall seconds per iteration), the wall time so far and to the end (N 41 or
N 45 of the control file the run started with) and when that is; the outputs
written; the largest velocity. Small plots of the time step and of the largest
velocity over the last iterations. Warnings: a time step that collapses, a
velocity that runs away, NaN, a log that has gone quiet (killed, crashed, hanging),
ctrl.txt changed since the run started (the changes listed). *Follow* reloads the
case when the run writes new output (File → Follow running simulation).
"""

from __future__ import annotations

import html
import math

import numpy as np
from PySide6.QtCore import QPointF, QRectF, QSize, Qt, QTimer
from PySide6.QtGui import QColor, QFont, QPainter, QPainterPath, QPen
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QPlainTextEdit,
    QProgressBar,
    QSizePolicy,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from lagoonviz.core.run_monitor import RunStatus, duration, finish_clock, run_status
from lagoonviz.gui.bridge import SessionBridge
from lagoonviz.gui.theme import COLORS

#: how often the panel looks again while it is shown [ms]
POLL_MS = 3000
STATE_COLORS = {
    "running": "#2e8b57",
    "finished": "#2b6fb6",
    "stopped": "#b8860b",
    "not responding": "#c0392b",
    "no run": "#6a7684",
}


class Sparkline(QWidget):
    """A small line plot of one quantity over the last iterations, its latest value
    in the corner."""

    def __init__(self, title: str, unit: str, parent: QWidget | None = None):
        super().__init__(parent)
        self.title, self.unit = title, unit
        self.x = np.zeros(0)
        self.y = np.zeros(0)
        self.setMinimumHeight(54)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

    def sizeHint(self) -> QSize:  # noqa: N802 (Qt naming)
        return QSize(220, 58)

    def set_data(self, x: np.ndarray | None, y: np.ndarray | None) -> None:
        self.x = np.asarray(x if x is not None else [], np.float64)
        self.y = np.asarray(y if y is not None else [], np.float64)
        self.update()

    def paintEvent(self, _event) -> None:  # noqa: N802 (Qt naming)
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        rect = QRectF(self.rect()).adjusted(1, 14, -1, -2)
        painter.setPen(QPen(QColor(COLORS["border"])))
        painter.drawRect(rect)
        font = QFont(self.font())
        font.setPointSizeF(max(7.0, font.pointSizeF() - 1.5))
        painter.setFont(font)
        painter.setPen(QColor(COLORS["dim"]))
        keep = np.isfinite(self.x) & np.isfinite(self.y)
        x, y = self.x[keep], self.y[keep]
        latest = f"{y[-1]:.4g} {self.unit}" if y.size else "–"
        painter.drawText(QRectF(2, 0, self.width() - 4, 13), Qt.AlignLeft, self.title)
        painter.drawText(QRectF(2, 0, self.width() - 4, 13), Qt.AlignRight, latest)
        if x.size < 2 or np.ptp(x) <= 0:
            return
        low, high = float(y.min()), float(y.max())
        if high - low <= 1e-15 * max(1.0, abs(high)):
            low, high = low - 0.5 * abs(low or 1.0), high + 0.5 * abs(high or 1.0)
        span_x, span_y = float(np.ptp(x)), high - low
        path = QPainterPath()
        for k, (xi, yi) in enumerate(zip(x, y, strict=True)):
            point = QPointF(
                rect.left() + (xi - x[0]) / span_x * rect.width(),
                rect.bottom() - (yi - low) / span_y * rect.height(),
            )
            path.moveTo(point) if k == 0 else path.lineTo(point)
        painter.setPen(QPen(QColor(COLORS["accent"]), 1.4))
        painter.drawPath(path)


class RunMonitorPanel(QWidget):
    def __init__(self, bridge: SessionBridge | None = None, parent: QWidget | None = None):
        super().__init__(parent)
        self.setObjectName("run_monitor")
        self.session = bridge.session if bridge else None
        #: set by the main window: switches File → Follow running simulation
        self.follow = None
        self.status: RunStatus | None = None

        self.case_box = QComboBox()
        self.case_box.setToolTip("The case whose latest run is shown")
        self.case_box.activated.connect(lambda _i: self.poll())
        self.state = QLabel()
        self.state.setTextFormat(Qt.RichText)
        self.refresh_button = QToolButton()
        self.refresh_button.setText("Look again")
        self.refresh_button.clicked.connect(self.poll)
        self.follow_box = QCheckBox("Follow")
        self.follow_box.setToolTip(
            "Load new outputs as the run writes them and stay at the last one "
            "(File → Follow running simulation)"
        )
        self.follow_box.toggled.connect(self._follow)
        top = QHBoxLayout()
        top.addWidget(self.case_box, 1)
        top.addWidget(self.state)
        top.addStretch(1)
        top.addWidget(self.follow_box)
        top.addWidget(self.refresh_button)

        self.progress = QProgressBar()
        self.progress.setRange(0, 1000)
        self.progress.setTextVisible(True)

        self.values: dict[str, QLabel] = {}
        grid = QGridLayout()
        names = [
            ("run", "Run"),
            ("iteration", "Iteration"),
            ("time", "Time"),
            ("dt", "Time step"),
            ("speed", "Speed"),
            ("elapsed", "Running for"),
            ("eta", "To the end"),
            ("outputs", "Outputs"),
            ("velocity", "Largest velocity"),
        ]
        for index, (key, label) in enumerate(names):
            row, column = divmod(index, 3)
            name = QLabel(label)
            name.setStyleSheet(f"color: {COLORS['dim']};")
            value = QLabel("–")
            value.setTextInteractionFlags(Qt.TextSelectableByMouse)
            grid.addWidget(name, row, 2 * column)
            grid.addWidget(value, row, 2 * column + 1)
            self.values[key] = value
        for column in (1, 3, 5):
            grid.setColumnStretch(column, 1)

        self.dt_plot = Sparkline("time step", "s")
        self.velocity_plot = Sparkline("largest velocity", "m/s")
        plots = QHBoxLayout()
        plots.addWidget(self.dt_plot)
        plots.addWidget(self.velocity_plot)

        self.warnings = QLabel()
        self.warnings.setWordWrap(True)
        self.warnings.setTextFormat(Qt.RichText)
        self.changes = QPlainTextEdit()
        self.changes.setReadOnly(True)
        self.changes.setMaximumHeight(90)
        self.changes.setToolTip(
            "ctrl.txt now against the copy the run started with (REEF3D_Case): "
            "- as the run has it, + as ctrl.txt has it now"
        )

        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 4, 6, 4)
        layout.addLayout(top)
        layout.addWidget(self.progress)
        layout.addLayout(grid)
        layout.addLayout(plots)
        layout.addWidget(self.warnings)
        layout.addWidget(self.changes)
        layout.addStretch(1)

        self.timer = QTimer(self)
        self.timer.setInterval(POLL_MS)
        self.timer.timeout.connect(self._tick)
        self.timer.start()
        if bridge is not None:
            bridge.sceneChanged.connect(self._on_scene)
        self._fill_cases()
        self.poll()

    def sizeHint(self) -> QSize:  # noqa: N802 (Qt naming)
        return QSize(600, 220)

    # ------------------------------------------------------------- cases
    def _on_scene(self, event: str, _item_id: str) -> None:
        if event in ("added", "removed", "reset", "changed"):
            self._fill_cases()

    def _cases(self) -> list[tuple[str, str]]:
        """(name, case folder) of the open cases."""
        found = []
        if self.session is None:
            return found
        for item in self.session.scene:
            if item.kind != "case":
                continue
            case = self.session.data.case(item.id)
            if case is not None:
                found.append((item.name, str(case.path)))
        return found

    def _fill_cases(self) -> None:
        cases = self._cases()
        current = self.case_box.currentData()
        box = self.case_box
        if [(box.itemText(i), box.itemData(i)) for i in range(box.count())] != cases:
            box.blockSignals(True)
            box.clear()
            for name, folder in cases:
                box.addItem(name, folder)
            index = box.findData(current)
            box.setCurrentIndex(index if index >= 0 else 0)
            box.blockSignals(False)
        box.setVisible(len(cases) > 1)

    def _solver(self, folder: str) -> str:
        if self.session is None:
            return ""
        for item in self.session.scene:
            if item.kind == "case":
                case = self.session.data.case(item.id)
                if case is not None and str(case.path) == folder:
                    return case.ctrl_solver
        return ""

    # ------------------------------------------------------------- polling
    def _tick(self) -> None:
        if self.isVisible():
            self.poll()

    def poll(self) -> None:
        folder = self.case_box.currentData()
        if not folder:
            self.status = None
            self._show(None)
            return
        try:
            self.status = run_status(folder, self._solver(folder))
        except Exception as error:  # noqa: BLE001 (a log being written must not stop it)
            self.state.setText(html.escape(f"could not read the logs: {error}"))
            return
        self._show(self.status)

    def _follow(self, on: bool) -> None:
        if self.follow is not None:
            self.follow(on)

    def set_following(self, on: bool) -> None:
        """The main window's Follow switched elsewhere: the box shows it."""
        self.follow_box.blockSignals(True)
        self.follow_box.setChecked(bool(on))
        self.follow_box.blockSignals(False)

    # ------------------------------------------------------------- showing
    def _show(self, status: RunStatus | None) -> None:
        values = self.values
        if status is None or status.state == "no run":
            self.state.setText(
                f'<span style="color:{STATE_COLORS["no run"]}">no run of this case yet</span>'
            )
            for label in values.values():
                label.setText("–")
            self.progress.setValue(0)
            self.progress.setFormat("")
            self.dt_plot.set_data(None, None)
            self.velocity_plot.set_data(None, None)
            self.warnings.setText("")
            self.changes.setVisible(False)
            return
        color = STATE_COLORS.get(status.state, COLORS["dim"])
        ended = f" ({status.ended})" if status.state == "stopped" and status.ended else ""
        self.state.setText(f'<b style="color:{color}">{status.state}{ended}</b>')
        started = status.started.astimezone().strftime("%d %b %H:%M") if status.started else ""
        values["run"].setText(
            f"{status.solver} {status.run_id[-4:]}" + (f", {started}" if started else "")
        )
        values["run"].setToolTip(status.run_id)
        values["iteration"].setText(
            _number(status.iteration)
            + (f" of {status.max_iterations:,}" if _limit(status.max_iterations) else "")
        )
        values["time"].setText(
            (f"{status.time:.4g} s" if status.time is not None else "–")
            + (f" of {status.end_time:g} s" if _limit(status.end_time) else "")
        )
        values["dt"].setText(f"{status.dt:.4g} s" if status.dt is not None else "–")
        if status.simulated_per_hour is not None:
            values["speed"].setText(
                f"{status.simulated_per_hour:.3g} s/h · {status.seconds_per_iteration:.3g} s/it"
            )
            values["speed"].setToolTip(
                "simulated seconds per wall hour · wall seconds per iteration "
                "(the last 50 iterations)"
            )
        else:
            values["speed"].setText("–")
        values["elapsed"].setText(duration(status.elapsed) if status.elapsed else "–")
        values["eta"].setText(
            f"{duration(status.eta)}, at {finish_clock(status.eta)}"
            if status.eta is not None
            else "–"
        )
        values["outputs"].setText(
            f"{status.outputs}"
            + (
                f", last at t = {status.last_output_time:.4g} s"
                if status.last_output_time is not None
                else ""
            )
        )
        values["velocity"].setText(
            f"{status.max_velocity:.4g} m/s" if status.max_velocity is not None else "–"
        )
        progress = status.progress
        self.progress.setValue(int(round(1000 * progress)) if progress is not None else 0)
        self.progress.setFormat(f"{100 * progress:.1f} %" if progress is not None else "no end set")
        history = status.history
        self.dt_plot.set_data(history.get("iteration"), history.get("dt"))
        self.velocity_plot.set_data(history.get("velocity_iteration"), history.get("velocity"))
        self.warnings.setText(
            "<br>".join(
                f'<span style="color:#c0392b">⚠ {html.escape(w)}</span>' for w in status.warnings
            )
        )
        self.changes.setVisible(bool(status.ctrl_changes))
        self.changes.setPlainText("\n".join(status.ctrl_changes))


def _number(value) -> str:
    return f"{value:,}" if value is not None else "–"


def _limit(value) -> bool:
    return value is not None and math.isfinite(value) and 0 < value < 1e8
