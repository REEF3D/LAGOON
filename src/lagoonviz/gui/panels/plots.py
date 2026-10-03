# SPDX-License-Identifier: GPL-3.0-or-later
"""Linked time-series plots: gauges, forces and motions under the 3D view.

One row per visible "Time series" item, all sharing the time axis. The orange
cursor is the scene time: it follows the timeline, and clicking or dragging in a
plot moves the timeline (and with it the 3D view). The legend shows each line's
value at the cursor.

Mouse: click/drag to set the time, wheel to zoom the time axis, right or middle
drag to pan, double-click to show everything. Drawn with QPainter, which is fast
enough for live updates; publication figures will be redrawn with matplotlib.
"""

from __future__ import annotations

import math

import numpy as np
from PySide6.QtCore import QPointF, QRectF, QSize, Qt
from PySide6.QtGui import QColor, QFont, QFontMetrics, QPainter, QPainterPath, QPen
from PySide6.QtWidgets import QSizePolicy, QWidget

from lagoonviz.core.data.manager import Trace
from lagoonviz.core.data.series import decimate
from lagoonviz.core.plotting import nice_step, nice_ticks, tick_label, value_range
from lagoonviz.gui.bridge import SessionBridge
from lagoonviz.gui.theme import COLORS, SERIES_COLORS

LEFT = 64  # room for y tick labels
RIGHT = 14
TOP = 22  # room for the row title and legend
BOTTOM = 24  # room for the time axis labels under the last row
ROW_GAP = 8


class PlotsPanel(QWidget):
    def __init__(self, bridge: SessionBridge | None = None, parent: QWidget | None = None):
        super().__init__(parent)
        self.setObjectName("plots")
        self.setMinimumHeight(90)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setMouseTracking(True)
        self.bridge = bridge
        self.session = bridge.session if bridge else None
        self.view: tuple[float, float] | None = None  # time window; None: everything
        self._rows: list[tuple[str, list[Trace]]] = []
        self._paths: dict[tuple, QPainterPath] = {}
        self._drag: tuple[str, float, tuple[float, float]] | None = None
        self.hover_time: float | None = None
        if bridge is not None:
            bridge.timeChanged.connect(lambda _t: self.update())
            bridge.sceneChanged.connect(lambda *_: self.refresh())
        self.refresh()

    def sizeHint(self) -> QSize:  # noqa: N802 (Qt naming)
        return QSize(600, 180)

    # ------------------------------------------------------------- data
    def refresh(self) -> None:
        """Reload the rows from the scene (items, columns, files)."""
        self._rows = []
        self._paths.clear()
        if self.session is not None:
            data = self.session.data
            for item_id in data.series_items():
                item = self.session.scene.get(item_id)
                if not item.visible:
                    continue
                case_item = data.case_item(item_id)
                if case_item is not None and not case_item.visible:
                    continue
                self._rows.append((item_id, data.traces(item_id)))
        self.update()

    @property
    def row_ids(self) -> list[str]:
        return [item_id for item_id, _ in self._rows]

    def full_range(self) -> tuple[float, float]:
        starts, ends = [], []
        for _, traces in self._rows:
            for trace in traces:
                if trace.time.size:
                    starts.append(float(trace.time[0]))
                    ends.append(float(trace.time[-1]))
        if not starts:
            return 0.0, 1.0
        low, high = min(starts), max(ends)
        return (low, high) if high > low else (low, low + 1.0)

    def time_window(self) -> tuple[float, float]:
        return self.view if self.view is not None else self.full_range()

    @property
    def linked(self) -> bool:
        """True if the plot time is the scene time (seconds known)."""
        return self.session is not None and self.session.data.time_in_seconds()

    # ------------------------------------------------------------- geometry
    def _plot_rects(self) -> list[QRectF]:
        n = len(self._rows)
        if not n:
            return []
        usable = self.height() - BOTTOM - (n - 1) * ROW_GAP
        height = max(usable / n, 20.0)
        rects = []
        for i in range(n):
            top = i * (height + ROW_GAP) + TOP
            rects.append(QRectF(LEFT, top, max(self.width() - LEFT - RIGHT, 10), height - TOP))
        return rects

    def time_at(self, x: float) -> float:
        low, high = self.time_window()
        width = max(self.width() - LEFT - RIGHT, 1)
        return low + (x - LEFT) / width * (high - low)

    def x_of(self, time: float) -> float:
        low, high = self.time_window()
        width = max(self.width() - LEFT - RIGHT, 1)
        return LEFT + (time - low) / (high - low) * width

    # ------------------------------------------------------------- painting
    def paintEvent(self, event) -> None:  # noqa: N802 (Qt naming)
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.fillRect(self.rect(), QColor(COLORS["base"]))
        font = QFont(self.font())
        font.setPointSizeF(max(font.pointSizeF() - 1.5, 8.0))
        painter.setFont(font)
        metrics = QFontMetrics(font)
        if not self._rows:
            painter.setPen(QColor(COLORS["dim"]))
            painter.drawText(
                self.rect(),
                Qt.AlignCenter | Qt.TextWordWrap,
                "No time series. Open a case with gauge or 6DOF output, "
                "or add one with Add → Time series.",
            )
            painter.end()
            return
        low, high = self.time_window()
        time_ticks = nice_ticks(low, high, max(2, int(self.width() / 110)))
        time_step = nice_step(high - low, max(2, int(self.width() / 110)))
        cursor = self.session.scene.time if self.linked else None
        rects = self._plot_rects()
        for (item_id, traces), rect in zip(self._rows, rects, strict=True):
            self._paint_row(painter, metrics, item_id, traces, rect, time_ticks, cursor)
        # time labels under the last row
        last = rects[-1]
        painter.setPen(QColor(COLORS["dim"]))
        for tick in time_ticks:
            x = self.x_of(tick)
            painter.drawText(
                QRectF(x - 40, last.bottom() + 3, 80, BOTTOM - 4),
                Qt.AlignHCenter | Qt.AlignTop,
                tick_label(tick, time_step),
            )
        painter.drawText(
            QRectF(self.width() - RIGHT - 60, last.bottom() + 3, 60, BOTTOM - 4),
            Qt.AlignRight | Qt.AlignTop,
            "t [s]",
        )
        if not self.linked and self.session is not None and self.session.data.n_frames():
            painter.drawText(
                QRectF(LEFT, 2, self.width() - LEFT - RIGHT, TOP - 4),
                Qt.AlignRight | Qt.AlignVCenter,
                "set the case's output interval to link plots and 3D view",
            )
        painter.end()

    def _paint_row(self, painter, metrics, item_id, traces, rect, time_ticks, cursor) -> None:
        grid = QColor(COLORS["border"])
        painter.fillRect(rect, QColor(COLORS["panel"]))
        low_v, high_v = value_range(t.values for t in traces)
        target = max(2, int(rect.height() / (1.8 * metrics.height())))
        value_ticks = nice_ticks(low_v, high_v, target)
        value_step = nice_step(high_v - low_v, target)

        def y_of(value: float) -> float:
            return rect.bottom() - (value - low_v) / (high_v - low_v) * rect.height()

        painter.setPen(QPen(grid, 1))
        for tick in time_ticks:
            x = self.x_of(tick)
            painter.drawLine(QPointF(x, rect.top()), QPointF(x, rect.bottom()))
        last_label = math.inf
        for tick in value_ticks:
            y = y_of(tick)
            painter.setPen(QPen(grid, 1))
            painter.drawLine(QPointF(rect.left(), y), QPointF(rect.right(), y))
            if abs(last_label - y) < metrics.height():
                continue  # no overlapping labels on low rows
            last_label = y
            painter.setPen(QColor(COLORS["dim"]))
            painter.drawText(
                QRectF(0, y - 8, LEFT - 6, 16),
                Qt.AlignRight | Qt.AlignVCenter,
                tick_label(tick, value_step),
            )

        painter.save()
        painter.setClipRect(rect.adjusted(0, -1, 0, 1))
        low_t, high_t = self.time_window()
        for index, trace in enumerate(traces):
            key = (item_id, trace.name, rect.width(), rect.height(), low_t, high_t, low_v, high_v)
            path = self._paths.get(key)
            if path is None:
                path = self._path(trace, rect, low_t, high_t, y_of)
                self._paths[key] = path
            color = QColor(SERIES_COLORS[index % len(SERIES_COLORS)])
            painter.setPen(QPen(color, 1.4))
            painter.drawPath(path)
        painter.restore()

        # title and legend with values at the cursor
        item = self.session.scene.get(item_id)
        x = rect.left() + 2
        painter.setPen(QColor(COLORS["text"]))
        title = item.name
        painter.drawText(QRectF(x, rect.top() - TOP + 2, 400, TOP - 4), Qt.AlignVCenter, title)
        x += metrics.horizontalAdvance(title) + 14
        for index, trace in enumerate(traces):
            color = QColor(SERIES_COLORS[index % len(SERIES_COLORS)])
            text = trace.name + (f" [{trace.unit}]" if trace.unit else "")
            if cursor is not None:
                value = _value_at(trace, cursor)
                text += f" = {_format_value(value)}"
            painter.fillRect(QRectF(x, rect.top() - TOP / 2 - 1, 10, 3), color)
            painter.setPen(QColor(COLORS["text"]))
            painter.drawText(
                QRectF(x + 14, rect.top() - TOP + 2, 400, TOP - 4), Qt.AlignVCenter, text
            )
            x += 14 + metrics.horizontalAdvance(text) + 14
        if not traces:
            painter.setPen(QColor(COLORS["dim"]))
            painter.drawText(rect, Qt.AlignCenter, "no data: check the item's file and columns")

        if cursor is not None and low_t <= cursor <= high_t:
            x = self.x_of(cursor)
            painter.setPen(QPen(QColor(COLORS["cursor"]), 1.6))
            painter.drawLine(QPointF(x, rect.top()), QPointF(x, rect.bottom()))

    def _path(self, trace: Trace, rect: QRectF, low_t, high_t, y_of) -> QPainterPath:
        time, values = trace.time, trace.values
        # only the visible window (plus one sample each side), decimated to pixels
        start = max(int(np.searchsorted(time, low_t)) - 1, 0)
        end = min(int(np.searchsorted(time, high_t)) + 1, time.size)
        time, values = decimate(time[start:end], values[start:end], int(rect.width()))
        xs = rect.left() + (time - low_t) / (high_t - low_t) * rect.width()
        # y_of is affine, so apply it to the whole array at once
        ys = y_of(0.0) - values * (y_of(0.0) - y_of(1.0))
        path = QPainterPath()
        drawing = False
        for x, y in zip(xs.tolist(), ys.tolist(), strict=True):
            if not math.isfinite(y):
                drawing = False
                continue
            if drawing:
                path.lineTo(x, y)
            else:
                path.moveTo(x, y)
                drawing = True
        return path

    # ------------------------------------------------------------- mouse
    def mousePressEvent(self, event) -> None:  # noqa: N802 (Qt naming)
        if event.button() == Qt.LeftButton and self._rows:
            self._drag = ("time", event.position().x(), self.time_window())
            self._set_time_from(event.position().x())
        elif event.button() in (Qt.RightButton, Qt.MiddleButton):
            self._drag = ("pan", event.position().x(), self.time_window())

    def mouseMoveEvent(self, event) -> None:  # noqa: N802 (Qt naming)
        x = event.position().x()
        self.hover_time = self.time_at(x) if self._rows else None
        if self._drag is None:
            return
        mode, start_x, (low, high) = self._drag
        if mode == "time":
            self._set_time_from(x)
        else:
            width = max(self.width() - LEFT - RIGHT, 1)
            shift = (start_x - x) / width * (high - low)
            self.view = (low + shift, high + shift)
            self.update()

    def mouseReleaseEvent(self, event) -> None:  # noqa: N802 (Qt naming)
        self._drag = None

    def mouseDoubleClickEvent(self, event) -> None:  # noqa: N802 (Qt naming)
        self.view = None
        self.update()

    def wheelEvent(self, event) -> None:  # noqa: N802 (Qt naming)
        if not self._rows:
            return
        steps = event.angleDelta().y() / 120.0
        if steps == 0:
            return
        low, high = self.time_window()
        anchor = self.time_at(event.position().x())
        factor = 0.85**steps
        self.view = (anchor - (anchor - low) * factor, anchor + (high - anchor) * factor)
        self.update()

    def _set_time_from(self, x: float) -> None:
        if not self.linked:
            return
        low, high = self.full_range()
        time = min(max(self.time_at(x), low), high)
        self.session.set_time(time)


def _value_at(trace: Trace, time: float) -> float:
    if trace.time.size == 0 or time < trace.time[0] or time > trace.time[-1]:
        return float("nan")
    return float(np.interp(time, trace.time, trace.values))


def _format_value(value: float) -> str:
    if not math.isfinite(value):
        return "–"
    magnitude = abs(value)
    if magnitude != 0 and (magnitude >= 1e5 or magnitude < 1e-3):
        return f"{value:.3e}"
    return f"{value:.4g}"
