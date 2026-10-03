# SPDX-License-Identifier: GPL-3.0-or-later
"""Wave profiles along lines (WSFLINE) at the current time, under the 3D view.

One row per profile item: the computed elevation along x for each of its lines,
with the wave theory dashed where FNPF wrote it. The value axis is fixed from a
spread of times, so the waves move through a steady frame while playing. Lines
have the colours they have in the 3D view.
"""

from __future__ import annotations

import math

import numpy as np
from PySide6.QtCore import QPointF, QRectF, QSize, Qt
from PySide6.QtGui import QColor, QFont, QFontMetrics, QPainter, QPainterPath, QPen
from PySide6.QtWidgets import QSizePolicy, QWidget

from lagoonviz.core.plotting import nice_step, nice_ticks, tick_label
from lagoonviz.gui.bridge import SessionBridge
from lagoonviz.gui.perf import PERF
from lagoonviz.gui.theme import COLORS, SERIES_COLORS

LEFT = 64
RIGHT = 14
TOP = 22
BOTTOM = 24
ROW_GAP = 8


class ProfilesPanel(QWidget):
    def __init__(self, bridge: SessionBridge | None = None, parent: QWidget | None = None):
        super().__init__(parent)
        self.setObjectName("profiles")
        self.setMinimumHeight(90)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.session = bridge.session if bridge else None
        self._rows: list[str] = []
        self._ranges: dict[tuple, tuple[float, float]] = {}
        if bridge is not None:
            bridge.timeChanged.connect(lambda _t: self.update())
            bridge.sceneChanged.connect(lambda *_: self.refresh())
        self.refresh()

    def sizeHint(self) -> QSize:  # noqa: N802 (Qt naming)
        return QSize(600, 180)

    # ------------------------------------------------------------- data
    def refresh(self) -> None:
        self._rows = []
        if self.session is not None:
            data = self.session.data
            for item_id in data.profile_items():
                item = self.session.scene.get(item_id)
                case_item = data.case_item(item_id)
                if item.visible and (case_item is None or case_item.visible):
                    self._rows.append(item_id)
        self.update()

    @property
    def row_ids(self) -> list[str]:
        return list(self._rows)

    def _value_range(self, item_id: str, numbers: list[int]) -> tuple[float, float]:
        """Elevation range of the shown lines over the run (kept once found)."""
        data = self.session.data
        found = data.profile_set(item_id)
        theory = bool(self.session.scene.get(item_id).properties.get("theory", True))
        key = (item_id, found.key if found else "", tuple(numbers), theory)
        if key not in self._ranges:
            lows, highs = [], []
            for number in numbers:
                low, high = found.value_range(number) if found else (0.0, 1.0)
                lows.append(low)
                highs.append(high)
            low, high = (min(lows), max(highs)) if lows else (0.0, 1.0)
            pad = 0.06 * (high - low)
            self._ranges[key] = (low - pad, high + pad)
        return self._ranges[key]

    # ------------------------------------------------------------- painting
    def paintEvent(self, event) -> None:  # noqa: N802 (Qt naming)
        with PERF.track("profiles"):
            self._paint()

    def _paint(self) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.fillRect(self.rect(), QColor(COLORS["base"]))
        font = QFont(self.font())
        font.setPointSizeF(max(font.pointSizeF() - 1.5, 8.0))
        painter.setFont(font)
        metrics = QFontMetrics(font)
        data = self.session.data if self.session is not None else None
        rows = []
        if data is not None:
            for item_id in self._rows:
                lines = data.profile_lines(item_id)
                if lines:
                    rows.append((item_id, lines))
        if not rows:
            painter.setPen(QColor(COLORS["dim"]))
            painter.drawText(
                self.rect(),
                Qt.AlignCenter | Qt.TextWordWrap,
                "No wave profiles. Open a case with WSFLINE output, "
                "or add one with Add → Wave profile.",
            )
            painter.end()
            return
        lows = [float(np.nanmin(line.x)) for _, lines in rows for line in lines if line.x.size]
        highs = [float(np.nanmax(line.x)) for _, lines in rows for line in lines if line.x.size]
        x_low, x_high = (min(lows), max(highs)) if lows else (0.0, 1.0)
        if x_high <= x_low:
            x_high = x_low + 1.0
        width = max(self.width() - LEFT - RIGHT, 10)

        def x_of(x: float) -> float:
            return LEFT + (x - x_low) / (x_high - x_low) * width

        target = max(2, int(self.width() / 110))
        x_ticks = nice_ticks(x_low, x_high, target)
        x_step = nice_step(x_high - x_low, target)
        n = len(rows)
        height = max((self.height() - BOTTOM - (n - 1) * ROW_GAP) / n, 20.0)
        rect = None
        for i, (item_id, lines) in enumerate(rows):
            top = i * (height + ROW_GAP) + TOP
            rect = QRectF(LEFT, top, width, height - TOP)
            self._paint_row(painter, metrics, item_id, lines, rect, x_of, x_ticks)
        painter.setPen(QColor(COLORS["dim"]))
        for tick in x_ticks:
            painter.drawText(
                QRectF(x_of(tick) - 40, rect.bottom() + 3, 80, BOTTOM - 4),
                Qt.AlignHCenter | Qt.AlignTop,
                tick_label(tick, x_step),
            )
        painter.drawText(
            QRectF(self.width() - RIGHT - 60, rect.bottom() + 3, 60, BOTTOM - 4),
            Qt.AlignRight | Qt.AlignTop,
            "x [m]",
        )
        painter.end()

    def _paint_row(self, painter, metrics, item_id, lines, rect, x_of, x_ticks) -> None:
        data = self.session.data
        item = self.session.scene.get(item_id)
        theory = bool(item.properties.get("theory", True))
        painter.fillRect(rect, QColor(COLORS["panel"]))
        low, high = self._value_range(item_id, [line.number for line in lines])
        target = max(2, int(rect.height() / (1.8 * metrics.height())))
        ticks = nice_ticks(low, high, target)
        step = nice_step(high - low, target)

        def y_of(value: float) -> float:
            return rect.bottom() - (value - low) / (high - low) * rect.height()

        grid = QColor(COLORS["border"])
        painter.setPen(QPen(grid, 1))
        for tick in x_ticks:
            painter.drawLine(QPointF(x_of(tick), rect.top()), QPointF(x_of(tick), rect.bottom()))
        last = math.inf
        for tick in ticks:
            y = y_of(tick)
            painter.setPen(QPen(grid, 1))
            painter.drawLine(QPointF(rect.left(), y), QPointF(rect.right(), y))
            if abs(last - y) < metrics.height():
                continue
            last = y
            painter.setPen(QColor(COLORS["dim"]))
            painter.drawText(
                QRectF(0, y - 8, LEFT - 6, 16),
                Qt.AlignRight | Qt.AlignVCenter,
                tick_label(tick, step),
            )

        painter.save()
        painter.setClipRect(rect.adjusted(0, -1, 0, 1))
        for index, line in enumerate(lines):
            color = QColor(SERIES_COLORS[index % len(SERIES_COLORS)])
            painter.setPen(QPen(color, 1.6))
            painter.drawPath(_path(line.x, line.eta, x_of, y_of))
            if theory and line.theory is not None:
                pen = QPen(color, 1.2, Qt.DashLine)
                painter.setPen(pen)
                painter.drawPath(_path(line.x, line.theory, x_of, y_of))
        painter.restore()

        # title and legend
        profile = data.profile(item_id)
        x = rect.left() + 2
        title = item.name + (f" · t = {profile.time:.2f} s" if profile is not None else "")
        painter.setPen(QColor(COLORS["text"]))
        painter.drawText(QRectF(x, rect.top() - TOP + 2, 400, TOP - 4), Qt.AlignVCenter, title)
        x += metrics.horizontalAdvance(title) + 14
        for index, line in enumerate(lines):
            color = QColor(SERIES_COLORS[index % len(SERIES_COLORS)])
            text = f"line {line.number} (y = {line.y:g})"
            painter.fillRect(QRectF(x, rect.top() - TOP / 2 - 1, 10, 3), color)
            painter.setPen(QColor(COLORS["text"]))
            painter.drawText(
                QRectF(x + 14, rect.top() - TOP + 2, 400, TOP - 4), Qt.AlignVCenter, text
            )
            x += 14 + metrics.horizontalAdvance(text) + 14
        if theory and any(line.theory is not None for line in lines):
            painter.setPen(QColor(COLORS["dim"]))
            painter.drawText(
                QRectF(x, rect.top() - TOP + 2, 200, TOP - 4),
                Qt.AlignVCenter,
                "dashed: wave theory",
            )


def _path(x: np.ndarray, values: np.ndarray, x_of, y_of) -> QPainterPath:
    path = QPainterPath()
    drawing = False
    for xi, vi in zip(x, values, strict=False):
        if not (math.isfinite(xi) and math.isfinite(vi)):
            drawing = False
            continue
        point = QPointF(x_of(float(xi)), y_of(float(vi)))
        if drawing:
            path.lineTo(point)
        else:
            path.moveTo(point)
            drawing = True
    return path
