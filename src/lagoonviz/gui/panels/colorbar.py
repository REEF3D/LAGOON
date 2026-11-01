# SPDX-License-Identifier: GPL-3.0-or-later
"""Colour bars under the viewport: one per visible item that is coloured by a variable
and has its colour bar on, titled, numbered and ticked as the item says
(lagoonviz.core.legend)."""

from __future__ import annotations

from dataclasses import dataclass, field

from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QColor, QFontMetrics, QLinearGradient, QPainter, QPen, QPolygonF
from PySide6.QtWidgets import QSizePolicy, QWidget

from lagoonviz.core.colormaps import lut
from lagoonviz.gui.theme import COLORS

ROW = 22
#: a row with numbers between the ends: they go under the bar
ROW_TICKS = 34


@dataclass
class ColorBarEntry:
    label: str
    colormap: str
    vmin: float
    vmax: float
    #: (value, text) of the numbers along the bar; empty: the ends, as _number writes them
    ticks: list = field(default_factory=list)
    #: the bar's colours, (n, 4) RGBA from low to high (colour bands: in steps);
    #: None: the colour map, smooth
    table: object = None
    #: lines across the bar: the item's contour lines' levels, as fractions of it
    marks: list = field(default_factory=list)
    #: values beyond the range: "min", "max", "both" (the bar ends in a point) or not
    extend: str = "neither"


def _number(value: float) -> str:
    if value == 0:
        return "0"
    magnitude = abs(value)
    if magnitude >= 1e4 or magnitude < 1e-3:
        return f"{value:.2e}"
    return f"{value:.4g}"


class ColorBarStrip(QWidget):
    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self.setObjectName("colorbars")
        self.entries: list[ColorBarEntry] = []
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.setFixedHeight(0)

    def set_entries(self, entries: list[ColorBarEntry]) -> None:
        if entries == self.entries:
            return
        self.entries = entries
        self.setFixedHeight(sum(self._row(e) for e in entries) + (8 if entries else 0))
        self.update()

    @staticmethod
    def _row(entry: ColorBarEntry) -> int:
        return ROW_TICKS if len(entry.ticks) > 2 else ROW

    def paintEvent(self, event) -> None:  # noqa: N802 (Qt naming)
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor(COLORS["panel"]))
        metrics = QFontMetrics(self.font())
        label_width = max((metrics.horizontalAdvance(e.label) for e in self.entries), default=0)
        label_width = min(label_width + 16, self.width() // 3)
        ends = [t for e in self.entries for t in (e.ticks[:1] + e.ticks[-1:])]
        number_width = (
            max(
                [metrics.horizontalAdvance("-0.000e+00")]
                + [metrics.horizontalAdvance(t) for _, t in ends]
            )
            + 12
        )
        top = 4
        for entry in self.entries:
            painter.setPen(QColor(COLORS["text"]))
            painter.drawText(
                QRectF(8, top, label_width - 8, ROW),
                Qt.AlignVCenter | Qt.AlignLeft,
                metrics.elidedText(entry.label, Qt.ElideRight, label_width - 12),
            )
            bar = QRectF(
                label_width + number_width,
                top + 5,
                max(20, self.width() - label_width - 2 * number_width - 16),
                ROW - 10,
            )
            point = bar.height()  # the points at the ends: values beyond the range
            low_point = entry.extend in ("min", "both")
            high_point = entry.extend in ("max", "both")
            bar = bar.adjusted(point if low_point else 0, 0, -point if high_point else 0, 0)
            if entry.table is not None:  # in steps: a colour per stretch of the bar
                table = entry.table
                n = len(table)
                for i, (r, g, b, _a) in enumerate(table):
                    x0 = bar.left() + bar.width() * i / n
                    stretch = QRectF(x0, bar.top(), bar.width() / n + 0.5, bar.height())
                    painter.fillRect(stretch, QColor.fromRgbF(float(r), float(g), float(b)))
            else:
                gradient = QLinearGradient(bar.left(), 0, bar.right(), 0)
                table = lut(entry.colormap, 32)
                for i, (r, g, b, _a) in enumerate(table):
                    gradient.setColorAt(i / (len(table) - 1), QColor.fromRgbF(r, g, b))
                painter.fillRect(bar, gradient)
            painter.setPen(QPen(QColor(255, 255, 255, 90)))
            painter.drawRect(bar)
            ends = entry.table if entry.table is not None else lut(entry.colormap, 32)
            for on, tip, base, colour in (
                (low_point, bar.left() - point, bar.left(), ends[0]),
                (high_point, bar.right() + point, bar.right(), ends[-1]),
            ):
                if on:
                    r, g, b = (float(c) for c in colour[:3])
                    painter.setBrush(QColor.fromRgbF(r, g, b))
                    painter.drawPolygon(
                        QPolygonF(
                            [
                                QPointF(tip, bar.center().y()),
                                QPointF(base, bar.top()),
                                QPointF(base, bar.bottom()),
                            ]
                        )
                    )
            painter.setBrush(Qt.NoBrush)
            painter.setPen(QPen(QColor(28, 35, 48, 230), 1.0))
            for where in entry.marks:  # the item's contour lines: their levels
                x = bar.left() + where * bar.width()
                painter.drawLine(QPointF(x, bar.top()), QPointF(x, bar.bottom()))
            painter.setPen(QColor(COLORS["text"]))
            low = entry.ticks[0][1] if entry.ticks else _number(entry.vmin)
            high = entry.ticks[-1][1] if entry.ticks else _number(entry.vmax)
            painter.drawText(
                QRectF(label_width, top, number_width - 6, ROW),
                Qt.AlignVCenter | Qt.AlignRight,
                low,
            )
            painter.drawText(
                QRectF(bar.right() + 6, top, number_width, ROW),
                Qt.AlignVCenter | Qt.AlignLeft,
                high,
            )
            span = entry.vmax - entry.vmin
            for value, text in entry.ticks[1:-1]:  # the numbers between, under the bar
                x = bar.left() + (0.0 if span == 0 else (value - entry.vmin) / span) * bar.width()
                painter.drawLine(QPointF(x, bar.bottom()), QPointF(x, bar.bottom() + 3))
                painter.drawText(
                    QRectF(x - 60, bar.bottom() + 2, 120, ROW_TICKS - ROW + 4),
                    Qt.AlignHCenter | Qt.AlignTop,
                    text,
                )
            top += self._row(entry)
        painter.end()
