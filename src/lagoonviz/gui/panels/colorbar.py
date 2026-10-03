# SPDX-License-Identifier: GPL-3.0-or-later
"""Colour bars under the viewport: one per visible item that is coloured by a variable."""

from __future__ import annotations

from dataclasses import dataclass

from PySide6.QtCore import QRectF, Qt
from PySide6.QtGui import QColor, QFontMetrics, QLinearGradient, QPainter, QPen
from PySide6.QtWidgets import QSizePolicy, QWidget

from lagoonviz.core.colormaps import lut
from lagoonviz.gui.theme import COLORS

ROW = 22


@dataclass
class ColorBarEntry:
    label: str
    colormap: str
    vmin: float
    vmax: float


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
        self.setFixedHeight(len(entries) * ROW + (8 if entries else 0))
        self.update()

    def paintEvent(self, event) -> None:  # noqa: N802 (Qt naming)
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor(COLORS["panel"]))
        metrics = QFontMetrics(self.font())
        label_width = max((metrics.horizontalAdvance(e.label) for e in self.entries), default=0)
        label_width = min(label_width + 16, self.width() // 3)
        number_width = metrics.horizontalAdvance("-0.000e+00") + 12
        for row, entry in enumerate(self.entries):
            top = 4 + row * ROW
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
            gradient = QLinearGradient(bar.left(), 0, bar.right(), 0)
            table = lut(entry.colormap, 32)
            for i, (r, g, b, _a) in enumerate(table):
                gradient.setColorAt(i / (len(table) - 1), QColor.fromRgbF(r, g, b))
            painter.fillRect(bar, gradient)
            painter.setPen(QPen(QColor(255, 255, 255, 90)))
            painter.drawRect(bar)
            painter.setPen(QColor(COLORS["text"]))
            painter.drawText(
                QRectF(label_width, top, number_width - 6, ROW),
                Qt.AlignVCenter | Qt.AlignRight,
                _number(entry.vmin),
            )
            painter.drawText(
                QRectF(bar.right() + 6, top, number_width, ROW),
                Qt.AlignVCenter | Qt.AlignLeft,
                _number(entry.vmax),
            )
        painter.end()
