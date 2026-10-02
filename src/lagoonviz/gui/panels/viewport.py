# SPDX-License-Identifier: GPL-3.0-or-later
"""Central 3D viewport. Phase 0: a placeholder; the pygfx canvas replaces it in phase 1."""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QLinearGradient, QPainter, QPixmap
from PySide6.QtWidgets import QWidget

from lagoonviz.gui.resources import resource_path

TOP = QColor("#3b5f8f")
BOTTOM = QColor("#1c3354")


class ViewportPlaceholder(QWidget):
    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self.setObjectName("viewport")
        self.setMinimumSize(320, 240)
        self._icon = QPixmap(str(resource_path("viewport_mark.png")))
        self.message = "3D viewport\npygfx rendering arrives in phase 1"

    def paintEvent(self, event) -> None:  # noqa: N802 (Qt naming)
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        gradient = QLinearGradient(0, 0, 0, self.height())
        gradient.setColorAt(0.0, TOP)
        gradient.setColorAt(1.0, BOTTOM)
        painter.fillRect(self.rect(), gradient)
        size = min(160, self.height() // 3, self.width() // 3)
        if not self._icon.isNull() and size > 16:
            icon = self._icon.scaled(size, size, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            painter.setOpacity(0.45)
            painter.drawPixmap((self.width() - size) // 2, self.height() // 2 - size, icon)
            painter.setOpacity(1.0)
        painter.setPen(QColor(255, 255, 255, 200))
        text_rect = self.rect().adjusted(0, self.height() // 2 + 12, 0, 0)
        painter.drawText(text_rect, Qt.AlignHCenter | Qt.AlignTop, self.message)
        painter.end()
