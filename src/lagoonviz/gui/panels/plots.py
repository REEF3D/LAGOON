# SPDX-License-Identifier: GPL-3.0-or-later
"""Linked 2D plots (gauges, forces, spectra). Phase 0: placeholder; fastplotlib in phase 2."""

from __future__ import annotations

from PySide6.QtCore import QSize, Qt
from PySide6.QtWidgets import QLabel, QVBoxLayout, QWidget


class PlotsPanel(QWidget):
    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self.setObjectName("plots")
        label = QLabel("Linked 2D plots (gauges, forces, spectra) arrive in phase 2.")
        label.setAlignment(Qt.AlignCenter)
        label.setEnabled(False)
        layout = QVBoxLayout(self)
        layout.addWidget(label)

    def sizeHint(self) -> QSize:  # noqa: N802 (Qt naming)
        return QSize(600, 140)
