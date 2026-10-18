# SPDX-License-Identifier: GPL-3.0-or-later
"""The Probe tool's box beside the 3D view: every value at the clicked point.

Like ParaView's point inspection: with *Probe* on, a click on a surface in the 3D
view lists the item, the point and all the arrays of its dataset there (vectors as
magnitude and components). The values follow the timeline; *Copy* puts them on the
clipboard as tab-separated lines.
"""

from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QGuiApplication
from PySide6.QtWidgets import (
    QAbstractItemView,
    QFrame,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QTableWidget,
    QTableWidgetItem,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from lagoonviz.gui.theme import COLORS

#: what the box says before the first click
HELP = "Click on a surface in the 3D view to list every value there. Drags still turn the view."


class ProbeBox(QFrame):
    """Item, point and all values at the probed point."""

    #: the × button: the Probe tool is to end
    closed = Signal()

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self.setObjectName("probeBox")
        self.setFixedWidth(260)
        self.probe = None

        top = QHBoxLayout()
        top.setContentsMargins(0, 0, 0, 0)
        top.setSpacing(2)
        self.title = QLabel("Probe")
        self.title.setObjectName("probeTitle")
        self.title.setWordWrap(True)
        top.addWidget(self.title, 1)
        self.copy_button = QToolButton()
        self.copy_button.setText("Copy")
        self.copy_button.setAutoRaise(True)
        self.copy_button.setToolTip("Copy the values (tab-separated name and value)")
        self.copy_button.clicked.connect(self.copy)
        self.copy_button.setEnabled(False)
        top.addWidget(self.copy_button)
        close = QToolButton()
        close.setText("×")
        close.setAutoRaise(True)
        close.setToolTip("End Probe (Esc)")
        close.clicked.connect(self.closed.emit)
        top.addWidget(close)

        self.note = QLabel(HELP)
        self.note.setObjectName("probeNote")
        self.note.setWordWrap(True)

        self.table = QTableWidget(0, 2)
        self.table.setHorizontalHeaderLabels(["Variable", "Value"])
        self.table.verticalHeader().hide()
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setShowGrid(False)
        self.table.setAlternatingRowColors(True)
        self.table.setWordWrap(False)
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(1, QHeaderView.Stretch)
        self.table.verticalHeader().setDefaultSectionSize(20)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 6, 6, 6)
        layout.setSpacing(4)
        layout.addLayout(top)
        layout.addWidget(self.note)
        layout.addWidget(self.table, 1)
        self.apply_theme_colors()

    def apply_theme_colors(self) -> None:
        self.setStyleSheet(
            f"#probeBox {{ background: {COLORS['panel']};"
            f" border-left: 1px solid {COLORS['border']}; }}"
            f"#probeTitle {{ color: {COLORS['text']}; font-weight: bold; }}"
            f"#probeNote {{ color: {COLORS['dim']}; }}"
            f"QToolButton {{ color: {COLORS['text']}; padding: 2px 6px; }}"
        )

    def set_probe(self, probe, note: str = "") -> None:
        """Show a Probe (None: nothing probed, ``note`` says why)."""
        self.probe = probe
        self.copy_button.setEnabled(probe is not None)
        if probe is None:
            self.title.setText("Probe")
            self.note.setVisible(True)
            self.note.setText(note or HELP)
            self.table.setRowCount(0)
            return
        self.title.setText(probe.name)
        notes = [probe.case] if probe.case else []
        if probe.nearest:
            notes.append("nearest grid point")
        if not probe.complete:
            notes.append("reading all variables…")
        if note:
            notes.append(note)
        self.note.setText(" · ".join(notes))
        self.note.setVisible(bool(notes))
        rows = probe.rows()
        self.table.setRowCount(len(rows))
        for row, (name, value) in enumerate(rows):
            left = QTableWidgetItem(name)
            right = QTableWidgetItem(value)
            right.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
            right.setToolTip(f"{name} = {value}")
            self.table.setItem(row, 0, left)
            self.table.setItem(row, 1, right)

    def clear(self, note: str = "") -> None:
        self.set_probe(None, note)

    def copy(self) -> None:
        if self.probe is not None:
            QGuiApplication.clipboard().setText(self.probe.text())
