# SPDX-License-Identifier: GPL-3.0-or-later
# Architect: Hans Bihs
"""View → Legend: where films put the time and the colour bars and how they look,
and whether the 3D view shows its bars (lagoonviz.core.legend). Each item's own
bar (on or off, title, numbers, ticks) is in its Properties. The dialog stays open
beside the window; the settings are kept with the session.
"""

from __future__ import annotations

from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFormLayout,
    QLabel,
    QPushButton,
    QWidget,
)

from lagoonviz.core import Session
from lagoonviz.core.legend import CORNERS, LIMITS, TIME_PLACES, VIEW_PLACES

NOTE = (
    "Films (File → Export animation, with Time and colour bars) follow these. Each "
    "item's bar, its title, numbers and ticks are in its Properties (Colour bar, Bar "
    "title, Bar numbers, Bar ticks); figures place theirs in the figure editor."
)


class LegendDialog(QDialog):
    def __init__(self, parent: QWidget, session: Session) -> None:
        super().__init__(parent)
        self.setWindowTitle("Legend")
        self.session = session
        self._quiet = False
        form = QFormLayout(self)
        self.bars = self._choices(CORNERS, "Corner of the films' colour bars")
        self.time = self._choices(TIME_PLACES, "Corner of the films' time, or off")
        self.length = self._number("length", 100.0, " %", "A bar's length, of the film's width")
        self.thickness = self._number("thickness", 100.0, " %", "The bars' thickness")
        self.text = self._number("text", 100.0, " %", "The size of the bars' text and the time")
        self.box = QCheckBox("A light box behind the bars and the time")
        self.box.toggled.connect(self._changed)
        self.view = self._choices(VIEW_PLACES, "The colour bars of the 3D view")
        form.addRow("Colour bars", self.bars)
        form.addRow("Time", self.time)
        form.addRow("Bar length", self.length)
        form.addRow("Bar thickness", self.thickness)
        form.addRow("Text size", self.text)
        form.addRow("", self.box)
        form.addRow("3D view", self.view)
        note = QLabel(NOTE)
        note.setWordWrap(True)
        note.setEnabled(False)
        form.addRow(note)
        buttons = QDialogButtonBox(QDialogButtonBox.Close)
        self.reset_button = QPushButton("As by default")
        buttons.addButton(self.reset_button, QDialogButtonBox.ResetRole)
        self.reset_button.clicked.connect(self.reset)
        buttons.rejected.connect(self.close)
        form.addRow(buttons)
        self.show_layout()

    def _choices(self, choices, tip: str) -> QComboBox:
        box = QComboBox()
        for choice in choices:
            box.addItem(choice, choice)
        box.setToolTip(tip)
        box.currentIndexChanged.connect(self._changed)
        return box

    def _number(self, key: str, scale: float, suffix: str, tip: str) -> QDoubleSpinBox:
        low, high = LIMITS[key]
        spin = QDoubleSpinBox()
        spin.setRange(low * scale, high * scale)
        spin.setDecimals(0)
        spin.setSingleStep(5)
        spin.setSuffix(suffix)
        spin.setToolTip(tip)
        spin.valueChanged.connect(self._changed)
        return spin

    def show_layout(self) -> None:
        """The fields at the session's legend layout."""
        layout = self.session.legend_layout()
        self._quiet = True
        try:
            for box, key in ((self.bars, "bars"), (self.time, "time"), (self.view, "view")):
                box.setCurrentIndex(max(box.findData(layout[key]), 0))
            self.length.setValue(100.0 * layout["length"])
            self.thickness.setValue(100.0 * layout["thickness"])
            self.text.setValue(100.0 * layout["text"])
            self.box.setChecked(bool(layout["box"]))
        finally:
            self._quiet = False

    def _changed(self, *_args) -> None:
        if self._quiet:
            return
        self.session.set_legend(
            bars=self.bars.currentData(),
            time=self.time.currentData(),
            length=self.length.value() / 100.0,
            thickness=self.thickness.value() / 100.0,
            text=self.text.value() / 100.0,
            box=self.box.isChecked(),
            view=self.view.currentData(),
        )

    def reset(self) -> None:
        self.session.set_legend(reset=True)
        self.show_layout()
