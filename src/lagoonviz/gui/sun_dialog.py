# SPDX-License-Identifier: GPL-3.0-or-later
# Architect: Hans Bihs
"""View → Sun: where the light comes from, for the 3D view, films and path tracing.

Two sliders, the direction (azimuth, measured from +x towards +y) and the height
above the horizon (elevation); the 3D view follows while they move. *Look's own*
puts the sun back where the look has it. The dialog stays open beside the window.
"""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSlider,
    QWidget,
)

from lagoonviz.core import Session

OWN = "The look's own sun."
YOURS = "Your sun: films and path tracing use it too."


class SunDialog(QDialog):
    def __init__(self, parent: QWidget, session: Session, look=lambda: None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Sun")
        self.session = session
        self.look = look  # the viewport's look now (its own sun)
        self._quiet = False
        form = QFormLayout(self)
        self.azimuth, azimuth_row = self._row(0.0, 360.0, "°", "From +x (0°) towards +y (90°)")
        self.elevation, elevation_row = self._row(
            1.0, 90.0, "°", "Height above the horizon: low for long shadows, 90° overhead"
        )
        form.addRow("Direction", azimuth_row)
        form.addRow("Height", elevation_row)
        self.note = QLabel()
        self.note.setWordWrap(True)
        form.addRow("", self.note)
        buttons = QDialogButtonBox(QDialogButtonBox.Close)
        self.own = QPushButton("Look's own")
        self.own.setToolTip("Put the sun back where the viewport look has it")
        buttons.addButton(self.own, QDialogButtonBox.ResetRole)
        self.own.clicked.connect(self.reset)
        buttons.rejected.connect(self.close)
        form.addRow(buttons)
        self.show_sun()

    def _row(self, low: float, high: float, suffix: str, tip: str):
        slider = QSlider(Qt.Horizontal)
        slider.setRange(round(low * 10), round(high * 10))
        spin = QDoubleSpinBox()
        spin.setRange(low, high)
        spin.setDecimals(1)
        spin.setSuffix(suffix)
        spin.setWrapping(high == 360.0)
        for widget in (slider, spin):
            widget.setToolTip(tip)
        slider.valueChanged.connect(lambda v: spin.setValue(v / 10))
        spin.valueChanged.connect(lambda v: slider.setValue(round(v * 10)))
        spin.valueChanged.connect(lambda _v: self._changed())
        row = QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        row.addWidget(slider, 1)
        row.addWidget(spin)
        holder = QWidget()
        holder.setLayout(row)
        return spin, holder

    def show_sun(self) -> None:
        """The sliders at the session's sun (or the look's)."""
        azimuth, elevation = self.session.sun_angles(self.look())
        self._quiet = True
        self.azimuth.setValue(azimuth)
        self.elevation.setValue(max(elevation, 1.0))
        self._quiet = False
        own = self.session.scene.sun is None
        self.note.setText(OWN if own else YOURS)
        self.own.setEnabled(not own)

    def _changed(self) -> None:
        if self._quiet:
            return
        self.session.set_sun(self.azimuth.value() % 360.0, self.elevation.value())
        self.note.setText(YOURS)
        self.own.setEnabled(True)

    def reset(self) -> None:
        self.session.set_sun(None, None)
        self.show_sun()
