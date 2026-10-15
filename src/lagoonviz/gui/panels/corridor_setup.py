# SPDX-License-Identifier: GPL-3.0-or-later
# Architect: Hans Bihs
"""The River tab of the set-up: a river corridor shaped by points on its two banks,
dragged in the 2D top view, and the curvilinear grid it makes for REEF3D (G 50 2,
curvgrid.dat; ``core.corridor``).

- *Start* lays a straight corridor along the domain, *Draw* one over an area dragged
  in the 2D view (it flows along the area's longer side).
- *Drag points*: press on a point of a bank in the 2D top view and drag it; the bank
  follows as a spline through the points or a smoother B-spline (*Curve*). Elsewhere
  the view pans as always; Esc stops.
- *Points per bank* spaces that many points along the banks as they are now.
- *Cells along* and *across*, *Smoothing* (Winslow's elliptic grid equations) and
  the bed (a slope from the upstream end, or geo.dat's points) set the grid, which
  the set-up draws as it changes; *Make grid* writes curvgrid.dat, G 50 2 and
  DIVEMesh's logical box (B 10, B 1).

Every change is an undo step; corridor.json and curvgrid.dat are written at once.
"""

from __future__ import annotations

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from lagoonviz.core import corridor as corridors

#: the purpose of an area drawn in the 2D view for this tab
PURPOSE = "river corridor"
CURVE_LABELS = {
    "through": "Through the points (Catmull-Rom)",
    "bspline": "Smooth, pulled by the points (B-spline)",
}
BED_LABELS = {"slope": "Sloping from the upstream end", "geo": "geo.dat's points"}


class CorridorTab(QWidget):
    """The River tab of the set-up panel (module doc)."""

    #: draw an area in the 2D top view: (case id, purpose)
    drawArea = Signal(str, str)
    #: drag the corridor's points in the 2D view (case id), or stop
    editPoints = Signal(str)
    stopEditing = Signal()
    #: a message for the panel's status line: (text, error)
    message = Signal(str, bool)

    def __init__(self, panel, parent=None) -> None:
        super().__init__(parent)
        self.panel = panel
        self.session = panel.session

        self.note = QLabel()
        self.note.setWordWrap(True)
        self.start_button = QPushButton("Start")
        self.start_button.setToolTip("A straight corridor along the domain")
        self.start_button.clicked.connect(self.start)
        self.draw_button = QPushButton("Draw")
        self.draw_button.setToolTip(
            "Drag an area in the 2D top view: a straight corridor over it, along its longer side"
        )
        self.draw_button.clicked.connect(self.draw)
        self.edit_button = QPushButton("Drag points")
        self.edit_button.setCheckable(True)
        self.edit_button.setToolTip(
            "Drag the points of the banks in the 2D top view (elsewhere the view pans; Esc stops)"
        )
        self.edit_button.toggled.connect(self._edit_toggled)
        self.remove_button = QPushButton("Remove")
        self.remove_button.setToolTip("Remove the corridor (corridor.json)")
        self.remove_button.clicked.connect(self.remove)
        row = QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        for button in (self.start_button, self.draw_button, self.edit_button, self.remove_button):
            row.addWidget(button)
        row.addStretch(1)

        # ---- the banks
        self.points = QSpinBox()
        self.points.setRange(corridors.MIN_POINTS, corridors.MAX_POINTS)
        self.points.setToolTip(
            "Points on each bank: they are spaced equally along the banks as they are now"
        )
        self.curve = QComboBox()
        for key, label in CURVE_LABELS.items():
            self.curve.addItem(label, key)
        self.curve.setToolTip(
            "Through the points: the bank passes every point (centripetal Catmull-Rom). "
            "B-spline: smoother, the points pull on it; it passes the first and last"
        )
        self.reverse_button = QPushButton("Reverse flow")
        self.reverse_button.setToolTip("Upstream and downstream (and so left and right) swapped")
        self.reverse_button.clicked.connect(self.reverse)
        banks = QGroupBox("Banks")
        banks_form = QFormLayout(banks)
        banks_form.addRow("Points per bank", self.points)
        banks_form.addRow("Curve", self.curve)
        banks_form.addRow("", self.reverse_button)

        # ---- the grid
        self.along = QSpinBox()
        self.along.setRange(4, 100_000)
        self.along.setToolTip("Cells along the river (i, downstream)")
        self.across = QSpinBox()
        self.across.setRange(4, 10_000)
        self.across.setToolTip("Cells across the river (j, from the right bank)")
        self.smoothing = QSpinBox()
        self.smoothing.setRange(0, 2000)
        self.smoothing.setToolTip(
            "Sweeps of Winslow's elliptic grid equations: smoother, more even cells in "
            "bends (0: straight lines across)"
        )
        self.bed = QComboBox()
        for key, label in BED_LABELS.items():
            self.bed.addItem(label, key)
        self.z_start = QDoubleSpinBox()
        self.z_start.setRange(-1e5, 1e5)
        self.z_start.setDecimals(3)
        self.z_start.setSuffix(" m")
        self.z_start.setToolTip("The bed's height at the upstream end")
        self.slope = QDoubleSpinBox()
        self.slope.setRange(-1.0, 1.0)
        self.slope.setDecimals(5)
        self.slope.setSingleStep(0.0001)
        self.slope.setToolTip("How much the bed falls per metre downstream (m/m)")
        self.quality = QLabel()
        self.quality.setWordWrap(True)
        self.make_button = QPushButton("Make grid")
        self.make_button.setToolTip(
            "Write curvgrid.dat, G 50 2 in ctrl.txt and DIVEMesh's logical box (B 10, B 1)"
        )
        self.make_button.clicked.connect(self.make_grid)
        grid = QGroupBox("Curvilinear grid (G 50 2)")
        grid_form = QFormLayout(grid)
        grid_form.addRow("Cells along", self.along)
        grid_form.addRow("Cells across", self.across)
        grid_form.addRow("Smoothing", self.smoothing)
        grid_form.addRow("Bed", self.bed)
        grid_form.addRow("Bed at upstream end", self.z_start)
        grid_form.addRow("Bed slope", self.slope)
        grid_form.addRow(self.quality)
        grid_form.addRow("", self.make_button)

        for box in (self.points, self.along, self.across, self.smoothing):
            box.setKeyboardTracking(False)  # typed: changed on Return or leaving it
            box.valueChanged.connect(lambda _v: self._commit())
        for box in (self.z_start, self.slope):
            box.setKeyboardTracking(False)
            box.valueChanged.connect(lambda _v: self._commit())
        for box in (self.curve, self.bed):
            box.activated.connect(lambda _i: self._commit())
        self._editors = (
            self.points,
            self.curve,
            self.along,
            self.across,
            self.smoothing,
            self.bed,
            self.z_start,
            self.slope,
        )

        body = QWidget()
        column = QVBoxLayout(body)
        column.setContentsMargins(0, 0, 0, 0)
        column.addWidget(banks)
        column.addWidget(grid)
        column.addStretch(1)
        self.form = body
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(body)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.addWidget(self.note)
        layout.addLayout(row)
        layout.addWidget(scroll, 1)

    # ------------------------------------------------------------- the case
    def corridor(self):
        case = self.panel.case_id
        if not case or case not in self.session.scene:
            return None
        try:
            return self.session.corridor(case)
        except ValueError:
            return None

    def refresh(self) -> None:
        """The form from the case's corridor."""
        has_case = bool(self.panel.case_id) and self.panel.case_id in self.session.scene
        corridor = self.corridor() if has_case else None
        solver = self.panel.solver() if has_case else ""
        for button in (self.start_button, self.draw_button):
            button.setEnabled(has_case)
        for widget in (self.edit_button, self.remove_button, self.form):
            widget.setEnabled(corridor is not None)
        if corridor is None and self.edit_button.isChecked():
            self.edit_button.setChecked(False)
        if not has_case:
            self.note.setText("Open a case to lay a river corridor in it.")
        elif corridor is None:
            self.note.setText(
                "A river corridor: two banks shaped by points you drag, and the "
                "curvilinear grid between them (G 50 2, SFLOW and NHFLOW). Start one "
                "along the domain, or Draw it."
            )
        else:
            text = "Drag the points of the banks in the 2D view (Drag points)."
            if solver and solver not in ("SFLOW", "NHFLOW"):
                text += f" REEF3D::{solver} has no curvilinear grids: SFLOW and NHFLOW do."
            self.note.setText(text)
        if corridor is None:
            self.quality.setText("")
            return
        for widget in self._editors:
            widget.blockSignals(True)
        self.points.setValue(len(corridor.left))
        self.curve.setCurrentIndex(self.curve.findData(corridor.curve))
        self.along.setValue(corridor.cells_along)
        self.across.setValue(corridor.cells_across)
        self.smoothing.setValue(corridor.smoothing)
        self.bed.setCurrentIndex(self.bed.findData(corridor.bed))
        self.z_start.setValue(corridor.z_start)
        self.slope.setValue(corridor.slope)
        for widget in self._editors:
            widget.blockSignals(False)
        sloping = corridor.bed == "slope"
        self.z_start.setEnabled(sloping)
        self.slope.setEnabled(sloping)
        try:
            x, y = corridor.grid()
            quality = corridors.quality(x, y)
        except (ValueError, FloatingPointError) as error:
            self.quality.setText(f"No grid: {error}")
            self.make_button.setEnabled(False)
            return
        self.quality.setText(quality.text())
        self.quality.setStyleSheet("color: #e5484d;" if quality.folded else "")
        self.make_button.setEnabled(not quality.folded)

    # ------------------------------------------------------------- edits
    def _run(self, action, done: str = "") -> bool:
        try:
            action()
        except ValueError as error:
            self.message.emit(str(error), True)
            return False
        if done:
            self.message.emit(done, False)
        return True

    def _commit(self) -> None:
        corridor = self.corridor()
        if corridor is None:
            return
        options = {
            "curve": self.curve.currentData(),
            "cells_along": self.along.value(),
            "cells_across": self.across.value(),
            "smoothing": self.smoothing.value(),
            "bed": self.bed.currentData(),
            "z_start": round(self.z_start.value(), 6),
            "slope": round(self.slope.value(), 8),
        }
        changed = {k: v for k, v in options.items() if getattr(corridor, k) != v}
        points = self.points.value() if self.points.value() != len(corridor.left) else None
        if changed or points is not None:
            self._run(
                lambda: self.session.set_corridor_options(
                    points=points, case=self.panel.case_id, **changed
                )
            )

    def start(self) -> None:
        self._run(
            lambda: self.session.start_corridor(case=self.panel.case_id),
            "A straight corridor along the domain: drag its points (Drag points)",
        )
        if self.corridor() is not None:
            self.edit_button.setChecked(True)

    def draw(self) -> None:
        if self.panel.case_id:
            self.drawArea.emit(self.panel.case_id, PURPOSE)

    def area_drawn(self, purpose: str, area) -> None:
        """An area dragged in the 2D view: a straight corridor over it."""
        if purpose != PURPOSE or not self.panel.case_id:
            return
        points = self.points.value() if self.corridor() is not None else 12
        if self._run(
            lambda: self.session.start_corridor(area, points, case=self.panel.case_id),
            "A straight corridor over the area: drag its points (Drag points)",
        ):
            self.edit_button.setChecked(True)

    def _edit_toggled(self, on: bool) -> None:
        if on and self.panel.case_id:
            self.editPoints.emit(self.panel.case_id)
        else:
            self.stopEditing.emit()

    def editing_stopped(self) -> None:
        """The 3D view stopped the dragging (Esc, another view)."""
        self.edit_button.blockSignals(True)
        self.edit_button.setChecked(False)
        self.edit_button.blockSignals(False)

    def point_moved(self, case_id: str, side: str, index: int, x: float, y: float) -> None:
        """A point dragged in the 2D view."""
        if case_id and case_id in self.session.scene:
            self._run(
                lambda: self.session.move_corridor_point(
                    side, index, round(x, 4), round(y, 4), case=case_id
                )
            )

    def reverse(self) -> None:
        self._run(
            lambda: self.session.reverse_corridor(case=self.panel.case_id),
            "The river flows the other way now",
        )

    def remove(self) -> None:
        self.edit_button.setChecked(False)
        self._run(lambda: self.session.remove_corridor(case=self.panel.case_id))

    def make_grid(self) -> None:
        quality = None

        def make():
            nonlocal quality
            quality = self.session.make_corridor_grid(case=self.panel.case_id)

        if self._run(make) and quality is not None:
            self.message.emit(
                f"curvgrid.dat written ({quality.cells[0]} x {quality.cells[1]} cells), "
                "G 50 2 and DIVEMesh's logical box set: Save, then Run DIVEMesh",
                False,
            )
