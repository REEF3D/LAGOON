# SPDX-License-Identifier: GPL-3.0-or-later
# Architect: Hans Bihs
"""The Blocks tab of the set-up: a multi-block grid (REEF3D's G 50 1, DIVEMesh's
R 20 1; ``core.multiblock``), kept as its topology in blocks.json: corners, edges
shaped by points, blocks whose faces are chains of edges. Two blocks that share an
edge meet there, node for node.

- *Start* lays one block over the domain, *Draw* one over an area dragged in the 2D
  view, *From corridor* makes the river corridor (River tab) block 0.
- *Drag points*: press on a corner or a point of an edge in the 2D top view and drag
  it; *Cut edge*: click an edge to cut it in two (a block can then meet only a part
  of a face); *Add block*: click an edge on the outline to add a block outside it
  (*Depth*, *Cells*). Esc stops.
- A block: its name, smoothing, and the boundary of each face without an interface
  (wall, inflow with its discharge, outflow with its water level); *Block 0* makes
  it the grid of the run; *Remove* takes it out.
- An edge: its cells (carried on to the opposite faces of the blocks), its curve
  and its grading; *Finer* and *Coarser* double or halve every edge's cells.
- *Rules* lists what DIVEMesh and REEF3D would refuse (red) or warn about, the set-up
  marks them where they are, with the cells' quality; *Make grid* writes blocks.dat,
  R 20 1, the sides of block 0, G 50 1 and the G 54 lines of the open faces.

Every change is an undo step; blocks.json and blocks.dat are written at once.
"""

from __future__ import annotations

from PySide6.QtCore import Signal
from PySide6.QtGui import QBrush, QColor
from PySide6.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from lagoonviz.core import multiblock as mb

#: the purpose of an area drawn in the 2D view for this tab
PURPOSE = "block"
BC_LABELS = {"wall": "Wall", "inflow": "Inflow (discharge)", "outflow": "Outflow (water level)"}
CURVE_LABELS = {
    "straight": "Straight",
    "through": "Through the points (Catmull-Rom)",
    "bspline": "Smooth, pulled by the points (B-spline)",
}
FACE_LABELS = {"i0": "Face -i", "i1": "Face +i", "j0": "Face -j", "j1": "Face +j"}
ERROR_COLOR = "#e5484d"
WARNING_COLOR = "#e5a000"


class FaceRow(QWidget):
    """One face of a block: its edges and cells, and its boundary."""

    changed = Signal(str, str, object)  # face, kind, value

    def __init__(self, face: str, parent=None) -> None:
        super().__init__(parent)
        self.face = face
        self.kind = QComboBox()
        for key, label in BC_LABELS.items():
            self.kind.addItem(label, key)
        self.value = QDoubleSpinBox()
        self.value.setRange(-1e6, 1e6)
        self.value.setDecimals(3)
        self.value.setKeyboardTracking(False)
        self.info = QLabel()
        self.info.setWordWrap(True)
        row = QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        row.addWidget(self.kind, 1)
        row.addWidget(self.value)
        column = QVBoxLayout(self)
        column.setContentsMargins(0, 0, 0, 0)
        column.addLayout(row)
        column.addWidget(self.info)
        self.kind.activated.connect(lambda _i: self._emit())
        self.value.valueChanged.connect(lambda _v: self._emit())

    def show_face(self, topology, block: int) -> None:
        b = topology.blocks[block]
        shared = topology.interface_faces(block)[self.face]
        chain = topology.blocks[block].faces[self.face]
        cells = topology.face_cells(block, self.face)
        text = f"{', '.join(chain)} · {cells} cells"
        if shared:
            text += " · interface: " + ", ".join(shared)
        self.info.setText(text)
        kind = b.bc_type(self.face)
        value = b.bc_value(self.face)
        for widget in (self.kind, self.value):
            widget.blockSignals(True)
        self.kind.setCurrentIndex(self.kind.findData(kind))
        self.value.setValue(value if value is not None else 0.0)
        self.value.setEnabled(kind != "wall")
        self.value.setSuffix(" m3/s" if kind == "inflow" else " m")
        self.kind.setEnabled(len(shared) < len(chain))  # not on a face all interface
        for widget in (self.kind, self.value):
            widget.blockSignals(False)

    def _emit(self) -> None:
        kind = self.kind.currentData()
        self.value.setEnabled(kind != "wall")
        self.changed.emit(self.face, kind, None if kind == "wall" else self.value.value())


class MultiblockTab(QWidget):
    """The Blocks tab of the set-up panel (module doc)."""

    #: draw an area in the 2D top view: (case id, purpose)
    drawArea = Signal(str, str)
    #: edit the blocks in the 2D view: (case id, mode "drag", "cut", "extend"), or stop
    editBlocks = Signal(str, str)
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
        self.start_button.setToolTip("One block over the domain")
        self.start_button.clicked.connect(self.start)
        self.draw_button = QPushButton("Draw")
        self.draw_button.setToolTip(
            "Drag an area in the 2D top view: a block over it (the first, or one more)"
        )
        self.draw_button.clicked.connect(self.draw)
        self.corridor_button = QPushButton("From corridor")
        self.corridor_button.setToolTip("The river corridor (River tab) as block 0")
        self.corridor_button.clicked.connect(self.from_corridor)
        self.remove_button = QPushButton("Remove")
        self.remove_button.setToolTip("Remove the blocks (blocks.json)")
        self.remove_button.clicked.connect(self.remove)
        row = QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        for button in (
            self.start_button,
            self.draw_button,
            self.corridor_button,
            self.remove_button,
        ):
            row.addWidget(button)
        row.addStretch(1)

        # ---- editing in the 2D view
        self.drag_button = QPushButton("Drag points")
        self.cut_button = QPushButton("Cut edge")
        self.extend_button = QPushButton("Add block")
        tips = (
            "Drag the corners and the points of the edges in the 2D top view (Esc stops)",
            "Click an edge in the 2D top view to cut it in two at the node nearest the "
            "click: a block can then meet a part of a face",
            "Click an edge on the outline in the 2D top view: a block outside it, Depth "
            "deep with Cells cells",
        )
        self._modes = {}
        for button, mode, tip in zip(
            (self.drag_button, self.cut_button, self.extend_button),
            ("drag", "cut", "extend"),
            tips,
            strict=True,
        ):
            button.setCheckable(True)
            button.setToolTip(tip)
            button.toggled.connect(lambda on, m=mode: self._mode_toggled(m, on))
            self._modes[mode] = button
        self.depth = QDoubleSpinBox()
        self.depth.setRange(0.001, 1e6)
        self.depth.setValue(50.0)
        self.depth.setSuffix(" m")
        self.depth.setToolTip("How far the new block reaches out from the edge")
        self.depth_cells = QSpinBox()
        self.depth_cells.setRange(1, 100_000)
        self.depth_cells.setValue(10)
        self.depth_cells.setToolTip("Cells of the new block outwards")
        edit_box = QGroupBox("In the 2D view")
        edit_form = QFormLayout(edit_box)
        modes = QHBoxLayout()
        for button in self._modes.values():
            modes.addWidget(button)
        edit_form.addRow(modes)
        edit_form.addRow("Depth", self.depth)
        edit_form.addRow("Cells", self.depth_cells)

        # ---- a block
        self.block = QComboBox()
        self.block.setToolTip("The block to change")
        self.block.currentIndexChanged.connect(lambda _i: self._show_block())
        self.block_name = QLineEdit()
        self.block_name.editingFinished.connect(self._rename)
        self.smoothing = QSpinBox()
        self.smoothing.setRange(0, 5000)
        self.smoothing.setKeyboardTracking(False)
        self.smoothing.setToolTip(
            "Sweeps of Winslow's elliptic grid equations (the nodes of the walls slide "
            "along them): smoother, more even cells"
        )
        self.smoothing.valueChanged.connect(lambda _v: self._commit_block())
        self.native_button = QPushButton("Block 0")
        self.native_button.setToolTip("Make it block 0, the grid of the run (DIVEMesh's box)")
        self.native_button.clicked.connect(self.make_native)
        self.remove_block_button = QPushButton("Remove block")
        self.remove_block_button.clicked.connect(self.remove_block)
        self.faces = {face: FaceRow(face) for face in mb.FACES}
        for row_widget in self.faces.values():
            row_widget.changed.connect(self._set_face)
        block_box = QGroupBox("Block")
        block_form = QFormLayout(block_box)
        block_form.addRow("Block", self.block)
        block_form.addRow("Name", self.block_name)
        block_form.addRow("Smoothing", self.smoothing)
        for face, row_widget in self.faces.items():
            block_form.addRow(FACE_LABELS[face], row_widget)
        buttons = QHBoxLayout()
        buttons.addWidget(self.native_button)
        buttons.addWidget(self.remove_block_button)
        buttons.addStretch(1)
        block_form.addRow(buttons)

        # ---- an edge
        self.edge = QComboBox()
        self.edge.setToolTip("The edge to change")
        self.edge.currentIndexChanged.connect(lambda _i: self._show_edge())
        self.cells = QSpinBox()
        self.cells.setRange(1, 100_000)
        self.cells.setKeyboardTracking(False)
        self.cells.setToolTip("Cells along the edge (carried on to the opposite faces)")
        self.curve = QComboBox()
        for key, label in CURVE_LABELS.items():
            self.curve.addItem(label, key)
        self.ratio = QDoubleSpinBox()
        self.ratio.setRange(0.01, 100.0)
        self.ratio.setDecimals(3)
        self.ratio.setSingleStep(0.1)
        self.ratio.setKeyboardTracking(False)
        self.ratio.setToolTip("Grading: the last cell's length over the first's (1: equal)")
        self.cells.valueChanged.connect(lambda _v: self._commit_edge())
        self.ratio.valueChanged.connect(lambda _v: self._commit_edge())
        self.curve.activated.connect(lambda _i: self._commit_edge())
        self.finer_button = QPushButton("Finer")
        self.finer_button.setToolTip("Twice the cells on every edge")
        self.finer_button.clicked.connect(lambda: self.scale(2.0))
        self.coarser_button = QPushButton("Coarser")
        self.coarser_button.setToolTip("Half the cells on every edge")
        self.coarser_button.clicked.connect(lambda: self.scale(0.5))
        edge_box = QGroupBox("Edge")
        edge_form = QFormLayout(edge_box)
        edge_form.addRow("Edge", self.edge)
        edge_form.addRow("Cells", self.cells)
        edge_form.addRow("Curve", self.curve)
        edge_form.addRow("Grading", self.ratio)
        scale = QHBoxLayout()
        scale.addWidget(self.finer_button)
        scale.addWidget(self.coarser_button)
        scale.addStretch(1)
        edge_form.addRow("All edges", scale)

        # ---- the rules and the grid
        self.quality = QLabel()
        self.quality.setWordWrap(True)
        self.rules = QListWidget()
        self.rules.setMinimumHeight(60)
        self.rules.setToolTip("What DIVEMesh and REEF3D would refuse (red) or warn about")
        self.make_button = QPushButton("Make grid")
        self.make_button.setToolTip(
            "Write blocks.dat, R 20 1 and the sides of block 0 in control.txt, G 50 1 "
            "and the open faces of the other blocks (G 54) in ctrl.txt"
        )
        self.make_button.clicked.connect(self.make_grid)
        grid_box = QGroupBox("Multi-block grid (G 50 1)")
        grid_form = QFormLayout(grid_box)
        grid_form.addRow(self.quality)
        grid_form.addRow("Rules", self.rules)
        grid_form.addRow("", self.make_button)

        body = QWidget()
        column = QVBoxLayout(body)
        column.setContentsMargins(0, 0, 0, 0)
        for box in (edit_box, block_box, edge_box, grid_box):
            column.addWidget(box)
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
    def topology(self):
        case = self.panel.case_id
        if not case or case not in self.session.scene:
            return None
        try:
            return self.session.multiblock(case)
        except ValueError:
            return None

    def refresh(self) -> None:
        """The form from the case's blocks."""
        has_case = bool(self.panel.case_id) and self.panel.case_id in self.session.scene
        topology = self.topology() if has_case else None
        for button in (self.start_button, self.draw_button):
            button.setEnabled(has_case)
        try:
            corridor = self.session.corridor(self.panel.case_id) if has_case else None
        except ValueError:
            corridor = None
        self.corridor_button.setEnabled(corridor is not None)
        self.remove_button.setEnabled(topology is not None)
        self.form.setEnabled(topology is not None)
        if topology is None:
            for button in self._modes.values():
                if button.isChecked():
                    button.setChecked(False)
        solver = self.panel.solver() if has_case else ""
        if not has_case:
            self.note.setText("Open a case to lay a multi-block grid in it.")
        elif topology is None:
            self.note.setText(
                "A multi-block grid: blocks of curvilinear cells that meet along their "
                "faces (G 50 1, SFLOW and NHFLOW). Start one block over the domain, Draw "
                "one, or make the river corridor block 0; then add blocks at its edges."
            )
        else:
            text = "Drag corners and points, cut edges and add blocks in the 2D view."
            if solver and solver not in ("SFLOW", "NHFLOW"):
                text += f" REEF3D::{solver} has no multi-block grids: SFLOW and NHFLOW do."
            self.note.setText(text)
        if topology is None:
            self.quality.setText("")
            self.rules.clear()
            self.block.clear()
            self.edge.clear()
            return
        self._fill(self.block, [f"{k}: {b.name}" for k, b in enumerate(topology.blocks)])
        self._fill(self.edge, list(topology.edges))
        self._show_block()
        self._show_edge()
        self._show_rules(topology)

    @staticmethod
    def _fill(box: QComboBox, names: list[str]) -> None:
        current = box.currentIndex()
        if [box.itemText(k) for k in range(box.count())] == names:
            return
        box.blockSignals(True)
        box.clear()
        box.addItems(names)
        box.setCurrentIndex(min(max(current, 0), len(names) - 1))
        box.blockSignals(False)

    def _show_block(self) -> None:
        topology = self.topology()
        k = self.block.currentIndex()
        if topology is None or not 0 <= k < len(topology.blocks):
            return
        block = topology.blocks[k]
        self.block_name.setText(block.name)
        self.smoothing.blockSignals(True)
        self.smoothing.setValue(block.smoothing)
        self.smoothing.blockSignals(False)
        ok = not topology.face_problems(k)
        for row_widget in self.faces.values():
            if ok:
                row_widget.show_face(topology, k)
            row_widget.setEnabled(ok)
        self.native_button.setEnabled(k > 0)
        self.remove_block_button.setEnabled(len(topology.blocks) > 1)

    def _show_edge(self) -> None:
        topology = self.topology()
        name = self.edge.currentText()
        if topology is None or name not in topology.edges:
            return
        edge = topology.edges[name]
        for widget in (self.cells, self.curve, self.ratio):
            widget.blockSignals(True)
        self.cells.setValue(edge.cells)
        self.curve.setCurrentIndex(self.curve.findData(edge.curve))
        self.ratio.setValue(edge.ratio)
        for widget in (self.cells, self.curve, self.ratio):
            widget.blockSignals(False)

    def _show_rules(self, topology) -> None:
        self.rules.clear()
        issues = mb.check(topology)
        for issue in issues:
            item = QListWidgetItem(
                ("Error: " if issue.level == "error" else "Warning: ") + issue.text
            )
            item.setForeground(
                QBrush(QColor(ERROR_COLOR if issue.level == "error" else WARNING_COLOR))
            )
            self.rules.addItem(item)
        if not issues:
            self.rules.addItem("No rule broken")
        errors = [i for i in issues if i.level == "error"]
        if errors:
            self.quality.setText(
                f"{mb.count(len(errors), 'rule')} broken: Make grid waits for them"
            )
            self.quality.setStyleSheet(f"color: {ERROR_COLOR};")
        else:
            self.quality.setText(mb.quality(topology).text())
            self.quality.setStyleSheet("")
        self.make_button.setEnabled(not errors)

    # ------------------------------------------------------------- edits
    def _run(self, action, done: str = "") -> bool:
        try:
            action()
        except (ValueError, KeyError, IndexError) as error:
            self.message.emit(str(error).strip("'\""), True)
            return False
        if done:
            self.message.emit(done, False)
        return True

    @property
    def case(self) -> str:
        return self.panel.case_id

    def start(self) -> None:
        if self._run(
            lambda: self.session.start_multiblock(case=self.case),
            "One block over the domain: add blocks at its edges (Add block)",
        ):
            self._modes["drag"].setChecked(True)

    def draw(self) -> None:
        if self.case:
            self.drawArea.emit(self.case, PURPOSE)

    def area_drawn(self, purpose: str, area) -> None:
        """An area dragged in the 2D view: the first block over it, or one more."""
        if purpose != PURPOSE or not self.case:
            return
        if self.topology() is None:
            self._run(lambda: self.session.start_multiblock(area, case=self.case), "A block")
        else:
            lx, ly = abs(area[1] - area[0]), abs(area[3] - area[2])
            step = max(lx, ly) / 20.0 or 1.0
            cells = (max(2, round(lx / step)), max(2, round(ly / step)))
            self._run(
                lambda: self.session.add_block(area, cells, case=self.case),
                "One more block: share its edges with the others in blocks.json, or start "
                "again with Add block at an edge",
            )

    def from_corridor(self) -> None:
        self._run(
            lambda: self.session.multiblock_from_corridor(case=self.case),
            "The river corridor is block 0: add blocks at its edges (Add block)",
        )

    def remove(self) -> None:
        for button in self._modes.values():
            button.setChecked(False)
        self._run(lambda: self.session.remove_multiblock(case=self.case))

    def _mode_toggled(self, mode: str, on: bool) -> None:
        if on:
            for other, button in self._modes.items():
                if other != mode and button.isChecked():
                    button.blockSignals(True)
                    button.setChecked(False)
                    button.blockSignals(False)
            if self.case:
                self.editBlocks.emit(self.case, mode)
        elif not any(b.isChecked() for b in self._modes.values()):
            self.stopEditing.emit()

    def editing_stopped(self) -> None:
        """The 3D view stopped the editing (Esc, another view)."""
        for button in self._modes.values():
            button.blockSignals(True)
            button.setChecked(False)
            button.blockSignals(False)

    def vertex_moved(self, case_id: str, name: str, x: float, y: float) -> None:
        if case_id and case_id in self.session.scene:
            self._run(
                lambda: self.session.move_block_vertex(name, round(x, 4), round(y, 4), case=case_id)
            )

    def point_moved(self, case_id: str, edge: str, index: int, x: float, y: float) -> None:
        if case_id and case_id in self.session.scene:
            self._run(
                lambda: self.session.move_block_point(
                    edge, index, round(x, 4), round(y, 4), case=case_id
                )
            )

    def edge_picked(self, case_id: str, mode: str, edge: str, x: float, y: float) -> None:
        """An edge clicked in the 2D view: cut it there, or add a block outside it."""
        if not case_id or case_id not in self.session.scene:
            return
        if mode == "cut":
            self._run(
                lambda: self.session.split_block_edge(edge, round(x, 4), round(y, 4), case=case_id),
                f"Edge {edge} cut in two",
            )
        elif mode == "extend":
            self._run(
                lambda: self.session.extrude_block(
                    edge, self.depth.value(), self.depth_cells.value(), case=case_id
                ),
                f"A block outside edge {edge}: drag its corners and points",
            )

    def _set_face(self, face: str, kind: str, value) -> None:
        k = self.block.currentIndex()
        if k < 0:
            return
        topology = self.topology()
        block = topology.blocks[k]
        if block.bc_type(face) == kind and (kind == "wall" or block.bc_value(face) == value):
            return
        self._run(lambda: self.session.set_block_face(k, face, kind, value, case=self.case))

    def _rename(self) -> None:
        k = self.block.currentIndex()
        topology = self.topology()
        name = self.block_name.text().strip()
        if k < 0 or topology is None or not name or name == topology.blocks[k].name:
            return
        self._run(lambda: self.session.set_block_options(k, name=name, case=self.case))

    def _commit_block(self) -> None:
        k = self.block.currentIndex()
        topology = self.topology()
        if k < 0 or topology is None:
            return
        if self.smoothing.value() != topology.blocks[k].smoothing:
            self._run(
                lambda: self.session.set_block_options(
                    k, smoothing=self.smoothing.value(), case=self.case
                )
            )

    def _commit_edge(self) -> None:
        topology = self.topology()
        name = self.edge.currentText()
        if topology is None or name not in topology.edges:
            return
        edge = topology.edges[name]
        options = {
            "cells": self.cells.value(),
            "curve": self.curve.currentData(),
            "ratio": round(self.ratio.value(), 6),
        }
        changed = {k: v for k, v in options.items() if getattr(edge, k) != v}
        if changed:
            self._run(lambda: self.session.set_block_edge(name, case=self.case, **changed))

    def scale(self, factor: float) -> None:
        self._run(
            lambda: self.session.scale_multiblock(factor, case=self.case),
            "Every edge twice as fine" if factor > 1 else "Every edge half as fine",
        )

    def make_native(self) -> None:
        k = self.block.currentIndex()
        if k > 0 and self._run(lambda: self.session.make_native_block(k, case=self.case)):
            self.block.setCurrentIndex(0)

    def remove_block(self) -> None:
        k = self.block.currentIndex()
        if k >= 0:
            self._run(lambda: self.session.remove_block(k, case=self.case))

    def make_grid(self) -> None:
        quality = None

        def make():
            nonlocal quality
            quality = self.session.make_multiblock_grid(case=self.case)

        if self._run(make) and quality is not None:
            self.message.emit(
                f"blocks.dat written ({mb.count(len(quality.blocks), 'block')}, "
                f"{mb.count(quality.cells, 'cell')}), "
                "R 20 1, G 50 1 and the open faces set: Save, then Run DIVEMesh",
                False,
            )
