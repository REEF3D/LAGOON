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
  (a circular arc too) and its grading, or its *spacing*: the cells at its ends (and
  a largest one), the cells following from its length, across the blocks; *Finer*
  and *Coarser* double or halve every edge's cells. *Pick edge* chooses it in the 2D
  view.
- *Along the water*: the edge laid on land a margin from the water below the level
  (F 60), bending no tighter than a radius (needs geo.dat); moving its corners
  lays it again.
- *Import blocks.dat*: a grid made elsewhere (its faces' nodes kept) to go on with.
- *Attach river*: a river block on a part of the chosen edge, along the river drawn
  in the River tab, its discharge in at its upstream face.
- *Tide*: block 0's open sea face as a Riemann edge (B 520) driven by the tide:
  from tidal constituents, a tide gauge or a file of levels, as background-1.dat
  (or B 511), *Write the tide*; *Make grid* writes the edge's zone.
- *Pieces*: a block written as several blocks (each runs on one process);
  *Balance* cuts the blocks for M 10's processes, and the tab says how even the
  processes' shares are.
- *Rules* lists what DIVEMesh and REEF3D would refuse (red) or warn about, the set-up
  marks them where they are, with the cells' quality; *Make grid* writes blocks.dat,
  R 20 1, the sides of block 0, G 50 1 and the G 54 lines of the open faces.

Every change is an undo step; blocks.json and blocks.dat are written at once.
"""

from __future__ import annotations

from PySide6.QtCore import QDateTime, Qt, Signal
from PySide6.QtGui import QBrush, QColor
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDateTimeEdit,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QPlainTextEdit,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from lagoonviz.core import multiblock as mb

#: the purpose of an area drawn in the 2D view for this tab
PURPOSE = "block"
BC_LABELS = {
    "wall": "Wall",
    "inflow": "Inflow (discharge)",
    "outflow": "Outflow (water level)",
    "open": "Open (its own condition in ctrl.txt)",
    "tide": "Tide (Riemann edge, B 520: the Tide box's background)",
}
CURVE_LABELS = {
    "straight": "Straight",
    "through": "Through the points (Catmull-Rom)",
    "bspline": "Smooth, pulled by the points (B-spline)",
    "arc": "Circular arc through its point",
    "nodes": "Through its nodes (imported)",
}
TIDE_SOURCES = {
    "constants": "Tidal constituents",
    "observed": "A tide gauge, observed",
    "predicted": "A tide gauge, predicted",
    "levels": "A file of levels",
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
        self.value.setEnabled(kind in ("inflow", "outflow"))
        self.value.setSuffix(" m3/s" if kind == "inflow" else " m")
        self.kind.setEnabled(len(shared) < len(chain))  # not on a face all interface
        for widget in (self.kind, self.value):
            widget.blockSignals(False)

    def _emit(self) -> None:
        kind = self.kind.currentData()
        valued = kind in ("inflow", "outflow")
        self.value.setEnabled(valued)
        self.changed.emit(self.face, kind, self.value.value() if valued else None)


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
        self.import_button = QPushButton("Import blocks.dat")
        self.import_button.setToolTip(
            "The case's blocks.dat (made elsewhere) as its blocks: the faces keep their "
            "nodes, the interfaces where faces share nodes, the boundaries from control.txt "
            "and ctrl.txt; the nodes' bed as geo.dat when there is none"
        )
        self.import_button.clicked.connect(self.import_blocks)
        row = QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        for button in (
            self.start_button,
            self.draw_button,
            self.corridor_button,
            self.import_button,
            self.remove_button,
        ):
            row.addWidget(button)
        row.addStretch(1)

        # ---- editing in the 2D view
        self.drag_button = QPushButton("Drag points")
        self.cut_button = QPushButton("Cut edge")
        self.extend_button = QPushButton("Add block")
        self.pick_button = QPushButton("Pick edge")
        tips = (
            "Drag the corners and the points of the edges in the 2D top view (Esc stops)",
            "Click an edge in the 2D top view to cut it in two at the node nearest the "
            "click: a block can then meet a part of a face",
            "Click an edge on the outline in the 2D top view: a block outside it, Depth "
            "deep with Cells cells",
            "Click an edge in the 2D top view to choose it under Edge",
        )
        self._modes = {}
        for button, mode, tip in zip(
            (self.drag_button, self.cut_button, self.extend_button, self.pick_button),
            ("drag", "cut", "extend", "pick"),
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
        self.pieces = QSpinBox()
        self.pieces.setRange(1, 999)
        self.pieces.setKeyboardTracking(False)
        self.pieces.setToolTip(
            "Written as this many blocks (equal parts): REEF3D runs each on one process. "
            "Block 0 is shared out over the processes by DIVEMesh itself"
        )
        self.split = QComboBox()
        self.split.addItem("along i", "i")
        self.split.addItem("along j", "j")
        self.split.setToolTip("Which way the block is cut into its pieces")
        self.pieces.valueChanged.connect(lambda _v: self._commit_pieces())
        self.split.activated.connect(lambda _i: self._commit_pieces())
        self.faces = {face: FaceRow(face) for face in mb.FACES}
        for row_widget in self.faces.values():
            row_widget.changed.connect(self._set_face)
        block_box = QGroupBox("Block")
        block_form = QFormLayout(block_box)
        block_form.addRow("Block", self.block)
        block_form.addRow("Name", self.block_name)
        block_form.addRow("Smoothing", self.smoothing)
        cut = QHBoxLayout()
        cut.addWidget(self.pieces)
        cut.addWidget(self.split)
        cut.addStretch(1)
        block_form.addRow("Pieces", cut)
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
        # spacing: the cells at the ends, the count from the length
        self.spaced = QCheckBox("By the cells at its ends")
        self.spaced.setToolTip(
            "Space the nodes by the length of the cells at the edge's ends (growing evenly "
            "from one to the other, or up to a largest cell); its cells follow from its "
            "length and go on across the blocks"
        )
        self.space_start = _metres("The first cell's length, at the edge's start")
        self.space_end = _metres("The last cell's length, at the edge's end")
        self.space_max = _metres(
            "The largest cell (none: growing evenly from the start to the end); the cells "
            "grow by Growth from each end up to it"
        )
        self.space_max.setMinimum(0.0)
        self.space_max.setSpecialValueText("none")
        self.growth = QDoubleSpinBox()
        self.growth.setRange(1.001, 2.0)
        self.growth.setDecimals(3)
        self.growth.setSingleStep(0.01)
        self.growth.setValue(mb.GROWTH)
        self.growth.setKeyboardTracking(False)
        self.growth.setToolTip("How much a cell grows on its neighbour towards the largest")
        self.space_across = QCheckBox("And the edges across the blocks")
        self.space_across.setChecked(True)
        self.space_across.setToolTip(
            "The edges across the blocks from it (whose cells go with its) get the same "
            "spacing, turned their way, unless they have one"
        )
        self.spaced.toggled.connect(lambda _on: self._commit_spacing())
        for box in (self.space_start, self.space_end, self.space_max, self.growth):
            box.valueChanged.connect(lambda _v: self._commit_spacing())
        self.sizes = QLabel()
        self.sizes.setWordWrap(True)
        self.fit_button = QPushButton("Cells from spacing")
        self.fit_button.setToolTip(
            "Every edge with a spacing gets the cells it asks for over its length as it is "
            "now (after corners were moved)"
        )
        self.fit_button.clicked.connect(self.fit_cells)
        # along the water
        self.level = QDoubleSpinBox()
        self.level.setRange(-1e6, 1e6)
        self.level.setDecimals(3)
        self.level.setKeyboardTracking(False)
        self.level.setToolTip("The water below this level (default F 60) is the water followed")
        self.margin = _metres("How far onto the land from the water the edge keeps")
        self.margin.setMinimum(0.0)
        self.radius = _metres(
            "The edge bends no tighter than this: land spits and inlets narrower than "
            "twice it are passed over"
        )
        self.radius.setMinimum(0.0)
        self.snap = QCheckBox("Its corners onto the shore")
        self.snap.setChecked(True)
        self.snap.setToolTip(
            "Move the edge's two corners to the shore's line; off: they stay, the edge "
            "joining the shore from them"
        )
        self.follow_button = QPushButton("Follow the water")
        self.follow_button.setToolTip(
            "Lay the edge along the water, between the points of the shore nearest its "
            "corners, the way round nearer to where it is (needs geo.dat)"
        )
        self.follow_button.clicked.connect(self.follow_water)
        self.unfollow_button = QPushButton("Stop following")
        self.unfollow_button.setToolTip("Keep the edge as it is, shaped by its points")
        self.unfollow_button.clicked.connect(self.stop_following)
        self.water_info = QLabel()
        self.water_info.setWordWrap(True)
        # a river onto a part of the edge
        self.river_width = _metres("How wide the river block is (its cells across from the edge's)")
        self.river_cells = QSpinBox()
        self.river_cells.setRange(0, 10_000)
        self.river_cells.setSpecialValueText("from the width")
        self.river_cells.setToolTip("Cells across the river: that many of the edge's cells")
        self.river_step = _metres("How long the river's cells are along it (none: twice across)")
        self.river_step.setMinimum(0.0)
        self.river_step.setSpecialValueText("twice across")
        self.discharge = QDoubleSpinBox()
        self.discharge.setRange(0.0, 1e6)
        self.discharge.setDecimals(3)
        self.discharge.setSuffix(" m3/s")
        self.discharge.setSpecialValueText("none")
        self.discharge.setToolTip("The discharge in at the river's upstream face (G 54 b 1 Q)")
        self.river_button = QPushButton("Attach river")
        self.river_button.setToolTip(
            "A river block on a part of the chosen edge (on the outline): along the river "
            "drawn in the River tab (Draw the river), else the corridor's line"
        )
        self.river_button.clicked.connect(self.attach_river)
        # the tide at the open sea
        self.tide_source = QComboBox()
        for key, label in TIDE_SOURCES.items():
            self.tide_source.addItem(label, key)
        self.tide_source.currentIndexChanged.connect(lambda _i: self._tide_rows())
        self.constants = QPlainTextEdit()
        self.constants.setPlaceholderText(
            "one constituent a line: name, amplitude (m), Greenwich phase (deg)\n"
            "M2 1.00 61\nS2 0.35 89\nN2 0.21 44\nK1 0.07 52\nO1 0.06 310"
        )
        self.constants.setMaximumHeight(110)
        self.gauge = QComboBox()
        self.gauge.setEditable(True)
        self.gauge.setToolTip('A tide gauge as "SOURCE:CODE", as Find lists them')
        self.find_button = QPushButton("Find")
        self.find_button.setToolTip("The tide gauges nearest to the case (needs its georeference)")
        self.find_button.clicked.connect(self.find_gauges)
        self.levels_file = QLineEdit()
        self.levels_file.setPlaceholderText("time level: seconds, or date and time (UTC)")
        browse = QPushButton("…")
        browse.setMaximumWidth(30)
        browse.clicked.connect(self._choose_levels)
        self.levels_browse = browse
        self.start = QDateTimeEdit()
        self.start.setTimeSpec(Qt.UTC)
        self.start.setDisplayFormat("yyyy-MM-dd HH:mm 'UTC'")
        self.start.setCalendarPopup(True)
        self.start.setDateTime(QDateTime.currentDateTimeUtc().addDays(-30))
        self.start.setToolTip("When the run starts (t = 0)")
        self.days = QDoubleSpinBox()
        self.days.setRange(0.01, 3650.0)
        self.days.setValue(3.0)
        self.days.setSuffix(" days")
        self.tide_step = QDoubleSpinBox()
        self.tide_step.setRange(1.0, 86400.0)
        self.tide_step.setValue(300.0)
        self.tide_step.setSuffix(" s")
        self.tide_step.setToolTip("A level every so many seconds in background-1.dat")
        self.ramp = QDoubleSpinBox()
        self.ramp.setRange(0.0, 1e7)
        self.ramp.setValue(10800.0)
        self.ramp.setSuffix(" s")
        self.ramp.setToolTip("The tide rises from nothing over this time at the start (B 510)")
        self.harmonic = QCheckBox("As constituents in REEF3D (B 511)")
        self.harmonic.setToolTip(
            "REEF3D sums the constituents itself (the nodal corrections of the start "
            "kept over the run), instead of reading background-1.dat"
        )
        self.current = QCheckBox("With the long-wave current")
        self.current.setToolTip(
            "Two columns (t eta): REEF3D adds the current eta sqrt(g/h) along the "
            "direction; off: none (t eta 0 0), right for a standing coastal tide"
        )
        self.travels = QCheckBox("Travels along the coast (B 515)")
        self.travels.setToolTip(
            "The tide holds at the reference point and travels along the direction at "
            "sqrt(g depth): a tide running along a coast past the open side"
        )
        self.direction = QDoubleSpinBox()
        self.direction.setRange(-360.0, 360.0)
        self.direction.setSuffix(" °")
        self.direction.setToolTip("Where the tide travels, from the model's x axis (B 510)")
        self.ref_x = _metres("Where the tide holds: x")
        self.ref_y = _metres("Where the tide holds: y")
        for box in (self.ref_x, self.ref_y):
            box.setRange(-1e9, 1e9)
        self.ref_depth = _metres("The depth whose long-wave speed the tide travels at")
        self.ref_depth.setValue(100.0)
        self.tide_button = QPushButton("Write the tide")
        self.tide_button.setToolTip(
            "background-1.dat and B 510 (B 511, B 515) in ctrl.txt: background 1, which "
            "block 0's tide faces take"
        )
        self.tide_button.clicked.connect(self.write_tide)
        self.tide_info = QLabel()
        self.tide_info.setWordWrap(True)
        # the processes
        self.balance_button = QPushButton("Balance")
        self.balance_button.setToolTip(
            "Cut the blocks 1.. into pieces so that M 10's processes get about as many "
            "cells each (block 0 is shared out by DIVEMesh, each piece runs on one)"
        )
        self.balance_button.clicked.connect(self.balance)
        self.balance_info = QLabel()
        self.balance_info.setWordWrap(True)
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
        edge_form.addRow("Spacing", self.spaced)
        edge_form.addRow("At the start", self.space_start)
        edge_form.addRow("At the end", self.space_end)
        edge_form.addRow("Largest", self.space_max)
        edge_form.addRow("Growth", self.growth)
        edge_form.addRow("", self.space_across)
        edge_form.addRow("", self.sizes)
        scale = QHBoxLayout()
        scale.addWidget(self.finer_button)
        scale.addWidget(self.coarser_button)
        scale.addWidget(self.fit_button)
        scale.addStretch(1)
        edge_form.addRow("All edges", scale)
        water_box = QGroupBox("Along the water")
        water_form = QFormLayout(water_box)
        water_form.addRow("Water below", self.level)
        water_form.addRow("Margin", self.margin)
        water_form.addRow("Bend radius", self.radius)
        water_form.addRow("", self.snap)
        follow = QHBoxLayout()
        follow.addWidget(self.follow_button)
        follow.addWidget(self.unfollow_button)
        follow.addStretch(1)
        water_form.addRow(follow)
        water_form.addRow(self.water_info)
        self.water_box = water_box
        river_box = QGroupBox("River")
        river_form = QFormLayout(river_box)
        river_form.addRow("Width", self.river_width)
        river_form.addRow("Cells across", self.river_cells)
        river_form.addRow("Cell length", self.river_step)
        river_form.addRow("Discharge", self.discharge)
        river_form.addRow("", self.river_button)
        self.river_width.setValue(200.0)
        tide_box = QGroupBox("Tide")
        tide_form = self.tide_form = QFormLayout(tide_box)
        tide_form.addRow("From", self.tide_source)
        tide_form.addRow("Constituents", self.constants)
        gauge_row = QHBoxLayout()
        gauge_row.addWidget(self.gauge, 1)
        gauge_row.addWidget(self.find_button)
        self.gauge_row = QWidget()
        self.gauge_row.setLayout(gauge_row)
        tide_form.addRow("Tide gauge", self.gauge_row)
        file_row = QHBoxLayout()
        file_row.addWidget(self.levels_file, 1)
        file_row.addWidget(self.levels_browse)
        self.file_row = QWidget()
        self.file_row.setLayout(file_row)
        tide_form.addRow("Levels", self.file_row)
        tide_form.addRow("Start", self.start)
        tide_form.addRow("Length", self.days)
        tide_form.addRow("Every", self.tide_step)
        tide_form.addRow("Ramp", self.ramp)
        tide_form.addRow("", self.harmonic)
        tide_form.addRow("", self.current)
        tide_form.addRow("", self.travels)
        tide_form.addRow("Direction", self.direction)
        tide_form.addRow("Holds at x", self.ref_x)
        tide_form.addRow("Holds at y", self.ref_y)
        tide_form.addRow("At the speed of", self.ref_depth)
        tide_form.addRow("", self.tide_button)
        tide_form.addRow(self.tide_info)
        self.travels.toggled.connect(lambda _on: self._tide_rows())
        self._tide_rows()

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
        balance_row = QHBoxLayout()
        balance_row.addWidget(self.balance_button)
        balance_row.addWidget(self.balance_info, 1)
        grid_form.addRow("Processes", balance_row)
        grid_form.addRow("", self.make_button)

        body = QWidget()
        column = QVBoxLayout(body)
        column.setContentsMargins(0, 0, 0, 0)
        for box in (edit_box, block_box, edge_box, water_box, river_box, tide_box, grid_box):
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
        self.import_button.setEnabled(self._has_file(mb.GRID_FILE))
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
        for widget in (self.smoothing, self.pieces, self.split):
            widget.blockSignals(True)
        self.smoothing.setValue(block.smoothing)
        self.pieces.setValue(block.pieces)
        self.split.setCurrentIndex(self.split.findData(block.split))
        for widget in (self.smoothing, self.pieces, self.split):
            widget.blockSignals(False)
        self.pieces.setEnabled(k > 0)
        self.split.setEnabled(k > 0)
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
        widgets = (
            self.cells,
            self.curve,
            self.ratio,
            self.spaced,
            self.space_start,
            self.space_end,
            self.space_max,
            self.growth,
            self.margin,
            self.radius,
            self.level,
            self.snap,
        )
        for widget in widgets:
            widget.blockSignals(True)
        self.cells.setValue(edge.cells)
        self.curve.setCurrentIndex(self.curve.findData(edge.curve))
        self.ratio.setValue(edge.ratio)
        first, last = edge.end_sizes(topology.vertices)
        length = edge.length(topology.vertices)
        spacing = edge.spacing or {}
        self.spaced.setChecked(bool(spacing))
        self.space_start.setValue(spacing.get("start", mb.curvi.nice_step(first)))
        self.space_end.setValue(spacing.get("end", mb.curvi.nice_step(last)))
        self.space_max.setValue(spacing.get("max", 0.0))
        self.growth.setValue(spacing.get("growth", mb.GROWTH))
        for box in (self.space_start, self.space_end, self.space_max, self.growth):
            box.setEnabled(bool(spacing))
        self.ratio.setEnabled(not spacing)
        self.cells.setEnabled(not spacing)
        text = (
            f"{length:,.0f} m long; cells now {first:.3g} m at the start, {last:.3g} m at the end"
        )
        wanted = edge.wanted_cells(topology.vertices)
        if wanted is not None and wanted != edge.cells:
            text += f" (its spacing asks for {wanted} cells: Cells from spacing)"
        self.sizes.setText(text)
        water = edge.water or {}
        self.level.setValue(water.get("level", self._water_level()))
        self.margin.setValue(water.get("margin", mb.curvi.nice_step(0.02 * length)))
        self.radius.setValue(water.get("radius", mb.curvi.nice_step(0.05 * length)))
        self.snap.setChecked(water.get("corners", True))
        for widget in widgets:
            widget.blockSignals(False)
        geo = self._has_file("geo.dat")
        self.follow_button.setEnabled(geo)
        self.unfollow_button.setEnabled(bool(water))
        if water:
            self.water_info.setText(
                f"{name} follows the water: {water['margin']:g} m onto the land, bending no "
                f"tighter than {water['radius']:g} m; moving its corners lays it again"
            )
        elif not geo:
            self.water_info.setText("Following the water needs the bed: geo.dat (Map tab)")
        else:
            self.water_info.setText(f"{name} is shaped by its points")

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
        try:
            processes = self.session.processes(self.case)
            self.balance_info.setText(
                mb.balance(topology, processes).text() if not errors else f"M 10: {processes}"
            )
        except (ValueError, KeyError):
            self.balance_info.setText("")

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
        if mode == "pick":
            index = self.edge.findText(edge)
            if index >= 0:
                self.edge.setCurrentIndex(index)
            self.message.emit(f"Edge {edge}", False)
        elif mode == "cut":
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

    def _has_file(self, name: str) -> bool:
        if not self.case:
            return False
        folder = self.session.data.case_folder(self.case)
        return folder is not None and (folder / name).is_file()

    def _water_level(self) -> float:
        try:
            return self.session.water_level(self.case) if self.case else 0.0
        except ValueError:
            return 0.0

    def _commit_spacing(self) -> None:
        topology = self.topology()
        name = self.edge.currentText()
        if topology is None or name not in topology.edges:
            return
        edge = topology.edges[name]
        if not self.spaced.isChecked():
            if edge.spacing:
                self._run(lambda: self.session.set_block_spacing(name, None, case=self.case))
            return
        largest = self.space_max.value() or None
        wanted = {"start": self.space_start.value(), "end": self.space_end.value()}
        if largest:
            wanted |= {"max": largest, "growth": round(self.growth.value(), 6)}
        if edge.spacing == wanted:
            return
        self._run(
            lambda: self.session.set_block_spacing(
                name,
                wanted["start"],
                wanted["end"],
                largest=largest,
                growth=wanted.get("growth"),
                across=self.space_across.isChecked(),
                case=self.case,
            )
        )

    def _commit_pieces(self) -> None:
        k = self.block.currentIndex()
        topology = self.topology()
        if k <= 0 or topology is None:
            return
        block = topology.blocks[k]
        pieces, split = self.pieces.value(), self.split.currentData()
        if (block.pieces, block.split) != (pieces, split):
            self._run(
                lambda: self.session.set_block_options(
                    k, pieces=pieces, split=split, case=self.case
                )
            )

    def balance(self) -> None:
        self._run(
            lambda: self.session.balance_blocks(case=self.case),
            f"The blocks cut for {self.session.processes(self.case)} processes (M 10)",
        )

    def attach_river(self) -> None:
        name = self.edge.currentText()
        if not name:
            return
        cells = self.river_cells.value() or None
        self._run(
            lambda: self.session.attach_river(
                name,
                cells=cells,
                width=None if cells else self.river_width.value(),
                step=self.river_step.value() or None,
                discharge=self.discharge.value() or None,
                case=self.case,
            ),
            f"A river block on edge {name}: drag its points, or change its cells",
        )

    def _tide_rows(self) -> None:
        source = self.tide_source.currentData()
        layout = self.tide_form
        layout.setRowVisible(self.constants, source == "constants")
        layout.setRowVisible(self.gauge_row, source in ("observed", "predicted"))
        layout.setRowVisible(self.file_row, source == "levels")
        layout.setRowVisible(self.harmonic, source == "constants")
        for widget in (self.ref_x, self.ref_y, self.ref_depth):
            layout.setRowVisible(widget, self.travels.isChecked())

    def _choose_levels(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Water levels", "", "Levels (*.txt *.dat *.csv);;All (*)"
        )
        if path:
            self.levels_file.setText(path)

    def find_gauges(self) -> None:
        found = []

        def find():
            found.extend(self.session.find_gauges(self.case))

        if self._run(find, ""):
            self.gauge.clear()
            for station in found:
                self.gauge.addItem(
                    f"{station.source}:{station.code} {station.describe()}",
                    f"{station.source}:{station.code}",
                )
            self.message.emit(f"{len(found)} tide gauges near the case", False)

    def write_tide(self) -> None:
        source = self.tide_source.currentData()
        start = self.start.dateTime().toUTC().toString(Qt.ISODate)
        options = {
            "days": self.days.value(),
            "step": self.tide_step.value(),
            "ramp": self.ramp.value(),
            "direction": self.direction.value(),
            "current": self.current.isChecked(),
        }
        if self.travels.isChecked():
            options |= {
                "reference": (self.ref_x.value(), self.ref_y.value()),
                "depth": self.ref_depth.value(),
            }
        if source == "constants":
            options |= {
                "constants": self.constants.toPlainText(),
                "harmonic": self.harmonic.isChecked(),
            }
        elif source == "levels":
            options["levels"] = self.levels_file.text().strip()
        else:
            gauge = self.gauge.currentData() or self.gauge.currentText().split()[0]
            options |= {"gauge": gauge, "predicted": source == "predicted"}
        found = []

        def write():
            found.append(self.session.write_tide(start, case=self.case, **options))

        if self._run(write) and found:
            self.tide_info.setText(found[0].text())
            self.message.emit(
                "The tide is background 1: a tide face of block 0 (Face -i: Tide) takes it "
                "at Make grid",
                False,
            )

    def fit_cells(self) -> None:
        self._run(
            lambda: self.session.fit_block_cells(case=self.case),
            "The edges with a spacing have the cells it asks for",
        )

    def follow_water(self) -> None:
        name = self.edge.currentText()
        if not name:
            return
        level = self.level.value()
        given = None if abs(level - self._water_level()) < 1e-9 else level
        self._run(
            lambda: self.session.follow_water(
                name,
                margin=self.margin.value(),
                radius=self.radius.value(),
                level=given,
                corners=self.snap.isChecked(),
                case=self.case,
            ),
            f"{name} follows the water: drag its corners along the shore",
        )

    def stop_following(self) -> None:
        name = self.edge.currentText()
        if name:
            self._run(lambda: self.session.set_block_edge(name, water=None, case=self.case))

    def import_blocks(self) -> None:
        count = 0

        def read():
            nonlocal count
            count = self.session.import_blocks(case=self.case)

        if self._run(read):
            self.message.emit(
                f"{mb.count(count, 'block')} read from blocks.dat: the faces keep their "
                "nodes; change them here, then Make grid",
                False,
            )

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


def _metres(tip: str) -> QDoubleSpinBox:
    box = QDoubleSpinBox()
    box.setRange(0.001, 1e7)
    box.setDecimals(3)
    box.setSuffix(" m")
    box.setKeyboardTracking(False)
    box.setToolTip(tip)
    return box
