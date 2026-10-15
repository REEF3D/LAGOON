# SPDX-License-Identifier: GPL-3.0-or-later
# Architect: Hans Bihs
"""The Porous tab of the set-up: the porous structures (VRANS) of a case, added,
drawn, changed and removed as the ctrl.txt lines that make them.

REEF3D::CFD: boxes (B 270), vertical cylinders (B 274), wedges along x and y
(B 281, B 282) and plates (B 291), each with its own material (porosity n, d50,
α, β). REEF3D::NHFLOW: boxes, cylinders, spheres and wedges (B 210 - B 218) of one
material for all (B 201). *Add* puts one in the middle of the domain, from the bed
to the still water; *Draw* drags its plan in the 2D top view of the 3D view. The
selected one lights up in the 3D view; its values change in the form below. Every
change is an undo step, saved with the input files (``core.porous``).
"""

from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QComboBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QScrollArea,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from lagoonviz.core import porous

#: the purpose of an area drawn in the 2D view for this tab
PURPOSE = "porous structure"
KEY_ROLE = Qt.UserRole
FILE = "ctrl.txt"


class PorousTab(QWidget):
    """The Porous tab of the set-up panel (module doc)."""

    #: draw an area in the 2D top view: (case id, purpose)
    drawArea = Signal(str, str)
    #: a message for the panel's status line: (text, error)
    message = Signal(str, bool)

    def __init__(self, panel, parent=None) -> None:
        super().__init__(parent)
        self.panel = panel
        self.session = panel.session
        #: (code, occurrence) of the selected structure, or None
        self._key: tuple[str, int] | None = None
        self._solver = None
        #: the kind a drawing in the 2D view is for
        self._drawing_kind = ""
        self._shape_fields: list[QLineEdit] = []
        self._material_fields: list[QLineEdit] = []

        self.note = QLabel()
        self.note.setWordWrap(True)

        self.kind_box = QComboBox()
        self.kind_box.setToolTip("The kind of porous structure to add")
        self.add_button = QPushButton("Add")
        self.add_button.setToolTip(
            "Add one in the middle of the domain, from the bed to the still water"
        )
        self.add_button.clicked.connect(lambda: self.add())
        self.draw_button = QPushButton("Draw")
        self.draw_button.setToolTip(
            "Drag its plan in the 2D top view of the 3D view (Esc stops); it reaches "
            "from the bed to the still water"
        )
        self.draw_button.clicked.connect(self.draw)
        self.remove_button = QPushButton("Remove")
        self.remove_button.setToolTip("Remove the selected structure from ctrl.txt")
        self.remove_button.clicked.connect(self.remove_selected)
        row = QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        row.addWidget(self.kind_box, 1)
        row.addWidget(self.add_button)
        row.addWidget(self.draw_button)
        row.addWidget(self.remove_button)

        self.tree = QTreeWidget()
        self.tree.setObjectName("porousStructures")
        self.tree.setHeaderLabels(["Structure", "Where"])
        self.tree.setRootIsDecorated(False)
        self.tree.setUniformRowHeights(True)
        self.tree.setMinimumHeight(110)
        self.tree.itemSelectionChanged.connect(self._on_select)

        self.shape_box = QGroupBox("Shape")
        self.shape_form = QFormLayout(self.shape_box)
        self.material_box = QGroupBox("Material")
        self.material_form = QFormLayout(self.material_box)
        self.material_note = QLabel()
        self.material_note.setWordWrap(True)
        self.material_note.setEnabled(False)
        self.material_form.addRow(self.material_note)
        for name in porous.MATERIAL:
            field = QLineEdit()
            field.editingFinished.connect(self._commit_material)
            self._material_fields.append(field)
            self.material_form.addRow(name, field)
        tips = (
            "Porosity: the share of the volume that is water (0 - 1; 0.4 for rubble)",
            "Mean diameter of the stones in m",
            "Linear (laminar) resistance coefficient α of the Forchheimer terms (500)",
            "Nonlinear (turbulent) resistance coefficient β of the Forchheimer terms (2.0)",
        )
        for field, tip in zip(self._material_fields, tips, strict=True):
            field.setToolTip(tip)

        body = QWidget()
        column = QVBoxLayout(body)
        column.setContentsMargins(0, 0, 0, 0)
        column.addWidget(self.shape_box)
        column.addWidget(self.material_box)
        column.addStretch(1)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(body)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.addWidget(self.note)
        layout.addLayout(row)
        layout.addWidget(self.tree, 2)
        layout.addWidget(scroll, 3)

    # ------------------------------------------------------------- the case
    def control(self):
        case = self.panel.case_id
        if not case or case not in self.session.scene:
            return None
        try:
            return self.session.input_file(FILE, case)
        except (OSError, ValueError):
            return None

    def kinds(self) -> list[porous.PorousKind]:
        return porous.kinds_for(self.panel.solver())

    def refresh(self) -> None:
        """The structures of the case again from ctrl.txt, the selection kept."""
        control = self.control()
        solver = self.panel.solver() if control is not None else ""
        kinds = porous.kinds_for(solver)
        if solver != self._solver:
            self._solver = solver
            self.kind_box.clear()
            for kind in kinds:
                self.kind_box.addItem(kind.label(), kind.code)
        if control is None:
            self.note.setText("Open a case to add porous structures to it.")
        elif not kinds:
            module = solver or "the module"
            self.note.setText(
                f"REEF3D::{module} has no porous structures: they are for CFD (B 270 - "
                "B 291) and NHFLOW (B 210 - B 218). Set the module with A 10 in ctrl.txt."
            )
        elif solver == "NHFLOW":
            self.note.setText(
                "Porous structures of REEF3D::NHFLOW (VRANS): one material for all of them (B 201)."
            )
        else:
            self.note.setText(
                "Porous structures of REEF3D::CFD (VRANS), each with its own material."
            )
        for widget in (self.kind_box, self.add_button, self.draw_button):
            widget.setEnabled(bool(kinds))

        found = porous.structures(control, solver) if control is not None and kinds else []
        self.tree.blockSignals(True)
        self.tree.clear()
        chosen = None
        for structure in found:
            item = QTreeWidgetItem([structure.label(), structure.kind.describe(structure.values)])
            key = (structure.kind.code, structure.occurrence)
            item.setData(0, KEY_ROLE, key)
            item.setToolTip(0, " ".join(f"{v:g}" for v in structure.values))
            self.tree.addTopLevelItem(item)
            if key == self._key:
                chosen = item
        self.tree.resizeColumnToContents(0)
        if chosen is not None:
            self.tree.setCurrentItem(chosen)
        else:
            self._key = None
        self.tree.blockSignals(False)
        self._show()

    def selected(self) -> porous.Structure | None:
        control = self.control()
        if self._key is None or control is None:
            return None
        code, occurrence = self._key
        lines = control.entries(code)
        if occurrence >= len(lines):
            return None
        return porous.Structure(
            porous.BY_CODE[code], occurrence, tuple(lines[occurrence].numbers() or ())
        )

    # ------------------------------------------------------------- the form
    def _show(self) -> None:
        """The selected structure's shape and material in the form."""
        structure = self.selected()
        self.remove_button.setEnabled(structure is not None)
        kind = structure.kind if structure is not None else None
        names = kind.geometry if kind is not None else ()
        if [self.shape_form.labelForField(f).text() for f in self._shape_fields] != list(names):
            while self.shape_form.rowCount():
                self.shape_form.removeRow(0)
            self._shape_fields = []
            for name in names:
                field = QLineEdit()
                field.editingFinished.connect(self._commit_shape)
                self._shape_fields.append(field)
                self.shape_form.addRow(name, field)
        self.shape_box.setTitle(f"Shape · {kind.label()}" if kind is not None else "Shape")
        self.shape_box.setVisible(kind is not None)
        geometry, material = kind.split(structure.values) if kind is not None else ([], [])
        for field, value in zip(self._shape_fields, geometry, strict=False):
            if not field.hasFocus():
                field.setText(f"{value:g}")

        nhflow = self.panel.solver() == "NHFLOW"
        control = self.control()
        if nhflow and control is not None:
            material = control.values(porous.NHFLOW_MATERIAL) or []
            self.material_box.setTitle("Material of all porous structures (B 201)")
            self.material_note.setText(
                ""
                if material
                else "No B 201 yet: REEF3D takes n = 1 (no porous media). "
                "Adding a structure adds B 201 too."
            )
            show = bool(material) or structure is not None
        else:
            self.material_box.setTitle("Material")
            self.material_note.setText("")
            show = structure is not None
        self.material_note.setVisible(bool(self.material_note.text()))
        self.material_box.setVisible(show)
        for index, field in enumerate(self._material_fields):
            if field.hasFocus():
                continue
            field.setText(f"{material[index]:g}" if index < len(material) else "")
            field.setPlaceholderText(f"{porous.DEFAULT_MATERIAL[index]:g}")

    @staticmethod
    def _numbers(fields) -> list[float] | None:
        try:
            return [float(f.text()) for f in fields]
        except ValueError:
            return None

    def _commit_shape(self) -> None:
        structure = self.selected()
        if structure is None:
            return
        geometry = self._numbers(self._shape_fields)
        if geometry is None:
            self.message.emit(f"{structure.kind.code}: the values must be numbers", True)
            return
        if "radius [m]" in structure.kind.geometry and geometry[-1] <= 0:
            self.message.emit(f"{structure.kind.code}: the radius must be larger than 0", True)
            return
        _, material = structure.kind.split(structure.values)
        values = structure.kind.join(geometry, material or None)
        if list(values) != list(structure.values):
            self.session.set_input(
                FILE, structure.kind.code, values, structure.occurrence, case=self.panel.case_id
            )

    def _commit_material(self) -> None:
        material = self._numbers(self._material_fields)
        if material is None:
            if any(f.text().strip() for f in self._material_fields):
                self.message.emit("The material's values must be numbers", True)
            return
        n, d50 = material[0], material[1]
        if not 0.0 < n <= 1.0 or d50 <= 0.0:
            self.message.emit("The porosity must be in (0, 1] and d50 larger than 0", True)
            return
        case = self.panel.case_id
        if self.panel.solver() == "NHFLOW":
            control = self.control()
            if control is not None and control.values(porous.NHFLOW_MATERIAL) != material:
                self.session.set_input(FILE, porous.NHFLOW_MATERIAL, material, case=case)
            return
        structure = self.selected()
        if structure is None:
            return
        geometry, old = structure.kind.split(structure.values)
        if old != material:
            values = structure.kind.join(geometry, material)
            self.session.set_input(
                FILE, structure.kind.code, values, structure.occurrence, case=case
            )

    # ------------------------------------------------------------- selection
    def _on_select(self) -> None:
        items = self.tree.selectedItems()
        key = items[0].data(0, KEY_ROLE) if items else None
        self._key = (str(key[0]), int(key[1])) if key else None
        self._show()
        self.highlight()

    def select(self, code: str, occurrence: int) -> None:
        self._key = (code, occurrence)
        self.refresh()
        self.highlight()

    def highlight(self) -> None:
        """The selected structure lit up in the 3D view."""
        if self.panel.case_id:
            code, occurrence = self._key or ("", 0)
            self.session.data.set_highlight(self.panel.case_id, FILE, code, occurrence)

    # ------------------------------------------------------------- edits
    def add(self, code: str = "", values=None) -> int | None:
        """A structure of the kind chosen (or ``code``), in the middle of the domain
        or with ``values``; selected. Returns its occurrence."""
        code = code or self.kind_box.currentData() or ""
        if not code or self.control() is None:
            return None
        try:
            occurrence = self.session.add_porous(code, values, case=self.panel.case_id)
        except ValueError as error:
            self.message.emit(str(error), True)
            return None
        self.select(code, occurrence)
        self.message.emit(
            f"Added a {porous.BY_CODE[code].name.lower()} ({code}): change it below, Save "
            "writes ctrl.txt",
            False,
        )
        return occurrence

    def draw(self) -> None:
        code = self.kind_box.currentData() or ""
        if not code or not self.panel.case_id:
            return
        self._drawing_kind = code
        self.drawArea.emit(self.panel.case_id, PURPOSE)

    def area_drawn(self, purpose: str, area) -> None:
        """The plan dragged in the 2D view: the structure over it, from the bed to
        the still water (in a 2D case across the whole domain)."""
        code, self._drawing_kind = self._drawing_kind, ""
        if purpose != PURPOSE or not code or not self.panel.case_id:
            return
        x0, x1, y0, y1 = (float(v) for v in area)
        setup = self.session.data.case_setup(self.panel.case_id)
        control = self.session.input_file("control.txt", self.panel.case_id)
        step = control.value("B 1") or 0.01
        if setup is not None and setup.two_d and setup.domain is not None:
            y0, y1 = setup.domain[2], setup.domain[3]
        x0, x1, y0, y1 = (porous.snap(v, step) for v in (x0, x1, y0, y1))
        if x1 <= x0 or y1 <= y0:
            self.message.emit("The area drawn is smaller than one cell", True)
            return
        values = self.session.porous_default(code, self.panel.case_id, (x0, x1, y0, y1))
        self.add(code, values)

    def remove_selected(self) -> None:
        structure = self.selected()
        if structure is None:
            return
        self.session.remove_input(
            FILE, structure.kind.code, structure.occurrence, case=self.panel.case_id
        )
        self._key = None
        self.refresh()
        self.highlight()
