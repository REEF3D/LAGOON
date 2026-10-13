# SPDX-License-Identifier: GPL-3.0-or-later
# Architect: Hans Bihs
"""The Map tab of the set-up: where a case lies on the Earth, its domain on the map,
and its bathymetry.

- *Find the place*: a place name (Kartverket, OpenStreetMap) or latitude, longitude;
  the domain's centre goes there (in the place's UTM zone when the case has no CRS
  yet), with a background map. A case started on the map (File → New case from
  map) is a draft until *Create case…* puts it in a folder of its own.
- *Georeference*: the CRS (EPSG code, or WKT with pyproj), the model's origin in it,
  its rotation and vertical datum; *Apply* writes ``georef.json``.
- *Domain*: *Draw on the map* — drag the domain's rectangle in the 2D top view
  (B 10), Shift + drag turns the model on the map about the domain's centre; or type
  the extent and the rotation.
- *Background map*: a map or satellite image round the domain (a "Background map"
  item, in the set-up and over the results).
- *Bathymetry*: downloaded for the domain (Kartverket's land and sea model in
  Norway, EMODnet in European seas), or a GeoTIFF, ASCII grid or x y z file of
  yours, carried onto a grid over the domain and written as geo.dat in model
  coordinates (G 10 1). *Run DIVEMesh*
  then makes the grid.
"""

from __future__ import annotations

import logging
import threading
from pathlib import Path

from PySide6.QtCore import QLocale, QObject, Qt, Signal
from PySide6.QtGui import QColor, QDoubleValidator, QGuiApplication
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QProgressDialog,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from lagoonviz.core.geo.bathy import Cancelled
from lagoonviz.core.geo.crs import (
    COMMON,
    CrsError,
    describe,
    is_known,
)
from lagoonviz.core.geo.places import search
from lagoonviz.core.kinds import BASEMAP_SOURCES
from lagoonviz.gui.theme import COLORS

log = logging.getLogger(__name__)

#: where geo.dat comes from: downloaded for the domain, or a file
BATHYMETRY_CHOICES = {
    "Download: best available": "best",
    "Download: Kartverket, land and sea (Norway)": "kartverket",
    "Download: EMODnet (European seas)": "emodnet",
    "A file of mine…": "file",
}
METHODS = {"Linear (TIN)": "tin", "Inverse distance": "idw", "Nearest point": "nearest"}


class _Runner(QObject):
    """Carries a worker thread's progress and end to the window's thread."""

    progressed = Signal(int, str)
    done = Signal(object)


class NumberField(QLineEdit):
    """A number typed as text (metres, degrees): narrow, any size of number, the unit
    after it. ``valueChanged`` when edited and confirmed; an empty field is 0 (or its
    placeholder's meaning)."""

    valueChanged = Signal(float)

    def __init__(self, low: float, high: float, decimals: int = 3, unit: str = "") -> None:
        super().__init__()
        self.low, self.high, self.decimals = float(low), float(high), int(decimals)
        validator = QDoubleValidator(self.low, self.high, self.decimals, self)
        validator.setNotation(QDoubleValidator.Notation.StandardNotation)
        validator.setLocale(QLocale.c())  # a point as the decimal mark, as in the files
        self.setValidator(validator)
        self.setMinimumWidth(60)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        if unit:
            self.setToolTip(unit.strip())
            self.addAction(_unit_icon(unit), QLineEdit.ActionPosition.TrailingPosition)
        self.editingFinished.connect(lambda: self.valueChanged.emit(self.value()))

    def value(self) -> float:
        try:
            return min(max(float(self.text().strip() or 0.0), self.low), self.high)
        except ValueError:
            return 0.0

    def setValue(self, value: float) -> None:  # noqa: N802 (as QDoubleSpinBox)
        text = f"{float(value):.{self.decimals}f}".rstrip("0").rstrip(".")
        self.setText("0" if text in ("", "-0") else text)
        if not self.signalsBlocked():
            self.valueChanged.emit(self.value())


def _unit_icon(unit: str):
    """The unit (m, °) drawn small at the end of a field."""
    from PySide6.QtGui import QIcon, QPainter, QPixmap

    pixmap = QPixmap(32, 32)
    pixmap.fill(Qt.transparent)
    painter = QPainter(pixmap)
    painter.setPen(QColor(COLORS["dim"]))
    font = painter.font()
    font.setPixelSize(20)
    painter.setFont(font)
    painter.drawText(pixmap.rect(), Qt.AlignCenter, unit.strip())
    painter.end()
    return QIcon(pixmap)


def _combo() -> QComboBox:
    """A combo box that narrows with the panel (long entries are elided)."""
    box = QComboBox()
    box.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLengthWithIcon)
    box.setMinimumContentsLength(10)
    box.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
    return box


def _spin(low: float, high: float, decimals: int = 3, suffix: str = "") -> NumberField:
    return NumberField(low, high, decimals, suffix)


class MapTab(QWidget):
    """The Map tab of the set-up panel (module doc)."""

    #: the user wants to draw the domain in the 2D view (the main window starts it)
    drawDomain = Signal(str)
    #: a message for the panel's status line: (text, error)
    message = Signal(str, bool)
    #: the user wants to start a case on the map (the main window asks where)
    newCase = Signal()

    def __init__(self, panel, parent=None) -> None:
        super().__init__(parent)
        self.panel = panel
        self.session = panel.session
        self._bathymetry: Path | None = None
        self._shown_georef: tuple | None = None
        self._shown_domain: tuple | None = None

        # ---- georeference
        self.crs = _combo()
        self.crs.setEditable(True)
        self.crs.setInsertPolicy(QComboBox.NoInsert)
        for code, name in COMMON.items():
            self.crs.addItem(f"EPSG:{code}  ·  {name}", f"EPSG:{code}")
        self.crs.setToolTip(
            "The coordinate reference system: an EPSG code (EPSG:25833 = ETRS89 / UTM 33N), "
            "or WKT with pyproj installed"
        )
        self.crs.setCurrentIndex(-1)  # none until the case has one, or one is chosen
        self.crs.lineEdit().setPlaceholderText("EPSG:25833")
        self.crs.currentTextChanged.connect(self._describe_crs)
        self.crs_name = QLabel()
        self.crs_name.setWordWrap(True)
        self.x0 = _spin(-1e9, 1e9, 3, " m")
        self.y0 = _spin(-1e9, 1e9, 3, " m")
        self.rotation = _spin(-360.0, 360.0, 3, " °")
        self.rotation.setToolTip("Counter-clockwise from east to the model's x-axis")
        self.rotation.valueChanged.connect(self._describe_bearing)
        self.bearing = QLabel()
        self.datum = QLineEdit()
        self.datum.setPlaceholderText("EPSG:5941 (NN2000), LAT, MSL …")
        self.z_offset = _spin(-1e5, 1e5, 3, " m")
        self.z_offset.setToolTip("The height of the model's z = 0 in the vertical datum")
        self.apply_button = QPushButton("Apply")
        self.apply_button.setToolTip("Write georef.json in the case folder")
        self.apply_button.clicked.connect(self.apply)
        self.remove_button = QPushButton("Remove")
        self.remove_button.setToolTip("Take georef.json away")
        self.remove_button.clicked.connect(self.remove)
        self.place = QLineEdit()
        self.place.setPlaceholderText("Nidelva, Trondheim havn … or 63.44, 10.40")
        self.place.setToolTip(
            "A place name (Kartverket in Norway, OpenStreetMap elsewhere) or latitude, "
            "longitude: the centre of the domain goes there (in its UTM zone when the case "
            "has no CRS yet)"
        )
        self.place.returnPressed.connect(self.place_domain)
        self.place_button = QPushButton("Find")
        self.place_button.clicked.connect(self.place_domain)
        self.results = _combo()
        self.results.setToolTip("The places found: choose one to put the domain there")
        self.results.activated.connect(self._result_chosen)
        self.results.hide()
        # a case started on the map lives in the drafts folder until it gets its own
        self.draft = QLabel()
        self.draft.setWordWrap(True)
        self.create_button = QPushButton("Create case…")
        self.create_button.setToolTip(
            "Put the case in a folder of its own: input files, georef.json, geo.dat and "
            "DIVEMesh's grid are copied there"
        )
        self.create_button.clicked.connect(self.create_case)
        georef = QGroupBox("Georeference")
        form = QFormLayout(georef)
        form.addRow("CRS", self.crs)
        form.addRow("", self.crs_name)
        origin = QHBoxLayout()
        origin.addWidget(QLabel("X0"))
        origin.addWidget(self.x0, 1)
        origin.addWidget(QLabel("Y0"))
        origin.addWidget(self.y0, 1)
        form.addRow("Origin", origin)
        form.addRow("Rotation", self.rotation)
        form.addRow("", self.bearing)
        form.addRow("Vertical datum", self.datum)
        form.addRow("z = 0 at", self.z_offset)
        buttons = QHBoxLayout()
        buttons.addStretch(1)
        buttons.addWidget(self.remove_button)
        buttons.addWidget(self.apply_button)
        form.addRow(buttons)

        # ---- domain
        self.draw_button = QPushButton("Draw on the map")
        self.draw_button.setToolTip(
            "Drag the domain's rectangle in the 2D top view; Shift + drag turns the model "
            "on the map about the domain's centre; Esc stops"
        )
        self.draw_button.clicked.connect(lambda: self.drawDomain.emit(self.case_id))
        self.extent = [_spin(-1e8, 1e8, 3, " m") for _ in range(4)]
        self.extent_button = QPushButton("Set B 10")
        self.extent_button.clicked.connect(self.set_extent)
        self.turn_to = _spin(-360.0, 360.0, 2, " °")
        self.turn_button = QPushButton("Turn")
        self.turn_button.setToolTip(
            "Turn the model on the map to this rotation, about the domain's centre (the "
            "domain stays where it is on the map)"
        )
        self.turn_button.clicked.connect(self.turn)
        domain = QGroupBox("Domain")
        form = QFormLayout(domain)
        form.addRow(self.draw_button)
        xs = QHBoxLayout()
        xs.addWidget(self.extent[0], 1)
        xs.addWidget(QLabel("…"))
        xs.addWidget(self.extent[1], 1)
        form.addRow("x", xs)
        ys = QHBoxLayout()
        ys.addWidget(self.extent[2], 1)
        ys.addWidget(QLabel("…"))
        ys.addWidget(self.extent[3], 1)
        form.addRow("y", ys)
        row = QHBoxLayout()
        row.addStretch(1)
        row.addWidget(self.extent_button)
        form.addRow(row)
        row = QHBoxLayout()
        row.addWidget(self.turn_to, 1)
        row.addWidget(self.turn_button)
        form.addRow("Turn to", row)

        # ---- background map
        self.source = _combo()
        self.source.addItems(list(BASEMAP_SOURCES))
        self.map_button = QPushButton("Show map")
        self.map_button.setToolTip(
            "A background map round the domain (an item: its properties set the margin, "
            "opacity and placement); shown in the set-up and over the results"
        )
        self.map_button.clicked.connect(self.show_map)
        background = QGroupBox("Background map")
        row = QHBoxLayout(background)
        row.addWidget(self.source, 1)
        row.addWidget(self.map_button)

        # ---- bathymetry
        self.bathy_source = _combo()
        for label, key in BATHYMETRY_CHOICES.items():
            self.bathy_source.addItem(label, key)
        self.bathy_source.setToolTip(
            "Download the bathymetry for the domain (no file needed): Kartverket's land "
            "and sea model in Norway, EMODnet in European seas; or a file of yours"
        )
        self.bathy_source.currentIndexChanged.connect(lambda _i: self._source_changed())
        self.file_label = QLabel("No file chosen")
        self.file_label.setWordWrap(True)
        self.file_button = QPushButton("Choose…")
        self.file_button.clicked.connect(self.choose_bathymetry)
        self.file_crs = QLineEdit()
        self.file_crs.setPlaceholderText("the file's own, else the case's")
        self.file_crs.setToolTip(
            "The CRS of the file's coordinates (EPSG:4326 for longitude, latitude; "
            "'model' for model coordinates)"
        )
        self.method = _combo()
        self.method.addItems(list(METHODS))
        self.method.setToolTip(
            "How scattered soundings are carried onto the grid (a raster is sampled bilinearly)"
        )
        self.spacing = _spin(0.0, 1e5, 3, " m")
        self.spacing.setPlaceholderText("DIVEMesh's cell size (B 1)")
        self.depths = QCheckBox("The file gives depths (positive down)")
        self.geo_button = QPushButton("Make geo.dat")
        self.geo_button.setToolTip(
            "Interpolate the file onto a grid over the domain and write geo.dat in model "
            "coordinates (and G 10 1); then Run DIVEMesh"
        )
        self.geo_button.clicked.connect(self.make_geo_dat)
        bathymetry = QGroupBox("Bathymetry")
        form = QFormLayout(bathymetry)
        form.addRow("From", self.bathy_source)
        self.file_row = QWidget()
        file_form = QFormLayout(self.file_row)
        file_form.setContentsMargins(0, 0, 0, 0)
        row = QHBoxLayout()
        row.addWidget(self.file_label, 1)
        row.addWidget(self.file_button)
        file_form.addRow("File", row)
        file_form.addRow("Its CRS", self.file_crs)
        file_form.addRow("Interpolation", self.method)
        file_form.addRow(self.depths)
        form.addRow(self.file_row)
        form.addRow("Spacing", self.spacing)
        row = QHBoxLayout()
        row.addStretch(1)
        row.addWidget(self.geo_button)
        form.addRow(row)

        #: what to do first, while the buttons wait for it (no case, no georeference)
        self.hint = QLabel()
        self.hint.setWordWrap(True)
        self.hint.setStyleSheet("color: #e07b39; font-weight: 600;")
        inner = QWidget()
        layout = QVBoxLayout(inner)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.addWidget(self.hint)
        self.start_button = QPushButton("New case from map…")
        self.start_button.setToolTip(
            "Find a place, start a case there with a domain and a map (File → New case from map)"
        )
        self.start_button.clicked.connect(lambda: self.newCase.emit())
        layout.addWidget(self.start_button)
        find = QGroupBox("Find the place")
        rows = QVBoxLayout(find)
        row = QHBoxLayout()
        row.addWidget(self.place, 1)
        row.addWidget(self.place_button)
        rows.addLayout(row)
        rows.addWidget(self.results)
        self.draft_box = QWidget()
        row = QHBoxLayout(self.draft_box)
        row.setContentsMargins(0, 0, 0, 0)
        row.addWidget(self.draft, 1)
        row.addWidget(self.create_button)
        layout.addWidget(self.draft_box)
        layout.addWidget(find)
        for group in (georef, domain, background, bathymetry):
            layout.addWidget(group)
        layout.addStretch(1)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        # the fields narrow with the panel: every button stays in view
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setWidget(inner)
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(scroll)
        self.refresh()

    # ------------------------------------------------------------- showing
    @property
    def case_id(self) -> str:
        return self.panel.case_id

    def _crs_text(self) -> str:
        data = self.crs.currentData()
        text = self.crs.currentText().strip()
        if data and text.startswith(str(data)):
            return str(data)
        return text.split("·")[0].strip()

    def _describe_crs(self, _text: str = "") -> None:
        text = self._crs_text()
        if not text:
            self.crs_name.setText("")
            return
        known = is_known(text)
        name = describe(text) if known else "not known: install pyproj for this CRS"
        self.crs_name.setText(name)

    def _describe_bearing(self, value: float) -> None:
        self.bearing.setText(f"x-axis to {(90.0 - value) % 360.0:.1f}° (compass)")

    def refresh(self) -> None:
        """The fields from the case's georef.json and B 10."""
        has_case = bool(self.case_id)
        for widget in (
            self.apply_button,
            self.place_button,
            self.draw_button,
            self.extent_button,
            self.map_button,
            self.file_button,
        ):
            widget.setEnabled(has_case)
        georef = self.session.data.georef(self.case_id) if has_case else None
        self.remove_button.setEnabled(georef is not None)
        self.turn_button.setEnabled(georef is not None)
        if not has_case:
            hint = (
                "Start a case on the map with File → New case from map…, or open one "
                "(File → Open case…): the map works on a case."
            )
        elif georef is None:
            hint = (
                "Find the place first: a place name or latitude, longitude, then Find "
                "(or choose the CRS and origin under Georeference, then Apply)."
            )
        else:
            hint = ""
        self.hint.setText(hint)
        self.hint.setVisible(bool(hint))
        self.start_button.setVisible(not has_case)
        draft = has_case and self.session.is_draft(self.case_id)
        self.draft.setText(
            "A draft started on the map (not in a folder of its own yet): find the place, "
            "draw the domain, add the bathymetry, then"
            if draft
            else ""
        )
        self.draft_box.setVisible(draft)
        web = self.bathy_source.currentData() != "file"
        self.file_row.setVisible(not web)
        self.geo_button.setEnabled(
            has_case and (self._bathymetry is not None or (web and georef is not None))
        )
        # the fields follow the files when those change, not while being typed in
        shown = (self.case_id, georef)
        if shown == self._shown_georef:
            georef_changed = False
        else:
            georef_changed = True
            self._shown_georef = shown
        for box in (self.x0, self.y0, self.rotation, self.z_offset, self.turn_to):
            box.blockSignals(True)
        if georef is None and georef_changed:  # another case, or none: empty fields
            self.crs.setCurrentIndex(-1)
            self.crs.setEditText("")
            for box in (self.x0, self.y0, self.rotation, self.z_offset, self.turn_to):
                box.setValue(0.0)
            self.datum.clear()
        if georef is not None and georef_changed:
            index = self.crs.findData(georef.crs)
            if index >= 0:
                self.crs.setCurrentIndex(index)
            else:
                self.crs.setEditText(georef.crs)
            self.x0.setValue(georef.origin[0])
            self.y0.setValue(georef.origin[1])
            self.rotation.setValue(georef.rotation_deg)
            self.turn_to.setValue(georef.rotation_deg)
            self.datum.setText(georef.vertical_datum)
            self.z_offset.setValue(georef.z_offset)
        for box in (self.x0, self.y0, self.rotation, self.z_offset, self.turn_to):
            box.blockSignals(False)
        self._describe_crs()
        self._describe_bearing(self.rotation.value())
        domain = self.session.data.backgrounds.domain(self.case_id) if has_case else None
        if domain is not None and (self.case_id, domain) != self._shown_domain:
            self._shown_domain = (self.case_id, domain)
            for box, value in zip(self.extent, domain[:4], strict=True):
                box.setValue(float(value))

    # ------------------------------------------------------------- actions
    def _run(self, action, done: str = "") -> bool:
        try:
            action()
        except (ValueError, OSError, CrsError) as error:
            self.message.emit(str(error), True)
            return False
        if done:
            self.message.emit(done, False)
        self.refresh()
        return True

    def apply(self) -> None:
        crs = self._crs_text()
        if not crs:
            self.message.emit("Choose the CRS first (e.g. EPSG:25833)", True)
            return
        self._run(
            lambda: self.session.set_georef(
                crs,
                origin=(self.x0.value(), self.y0.value()),
                rotation_deg=self.rotation.value(),
                vertical_datum=self.datum.text().strip(),
                z_offset=self.z_offset.value(),
                case=self.case_id,
            ),
            "georef.json written",
        )

    def remove(self) -> None:
        self._run(lambda: self.session.clear_georef(case=self.case_id), "georef.json removed")

    def place_domain(self) -> None:
        """Find the place typed (a name, or latitude, longitude) and put the domain's
        centre there; with several places found, the list offers the others."""
        text = self.place.text().strip()
        if not text or not self.case_id:
            self.message.emit("Type a place name, or latitude, longitude", True)
            return
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            found = search(text)
        except OSError as error:
            self.message.emit(f"{error} (offline? type latitude, longitude)", True)
            return
        finally:
            QApplication.restoreOverrideCursor()
        if not found:
            self.message.emit(f"Nothing found for {text!r}", True)
            return
        self.results.blockSignals(True)
        self.results.clear()
        for place in found:
            self.results.addItem(place.label(), place)
        self.results.setCurrentIndex(0)
        self.results.blockSignals(False)
        self.results.setVisible(len(found) > 1)
        self._place(found[0])

    def _result_chosen(self, index: int) -> None:
        place = self.results.itemData(index)
        if place is not None:
            self._place(place)

    def _place(self, place) -> None:
        has_georef = self.session.data.georef(self.case_id) is not None
        crs = None if has_georef else (self._crs_text() or None)
        self._run(
            lambda: self.session.place_case(place, case=self.case_id, crs=crs),
            f"{place.name}: the domain's centre is there",
        )

    def create_case(self) -> None:
        """The draft case into a folder of its own."""
        folder = QFileDialog.getExistingDirectory(
            self,
            "Folder of the case (make one with New Folder; it must be empty)",
            str(Path.home()),
        )
        if not folder:
            return
        self._run(
            lambda: self.session.create_case(folder, case=self.case_id),
            f"The case is in {folder} now",
        )

    def set_extent(self) -> None:
        x0, x1, y0, y1 = (box.value() for box in self.extent)
        self._run(
            lambda: self.session.set_domain(x0, x1, y0, y1, case=self.case_id),
            "B 10 set (Save or Run DIVEMesh writes control.txt)",
        )

    def domain_drawn(self, box) -> None:
        """The domain dragged in the 2D view: B 10."""
        if not self.case_id:
            return
        x0, x1, y0, y1 = (float(v) for v in box)
        # round to the cell size, else to centimetres
        step = self.session.input_file("control.txt", self.case_id).value("B 1") or 0.01
        step = step if step > 0 else 0.01

        def snap(v):
            return round(round(v / step) * step, 6)

        x0, x1, y0, y1 = snap(x0), snap(x1), snap(y0), snap(y1)
        if x1 <= x0 or y1 <= y0:
            self.message.emit("The domain drawn is smaller than one cell", True)
            return
        for spin, value in zip(self.extent, (x0, x1, y0, y1), strict=True):
            spin.setValue(value)
        self.set_extent()

    def model_turned(self, angle: float, centre) -> None:
        """Shift + drag in the 2D view: the model turned on the map."""
        georef = self.session.data.georef(self.case_id) if self.case_id else None
        if georef is None:
            self.message.emit("Set the georeference first (Apply)", True)
            return
        self._run(
            lambda: self.session.turn_model(
                georef.rotation_deg + angle, about=centre, case=self.case_id
            ),
            f"Turned the model by {angle:+.1f}°",
        )

    def turn(self) -> None:
        self._run(
            lambda: self.session.turn_model(self.turn_to.value(), case=self.case_id),
            f"Turned the model to {self.turn_to.value():.2f}°",
        )

    def show_map(self) -> None:
        """A background map item for the case (or the existing one, its map changed)."""
        if self.session.data.georef(self.case_id) is None:
            self.message.emit("Set the georeference first (CRS and origin, Apply)", True)
            return
        source = self.source.currentText()
        found = [
            i
            for i in self.session.scene
            if i.kind == "basemap"
            and (found_case := self.session.data.case_item(i.id)) is not None
            and found_case.id == self.case_id
        ]
        if found:
            item = found[0]
            if source in ("GeoTIFF file", "Tile server URL") and not item.properties.get("file"):
                self._ask_map_file(item.id, source)
            self.session.set_property(item.id, "source", source)
            if not item.visible:
                self.session.set_visible(item.id, True)
        else:
            item_id = self.session.add_background_map(source, case=self.case_id)
            if source in ("GeoTIFF file", "Tile server URL"):
                self._ask_map_file(item_id, source)
        self.message.emit(f"{source}: fetching the map round the domain…", False)

    def _ask_map_file(self, item_id: str, source: str) -> None:
        if source == "GeoTIFF file":
            chosen, _ = QFileDialog.getOpenFileName(
                self,
                "A georeferenced picture (orthophoto, chart)",
                str(Path.home()),
                "GeoTIFF (*.tif *.tiff)",
            )
        else:
            chosen = QGuiApplication.clipboard().text().strip()
            if "{z}" not in chosen:
                chosen = ""
                self.message.emit(
                    "Give the tile URL ({z}, {x}, {y}) in the item's File or URL", False
                )
        if chosen:
            self.session.set_property(item_id, "file", chosen)

    def choose_bathymetry(self) -> None:
        chosen, _ = QFileDialog.getOpenFileName(
            self,
            "Bathymetry or terrain",
            str(self.session.data.case_folder(self.case_id) or Path.home()),
            "Bathymetry (*.tif *.tiff *.asc *.xyz *.txt *.csv *.dat *.pts);;All files (*)",
        )
        if not chosen:
            return
        self._bathymetry = Path(chosen)
        self.file_label.setText(self._bathymetry.name)
        self.file_label.setToolTip(chosen)
        self.refresh()

    def _source_changed(self) -> None:
        self.refresh()

    def make_geo_dat(self) -> None:
        """Ask where geo.dat is stored (a draft case moves to that folder), then make
        it, downloaded for the domain or from the file chosen, in a worker thread with
        a progress bar (Cancel keeps the geo.dat there was)."""
        key = self.bathy_source.currentData()
        if key == "file" and self._bathymetry is None:
            return
        chosen = self._where_geo_dat()
        if chosen is None:
            return
        spacing = self.spacing.value() or None
        try:
            if key == "file":
                job = self.session.geo_dat_job(
                    self._bathymetry,
                    crs=self.file_crs.text().strip() or None,
                    method=METHODS[self.method.currentText()],
                    spacing=spacing,
                    depths=self.depths.isChecked(),
                    case=self.case_id,
                )
            else:
                job = self.session.geo_dat_job(key, spacing=spacing, case=self.case_id)
        except (ValueError, OSError, CrsError) as error:
            self.message.emit(str(error), True)
            return
        dialog = QProgressDialog("Making geo.dat…", "Cancel", 0, 1000, self)
        dialog.setWindowTitle("geo.dat")
        dialog.setWindowModality(Qt.WindowModal)
        dialog.setMinimumDuration(0)
        dialog.setAutoClose(False)
        dialog.setAutoReset(False)
        dialog.setValue(0)
        stop = threading.Event()
        runner = _Runner(self)
        runner.progressed.connect(
            lambda value, text: (dialog.setValue(value), dialog.setLabelText(text))
        )
        runner.done.connect(lambda error: self._geo_dat_done(job, error, dialog, chosen))
        dialog.canceled.connect(stop.set)

        def progress(fraction: float, text: str) -> None:
            if stop.is_set():
                raise Cancelled()
            runner.progressed.emit(int(round(1000 * fraction)), text)

        def work() -> None:
            try:
                job.run(progress)
            except BaseException as error:  # noqa: BLE001 (reported in the window)
                runner.done.emit(error)
                return
            runner.done.emit(None)

        self._geo_runner = runner  # kept until done
        threading.Thread(target=work, name="lagoon-geo.dat", daemon=True).start()

    def _geo_dat_done(self, job, error, dialog, chosen: Path) -> None:
        """Back in the window's thread: G 10 1 and the message, or why not."""
        dialog.close()
        self._geo_runner = None
        if isinstance(error, Cancelled):
            self.message.emit("Cancelled: geo.dat is as it was", False)
            return
        if error is not None:
            log.warning("geo.dat: %s", error)
            self.message.emit(str(error) or type(error).__name__, True)
            return
        try:
            count = job.finish()
        except (ValueError, OSError) as failed:
            self.message.emit(str(failed), True)
            return
        self.refresh()
        folder = self.session.data.case_folder(self.case_id)
        stored = folder / "geo.dat"
        note = ""
        if chosen.resolve() != stored.resolve():  # elsewhere, or another name: a copy
            import shutil

            try:
                shutil.copyfile(stored, chosen)
                note = f", a copy as {chosen}"
            except OSError as failed:
                note = f" (the copy as {chosen} failed: {failed})"
        sources = " and ".join(self.session.geo_sources) or "the file"
        self.message.emit(
            f"geo.dat: {count:,} points from {sources}, stored in {folder}{note}; "
            "G 10 1 in control.txt. Now Run DIVEMesh",
            False,
        )

    def _where_geo_dat(self) -> Path | None:
        """Where geo.dat is stored, asked in a save dialog; None: cancelled.

        DIVEMesh reads geo.dat in the case folder: a draft case (started on the map)
        moves to the folder chosen, with its input files and georef.json; for a case
        with its own folder the dialog starts there, and a place elsewhere (or another
        name) gets a copy."""
        folder = self.session.data.case_folder(self.case_id)
        if folder is None:
            return None
        draft = self.session.is_draft(self.case_id)
        start = Path.home() / "geo.dat" if draft else folder / "geo.dat"
        title = (
            "Store geo.dat: the case goes to this folder (with control.txt, ctrl.txt, georef.json)"
            if draft
            else "Store geo.dat (DIVEMesh reads the one in the case folder)"
        )
        chosen, _ = QFileDialog.getSaveFileName(
            self, title, str(start), "geo.dat (*.dat);;All files (*)"
        )
        if not chosen:
            return None
        chosen = Path(chosen)
        target = chosen.parent
        if draft:
            clashes = self.session.case_file_clashes(target, self.case_id)
            clashes = [c for c in clashes if c != "geo.dat"]  # the dialog asked already
            if clashes:
                answer = QMessageBox.question(
                    self,
                    "Store geo.dat",
                    f"{target} already has {', '.join(clashes)}. Put this case there and "
                    "overwrite them?",
                    QMessageBox.Yes | QMessageBox.Cancel,
                    QMessageBox.Cancel,
                )
                if answer != QMessageBox.Yes:
                    return None
            if not self._run(
                lambda: self.session.create_case(target, case=self.case_id, overwrite=True),
                f"The case is in {target} now",
            ):
                return None
        return chosen
