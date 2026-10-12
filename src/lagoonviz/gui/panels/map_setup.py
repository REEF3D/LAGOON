# SPDX-License-Identifier: GPL-3.0-or-later
# Architect: Hans Bihs
"""The Map tab of the set-up: where a case lies on the Earth, its domain on the map,
and its bathymetry.

- *Georeference*: the CRS (EPSG code, or WKT with pyproj), the model's origin in it,
  its rotation and vertical datum; *Apply* writes ``georef.json``. *Place* puts the
  domain's centre at a longitude and latitude (and picks the UTM zone when no CRS is
  set yet).
- *Domain*: *Draw on the map* — drag the domain's rectangle in the 2D top view
  (B 10), Shift + drag turns the model on the map about the domain's centre; or type
  the extent and the rotation.
- *Background map*: a map or satellite image round the domain (a "Background map"
  item, in the set-up and over the results).
- *Bathymetry*: a GeoTIFF, ASCII grid or x y z file, interpolated onto a grid over
  the domain and written as geo.dat in model coordinates (G 10 1). *Run DIVEMesh*
  then makes the grid.
"""

from __future__ import annotations

import logging
import math
from pathlib import Path

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QGuiApplication
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from lagoonviz.core.geo.crs import (
    COMMON,
    CrsError,
    describe,
    is_geographic,
    is_known,
    transformer,
    utm_for,
)
from lagoonviz.core.kinds import BASEMAP_SOURCES

log = logging.getLogger(__name__)

METHODS = {"Linear (TIN)": "tin", "Inverse distance": "idw", "Nearest point": "nearest"}


def _spin(low: float, high: float, decimals: int = 3, suffix: str = "") -> QDoubleSpinBox:
    box = QDoubleSpinBox()
    box.setRange(low, high)
    box.setDecimals(decimals)
    box.setSuffix(suffix)
    box.setKeyboardTracking(False)
    box.setButtonSymbols(QDoubleSpinBox.NoButtons)
    return box


class MapTab(QWidget):
    """The Map tab of the set-up panel (module doc)."""

    #: the user wants to draw the domain in the 2D view (the main window starts it)
    drawDomain = Signal(str)
    #: a message for the panel's status line: (text, error)
    message = Signal(str, bool)

    def __init__(self, panel, parent=None) -> None:
        super().__init__(parent)
        self.panel = panel
        self.session = panel.session
        self._bathymetry: Path | None = None
        self._shown_georef: tuple | None = None
        self._shown_domain: tuple | None = None

        # ---- georeference
        self.crs = QComboBox()
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
        self.place.setPlaceholderText("63.4400, 10.4000  (latitude, longitude)")
        self.place.setToolTip(
            "Put the centre of the domain here (degrees); the UTM zone is chosen when no "
            "CRS is set yet"
        )
        self.place_button = QPushButton("Place")
        self.place_button.clicked.connect(self.place_domain)
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
        turn = QHBoxLayout()
        turn.addWidget(self.rotation, 1)
        turn.addWidget(self.bearing)
        form.addRow("Rotation", turn)
        form.addRow("Vertical datum", self.datum)
        form.addRow("z = 0 at", self.z_offset)
        place = QHBoxLayout()
        place.addWidget(self.place, 1)
        place.addWidget(self.place_button)
        form.addRow("Place at", place)
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
        self.source = QComboBox()
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
        self.method = QComboBox()
        self.method.addItems(list(METHODS))
        self.method.setToolTip(
            "How scattered soundings are carried onto the grid (a raster is sampled bilinearly)"
        )
        self.spacing = _spin(0.0, 1e5, 3, " m")
        self.spacing.setSpecialValueText("DIVEMesh's cell size (B 1)")
        self.depths = QCheckBox("The file gives depths (positive down)")
        self.geo_button = QPushButton("Make geo.dat")
        self.geo_button.setToolTip(
            "Interpolate the file onto a grid over the domain and write geo.dat in model "
            "coordinates (and G 10 1); then Run DIVEMesh"
        )
        self.geo_button.clicked.connect(self.make_geo_dat)
        bathymetry = QGroupBox("Bathymetry")
        form = QFormLayout(bathymetry)
        row = QHBoxLayout()
        row.addWidget(self.file_label, 1)
        row.addWidget(self.file_button)
        form.addRow("File", row)
        form.addRow("Its CRS", self.file_crs)
        form.addRow("Interpolation", self.method)
        form.addRow("Spacing", self.spacing)
        form.addRow(self.depths)
        row = QHBoxLayout()
        row.addStretch(1)
        row.addWidget(self.geo_button)
        form.addRow(row)

        inner = QWidget()
        layout = QVBoxLayout(inner)
        layout.setContentsMargins(6, 6, 6, 6)
        for group in (georef, domain, background, bathymetry):
            layout.addWidget(group)
        layout.addStretch(1)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
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
        self.geo_button.setEnabled(has_case and self._bathymetry is not None)
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
        """The domain's centre at a latitude and longitude."""
        parts = self.place.text().replace(";", ",").replace(",", " ").split()
        try:
            lat, lon = float(parts[0]), float(parts[1])
        except (IndexError, ValueError):
            self.message.emit("Give the place as latitude, longitude (degrees)", True)
            return
        if not (-90 <= lat <= 90 and -180 <= lon <= 180):
            self.message.emit("Latitude -90 … 90, longitude -180 … 180", True)
            return
        crs = self._crs_text()
        if not crs or is_geographic(crs):
            crs = utm_for(lon, lat)  # a CRS in metres
        try:
            big_x, big_y = transformer("EPSG:4326", crs)(lon, lat)
        except CrsError as error:
            self.message.emit(str(error), True)
            return
        domain = self.session.data.backgrounds.domain(self.case_id)
        cx, cy = ((domain[0] + domain[1]) / 2, (domain[2] + domain[3]) / 2) if domain else (0, 0)
        r = math.radians(self.rotation.value())
        # the origin that puts the model's (cx, cy) at the place
        x0 = float(big_x) - (cx * math.cos(r) - cy * math.sin(r))
        y0 = float(big_y) - (cx * math.sin(r) + cy * math.cos(r))
        index = self.crs.findData(crs)
        if index >= 0:
            self.crs.setCurrentIndex(index)
        else:
            self.crs.setEditText(crs)
        self.x0.setValue(x0)
        self.y0.setValue(y0)
        self.apply()

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

    def make_geo_dat(self) -> None:
        if self._bathymetry is None:
            return
        crs = self.file_crs.text().strip() or None
        method = METHODS[self.method.currentText()]
        spacing = self.spacing.value() or None
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            count = [0]

            def make():
                count[0] = self.session.make_geo_dat(
                    self._bathymetry,
                    crs=crs,
                    method=method,
                    spacing=spacing,
                    depths=self.depths.isChecked(),
                    case=self.case_id,
                )

            ok = self._run(make)
        finally:
            QApplication.restoreOverrideCursor()
        if ok:
            self.message.emit(
                f"geo.dat: {count[0]:,} points in model coordinates, G 10 1; now Run DIVEMesh",
                False,
            )
