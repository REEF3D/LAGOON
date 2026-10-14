# SPDX-License-Identifier: GPL-3.0-or-later
# Architect: Hans Bihs
"""The Sea tab of the set-up: the sea at the case from MET Norway's hindcasts, for
the waves and the boundaries of a REEF3D run.

- *Find storms*: the largest wave events of NORA3 (waves and wind, 3 km, hourly,
  1959 on) at the wave generation side of the case; one chosen sets the window.
- *Show the conditions*: NORA3's Hs, Tp, direction and wind and NorKyst's water
  level and current (800 m, hourly, 2017-02-20 on) over the window, as time series
  items (tables in the case's ``LAGOON_Sea`` folder, time 0 at the window's start).
- *Wave spectrum*: NORA3's 2D spectrum at the storm's peak as REEF3D's
  ``spectrum-file-2d.dat``, ctrl.txt's wave generation set to it.
- *Boundary hydrographs*: NorKyst's water level and discharge along the sides of
  the domain as ``hydrograph_FSF_<n>.dat`` and ``hydrograph_Q_<n>.dat``, ctrl.txt's
  patches (B 440, B 421, B 422) set to them.

The case needs a georeference (the Map tab). The data come from
thredds.met.no while it runs; ctrl.txt changes are one undo step each, saved with
the input files.
"""

from __future__ import annotations

import datetime as dt
import logging
import threading

from PySide6.QtCore import QDateTime, Qt, QTimeZone, Signal
from PySide6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QDateTimeEdit,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QProgressDialog,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from lagoonviz.core.geo.bathy import Cancelled
from lagoonviz.gui.panels.map_setup import _Runner

log = logging.getLogger(__name__)

ATTRIBUTION = (
    "Data: MET Norway (NORA3, NorKyst), thredds.met.no, under NLOD 2.0 / CC BY 4.0. "
    "NORA3 from 1959 (a month or two behind), NorKyst from 2017-02-20."
)
#: hours before and after a storm's peak for the window
BEFORE_HOURS, AFTER_HOURS = 24, 12


def _qt(when: dt.datetime) -> QDateTime:
    return QDateTime.fromSecsSinceEpoch(int(when.timestamp()), QTimeZone.utc())


def _py(value: QDateTime) -> dt.datetime:
    return dt.datetime.fromtimestamp(value.toSecsSinceEpoch(), dt.UTC)


def _time_edit(when: dt.datetime) -> QDateTimeEdit:
    edit = QDateTimeEdit(_qt(when))
    edit.setTimeZone(QTimeZone.utc())
    edit.setDisplayFormat("yyyy-MM-dd HH:00 'UTC'")
    edit.setCalendarPopup(True)
    return edit


class SeaTab(QWidget):
    """The Sea tab of the set-up panel (module doc)."""

    #: a message for the panel's status line: (text, error)
    message = Signal(str, bool)

    def __init__(self, panel, parent=None) -> None:
        super().__init__(parent)
        from lagoonviz.core import sea

        self.panel = panel
        self.session = panel.session
        self.storms: list = []
        self._runner = None
        self._stop: threading.Event | None = None

        self.where = QLabel()
        self.where.setWordWrap(True)

        # ---- storms
        first, last = sea.default_years()
        self.first_year = QSpinBox()
        self.last_year = QSpinBox()
        for box, value in ((self.first_year, first), (self.last_year, last)):
            box.setRange(1959, dt.datetime.now(dt.UTC).year)
            box.setValue(value)
        self.count = QSpinBox()
        self.count.setRange(1, 50)
        self.count.setValue(10)
        self.storm_button = QPushButton("Find storms")
        self.storm_button.setToolTip(
            "The largest wave events of NORA3 at the case (at least 3 days apart); a "
            "year takes some seconds to read"
        )
        self.storm_button.clicked.connect(self.find_storms)
        years = QHBoxLayout()
        years.addWidget(self.first_year)
        years.addWidget(QLabel("to"))
        years.addWidget(self.last_year)
        years.addWidget(QLabel("largest"))
        years.addWidget(self.count)
        years.addStretch(1)
        self.table = QTableWidget(0, 5)
        self.table.setHorizontalHeaderLabels(
            ["Peak (UTC)", "Hs [m]", "Tp [s]", "From [°]", "Wind [m/s]"]
        )
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.verticalHeader().setVisible(False)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.table.setMinimumHeight(140)
        self.table.itemSelectionChanged.connect(self._storm_chosen)
        storms = QVBoxLayout()
        storms.addLayout(years)
        storms.addWidget(self.storm_button)
        storms.addWidget(self.table)
        storm_box = QGroupBox("Storms (NORA3)")
        storm_box.setLayout(storms)

        # ---- the window
        now = dt.datetime(last, 1, 1, tzinfo=dt.UTC)
        self.start = _time_edit(now)
        self.end = _time_edit(now + dt.timedelta(days=1))
        self.peak = _time_edit(now + dt.timedelta(hours=BEFORE_HOURS))
        self.peak.setToolTip("The hour of the spectrum (a storm chosen: its peak)")
        window = QFormLayout()
        window.addRow("Start", self.start)
        window.addRow("End", self.end)
        window.addRow("Spectrum at", self.peak)
        window_box = QGroupBox("Time window")
        window_box.setLayout(window)

        # ---- what to make
        self.show_button = QPushButton("Show the conditions")
        self.show_button.setToolTip(
            "NORA3 waves and wind, NorKyst water level and current at the case over the "
            "window, as time series (LAGOON_Sea folder)"
        )
        self.show_button.clicked.connect(self.show_conditions)
        self.spectrum_button = QPushButton("Write spectrum-file-2d.dat")
        self.spectrum_button.setToolTip(
            "NORA3's 2D spectrum at the case as REEF3D's spectrum file (B 85 11), the "
            "directions ±90° round the mean"
        )
        self.spectrum_button.clicked.connect(self.write_spectrum)
        self.sides = {}
        side_row = QHBoxLayout()
        side_row.addWidget(QLabel("Sides"))
        for number, (name, _) in sea.SIDES.items():
            box = QCheckBox(name)
            box.setToolTip(f"Side {number} (B 440 {number} {number} …)")
            self.sides[number] = box
            side_row.addWidget(box)
        side_row.addStretch(1)
        self.auto_sides = QCheckBox("Those in the water")
        self.auto_sides.setChecked(True)
        self.auto_sides.setToolTip("The sides more than half in NorKyst's water")
        self.auto_sides.toggled.connect(self._sides_enabled)
        self.boundary_button = QPushButton("Write boundary hydrographs")
        self.boundary_button.setToolTip(
            "NorKyst's water level and discharge along the sides: hydrograph_FSF_<n>.dat, "
            "hydrograph_Q_<n>.dat (patchBC; SFLOW and CFD per the user guide)"
        )
        self.boundary_button.clicked.connect(self.write_boundaries)
        self.set_ctrl = QCheckBox("Set ctrl.txt to them")
        self.set_ctrl.setChecked(True)
        self.set_ctrl.setToolTip(
            "Spectrum: B 85 11, B 92 31, B 93, B 105, B 130 1. Boundaries: B 440, B 421, "
            "B 422. One undo step each, saved with the input files"
        )
        make = QVBoxLayout()
        make.addWidget(self.show_button)
        make.addWidget(self.spectrum_button)
        make.addWidget(self.auto_sides)
        make.addLayout(side_row)
        make.addWidget(self.boundary_button)
        make.addWidget(self.set_ctrl)
        make_box = QGroupBox("For the case")
        make_box.setLayout(make)

        credit = QLabel(ATTRIBUTION)
        credit.setWordWrap(True)
        credit.setEnabled(False)

        inner = QWidget()
        layout = QVBoxLayout(inner)
        layout.addWidget(self.where)
        layout.addWidget(storm_box)
        layout.addWidget(window_box)
        layout.addWidget(make_box)
        layout.addWidget(credit)
        layout.addStretch(1)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(inner)
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(scroll)
        self._sides_enabled(True)

    # ------------------------------------------------------------- showing
    @property
    def case_id(self) -> str:
        return self.panel.case_id

    def _sides_enabled(self, auto: bool) -> None:
        for box in self.sides.values():
            box.setEnabled(not auto)

    def refresh(self) -> None:
        """Where the sea is read, or why not."""
        from lagoonviz.core import sea

        ready = False
        if not self.case_id:
            self.where.setText("Open a case to read the sea at it.")
        else:
            try:
                spot = sea.place(self.session, self.case_id)
            except (ValueError, OSError) as error:
                self.where.setText(f"{error}. Georeference the case in the Map tab first.")
            else:
                ready = True
                self.where.setText(
                    f"Read at {spot.lat:.4f} N, {spot.lon:.4f} E: {spot.label} "
                    f"(x {spot.x:g}, y {spot.y:g})"
                )
        busy = self._runner is not None
        for button in (
            self.storm_button,
            self.show_button,
            self.spectrum_button,
            self.boundary_button,
        ):
            button.setEnabled(ready and not busy)

    def _storm_chosen(self) -> None:
        rows = self.table.selectionModel().selectedRows()
        if not rows or rows[0].row() >= len(self.storms):
            return
        storm = self.storms[rows[0].row()]
        self.peak.setDateTime(_qt(storm.when))
        self.start.setDateTime(_qt(storm.when - dt.timedelta(hours=BEFORE_HOURS)))
        self.end.setDateTime(_qt(storm.when + dt.timedelta(hours=AFTER_HOURS)))
        self.message.emit(storm.describe(), False)

    def _window(self) -> tuple[dt.datetime, dt.datetime] | None:
        start, end = _py(self.start.dateTime()), _py(self.end.dateTime())
        if end <= start:
            self.message.emit("The window ends before it starts", True)
            return None
        return start, end

    # ------------------------------------------------------------- jobs
    def _job(self, title: str, work, finish) -> None:
        """``work(progress)`` in a thread (reads THREDDS, writes files), then
        ``finish(result)`` in the window's thread (the scene, ctrl.txt)."""
        dialog = QProgressDialog(title + "…", "Cancel", 0, 1000, self)
        dialog.setWindowTitle(title)
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
        dialog.canceled.connect(stop.set)
        box: dict = {}

        def ended(error) -> None:
            dialog.close()
            self._runner = None
            try:
                if isinstance(error, Cancelled):
                    self.message.emit(f"Cancelled: {title.lower()}", False)
                elif error is not None:
                    log.warning("%s: %s", title, error)
                    self.message.emit(str(error) or type(error).__name__, True)
                else:
                    finish(box["result"])
            except (ValueError, OSError) as failed:
                self.message.emit(str(failed), True)
            finally:
                self.panel.refresh()

        runner.done.connect(ended)

        def progress(fraction: float, text: str) -> None:
            if stop.is_set():
                raise Cancelled()
            runner.progressed.emit(int(round(1000 * min(max(fraction, 0.0), 1.0))), text)

        def run() -> None:
            try:
                box["result"] = work(progress)
            except BaseException as error:  # noqa: BLE001 (reported in the window)
                runner.done.emit(error)
                return
            runner.done.emit(None)

        self._runner = runner  # kept until done
        self.refresh()
        threading.Thread(target=run, name="lagoon-sea", daemon=True).start()

    def find_storms(self) -> None:
        from lagoonviz.core import sea

        case_id = self.case_id
        first, last = sorted((self.first_year.value(), self.last_year.value()))
        count = self.count.value()

        def work(progress):
            return sea.find_storms(self.session, case_id, first, last, count, progress)

        def finish(result) -> None:
            found, point, _ = result
            self.storms = list(found)
            self.table.setRowCount(len(self.storms))
            for row, s in enumerate(self.storms):
                cells = [
                    f"{s.when:%Y-%m-%d %H:00}",
                    f"{s.hs:.1f}",
                    f"{s.tp:.1f}",
                    f"{s.direction:.0f}",
                    f"{s.wind:.0f} from {s.wind_from:.0f}°",
                ]
                for column, text in enumerate(cells):
                    self.table.setItem(row, column, QTableWidgetItem(text))
            self.message.emit(
                f"{len(self.storms)} storms {first}–{last} at the NORA3 point "
                f"{point.lat:.3f} N {point.lon:.3f} E ({point.distance:.1f} km away). "
                "Choose one for the window",
                False,
            )

        self._job("Finding storms", work, finish)

    def show_conditions(self) -> None:
        from lagoonviz.core import sea

        window = self._window()
        if window is None:
            return
        case_id = self.case_id

        def work(progress):
            return sea.conditions(self.session, case_id, *window, progress)

        def finish(written) -> None:
            self.session.sea_notes = written.notes
            items = self.session._show_sea_tables(case_id, written.files)
            note = ("; " + "; ".join(written.notes)) if written.notes else ""
            self.message.emit(f"{len(items)} time series of the sea shown{note}", False)

        self._job("Reading the sea", work, finish)

    def write_spectrum(self) -> None:
        from lagoonviz.core import sea

        case_id = self.case_id
        when = _py(self.peak.dateTime())
        apply = self.set_ctrl.isChecked()

        def work(progress):
            progress(0.1, f"NORA3 spectra {when:%Y-%m-%d}")
            return sea.spectrum(self.session, case_id, when, apply=False)

        def finish(facts) -> None:
            if apply:
                sea.apply_spectrum(self.session, case_id, facts)
            self.message.emit(
                f"{facts['file'].name}: Hs {facts['Hs']:.2f} m, Tp {facts['Tp']:.1f} s, "
                f"going {facts['direction']:.0f}° from the x-axis (NORA3 "
                f"{facts['lat']:.3f} N {facts['lon']:.3f} E)"
                + ("; ctrl.txt set (B 85 11, B 93, B 105)" if apply else ""),
                False,
            )

        self._job("Wave spectrum", work, finish)

    def write_boundaries(self) -> None:
        from lagoonviz.core import sea

        window = self._window()
        if window is None:
            return
        sides = None
        if not self.auto_sides.isChecked():
            sides = [n for n, box in self.sides.items() if box.isChecked()]
            if not sides:
                self.message.emit("Choose the sides first", True)
                return
        case_id = self.case_id
        apply = self.set_ctrl.isChecked()

        def work(progress):
            return sea.boundaries(
                self.session, case_id, *window, sides=sides, apply=False, progress=progress
            )

        def finish(chosen) -> None:
            if not chosen:
                self.message.emit("No side of the domain is in NorKyst's water", True)
                return
            if apply:
                sea.apply_boundaries(self.session, case_id, chosen)
            folder = self.session.data.case_folder(case_id)
            stamp = f"{window[0]:%Y%m%d%H}"
            self.session._show_sea_tables(
                case_id, [folder / sea.FOLDER / f"Boundaries-{stamp}.dat"]
            )
            names = ", ".join(f"{s.side} {s.name}" for s in chosen)
            self.message.emit(
                f"Hydrographs for {names}"
                + ("; ctrl.txt patches set (B 440, B 421, B 422)" if apply else ""),
                False,
            )

        self._job("Boundary hydrographs", work, finish)
