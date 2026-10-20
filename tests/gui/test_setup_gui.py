"""The set-up in the window: the Set-up / Results switch, the Set-up panel editing the
input files, and DIVEMesh run from it (a stand-in program)."""

import os
import stat
import sys
from pathlib import Path

import pytest
from PySide6.QtCore import QSettings

from lagoonviz.core import Session
from lagoonviz.gui.main_window import MainWindow

TESTS = Path(__file__).resolve().parents[1]
CONTROL = "C 11 6\nC 14 7\nB 1 0.5\nB 10 0.0 10.0 0.0 2.0 0.0 1.0\nM 10 1\n"
CTRL = "A 10 5\nF 60 0.6\nP 51 2.0 1.0\nP 51 6.0 1.0\n"


@pytest.fixture
def window(qtbot, tmp_path):
    settings = QSettings(str(tmp_path / "settings.ini"), QSettings.IniFormat)
    win = MainWindow(Session(), settings=settings)
    win.ask_before_discard = False
    qtbot.addWidget(win)
    win.show()
    return win


@pytest.fixture
def flume(tmp_path):
    case = tmp_path / "flume"
    case.mkdir()
    (case / "control.txt").write_text(CONTROL)
    (case / "ctrl.txt").write_text(CTRL)
    return case


def _entry(editor, code, occurrence=0):
    for top in range(editor.tree.topLevelItemCount()):
        section = editor.tree.topLevelItem(top)
        for n in range(section.childCount()):
            entry = section.child(n)
            if tuple(entry.data(0, 0x0100)) == (code, occurrence):
                return entry
    return None


def test_a_case_that_has_not_run_opens_in_the_set_up(window, flume):
    window.open_case(flume)
    assert window.session.mode == "setup"
    assert window.act_mode_setup.isChecked() and not window.act_mode_results.isChecked()
    assert not window.timeline.isEnabled() and window.act_run_divemesh.isVisible()
    assert window.docks["setup"].isVisible()
    panel = window.setup_panel
    assert panel.case_box.count() == 1 and panel.case_id
    assert _entry(panel.editors["ctrl.txt"], "P 51", 1) is not None
    assert _entry(panel.editors["control.txt"], "B 10") is not None
    window.act_mode_results.trigger()
    assert window.session.mode == "results" and window.timeline.isEnabled()
    assert not window.act_run_divemesh.isVisible()
    # the right panel goes from the set-up to the properties
    assert not window.docks["setup"].isVisible() and window.docks["properties"].isVisible()
    window.act_mode_setup.trigger()
    assert window.docks["setup"].isVisible()


def test_the_results_are_greyed_in_the_set_up(window, reef3d_run):
    (reef3d_run / "ctrl.txt").write_text(CTRL)
    window.open_case(reef3d_run)
    assert window.session.mode == "results"
    window.act_mode_setup.trigger()
    tree = window.scene_tree
    surface = next(i for i in window.session.scene if i.kind == "free_surface")
    entry = tree._entry(surface.id)
    assert "shown in Results" in entry.toolTip(0)
    assert not window.session.data.drawn(surface.id)


def test_editing_a_command_in_the_panel(window, flume):
    window.open_case(flume)
    panel = window.setup_panel
    editor = panel.editors["ctrl.txt"]
    editor.tree.setCurrentItem(_entry(editor, "P 51", 1))
    assert window.session.data.highlight[2:] == ("P 51", 1)  # lit in the 3D view
    x, y = editor._fields
    x.setText("7.5")
    x.editingFinished.emit()
    ctrl = window.session.input_file("ctrl.txt")
    assert ctrl.all("P 51") == [[2.0, 1.0], [7.5, 1.0]]
    assert panel.tabs.tabText(1).endswith("*") and panel.save_button.isEnabled()
    assert _entry(editor, "P 51", 1).text(1) == "7.5 1.0"
    window.act_undo.trigger()
    assert ctrl.all("P 51")[1] == [6.0, 1.0]
    window.act_redo.trigger()
    editor._add_from("vertical velocity profile")  # P 67 (NHFLOW): its default values
    assert "P 67" in window.session.input_file("ctrl.txt")
    editor.remove_selected()
    assert "P 67" not in window.session.input_file("ctrl.txt")
    panel.save()
    assert "P 51 7.5 1.0" in (flume / "ctrl.txt").read_text()
    assert not panel.save_button.isEnabled()


def test_a_choice_is_set_from_its_list(window, flume):
    window.open_case(flume)
    editor = window.setup_panel.editors["control.txt"]
    editor.tree.setCurrentItem(_entry(editor, "C 14"))
    (box,) = editor._fields
    box.setCurrentIndex(box.findData("21"))  # wall
    box.activated.emit(box.currentIndex())
    assert window.session.input_file("control.txt").int("C 14") == 21


def test_the_text_of_a_file(window, flume):
    window.open_case(flume)
    editor = window.setup_panel.editors["control.txt"]
    editor.text_button.setChecked(True)
    assert "B 10 0.0 10.0" in editor.text.toPlainText()
    editor.text.setPlainText(CONTROL.replace("B 1 0.5", "B 1 0.25"))
    editor.apply_text.click()
    setup = window.session.data.case_setup(window.setup_panel.case_id)
    assert setup.cells == (40, 8, 4)


FAKE = """\
import sys
sys.path.insert(0, {tests!r})
from pathlib import Path
import numpy as np
from reef3d_fixtures import write_grid_preview
print("read control")
write_grid_preview(Path("."), np.linspace(0, 10, 21), np.linspace(0, 2, 5), np.linspace(0, 1, 3))
print("--DONE--")
"""


@pytest.mark.skipif(os.name == "nt", reason="the stand-in is a shell script")
def test_running_divemesh_from_the_panel(window, flume, tmp_path, qtbot, monkeypatch):
    script = tmp_path / "fake_divemesh.py"
    script.write_text(FAKE.format(tests=str(TESTS)))
    program = tmp_path / "DiveMESH"
    program.write_text(f"#!/bin/sh\nexec {sys.executable} {script}\n")
    program.chmod(program.stat().st_mode | stat.S_IEXEC)
    monkeypatch.setenv("LAGOON_DIVEMESH", str(program))
    window.open_case(flume)
    panel = window.setup_panel
    window.session.set_input("control.txt", "B 1", [0.25])  # saved before the run
    assert panel.run_divemesh()
    qtbot.waitUntil(lambda: panel.process is None, timeout=20000)
    assert "--DONE--" in panel.log.toPlainText()
    assert "B 1 0.25" in (flume / "control.txt").read_text()
    grid = next(i for i in window.session.scene if i.kind == "grid")
    assert window.session.data.grid(grid.id).source == "DIVEMesh"
    assert panel.settings.value("divemesh/program") == str(program)


def test_the_map_tab_places_the_case(window, flume, monkeypatch):
    from lagoon_format.geo import Georef

    window.open_case(flume)
    panel = window.setup_panel
    tab = panel.map_tab
    assert (
        panel.tabs.tabText(panel.tabs.indexOf(tab)) == "Map" and not tab.remove_button.isEnabled()
    )
    assert tab.place_button.isEnabled() and "Find the place" in tab.hint.text()
    assert not tab.session.is_draft()
    # the domain's centre at Trondheim: the UTM zone is chosen
    tab.place.setText("63.44, 10.40")
    tab.place_domain()
    georef = window.session.georef()
    assert georef is not None and georef.crs == "EPSG:25832"
    assert Georef.from_json((flume / "georef.json").read_text()) == georef
    assert tab.remove_button.isEnabled() and "UTM zone 32N" in tab.crs_name.text()
    assert tab.hint.text() == ""
    # the domain dragged in the 2D view, snapped to the cell size (B 1 0.5)
    tab.domain_drawn((1.12, 20.3, -2.1, 4.9))
    assert window.session.input_file("control.txt").values("B 10")[:4] == [1.0, 20.5, -2.0, 5.0]
    assert [f.text() for f in tab.extent] == ["1", "20.5", "-2", "5"]  # shown, readable
    assert tab.x0.text() == f"{georef.origin[0]:.3f}".rstrip("0").rstrip(".")
    # Shift + drag: the model turned on the map about the domain's centre
    centre = georef.to_world(10.75, 1.5)
    tab.model_turned(15.0, (10.75, 1.5))
    assert window.session.georef().rotation_deg == pytest.approx(15.0)
    assert window.session.georef().to_world(10.75, 1.5) == pytest.approx(centre)
    # the background map: an item of the case
    tab.source.setCurrentText("OpenStreetMap")
    tab.show_map()
    maps = [i for i in window.session.scene if i.kind == "basemap"]
    assert len(maps) == 1 and maps[0].properties["source"] == "OpenStreetMap"
    window.act_undo.trigger()  # the map
    window.act_undo.trigger()  # the turn
    assert window.session.georef().rotation_deg == 0.0


def test_the_map_tab_says_to_open_a_case_first(window):
    tab = window.setup_panel.map_tab
    assert not tab.place_button.isEnabled() and not tab.map_button.isEnabled()
    assert "New case from map" in tab.hint.text()


def test_a_case_from_the_map(window, tmp_path, monkeypatch):
    from PySide6.QtWidgets import QFileDialog, QInputDialog

    monkeypatch.setattr(QInputDialog, "getText", lambda *a, **k: ("63.44, 10.40", True))
    case_id = window.new_case_from_map_dialog()
    session = window.session
    assert case_id and session.is_draft(case_id) and session.mode == "setup"
    tab = window.setup_panel.map_tab
    assert window.setup_panel.tabs.currentWidget() is tab
    assert session.georef(case_id).crs == "EPSG:25832"
    assert tab.create_button.isEnabled() and "draft" in tab.draft.text()
    target = tmp_path / "nidelva"
    monkeypatch.setattr(QFileDialog, "getExistingDirectory", lambda *a, **k: str(target))
    tab.create_case()
    assert not session.is_draft(case_id) and (target / "georef.json").is_file()
    assert tab.draft.text() == ""


def test_make_geo_dat_asks_where_and_moves_a_draft_there(window, tmp_path, monkeypatch, qtbot):
    import numpy as np
    from PySide6.QtWidgets import QFileDialog

    sys.path.insert(0, str(TESTS / "core"))
    from test_geo import plane, write_geotiff

    x0, y0, size, n = 269000.0, 7042000.0, 10.0, 300
    cols, rows = np.meshgrid(np.arange(n) + 0.5, np.arange(n) + 0.5)
    terrain = tmp_path / "terrain.tif"
    write_geotiff(terrain, plane(x0 + cols * size, y0 - rows * size), x0, y0, size)
    session = window.session
    case_id = session.new_case_from_map((63.425, 10.40), length=400.0)
    window.setup_panel.refresh()
    tab = window.setup_panel.map_tab
    tab.bathy_source.setCurrentIndex(tab.bathy_source.findData("file"))
    tab._bathymetry = terrain
    target = tmp_path / "nidelva"
    target.mkdir()
    (target / "notes.txt").write_text("mine")  # other files may be there
    monkeypatch.setattr(
        QFileDialog, "getSaveFileName", lambda *a, **k: (str(target / "geo.dat"), "")
    )
    tab.make_geo_dat()
    assert not session.is_draft(case_id) and session.data.case_folder(case_id) == target
    # made in a worker thread, a progress bar showing
    qtbot.waitUntil(lambda: " from terrain.tif" in panel_message(window), timeout=20000)
    assert (target / "geo.dat").is_file() and (target / "georef.json").is_file()
    assert (target / "notes.txt").read_text() == "mine"
    assert str(target) in panel_message(window)
    # its points: an item in the scene, with their spacing in the properties
    geodat = next(i for i in session.scene if i.kind == "geodat")
    window.scene_tree.select(geodat.id)
    assert "Spacing: 4 m (a regular grid)" in window.properties.info.text()


def panel_message(window) -> str:
    return window.setup_panel.message.text()


def test_the_results_leave_the_map(window):
    session = window.session
    session.new_case_from_map((63.44, 10.40))
    window._apply_mode()
    basemap = next(i for i in session.scene if i.kind == "basemap")
    assert session.mode == "setup" and basemap.visible
    window.act_mode_results.trigger()
    assert session.scene.get(basemap.id).visible  # the map stays under the results
    window.act_mode_setup.trigger()
    assert session.scene.get(basemap.id).visible


def test_a_clicked_item_brings_its_properties(window):
    session = window.session
    case = session.new_case_from_map((63.44, 10.40))
    window._apply_mode()
    basemap = next(i for i in session.scene if i.kind == "basemap")
    setup_dock, properties = window.docks["setup"], window.docks["properties"]
    tree = window.scene_tree

    def click(item_id):
        tree.select(item_id)
        tree.itemClicked.emit(tree._entry(item_id), 0)

    # the set-up: the case stays with the Set-up panel, the map shows its properties
    click(case)
    assert setup_dock.widget().isVisible() and not properties.widget().isVisible()
    click(basemap.id)
    assert properties.widget().isVisible() and window.properties.item_id == basemap.id
    # the results, with the Set-up panel brought back over the properties
    window.act_mode_results.trigger()
    setup_dock.show()
    setup_dock.raise_()
    click(case)
    assert properties.widget().isVisible() and window.properties.item_id == case


def test_the_map_tab_starts_at_trondheim(window):
    panel = window.setup_panel
    panel.tabs.setCurrentWidget(panel.map_tab)
    session = window.session
    case = next(i.id for i in session.scene if i.kind == "case")
    assert session.is_draft(case) and session.georef(case).crs == "EPSG:25832"
    basemap = next(i for i in session.scene if i.kind == "basemap")
    assert basemap.properties["source"] == "Kartverket topo"


def test_the_sea_tab(window, tmp_path, monkeypatch, qtbot):
    """Storms, the window from one, the spectrum and ctrl.txt (a stand-in THREDDS)."""
    import datetime as dt
    import json

    import numpy as np
    from metocean_fake import FakeThredds, nora3_month, spectra_day

    from lagoonviz.core.geo import metocean as mo
    from lagoonviz.core.geo.crs import transformer

    fake = FakeThredds()
    nora3_month(fake, 2020, 1, storm_hour=100)
    spectra_day(fake, dt.date(2020, 1, 5))
    monkeypatch.setattr(mo, "FETCH", fake)
    case = tmp_path / "fjord"
    case.mkdir()
    (case / "control.txt").write_text("B 1 50\nB 10 0 2000 0 1000 -30 30\nC 11 6\nC 14 7\n")
    (case / "ctrl.txt").write_text("A 10 5\nF 60 30.0\n")
    x, y = transformer("EPSG:4326", "EPSG:25832")(3.95, 61.0)
    georef = {"version": 1, "crs": "EPSG:25832", "origin": [float(x), float(y)]}
    (case / "georef.json").write_text(json.dumps(georef))
    window.open_case(case)
    tab = window.setup_panel.sea_tab
    tab.refresh()
    assert "x min" in tab.where.text() and tab.storm_button.isEnabled()
    tab.first_year.setValue(2020)
    tab.last_year.setValue(2020)
    tab.count.setValue(2)
    tab.storm_button.click()
    qtbot.waitUntil(lambda: tab._runner is None and tab.table.rowCount() == 2, timeout=20000)
    assert tab.table.item(0, 0).text() == "2020-01-05 04:00"
    tab.table.selectRow(0)
    assert tab.start.dateTime().toSecsSinceEpoch() == int(
        dt.datetime(2020, 1, 4, 4, tzinfo=dt.UTC).timestamp()
    )
    tab.spectrum_button.click()
    qtbot.waitUntil(lambda: tab._runner is None, timeout=20000)
    assert (case / "spectrum-file-2d.dat").is_file()
    ctrl = window.session.input_file("ctrl.txt")
    assert ctrl.values("B 85") == [11.0]
    assert np.isclose(ctrl.values("B 93")[0], 4.0, atol=0.01)


def test_the_sea_tab_anywhere(window, tmp_path, monkeypatch, qtbot):
    """Away from Norway: Open-Meteo's storms and waves (JONSWAP), a river's
    hydrograph, tide gauges (stand-ins for the services)."""
    import json

    from openmeteo_fake import FakeGauges, FakeOpenMeteo

    from lagoonviz.core.geo import gauges as gg
    from lagoonviz.core.geo import openmeteo as om
    from lagoonviz.core.geo.crs import transformer

    monkeypatch.setattr(om, "FETCH", FakeOpenMeteo())
    monkeypatch.setattr(gg, "FETCH", FakeGauges(40.45, -73.85))
    case = tmp_path / "bay"
    case.mkdir()
    (case / "control.txt").write_text("B 1 50\nB 10 0 2000 0 1000 -20 10\nC 11 6\nC 14 7\n")
    (case / "ctrl.txt").write_text("A 10 5\nF 60 0.0\n")
    x, y = transformer("EPSG:4326", "EPSG:32618")(-73.85, 40.45)
    georef = {"version": 1, "crs": "EPSG:32618", "origin": [float(x), float(y)]}
    (case / "georef.json").write_text(json.dumps(georef))
    window.open_case(case)
    tab = window.setup_panel.sea_tab
    tab.refresh()
    assert "Open-Meteo" in tab.where.text() and tab.storm_box.title() == "Storms (ERA5)"
    assert tab.spectrum_button.text() == "Set the waves (JONSWAP)"
    tab.first_year.setValue(2023)
    tab.last_year.setValue(2023)
    tab.count.setValue(2)
    tab.storm_button.click()
    qtbot.waitUntil(lambda: tab._runner is None and tab.table.rowCount() == 2, timeout=20000)
    assert tab.table.item(0, 0).text() == "2023-03-10 06:00"
    tab.table.selectRow(0)
    tab.spectrum_button.click()
    qtbot.waitUntil(lambda: tab._runner is None, timeout=20000)
    assert window.session.input_file("ctrl.txt").values("B 85") == [2.0]
    tab.river_button.click()
    qtbot.waitUntil(lambda: tab._runner is None, timeout=20000)
    assert (case / "hydrograph.dat").is_file()
    tab.gauge_button.click()
    qtbot.waitUntil(lambda: tab._runner is None and tab.gauge_list.count() > 0, timeout=20000)
    assert "Kartverket" in tab.gauge_list.itemText(0)
    tab.gauge_list.setCurrentIndex(1)  # NOAA
    tab.levels_button.click()
    qtbot.waitUntil(lambda: tab._runner is None, timeout=20000)
    assert any(i.name.startswith("NOAA gauge") for i in window.session.scene)
    # back to MET Norway by hand: the spectrum file again
    tab.source.setCurrentIndex(tab.source.findData("MET Norway"))
    assert tab.spectrum_button.text() == "Write spectrum-file-2d.dat"


def test_a_turned_model_makes_geo_dat_again_before_divemesh(
    window, flume, tmp_path, qtbot, monkeypatch
):
    """geo.dat made, the model turned on the map: the Map tab says geo.dat no longer
    fits, and Run DIVEMesh makes it again first."""
    import numpy as np

    script = tmp_path / "fake_divemesh.py"
    script.write_text(FAKE.format(tests=str(TESTS)))
    program = tmp_path / "DiveMESH"
    program.write_text(f"#!/bin/sh\nexec {sys.executable} {script}\n")
    program.chmod(program.stat().st_mode | stat.S_IEXEC)
    monkeypatch.setenv("LAGOON_DIVEMESH", str(program))
    window.open_case(flume)
    session = window.session
    session.set_georef("EPSG:25832", origin=(570000.0, 7030000.0))
    xs, ys = np.meshgrid(np.arange(-20.0, 31.0, 1.0), np.arange(-20.0, 23.0, 1.0))
    soundings = np.column_stack(
        [570000.0 + xs.ravel(), 7030000.0 + ys.ravel(), -1.0 + 0.1 * xs.ravel()]
    )
    survey = tmp_path / "survey.xyz"
    np.savetxt(survey, soundings)
    session.make_geo_dat(survey, crs="EPSG:25832")
    tab = window.setup_panel.map_tab
    tab.refresh()
    assert tab.geo_state.isHidden()
    tab.model_turned(20.0, (5.0, 1.0))
    tab.refresh()
    assert not tab.geo_state.isHidden() and "turned 0°" in tab.geo_state.text()
    panel = window.setup_panel
    panel.stale_geo_dat = "remake"
    assert panel.run_divemesh()
    qtbot.waitUntil(
        lambda: panel.process is None and "--DONE--" in panel.log.toPlainText(), timeout=30000
    )
    assert session.geo_dat_state() == "" and session.geo_dat_recipe()["rotation_deg"] == 20.0
    assert tab.geo_state.isHidden()


def test_the_coast_of_a_download_and_files_over_it(window, tmp_path, monkeypatch):
    """The Map tab's *Near the coast*, Dean's A, *Trust the sea from*, the files over
    the download and ICESat-2 go to the geo.dat job."""
    from PySide6.QtWidgets import QDialog, QFileDialog

    from lagoonviz.gui.panels import map_setup

    session = window.session
    session.new_case_from_map((63.425, 10.40), length=400.0)
    window.setup_panel.refresh()
    tab = window.setup_panel.map_tab
    tab.bathy_source.setCurrentIndex(tab.bathy_source.findData("best"))
    assert tab.web_row.isVisibleTo(tab) and not tab.file_row.isVisibleTo(tab)
    assert tab.nearshore.currentData() == "dean" and tab.dean_a.value() == 0.1
    chart = tmp_path / "chart.xyz"
    chart.write_text("271200 7040200 -5\n")
    atl24 = tmp_path / "ATL24_x.h5"
    atl24.write_bytes(b"")
    monkeypatch.setattr(
        QFileDialog, "getOpenFileNames", lambda *a, **k: ([str(chart), str(atl24)], "")
    )

    def asked(dialog):
        dialog.crs.setText("EPSG:32643")
        dialog.depths.setChecked(True)
        dialog.below_msl.setValue(2.6)
        return QDialog.Accepted

    monkeypatch.setattr(map_setup.OverlayDialog, "exec", asked)
    tab.add_overlay()
    assert tab.overlay_list.count() == 2
    assert "ICESat-2" in tab.overlay_list.item(1).text()
    assert "2.6 m below MSL" in tab.overlay_list.item(0).text()
    tab.nearshore.setCurrentIndex(tab.nearshore.findData("smooth"))
    assert not tab.web_form.isRowVisible(tab.dean_a)
    tab.trust_depth.setValue(2.5)
    tab.icesat2.setChecked(True)
    given = {}

    def job(source, **options):
        given.update(options, source=source)
        raise ValueError("stop here")

    monkeypatch.setattr(session, "geo_dat_job", job)
    monkeypatch.setattr(tab, "_where_geo_dat", lambda: tmp_path / "geo.dat")
    tab.make_geo_dat()
    assert given["source"] == "best" and given["nearshore"] == "smooth"
    assert given["dean_a"] is None and given["trust_depth"] == 2.5 and given["icesat2"]
    assert given["overlays"][0] == {
        "path": str(chart),
        "crs": "EPSG:32643",
        "depths": True,
        "below_msl": 2.6,
    }
    tab.overlay_list.setCurrentRow(0)
    tab.remove_overlay()
    assert tab.overlay_list.count() == 1 and tab._overlays[0]["path"] == str(atl24)


def test_the_porous_tab_adds_draws_and_changes_structures(window, flume):
    window.open_case(flume)
    panel = window.setup_panel
    tab = panel.porous_tab
    assert panel.tabs.tabText(panel.tabs.indexOf(tab)) == "Porous"
    # NHFLOW (A 10 5): its shapes, one material for all (B 201)
    assert tab.kind_box.count() == 6 and tab.kind_box.currentData() == "B 210"
    assert "B 201" in tab.note.text() and not tab.remove_button.isEnabled()
    assert tab.add() == 0
    ctrl = window.session.input_file("ctrl.txt")
    assert ctrl.values("B 201") == [0.4, 0.05, 500.0, 2.0] and len(ctrl.entries("B 210")) == 1
    assert tab.tree.topLevelItemCount() == 1 and tab.remove_button.isEnabled()
    assert len(tab._shape_fields) == 6 and tab._material_fields[0].text() == "0.4"
    assert "B 201" in tab.material_box.title()
    # a value changed in the form
    tab._shape_fields[1].setText("6")
    tab._commit_shape()
    assert window.session.input_file("ctrl.txt").values("B 210")[1] == 6.0
    tab._material_fields[0].setText("0.45")
    tab._commit_material()
    assert window.session.input_file("ctrl.txt").values("B 201")[0] == 0.45
    tab._material_fields[0].setText("1.5")
    tab._commit_material()
    assert "porosity" in panel.message.text()
    # drawn in the 2D view: a vertical cylinder in the area, snapped to the cells
    index = tab.kind_box.findData("B 213")
    tab.kind_box.setCurrentIndex(index)
    tab.draw()
    window.viewport.areaDrawn.emit("porous structure", (3.02, 4.1, 0.4, 1.6))
    values = window.session.input_file("ctrl.txt").values("B 213")
    assert values == [3.5, 1.0, 0.0, 0.6, 0.5]
    assert tab.tree.topLevelItemCount() == 2 and tab.selected().kind.code == "B 213"
    tab.remove_selected()
    assert "B 213" not in window.session.input_file("ctrl.txt")
    assert tab.tree.topLevelItemCount() == 1
    # SFLOW has none
    window.session.set_input("ctrl.txt", "A 10", [2])
    assert not tab.add_button.isEnabled() and "no porous structures" in tab.note.text()


def test_the_solids_tab_adds_divemesh_and_direct_forcing_solids(window, flume):
    window.open_case(flume)
    panel = window.setup_panel
    tab = panel.solids_tab
    assert panel.tabs.tabText(panel.tabs.indexOf(tab)) == "Solids"
    # NHFLOW: its direct forcing first, then DIVEMesh's
    assert tab.kind_box.currentData() == "A 581" and tab.kind_box.findData("S 33") > 0
    assert "A 580" in tab.note.text() and not tab.material_box.isVisibleTo(tab)
    tab.kind_box.setCurrentIndex(tab.kind_box.findData("S 33"))
    tab.draw()
    window.viewport.areaDrawn.emit("solid", (4.0, 5.0, 0.5, 1.5))
    control = window.session.input_file("control.txt")
    assert control.values("S 33") == [4.5, 1.0, 0.5]
    assert "Run DIVEMesh" in panel.message.text()
    assert len(tab._shape_fields) == 3 and tab.selected().kind.file == "control.txt"
    # an area drawn for the Porous tab is not the Solids tab's
    window.viewport.areaDrawn.emit("porous structure", (1.0, 2.0, 0.5, 1.5))
    assert "B 210" not in window.session.input_file("ctrl.txt")
    tab._shape_fields[2].setText("0.75")
    tab._commit_shape()
    assert window.session.input_file("control.txt").values("S 33") == [4.5, 1.0, 0.75]
    tab.add("A 584")
    assert window.session.input_file("ctrl.txt").values("A 584")[4] > 0
    assert tab.tree.topLevelItemCount() == 2
    tab.remove_selected()
    assert "A 584" not in window.session.input_file("ctrl.txt")


def test_the_river_tab_lays_a_corridor_and_makes_its_grid(window, flume):
    window.open_case(flume)
    panel = window.setup_panel
    tab = panel.river_tab
    assert panel.tabs.tabText(panel.tabs.indexOf(tab)) == "River"
    assert not tab.form.isEnabled() and tab.start_button.isEnabled()
    tab.start()
    corridor = window.session.corridor()
    assert corridor is not None and tab.edit_button.isChecked() and tab.form.isEnabled()
    assert tab.points.value() == 12 and "cells" in tab.quality.text()
    # a point dragged in the 2D view
    window.viewport.corridorPointMoved.emit(panel.case_id, "left", 5, 5.0, 1.9)
    assert tuple(window.session.corridor().left[5]) == (5.0, 1.9)
    tab.points.setValue(16)
    assert len(window.session.corridor().left) == 16
    tab.curve.setCurrentIndex(tab.curve.findData("bspline"))
    tab.curve.activated.emit(tab.curve.currentIndex())
    assert window.session.corridor().curve == "bspline"
    tab.across.setValue(8)
    tab.along.setValue(40)
    assert window.session.corridor().cells_across == 8
    tab.make_grid()
    assert "curvgrid.dat written (40 x 8 cells)" in panel.message.text()
    assert window.session.input_file("ctrl.txt").values("G 50") == [2.0]
    # Esc in the 3D view: the button follows
    window.viewport.corridorEditingStopped.emit()
    assert not tab.edit_button.isChecked()
    # an area drawn for the corridor
    window.viewport.areaDrawn.emit("river corridor", (1.0, 9.0, 0.5, 1.5))
    assert window.session.corridor().left[0].tolist() == [1.0, 1.5]
    tab.remove()
    assert window.session.corridor() is None and not tab.form.isEnabled()


def test_the_blocks_tab_lays_blocks_and_makes_their_grid(window, flume):
    window.open_case(flume)
    panel = window.setup_panel
    tab = panel.blocks_tab
    assert panel.tabs.tabText(panel.tabs.indexOf(tab)) == "Blocks"
    assert not tab.form.isEnabled() and tab.start_button.isEnabled()
    tab.start()
    topology = window.session.multiblock()
    assert topology is not None and tab.drag_button.isChecked() and tab.form.isEnabled()
    assert len(topology.blocks) == 1 and "cells" in tab.quality.text()
    # Add block: an edge clicked in the 2D view
    tab.extend_button.setChecked(True)
    assert not tab.drag_button.isChecked()
    tab.depth.setValue(3.0)
    tab.depth_cells.setValue(8)
    window.viewport.blockEdgePicked.emit(panel.case_id, "extend", "b0_e", 10.0, 1.0)
    topology = window.session.multiblock()
    assert len(topology.blocks) == 2 and tab.block.count() == 2
    # a corner dragged
    corner = topology.edges["b0_e_x"].end
    window.viewport.blockVertexMoved.emit(panel.case_id, corner, 13.2, 2.2)
    assert tuple(window.session.multiblock().vertices[corner]) == (13.2, 2.2)
    # the water level on block 1's +i face
    tab.block.setCurrentIndex(1)
    row = tab.faces["i1"]
    assert row.isEnabled() and row.kind.isEnabled()
    row.value.setValue(0.6)
    row.kind.setCurrentIndex(row.kind.findData("outflow"))
    row.kind.activated.emit(row.kind.currentIndex())
    assert window.session.multiblock().blocks[1].bc_value("i1") == 0.6
    assert tab.make_button.isEnabled()
    tab.make_grid()
    assert "blocks.dat written (2 blocks" in panel.message.text()
    assert (flume / "blocks.dat").is_file()
    assert window.session.input_file("ctrl.txt").values("G 50") == [1.0]
    assert window.session.input_file("control.txt").values("R 20") == [1.0]
    # Esc in the 3D view: the buttons follow
    window.viewport.blocksEditingStopped.emit()
    assert not any(b.isChecked() for b in (tab.drag_button, tab.cut_button, tab.extend_button))
    tab.remove()
    assert window.session.multiblock() is None and not tab.form.isEnabled()


def test_the_river_tab_follows_the_thalweg(window, flume):
    import numpy as np

    x, y = np.linspace(0, 10, 41), np.linspace(0, 2, 9)
    gx, gy = np.meshgrid(x, y)
    z = 0.2 + 0.5 * (gy - 1.0) ** 2  # a channel along y = 1
    np.savetxt(flume / "geo.dat", np.column_stack([gx.ravel(), gy.ravel(), z.ravel()]))
    window.open_case(flume)
    tab = window.setup_panel.river_tab
    assert tab.auto_box.isEnabled()
    tab.margin.setValue(0.1)
    tab.auto_points.setValue(10)
    tab.follow_thalweg()
    corridor = window.session.corridor()
    assert corridor is not None and len(corridor.left) == 10 and tab.edit_button.isChecked()
    assert abs(float(np.mean(corridor.left[:, 1] + corridor.right[:, 1])) / 2 - 1.0) < 0.2


def test_the_river_tab_draws_the_river_first_then_its_bed_and_creates_it(window, flume):
    import numpy as np

    window.open_case(flume)
    tab = window.setup_panel.river_tab
    assert tab.trace_button.isEnabled() and not tab.continue_button.isEnabled()
    assert not tab.bed_button.isEnabled() and not tab.create_button.isEnabled()
    tab.trace_button.setChecked(True)  # no GPU here: the view says so and stops
    assert not tab.trace_button.isChecked()
    # the view traced the river: a line only
    line = [(0.5, 1.3), (5.0, 0.7), (9.5, 1.2)]
    window.viewport.lineDrawn.emit("river", line)
    assert window.session.corridor() is None
    assert window.session.river_line() == line
    assert tab.continue_button.isEnabled() and tab.create_button.isEnabled()
    window.viewport.lineDrawn.emit("river", None)  # stopped without a line: kept
    assert window.session.river_line() == line
    # continued: the view gets the line to take up
    taken = []
    window.viewport.start_line_drawing = lambda case, purpose, points=None: taken.append(points)
    tab.drawLine.disconnect()
    tab.drawLine.connect(window.viewport.start_line_drawing)
    tab.continue_button.setChecked(True)
    assert taken == [line]
    window.viewport.lineDrawn.emit("river", [*line, (9.8, 1.0)])
    assert window.session.river_line()[-1] == (9.8, 1.0)
    assert not tab.continue_button.isChecked()

    # Domain and geo.dat: the domain around the river, then the Map tab's geo.dat
    def make_geo_dat(then=None):
        x, y = np.linspace(-2, 12, 57), np.linspace(-2, 4, 25)
        gx, gy = np.meshgrid(x, y)
        z = 0.2 + 0.5 * (gy - 1.0) ** 2  # a channel along y = 1
        np.savetxt(flume / "geo.dat", np.column_stack([gx.ravel(), gy.ravel(), z.ravel()]))
        window.session.data.case_setup(window.setup_panel.case_id).geo_changed()
        then()
        return True

    window.setup_panel.map_tab.make_geo_dat = make_geo_dat
    tab.width.setValue(1.6)
    tab.domain_and_bed()
    x0, x1, y0, y1 = window.session.input_file("control.txt").values("B 10")[:4]
    assert x0 < 0.5 and x1 > 9.5 and y0 < 0 and y1 > 2  # beyond the river
    # Create river: the corridor and its grid, fitted to the bed
    tab.margin.setValue(0.1)
    tab.create_river()
    created = window.session.corridor()
    assert created is not None and created.extra["made"]["how"] == "fitted"
    centre = created.centreline(50)[5:-5]
    assert np.abs(centre[:, 1] - 1.0).max() < 0.25
    assert [i for i in window.session.scene if i.kind == "corridor"]


def test_the_corridor_is_an_item_with_a_reset(window, flume):
    from PySide6.QtWidgets import QPushButton

    window.open_case(flume)
    item = window.add_item("corridor")  # Add → River corridor
    assert window.session.scene.get(item).kind == "corridor"
    tab = window.setup_panel.river_tab
    assert tab.reset_button.isEnabled() and tab.form.isEnabled()
    start = window.session.corridor().left.copy()
    window.session.move_corridor_point("left", 2, 3.0, 1.95)
    window.properties.show_item(item)
    reset = next(b for b in window.properties.findChildren(QPushButton) if b.text() == "Reset")
    reset.click()
    assert (window.session.corridor().left == start).all()
    assert "points on each bank" in window.properties.info.text()
    window.session.remove_item(item)
    assert window.session.corridor() is None and not tab.form.isEnabled()


def test_the_rotation_sliders_show_the_turn_and_keep_it_when_let_go(window, flume):
    window.open_case(flume)
    session = window.session
    tab = window.setup_panel.map_tab
    assert not tab.rotation_slider.isEnabled()  # no georeference yet
    session.set_georef("EPSG:25833", origin=(270000.0, 7040000.0), rotation_deg=30.0)
    case = window.setup_panel.case_id
    assert tab.rotation_slider.isEnabled() and tab.rotation_slider.value() == 300
    # dragged: shown turned, nothing written
    tab.rotation_slider.setSliderDown(True)
    tab.rotation_slider.setValue(400)
    assert session.georef().rotation_deg == 30.0
    assert session.data.backgrounds.shown_georef(case).rotation_deg == 40.0
    assert tab.rotation.value() == 40.0
    # let go: written, the trial over
    tab.rotation_slider.setSliderDown(False)
    tab.rotation_slider.sliderReleased.emit()
    assert session.georef().rotation_deg == 40.0
    # about the middle of the view (no 3D view here: the domain's centre)
    tab.view_centre = lambda: (2.0, 0.5)
    spot = session.georef().to_world(2.0, 0.5)
    tab.rotation_slider.setValue(450)  # by the keys: kept at once
    assert session.georef().rotation_deg == 45.0
    assert session.georef().to_world(2.0, 0.5) == pytest.approx(spot)
    assert session.data.backgrounds.shown_georef(case) == session.georef()
    # the domain's slider: about the domain's centre
    centre = session.georef().to_world(5.0, 1.0)
    tab.turn_slider.setValue(-200)  # moved by the keys: kept at once
    assert session.georef().rotation_deg == -20.0
    assert session.georef().to_world(5.0, 1.0) == pytest.approx(centre)
    turned = []
    window.viewport.statusMessage.connect(turned.append)
    tab.north_button.click()  # the 3D view turns north up (no GPU here: said so)


def test_the_river_tab_squares_the_grid(window, flume):
    window.open_case(flume)
    tab = window.setup_panel.river_tab
    assert not tab.square_button.isEnabled()
    tab.start()
    assert window.session.corridor().sections == "even" and tab.square_button.isEnabled()
    window.session.set_corridor_options(smoothing=40)
    assert tab.square_button.text() == "Tidy the grid"
    tab.square_button.click()
    corridor = window.session.corridor()
    assert corridor.sections == "normal" and corridor.smoothing == 20
    assert tab.sections.currentData() == "normal"


def test_one_grid_in_the_river_tab_and_a_note_when_curvgrid_is_another(window, flume):
    window.open_case(flume)
    tab = window.setup_panel.river_tab
    tab.start()
    tab.make_grid()
    assert "curvgrid.dat is another grid" not in tab.quality.text()
    window.session.move_corridor_point("left", 3, 3.0, 1.95)
    assert "curvgrid.dat is another grid" in tab.quality.text()


def test_the_input_check_from_the_panel(window, flume):
    """Check: both input files, DIVEMesh's grid, the cost, in a window."""
    window.open_case(flume)
    panel = window.setup_panel
    assert panel.check_button.isEnabled()
    report = panel.check_input()
    assert report is not None and any("run DIVEMesh first" in p.message for p in report.errors)
    assert panel._check_dialog.isVisible()
    assert "Input check" in panel.message.text()


def test_the_blocks_tab_spaces_edges_follows_the_water_and_imports(window, tmp_path):
    import numpy as np

    from lagoonviz.core import multiblock as mb

    case = tmp_path / "estuary"
    case.mkdir()
    (case / "control.txt").write_text("C 11 2\nB 1 1.0\nB 10 0.0 1.0 0.0 1.0 0.0 5.0\nR 20 1\n")
    (case / "ctrl.txt").write_text("A 10 5\nG 50 1\nF 60 10.0\n")
    xs, ys = np.arange(-200.0, 1201.0, 10.0), np.arange(-400.0, 701.0, 10.0)
    gx, gy = np.meshgrid(xs, ys)
    water = (gy < 0) | ((np.abs(gx - 500) < 100) & (gy < 300))  # the sea and a bay
    np.savetxt(
        case / "geo.dat", np.column_stack([gx.ravel(), gy.ravel(), np.where(water, 0, 20).ravel()])
    )
    window.open_case(case)
    panel = window.setup_panel
    tab = panel.blocks_tab
    window.session.start_multiblock([0, 1000, -300, 150], cells=(20, 10))
    tab.refresh()
    assert not tab.import_button.isEnabled()  # no blocks.dat yet
    # an edge picked in the 2D view
    tab.pick_button.setChecked(True)
    window.viewport.blockEdgePicked.emit(panel.case_id, "pick", "b0_n", 500.0, 150.0)
    assert tab.edge.currentText() == "b0_n"
    assert tab.follow_button.isEnabled() and tab.level.value() == 10.0
    # along the water
    tab.margin.setValue(40.0)
    tab.radius.setValue(60.0)
    tab.follow_water()
    edge = window.session.multiblock().edges["b0_n"]
    assert edge.water == {"margin": 40.0, "radius": 60.0}
    assert "follows the water" in tab.water_info.text() and tab.unfollow_button.isEnabled()
    tab.stop_following()
    assert window.session.multiblock().edges["b0_n"].water is None
    # spaced by the cells at its ends, across the block
    tab.space_start.setValue(20.0)
    tab.space_end.setValue(80.0)
    tab.spaced.setChecked(True)
    t = window.session.multiblock()
    assert t.edges["b0_n"].spacing == {"start": 20.0, "end": 80.0}
    assert t.edges["b0_s"].spacing == {"start": 20.0, "end": 80.0}
    assert t.edges["b0_s"].cells == t.edges["b0_n"].cells and not tab.cells.isEnabled()
    assert "cells now" in tab.sizes.text()
    tab.fit_cells()
    # an arc
    tab.edge.setCurrentIndex(tab.edge.findText("b0_w"))
    tab.curve.setCurrentIndex(tab.curve.findData("arc"))
    tab.curve.activated.emit(tab.curve.currentIndex())
    assert window.session.multiblock().edges["b0_w"].curve == "arc"
    # the open face (a tide edge)
    row = tab.faces["i0"]
    row.kind.setCurrentIndex(row.kind.findData("open"))
    row.kind.activated.emit(row.kind.currentIndex())
    assert window.session.multiblock().blocks[0].bc_type("i0") == "open"
    assert not row.value.isEnabled()
    # a blocks.dat made elsewhere, imported
    tab.make_grid()
    assert (case / "blocks.dat").is_file()
    tab.remove()
    tab.refresh()
    assert tab.import_button.isEnabled()
    tab.import_blocks()
    topology = window.session.multiblock()
    assert topology is not None and all(e.curve == "nodes" for e in topology.edges.values())
    assert "read from blocks.dat" in panel.message.text()
    assert topology.blocks[0].bc_type("i0") == "open"
    assert mb.check(topology) == [] or not [i for i in mb.check(topology) if i.level == "error"]
