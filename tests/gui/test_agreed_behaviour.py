"""Agreed behaviour (part 1 of the regression suite).

One test per behaviour that was asked for and accepted, named after it, so a failure
says which agreement broke. Behaviours already covered elsewhere are listed here
with the test that guards them, so this file is the index:

In test_main_window.py:

* opening a case shows its free surface: test_opening_a_case_shows_its_free_surface
* timeline click jumps to the click: test_timeline_click_jumps_to_the_click
* holding the slider pauses playing: test_holding_the_slider_pauses_playing
* exaggeration slider and Sync: test_case_exaggeration_slider_and_sync
* colour map and Colour by independent: test_bed_colour_map_and_colour_by
* full screen, only view and player: test_full_screen_viewer_shows_only_the_view_and_the_player
* automatic range shown, kept when unticked: test_properties_show_the_automatic_range
* reload follows a running simulation: test_reload_follows_a_running_simulation
* compare two cases side by side: test_compare_two_cases_side_by_side
* theme switches at once, remembered: test_theme_switches_now_and_is_remembered

Elsewhere:

* 2D view does not rotate, orbit through the poles: render/test_controls.py
* looks (colours, lights, shadows, views): render/test_reference_images.py
* window structure, properties of every kind: gui/test_snapshots.py
"""

import pytest
from PySide6.QtCore import QSettings
from PySide6.QtGui import QKeySequence

from lagoonviz.core import Session
from lagoonviz.gui.main_window import MainWindow


@pytest.fixture
def window(qtbot, tmp_path):
    settings = QSettings(str(tmp_path / "settings.ini"), QSettings.IniFormat)
    win = MainWindow(Session(), settings=settings)
    win.ask_before_discard = False
    qtbot.addWidget(win)
    win.show()
    return win


def test_full_screen_brings_back_only_the_panels_that_were_open(window):
    """Leaving full screen restores the panels as they were, closed ones stay closed."""
    window.docks["console"].hide()
    window.toggle_full_screen(True)
    window.toggle_full_screen(False)
    shown = {name for name, dock in window.docks.items() if not dock.isHidden()}
    assert "console" not in shown
    assert {"scene", "properties", "timeline"} <= shown


def test_closing_in_full_screen_keeps_the_panels_for_next_time(qtbot, tmp_path):
    settings = QSettings(str(tmp_path / "settings.ini"), QSettings.IniFormat)
    first = MainWindow(Session(), settings=settings)
    first.ask_before_discard = False
    qtbot.addWidget(first)
    first.show()
    first.toggle_full_screen(True)
    first.close()
    second = MainWindow(Session(), settings=settings)
    qtbot.addWidget(second)
    second.show()
    for name in ("scene", "properties", "timeline"):
        assert not second.docks[name].isHidden(), name
    assert not second.toolbar.isHidden()


def test_full_screen_keys(window):
    """Ctrl+Shift+F (Cmd+Shift+F on a Mac; not the macOS system shortcut), Esc leaves."""
    assert window.act_full_screen.shortcut() == QKeySequence("Ctrl+Shift+F")
    assert window.act_leave_full_screen.shortcut() == QKeySequence("Esc")


def test_defaults_of_a_new_case(window, reef3d_run):
    """Free surface in the REEF3D colours by eta with dry land hidden; plain sand bed
    with wet land shown; one vertical scale for the whole case."""
    window.open_case(reef3d_run)
    scene = window.session.scene
    surface = scene.get("free_surface-1").properties
    assert (surface["variable"], surface["colormap"]) == ("eta", "reef3d")
    assert surface["auto_range"] is True and surface["hide_dry"] is True
    bed = next(i for i in scene if i.kind == "bathymetry").properties
    assert (bed["variable"], bed["colormap"], bed["hide_wet"]) == ("none", "sand", False)
    case = scene.get("case-1").properties
    assert case["sync_exaggeration"] is True and case["exaggeration"] == 1.0


def test_light_theme_and_studio_look_by_default(window):
    from lagoonviz.gui.theme import COLORS, THEMES

    assert window.theme_actions["light"].isChecked()
    assert COLORS == THEMES["light"]
    assert window.look_actions["studio"].isChecked()


def test_colour_maps_offered(window, reef3d_run):
    """Rainbow (Turbo) and white are offered; cividis was replaced by rainbow."""
    window.open_case(reef3d_run)
    window.scene_tree.select("free_surface-1")
    combo = window.properties.editors["colormap"]
    names = [combo.itemText(i) for i in range(combo.count())]
    assert {"reef3d", "rainbow", "white", "greys", "sand"} <= set(names)
    assert "cividis" not in names


def test_slider_drag_is_one_undo_step(window, reef3d_run):
    window.open_case(reef3d_run)
    window.scene_tree.select("case-1")
    both = window.properties.editors["exaggeration"]
    for position in (100, 300, 600, 800):
        both.slider.setValue(position)
    both.slider.sliderReleased.emit()
    window.session.undo()
    assert window.session.scene.get("case-1").properties["exaggeration"] == 1.0


def test_animations_are_saved_next_to_the_case(window, reef3d_run):
    """Not in the (unwritable) start folder of the app: 'Broken pipe' before."""
    from lagoonviz.gui.export_movie import movie_folder

    window.open_case(reef3d_run)
    assert movie_folder(window.session).resolve() == reef3d_run.resolve()
