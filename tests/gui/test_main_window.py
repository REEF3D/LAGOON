"""GUI tests (offscreen). The GUI must only drive the core through Session."""

import pytest
from PySide6.QtCore import QSettings, Qt

from lagoonviz.core import Session
from lagoonviz.gui.main_window import MainWindow


@pytest.fixture
def window(qtbot, tmp_path):
    settings = QSettings(str(tmp_path / "settings.ini"), QSettings.IniFormat)
    win = MainWindow(Session(), settings=settings)
    win.ask_before_discard = False  # no modal dialog when the test closes the window
    qtbot.addWidget(win)
    win.show()
    return win


def test_panels_exist(window):
    assert set(window.docks) == {"scene", "properties", "plots", "timeline", "console"}
    assert window.centralWidget().objectName() == "viewport"
    assert window.windowTitle() == "Untitled - LAGOON"


def test_add_action_updates_tree_and_selects(window):
    window.add_actions["free_surface"].trigger()
    tree = window.scene_tree.tree
    assert tree.topLevelItemCount() == 1
    assert window.scene_tree.selected_id() == "free_surface-1"
    assert window.properties.item_id == "free_surface-1"
    assert window.windowTitle().endswith("* - LAGOON")


def test_undo_redo_actions(window):
    window.add_actions["gauge"].trigger()
    assert window.act_undo.isEnabled() and window.act_undo.text() == "&Undo Add Gauge"
    window.act_undo.trigger()
    assert window.scene_tree.tree.topLevelItemCount() == 0
    assert window.act_redo.isEnabled()
    window.act_redo.trigger()
    assert window.scene_tree.tree.topLevelItemCount() == 1


def test_visibility_checkbox_runs_command(window):
    item_id = window.session.add_item("slice")
    entry = window.scene_tree.tree.topLevelItem(0)
    entry.setCheckState(0, Qt.Unchecked)
    assert window.session.scene.get(item_id).visible is False
    assert window.session.stack.undo_label == "Hide"


def test_properties_editor_runs_command(window):
    item_id = window.session.add_item("free_surface")
    window.properties.editors["colormap"].setCurrentText("viridis")
    assert window.session.scene.get(item_id).properties["colormap"] == "viridis"
    window.properties.editors["opacity"].setValue(0.4)
    assert window.session.scene.get(item_id).properties["opacity"] == pytest.approx(0.4)
    # a change made in a script shows up in the panel
    window.session.set_property(item_id, "colormap", "greys")
    assert window.properties.editors["colormap"].currentText() == "greys"


def test_console_runs_python_and_logs_commands(window):
    window.console.run("session.add_item('gauge', x=2.5)")
    assert "gauge-1" in window.session.scene
    log = window.console.log.toPlainText()
    assert "session.add_item('gauge', x=2.5)" in log
    window.console.run("1/0")
    assert "ZeroDivisionError" in window.console.log.toPlainText()


def test_timeline_sets_time(window):
    window.timeline.slider.setValue(500)
    assert window.session.scene.time == pytest.approx(30.0)
    window.session.set_time(15.0)
    assert window.timeline.slider.value() == 250


def test_save_open_and_export_script(window, tmp_path):
    item_id = window.session.add_item("free_surface", colormap="balance")
    path = tmp_path / "case.lagoon-session"
    assert window.save_session_to(path)
    assert window.windowTitle() == "case.lagoon-session - LAGOON"

    window.session.remove_item(item_id)
    window.open_session(path)
    assert item_id in window.session.scene
    assert not window.dirty

    window.session.set_property(item_id, "opacity", 0.5)
    script = tmp_path / "replay.py"
    window.export_script(script)
    replayed = Session.from_script(script.read_text())
    assert replayed.scene.to_dict() == window.session.scene.to_dict()


def test_layout_saved_between_runs(qtbot, tmp_path):
    settings = QSettings(str(tmp_path / "s.ini"), QSettings.IniFormat)
    first = MainWindow(Session(), settings=settings)
    qtbot.addWidget(first)
    first.docks["console"].hide()
    first.close()
    second = MainWindow(Session(), settings=settings)
    qtbot.addWidget(second)
    assert second.docks["console"].isHidden()
    second.reset_layout()
    assert not second.docks["console"].isHidden()
