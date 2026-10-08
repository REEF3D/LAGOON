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
