"""GUI tests (offscreen). The GUI must only drive the core through Session."""

from pathlib import Path

import numpy as np
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
    assert set(window.docks) == {
        "scene",
        "properties",
        "plots",
        "profiles",
        "timeline",
        "console",
    }
    assert window.centralWidget().objectName() == "viewports"
    assert window.viewport.objectName() == "viewport"
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


def test_timeline_click_jumps_to_the_click(window, reef3d_run, qtbot):
    from PySide6.QtCore import QPoint

    window.open_case(reef3d_run)
    slider = window.timeline.slider
    slider.resize(400, slider.height())
    slider.setValue(slider.minimum())
    x = slider.width() - 1  # the far end, away from the handle
    qtbot.mouseClick(slider, Qt.LeftButton, pos=QPoint(x, slider.height() // 2))
    assert slider.value() == slider.value_at(x) == slider.maximum()


def test_pressing_beside_the_handle_while_playing_drags(window, reef3d_run, qtbot):
    """Held beside the handle, Qt's slider stepped towards the mouse again and again
    while playing moved the handle away: it jumped back and forth, never forward."""
    from PySide6.QtCore import QPoint

    window.open_case(reef3d_run)
    timeline, slider = window.timeline, window.timeline.slider
    slider.resize(400, slider.height())
    timeline.go_to_frame(0)
    middle = QPoint(slider.width() // 2, slider.height() // 2)
    qtbot.mousePress(slider, Qt.LeftButton, pos=middle)
    assert slider.isSliderDown() and slider.value() == slider.value_at(middle.x())
    held = slider.value()
    window.session.set_time(float(timeline.times[0]))  # playing shows another frame
    timeline._advance()  # and does not step while held
    assert slider.value() == held
    qtbot.mouseRelease(slider, Qt.LeftButton, pos=middle)
    assert not slider.isSliderDown()
    assert timeline.frame_of(window.session.scene.time) == held  # plays on from here


def test_holding_the_slider_pauses_playing(window, reef3d_run):
    """Playing and dragging the slider used to fight: the handle jumped back and forth."""
    window.open_case(reef3d_run)
    timeline = window.timeline
    timeline.go_to_frame(0)
    timeline.slider.setSliderDown(True)
    timeline._advance()
    assert timeline.frame_of(window.session.scene.time) == 0  # waits while held
    timeline.slider.setValue(2)  # dragged to frame 2
    window.session.set_time(float(timeline.times[1]))  # a frame arriving meanwhile
    assert timeline.slider.value() == 2  # the handle stays under the mouse
    timeline.slider.setSliderDown(False)
    timeline.go_to_frame(2)
    timeline._advance()
    assert timeline.frame_of(window.session.scene.time) == 0  # on from there (wraps)


def test_slider_keeps_its_width_while_playing(window, reef3d_run):
    """The time label used to resize with each text, and the slider with it: the
    handle jiggled back and forth while playing."""
    window.open_case(reef3d_run)
    timeline = window.timeline
    widths = set()
    for frame in range(3):
        timeline.go_to_frame(frame)
        widths.add(timeline.label.width())
        assert timeline.label.sizeHint().width() <= timeline.label.width() + 1
    assert len(widths) == 1


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


# ----------------------------------------------------------------- phase 1: cases
def test_viewport_placeholder_without_screen(window):
    from lagoonviz.gui.panels.viewport import ViewportPlaceholder

    assert isinstance(window.viewport, ViewportPlaceholder)
    assert "offscreen" in window.viewport.reason


def test_open_case_fills_scene_and_timeline(window, reef3d_run):
    case_id = window.open_case(reef3d_run)
    assert case_id == "case-1"
    assert [i.kind for i in window.session.scene] == ["case", "bathymetry", "free_surface"]
    tree = window.scene_tree.tree
    assert tree.topLevelItemCount() == 1  # the case, with its items below it
    assert [tree.topLevelItem(0).child(i).text(0) for i in range(2)] == [
        "Bed",
        "Free surface",
    ]
    timeline = window.timeline
    assert timeline.n_frames == 3 and not timeline.end.isVisibleTo(window)
    timeline.step(1)
    assert window.session.scene.time == 1
    timeline.go_to_frame(10)
    assert window.session.scene.time == 2
    assert timeline.label.text() == "frame 2 / 2"
    timeline.slider.setValue(0)
    assert window.session.scene.time == 0
    assert "Open run" in window.act_undo.text()


def test_timeline_in_seconds_with_print_interval(window, tmp_path):
    from reef3d_fixtures import make_run

    window.open_case(make_run(tmp_path / "timed", print_interval=0.5, volume=False))
    window.timeline.go_to_frame(2)
    assert window.session.scene.time == 1.0
    assert window.timeline.label.text() == "t = 1.00 s  (frame 2 / 2)"


def test_camera_keyframes(window, reef3d_run):
    window.open_case(reef3d_run)
    session, timeline = window.session, window.timeline
    if getattr(window.viewport, "view", None) is None:  # no GPU: a saved camera instead
        session.set_camera({"position": [0.0, 0.0, 10.0], "rotation": [0.0, 0.0, 0.0, 1.0]})
    assert not timeline.path_button.isEnabled() and not window.act_camera_path.isEnabled()
    window.dirty = False
    timeline.go_to_frame(0)
    timeline.key_button.click()
    timeline.go_to_frame(2)
    window.act_add_key.trigger()
    assert [k["time"] for k in session.scene.camera_keys] == [0.0, 2.0]
    assert timeline.slider.markers == [0, 2]
    timeline.slider.grab()  # the markers paint
    assert timeline.path_button.isChecked() and window.act_camera_path.isChecked()
    assert window.windowTitle().endswith("* - LAGOON")  # keyframes are worth saving
    timeline.path_button.click()
    assert not session.scene.camera_path and not window.act_camera_path.isChecked()
    window.act_camera_path.trigger()
    assert session.scene.camera_path and timeline.path_button.isChecked()
    timeline.go_to_frame(1)
    view = getattr(window.viewport, "view", None)
    if view is not None:  # the 3D view is where the path is
        np.testing.assert_allclose(
            view.camera_state()["position"], session.camera_at()["position"], atol=1e-5
        )
    assert timeline.key_here() is None and not window.remove_camera_key()
    timeline.go_to_frame(2)
    assert window.remove_camera_key()
    assert timeline.slider.markers == [0]
    window.act_clear_keys.trigger()
    assert timeline.slider.markers == [] and not timeline.path_button.isEnabled()
    assert not window.act_remove_key.isEnabled()
    window.act_orbit.trigger()  # once around over the timeline (0 to 2 s)
    keys = session.scene.camera_keys
    assert len(keys) == 13 and keys[0]["time"] == 0.0 and keys[-1]["time"] == 2.0
    assert timeline.path_button.isChecked() and len(timeline.slider.markers) == 3  # 3 frames


def test_playback_advances_and_wraps(window, reef3d_run):
    window.open_case(reef3d_run)
    window.timeline._advance()
    assert window.session.scene.time == 1
    window.timeline.go_to_frame(2)
    window.timeline._advance()
    assert window.session.scene.time == 0


def test_properties_list_data_variables(window, reef3d_run):
    window.open_case(reef3d_run)
    window.scene_tree.select("free_surface-1")
    combo = window.properties.editors["variable"]
    choices = [combo.itemText(i) for i in range(combo.count())]
    assert choices == ["none", "velocity", "velocity.x", "velocity.y", "velocity.z", "eta"]
    combo.setCurrentText("velocity.x")
    props = window.session.scene.get("free_surface-1").properties
    assert props["variable"] == "velocity.x"
    window.properties.fit_button.click()
    assert props["auto_range"] is False and props["vmin"] < props["vmax"]
    assert "frames" in window.properties.info.text()


def test_case_exaggeration_slider_and_sync(window, reef3d_run):
    case_id = window.open_case(reef3d_run)
    session = window.session
    window.scene_tree.select(case_id)
    assert "sync_exaggeration" not in window.properties.editors  # the Sync button instead
    both = window.properties.editors["exaggeration"]
    both.slider.setValue(both.slider.maximum() // 2)  # log scale: the middle is 3.16
    assert session.scene.get(case_id).properties["exaggeration"] == pytest.approx(3.16)
    assert both.spin.value() == pytest.approx(3.16)
    both.spin.setValue(10.0)  # typing moves the slider
    assert both.slider.value() == both.to_position(10.0)

    bed = next(i.id for i in session.scene if i.kind == "bathymetry")
    window.scene_tree.select(bed)
    panel = window.properties
    editor = panel.editors["exaggeration"]
    assert editor.isEnabled() and editor.value() == pytest.approx(10.0)  # the case's
    assert panel.sync_button.isChecked()
    editor.spin.setValue(5.0)  # synced: moves the whole case
    assert session.scene.get(case_id).properties["exaggeration"] == pytest.approx(5.0)
    assert session.data.exaggeration("free_surface-1") == pytest.approx(5.0)

    panel.sync_button.setChecked(False)  # each item its own, starting from 5
    assert session.scene.get(case_id).properties["sync_exaggeration"] is False
    editor.spin.setValue(2.0)
    assert session.data.exaggeration(bed) == pytest.approx(2.0)
    assert session.data.exaggeration("free_surface-1") == pytest.approx(5.0)
    session.undo()
    session.undo()  # sync back on
    assert panel.sync_button.isChecked() and editor.value() == pytest.approx(5.0)


def test_opening_a_case_shows_its_free_surface(window, reef3d_run):
    window.open_case(reef3d_run)
    assert window.scene_tree.selected_id() == "free_surface-1"
    assert window.properties.item_id == "free_surface-1"


def test_bed_colour_map_and_colour_by(window, reef3d_run):
    window.open_case(reef3d_run)
    session = window.session
    bed = next(i.id for i in session.scene if i.kind == "bathymetry")
    window.scene_tree.select(bed)
    editors = window.properties.editors
    # plain sand: the colour map still works, it picks the bed's one colour
    assert editors["colormap"].isEnabled() and not editors["vmin"].isEnabled()
    editors["colormap"].setCurrentText("white")
    assert session.data.display(bed, session.data.load(bed)).color == "#ffffff"
    assert session.scene.get(bed).properties["variable"] == "none"
    editors["variable"].setCurrentText("depth")
    editors["colormap"].setCurrentText("rainbow")
    props = window.session.scene.get(bed).properties
    assert (props["variable"], props["colormap"]) == ("depth", "rainbow")
    editors["hide_wet"].setChecked(True)
    assert props["hide_wet"] is True


def test_open_case_error_is_reported(window, tmp_path, monkeypatch):
    from PySide6.QtWidgets import QMessageBox

    shown = []
    monkeypatch.setattr(QMessageBox, "warning", lambda *args: shown.append(args[2]))
    assert window.open_case(tmp_path / "missing") is None
    assert "does not exist" in shown[0]
    assert len(window.session.scene) == 0


def test_save_screenshot(window, tmp_path):
    path = tmp_path / "shot.png"
    assert window.save_screenshot(path)
    assert path.read_bytes()[:4] == b"\x89PNG"
    # in the recorded script: the 3D view rendered off screen
    script = window.session.to_script()
    assert "File > Save screenshot" in script and str(path) in script
    compile(script, "<script>", "exec")


def test_dev_remote_runs_actions(window, reef3d_run, tmp_path):
    import json

    from lagoonviz.gui.devremote import DevRemote

    folder = tmp_path / "remote"
    remote = DevRemote(window, folder)
    request = {
        "actions": [
            {"open_case": str(reef3d_run)},
            {"frame": 2},
            {"set": ["free_surface-1", "exaggeration", 3.0]},
            {"screenshot": "a.png"},
            {"exec": "print('no code runs')"},
        ]
    }
    (folder / "request.json").write_text(json.dumps(request))
    remote._tick()
    result = json.loads((folder / "result.json").read_text())
    assert result["timeline"] == "frame 2 / 2"
    assert (folder / "a.png").is_file()
    assert result["results"][-1].startswith("ERROR unknown action 'exec'")
    assert not (folder / "request.json").exists()


# ----------------------------------------------------------------- phase 2: time series
@pytest.fixture
def series_run(tmp_path):
    from reef3d_fixtures import make_run

    return make_run(tmp_path / "body", frames=5, volume=False, print_interval=0.5, series=True)


def test_plots_show_series_and_follow_time(window, series_run, qtbot):
    from PySide6.QtCore import QPoint

    window.open_case(series_run)
    plots = window.plots
    assert plots.row_ids == ["series-1", "series-2", "series-3"]
    assert plots.linked
    plots.resize(800, 300)
    image = plots.grab().toImage()  # paints without errors
    assert image.width() == 800
    # clicking in a plot moves the scene time (and the timeline)
    x = int(plots.x_of(1.0))
    qtbot.mouseClick(plots, Qt.LeftButton, pos=QPoint(x, 100))
    assert window.session.scene.time == pytest.approx(1.0, abs=0.05)
    assert window.timeline.frame_of(window.session.scene.time) == 2
    # zoom and reset
    low, high = plots.time_window()
    from PySide6.QtCore import QPointF
    from PySide6.QtGui import QWheelEvent

    wheel = QWheelEvent(
        QPointF(x, 100),
        QPointF(x, 100),
        QPoint(0, 0),
        QPoint(0, 240),
        Qt.NoButton,
        Qt.NoModifier,
        Qt.NoScrollPhase,
        False,
    )
    plots.wheelEvent(wheel)
    zoomed = plots.time_window()
    assert zoomed[1] - zoomed[0] < high - low
    qtbot.mouseDClick(plots, Qt.LeftButton, pos=QPoint(x, 100))
    assert plots.view is None


def test_plots_draw_their_lines_once_while_playing(window, series_run):
    """Playing redrew every line of every plot at each frame (330 ms for a gauge file
    of a long run): now only the cursor and the legend values are drawn again."""
    window.open_case(series_run)
    plots = window.plots
    plots.resize(600, 300)
    plots.grab()  # paints: lines into the cached picture
    picture = plots._static
    assert picture is not None
    for time in (0.5, 1.0, 1.5):
        window.session.set_time(time)
        plots.grab()
    assert plots._static is picture  # not drawn again
    plots.resize(700, 300)
    plots.grab()
    assert plots._static is not picture  # a new size draws the lines again


def test_probe_time_series_in_the_plots(window, series_run, qtbot):
    """The orange dot on the free surface plots eta there over the run, and a box
    chooses which plots are shown."""
    window.open_case(series_run)
    session, plots = window.session, window.plots
    session.data.load("free_surface-1")
    probe = session.data.probe("free_surface-1", 5, (1.0, 0.0, 0.0))
    plots.set_probe(probe)
    qtbot.waitUntil(lambda: "probe" in plots.row_ids, timeout=5000)
    assert plots.row_ids[0] == "probe" and len(plots.row_ids) > 1  # with the gauges
    labels = [plots.show_box.itemText(i) for i in range(plots.show_box.count())]
    assert labels[:2] == ["All plots", "Probe (orange dot)"] and not plots.show_box.isHidden()
    plots.show_box.setCurrentIndex(1)
    plots.refresh()
    assert plots.row_ids == ["probe"]
    plots.show_box.setCurrentIndex(2)  # the gauges only
    plots.refresh()
    assert "probe" not in plots.row_ids
    plots.set_probe(None)
    assert "Probe (orange dot)" not in [
        plots.show_box.itemText(i) for i in range(plots.show_box.count())
    ]


def test_hidden_series_leave_the_plot(window, series_run):
    window.open_case(series_run)
    window.session.set_visible("series-2", False)
    assert window.plots.row_ids == ["series-1", "series-3"]
    window.session.set_property("series-1", "columns", "P2, P3")
    assert [t.name for t in window.plots._rows[0][1]] == ["P2", "P3"]


def test_series_properties_list_files(window, series_run):
    window.open_case(series_run)
    window.scene_tree.select("series-2")
    combo = window.properties.editors["source"]
    labels = [combo.itemText(i) for i in range(combo.count())]
    assert "NHFLOW Body 0 forces (REEF3D_6DOF_forces_0.dat)" in labels
    assert "Columns: Fx, Fy, Fz" in window.properties.info.text()


def test_empty_plot_paints_hint(window):
    window.plots.resize(400, 200)
    assert window.plots.row_ids == []
    window.plots.grab()


def test_theme_switches_now_and_is_remembered(window):
    from lagoonviz.gui.theme import COLORS, THEMES

    window.theme_actions["dark"].trigger()
    assert window.settings.value("view/theme") == "dark"
    assert (
        window.theme_actions["dark"].isChecked() and not window.theme_actions["light"].isChecked()
    )
    assert COLORS["base"] == THEMES["dark"]["base"]  # applied at once, no restart
    assert window.settings.value("view/look") == "ocean"  # the 3D view follows
    window.theme_actions["light"].trigger()
    assert COLORS == THEMES["light"] and window.settings.value("view/look") == "studio"


def test_themes_apply(qapp):
    from lagoonviz.gui.theme import COLORS, SERIES_COLORS, THEMES, apply_theme

    assert apply_theme(qapp, "dark") == "dark" and COLORS["base"] == THEMES["dark"]["base"]
    assert SERIES_COLORS[0] == "#5aa2e6"
    assert apply_theme(qapp, "nonsense") == "light" and COLORS == THEMES["light"]


def test_add_slice_from_toolbar(window, reef3d_run):
    window.open_case(reef3d_run)
    window.add_actions["slice"].trigger()
    item = window.session.scene.get("slice-1")
    assert item.properties["normal"] == "y" and item.properties["position"] == 2.0
    assert window.scene_tree.selected_id() == "slice-1"
    combo = window.properties.editors["variable"]
    choices = [combo.itemText(i) for i in range(combo.count())]
    assert "pressure" in choices and "velocity.z" in choices


def test_viewport_look_is_remembered(window):
    window.look_actions["ocean"].trigger()
    assert window.settings.value("view/look") == "ocean"
    assert window.look_actions["ocean"].isChecked()


def test_timeline_starts_where_the_shown_run_has_data(window, tmp_path):
    import os
    import time

    from reef3d_fixtures import make_run

    # 28 outputs left from the long run, 2 from a restart (< 10 %): the long run shows
    run = make_run(tmp_path / "restarted", frames=30, volume=False)
    folder = run / "REEF3D_NHFLOW_VTP_FSF"
    now = time.time()
    for path in folder.iterdir():
        step = int(path.name.split("-")[3][:8])
        stamp = now + step if step < 2 else now - 3600 + step
        os.utime(path, (stamp, stamp))
    window.open_case(folder)
    timeline = window.timeline
    assert timeline.first_frame == 2 and timeline.slider.minimum() == 2
    timeline.go_to_frame(0)
    assert window.session.scene.time == 2.0
    timeline.go_to_frame(timeline.n_frames - 1)
    timeline._advance()  # wraps to the first frame with data, not to 0
    assert window.session.scene.time == 2.0


def test_export_dialog_and_overlay(window, reef3d_run):
    import numpy as np

    from lagoonviz.gui.export_movie import ExportAnimationDialog, make_overlay, movie_folder

    window.open_case(reef3d_run)
    assert movie_folder(window.session) == reef3d_run  # next to the case, if writable
    dialog = ExportAnimationDialog(window, window.session, str(reef3d_run), (640, 360))
    settings = dialog.settings()
    assert settings["size"] == (1920, 1080) and (settings["start"], settings["end"]) == (0, 2)
    assert settings["path"].endswith(".mp4")
    # the film has the 3D view's shape by default, so it frames what the view shows
    square = ExportAnimationDialog(window, window.session, str(reef3d_run), (800, 600))
    assert square.settings()["size"] == (1920, 1440)
    image = np.full((180, 320, 4), 255, np.uint8)
    drawn = make_overlay(window.session)(image.copy(), 1.0)
    assert drawn.shape == image.shape and (drawn != 255).any()  # time and colour bar


def test_export_dialog_path_traced(window, reef3d_run, monkeypatch, tmp_path):
    from PySide6.QtWidgets import QDialogButtonBox

    from lagoonviz.gui.export_movie import ExportAnimationDialog
    from lagoonviz.render import pathtrace

    window.open_case(reef3d_run)
    monkeypatch.setattr(pathtrace, "find_blender", lambda: None)
    dialog = ExportAnimationDialog(window, window.session, str(reef3d_run), (640, 360))
    export = dialog.findChild(QDialogButtonBox).button(QDialogButtonBox.Ok)
    assert dialog.settings()["engine"] == "pygfx" and dialog.cycles_row.isHidden()
    dialog.engine.setCurrentIndex(1)
    assert not dialog.cycles_row.isHidden() and not dialog.overlay.isEnabled()
    assert not export.isEnabled() and "not found" in dialog.blender_label.text()
    settings = dialog.settings()
    assert settings["engine"] == "cycles" and settings["samples"] == 128
    assert settings["overlay"] is False  # Cycles films show the scene alone
    blender = tmp_path / "blender"
    blender.write_text("")
    monkeypatch.setattr(pathtrace, "find_blender", lambda: str(blender))
    found = ExportAnimationDialog(window, window.session, str(reef3d_run), (640, 360))
    found.engine.setCurrentIndex(1)
    assert found.findChild(QDialogButtonBox).button(QDialogButtonBox.Ok).isEnabled()
    found.engine.setCurrentIndex(0)
    assert found.overlay.isEnabled()


def test_profiles_panel_shows_wave_profiles(window, tmp_path):
    from reef3d_fixtures import make_run

    run = make_run(
        tmp_path / "fnpf", solver="FNPF", frames=6, volume=False, print_interval=0.5, profiles=True
    )
    window.open_case(run)
    item = next(i for i in window.session.scene if i.kind == "profile")
    assert window.profiles.row_ids == [item.id]
    window.session.set_time(1.0)
    window.profiles.resize(500, 200)
    window.profiles.grab()  # paints lines, dashed theory and legend
    window.session.set_visible(item.id, False)
    assert window.profiles.row_ids == []
    window.profiles.grab()  # the hint


def test_compare_two_cases_side_by_side(window, tmp_path):
    from reef3d_fixtures import make_run

    first = window.open_case(make_run(tmp_path / "a", frames=3, volume=False))
    second = window.open_case(make_run(tmp_path / "b", solver="FNPF", frames=3, volume=False))
    window.toggle_compare(True)
    assert window.act_compare.isChecked() and not window.viewport2.isHidden()
    assert window.viewport.case_filter == first and window.viewport2.case_filter == second
    window.viewport2.set_case_filter("")
    assert window.viewport2.case_filter == ""
    window.screenshot()  # both views pasted in
    window.toggle_compare(False)
    assert window.viewport2.isHidden() and window.viewport.case_filter == ""


def test_reload_follows_a_running_simulation(window, tmp_path):
    import os
    import shutil

    from reef3d_fixtures import make_run

    full = make_run(tmp_path / "full", frames=5, volume=False)
    running = tmp_path / "running"
    shutil.copytree(full, running)
    folders = [running / "REEF3D_NHFLOW_VTP_FSF", running / "REEF3D_NHFLOW_VTP_BED"]
    for folder in folders:
        for path in folder.glob("*0000000[34]*"):
            path.unlink()
    window.open_case(running)
    timeline = window.timeline
    assert timeline.n_frames == 3
    timeline.go_to_frame(2)  # at the end
    window.set_follow(True)
    for folder in folders:
        for path in (full / folder.name).glob("*0000000[34]*"):
            shutil.copy(path, folder)
        os.utime(folder, (os.path.getmtime(folder) + 5,) * 2)
    window._check_for_new_output()  # what the follow timer does
    assert timeline.n_frames == 5
    assert timeline.frame_of(window.session.scene.time) == 4  # followed to the new end
    window.set_follow(False)
    window.act_reload.trigger()  # nothing new: stays
    assert timeline.n_frames == 5


def test_full_screen_viewer_shows_only_the_view_and_the_player(window):
    window.show()
    window.act_full_screen.trigger()
    assert window.full_screen and window.act_full_screen.isChecked()
    shown = [name for name, dock in window.docks.items() if not dock.isHidden()]
    assert shown == ["timeline"] and window.toolbar.isHidden()
    assert not window.timeline.exit_full_screen.isHidden()
    player = window.docks["timeline"]
    assert player.maximumHeight() <= window.timeline.sizeHint().height()  # no more room
    window.act_leave_full_screen.trigger()  # Esc
    assert player.maximumHeight() > 10000
    assert window.timeline.exit_full_screen.isHidden()
    window.act_full_screen.trigger()
    window.timeline.exit_full_screen.click()  # the button does the same
    assert not window.full_screen
    assert not window.full_screen and not window.toolbar.isHidden()
    for name in ("scene", "properties", "plots", "timeline", "console"):
        assert not window.docks[name].isHidden(), name


def test_properties_show_the_automatic_range(window, reef3d_run):
    window.open_case(reef3d_run)
    window.scene_tree.select("free_surface-1")
    low, high = window.properties.editors["vmin"], window.properties.editors["vmax"]
    assert low.isEnabled() and high.isEnabled()  # editable: typing fixes the range
    shown = window.session.data.automatic_range("free_surface-1")
    assert (low.value(), high.value()) == pytest.approx(shown, abs=1e-3)
    window.properties.editors["auto_range"].setChecked(False)
    assert low.isEnabled() and window.session.scene.get("free_surface-1").properties[
        "vmax"
    ] == pytest.approx(shown[1], rel=1e-5)


def test_typing_a_range_bound_fixes_the_range(window, reef3d_run):
    """Bug: range min and max could not be changed while the range was automatic."""
    window.open_case(reef3d_run)
    window.scene_tree.select("free_surface-1")
    shown = window.session.data.automatic_range("free_surface-1")
    window.properties.editors["vmin"].setValue(-0.5)  # as typed and Return
    props = window.session.scene.get("free_surface-1").properties
    assert props["auto_range"] is False and props["vmin"] == -0.5
    assert props["vmax"] == pytest.approx(shown[1], abs=1e-3)  # the other bound stays
    assert not window.properties.editors["auto_range"].isChecked()
    window.properties.editors["vmax"].setValue(0.75)
    props = window.session.scene.get("free_surface-1").properties
    assert (props["vmin"], props["vmax"]) == (-0.5, 0.75)
    shown_now = window.session.data.display(
        "free_surface-1", window.session.data.load("free_surface-1")
    )
    assert (shown_now.vmin, shown_now.vmax) == (-0.5, 0.75)


def test_convert_to_lagoon_format(window, reef3d_run, qtbot):
    """File > Convert to LAGOON format writes the store next to the output and the
    case then reads it."""
    from lagoonviz.core.data.lagoon import StoreDataset

    window.open_case(reef3d_run)
    job = window.convert_case(ask=False)
    qtbot.waitUntil(
        lambda: job.finished and not window._convert_job.thread.is_alive(), timeout=10000
    )
    qtbot.wait(300)  # the poll timer reloads the case
    assert job.error is None, job.error
    assert (reef3d_run / "REEF3D_NHFLOW.lagoon" / "zarr.json").is_file()
    case = window.session.data.case("case-1")
    assert isinstance(case.datasets["nhflow_free_surface"], StoreDataset)
    assert "smaller" in window.statusBar().currentMessage()


def test_export_figure(window, tmp_path, qtbot):
    """File > Export figure: the plots and the wave profiles as paper figures."""
    pytest.importorskip("matplotlib")
    from reef3d_fixtures import make_run

    from lagoonviz.gui.export_figure import available_contents, make_figure
    from lagoonviz.render.figure import save_figure

    run = make_run(tmp_path / "run", frames=4, print_interval=0.5, series=True, profiles=True)
    window.open_case(run)
    window.session.set_time(1.0)
    window.plots.refresh()
    keys = [key for key, _ in available_contents(window.session, bool(window.plots.shown_rows()))]
    assert keys[:4] == ["series", "spectra", "validation", "profiles"]
    settings = {
        "width_mm": 90.0,
        "height_mm": None,
        "font_size": 8.0,
        "font_family": "sans-serif",
        "dpi": 150,
        "usetex": False,
        "title": "",
    }
    for content in ("series", "spectra", "profiles"):
        figure, spec = make_figure(window, {**settings, "content": content})
        path = save_figure(figure, tmp_path / f"{content}.pdf", spec)
        assert path.stat().st_size > 1000
    # a plotted gauge against a "measured" file (here: the same gauge, written out)
    trace = window.plots.shown_rows()[0][1][0]
    lab = tmp_path / "lab.txt"
    lab.write_text(
        "time WG1\n" + "".join(f"{t} {v}\n" for t, v in zip(trace.time, trace.values, strict=True))
    )
    figure, spec = make_figure(
        window,
        {
            **settings,
            "content": "validation",
            "style": "Journal (serif)",
            "font_family": None,  # the style's
            "line": (0, 0),
            "measured": str(lab),
            "measured_column": "WG1",
        },
    )
    assert figure.comparison.rmse < 1e-6 and spec.font_family == "serif"
    # the dialog: choosing a style sets its font (which can then still be changed)
    from lagoonviz.gui.export_figure import ExportFigureDialog

    dialog = ExportFigureDialog(window, available_contents(window.session, True), str(tmp_path))
    qtbot.addWidget(dialog)
    dialog.style.setCurrentText("Journal (serif)")
    assert dialog.settings()["font_family"] == "serif"
    dialog.style.setCurrentText("REEF3D")
    assert dialog.settings()["font_family"] == "sans-serif"
    # File > Export script makes the figure again
    from lagoonviz.gui.export_figure import record_figure

    validation = {
        **settings,
        "content": "validation",
        "style": "Journal (serif)",
        "line": (0, 0),
        "measured": str(lab),
        "measured_column": "WG1",
    }
    record_figure(window, validation, tmp_path / "validation.pdf")
    script = window.session.to_script()
    assert "File > Export figure: validation" in script and f"t.name == {trace.name!r}" in script
    namespace = {"__name__": "__main__"}
    exec(compile(script, "<script>", "exec"), namespace)
    assert (tmp_path / "validation.pdf").stat().st_size > 1000
    assert window.act_export_figure.shortcut().toString() == "Ctrl+Shift+P"


def test_figure_editor(window, tmp_path, qtbot):
    """File > Figure editor: lines dropped onto panels, a live preview, templates."""
    pytest.importorskip("matplotlib")
    from reef3d_fixtures import make_run

    from lagoonviz.core.figure_template import FigureTemplate, Line
    from lagoonviz.gui.figure_editor import MIME, _decode, _encode

    run = make_run(tmp_path / "run", frames=4, print_interval=0.5, series=True)
    window.open_case(run)
    editor = window.open_figure_editor()
    qtbot.addWidget(editor)
    assert window.open_figure_editor() is editor  # one per window
    # what can be drawn: the case's time-series files and their columns
    top = editor.sources.topLevelItem(0)
    assert top is not None and top.childCount() > 0
    line = top.child(0).data(0, Qt.UserRole)
    assert isinstance(line, Line)
    assert _decode(_encode(line)) == line
    assert editor.sources.mimeData([top.child(0)]).hasFormat(MIME)
    # a line onto empty space: a new time panel; another onto that panel
    editor.add_line(Line(line.source, line.column))
    editor.add_line(Line(line.source, line.column, label="again", linestyle="--"), 0)
    assert len(editor.template.panels) == 1 and len(editor.template.panels[0].lines) == 2
    assert editor.settings_stack.currentWidget() is editor.line_form
    assert editor.label_box.text() == "again"
    # a spectrum panel, moved up, then removed
    editor.add_panel("spectrum")
    editor.add_line(Line(line.source, line.column), 1)
    editor.tree.setCurrentItem(editor.tree.topLevelItem(1))
    editor.move_selected(-1)
    assert editor.template.panels[0].kind == "spectrum"
    editor.remove_selected()
    assert [p.kind for p in editor.template.panels] == ["time"]
    # the panel's settings change the template
    editor.tree.setCurrentItem(editor.tree.topLevelItem(0))
    editor.ymax_box.setText("0.5")
    editor.ymax_box.editingFinished.emit()
    assert editor.template.panels[0].ymax == 0.5
    # the preview, drawn after the edits settle
    editor.draw_preview()
    assert not editor.preview.pixmap().isNull()
    assert editor.problems.text() == ""
    # a template round trip and the figure written
    saved = editor.save_template(str(tmp_path / "gauge"))
    assert saved.endswith(".lagoonfig")
    editor.template = FigureTemplate()
    editor.open_template(saved)
    assert len(editor.template.panels[0].lines) == 2
    assert editor.template.panels[0].lines[1].label == "again"
    written = editor.export(str(tmp_path / "gauge.pdf"))
    assert Path(written).stat().st_size > 1000
    assert "File > Figure editor: export" in window.session.to_script()


def test_figure_editor_paper_figure(window, tmp_path, qtbot):
    """The figure editor: wave profiles and a 3D view next to a gauge, in two columns,
    lettered, with a note."""
    pytest.importorskip("matplotlib")
    from reef3d_fixtures import make_run

    from lagoonviz.core.figure_template import FigureTemplate, Line

    run = make_run(
        tmp_path / "fnpf", solver="FNPF", frames=4, print_interval=0.5, series=True, profiles=True
    )
    window.open_case(run)
    window.session.set_time(1.0)
    editor = window.open_figure_editor()
    qtbot.addWidget(editor)
    # the wave profiles are in the list, with FNPF's wave theory
    names = [
        editor.sources.topLevelItem(i).text(0) for i in range(editor.sources.topLevelItemCount())
    ]
    assert "Wave profiles" in names
    profiles = editor.sources.topLevelItem(names.index("Wave profiles"))
    first = profiles.child(0).data(0, Qt.UserRole)
    theory = profiles.child(1).data(0, Qt.UserRole)
    assert first.data == "profile" and theory.theory and theory.linestyle == "--"
    # a gauge, then a profile line onto it: the profile gets a panel of its own
    gauges = editor.sources.topLevelItem(0).child(0).data(0, Qt.UserRole)
    editor.add_line(Line(gauges.source, gauges.column))
    editor.add_line(Line(first.source, first.column, data="profile"), 0)
    editor.add_line(Line(theory.source, theory.column, data="profile", linestyle="--"), 1)
    assert [p.kind for p in editor.template.panels] == ["time", "profile"]
    assert len(editor.template.panels[1].lines) == 2
    # a 3D view panel, taking the view as it is now; only its settings show
    index = editor.add_panel("view")
    editor.tree.setCurrentItem(editor.tree.topLevelItem(index))
    assert editor.panel_layout.isRowVisible(editor.take_view_button)
    assert not editor.panel_layout.isRowVisible(editor.ymin_box)
    window.session.set_time(1.5)
    editor.take_view()
    view = editor.template.panels[index]
    assert view.time == 1.5 and view.view == "camera"
    window.session.set_time(1.0)
    # the view over both columns, a note on the profile, letters
    editor.span_box.setValue(2)
    assert view.span == 2
    editor.tree.setCurrentItem(editor.tree.topLevelItem(1))
    assert editor.panel_layout.isRowVisible(editor.moment_box)
    assert editor.start_box.suffix() == " m"
    editor.notes_box.setPlainText("0.5, 0.9: $\\lambda$ = 4 m")
    assert editor.template.panels[1].notes[0].x == 0.5
    editor.tree.setCurrentItem(None)
    editor.columns_box.setValue(2)
    editor.letters_box.setChecked(True)
    assert editor.template.columns == 2 and editor.template.letters
    editor.draw_preview()
    assert not editor.preview.pixmap().isNull()
    written = editor.export(str(tmp_path / "paper.pdf"))
    assert Path(written).stat().st_size > 2000
    saved = editor.save_template(str(tmp_path / "paper"))
    assert FigureTemplate.load(saved).panels[2].kind == "view"
