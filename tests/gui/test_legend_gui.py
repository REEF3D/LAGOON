"""Legends in the window: View → Legend, the 3D view's colour bars, the Plots
panel's legends, and the figure editor's legend and colour bar settings."""

import pytest
from PySide6.QtCore import QSettings
from reef3d_fixtures import show_plots

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


@pytest.fixture
def run(tmp_path):
    from reef3d_fixtures import make_run

    return make_run(tmp_path / "run", frames=4, print_interval=0.5, series=True)


def test_the_legend_dialog(window, qtbot):
    dialog = window.show_legend_dialog()
    qtbot.addWidget(dialog)
    assert window.show_legend_dialog() is dialog  # one per window
    dialog.bars.setCurrentIndex(dialog.bars.findData("upper right"))
    dialog.length.setValue(40)
    dialog.box.setChecked(True)
    assert window.session.scene.legend == {"bars": "upper right", "length": 0.4, "box": True}
    dialog.reset()
    assert window.session.scene.legend == {} and dialog.length.value() == 28
    window.session.set_legend(time="off")  # from a script: the dialog shows it
    dialog.show_layout()
    assert dialog.time.currentData() == "off"


def test_the_view_bars_follow_the_items(window, run, qtbot):
    window.open_case(run)
    show_plots(window.session)  # (they open switched off)
    session = window.session
    item = next(i for i in session.scene if i.properties.get("variable", "none") != "none")
    strip = window.viewport.colorbars
    qtbot.waitUntil(lambda: len(strip.entries) >= 1, timeout=10000)
    session.set_properties(item.id, legend_title=r"$\eta$ [m]", legend_ticks=3)
    qtbot.waitUntil(lambda: strip.entries and strip.entries[0].label == "η [m]", timeout=10000)
    assert len(strip.entries[0].ticks) == 3 and strip.height() > 30
    strip.grab()  # paints without errors
    session.set_legend(view="off")
    qtbot.waitUntil(lambda: not strip.entries, timeout=10000)
    session.set_legend(view="below the view")
    session.set_properties(item.id, legend=False)
    qtbot.waitUntil(lambda: all(e.label != "η [m]" for e in strip.entries), timeout=10000)


def test_the_plots_legends(window, run):
    window.open_case(run)
    show_plots(window.session)  # (they open switched off)
    session = window.session
    series = next(i for i in session.scene if i.kind == "series")
    plots = window.plots
    plots.resize(800, 300)
    for place in ("in the plot", "off", "above the plot"):
        session.set_properties(series.id, plot_legend=place, labels="P1 = gauge 1")
        plots.grab()  # paints without errors
    session.set_properties(series.id, legend_values=False)
    plots.grab()


def test_the_figure_editors_legend(window, run, qtbot):
    pytest.importorskip("matplotlib")
    from lagoonviz.core.figure_template import Line

    window.open_case(run)
    show_plots(window.session)  # (they open switched off)
    editor = window.open_figure_editor()
    qtbot.addWidget(editor)
    line = editor.sources.topLevelItem(0).child(0).data(0, 0x0100)
    editor.add_line(Line(line.source, line.column))
    editor.add_line(Line(line.source, line.column, label="again"), 0)
    editor.tree.setCurrentItem(editor.tree.topLevelItem(0))
    editor.legend_box.setCurrentIndex(editor.legend_box.findData("outside right"))
    editor.legend_columns_box.setValue(2)
    editor.legend_frame_box.setChecked(True)
    panel = editor.template.panels[0]
    assert (panel.legend, panel.legend_columns, panel.legend_frame) == ("outside right", 2, True)
    editor.tree.setCurrentItem(editor.tree.topLevelItem(0).child(1))
    editor.in_legend_box.setChecked(False)
    assert panel.lines[1].legend is False
    editor.add_panel("view")
    editor.tree.setCurrentItem(editor.tree.topLevelItem(1))
    editor.bar_place_box.setCurrentIndex(editor.bar_place_box.findData("inside lower left"))
    editor.bar_length_box.setValue(50)
    assert editor.template.panels[1].bar_place == "inside lower left"
    assert editor.template.panels[1].bar_length == 0.5
