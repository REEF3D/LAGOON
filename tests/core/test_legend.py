"""Legends: the colour bars (each item's title, numbers, ticks, on or off), the
films' layout (corners, length, text, box), line legends of figures and plots."""

import numpy as np
import pytest
from reef3d_fixtures import make_run, show_plots

from lagoonviz.core import Session
from lagoonviz.core.figure_template import FigureTemplate, Line, Panel
from lagoonviz.core.legend import (
    LAYOUT,
    bar_ticks,
    colour_bars,
    format_number,
    layout,
    line_name,
    line_names,
    place,
)


@pytest.fixture
def session(tmp_path):
    run = make_run(tmp_path / "run", frames=4, print_interval=0.5, series=True, profiles=True)
    s = Session()
    s.open_case(run)
    show_plots(s)
    s.set_time(1.0)
    return s


def _coloured(session):
    return next(i for i in session.scene if i.properties.get("variable", "none") != "none")


def test_numbers_and_ticks():
    assert format_number(0.17934) == "0.1793"
    assert format_number(2.5, "integer") == "2" or format_number(2.5, "integer") == "3"
    assert format_number(-0.004, "2 decimals") == "0.00"  # no "-0.00"
    assert format_number(1234.5, "1 decimal") == "1234.5"
    assert format_number(0.00012, "scientific") == "1.20e-04"
    assert [t for _, t in bar_ticks(-0.18, 0.18)] == ["-0.18", "0.18"]
    assert [v for v, _ in bar_ticks(0.0, 2.0, 1)] == [0.0, 1.0, 2.0]
    ticks = bar_ticks(0.0, 1.0, 5, "2 decimals")
    assert [t for _, t in ticks] == ["0.00", "0.25", "0.50", "0.75", "1.00"]


def test_an_items_colour_bar(session):
    item = _coloured(session)
    bars = colour_bars(session)
    assert bars[0].label == f"{item.name} · {item.properties['variable']}"
    assert [t for _, t in bars[0].ticks] and len(bars[0].ticks) == 2
    session.set_properties(
        item.id, legend_title=r"$\eta$ [m]", legend_ticks=3, legend_format="2 decimals"
    )
    bar = colour_bars(session)[0]
    assert bar.label == r"$\eta$ [m]" and len(bar.ticks) == 3
    assert all(len(text.split(".")[-1]) == 2 for _, text in bar.ticks)
    session.set_properties(item.id, legend=False)
    assert all(b.item_id != item.id for b in colour_bars(session))
    session.undo()
    assert any(b.item_id == item.id for b in colour_bars(session))
    with pytest.raises(ValueError):
        session.set_properties(item.id, legend_format="roman")


def test_the_layout_of_films(session, tmp_path):
    assert session.legend_layout() == LAYOUT
    whole = session.set_legend(bars="lower right", time="off", length=0.4, box=True)
    assert whole["bars"] == "lower right" and whole["time"] == "off"
    assert session.scene.legend == {
        "bars": "lower right",
        "time": "off",
        "length": 0.4,
        "box": True,
    }
    with pytest.raises(ValueError):
        session.set_legend(bars="middle")
    with pytest.raises(ValueError):
        session.set_legend(colour="red")
    assert layout({"length": 5.0})["length"] == 0.95  # held to its limits
    # kept with the session and in its script
    path = tmp_path / "s.lagoon-session"
    session.save(path)
    again = Session.open(path)
    assert again.scene.legend == session.scene.legend
    assert "session.set_legend(bars='lower right'" in session.to_script()
    assert session.set_legend(reset=True) == LAYOUT and session.scene.legend == {}


def test_placed_on_an_image():
    # the default: bars lower left, the time upper left, in a 1080-line image
    spots = place(1920, 1080, 2)
    assert spots.time.x == pytest.approx(28) and spots.time.y == pytest.approx(28)
    assert spots.bars[1].bottom < 1080 - 28 and spots.bars[0].y < spots.bars[1].y
    assert spots.bars[0].w == pytest.approx(0.28 * 1920) and spots.frame is None
    # the other corner, longer and thicker, larger text and a box; 4K scales up
    spots = place(
        3840,
        2160,
        1,
        {"bars": "upper right", "length": 0.5, "thickness": 2, "text": 1.5, "box": True},
    )
    assert spots.bars[0].right == pytest.approx(3840 - 56 - 16)
    assert spots.bars[0].h == pytest.approx(14 * 2 * 2) and spots.font == pytest.approx(60)
    assert spots.frame is not None and spots.time_frame is not None
    # the time and the bars in one corner: the time first, the bars after it
    spots = place(1920, 1080, 1, {"bars": "upper left"})
    assert spots.titles[0].y > spots.time.bottom
    spots = place(1920, 1080, 1, {"bars": "lower left", "time": "lower left"})
    assert spots.time.bottom < spots.titles[0].y
    assert place(1920, 1080, 1, {"time": "off"}).time is None


def test_film_frames_get_the_legend(session):
    from lagoonviz.render.legend_overlay import LegendOverlay, plain

    assert plain(r"$\eta$ [m]") == "η [m]" and plain(r"$H_{m0}$ [m$^2$]") == "Hm0 [m²]"
    image = np.full((360, 640, 4), 200, np.uint8)
    drawn = LegendOverlay(session)(image, 1.0)
    assert drawn.shape == image.shape and drawn.dtype == np.uint8
    changed = np.any(drawn != image, axis=2)
    rows, cols = np.nonzero(changed)
    assert rows.min() < 60 and rows.max() > 300  # the time on top, a bar at the bottom
    assert cols.max() < 640 / 2  # all on the left
    session.set_legend(bars="lower right", time="off")
    drawn = LegendOverlay(session)(image, 1.0)
    rows, cols = np.nonzero(np.any(drawn != image, axis=2))
    assert rows.min() > 200 and cols.min() > 640 / 2
    session.set_properties(_coloured(session).id, legend=False)
    assert np.array_equal(LegendOverlay(session)(image, 1.0), image)  # nothing to draw


def test_line_names():
    assert line_names("P1 = gauge 1; P4=gauge 4\nFx = force") == {
        "P1": "gauge 1",
        "P4": "gauge 4",
        "Fx": "force",
    }
    assert line_names("nonsense; =x; y=") == {}
    assert line_name(None, "P1") == "P1"


def test_figure_legends(session, tmp_path):
    pytest.importorskip("matplotlib")
    from lagoonviz.core.figure_template import LEGEND_PLACES
    from lagoonviz.render.figure import series_figure, template_figure, template_spec

    series = next(i for i in session.scene if i.kind == "series")
    table = session.data.series(series.id)
    first, second = list(table.columns)[:2]
    session.set_properties(series.id, labels=f"{first} = gauge A", columns=f"{first}, {second}")
    figure = series_figure(session)
    labels = [t.get_text() for t in figure.get_axes()[0].get_legend().get_texts()]
    assert labels[0] == "gauge A"
    session.set_properties(series.id, plot_legend="off")
    assert series_figure(session).get_axes()[0].get_legend() is None
    # template panels: the legend where asked, lines left out of it
    source = session.data.series_file(series.id).key
    lines = [Line(source, first, label="one"), Line(source, second, legend=False)]
    for place_ in ("outside right", "above", "below", "upper left"):
        panel = Panel(
            lines=list(lines), legend=place_, legend_columns=2, legend_frame=True, legend_size=6.0
        )
        template = FigureTemplate(panels=[panel])
        figure = template_figure(session, template, template_spec(template, dpi=100))
        legend = figure.get_axes()[0].get_legend()
        assert [t.get_text() for t in legend.get_texts()] == ["one"]
        assert legend.get_frame_on() and legend.get_texts()[0].get_fontsize() == 6.0
    panel = Panel(lines=list(lines), legend="off")
    figure = template_figure(session, FigureTemplate(panels=[panel]))
    assert figure.get_axes()[0].get_legend() is None
    # one line: "auto" leaves the legend out, a place asked for shows it
    one = Panel(lines=[Line(source, first)])
    assert template_figure(session, FigureTemplate(panels=[one])).get_axes()[0].get_legend() is None
    one.legend = "upper right"
    assert template_figure(session, FigureTemplate(panels=[one])).get_axes()[0].get_legend()
    assert set(LEGEND_PLACES) >= {"auto", "off", "outside right", "above", "below"}
    # a template keeps them
    path = FigureTemplate(panels=[panel]).save(tmp_path / "f.lagoon-figure")
    loaded = FigureTemplate.load(path)
    assert loaded.panels[0].legend == "off" and loaded.panels[0].lines[1].legend is False


@pytest.mark.parametrize("where", ["below", "right", "inside lower left", "inside upper right"])
def test_view_panel_colour_bars(session, where):
    pytest.importorskip("matplotlib")
    from lagoonviz.render.figure import template_figure

    item = _coloured(session)
    session.set_properties(item.id, legend_ticks=3, legend_format="1 decimal")
    panel = Panel(kind="view", bar_place=where, bar_length=0.5)
    image = np.full((90, 160, 4), 200, np.uint8)
    figure = template_figure(session, FigureTemplate(panels=[panel]), images={0: image})
    every = [*figure.get_axes(), *(c for ax in figure.get_axes() for c in ax.child_axes)]
    bars = [ax for ax in every if getattr(ax, "_colorbar", None) is not None]
    assert len(bars) == 1
    axis = bars[0].yaxis if where == "right" else bars[0].xaxis
    texts = [t.get_text() for t in axis.get_ticklabels()]
    assert len(texts) == 3 and all(len(t.split(".")[-1]) == 1 for t in texts)
    assert figure.missing == []
