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


def test_colour_bands_and_the_contour_legend(session):
    """Colour bands: the colours in steps between round levels of the variable an
    item is coloured by (filled contours), the bar in steps too, numbered at round
    values, ending in a point where the frame goes beyond the range; contour lines of
    that variable mark their levels on a smooth bar."""
    from lagoonviz.core.colormaps import band_edges, banded_lut, colors, lut

    item = _coloured(session)
    variable = item.properties["variable"]
    mesh = session.data.load(item.id)
    assert session.data.colour_levels(item.id, mesh) == []  # neither bands nor lines
    plain = colour_bars(session)[0]
    assert plain.levels == [] and not plain.bands
    np.testing.assert_allclose(plain.table(64), lut(plain.colormap, 64))
    # contour lines of the variable it is coloured by: their levels on the bar
    session.set_property(item.id, "contour_by", variable)
    bar = colour_bars(session)[0]
    lines = session.data.contour_values(item.id, variable, mesh)
    assert bar.levels == [v for v in lines if bar.vmin < v < bar.vmax] and bar.levels
    assert not bar.bands
    assert session.data.display(item.id, mesh).bands == ()  # smooth colours
    # colour bands: the same levels, the surface's colours in steps
    session.set_property(item.id, "colour_bands", True)
    bar = colour_bars(session)[0]
    assert bar.bands and bar.levels
    shown = session.data.display(item.id, mesh)
    assert shown.bands == band_edges(bar.levels, bar.vmin, bar.vmax)
    assert all(0 < u < 1 for u in shown.bands) and list(shown.bands) == sorted(shown.bands)
    table = bar.table(1024)
    steps = len(np.unique(table[:, :3].round(6), axis=0))
    assert steps == len(bar.levels) + 1
    # each point's colour is its band's: equal within a band
    rgba = colors(shown.values, shown.colormap, shown.vmin, shown.vmax, shown.bands)
    band = np.searchsorted(
        [bar.vmin + u * (bar.vmax - bar.vmin) for u in shown.bands], shown.values
    )
    for b in np.unique(band):
        assert len(np.unique(rgba[band == b].round(6), axis=0)) == 1
    # the steps' colours: the colour map at each band's middle
    edges = (0.0, 0.25, 0.75, 1.0)
    stepped = banded_lut("viridis", (0.25, 0.75), 401)
    for k, (a, b) in enumerate(zip(edges, edges[1:], strict=False)):
        middle = lut("viridis", 4096)[round(0.5 * (a + b) * 4095)]
        inside = (np.arange(401) / 400 > a + 1e-9) & (np.arange(401) / 400 < b - 1e-9)
        np.testing.assert_allclose(stepped[inside], np.repeat(middle[None], inside.sum(), 0)), k
    # beyond the range: the bar's ends
    low, high = np.percentile(shown.values, [30, 70])
    session.set_properties(item.id, auto_range=False, vmin=float(low), vmax=float(high))
    assert colour_bars(session)[0].extend == "both"
    session.set_properties(item.id, vmin=-1e9, vmax=1e9)
    assert colour_bars(session)[0].extend == "neither"


def test_the_printed_contour_legend(session):
    """A figure's colour bar as a printed map has it (Fig. 2 of an abstract): in
    steps with colour bands, ticks at round values, points at the ends beyond the
    range; with contour lines a line at each level on a smooth bar."""
    pytest.importorskip("matplotlib")
    from matplotlib.colors import BoundaryNorm, ListedColormap

    from lagoonviz.render.figure import template_figure

    item = _coloured(session)
    variable = item.properties["variable"]
    image = np.full((90, 160, 4), 200, np.uint8)

    def printed():
        panel = Panel(kind="view", bar_place="right", bar_length=1.0)
        figure = template_figure(session, FigureTemplate(panels=[panel]), images={0: image})
        every = [*figure.get_axes(), *(c for ax in figure.get_axes() for c in ax.child_axes)]
        return next(ax._colorbar for ax in every if getattr(ax, "_colorbar", None) is not None)

    session.set_properties(item.id, contour_by=variable, colour_bands=True)
    bar = colour_bars(session)[0]
    top = bar.vmax - 0.3 * (bar.vmax - bar.vmin)
    session.set_properties(item.id, auto_range=False, vmin=bar.vmin, vmax=top)
    bar = colour_bars(session)[0]
    assert bar.extend in ("max", "both")
    colourbar = printed()
    assert isinstance(colourbar.norm, BoundaryNorm) and isinstance(colourbar.cmap, ListedColormap)
    assert colourbar.cmap.N == len(bar.levels) + 1
    assert colourbar.extend == bar.extend
    ticks = colourbar.get_ticks()
    from lagoonviz.core.plotting import nice_ticks

    assert 2 <= len(ticks) <= 7  # round values, not every band's edge
    np.testing.assert_allclose(ticks, nice_ticks(bar.vmin, bar.vmax, 5))
    # no bands: smooth, the contour lines marked
    session.set_property(item.id, "colour_bands", False)
    colourbar = printed()
    assert not isinstance(colourbar.norm, BoundaryNorm)
    assert colourbar.lines and len(colourbar.lines[0].get_segments()) == len(bar.levels)


def test_the_contour_legend_on_films_and_under_the_view(session):
    """The 3D view's colour bars (under the view, on films and saved images) as the
    printed one: round numbers between the ends, a line at each contour level on a
    smooth bar, steps with colour bands, a point at an end beyond the range."""
    from lagoonviz.core.legend import place, round_ticks
    from lagoonviz.render.legend_overlay import MARK, LegendOverlay

    # Fig. 2's depth: 0 20 40 60 numbered, the end at 65 not (60 is close)
    assert round_ticks(0.0, 65.0) == [
        (0.0, "0"),
        (20.0, "20"),
        (40.0, "40"),
        (60.0, "60"),
        (65.0, ""),
    ]
    ticks = round_ticks(-0.06, 0.14)
    assert [t for _, t in ticks] == ["", "-0.05", "0", "0.05", "0.1", "0.14"] or ticks[-1][
        1
    ] == "0.14"
    assert ticks[0] == (-0.06, "")  # -0.05 is close to the end
    item = _coloured(session)
    variable = item.properties["variable"]
    session.set_property(item.id, "contour_by", variable)
    bar = colour_bars(session)[0]
    assert bar.marks() and all(0 < m < 1 for m in bar.marks())
    assert len(bar.ticks) > 2  # round numbers between the ends
    np.testing.assert_allclose(
        [v for v, _ in bar.ticks], [v for v, _ in round_ticks(bar.vmin, bar.vmax)]
    )
    image = np.full((360, 640, 4), 200, np.uint8)
    spots = place(640, 360, 1, session.legend_layout())
    strip = spots.bars[0]
    row = int(strip.y + strip.h / 2)
    drawn = LegendOverlay(session)(image, 1.0)
    # the contour lines' levels: dark lines across the bar
    for where in bar.marks():
        x = int(round(strip.x + where * strip.w))
        near = drawn[row, max(x - 1, 0) : x + 2, :3].astype(int)
        assert (np.abs(near - np.array(MARK[:3])).sum(axis=1) < 60).any()
    # beyond the range at both ends: the bar ends in points of its end colours
    values = session.data.load(item.id).scalar(variable)
    low, high = np.percentile(values, [30, 70])
    session.set_properties(item.id, auto_range=False, vmin=float(low), vmax=float(high))
    bar = colour_bars(session)[0]
    assert bar.extend == "both"
    drawn = LegendOverlay(session)(image, 1.0)
    table = (np.asarray(bar.table(1024))[:, :3] * 255).round().astype(int)
    left = drawn[row, int(strip.x - strip.h / 3), :3].astype(int)
    right = drawn[row, int(strip.right + strip.h / 3), :3].astype(int)
    assert np.abs(left - table[0]).sum() < 12 and np.abs(right - table[-1]).sum() < 12
    # colour bands: no lines, the bar in steps
    session.set_property(item.id, "colour_bands", True)
    bar = colour_bars(session)[0]
    assert bar.marks() == [] and bar.bands
