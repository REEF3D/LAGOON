"""Publication figures (matplotlib): time series, wave profiles, the 3D view."""

import numpy as np
import pytest
from reef3d_fixtures import make_run, show_plots

pytest.importorskip("matplotlib")

from lagoonviz.core import Session  # noqa: E402
from lagoonviz.render.figure import (  # noqa: E402
    FigureSpec,
    colormap,
    colour_bars,
    profile_figure,
    save_figure,
    series_figure,
    view_figure,
)


@pytest.fixture
def session(tmp_path):
    run = make_run(tmp_path / "run", frames=4, print_interval=0.5, series=True, profiles=True)
    s = Session()
    s.open_case(run)
    show_plots(s)
    s.set_time(1.0)
    return s


def test_time_series_figure(session, tmp_path):
    spec = FigureSpec(width_mm=90, font_size=8)
    figure = series_figure(session, spec)
    width, height = figure.get_size_inches()
    assert width == pytest.approx(90 / 25.4)
    axes = figure.get_axes()
    assert len(axes) >= 1 and axes[-1].get_xlabel() == "t [s]"
    lines = [line for line in axes[0].get_lines() if line.get_linestyle() == "-"]
    assert lines and lines[0].get_color() == spec.palette[0]
    for suffix in (".pdf", ".svg", ".png"):
        path = save_figure(figure, tmp_path / f"gauges{suffix}", spec)
        assert path.stat().st_size > 1000
    assert save_figure(figure, tmp_path / "noext", spec).suffix == ".pdf"


def test_wave_profile_figure(session, tmp_path):
    figure = profile_figure(session, FigureSpec(width_mm=140))
    ax = figure.get_axes()[0]
    assert ax.get_xlabel() == "x [m]" and "eta" in ax.get_ylabel()
    assert len(ax.get_lines()) >= 2  # two lines
    assert ax.get_title(loc="left").startswith("t = ")
    save_figure(figure, tmp_path / "profiles.pdf")


def test_view_figure_with_colour_bars_and_axes(session, tmp_path):
    bars = colour_bars(session)
    assert bars and bars[0].label.endswith("eta")
    image = np.full((90, 160, 4), 200, np.uint8)
    spec = FigureSpec(width_mm=90, dpi=200)
    figure = view_figure(session, spec, image=image, extent=(0.0, 8.0, 0.0, 4.0))
    ax = figure.get_axes()[0]
    assert ax.get_xlabel() == "x [m]" and ax.get_ylabel() == "y [m]"
    assert len(figure.get_axes()) == 1 + len(bars)  # a colour bar per coloured item
    perspective = view_figure(session, spec, image=image)
    assert not perspective.get_axes()[0].axison
    save_figure(figure, tmp_path / "view.png", spec)


def test_lagoon_colour_maps_in_matplotlib():
    cmap = colormap("reef3d")
    middle = cmap(0.5)
    assert all(abs(c - 0.97) < 0.03 for c in middle[:3])  # still water white


def test_nothing_to_draw(tmp_path):
    s = Session()
    with pytest.raises(ValueError):
        series_figure(s)


def test_house_styles():
    from lagoonviz.render.figure import STYLES

    journal = FigureSpec.from_style("Journal (serif)", width_mm=140)
    assert journal.font_family == "serif" and journal.width_mm == 140
    assert journal.palette[0] == "#000000" and journal.rc()["lines.linewidth"] == 0.8
    slides = FigureSpec.from_style("Presentation", font_size=None)  # None: the style's
    assert slides.font_size == 14 and slides.rc()["axes.grid"] is True
    assert set(STYLES) >= {"REEF3D", "Journal (serif)", "Presentation"}
    assert FigureSpec.from_style("unknown").font_family == "sans-serif"


def test_spectrum_figure(session, tmp_path):
    from lagoonviz.render.figure import spectrum_figure

    figure = spectrum_figure(session, FigureSpec.from_style("REEF3D"))
    axes = figure.get_axes()
    assert axes[-1].get_xlabel() == "f [Hz]"
    assert "H_{m0}" in axes[0].get_legend().get_texts()[0].get_text()
    save_figure(figure, tmp_path / "spectra.pdf")


def test_validation_figure(session, tmp_path):
    from lagoonviz.core.analysis import read_measurements
    from lagoonviz.render.figure import series_rows, validation_figure

    trace = series_rows(session)[0][1][0]  # the first wave gauge
    lab = tmp_path / "lab.dat"
    t = np.linspace(0, 1.5, 31)
    lab.write_text(
        "t WG\n"
        + "".join(
            f"{a} {b}\n"
            for a, b in zip(t, np.interp(t, trace.time, trace.values) + 0.002, strict=True)
        )
    )
    time, columns = read_measurements(lab)
    figure = validation_figure(trace, time, columns["WG"], FigureSpec(), "flume")
    stats = figure.comparison
    assert stats.bias == pytest.approx(-0.002, abs=1e-4) and stats.r == pytest.approx(1, abs=1e-3)
    texts = [t.get_text() for t in figure.get_axes()[0].texts]
    assert any("RMSE" in text and "d =" in text for text in texts)
    save_figure(figure, tmp_path / "validation.pdf")


def test_the_extent_of_the_2d_top_view():
    """LAGOON's orthographic camera is a perspective camera with fov 0: from the top
    the figure gets axes in metres."""
    from types import SimpleNamespace

    from lagoonviz.render.figure import view_extent

    world = SimpleNamespace(forward=(0.0, 0.0, -1.0), position=(10.0, 5.0, 50.0))
    camera = SimpleNamespace(fov=0, width=20.0, height=5.0, zoom=1.0, world=world)
    assert view_extent(camera, (400, 200)) == pytest.approx((0.0, 20.0, 0.0, 10.0))
    camera.fov = 45
    assert view_extent(camera, (400, 200)) is None
