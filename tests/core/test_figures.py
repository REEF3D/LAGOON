"""Publication figures (matplotlib): time series, wave profiles, the 3D view."""

import numpy as np
import pytest
from reef3d_fixtures import make_run

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
