"""Figure templates: what a figure shows, saved apart from the run."""

import pytest
from reef3d_fixtures import make_run

from lagoonviz.core import Session
from lagoonviz.core.figure_template import (
    FigureTemplate,
    Line,
    Panel,
    resolve,
    series_columns,
)


@pytest.fixture
def session(tmp_path):
    run = make_run(tmp_path / "run", frames=4, print_interval=0.5, series=True)
    s = Session()
    s.open_case(run)
    return s


def test_what_can_be_drawn(session):
    columns = series_columns(session)
    gauges = next(k for k, (name, _) in columns.items() if name == "Wave gauges")
    assert columns[gauges][1][:3] == ["P1", "P2", "P3"]


def test_round_trip_and_measured_files_next_to_it(session, tmp_path):
    lab = tmp_path / "figs" / "lab.txt"
    lab.parent.mkdir()
    lab.write_text("t WG\n0 0\n1 1\n")
    template = FigureTemplate(
        panels=[
            Panel(
                lines=[Line("nhflow_wsf_hg", "P1", label="computed"), Line("", "WG", file=str(lab))]
            ),
            Panel(kind="spectrum", start=0.5),
        ],
        style="Journal (serif)",
        width_mm=140,
    )
    path = template.save(tmp_path / "figs" / "gauge")
    assert path.suffix == ".lagoonfig"
    assert '"file": "lab.txt"' in path.read_text()  # relative: template and data move together
    again = FigureTemplate.load(path)
    assert again.style == "Journal (serif)" and again.panels[1].kind == "spectrum"
    assert again.panels[0].lines[0].label == "computed"
    assert again.panels[0].lines[1].file == str(lab.resolve())
    with pytest.raises(ValueError):
        FigureTemplate.from_dict({"panels": []})


def test_resolving_lines(session, tmp_path):
    key = next(iter(series_columns(session)))
    column = series_columns(session)[key][1][0]
    data = resolve(session, Line(key, column))
    assert data.time.size == data.values.size > 2 and data.name == column
    lab = tmp_path / "lab.txt"
    lab.write_text("t WG\n0 0.1\n1 0.2\n")
    measured = resolve(session, Line("", "WG", file=str(lab)))
    assert list(measured.values) == [0.1, 0.2] and measured.name == "lab WG"
    with pytest.raises(KeyError):
        resolve(session, Line("nothing", "P1"))
    with pytest.raises(KeyError):
        resolve(session, Line(key, "nothing"))


def test_template_figure(session, tmp_path):
    pytest.importorskip("matplotlib")
    from lagoonviz.render.figure import save_figure, template_figure, template_spec

    key = next(k for k, (name, _) in series_columns(session).items() if name == "Wave gauges")
    data = resolve(session, Line(key, "P1"))
    lab = tmp_path / "lab.txt"
    lab.write_text(
        "t WG\n"
        + "".join(f"{t} {v + 0.001}\n" for t, v in zip(data.time, data.values, strict=True))
    )
    template = FigureTemplate(
        panels=[
            Panel(
                lines=[Line(key, "P1"), Line("", "WG", file=str(lab))],
                compare=True,
                ylabel=r"$\eta$ [m]",
            ),
            Panel(lines=[Line(key, "P2", color="#ff0000", linestyle="--")], ymin=-1, ymax=1),
            Panel(kind="spectrum", lines=[Line(key, "P1"), Line("gone", "P9")]),
        ],
        title="Run 12",
    )
    figure = template_figure(session, template)
    axes = figure.get_axes()
    assert len(axes) == 3 and axes[0].get_ylabel() == r"$\eta$ [m]"
    assert any("RMSE" in t.get_text() for t in axes[0].texts)
    assert axes[1].get_lines()[0].get_color() == "#ff0000" and axes[1].get_ylim() == (-1, 1)
    assert axes[2].get_xlabel() == "f [Hz]"
    assert figure.missing and "P9" in figure.missing[0]
    save_figure(figure, tmp_path / "t.pdf", template_spec(template))
    with pytest.raises(ValueError):
        template_figure(session, FigureTemplate())


# ======================================================================== paper figures
def _fnpf(tmp_path):
    run = make_run(
        tmp_path / "fnpf", solver="FNPF", frames=4, print_interval=0.5, series=True, profiles=True
    )
    s = Session()
    s.open_case(run)
    s.set_time(1.0)
    return s


def test_grid_layout():
    from lagoonviz.core.figure_template import layout

    template = FigureTemplate(
        panels=[Panel(span=2), Panel(), Panel(), Panel(), Panel(span=5)], columns=2
    )
    assert layout(template) == [[(0, 0, 2)], [(1, 0, 1), (2, 1, 1)], [(3, 0, 1)], [(4, 0, 2)]]
    assert layout(FigureTemplate(panels=[Panel(), Panel()])) == [[(0, 0, 1)], [(1, 0, 1)]]


def test_wave_profile_lines(tmp_path):
    from lagoonviz.core.figure_template import profile_columns

    session = _fnpf(tmp_path)
    found = profile_columns(session)
    key = next(iter(found))
    columns = [c for c, _ in found[key][1]]
    assert columns[:2] == ["1", "1 theory"]
    line = Line(key, "1", data="profile")
    early, late = resolve(session, line, 0.2), resolve(session, line, 1.4)
    assert early.unit == "m" and early.time.size == early.values.size > 10
    assert not (early.values == late.values).all()  # another moment, another profile
    theory = resolve(session, Line(key, "1 theory", data="profile"), 1.4)
    assert theory.name == "theory" and Line(key, "1 theory", data="profile").theory
    assert Line(key, "1", data="profile").title() == "line 1"
    with pytest.raises(KeyError):
        resolve(session, Line(key, "9", data="profile"), 1.0)


def test_a_paper_figure(tmp_path):
    """Two columns: a 3D view over the whole width, a profile and a gauge under it,
    a spectrum; lettered, with notes; drawn and written."""
    import numpy as np

    from lagoonviz.core.figure_template import Note, profile_columns
    from lagoonviz.render.figure import colour_bars, save_figure, template_figure, template_spec

    session = _fnpf(tmp_path)
    gauges = next(k for k, (name, _) in series_columns(session).items() if name == "Wave gauges")
    profiles = next(iter(profile_columns(session)))
    item = next(i for i in session.scene if i.kind == "free_surface")
    session.set_property(item.id, "variable", "eta")
    template = FigureTemplate(
        panels=[
            Panel(kind="view", span=2, time=1.0, camera={"position": [1, 2, 3]}),
            Panel(
                kind="profile",
                time=1.2,
                lines=[
                    Line(profiles, "1", data="profile"),
                    Line(profiles, "1 theory", data="profile"),
                ],
                notes=[Note("$\\lambda$ = 4 m", 0.05, 0.9)],
            ),
            Panel(lines=[Line(gauges, "P1")], title="Gauge 1"),
            Panel(kind="spectrum", lines=[Line(gauges, "P1")], span=2),
        ],
        columns=2,
        letters=True,
        width_mm=190.0,
    )
    spec = template_spec(template, dpi=100)
    image = np.zeros((90, 160, 4), np.uint8)
    image[..., 3] = 255
    figure = template_figure(session, template, spec, images={0: image})
    assert figure.missing == []
    titles = [ax.get_title(loc="left") for ax in figure.get_axes() if ax.get_title(loc="left")]
    assert titles == ["(a) t = 1.00 s", "(b) t = 1.20 s", "(c) Gauge 1", "(d)"]
    bars = len(colour_bars(session))
    assert bars >= 1 and len(figure.get_axes()) == 4 + bars
    view, profile, gauge = (
        figure.get_axes()[0],
        figure.get_axes()[1 + bars],
        figure.get_axes()[2 + bars],
    )
    # the profile and the gauge side by side, under the view
    assert profile.get_position().y0 == pytest.approx(gauge.get_position().y0, abs=0.02)
    assert profile.get_position().x1 < gauge.get_position().x0
    assert view.get_position().y0 > profile.get_position().y1
    assert profile.get_xlabel() == "x [m]" and len(profile.get_lines()) == 2
    assert profile.get_lines()[1].get_linestyle() == "--"  # the wave theory, dashed
    assert any(t.get_text() == "$\\lambda$ = 4 m" for t in profile.texts)
    path = save_figure(figure, tmp_path / "paper.pdf", spec)
    assert path.stat().st_size > 2000
    # the session's view is as it was
    assert session.scene.time == 1.0 and session.scene.camera is None
    # kept in a template file, made again from the recorded script code
    saved = FigureTemplate.load(template.save(tmp_path / "paper"))
    assert saved == template
    assert saved.panels[1].notes[0].text == "$\\lambda$ = 4 m"


def test_view_panel_without_a_renderer(tmp_path, monkeypatch):
    """No GPU renderer: the view panel is left out and named, the rest is drawn."""
    import lagoonviz.render.figure as module

    def unavailable(*args, **kwargs):
        raise RuntimeError("no GPU adapter")

    monkeypatch.setattr(module, "render_view", unavailable)
    session = _fnpf(tmp_path)
    gauges = next(k for k, (name, _) in series_columns(session).items() if name == "Wave gauges")
    template = FigureTemplate(panels=[Panel(kind="view"), Panel(lines=[Line(gauges, "P1")])])
    figure = module.template_figure(session, template, module.template_spec(template, dpi=80))
    assert figure.missing == ["panel 1: no GPU adapter"]


@pytest.mark.parametrize("columns", [1, 2])
def test_titles_stay_inside_the_figure(tmp_path, columns):
    """Images side by side shrink to their aspect after the layout: their titles must
    still fit."""
    import numpy as np
    from matplotlib.backends.backend_agg import FigureCanvasAgg

    from lagoonviz.render.figure import _context, template_figure, template_spec

    session = _fnpf(tmp_path)
    template = FigureTemplate(
        panels=[Panel(kind="view", time=1.0), Panel(kind="view", time=1.5)],
        columns=columns,
        letters=True,
        width_mm=190.0,
    )
    for dpi in (100, 300):
        spec = template_spec(template, dpi=dpi)
        image = np.full((300, 534, 4), 200, np.uint8)
        with _context(spec):
            figure = template_figure(session, template, spec, images={0: image, 1: image})
            canvas = FigureCanvasAgg(figure)
            canvas.draw()
            top = max(
                ax._left_title.get_window_extent(canvas.get_renderer()).y1
                for ax in figure.get_axes()
            )
        assert top <= figure.get_size_inches()[1] * figure.dpi


def test_notes_and_what_fits_where():
    from lagoonviz.core.figure_template import Note, fits, notes_text, parse_notes

    text = "$H_s$ = 2 m\n\n0.6, 0.2: $T_p = 10$ s\nthird"
    notes = parse_notes(text)
    assert notes == [
        Note("$H_s$ = 2 m", 0.03, 0.95),
        Note("$T_p = 10$ s", 0.6, 0.2),
        Note("third", 0.03, 0.95 - 0.16),
    ]
    assert notes_text(notes) == "$H_s$ = 2 m\n0.6, 0.2: $T_p = 10$ s\nthird"
    assert parse_notes(notes_text(notes)) == notes
    gauge, profile = Line("hg", "P1"), Line("wsfline", "1", data="profile")
    measured = Line("", "WG1", file="lab.txt")
    assert fits(gauge, "time") and fits(gauge, "spectrum") and not fits(gauge, "profile")
    assert fits(profile, "profile") and not fits(profile, "time")
    assert fits(measured, "time") and fits(measured, "profile")
    assert not any(fits(line, "view") for line in (gauge, profile, measured))
