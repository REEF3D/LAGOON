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
