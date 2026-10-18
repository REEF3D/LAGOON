"""Data tables for 2D plots: files, expressions, fits, REEF3D series, runs collected."""

import numpy as np
import pytest
from reef3d_fixtures import make_run

from lagoonviz.core import Session
from lagoonviz.core.tables import DataTable, TableSpec, evaluate, fit, read_table


def test_reading_text_tables(tmp_path):
    path = tmp_path / "flume.dat"
    path.write_text("# test 3\ntime\tFz [N]\teta (m)\n0 1 0.1\n1 2 0.2\n2 3 0.3\n")
    table = read_table(path)
    assert table.names == ["time", "Fz", "eta"] and table.units == {"Fz": "N", "eta": "m"}
    assert table.rows == 3 and table.columns["Fz"][2] == 3.0
    plain = tmp_path / "plain.csv"
    plain.write_text("1,2\n3,4\n")
    assert read_table(plain).names == ["c1", "c2"]
    european = tmp_path / "eu.csv"
    european.write_text("t;eta\n0,5;1,25\n1,0;2,5\n")
    table = read_table(european)
    assert table.columns["eta"][0] == 1.25 and table.columns["t"][1] == 1.0
    ragged = tmp_path / "ragged.txt"
    ragged.write_text("a b c\n1 2 3\n4 5\n")
    assert np.isnan(read_table(ragged).columns["c"][1])


def test_expressions():
    columns = {
        "t": np.arange(5.0),
        "P1": np.array([0.0, 1.0, 0.0, -1.0, 0.0]),
        "Fz [N]": np.ones(5),
    }
    assert np.allclose(evaluate("2 * P1 + 1", columns), [1, 3, 1, -1, 1])
    assert evaluate("max(P1) - min(P1)", columns) == 2.0
    assert np.allclose(evaluate("where(P1 > 0, P1, 0)", columns), [0, 1, 0, 0, 0])
    assert np.allclose(evaluate('col("Fz [N]") * g', columns), 9.81)
    assert evaluate("rms(P1)", columns) == pytest.approx(np.sqrt(0.4))
    assert np.allclose(evaluate("(P1 > -0.5) & (t < 3)", columns), [1, 1, 1, 0, 0])
    assert np.allclose(evaluate("P1 if t else -1", columns), [-1, 1, 0, -1, 0])
    for bad in ("__import__('os')", "P1.real", "open('x')", "lambda: 1", "nonsense + 1"):
        with pytest.raises((ValueError, KeyError)):
            evaluate(bad, columns)


def test_formulas_in_a_table():
    table = DataTable("t", {"x": np.arange(4.0), "y": np.array([1.0, 2.0, 4.0, 8.0])})
    table.add_formulas({"y2": "2 * y", "dy": "diff(y)", "H": "max(y) - min(y)", "bad": "z + 1"})
    assert np.allclose(table.columns["y2"], [2, 4, 8, 16])
    assert np.allclose(table.columns["dy"][:3], [1, 2, 4]) and np.isnan(table.columns["dy"][3])
    assert np.allclose(table.columns["H"], 7.0)
    assert "bad" not in table.columns and "z" in table.errors["bad"]
    assert table.text().splitlines()[0] == "x\ty\ty2\tdy\tH"


def test_fits():
    x = np.linspace(1, 10, 50)
    line = fit(x, 3 * x + 1)
    assert line.params["a1"] == pytest.approx(3) and line.params["a0"] == pytest.approx(1)
    assert line.r2 == pytest.approx(1) and line.equation == "y = 3 x + 1"
    assert np.allclose(line(np.array([2.0])), 7)
    curve = fit(x, 2 * x**2 - x, "quadratic")
    assert np.allclose(curve(x), 2 * x**2 - x)
    growth = fit(x, 0.5 * np.exp(0.3 * x), "exponential")
    assert growth.params["b"] == pytest.approx(0.3)
    power = fit(x, 4 * x**1.5, "power")
    assert power.params["a"] == pytest.approx(4) and power.params["b"] == pytest.approx(1.5)
    with pytest.raises(ValueError):
        fit([1.0], [1.0])


def test_reef3d_series_and_collected_runs(tmp_path):
    first = make_run(tmp_path / "a", frames=3, print_interval=1.0, series=True)
    second = make_run(tmp_path / "b", frames=3, print_interval=1.0, series=True)
    s = Session()
    s.open_case(first)
    s.open_case(second)
    key = next(k for k in s.data.case("case-1").series if "forces" in k)
    table = TableSpec("forces", f"series:{key}", formulas={"Fz kN": "Fz / 1000"}).load(s)
    assert table.names[:2] == ["time", "Fx"] and "Fz kN" in table.names
    assert table.units["time"] == "s"
    runs = TableSpec(
        "study",
        "collect:",
        collect={"Fz max": f"{key}: max(Fz)", "bad": f"{key}: nothing(Fz)"},
        start=0.0,
        end=1.0,
    ).load(s)
    assert list(runs.columns["case"]) == [1.0, 2.0] and len(runs.labels) == 2
    assert runs.columns["Fz max"] == pytest.approx([2010.0, 2010.0])
    assert "bad" in runs.errors


def test_a_spec_round_trip_and_its_file(tmp_path):
    path = tmp_path / "m.dat"
    path.write_text("t a\n0 1\n1 2\n")
    spec = TableSpec("m", "file:m.dat", formulas={"b": "a * 2"})
    again = TableSpec.from_dict(spec.to_dict())
    table = again.load(base=tmp_path)
    assert list(table.columns["b"]) == [2.0, 4.0]
    assert spec.stamp(tmp_path) == path.stat().st_mtime


def test_an_xy_figure_from_tables(tmp_path):
    """DataGraph's job: columns over columns, points and a fit, a right axis, bars."""
    from lagoonviz.core.figure_template import FigureTemplate, Line, Panel
    from lagoonviz.render.figure import template_figure

    path = tmp_path / "study.dat"
    path.write_text("dx H F\n0.1 1.0 10\n0.05 1.2 11\n0.025 1.25 11.5\n0.0125 1.27 11.6\n")
    template = FigureTemplate(
        tables=[{"name": "study", "source": f"file:{path}", "formulas": {"N": "1 / dx"}}],
        panels=[
            Panel(
                kind="xy",
                lines=[
                    Line("", "H", table="study", x="N", plot="scatter", fit="power"),
                    Line("", "F", table="study", x="N", axis="right", linestyle="--"),
                ],
                logx=True,
                y2label="F [kN]",
            ),
            Panel(kind="xy", lines=[Line("", "H", table="study", x="dx", plot="bar")]),
        ],
    )
    s = Session()
    figure = template_figure(s, template)
    assert figure.missing == []
    ax = figure.axes[0]
    assert ax.get_xscale() == "log" and ax.get_xlabel() == "N"
    assert "F [kN]" in [a.get_ylabel() for a in figure.axes]  # the right axis of panel 1
    labels = [t.get_text() for t in ax.get_legend().get_texts()]
    assert any("$R^2$" in label for label in labels) and "F" in labels
    # kept with the figure, the file relative to it
    saved = template.save(tmp_path / "study.lagoonfig")
    assert '"file:study.dat"' in saved.read_text()
    again = FigureTemplate.load(saved)
    assert again.tables[0]["source"] == f"file:{path.resolve()}"
    assert again.panels[0].lines[0].fit == "power" and again.panels[0].logx
    path.unlink()
    broken = template_figure(s, again)
    assert any("table study" in m for m in broken.missing)
