"""Measured data on the gauge plots: dashed lines and how well they match."""

import numpy as np
import pytest
from reef3d_fixtures import GAUGES, gauge_eta, make_run, show_plots

from lagoonviz.core import Session
from lagoonviz.core.analysis import compare, match_columns


def case_with_series(tmp_path):
    root = make_run(tmp_path / "flume", frames=3, print_interval=1.0, series=True)
    s = Session()
    s.open_case(root)
    show_plots(s)
    gauges = next(
        i.id for i in s.scene if i.kind == "series" and "wsf" in str(i.properties["source"])
    )
    return root, s, gauges


def write_measured(path, header=True, delay=0.0, scale=1.0):
    t = np.linspace(0.0, 2.0, 201)
    columns = [scale * gauge_eta(x, t - delay) for x, _ in GAUGES]
    lines = ["# flume test 3"]
    if header:
        lines.append("t " + " ".join(f"P{i + 1}" for i in range(len(GAUGES))))
    for tk, *row in zip(t, *columns, strict=True):
        lines.append(" ".join(f"{v:.6g}" for v in (tk, *row)))
    path.write_text("\n".join(lines) + "\n")
    return path


def test_measured_lines_and_their_match(tmp_path):
    root, s, gauges = case_with_series(tmp_path)
    write_measured(root / "measured.dat")
    s.set_property(gauges, "measured", "measured.dat")  # relative to the case folder
    model = s.data.traces(gauges)
    measured = s.data.measured_traces(gauges)
    assert [t.measured_for for t in measured] == [t.name for t in model]
    assert measured[0].name == f"{model[0].name} measured" and measured[0].unit == model[0].unit
    found = s.data.comparisons(gauges)
    assert set(found) == {t.name for t in model}
    stats = found[model[0].name]
    assert stats.rmse < 0.01 and stats.r > 0.99 and stats.skill > 0.99
    # only over the time asked
    assert s.data.comparisons(gauges, 0.0, 1.0)[model[0].name].end <= 1.0


def test_columns_in_order_by_name_or_as_told(tmp_path):
    root, s, gauges = case_with_series(tmp_path)
    write_measured(root / "plain.dat", header=False)
    s.set_property(gauges, "measured", str(root / "plain.dat"))  # absolute
    measured = s.data.measured_traces(gauges)
    model = s.data.traces(gauges)
    assert [t.measured_for for t in measured] == [t.name for t in model]  # in order
    s.set_property(gauges, "measured_columns", f"{model[0].name} = 2")
    measured = s.data.measured_traces(gauges)
    assert len(measured) == 1 and np.allclose(
        measured[0].values[:5], gauge_eta(GAUGES[1][0], np.linspace(0, 2, 201))[:5], atol=1e-6
    )
    assert match_columns(["P1", "P2"], ["P2", "P1"]) == [("P1", "P1"), ("P2", "P2")]
    assert match_columns(["P1", "P2"], ["1", "2", "3"]) == [("P1", "1"), ("P2", "2")]


def test_time_shift_and_lag(tmp_path):
    root, s, gauges = case_with_series(tmp_path)
    write_measured(root / "late.dat", delay=0.2)
    s.set_property(gauges, "measured", "late.dat")
    name = s.data.traces(gauges)[0].name
    late = s.data.comparisons(gauges)[name]
    s.set_property(gauges, "measured_shift", -0.2)
    shifted = s.data.comparisons(gauges)[name]
    assert shifted.rmse < late.rmse
    t = np.linspace(0, 60, 6001)
    lag = compare(t, np.sin(t - 0.3), t, np.sin(t)).lag
    assert lag == pytest.approx(0.3, abs=0.01)


def test_a_missing_file_draws_nothing(tmp_path):
    root, s, gauges = case_with_series(tmp_path)
    s.set_property(gauges, "measured", "nowhere.dat")
    assert s.data.measured_traces(gauges) == [] and s.data.comparisons(gauges) == {}


def test_the_figure_draws_them_dashed(tmp_path):
    from lagoonviz.render.figure import series_figure, series_rows

    root, s, gauges = case_with_series(tmp_path)
    write_measured(root / "measured.dat")
    s.set_property(gauges, "measured", "measured.dat")
    rows = dict(series_rows(s))
    assert any(t.measured_for for t in rows[gauges])
    figure = series_figure(s, rows=[(gauges, rows[gauges])])
    styles = {line.get_linestyle() for line in figure.axes[0].get_lines()}
    assert "--" in styles
