"""Time series: REEF3D gauge, 6DOF and force tables, and their scene items."""

import numpy as np
import pytest
from reef3d_fixtures import GAUGES, gauge_eta, make_run, show_plots

from lagoonviz.core import Session
from lagoonviz.core.data.reef3d import open_case
from lagoonviz.core.data.series import (
    decimate,
    default_columns,
    find_series,
    parse_columns,
    read_series,
)


@pytest.fixture
def run(tmp_path):
    return make_run(tmp_path / "run", frames=5, volume=False, print_interval=0.5, series=True)


def test_wave_gauges(run):
    table = read_series(run / "REEF3D_NHFLOW_WSF" / "REEF3D-NHFLOW-WSF-HG.dat")
    assert table.names == ["P1", "P2", "P3"]
    assert table.locations == {"P1": (1.0, 2.0), "P2": (4.0, 2.0), "P3": (7.0, 2.0)}
    assert table.time[0] == 0 and table.time[-2] == pytest.approx(2.0)
    np.testing.assert_allclose(
        table.columns["P2"][:-1], gauge_eta(GAUGES[1][0], table.time[:-1]), atol=1e-5
    )
    # REEF3D's -1e20 and nan mean "no value"
    assert np.isnan(table.columns["P1"][-1]) and np.isnan(table.columns["P2"][-1])
    assert table.value_at("P1", 0.25) == pytest.approx(
        np.interp(0.25, table.time, table.columns["P1"])
    )
    assert np.isnan(table.value_at("P1", 99.0))


def test_6dof_files_units_and_defaults(run):
    folder = run / "REEF3D_NHFLOW_6DOF"
    velocity = read_series(folder / "REEF3D_6DOF_velocity_0.dat")
    assert velocity.names == ["Ue", "We", "Qe"]
    assert velocity.units == {"Ue": "m/s", "We": "m/s", "Qe": "rad/s"}
    assert default_columns(velocity) == ["We"]
    forces = read_series(folder / "REEF3D_6DOF_forces_0.dat")
    assert default_columns(forces) == ["Fx", "Fz"]
    assert default_columns(read_series(folder / "REEF3D_6DOF_position_0.dat")) == ["ZG"]
    assert parse_columns("Fz; Fy , nope", forces) == ["Fz", "Fy"]


def test_force_file_with_iteration_column(tmp_path):
    path = tmp_path / "REEF3D_FNPF_Force_ALE" / "REEF3D_FNPF_ALE_Force-1.dat"
    path.parent.mkdir()
    path.write_text(
        "x \t y \t Cd \t Cm\n15.281 \t 0.05 \t 0.7 \t 2\n\n\nit \t time \t Fx \t Fy \n"
        "1 \t 0 \t 1.5 \t 0\n2 \t 0.002 \t 2.5 \t 0\n3 \t 0.004"
    )  # last row unfinished
    table = read_series(path)
    assert table.names == ["Fx", "Fy"]
    np.testing.assert_allclose(table.time, [0, 0.002])
    found = find_series(tmp_path)
    assert list(found) == ["fnpf_ale_force_1"]
    assert found["fnpf_ale_force_1"].name == "Morison force 1"  # (P 85)


def test_not_a_series(tmp_path):
    path = tmp_path / "notes.dat"
    path.write_text("hello\n1 2 3\n")
    with pytest.raises(ValueError):
        read_series(path)


def test_found_in_case_and_named(run):
    case = open_case(run)
    assert set(case.series) == {
        "nhflow_wsf_hg",
        "nhflow_6dof_forces_0",
        "nhflow_6dof_position_0",
        "nhflow_6dof_velocity_0",
    }
    names = [s.name for s in case.main_series()]
    assert names == ["Wave gauges", "Body 0 forces", "Body 0 motion"]


def test_open_case_adds_plots_and_traces(run):
    s = Session()
    s.open_case(run)
    series = [i for i in s.scene if i.kind == "series"]
    assert [i.name for i in series] == ["Wave gauges", "Body 0 forces", "Body 0 motion"]
    traces = s.data.traces(series[1].id)
    assert [t.name for t in traces] == ["Fx", "Fz"]
    s.set_property(series[0].id, "columns", "P3, P1")
    assert [t.name for t in s.data.traces(series[0].id)] == ["P3", "P1"]
    assert s.data.time_in_seconds()
    replayed = Session.from_script(s.to_script())
    assert replayed.scene.to_dict() == s.scene.to_dict()


def test_only_series_still_opens(tmp_path):
    run = make_run(tmp_path / "gauges_only", frames=3, volume=False, series=True)
    for folder in run.glob("REEF3D_NHFLOW_VTP_*"):
        for path in folder.iterdir():
            path.unlink()
        folder.rmdir()
    s = Session()
    s.open_case(run / "REEF3D_NHFLOW_6DOF")
    assert s.data.time_range() == (0.0, pytest.approx(3.0), "s")
    assert s.data.series_end() == pytest.approx(3.0)  # the gauge file's last row


def test_file_is_read_again_when_it_grows(run):
    s = Session()
    s.open_case(run)
    path = run / "REEF3D_NHFLOW_6DOF" / "REEF3D_6DOF_forces_0.dat"
    before = len(s.data.series("series-2"))
    import os

    with open(path, "a") as handle:
        handle.write("99 \t 1 \t 0 \t 2\n")
    stat = path.stat()
    os.utime(path, (stat.st_atime, stat.st_mtime + 5))
    assert len(s.data.series("series-2")) == before + 1


def test_decimate_keeps_extremes():
    t = np.linspace(0, 1, 100_001)
    v = np.sin(40 * t)
    v[50_000] = 5.0
    dt, dv = decimate(t, v, 200)
    assert dt.size == 400 and dv.max() == 5.0 and dv.min() == pytest.approx(v.min(), abs=1e-6)
    assert np.all(np.diff(dt) >= 0)


def test_gauges_in_3d_follow_time_and_plot_colours(run):
    from reef3d_fixtures import DEPTH

    from lagoonviz.core.colormaps import SERIES_PALETTES

    s = Session()
    s.open_case(run)
    gauges = next(i for i in s.scene if i.kind == "series" and i.name == "Wave gauges")
    assert not gauges.visible  # there, switched off: a tick shows them
    show_plots(s)
    surface = next(i for i in s.scene if i.kind == "free_surface")
    # the still water level comes from the free surface, once it was shown
    s.data.display(surface.id, s.data.load(surface.id))
    s.set_time(1.0)
    marks = s.data.gauge_marks(gauges.id)
    assert marks.names == ["P1", "P2", "P3"]
    assert marks.colors == SERIES_PALETTES["light"][:3]
    assert marks.visible
    xs = [x for x, _ in GAUGES]
    np.testing.assert_allclose(marks.dots[:, 0], xs)
    np.testing.assert_allclose(marks.dots[:, 2], DEPTH + gauge_eta(np.array(xs), 1.0), atol=1e-5)
    assert np.all(marks.staffs[:, 0, 2] < marks.dots[:, 2])
    assert np.all(marks.staffs[:, 1, 2] > marks.dots[:, 2])

    # the plot shows P3 first: so does its colour in 3D
    s.set_property(gauges.id, "columns", "P3")
    marks = s.data.gauge_marks(gauges.id, ["#000000", "#ffffff"])
    assert marks.names == ["P3"] and marks.colors == ["#000000"]
    s.set_property(gauges.id, "gauges_3d", False)
    assert not s.data.gauge_marks(gauges.id).visible
    # files without positions have no marks
    forces = next(i for i in s.scene if i.name == "Body 0 forces")
    assert s.data.gauge_marks(forces.id) is None
