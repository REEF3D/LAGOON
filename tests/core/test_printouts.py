"""REEF3D's print-outs as REEF3D writes them (the C++ print code, 2026): probes,
run-up and bed gauges, the wave theory's time series, free surface lines along x
and y and time-averaged ones; named as the REEF3D User Guide names them."""

import numpy as np
import pytest

from lagoonviz.core.data.profiles import find_profiles, read_profile
from lagoonviz.core.data.reef3d import open_case
from lagoonviz.core.data.series import find_series, read_series

TIMES = (0.0, 0.05, 0.1, 0.15)


def _write(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return path


def _probe(kind, header, columns):
    """A probe file: its id, position, two blank lines, the header, the rows."""
    rows = "".join(
        f"{t:.9g}" + "".join(f" \t {v:.6g}" for v in column(t)) + "\n"
        for t in TIMES
        for column in [columns]
    )
    return (
        f"{kind} ID:  1\n\nx_coord     y_coord     z_coord\n1\t 4.5\t 0.5\t -0.2\n\n\n"
        + (header + "\n" if header else "")
        + rows
    )


@pytest.fixture
def case(tmp_path):
    root = tmp_path / "case"
    probes = root / "REEF3D_NHFLOW_ProbePoint"
    # nhflow_probe_vel.cpp: "t \t U \t V \t W "
    _write(
        probes / "REEF3D-NHFLOW-Vel-Probe-1.dat",
        _probe("Vel Probe", "t \t U \t V \t W ", lambda t: (np.sin(t), 0.1, -t)),
    )
    _write(
        probes / "REEF3D-NHFLOW-Vel-Probe-Theory-1.dat",
        _probe("Vel Probe", "t \t U \t V \t W ", lambda t: (np.cos(t), 0.0, t)),
    )
    # nhflow_probe_press.cpp: headed "t \t U \t V \t W " but the rows: time, pressure
    _write(
        probes / "REEF3D-NHFLOW-Probe_Press-1.dat",
        _probe("Vel Probe", "t \t U \t V \t W ", lambda t: (9810.0 + t,)),
    )
    # probe_pressure.cpp (CFD): no header at all
    _write(
        root / "REEF3D_CFD_PressureProbe" / "REEF3D-CFD-Probe-Pressure-1.dat",
        _probe("Point Probe", "", lambda t: (100.0 * t,)),
    )
    # nhflow_print_runup_gage_x.cpp: the columns named, the time not (one gauge: with
    # more, REEF3D writes the z of one and the x of the next without a space between)
    runup = "number of runup-probes:  1\n\nline_No     y_coord\n1\t 0.5\n\n\n"
    runup += "X 1\t P 1 \t \t \n\n"
    runup += "".join(f"{t:.9g}\t{10 + t}  \t  {0.1 * t}\n" for t in TIMES)
    _write(root / "REEF3D_NHFLOW_RUNUP" / "REEF3D-NHFLOW-runup-x.dat", runup)
    # bedprobe_point.cpp
    bed = "number of gauges:  1\n\nx_coord     y_coord\n1\t 3.0\t 0.5\n\n\ntime\t P1\n"
    bed += "".join(f"{t}\t {-0.1 * t}  \t  \n" for t in TIMES)
    _write(root / "REEF3D_NHFLOW_Sediment" / "REEF3D-NHFLOW-Sediment-Point.dat", bed)
    # iowave_timeseries.cpp: positions without a number, "t \t eta"
    theory = "Wave Timeseries ID:  0\n\nx_coord     y_coord     T\n\t 2.0\t 0.5\t 12800\n\n"
    theory += "t \t eta\n\n" + "".join(f"{t} \t {np.sin(t)}\n" for t in TIMES)
    _write(root / "REEF3D_Log-Wave" / "REEF3D-Wave-Timeseries-1.dat", theory)
    return root


def test_probes_and_gauges_are_read(case):
    vel = read_series(case / "REEF3D_NHFLOW_ProbePoint" / "REEF3D-NHFLOW-Vel-Probe-1.dat")
    assert vel.names == ["U", "V", "W"] and len(vel) == len(TIMES)
    np.testing.assert_allclose(vel.time, TIMES)
    np.testing.assert_allclose(vel.columns["W"], [-t for t in TIMES])
    assert vel.locations == {"P1": (4.5, 0.5, -0.2)}
    press = read_series(case / "REEF3D_NHFLOW_ProbePoint" / "REEF3D-NHFLOW-Probe_Press-1.dat")
    assert press.names == ["p"]
    np.testing.assert_allclose(press.columns["p"], [9810.0 + t for t in TIMES])
    cfd = read_series(case / "REEF3D_CFD_PressureProbe" / "REEF3D-CFD-Probe-Pressure-1.dat")
    assert cfd.names == ["p"] and len(cfd) == len(TIMES)
    runup = read_series(case / "REEF3D_NHFLOW_RUNUP" / "REEF3D-NHFLOW-runup-x.dat")
    assert runup.names == ["X 1", "P 1"]
    np.testing.assert_allclose(runup.columns["X 1"], [10 + t for t in TIMES])
    bed = read_series(case / "REEF3D_NHFLOW_Sediment" / "REEF3D-NHFLOW-Sediment-Point.dat")
    assert bed.names == ["P1"] and bed.locations == {"P1": (3.0, 0.5)}
    theory = read_series(case / "REEF3D_Log-Wave" / "REEF3D-Wave-Timeseries-1.dat")
    assert theory.names == ["eta"] and theory.locations == {"P1": (2.0, 0.5, 12800.0)}


def test_a_file_being_written_and_not_a_series(tmp_path):
    path = _write(
        tmp_path / "REEF3D-CFD-Probe-Pressure-1.dat",
        _probe("Point Probe", "", lambda t: (t,)) + "0.2 \t",  # a row half written
    )
    assert len(read_series(path)) == len(TIMES)
    notes = _write(tmp_path / "notes.dat", "hello\n1 2 3\n")
    with pytest.raises(ValueError):
        read_series(notes)


def test_named_as_reef3d_names_them(case):
    names = {f.path.name: f.name for f in find_series(case).values()}
    assert names == {
        "REEF3D-NHFLOW-Vel-Probe-1.dat": "Velocity probe 1",
        "REEF3D-NHFLOW-Vel-Probe-Theory-1.dat": "Velocity probe 1 (theory)",
        "REEF3D-NHFLOW-Probe_Press-1.dat": "Pressure probe 1",
        "REEF3D-CFD-Probe-Pressure-1.dat": "Pressure probe 1",
        "REEF3D-NHFLOW-runup-x.dat": "Run-up gauges",
        "REEF3D-NHFLOW-Sediment-Point.dat": "Bed level gauges",
        "REEF3D-Wave-Timeseries-1.dat": "Theoretical wave time series 1",
    }
    case_ = open_case(case)
    shown = {f.name for f in case_.main_series(limit=20)}
    assert "Theoretical wave time series 1" not in shown  # REEF3D writes one always
    assert "Velocity probe 1" in shown


@pytest.mark.parametrize(
    "stem, name",
    [
        ("REEF3D-FNPF-WSF-HG", "Wave gauges"),
        ("REEF3D-FNPF-WSF-HG-THEORY", "Wave gauges (theory)"),
        ("REEF3D-CFD-Probe-Point-2", "Point probe 2"),
        ("REEF3D-SFLOW-Probe-Point-3", "Depth-averaged point probe 3"),
        ("REEF3D_CFD_Force-1", "Force box 1"),
        ("REEF3D_NHFLOW_ALE_Force-2", "Morison force 2"),
        ("REEF3D_Runup-1", "Cylinder run-up 1"),
        ("REEF3D-NHFLOW-runup-max-x", "Maximum run-up gauges"),
        ("REEF3D-CFD-Sediment-Bedshear-Max", "Maximum bed shear stress"),
        ("REEF3D-CFD-Sediment-Bedshear", "Bed shear stress gauges"),
        ("REEF3D-CFD-Sediment-Max", "Maximum bed change"),
    ],
)
def test_reef3d_names(tmp_path, stem, name):
    from lagoonviz.core.data.series import SeriesFile

    assert SeriesFile("k", tmp_path / f"{stem}.dat", "CFD", "probes").name == name


def _line_file(folder, number, time, along, across, theory=False):
    """A free surface line file as nhflow_print_wsfline(_y).cpp writes it."""
    axis = "Y" if folder.name.endswith("_Y") else "X"
    coord = "x_coord" if axis == "Y" else "y_coord"
    text = f"simtime:  {time}\nnumber of wsf-lines:  1\n\nline_No     {coord}\n1\t {across}\n\n\n"
    text += f"{axis} 1\t P 1 \t \t " + ("\t \t W 1" if theory else "") + "\n\n"
    for a in along:
        text += f"{a} \t {0.1 * np.sin(a + time)} \t  " + (
            f"{0.1 * np.cos(a)} \t  " if theory else ""
        )
        text += "\n"
    return _write(folder / f"REEF3D-NHFLOW-wsfline-{number:08d}.dat", text)


def test_free_surface_lines_along_y_and_time_averaged(tmp_path):
    case = tmp_path / "case"
    for number, time in enumerate((0.0, 0.5, 1.0)):
        _line_file(case / "REEF3D_NHFLOW_WSFLINE", number, time, np.linspace(0, 8, 9), 0.5)
        _line_file(case / "REEF3D_NHFLOW_WSFLINE_Y", number, time, np.linspace(0, 4, 5), 2.0, True)
    # nhflow_print_timeavg_wsfline.cpp: one file per line, at the end of its window
    folder = case / "REEF3D_NHFLOW_TIME_AVG_WSFLINE"
    for line in (1, 2):
        text = (
            f"Time-Averaged Water Surface Lineprobe ID:  {line}\n\nsimtime:  10.0\n"
            f"starttime:  2.0\naveraging_time:  8.0\n\nline_No     y_coord\n{line}\t {line}.5\n\n"
            f"X {line}\t P {line}\n\n" + "".join(f"{x} \t {0.01 * x}\n" for x in range(5))
        )
        _write(folder / f"REEF3D-NHFLOW-TimeAvgWsfLine-{line}-500.dat", text)
    sets = find_profiles(case)
    names = {key: s.name for key, s in sets.items()}
    assert names == {
        "nhflow_wsfline": "Free surface lines",
        "nhflow_wsfline_y": "Free surface lines in y",
        "nhflow_time_avg_wsfline_1": "Time-averaged free surface line 1",
        "nhflow_time_avg_wsfline_2": "Time-averaged free surface line 2",
    }
    along_y = sets["nhflow_wsfline_y"].at(0.5).lines[1]
    assert along_y.axis == "y" and along_y.across == "x" and along_y.y == 2.0
    np.testing.assert_allclose(along_y.x, np.linspace(0, 4, 5))
    assert along_y.theory is not None and along_y.eta.shape == (5,)
    averaged = sets["nhflow_time_avg_wsfline_2"].at(10.0).lines[2]
    assert averaged.y == 2.5 and averaged.eta[-1] == pytest.approx(0.04)


def test_cfd_lines_along_y_with_one_coordinate_column(tmp_path):
    # print_wsfline_y.cpp: "y_coord \t \t P 1", rows: y, elevation of each line
    text = "simtime:  1.5\nnumber of wsf-lines_y:  1\n\nline_No     x_coord\n1\t 3.0\n\n\n"
    text += "y_coord\t \t P 1\n\n" + "".join(f"{y} \t {0.2 * y} \t  \n" for y in range(4))
    path = _write(tmp_path / "REEF3D_CFD_WSFLINE_Y" / "REEF3D-CFD-wsfline_y-00000001.dat", text)
    line = read_profile(path, "y").lines[1]
    assert line.y == 3.0 and line.x.tolist() == [0, 1, 2, 3] and line.eta[-1] == pytest.approx(0.6)


def test_lines_along_y_in_3d(tmp_path):
    from reef3d_fixtures import make_run

    from lagoonviz.core import Session

    run = make_run(tmp_path / "run", frames=3)
    for number, time in enumerate((0.0, 1.0, 2.0)):
        _line_file(run / "REEF3D_NHFLOW_WSFLINE_Y", number, time, np.linspace(0, 4, 5), 2.0)
    s = Session()
    s.open_case(run)
    item = next(i for i in s.scene if i.kind == "profile")
    assert item.name == "Free surface lines in y"
    marks = s.data.profile_marks(item.id)
    xs, ys = marks.staffs[..., 0], marks.staffs[..., 1]
    assert np.allclose(xs, 2.0) and ys.min() == 0.0 and ys.max() == 4.0  # along y at x = 2
