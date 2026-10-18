"""The Run monitor's reading of a run: run log, main and max logs, control files."""

import json
import os
import time

import numpy as np
import pytest

from lagoonviz.core.run_monitor import (
    ctrl_changes,
    duration,
    read_tail,
    run_status,
)

RUN = "20261017T154840Z-4f21"


def write_case(root, iterations=200, dt=0.005, itertime=2.0, end=None, ctrl_now=None,
               solver="CFD", dt_tail=None, speed_tail=None):  # fmt: skip
    root.mkdir(parents=True, exist_ok=True)
    started = "P 30 0.05\nN 41 6.0\nN 45 100000\nA 10 6\n"
    (root / "ctrl.txt").write_text(ctrl_now if ctrl_now is not None else started)
    case = root / "REEF3D_Case"
    case.mkdir(exist_ok=True)
    copy = case / f"ctrl-{RUN}.txt"
    copy.write_text(started)
    records = [
        {
            "type": "run", "format": "reef3d-run", "version": 1, "run": RUN,
            "solver": solver, "started": "2026-10-17T15:48:40Z",
            "ctrl": {"file": "ctrl.txt", "copy": f"REEF3D_Case/ctrl-{RUN}.txt"},
        },
        {"type": "output", "run": RUN, "step": 0, "time": 0.0, "streams": ["v"]},
        {"type": "output", "run": RUN, "step": 1, "time": 0.05, "streams": ["v"]},
    ]  # fmt: skip
    if end:
        records.append({"type": "end", "run": RUN, "status": end})
    (case / f"REEF3D_{solver}_run.jsonl").write_text("".join(json.dumps(r) + "\n" for r in records))
    logs = root / ("REEF3D_SFLOW_Log" if solver == "SFLOW" else "REEF3D_Log")
    logs.mkdir(exist_ok=True)
    steps = np.full(iterations, dt)
    if dt_tail is not None:
        steps[-20:] = dt_tail
    lines = ["REEF3D version:  v_261017", "", "", "number of cells:  300000", "",
             "#iteration \t #timestep \t #simtime \t #itertime \t #piter"]  # fmt: skip
    simtime = np.cumsum(steps)
    for k in range(iterations):
        lines.append(f"{k + 1} \t {steps[k]:.5f} \t {simtime[k]:.5f} \t {itertime:.4f} \t 10")
    prefix = "REEF3D_SFLOW_" if solver == "SFLOW" else "REEF3D_"
    (logs / f"{prefix}mainlog.dat").write_text("\n".join(lines) + "\n")
    speeds = np.full(iterations, 1.5)
    if speed_tail is not None:
        speeds[-5:] = speed_tail
    lines = ["number of cells:  300000", "", "#iteration \t #umax \t\t #vmax \t\t #wmax"]
    lines += [f"{k + 1}\t \t{speeds[k]} \t 0.5 \t 0.4 \t 1e-5" for k in range(iterations)]
    (logs / f"{prefix}maxlog.dat").write_text("\n".join(lines) + "\n")
    return root


def test_a_run_going_on(tmp_path):
    root = write_case(tmp_path / "case")
    status = run_status(root)
    assert status.state == "running" and status.solver == "CFD" and status.run_id == RUN
    assert status.iteration == 200 and status.time == pytest.approx(1.0, abs=1e-6)
    assert status.dt == pytest.approx(0.005) and status.end_time == 6.0
    assert status.seconds_per_iteration == pytest.approx(2.0)
    assert status.simulated_per_hour == pytest.approx(0.005 / 2.0 * 3600)
    assert status.eta == pytest.approx((6.0 - 1.0) / 0.0025)  # 5 s at 2.5 ms per s
    assert status.progress == pytest.approx(1.0 / 6.0)
    assert status.outputs == 2 and status.last_output_time == 0.05
    assert status.max_velocity == pytest.approx(1.5)
    assert status.warnings == [] and status.ctrl_changes == []
    assert list(status.history["dt"][-3:]) == pytest.approx([0.005] * 3)


def test_finished_and_stopped(tmp_path):
    assert run_status(write_case(tmp_path / "a", end="finished")).state == "finished"
    stopped = run_status(write_case(tmp_path / "b", end="diverged"))
    assert stopped.state == "stopped" and stopped.ended == "diverged" and stopped.eta is None


def test_a_quiet_log_is_not_responding(tmp_path):
    root = write_case(tmp_path / "case")
    log = root / "REEF3D_Log" / "REEF3D_mainlog.dat"
    old = time.time() - 3600
    os.utime(log, (old, old))
    status = run_status(root)
    assert status.state == "not responding" and status.eta is None
    assert any("killed" in w for w in status.warnings)


def test_warnings_of_a_run_blowing_up(tmp_path):
    root = write_case(tmp_path / "case", dt_tail=0.0001, speed_tail=40.0)
    status = run_status(root)
    assert any("time step dropped" in w for w in status.warnings)
    assert any("largest velocity rose" in w for w in status.warnings)


def test_ctrl_changed_after_the_start(tmp_path):
    """The case of 2026-10-17: P 30 0.05 at the start, 0.025 ten seconds later."""
    now = "P 30 0.025\nN 41 6.0\nN 45 100000\nA 10 6\n// a comment\n"
    status = run_status(write_case(tmp_path / "case", ctrl_now=now))
    assert status.ctrl_changes == ["-P 30 0.05", "+P 30 0.025"]
    assert any("ctrl.txt was changed" in w for w in status.warnings)
    assert status.end_time == 6.0  # the run's N 41, from its copy
    assert ctrl_changes("A 10 6  // CFD\n\nP 30 0.1", "A 10 6\nP 30   0.1") == []


def test_sflow_logs_and_no_run(tmp_path):
    status = run_status(write_case(tmp_path / "sflow", solver="SFLOW"))
    assert status.iteration == 200 and status.solver == "SFLOW"
    empty = tmp_path / "empty"
    empty.mkdir()
    assert run_status(empty).state == "no run"


def test_the_tail_of_a_long_log(tmp_path):
    path = tmp_path / "REEF3D_mainlog.dat"
    lines = ["header"] + [f"{k} \t 0.001 \t {k * 0.001:.5f} \t 1.0" for k in range(1, 200_001)]
    path.write_text("\n".join(lines) + "\n12")  # a line being written
    tail = read_tail(path)
    assert tail.iteration[-1] == 200_000 and len(tail.iteration) < 200_000
    assert tail.columns.shape[1] == 3


def test_durations():
    assert duration(42) == "42 s" and duration(600) == "10 min"
    assert duration(3 * 3600 + 1200) == "3 h 20 min" and duration(3 * 86400) == "3 d 0 h"
