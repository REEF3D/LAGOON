"""The input check before a run: both input files, DIVEMesh's grid, the cost."""

import json
import os
import time

from lagoonviz.core import Session
from lagoonviz.core.input_check import check_case


def write_inputs(root, ranks_ctrl=2, ranks_control=2, grids=2, ctrl_extra="P 30 0.1\nN 41 5.0\n"):
    root.mkdir(parents=True, exist_ok=True)
    (root / "control.txt").write_text(
        f"C 11 1\nB 1 0.05\nB 10 0.0 4.0 0.0 1.0 0.0 2.0\nM 10 {ranks_control}\n"
    )
    (root / "ctrl.txt").write_text(f"A 10 6\nM 10 {ranks_ctrl}\n{ctrl_extra}")
    folder = root / "DIVEMesh_Grid"
    folder.mkdir(exist_ok=True)
    (folder / "grid-geometry.dat").write_text("")
    old = time.time() - 60
    os.utime(root / "control.txt", (old - 60, old - 60))
    for rank in range(1, grids + 1):
        path = folder / f"grid-{rank:06d}.dat"
        path.write_text("")
        os.utime(path, (old, old))
    return root


def levels(report):
    return [(p.level, p.where) for p in report.problems]


def test_a_good_case_and_its_cost(tmp_path):
    report = check_case(write_inputs(tmp_path / "case"))
    assert report.ok and report.solver == "CFD"
    assert [p.level for p in report.problems] == ["note"]  # the LAGOON store is off
    facts = dict(report.facts)
    assert facts["Grid"] == "80 x 20 x 40 = 64,000 cells"
    assert facts["Per process"] == "32,000 cells on each of 2"


def test_processes_and_grid_files(tmp_path):
    report = check_case(write_inputs(tmp_path / "a", ranks_ctrl=4, ranks_control=2))
    assert ("error", "M 10") in levels(report)
    report = check_case(write_inputs(tmp_path / "b", ranks_ctrl=3, ranks_control=3, grids=2))
    assert any("no grid-000003.dat" in p.message for p in report.errors)
    report = check_case(write_inputs(tmp_path / "c", ranks_ctrl=1, ranks_control=1, grids=2))
    assert any("beyond M 10 1" in p.message for p in report.problems)
    root = write_inputs(tmp_path / "d")
    (root / "control.txt").write_text((root / "control.txt").read_text())  # touched now
    report = check_case(root)
    assert any(
        p.where == "control.txt" and "run DIVEMesh again" in p.message for p in report.errors
    )
    root = write_inputs(tmp_path / "e")
    for path in (root / "DIVEMesh_Grid").glob("grid-0*.dat"):
        path.unlink()
    assert any("run DIVEMesh first" in p.message for p in check_case(root).errors)


def test_no_output_no_end_and_unknown_commands(tmp_path):
    report = check_case(write_inputs(tmp_path / "case", ctrl_extra="Q 9999 1\n"))
    messages = " | ".join(p.message for p in report.problems)
    assert "no 3D output" in messages and "no end" in messages
    assert "Q 9999 is not in LAGOON's list of REEF3D commands" in messages
    missing = tmp_path / "empty"
    missing.mkdir()
    assert len(check_case(missing).errors) == 2  # neither file


def test_the_time_from_the_last_run(tmp_path):
    root = write_inputs(tmp_path / "case")
    logs = root / "REEF3D_Log"
    logs.mkdir()
    lines = ["REEF3D version:  x", "", "number of cells:  64000", ""]
    lines += [f"{k} \t 0.01 \t {0.01 * k:.5f} \t 0.5 \t 10" for k in range(1, 300)]
    (logs / "REEF3D_mainlog.dat").write_text("\n".join(lines) + "\n")
    runs = root / "REEF3D_Case"
    runs.mkdir()
    record = {"type": "run", "run": "r1", "solver": "CFD", "started": "2026-10-17T10:00:00Z",
              "ranks": 2}  # fmt: skip
    (runs / "REEF3D_CFD_run.jsonl").write_text(json.dumps(record) + "\n")
    facts = dict(check_case(root).facts)
    assert facts["Time per iteration"].startswith("about 0.5 s")
    assert facts["Run time"].startswith("about 4 min")  # 5 s / 0.01 s x 0.5 s = 250 s


def test_from_the_session(tmp_path):
    root = write_inputs(tmp_path / "case")
    s = Session()
    s.open_case(root)
    assert s.check_input().ok
