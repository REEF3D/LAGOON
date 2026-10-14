"""DIVEMesh's grid: from control.txt before it has run, from its preview after; the
grid item; running DIVEMesh (a stand-in program) from a session."""

import os
import stat
import sys
from pathlib import Path

import numpy as np
import pytest
from reef3d_fixtures import write_grid_preview

from lagoonviz.core import Session
from lagoonviz.core.data.divemesh_grid import (
    MAX_LINES,
    grid_lines,
    line_step,
    read_preview,
    uniform_grid,
)
from lagoonviz.core.divemesh_run import LOG, find_divemesh

TESTS = Path(__file__).resolve().parents[1]
CONTROL = "B 1 0.5\nB 10 0.0 10.0 0.0 2.0 0.0 1.0\nM 10 1\n"


def test_a_uniform_grid_from_control_txt():
    grid = uniform_grid((0.0, 10.0, 0.0, 2.0, 0.0, 1.0), (20, 4, 2))
    assert grid.shape == (21, 5, 3) and grid.cells == (20, 4, 2)
    assert grid.spacing() == [(0.5, 0.5), (0.5, 0.5), (0.5, 0.5)]
    bottom = grid_lines(grid, "bottom")
    # 5 lines along x of 20 segments, 21 lines along y of 4 segments
    assert len(bottom) == 5 * 20 + 21 * 4
    assert np.allclose(bottom[..., 2], 0.0)
    back = grid_lines(grid, "back")  # x min and y max
    assert np.allclose(back[(back[..., 0] != 0.0).any(axis=1)][..., 1], 2.0)


def test_a_fine_grid_draws_every_nth_line():
    grid = uniform_grid((0.0, 1.0, 0.0, 1.0, 0.0, 1.0), (2000, 10, 10))
    step = line_step(grid)
    assert step == int(np.ceil(2001 / MAX_LINES))
    lines = grid_lines(grid, "bottom")
    across = lines[lines[:, 0, 1] == lines[:, 1, 1]]  # along x: as many as y lines
    along_y = len(lines) - len(across)
    assert along_y == 10 * (len(range(0, 2001, step)) + (2000 % step != 0))
    assert grid_lines(grid, "bottom", every=1).shape[0] > len(lines)


def test_the_preview_divemesh_writes(tmp_path):
    x = np.concatenate([np.linspace(0, 5, 11), np.linspace(5.25, 10, 20)])  # finer after 5
    write_grid_preview(tmp_path, x, np.linspace(0, 2, 5), np.linspace(0, 1, 3))
    grid = read_preview(tmp_path)
    assert grid.source == "DIVEMesh" and grid.shape == (31, 5, 3)
    (dx_min, dx_max), *_ = grid.spacing()
    assert dx_min == pytest.approx(0.25) and dx_max == pytest.approx(0.5)
    low, high = grid.bounds()
    assert low.tolist() == [0, 0, 0] and high.tolist() == [10, 2, 1]
    assert np.allclose(grid.axis(0), x)


def test_a_case_that_has_not_run_shows_its_grid(tmp_path):
    (tmp_path / "control.txt").write_text(CONTROL)
    session = Session()
    session.open_case(tmp_path)
    grid = next(i for i in session.scene if i.kind == "grid")
    marks = session.data.setup_marks(grid.id)
    (color, lines), *_ = marks.lines
    assert len(lines) == len(grid_lines(session.data.grid(grid.id), "bottom and back"))
    assert session.data.grid(grid.id).source == "control.txt"
    session.set_property(grid.id, "faces", "bottom")
    assert len(session.data.setup_marks(grid.id).lines[0][1]) < len(lines)


@pytest.mark.parametrize("solver, top", [(5, 0.6), (3, 0.6), (6, 1.0)])
def test_sigma_grids_reach_from_the_bed_to_the_still_water(tmp_path, solver, top):
    """NHFLOW and FNPF: REEF3D scales DIVEMesh's heights to σ and stretches them from
    the bed to the still water (F 60): the grid drawn there, as ParaView shows it,
    its levels spaced as DIVEMesh spaced them. CFD: as DIVEMesh made it."""
    (tmp_path / "control.txt").write_text("B 10 0.0 10.0 0.0 2.0 0.0 1.0\nB 1 0.5\nM 10 2\n")
    (tmp_path / "ctrl.txt").write_text(f"A 10 {solver}\nF 60 0.6\n")
    levels = np.array([0.0, 0.1, 0.3, 0.6, 1.0])  # finer at the bed
    write_grid_preview(tmp_path, np.linspace(0, 10, 21), np.linspace(0, 2, 5), levels)
    session = Session()
    session.open_case(tmp_path)
    item = next(i for i in session.scene if i.kind == "grid")
    grid = session.data.grid(item.id)
    low, high = grid.bounds()
    assert low[2] == pytest.approx(0.0) and high[2] == pytest.approx(top)
    np.testing.assert_allclose(grid.axis(2), levels * top, atol=1e-6)
    lines = session.data.setup_marks(item.id).lines[0][1]
    assert lines[..., 2].max() == pytest.approx(top, abs=1e-6)


def test_a_stretched_grid_waits_for_divemesh(tmp_path):
    (tmp_path / "control.txt").write_text(CONTROL + "B 101 1\nB 111 2.0\n")
    session = Session()
    session.open_case(tmp_path)
    grid = next(i for i in session.scene if i.kind == "grid")
    assert session.data.grid(grid.id) is None and session.data.setup_marks(grid.id) is None


FAKE = """\
import sys
sys.path.insert(0, {tests!r})
from pathlib import Path
import numpy as np
from reef3d_fixtures import write_grid_preview
control = Path("control.txt").read_text()
print("read control")
assert "B 1 0.25" in control, control
write_grid_preview(Path("."), np.linspace(0, 10, 41), np.linspace(0, 2, 9), np.linspace(0, 1, 5))
print("--DONE--")
"""


@pytest.fixture
def fake_divemesh(tmp_path, monkeypatch):
    """A stand-in for DIVEMesh: prints, checks control.txt, writes a grid preview."""
    script = tmp_path / "fake_divemesh.py"
    script.write_text(FAKE.format(tests=str(TESTS)))
    program = tmp_path / "DiveMESH"
    program.write_text(f"#!/bin/sh\nexec {sys.executable} {script}\n")
    program.chmod(program.stat().st_mode | stat.S_IEXEC)
    monkeypatch.setenv("LAGOON_DIVEMESH", str(program))
    return program


@pytest.mark.skipif(os.name == "nt", reason="the stand-in is a shell script")
def test_running_divemesh_from_a_session(tmp_path, fake_divemesh):
    case = tmp_path / "case"
    case.mkdir()
    (case / "control.txt").write_text(CONTROL)
    assert find_divemesh() == fake_divemesh
    session = Session()
    case_id = session.open_case(case)
    setup = session.data.case_setup(case_id)
    setup.control.set("B 1", [0.25])  # an edit not saved yet: written before the run
    lines = []
    assert session.run_divemesh(case_id, on_line=lines.append) == 0
    assert lines == ["read control", "--DONE--"]
    assert "--DONE--" in (case / LOG).read_text()
    assert "B 1 0.25" in (case / "control.txt").read_text()
    grid = next(i for i in session.scene if i.kind == "grid")
    assert session.data.grid(grid.id).source == "DIVEMesh"
    assert session.data.grid(grid.id).cells == (40, 8, 4)


def test_divemesh_not_found(tmp_path, monkeypatch):
    monkeypatch.setenv("LAGOON_DIVEMESH", str(tmp_path / "nothing"))
    monkeypatch.setenv("PATH", str(tmp_path))
    monkeypatch.setenv("HOME", str(tmp_path))
    assert find_divemesh() is None
    (tmp_path / "control.txt").write_text(CONTROL)
    session = Session()
    session.open_case(tmp_path)
    with pytest.raises(FileNotFoundError, match="LAGOON_DIVEMESH"):
        session.run_divemesh()
