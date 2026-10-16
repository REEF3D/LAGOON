"""NHFLOW's direct forcing solids (ctrl.txt A 581 - A 589): shown as a structure of
the case, before a run built from ctrl.txt, after it from the file NHFLOW writes
(REEF3D_NHFLOW_FORCING_VTP), unless ctrl.txt was changed since."""

import os
import shutil

import numpy as np
from reef3d_fixtures import make_run, write_piece

from lagoonviz.core import Session
from lagoonviz.core.data.forcing import FORCING_VTP, find_forcing, forcing_from_ctrl

CTRL = """A 10 5
A 581 1.0 2.0 1.0 2.0 0.0 1.5
A 583 4.0 0.5 0.5 3.5 0.3
A 584 6.0 2.0 0.0 2.5 0.4
A 585 7.0 1.0 0.0 0.2 7.5 1.5 2.0 0.1
A 586 3.0 3.0 1.0 0.5
A 587 0.0 1.0 3.0 4.0 0.0 1.0
A 588 5.0 6.0 3.0 4.0 1.0 0.0
A 589 2.0 3.0 0.0 1.0 0.0 1.0
P 30 0.5
"""


def test_the_shapes_from_ctrl(tmp_path):
    ctrl = tmp_path / "ctrl.txt"
    ctrl.write_text(CTRL)
    points, triangles, skipped = forcing_from_ctrl(ctrl)
    assert not skipped and len(triangles) > 100
    # the box: its corners there; the sphere's top at 1.5; the cylinder to 2.5
    assert np.isclose(points[:, 2].max(), 2.5)
    near = np.linalg.norm(points - [3.0, 3.0, 1.0], axis=1)
    assert np.isclose(near[near < 0.6].max(), 0.5, atol=1e-5)
    # the jacket member: its ends at the two points
    member = points[(points[:, 0] > 6.7) & (points[:, 0] < 7.8)]
    tilt = np.arctan2(np.hypot(0.5, 0.5), 2.0)  # the end discs lie across the axis
    assert np.isclose(member[:, 2].min(), -0.2 * np.sin(tilt), atol=0.005)
    assert np.isclose(member[:, 2].max(), 2.0 + 0.1 * np.sin(tilt), atol=0.005)
    # every solid closed: each edge in two triangles (the shapes are watertight)
    keys = np.round(points, 5)
    _, ids = np.unique(keys, axis=0, return_inverse=True)
    ids = ids.reshape(-1)[triangles]
    edges = np.sort(np.concatenate([ids[:, [0, 1]], ids[:, [1, 2]], ids[:, [2, 0]]]), axis=1)
    _, counts = np.unique(edges, axis=0, return_counts=True)
    assert (counts == 2).all()


def test_a_case_shows_its_forcing_solids(tmp_path):
    # before a run: from ctrl.txt
    case = tmp_path / "case"
    case.mkdir()
    (case / "ctrl.txt").write_text(CTRL)
    (case / "control.txt").write_text("B 1 0.25\nB 10 0.0 8.0 0.0 4.0 0.0 3.0\nM 10 1\n")
    s = Session()
    s.open_case(case)
    forcing = [i for i in s.scene if i.kind == "structure"]
    assert len(forcing) == 1 and forcing[0].properties["dataset"] == "nhflow_forcing"
    mesh = s.data.load(forcing[0].id)
    assert mesh.n_triangles > 100
    # after a run: shown with the results, from NHFLOW's file
    root = make_run(tmp_path / "run", solver="NHFLOW", frames=2, print_interval=0.5)
    shutil.copy(case / "ctrl.txt", root / "ctrl.txt")
    folder = root / FORCING_VTP.split("/")[0]
    folder.mkdir()
    corners = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], np.float64)
    write_piece(root / FORCING_VTP, "PolyData", corners, {}, np.arange(3), np.array([3]))
    old = os.path.getmtime(root / "ctrl.txt") - 60
    os.utime(root / "ctrl.txt", (old, old))
    assert find_forcing(root) == root / FORCING_VTP
    s = Session()
    s.open_case(root)
    structures = [i for i in s.scene if i.kind == "structure"]
    assert [i.properties["dataset"] for i in structures] == ["nhflow_forcing"]
    assert s.data.load(structures[0].id).n_triangles == 1  # the file NHFLOW wrote
    assert [i for i in s.scene if i.kind == "free_surface"]
    # ctrl.txt changed since: built from it again
    older = os.path.getmtime(root / "ctrl.txt") - 60
    os.utime(root / FORCING_VTP, (older, older))
    assert find_forcing(root) == root / "ctrl.txt"
    # not an NHFLOW case: none
    (root / "ctrl.txt").write_text(CTRL.replace("A 10 5", "A 10 6"))
    assert find_forcing(root) is None
