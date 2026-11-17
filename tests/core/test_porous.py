# LAGOON — CFD's porous structures (VRANS, ctrl.txt B 270 - B 291), layer by layer
# Architect: Hans Bihs
"""REEF3D::CFD gives each porous structure its own material, the later one holding
where they overlap (the boxes first, then the cylinders, wedges and plates): a
rubble mound's armour, filter and core. LAGOON draws what is left of each as REEF3D
fills its cells, coloured by its material, with its outlines where one layer meets
the next."""

from __future__ import annotations

import numpy as np
import pytest
from reef3d_fixtures import make_run

from lagoonviz.core import Session
from lagoonviz.core.data.porous import find_porous, porous_mesh, read_porous

# a rubble mound in a flume (y 0 - 1): armour, filter, core, each a crest box and
# two slope wedges; the boxes come first in REEF3D's order, whatever the file's
MOUND = """A 10 6
B 281 7.0 8.375 0.0 1.0 0.0 1.1     0.45 0.05 500 2
B 270 8.375 8.575 0.0 1.0 0.0 1.1   0.45 0.05 500 2
B 281 8.575 9.95 0.0 1.0 1.1 0.0    0.45 0.05 500 2
B 270 8.415 8.535 0.0 1.0 0.0 1.02  0.40 0.02 500 2
B 281 7.11 8.415 0.0 1.0 0.0 1.02   0.40 0.02 500 2
B 281 8.535 9.84 0.0 1.0 1.02 0.0   0.40 0.02 500 2
B 270 8.435 8.515 0.0 1.0 0.0 0.98  0.35 0.005 500 2 // core
B 281 7.18 8.435 0.0 1.0 0.0 0.98   0.35 0.005 500 2
B 281 8.515 9.77 0.0 1.0 0.98 0.0   0.35 0.005 500 2
"""


def _front(mesh, y=0.0):
    """The triangles on the side y = 0 (x, z), the layer of each, the outlines there."""
    p = mesh.points.astype(np.float64)
    t = p[mesh.triangles.astype(np.int64)]
    front = (np.abs(t[..., 1] - y) < 1e-6).all(axis=1)
    layer = mesh.point_data["layer"][mesh.triangles[front, 0]]
    lines = p[mesh.outlines.astype(np.int64)]
    return t[front][..., [0, 2]], layer, lines[(np.abs(lines[..., 1] - y) < 1e-6).all(axis=1)]


def _area(t):
    a, b, c = t[:, 0], t[:, 1], t[:, 2]
    cross = (b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1]) - (c[:, 0] - a[:, 0]) * (b[:, 1] - a[:, 1])
    return 0.5 * np.abs(cross)


def test_the_structures_in_reef3ds_order(tmp_path):
    ctrl = tmp_path / "ctrl.txt"
    ctrl.write_text(MOUND)
    found = read_porous(ctrl)
    assert [s.code for s in found] == [270] * 3 + [281] * 6  # boxes first
    assert found[2].material == (0.35, 0.005, 500.0, 2.0) and found[2].shape[-1] == 0.98
    assert find_porous(tmp_path) == ctrl
    ctrl.write_text(MOUND.replace("A 10 6", "A 10 5"))  # NHFLOW: B 201 and its own file
    assert read_porous(ctrl) == [] and find_porous(tmp_path) is None
    ctrl.write_text("B 270 0 1 0 1 0 1 0.4\n")  # without its material: not drawn
    assert read_porous(ctrl) == []


def test_the_layers_side_by_side(tmp_path):
    """The flume's side shows the three layers as bands, each once (no layer drawn
    inside another); the outlines where they meet and round them, not where a
    layer's crest box meets its slope wedges."""
    ctrl = tmp_path / "ctrl.txt"
    ctrl.write_text(MOUND)
    mesh = porous_mesh(read_porous(ctrl))
    triangles, layer, lines = _front(mesh)
    whole = 0.5 * (9.95 - 7.0 + 0.2) * 1.1  # the armour's outline: the mound
    filter_ = 0.5 * (9.84 - 7.11 + 0.12) * 1.02
    core = 0.5 * (9.77 - 7.18 + 0.08) * 0.98
    assert np.isclose(_area(triangles).sum(), whole, rtol=1e-5)  # all of it, once
    np.testing.assert_allclose(
        [_area(triangles[layer == n]).sum() for n in (1, 2, 3)],
        [whole - filter_, filter_ - core, core],
        rtol=1e-5,
    )
    assert set(np.unique(mesh.point_data["d50"])) == {
        np.float32(0.05),
        np.float32(0.02),
        np.float32(0.005),
    }
    # outlines: the three trapezoids' tops and the bed line, no vertical seams
    flat = lines[..., [0, 2]]
    upright = np.abs(flat[:, 0, 0] - flat[:, 1, 0]) < 1e-9
    assert not upright.any()
    length = np.linalg.norm(flat[:, 1] - flat[:, 0], axis=1).sum()
    slopes = [
        np.hypot(8.375 - 7.0, 1.1) + 0.2 + np.hypot(9.95 - 8.575, 1.1),
        np.hypot(8.415 - 7.11, 1.02) + 0.12 + np.hypot(9.84 - 8.535, 1.02),
        np.hypot(8.435 - 7.18, 0.98) + 0.08 + np.hypot(9.77 - 8.515, 0.98),
    ]
    assert np.isclose(length, sum(slopes) + (9.95 - 7.0), rtol=1e-5)


def test_a_later_structure_holds_where_they_overlap(tmp_path):
    """Two boxes of one material side by side are one layer (no seam); a box of
    another material inside them takes its place; a cylinder after the boxes too."""
    ctrl = tmp_path / "ctrl.txt"
    ctrl.write_text(
        "B 270 0 4 0 1 0 2   0.4 0.05 500 2\n"
        "B 270 4 6 0 1 0 2   0.4 0.05 500 2\n"
        "B 270 1 3 0 1 0 1   0.3 0.01 500 2\n"
        "B 274 5 0.5 0 3 0.3 0.5 0.1 500 2\n"
    )
    mesh = porous_mesh(read_porous(ctrl))
    triangles, layer, lines = _front(mesh)
    np.testing.assert_allclose(
        [_area(triangles[layer == n]).sum() for n in (1, 2)], [12.0 - 2.0, 2.0], rtol=1e-5
    )
    flat = lines[..., [0, 2]]
    assert not any(np.allclose(seg[:, 0], 4.0) for seg in flat)  # no seam at x = 4
    assert any(np.allclose(seg[:, 0], 1.0) for seg in flat)  # the inner box's side
    p = mesh.points[mesh.triangles.ravel()]
    cylinder = mesh.point_data["layer"][mesh.triangles.ravel()] == 3
    assert cylinder.any() and np.ptp(p[cylinder][:, 2]) > 2.9  # whole: the later one
    radius = np.hypot(p[cylinder][:, 0] - 5.0, p[cylinder][:, 1] - 0.5)
    assert radius.max() <= 0.3 + 1e-5
    # the boxes' top with a hole where the cylinder comes through
    t = mesh.points[mesh.triangles.astype(np.int64)].astype(np.float64)
    top = (np.abs(t[..., 2] - 2.0) < 1e-6).all(axis=1)
    top &= mesh.point_data["layer"][mesh.triangles[:, 0]] == 1
    hole = 0.5 * 48 * 0.3**2 * np.sin(2 * np.pi / 48)
    assert np.isclose(_area(t[top][..., :2]).sum(), 6.0 - hole, rtol=1e-5)


def test_the_porous_structures_are_shown_with_the_run(tmp_path):
    run = make_run(tmp_path / "cfd", solver="CFD", frames=2, print_interval=0.5)
    with open(run / "ctrl.txt", "a") as ctrl:
        ctrl.write(
            MOUND.replace("A 10 6\n", "")
            .replace("7.", "2.")
            .replace("8.", "3.")
            .replace("9.", "4.")
        )
    s = Session()
    s.open_case(run)
    item = next(i for i in s.scene if i.kind == "structure" and i.name == "Porous structures")
    assert item.visible and item.properties["variable"] == "d50"
    assert item.properties["grid_lines"] == "dark" and item.properties["opacity"] < 1.0
    mesh = s.data.load(item.id)
    assert mesh.n_triangles and set(mesh.point_data) >= {"porosity", "d50", "layer"}
    shown = s.data.display(item.id, mesh)
    lines = s.data.patch_outlines(item.id, mesh, shown)
    assert lines is not None and lines.facing and len(lines.segments) == len(mesh.outlines)
    # at every time: one frame
    assert s.data.dataset(item.id).label == "Porous structures (ctrl.txt)"
    s.set_time(0.5)
    assert s.data.load(item.id).n_triangles == mesh.n_triangles


# ======================================================================== NHFLOW
# Yue's breakwater in a 2D flume (TEST Case19): one body of a crest box and two
# wedges, its armour and filter as layers below the exposed surface, the core inside
YUE_CTRL = """A 10 5
B 200 1
B 201 0.414 0.0028 50.0 2.0 // core
B 202 0.075 0.3  0.0596 100.0 1.1// armour layer
B 202 0.045 0.33 0.02   600.0 1.1 // filter layer
B 209 1.1
B 210 8.46 8.54 0.0 0.005 0.0 1.1
B 217 7.0 8.46 0.0 0.005 0.0 1.1
B 217 8.54 10.0 0.0 0.005 1.1 0.0
F 60 0.7
"""
YUE_CONTROL = "B 2 2400 1 20\nB 10 0.0 12.0 0.0 0.005 0.0 1.4\n"


def _yue(tmp_path, ctrl=YUE_CTRL, control=YUE_CONTROL):
    (tmp_path / "ctrl.txt").write_text(ctrl)
    (tmp_path / "control.txt").write_text(control)
    return tmp_path / "ctrl.txt"


def _inner(t: float) -> float:
    """The area of the mound's part deeper than t below its slopes and crest (the
    bed exposes nothing): the trapezoid of the lines t inside them."""
    import math

    sine = math.sin(math.atan2(1.1, 1.46))
    run = 1.46 / 1.1
    width = 3.0 - 2.0 * t / sine
    top = min(1.1 - t, width / (2.0 * run))
    return width * top - run * top * top


def test_nhflow_layers_below_the_surface(tmp_path):
    from lagoonviz.core.data.porous import porous_from_ctrl, read_nhflow_porous

    ctrl = _yue(tmp_path)
    body = read_nhflow_porous(ctrl)
    assert [code for code, _ in body.shapes] == [210, 217, 217]
    assert body.depths() == [0.075, pytest.approx(0.12)]
    assert body.materials()[0] == (0.3, 0.0596, 1.0) and body.materials()[-1][2] == 3.0
    assert find_porous(tmp_path) == ctrl
    mesh = porous_from_ctrl(ctrl)
    triangles, layer, lines = _front(mesh)
    whole = 0.5 * (3.0 + 0.08) * 1.1
    assert np.isclose(_area(triangles).sum(), whole, rtol=1e-3)
    np.testing.assert_allclose(
        [_area(triangles[layer == n]).sum() for n in (1, 2, 3)],
        [whole - _inner(0.075), _inner(0.075) - _inner(0.12), _inner(0.12)],
        rtol=3e-3,  # (drawn on a grid of a sixth of the filter's thickness)
    )
    # the other end the same, the slopes and crest armour
    back, _, _ = _front(mesh, 0.005)
    assert np.isclose(_area(back).sum(), whole, rtol=1e-3)
    t = mesh.points[mesh.triangles.astype(np.int64)]
    sides = ~(
        (np.abs(t[..., 1]) < 1e-7).all(axis=1) | (np.abs(t[..., 1] - 0.005) < 1e-7).all(axis=1)
    )
    up = np.cross(t[:, 1] - t[:, 0], t[:, 2] - t[:, 0])[:, 2] > 0
    assert set(mesh.point_data["layer"][mesh.triangles[sides & up, 0]]) == {1.0}
    # outlines on an end: the outline, two layer boundaries (no seam at the crest box)
    flat = lines[..., [0, 2]]
    upright = np.abs(flat[:, 0, 0] - flat[:, 1, 0]) < 1e-9
    assert not (upright & (np.abs(flat[:, 0, 1] - flat[:, 1, 1]) > 0.01)).any()
    length = np.linalg.norm(flat[:, 1] - flat[:, 0], axis=1).sum()
    outline = 3.0 + 0.08 + 2 * np.hypot(1.46, 1.1)

    def boundary(t):  # the line t below the slopes and the crest, bed to bed
        sine = np.sin(np.arctan2(1.1, 1.46))
        width = 3.0 - 2.0 * t / sine
        top = min(1.1 - t, width / (2.0 * 1.46 / 1.1))
        return 2.0 * top / sine + max(width - 2.0 * top * 1.46 / 1.1, 0.0)

    assert np.isclose(length, outline + boundary(0.075) + boundary(0.12), rtol=5e-3)


def test_nhflow_what_exposes_the_body(tmp_path):
    """B 203 0: the bed faces expose too (the core off the bed); no B 202: the body
    the core's; a body that is not the same at every y: its outside alone."""
    from lagoonviz.core.data.porous import porous_from_ctrl

    mesh = porous_from_ctrl(_yue(tmp_path, YUE_CTRL + "B 203 0\n"))
    triangles, layer, _ = _front(mesh)
    low = triangles[layer == 3][..., 1].min()
    assert abs(low - 0.12) < 0.01  # the core 0.12 above the bed
    plain = "\n".join(line for line in YUE_CTRL.splitlines() if not line.startswith("B 202"))
    mesh = porous_from_ctrl(_yue(tmp_path, plain))
    assert set(mesh.point_data["layer"]) == {1.0} and np.allclose(
        mesh.point_data["porosity"], 0.414
    )
    # a 3D roundhead: a vertical cylinder, its outside armour
    head = YUE_CTRL.split("B 210")[0] + "B 213 5.0 1.0 0.0 1.0 0.5\n"
    control = "B 2 100 40 20\nB 10 0.0 10.0 0.0 2.0 0.0 1.4\n"
    mesh = porous_from_ctrl(_yue(tmp_path, head, control))
    assert mesh.n_triangles and np.ptp(mesh.points[:, 1]) > 0.9
    t = mesh.points[mesh.triangles.astype(np.int64)]
    up = np.cross(t[:, 1] - t[:, 0], t[:, 2] - t[:, 0])
    side = np.abs(up[:, 2]) < 1e-6 * np.linalg.norm(up, axis=1)
    assert set(mesh.point_data["layer"][mesh.triangles[side, 0]]) == {1.0}


def test_nhflow_porous_with_the_run(tmp_path):
    """Shown as the porous structures, NHFLOW's own file of the body (no layers) not
    besides."""
    from reef3d_fixtures import BOX, write_piece

    run = make_run(tmp_path / "nhflow", solver="NHFLOW", frames=2, print_interval=0.5)
    (run / "ctrl.txt").write_text((run / "ctrl.txt").read_text() + YUE_CTRL)
    (run / "control.txt").write_text(YUE_CONTROL)
    folder = run / "REEF3D_NHFLOW_VRANS_VTP"
    folder.mkdir()
    write_piece(
        folder / "REEF3D-NHFLOW-VRANS.vtp", "PolyData", BOX, {}, np.arange(4), np.array([4])
    )
    s = Session()
    s.open_case(run)
    structures = [i for i in s.scene if i.kind == "structure"]
    assert [i.name for i in structures] == ["Porous structures"]
    assert s.data.dataset(structures[0].id).key == "nhflow_porous"
    mesh = s.data.load(structures[0].id)
    assert set(np.unique(mesh.point_data["layer"])) == {1.0, 2.0, 3.0}
