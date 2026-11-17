# LAGOON — CFD's porous structures (VRANS, ctrl.txt B 270 - B 291), layer by layer
# Architect: Hans Bihs
"""REEF3D::CFD gives each porous structure its own material, the later one holding
where they overlap (the boxes first, then the cylinders, wedges and plates): a
rubble mound's armour, filter and core. LAGOON draws what is left of each as REEF3D
fills its cells, coloured by its material, with its outlines where one layer meets
the next."""

from __future__ import annotations

import numpy as np
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
