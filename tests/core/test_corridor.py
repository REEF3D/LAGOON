"""A river corridor: banks shaped by dragged points (a spline through them or a
B-spline), the curvilinear grid between them, curvgrid.dat and the set-up of a case
for REEF3D's G 50 2."""

import numpy as np
import pytest

from lagoonviz.core import Session
from lagoonviz.core import corridor as C

CONTROL = "C 11 1\nC 14 2\nB 1 1.0\nB 10 0.0 200.0 0.0 60.0 0.0 1.0\nM 10 1\n"
CTRL = "A 10 2\nF 60 2.0\n"


@pytest.fixture
def river(tmp_path):
    case = tmp_path / "river"
    case.mkdir()
    (case / "control.txt").write_text(CONTROL)
    (case / "ctrl.txt").write_text(CTRL)
    return case


def bend(cells_along=60, cells_across=12, **options):
    """A 90° bend: the outer (left) bank at r = 100 m, the inner at 60 m, flowing
    clockwise from (0, r) to (r, 0)."""
    a = np.radians(np.linspace(90, 0, 12))
    left = np.column_stack([100 * np.cos(a), 100 * np.sin(a)])
    right = np.column_stack([60 * np.cos(a), 60 * np.sin(a)])
    return C.Corridor(left, right, cells_along=cells_along, cells_across=cells_across, **options)


def test_the_curves_of_the_banks():
    p = np.array([[0, 0], [1, 2], [3, 3], [6, 1], [8, 4]], float)
    through = C.catmull_rom(p)
    for q in p:  # through every point
        assert np.linalg.norm(through - q, axis=1).min() < 1e-9
    smooth = C.bspline(p)
    assert np.allclose(smooth[0], p[0]) and np.allclose(smooth[-1], p[-1])
    assert np.linalg.norm(smooth - p[2], axis=1).min() > 0.1  # pulled, not passed
    # centripetal: no loop where two points nearly meet
    close = np.array([[0, 0], [10, 0], [10.01, 0.01], [20, 0]], float)
    line = C.catmull_rom(close)
    assert line[:, 0].max() <= 20 + 1e-9 and line[:, 0].min() >= -1e-9
    assert np.allclose(C.resample(np.array([[0, 0], [10, 0.0]]), 3), [[0, 0], [5, 0], [10, 0]])


def test_a_straight_corridor_and_its_grid():
    straight = C.default((0, 100, 0, 20))
    assert len(straight.left) == 12 and straight.left[0, 1] == 20 and straight.right[0, 1] == 0
    x, y = straight.grid()
    assert x.shape == (21, 101)  # cells across 20, along 100 (cells about square)
    q = C.quality(x, y)
    assert q.folded == 0 and q.min_angle == pytest.approx(90) and q.along == pytest.approx((1, 1))
    across = C.default((0, 10, 0, 50))  # flows along y: the left bank at x0
    assert across.left[0, 0] == 0 and across.left[-1, 1] == 50


def test_a_bend_does_not_fold_and_smoothing_evens_it():
    x, y = bend().grid()
    assert C.quality(x, y).folded == 0 and (C.cell_areas(x, y) > 0).all()
    # the banks the other way round: j from the left bank, still turning counter-clockwise
    swapped = bend()
    swapped = C.Corridor(swapped.right, swapped.left, cells_along=60, cells_across=12)
    assert (C.cell_areas(*swapped.grid()) > 0).all()
    smooth = bend(smoothing=200)
    xs, ys = smooth.grid()
    assert C.quality(xs, ys).folded == 0
    np.testing.assert_allclose(xs[0], x[0])  # the banks stay where they are
    np.testing.assert_allclose(ys[-1], y[-1])


def test_points_moved_resampled_and_reversed():
    corridor = bend()
    moved = corridor.moved("left", 5, 0.0, 120.0)
    assert tuple(moved.left[5]) == (0.0, 120.0) and tuple(corridor.left[5]) != (0.0, 120.0)
    more = corridor.with_points(24)
    assert len(more.left) == len(more.right) == 24
    np.testing.assert_allclose(np.hypot(*more.left.T), 100.0, rtol=2e-3)  # still the bend
    assert len(corridor.with_points(1000).left) == C.MAX_POINTS
    back = corridor.reversed()
    np.testing.assert_allclose(back.left[0], corridor.right[-1])
    with pytest.raises(ValueError, match="curve"):
        C.Corridor(corridor.left, corridor.right, curve="nurbs")
    with pytest.raises(ValueError, match="two points"):
        C.Corridor(corridor.left[:1], corridor.right)
    again = C.Corridor.from_dict(corridor.to_dict() | {"note": "kept"})
    assert again.extra == {"note": "kept"} and np.allclose(again.left, corridor.left)


def test_the_bed_and_curvgrid_dat(tmp_path):
    corridor = C.default((0, 100, 0, 20), z_start=-2.0, slope=0.01)
    x, y = corridor.grid()
    z = corridor.bed_heights(x, y)
    assert z[0, 0] == pytest.approx(-2.0) and z[5, -1] == pytest.approx(-3.0)
    points = np.array([[0, 0, 1.0], [200, 0, 1.0], [0, 100, 1.0], [200, 100, 1.0]])
    geo = C.Corridor(corridor.left, corridor.right, bed="geo").bed_heights(x, y, points)
    assert np.allclose(geo, 1.0)
    path = tmp_path / C.GRID_FILE
    path.write_text(C.grid_text(x, y, z))
    lines = path.read_text().splitlines()
    assert lines[0] == "100 20" and len(lines) == 1 + 101 * 21
    assert [float(v) for v in lines[2].split()] == pytest.approx([1.0, 0.0, -2.01])  # i fastest
    rx, ry, rz = C.read_grid(path)
    np.testing.assert_allclose(rx, x) and np.testing.assert_allclose(rz, z)
    assert C.read_grid(tmp_path / "none.dat") is None
    assert C.nice_step(0.83) == 1.0 and C.nice_step(2.46) == 2.0 and C.nice_step(3.2) == 5.0
    assert C.logical_box(60, 12, 2.0) == (0.0, 120.0, 0.0, 24.0)


def test_a_case_gets_a_corridor_and_its_grid(river):
    session = Session()
    case_id = session.open_case(river)
    session.start_corridor()
    corridor = session.corridor()
    assert (river / "corridor.json").is_file() and len(corridor.left) == 12
    assert corridor.left[0, 1] == 54.0 and corridor.right[0, 1] == 6.0 and corridor.z_start == 0.0
    session.move_corridor_point("left", 5, 90.0, 58.0)
    session.set_corridor_options(points=16, curve="bspline", smoothing=50, slope=0.001)
    assert len(session.corridor().left) == 16 and session.corridor().curve == "bspline"
    # the set-up draws the corridor: its banks, its points, the grid it makes
    marks = [m for m in session.data.case_setup(case_id).marks() if m.group == "corridor"]
    assert len(marks) == 2 and len(marks[0].points) == 32
    quality = session.make_corridor_grid()
    assert quality.folded == 0
    ni, nj = quality.cells
    assert (river / "curvgrid.dat").read_text().split("\n")[0] == f"{ni} {nj}"
    control = session.input_file("control.txt")
    step = control.value("B 1")
    assert control.values("B 10") == [0.0, ni * step, 0.0, nj * step, 0.0, 1.0]
    assert "G 50 2\n" in session.input_file("ctrl.txt").text()  # an integer
    setup = session.data.case_setup(case_id)
    assert setup.curvgrid is not None and setup.curvgrid[0].shape == (nj + 1, ni + 1)
    # the domain is where the nodes are, not DIVEMesh's logical box
    assert setup.domain[3] > 50 and setup.domain[2] == pytest.approx(6.0)
    groups = [m.group for m in setup.marks()]
    assert "curvgrid" in groups and session.data.grid(case_id) is None
    script = session.to_script()
    assert "session.start_corridor([0.0, 200.0, 6.0, 54.0], points=12)" in script
    assert "session.move_corridor_point('left', 5, 90.0, 58.0)" in script
    assert "session.make_corridor_grid()" in script
    session.undo()  # the grid, G 50 and the logical box: one step
    assert not (river / "curvgrid.dat").exists()
    assert session.input_file("ctrl.txt").text() == CTRL
    assert session.input_file("control.txt").text() == CONTROL
    assert session.data.case_setup(case_id).curvgrid is None
    for _ in range(3):
        session.undo()
    assert session.corridor() is None and not (river / "corridor.json").exists()
    session.redo()
    session.reverse_corridor(case=case_id)
    assert session.to_script().splitlines()[-1] == f"session.reverse_corridor(case={case_id!r})"


def test_a_folding_grid_is_refused(river):
    session = Session()
    session.open_case(river)
    session.start_corridor((0, 100, 0, 20), points=4)
    session.move_corridor_point("right", 2, 66.0, 30.0)  # the right bank over the left
    with pytest.raises(ValueError, match="fold"):
        session.make_corridor_grid()
    (river / "corridor.json").unlink()
    other = Session()
    other.open_case(river)
    with pytest.raises(ValueError, match="no river corridor"):
        other.make_corridor_grid()
