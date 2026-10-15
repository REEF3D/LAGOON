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
    marks = session.data.case_setup(case_id).marks()
    banks = [m for m in marks if m.group == "corridor"]
    assert len(banks) == 1 and len(banks[0].points) == 32
    assert [m for m in marks if m.group == "corridor_grid"]
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
    # one grid: the corridor's while there is one (not curvgrid.dat's as well)
    assert "corridor_grid" in groups and "curvgrid" not in groups
    assert session.data.grid(case_id) is None
    script = session.to_script()
    assert "session.start_corridor(points=12)" in script
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


def valley():
    """A valley 400 x 200 m: a channel 20 m wide meandering about y = 100 + 40 sin(2π x
    / 250), its bed 1 m below 0 in the middle, the floodplain at 1 m and above."""
    x, y = np.linspace(0, 400, 201), np.linspace(0, 200, 101)
    gx, gy = np.meshgrid(x, y)
    centre = 100 + 40 * np.sin(2 * np.pi * gx / 250)
    d = np.abs(gy - centre)
    z = np.where(d < 10, -1.0 + 0.5 * (d / 10) ** 2, 1.0 + 0.005 * (d - 10)) - 0.002 * gx
    return np.column_stack([gx.ravel(), gy.ravel(), z.ravel()])


def test_the_thalweg_and_a_corridor_along_it():
    domain = (0, 400, 0, 200, -2, 3)
    thalweg = C.find_thalweg(valley(), domain, level=0.0)
    centre = 100 + 40 * np.sin(2 * np.pi * thalweg.line[:, 0] / 250)
    assert np.abs(thalweg.line[:, 1] - centre).max() < 3.0  # in the channel all along
    assert thalweg.line[0, 0] < 2 and thalweg.line[-1, 0] > 398  # from x min to x max
    assert 7 < np.median(thalweg.left) < 12 and 7 < np.median(thalweg.right) < 12
    corridor = C.from_thalweg(thalweg, margin=5.0, count=16, smoothing=50)
    assert len(corridor.left) == 16 and corridor.bed == "geo"
    offset = corridor.left[:, 1] - (100 + 40 * np.sin(2 * np.pi * corridor.left[:, 0] / 250))
    assert (offset > 8).all() and (offset < 25).all()  # the water's edge and 5 m more
    x, y = corridor.grid()
    assert C.quality(x, y).folded == 0
    wide = C.from_thalweg(thalweg, width=60.0)
    across = np.linalg.norm(wide.bank_line("left") - wide.right[0], axis=1).min()
    assert across == pytest.approx(60, rel=0.05)


def test_a_case_gets_a_corridor_along_its_thalweg(river):
    (river / "control.txt").write_text(
        "C 11 1\nC 14 2\nB 1 2.0\nB 10 0.0 400.0 0.0 200.0 0.0 1.0\nG 10 1\nM 10 1\n"
    )
    (river / "ctrl.txt").write_text("A 10 2\nF 60 0.0\n")
    session = Session()
    session.open_case(river)
    with pytest.raises(ValueError, match="geo.dat"):
        session.auto_corridor()
    np.savetxt(river / "geo.dat", valley(), fmt="%.4f")
    session.input_file("control.txt")  # the set-up reads geo.dat again
    session.data.case_setup(session._case_id(None)).geo_changed()
    thalweg = session.auto_corridor(margin=5.0, points=14)
    assert thalweg.level == 0.0  # F 60
    corridor = session.corridor()
    assert len(corridor.left) == 14 and corridor.left[0, 0] < 5
    assert "session.auto_corridor(margin=5.0, points=14)" in (session.to_script())
    session.move_corridor_point("left", 3, *corridor.left[3] + 2.0)  # then dragged
    assert session.make_corridor_grid().folded == 0


def test_the_corridor_is_an_item_added_removed_and_reset(river):
    session = Session()
    case_id = session.open_case(river)
    item = session.add_item("corridor")  # Add -> River corridor: laid out at once
    assert session.scene.get(item).kind == "corridor" and session.corridor() is not None
    assert session.add_item("corridor") == item  # one per case
    # it draws the banks, their points and its grid; the Case set-up item does not
    marks = session.data.setup_marks(item)
    assert marks is not None and len(marks.points) and len(marks.lines) == 2
    setup_item = next(i.id for i in session.scene if i.kind == "setup")
    colours = [c for c, _ in session.data.setup_marks(setup_item).lines]
    assert C.CURVES and "#ffd84a" not in colours
    session.set_property(item, "show_grid", False)
    session.set_property(item, "show_points", False)
    marks = session.data.setup_marks(item)
    assert len(marks.lines) == 1 and not len(marks.points)
    # reset: as it was made, the dragged points dropped, the grid's settings kept
    start = session.corridor().left.copy()
    session.move_corridor_point("left", 4, 70.0, 59.0)
    session.set_corridor_options(cells_across=10)
    session.reset_corridor()
    np.testing.assert_allclose(session.corridor().left, start)
    assert session.corridor().cells_across == 10
    assert session.to_script().splitlines()[-1] == "session.reset_corridor()"
    # removed: the item and corridor.json, one undo step
    session.remove_item(item)
    assert not (river / "corridor.json").exists() and session.corridor() is None
    assert not [i for i in session.scene if i.kind == "corridor"]
    session.undo()
    assert (river / "corridor.json").exists() and item in session.scene
    # the corridor of a case opened again is its item
    again = Session()
    again.open_case(river)
    assert [i for i in again.scene if i.kind == "corridor"]
    # the River tab's Remove: the item goes with it
    again.remove_corridor()
    assert not [i for i in again.scene if i.kind == "corridor"]
    again.undo()
    assert [i for i in again.scene if i.kind == "corridor"]
    assert case_id


def test_a_corridor_drawn_over_an_area_resets_to_that_area(river):
    session = Session()
    session.open_case(river)
    session.start_corridor((20, 180, 10, 50), points=8)
    assert [i for i in session.scene if i.kind == "corridor"]
    first = session.corridor().right.copy()
    session.move_corridor_point("right", 2, 60.0, 0.0)
    session.reset_corridor()
    np.testing.assert_allclose(session.corridor().right, first)
    assert session.corridor().extra["made"] == {
        "how": "area",
        "points": 8,
        "area": [20.0, 180.0, 10.0, 50.0],
    }


def test_the_bed_of_many_points_is_quick():
    """geo.dat on a regular grid (as LAGOON writes it) is read bilinear, very many
    scattered points by their nearest neighbours: a million points take a second,
    not the half minute and more of triangulating them."""
    import time

    x, y = np.meshgrid(np.linspace(0, 30000, 1500), np.linspace(0, 15000, 750))
    grid = np.column_stack([x.ravel(), y.ravel(), np.sin(x.ravel() / 1000)])
    corridor = C.default((5000, 25000, 6000, 9000))
    gx, gy = corridor.grid()
    start = time.perf_counter()
    z = C.interpolate_bed(grid, gx, gy)
    assert time.perf_counter() - start < 5.0
    assert np.abs(z - np.sin(gx / 1000)).max() < 1e-3
    rng = np.random.default_rng(0)
    scattered = rng.uniform((0, 0), (30000, 15000), (400_000, 2))
    scattered = np.column_stack([scattered, np.sin(scattered[:, 0] / 1000)])
    start = time.perf_counter()
    z = C.interpolate_bed(scattered, gx, gy)
    assert time.perf_counter() - start < 5.0
    assert np.abs(z - np.sin(gx / 1000)).max() < 0.05


def _off_square(corridor):
    """How far the cross-sections are from square to the river (degrees, at the
    middle of the river), each."""
    x, y = corridor.grid()
    j = x.shape[0] // 2
    section = np.stack([x[-1] - x[0], y[-1] - y[0]], axis=1)
    along = np.stack([np.gradient(x[j]), np.gradient(y[j])], axis=1)
    cosine = np.abs((section * along).sum(1))
    cosine /= np.linalg.norm(section, axis=1) * np.linalg.norm(along, axis=1)
    return np.degrees(np.arcsin(np.clip(cosine, 0, 1)))


def _stretch(corridor):
    """The most a cell along a bank differs in length from its neighbour, and the
    longest cell along a bank over the shortest."""
    x, y = corridor.grid()
    along = np.hypot(np.diff(x[[0, -1]], axis=1), np.diff(y[[0, -1]], axis=1))
    ratio = along[:, 1:] / along[:, :-1]
    return float(np.maximum(ratio, 1 / ratio).max()), float((along.max(1) / along.min(1)).max())


def test_cross_sections_square_to_the_river():
    thalweg = C.find_thalweg(valley(), (0, 400, 0, 200, -2, 3), level=0.0)
    corridor = C.from_thalweg(thalweg, margin=20.0, count=14)
    assert corridor.sections == "normal"  # along the thalweg: square at once
    from dataclasses import replace

    even = replace(corridor, sections="even")
    assert np.median(_off_square(even)) > 15  # equal lengths: skewed in the bends
    square = _off_square(corridor)
    assert np.median(square) < 8.0 and square.max() < 15.0  # across the river, not skewed
    neighbour, spread = _stretch(corridor)
    assert neighbour < 1.25 and spread < 3.0  # and not squeezed or stretched along the banks
    x, y = corridor.grid()
    assert C.quality(x, y).folded == 0
    # a 90° bend: the cross-sections run to the centre of the bend
    a = np.radians(np.linspace(90, 0, 12))
    bend = C.Corridor(
        np.column_stack([100 * np.cos(a), 100 * np.sin(a)]),
        np.column_stack([40 * np.cos(a), 40 * np.sin(a)]),
        cells_along=60,
        cells_across=12,
        sections="normal",
    )
    x, y = bend.grid()
    radial = np.degrees(np.arctan2(y[0], x[0]) - np.arctan2(y[-1], x[-1]))
    assert np.abs(radial).max() < 4.0
    # a point dragged far out: still no folded cells
    dragged = corridor.moved("left", 6, *(corridor.left[6] + [0.0, 30.0]))
    assert C.quality(*dragged.grid()).folded == 0
    assert np.median(_off_square(dragged)) < 8.0 and _stretch(dragged)[0] < 1.25
    with pytest.raises(ValueError, match="sections"):
        replace(corridor, sections="radial")
    assert C.Corridor.from_dict({**corridor.to_dict(), "sections": "even"}).sections == "even"


def test_the_default_water_level_and_the_widest_bank():
    # heights to a sea level datum: the water at 0
    z = np.array([[-5.0, 2.0], [10.0, 20.0]])
    assert C.default_level(z, np.array([-5.0, -3.0])) == 0.0
    # an inland river: halfway between its bed and the top of the banks
    inland = np.array([[300.0, 302.0], [310.0, 320.0]])
    assert 300.0 < C.default_level(inland, np.array([300.0])) < 320.0
    # water spreading far over a flat marsh: the bank no farther than WIDEST half widths
    line = np.column_stack([np.linspace(0, 1000, 101), np.zeros(101)])
    left = np.full(101, 50.0)
    left[40:60] = 2000.0
    thalweg = C.Thalweg(line, np.zeros(101), left, np.full(101, 50.0), 0.0)
    corridor = C.from_thalweg(thalweg, margin=0.0, count=12)
    assert corridor.left[:, 1].max() <= C.WIDEST * 50.0 + 1e-6


def test_a_fine_grid_is_drawn_too(river):
    """A grid of more than 400,000 nodes was not drawn (and so vanished after Make
    grid): it is, every n-th line."""
    session = Session()
    case_id = session.open_case(river)
    session.start_corridor()
    session.set_corridor_options(cells_along=2320, cells_across=200)
    session.make_corridor_grid()
    marks = session.data.case_setup(case_id).marks()
    grid = [m for m in marks if m.group == "corridor_grid"]
    assert grid and 0 < len(grid[0].segments) < 2 * 121 * 2400
