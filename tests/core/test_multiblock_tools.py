"""Tools for multi-block grids of coasts and estuaries: edges spaced by the cells at
their ends, circular arcs, edges laid along the water (on land a margin from it,
smooth), and a blocks.dat made elsewhere read back into a topology."""

import json

import numpy as np
import pytest

from lagoonviz.core import Session
from lagoonviz.core import multiblock as M
from lagoonviz.core import waterline as W

CONTROL = "C 11 2\nC 12 21\nC 13 21\nC 14 21\nB 1 1.0\nB 10 0.0 1.0 0.0 1.0 0.0 5.0\nR 20 1\n"
CTRL = "A 10 5\nG 50 1\nF 60 10.0\nG 54 2 1 50.0\n"


# ---------------------------------------------------------------- spacing
def test_cells_from_the_spacing_at_the_ends():
    grow = M.check_spacing({"start": 60, "end": 1500})
    n = M.cells_for(50_000, grow)
    sizes = M.spacing_sizes(n, grow) * 50_000 / M.spacing_sizes(n, grow).sum()
    assert 100 < n < 120
    assert sizes[0] == pytest.approx(60, rel=0.05) and sizes[-1] == pytest.approx(1500, rel=0.05)
    assert M.cells_for(1000, {"start": 10, "end": 10}) == 100
    both = M.check_spacing({"start": 100, "end": 50, "max": 400, "growth": 1.1})
    n = M.cells_for(20_000, both)
    sizes = M.spacing_sizes(n, both) * 20_000 / M.spacing_sizes(n, both).sum()
    assert sizes[0] == pytest.approx(100, rel=0.05) and sizes[-1] == pytest.approx(50, rel=0.05)
    assert sizes.max() == pytest.approx(400, rel=0.05)
    assert (sizes[1:] / sizes[:-1]).max() <= 1.1 + 1e-9
    with pytest.raises(ValueError):
        M.check_spacing({"start": 0, "end": 5})
    with pytest.raises(ValueError):
        M.check_spacing({"start": 10, "end": 50, "max": 20})


def test_a_spacing_goes_across_the_blocks_turned_their_way():
    t = M.from_box((0, 1000, 0, 500), (10, 10)).extruded("b0_n", 300.0, 10)
    # block 1 runs along b0_n backwards or forwards: either way its far edge gets the
    # spacing its own way round
    t = t.with_spacing("b0_s", {"start": 10, "end": 100})
    across = t.across("b0_s")
    assert set(across) == {"b0_s", "b0_n", "b0_n_x"}
    for name in across:
        e = t.edges[name]
        first, last = e.end_sizes(t.vertices)
        small_at_x0 = (first < last) == (t.vertices[e.start][0] < t.vertices[e.end][0])
        assert small_at_x0  # fine at x = 0 on every edge, whichever way it runs
        assert e.cells == t.edges["b0_s"].cells
    assert t.edges["b0_s"].end_sizes(t.vertices)[0] == pytest.approx(10, rel=0.05)
    assert not [i for i in M.check(t) if i.level == "error"]
    # moved corners: the cells follow the spacing again when asked
    longer = t.moved_vertex("b0_se", 2000, 0).moved_vertex("b0_ne", 2000, 500)
    longer = longer.moved_vertex(t.edges["b0_n_x"].end, 2000, 800)
    assert longer.fitted_cells().edges["b0_s"].cells > t.edges["b0_s"].cells
    data = json.loads(t.text())
    assert data["edges"]["b0_s"]["spacing"] == {"start": 10.0, "end": 100.0}
    assert M.Topology.from_dict(data).edges["b0_s"].spacing == {"start": 10.0, "end": 100.0}


# ---------------------------------------------------------------- arcs
def test_circular_arcs():
    arc = M.arc_through((0, 0), (1, 1), (2, 0))
    assert np.allclose(np.hypot(arc[:, 0] - 1, arc[:, 1]), 1.0)
    assert arc[:, 1].min() >= -1e-9  # the way round through (1, 1)
    other = M.arc_through((0, 0), (1, -1), (2, 0))
    assert other[:, 1].max() <= 1e-9
    big = M.arc_through((0, 0), (-1, 1), (0, 2))  # more than a half circle is fine too
    assert big[:, 0].min() == pytest.approx(-1.0, abs=1e-3)
    assert np.allclose(M.arc_through((0, 0), (1, 0), (2, 0))[[0, -1]], [(0, 0), (2, 0)])
    # an edge made an arc bulges out of its block, and the block meshes
    t = M.from_box((0, 100, 0, 50), (10, 5)).with_edge("b0_w", curve="arc")
    ((px, _py),) = t.edges["b0_w"].points
    assert px == pytest.approx(-12.5)
    assert t.edges["b0_w"].line(t.vertices)[:, 0].min() == pytest.approx(-12.5, abs=0.01)
    assert not [
        i for i in M.check(t) if i.level == "error"
    ]


# ---------------------------------------------------------------- the water
def bay_bed(step=10.0):
    """Sea (z = 0) below y = 0, a bay 200 m wide reaching up to y = 300 at x 400 -
    600, land (z = 20) elsewhere: geo.dat's points on a raster."""
    xs = np.arange(-200.0, 1200.0 + 1, step)
    ys = np.arange(-400.0, 700.0 + 1, step)
    gx, gy = np.meshgrid(xs, ys)
    water = (gy < 0) | ((np.abs(gx - 500) < 100) & (gy < 300))
    z = np.where(water, 0.0, 20.0)
    return np.column_stack([gx.ravel(), gy.ravel(), z.ravel()])


def distance_to_water(p):
    """How far a point is from the sea or the bay of :func:`bay_bed`."""
    x, y = p[:, 0], p[:, 1]
    sea = np.maximum(y, 0.0)
    dx = np.maximum(np.abs(x - 500) - 100, 0.0)
    dy = np.maximum(y - 300, 0.0)
    bay = np.hypot(dx, dy)
    return np.minimum(sea, bay)


def test_a_line_along_the_water():
    points = bay_bed()
    guide = np.column_stack([np.linspace(0, 1000, 50), np.full(50, 150.0)])
    found = W.waterline(points, guide, 10.0, margin=40.0, radius=60.0)
    line = found.line
    assert found.wet == 0
    d = distance_to_water(line)
    assert d.min() >= 40.0 - 2 * found.step  # never nearer the water than the margin
    # it goes round the bay: up to near its head and back
    assert line[:, 1].max() == pytest.approx(340.0, abs=25.0)
    # the corners come to the shore; or stay where they were
    assert distance_to_water(line[[0, -1]]) == pytest.approx(40.0, abs=2 * found.step)
    kept = W.waterline(points, guide, 10.0, margin=40.0, radius=60.0, corners=False)
    assert np.allclose(kept.line[[0, -1]], guide[[0, -1]])
    assert distance_to_water(kept.line[5:-5]).min() >= 40.0 - 2 * kept.step
    # the bigger the radius, the smoother: a radius wider than the bay fills it
    smooth = W.waterline(points, guide, 10.0, margin=40.0, radius=400.0)
    assert smooth.line[:, 1].max() > 300.0  # (the bay is water: it still goes round it)
    with pytest.raises(ValueError, match="no water"):
        W.waterline(points, guide, -5.0, margin=40.0, radius=60.0)


# ---------------------------------------------------------------- import
def three_blocks():
    """Block 0, block 1 beside it, block 2 on a part of block 1's far face (a river
    into a harbour)."""
    t = M.from_box((0, 100, 0, 60), (20, 12)).extruded("b0_e", 50.0, 10)
    t = t.split_edge("b0_e_x", 150.0, 20.0)

    def edge_between(y0, y1):
        for name, e in t.edges.items():
            ys = sorted((t.vertices[e.start][1], t.vertices[e.end][1]))
            xs = (t.vertices[e.start][0], t.vertices[e.end][0])
            if xs == (150.0, 150.0) and np.allclose(ys, (y0, y1)):
                return name
        raise KeyError((y0, y1))

    t = t.split_edge(edge_between(20, 60), 150.0, 40.0)
    return t.extruded(edge_between(20, 40), 80.0, 16)


def test_a_blocks_dat_made_elsewhere_read_back():
    t = three_blocks()
    assert not [i for i in M.check(t) if i.level == "error"]
    nodes = [(x, y, np.zeros_like(x)) for x, y in t.grids()]
    bcs = {(2, "i0"): {"type": "inflow", "value": 5.0}}
    back = M.from_blocks(nodes, ["sea", "harbour", "river"], bcs, smoothing=0)
    assert [b.name for b in back.blocks] == ["sea", "harbour", "river"]
    assert M.interface_count(back) == M.interface_count(t) == 2
    # the faces keep their nodes exactly; the interfaces are shared edges
    for k in range(3):
        for face in M.FACES:
            assert np.allclose(back.face_nodes(k, face), t.face_nodes(k, face), atol=1e-6)
    cut = [f for f in M.FACES if len(t.blocks[1].faces[f]) == 3]
    assert cut and len(back.blocks[1].faces[cut[0]]) == 3  # cut where the river meets it
    assert all(e.curve == "nodes" for e in back.edges.values())
    assert back.blocks[2].bc == {"i0": {"type": "inflow", "value": 5.0}}
    assert not [i for i in M.check(back) if i.level == "error"]
    # more or fewer cells on an imported edge keep its grading
    edge = back.edges["b0_j0"]
    graded = M.Edge(edge.start, edge.end, 40, edge.points, "nodes")
    assert len(graded.nodes(back.vertices)) == 41
    # blocks.json keeps it
    assert M.Topology.from_dict(json.loads(back.text())).text() == back.text()


def test_the_boundaries_from_the_input_files():
    from lagoonviz.core.data.controlfile import ControlFile

    control = ControlFile("control.txt", CONTROL)
    ctrl = ControlFile("ctrl.txt", CTRL)
    bcs = M.bcs_from_inputs(control, ctrl)
    assert bcs == {
        (0, "i0"): {"type": "open", "value": 2.0},
        (2, "i0"): {"type": "inflow", "value": 50.0},
    }


def test_open_faces_keep_their_code():
    t = M.from_box((0, 100, 0, 50), (10, 5)).with_bc(0, "i0", "open", 2)
    assert not [i for i in M.check(t) if i.level == "error"]
    assert M.settings(t).sides["C 11"] == 2 and M.settings(t).g54 == []
    t = t.extruded("b0_e", 50.0, 10).with_bc(1, "j0", "open")
    assert any("only on block 0" in i.text for i in M.check(t))


# ---------------------------------------------------------------- the session
@pytest.fixture
def estuary(tmp_path):
    folder = tmp_path / "estuary"
    folder.mkdir()
    (folder / "control.txt").write_text(CONTROL)
    (folder / "ctrl.txt").write_text(CTRL)
    return folder


def test_the_session_imports_a_blocks_dat(estuary):
    t = three_blocks()
    nodes = [(x, y, 5.0 + 0.01 * x) for x, y in t.grids()]
    (estuary / "blocks.dat").write_text(M.blocks_text(nodes))
    s = Session()
    s.open_case(estuary)
    assert s.import_blocks(smoothing=50) == 3
    topology = s.multiblock()
    assert M.interface_count(topology) == 2 and topology.blocks[0].smoothing == 50
    assert topology.blocks[0].bc_type("i0") == "open"
    assert topology.blocks[2].bc == {"i0": {"type": "inflow", "value": 50.0}}
    # the bed of the nodes kept as geo.dat: Make grid gives it back
    assert (estuary / "geo.dat").is_file()
    s.make_multiblock_grid()
    made = M.read_blocks(estuary / "blocks.dat")
    for (x, _y, z), (_x0, _y0, _z0) in zip(made, nodes, strict=True):
        assert np.allclose(z, 5.0 + 0.01 * x, atol=0.05)
    assert "session.import_blocks(smoothing=50)" in s.to_script()
    s.undo()
    s.undo()
    assert not (estuary / "blocks.json").exists() and not (estuary / "geo.dat").exists()


def test_the_session_follows_the_water(estuary):
    np.savetxt(estuary / "geo.dat", bay_bed(10.0), fmt="%.1f")
    s = Session()
    s.open_case(estuary)
    s.start_multiblock([0, 1000, -300, 150], cells=(20, 10))
    s.follow_water("b0_n", margin=40, radius=60)
    t = s.multiblock()
    edge = t.edges["b0_n"]
    assert edge.water == {"margin": 40.0, "radius": 60.0} and edge.curve == "through"
    line = edge.line(t.vertices)
    # (within about a raster cell: a third of the smaller of margin and radius, and
    # half a cell of geo.dat, where the bed's height passes the level)
    assert distance_to_water(line).min() >= 40.0 - 20.0
    assert line[:, 1].max() > 300.0  # round the bay
    # a corner moved along the shore: the edge is laid again, the corner onto it
    s.move_block_vertex("b0_ne", 1000.0, 120.0)
    t = s.multiblock()
    assert distance_to_water(np.array([t.vertices["b0_ne"]]))[0] == pytest.approx(40, abs=15)
    # spacing, script and undo
    s.set_block_spacing("b0_s", 20.0, 80.0)
    assert s.multiblock().edges["b0_n"].spacing == {"start": 20.0, "end": 80.0}
    script = s.to_script()
    assert "session.follow_water('b0_n', margin=40.0, radius=60.0)" in script
    assert "session.set_block_spacing('b0_s', 20.0, 80.0)" in script
    # a point dragged by hand: the edge is shaped by hand from then on
    s.move_block_point("b0_n", 0, 100.0, 380.0)
    assert s.multiblock().edges["b0_n"].water is None
    s.undo()
    assert s.multiblock().edges["b0_n"].water is not None


def test_following_the_water_needs_the_bed(estuary):
    s = Session()
    s.open_case(estuary)
    s.start_multiblock([0, 1000, -300, 150], cells=(20, 10))
    with pytest.raises(ValueError, match="geo.dat"):
        s.follow_water("b0_n", margin=40, radius=60)
