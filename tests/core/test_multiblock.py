"""A multi-block grid: its topology in blocks.json (corners, edges, blocks with the
edge chains of their faces), the blocks meshed by transfinite interpolation and
Winslow, point matching by sharing edges, the rules of DIVEMesh and REEF3D, the
quality of the cells, blocks.dat and the set-up of a case for G 50 1 / R 20 1."""

import json

import numpy as np
import pytest

from lagoonviz.core import Session
from lagoonviz.core import corridor as C
from lagoonviz.core import multiblock as M

CONTROL = "C 11 21\nC 14 21\nB 1 1.0\nB 10 0.0 200.0 0.0 60.0 0.0 1.0\nM 10 1\n"
CTRL = "A 10 2\nF 60 2.0\n"


@pytest.fixture
def case(tmp_path):
    folder = tmp_path / "harbour"
    folder.mkdir()
    (folder / "control.txt").write_text(CONTROL)
    (folder / "ctrl.txt").write_text(CTRL)
    return folder


def two_boxes():
    """Block 0 over x 0 - 100, block 1 beside it over x 100 - 160, sharing an edge."""
    t = M.from_box((0, 100, 0, 50), (20, 10))
    return t.extruded("b0_e", 60.0, 12)


def test_tfi_keeps_the_faces_and_fills_a_square():
    s = np.column_stack([np.linspace(0, 1, 5), np.zeros(5)])
    n = np.column_stack([np.linspace(0, 1, 5), np.ones(5)])
    w = np.column_stack([np.zeros(4), np.linspace(0, 1, 4)])
    e = np.column_stack([np.ones(4), np.linspace(0, 1, 4)])
    x, y = M.tfi(s, n, w, e)
    assert x.shape == (4, 5)
    np.testing.assert_allclose(x[0], s[:, 0])
    np.testing.assert_allclose(y[:, -1], e[:, 1])
    np.testing.assert_allclose(x[2, 2], 0.5)
    assert (C.cell_areas(x, y) > 0).all()


def test_graded_edges_and_their_nodes():
    vertices = {"a": (0.0, 0.0), "b": (10.0, 0.0)}
    even = M.Edge("a", "b", 5).nodes(vertices)
    np.testing.assert_allclose(even[:, 0], [0, 2, 4, 6, 8, 10])
    graded = M.Edge("a", "b", 5, ratio=4.0).nodes(vertices)
    steps = np.diff(graded[:, 0])
    assert steps[-1] / steps[0] == pytest.approx(4.0) and graded[-1, 0] == 10.0
    curved = M.Edge("a", "b", 8, points=[[5.0, 3.0]]).nodes(vertices)
    assert curved[:, 1].max() > 2.5  # through the point


def test_blocks_share_the_nodes_of_their_interface():
    t = two_boxes()
    assert M.check(t) == []
    x0, y0 = t.block_nodes(0)
    x1, y1 = t.block_nodes(1)
    # block 1's j = 0 face is block 0's +i face, run backwards (j outwards to the left of i)
    face0 = np.column_stack([x0[:, -1], y0[:, -1]])
    face1 = np.column_stack([x1[0], y1[0]])
    np.testing.assert_array_equal(face0[::-1], face1)
    assert (C.cell_areas(x1, y1) > 0).all()  # right-handed
    q = M.quality(t)
    assert q.cells == 20 * 10 + 10 * 12 and q.folded == 0 and q.counts["good"] == q.cells


def test_cells_go_on_to_the_opposite_faces():
    t = two_boxes()
    finer = t.with_edge("b0_e", cells=16)
    assert finer.edges["b0_w"].cells == 16  # across block 0
    assert finer.edges["b0_e_x"].cells == 16  # and across block 1
    assert M.check(finer) == []
    twice = t.scaled(2)
    assert twice.size(0) == (40, 20) and twice.size(1) == (20, 24)


def test_the_rules():
    t = two_boxes()
    # block 0's interfaces must cover whole faces: cut its +i edge, add a block on half
    cut = t.split_edge("b0_n", 50.0, 50.0).extruded("b0_na", 20.0, 8)
    texts = [i.text for i in M.check(cut) if i.level == "error"]
    assert any("whole faces" in s for s in texts)
    # the same on block 1 is allowed
    side = t.split_edge("b0_e_x", 160.0, 25.0).extruded("b0_e_xa", 20.0, 8)
    assert [i for i in M.check(side) if i.level == "error"] == []
    # too shallow behind an interface with block 0
    shallow = M.from_box((0, 100, 0, 50), (20, 10)).extruded("b0_e", 10.0, 3)
    assert any("cells deep" in i.text and i.level == "error" for i in M.check(shallow))
    # open faces of the blocks 1..: inflow on -i, water level on +i, not on an interface
    wrong = t.with_bc(1, "j1", "outflow", 2.0)
    assert any("only on the +i face" in i.text for i in M.check(wrong))
    shared = t.with_bc(0, "i1", "outflow", 2.0)
    assert any("interface" in i.text for i in M.check(shared))
    nothing = t.with_bc(1, "i1", "outflow")
    assert any("no water level" in i.text for i in M.check(nothing))
    # a folded block
    folded = t.moved_vertex(t.edges["b0_e"].end, 20.0, -30.0)
    assert any("folded" in i.text or "left-handed" in i.text for i in M.check(folded))


def test_settings_for_divemesh_and_reef3d():
    t = two_boxes().with_bc(0, "i0", "inflow", 100.0)
    t = t.split_edge("b0_e_x", 160.0, 25.0)
    edge = [e for e in t.boundary_edges() if e.startswith("b0_e_x")]
    assert edge
    found = M.settings(t)
    assert found.sides["C 11"] == 1 and found.sides["C 14"] == 21 and found.inflow == 100.0
    assert found.box == pytest.approx((0, 160, 0, 50))
    # the far edge of the extrusion (cut in two) is block 1's +j face; its +i face may
    # carry a water level
    assert t.blocks[1].faces["j1"] == ["-b0_e_xb", "-b0_e_xa_2"]
    level = t.with_bc(1, "i1", "outflow", 10.0)
    assert M.settings(level).g54 == [[1, 2, 10.0]]
    assert [i for i in M.check(level) if i.level == "error"] == []


def test_the_file_and_blocks_dat(tmp_path):
    t = two_boxes()
    again = M.Topology.from_dict(json.loads(t.text()))
    assert again.text() == t.text()
    nodes = [(x, y, np.zeros_like(x)) for x, y in t.grids()]
    path = tmp_path / "blocks.dat"
    path.write_text(M.blocks_text(nodes))
    back = M.read_blocks(path)
    assert len(back) == 2 and back[1][0].shape == (13, 11)
    np.testing.assert_allclose(back[0][0], nodes[0][0], atol=1e-6)


def test_a_corridor_becomes_a_block():
    a = np.radians(np.linspace(90, 0, 12))
    left = np.column_stack([100 * np.cos(a), 100 * np.sin(a)])
    right = np.column_stack([60 * np.cos(a), 60 * np.sin(a)])
    corridor = C.Corridor(left, right, cells_along=60, cells_across=12, smoothing=100)
    t = M.from_corridor(corridor)
    assert M.check(t) == [] and t.size(0) == (60, 12)
    x, y = t.block_nodes(0)
    assert np.hypot(x[0], y[0]).max() == pytest.approx(60.0, rel=2e-3)  # j = 0: the right bank


def test_the_session_makes_the_grid_and_the_input_files(case):
    session = Session()
    session.open_case(case)
    session.start_multiblock([0, 100, 0, 50], cells=(20, 10))
    assert (case / "blocks.json").is_file()
    assert [i for i in session.scene if i.kind == "multiblock"]
    session.extrude_block("b0_e", 60.0, cells=12)
    session.set_block_face(0, "i0", "inflow", 50.0)
    session.set_block_face(1, "i1", "outflow", 2.5)
    issues, quality = session.multiblock_check()
    assert issues == [] and quality.cells == 320
    session.make_multiblock_grid()
    assert M.read_blocks(case / "blocks.dat")[1][0].shape == (13, 11)
    control = session.input_file("control.txt")
    ctrl = session.input_file("ctrl.txt")
    assert control.int("R 20") == 1 and control.int("C 11") == 1
    assert control.values("B 10")[:4] == pytest.approx([0, 160, 0, 50])
    assert ctrl.int("G 50") == 1 and ctrl.values("G 54") == [1, 2, 2.5]
    assert ctrl.int("B 60") == 1 and ctrl.value("W 10") == 50.0
    # one grid at a time: the blocks' item draws it, blocks.dat's is not drawn as well
    item = next(i for i in session.scene if i.kind == "multiblock")
    case_id = session.data.case_item(item.id).id
    groups = {m.group for m in session.data.case_setup(case_id).marks()}
    assert "multiblock_grid" in groups and "curvgrid" not in groups
    # undo: one step back to before Make grid
    session.undo()
    assert not (case / "blocks.dat").is_file()


def test_the_set_up_marks_of_the_blocks(case):
    session = Session()
    session.open_case(case)
    session.start_multiblock([0, 100, 0, 50], cells=(20, 10))
    session.extrude_block("b0_e", 4.0, cells=3)  # too shallow: a rule broken
    item = next(i for i in session.scene if i.kind == "multiblock")
    groups = {m.group for m in session.data.case_setup(session.data.case_item(item.id).id).marks()}
    assert {"multiblock", "multiblock_grid", "multiblock_issues"} <= groups
    marks = session.data.setup_marks(item.id)
    assert marks is not None
    with pytest.raises(ValueError, match="rule"):
        session.make_multiblock_grid()
