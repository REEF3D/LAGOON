"""Porous structures (VRANS) in the set-up: the ctrl.txt lines of CFD (B 270 - B 291,
material on each line) and NHFLOW (B 210 - B 218, material B 201), added as one undo
step, and drawn where REEF3D puts them."""

import numpy as np
import pytest

from lagoonviz.core import Session, porous
from lagoonviz.core.data.controlfile import ControlFile, catalog_for

#: a 2D flume, 20 m long and 1 m high, for REEF3D::CFD
CONTROL = "C 11 6\nC 14 7\nB 1 0.05\nB 10 0.0 20.0 0.0 0.05 0.0 1.0\nM 10 1\n"
CTRL = "A 10 6\nF 60 0.6\nP 51 2.0 0.025\n"


@pytest.fixture
def flume(tmp_path):
    case = tmp_path / "flume"
    case.mkdir()
    (case / "control.txt").write_text(CONTROL)
    (case / "ctrl.txt").write_text(CTRL)
    return case


def test_the_kinds_each_module_reads():
    assert [k.code for k in porous.kinds_for("CFD")] == [
        "B 270",
        "B 274",
        "B 281",
        "B 282",
        "B 291",
    ]
    assert [k.code for k in porous.kinds_for("nhflow")] == [
        "B 210",
        "B 212",
        "B 213",
        "B 215",
        "B 217",
        "B 218",
    ]
    assert porous.kinds_for("SFLOW") == [] and porous.kinds_for("") == []
    box = porous.BY_CODE["B 270"]
    assert box.count == 10 and box.material_on_line
    assert porous.BY_CODE["B 291"].count == 11 and porous.BY_CODE["B 210"].count == 6
    # every kind's line has as many values as REEF3D's reader takes
    catalog = catalog_for("ctrl.txt")
    for kind in porous.KINDS:
        assert catalog.get(kind.code).count == kind.count, kind.code


def test_a_structure_from_an_area():
    area, heights = (8.0, 10.0, 0.0, 1.0), (0.0, 0.6)
    box = porous.from_area(porous.BY_CODE["B 270"], area, heights)
    assert box == [8.0, 10.0, 0.0, 1.0, 0.0, 0.6, *porous.DEFAULT_MATERIAL]
    cylinder = porous.from_area(porous.BY_CODE["B 274"], area, heights, (0.5, 0.02, 1000, 1.1))
    assert cylinder == [9.0, 0.5, 0.0, 0.6, 0.5, 0.5, 0.02, 1000.0, 1.1]
    wedge = porous.from_area(porous.BY_CODE["B 217"], (10.0, 8.0, 1.0, 0.0), heights)
    assert wedge == [8.0, 10.0, 0.0, 1.0, 0.0, 0.6]  # rising along x, no material (B 201)
    plate = porous.from_area(porous.BY_CODE["B 291"], area, heights)
    assert plate[:7] == [8.0, 10.0, 0.0, 1.0, 0.0, 0.6, 0.06]
    sphere = porous.from_area(porous.BY_CODE["B 215"], area, heights)
    assert sphere == [9.0, 0.5, 0.3, 0.3]
    across = porous.from_area(porous.BY_CODE["B 212"], area, heights)
    assert across == [9.0, 0.3, 0.0, 1.0, 0.3]
    kind = porous.BY_CODE["B 270"]
    assert kind.split(box) == ([8.0, 10.0, 0.0, 1.0, 0.0, 0.6], list(porous.DEFAULT_MATERIAL))
    assert "x 8 – 10" in kind.describe(box) and "n 0.4, d50 0.05 m" in kind.describe(box)
    with pytest.raises(ValueError, match="6 shape values"):
        kind.join([1.0, 2.0])
    assert porous.snap(8.013, 0.05) == 8.0 and porous.snap(0.123, 0) == 0.12
    assert porous.default_area((0, 20, 0, 0.05, 0, 1), two_d=True) == (8.0, 12.0, 0, 0.05)


def test_adding_a_porous_structure_to_a_cfd_case(flume):
    session = Session()
    case_id = session.open_case(flume)
    occurrence = session.add_porous("B 270")
    assert occurrence == 0
    ctrl = session.input_file("ctrl.txt")
    # the middle fifth of the flume, all across it (2D), from the bed to the still water
    assert ctrl.all("B 270") == [[8.0, 12.0, 0.0, 0.05, 0.0, 0.6, 0.4, 0.05, 500.0, 2.0]]
    drawn = session.porous_default("B 281", area=(14.0, 16.0, 0.0, 0.05))
    assert session.add_porous("B 281", drawn) == 0
    assert "session.add_porous('B 270', [8.0, 12.0" in session.to_script()
    structures = porous.structures(session.input_file("ctrl.txt"), "CFD")
    assert [s.label() for s in structures] == ["Box (B 270 #1)", "Wedge along x (B 281 #1)"]
    # drawn in the set-up where REEF3D puts them
    marks = {m.code: m for m in session.data.case_setup(case_id).marks()}
    assert (
        marks["B 270"].group == "structures"
        and marks["B 281"].legend == "porous wedge (VRANS) (B 281)"
    )
    wedge = marks["B 281"].segments.reshape(-1, 3)
    top = wedge[np.isclose(wedge[:, 2], 0.6)]
    assert len(top) and np.allclose(top[:, 0], 16.0)  # high at its end, x = 16
    assert wedge[:, 2].min() == 0.0 and wedge[:, 0].min() == 14.0
    with pytest.raises(ValueError, match="takes 10 values"):
        session.add_porous("B 270", [1.0, 2.0])
    with pytest.raises(ValueError, match="not a porous structure"):
        session.add_porous("P 51")
    session.undo()
    session.undo()
    assert session.input_file("ctrl.txt").text() == CTRL
    session.redo()
    assert session.save_inputs() == [flume / "ctrl.txt"]
    assert "B 270 8.0 12.0 0.0 0.05 0.0 0.6 0.4 0.05 500.0 2.0" in (flume / "ctrl.txt").read_text()


def test_nhflow_gets_its_material_with_the_first_structure(flume):
    (flume / "ctrl.txt").write_text("A 10 5\nF 60 0.6\n")
    session = Session()
    session.open_case(flume)
    session.add_porous("B 213", [10.0, 0.025, 0.0, 0.6, 0.3])
    ctrl = session.input_file("ctrl.txt")
    assert ctrl.values("B 201") == list(porous.DEFAULT_MATERIAL)
    assert ctrl.values("B 213") == [10.0, 0.025, 0.0, 0.6, 0.3]
    session.set_input("ctrl.txt", "B 201", [0.45, 0.02, 500.0, 2.0])
    session.add_porous("B 210")  # B 201 is there: kept as it is
    assert session.input_file("ctrl.txt").values("B 201") == [0.45, 0.02, 500.0, 2.0]
    assert len(session.input_file("ctrl.txt").entries("B 201")) == 1
    session.undo()
    session.undo()
    session.undo()  # the cylinder and B 201 were one step
    assert "B 201" not in session.input_file("ctrl.txt")


def test_wedges_and_plates_are_drawn_as_reef3d_reads_them():
    from lagoonviz.core.data.setup import _shaped

    span = (0.0, 1.0)
    # B 282: along y, the top from z 0.8 at y 1 down to 0.2 at y 3, the base at 0.2
    segments, _, label = _shaped("wedge_y", [0, 2, 1, 3, 0.8, 0.2], span, span, span, 0.5)
    points = segments.reshape(-1, 3)
    assert points[:, 2].min() == pytest.approx(0.2) and points[:, 2].max() == pytest.approx(0.8)
    assert np.allclose(points[np.isclose(points[:, 2], 0.8)][:, 1], 1.0)
    assert len(segments) == 9  # a triangle at each end and three edges between
    # B 291: a band 0.1 thick above the line from (0, 0.2) to (4, 0.6)
    segments, _, label = _shaped("plate_x", [0, 4, 0, 1, 0.2, 0.6, 0.1], span, span, span, 0.5)
    points = segments.reshape(-1, 3)
    assert len(segments) == 12 and points[:, 2].max() == pytest.approx(0.7)
    assert label == (4, 1, pytest.approx(0.7))


def test_the_catalog_describes_cfd_vrans():
    catalog = catalog_for("ctrl.txt")
    box = catalog.get("B 270")
    assert box.title == "VRANS porous box" and box.modules == "C"
    assert box.names()[6:] == ["n", "d_50", "α", "β"]
    assert catalog.get("B 291").names()[6] == "d"
    ctrl = ControlFile("ctrl.txt", "A 10 6\nB 270 0 1 0 1 0 1 0.4 0.05 500 2\n")
    assert not [p for p in catalog.check(ctrl, "CFD") if p.code == "B 270"]
    assert [p for p in catalog.check(ctrl, "NHFLOW") if p.code == "B 270"]  # not NHFLOW's
