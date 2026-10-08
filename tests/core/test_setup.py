"""The set-up of a case from its input files: facts, marks, the "Case set-up" item,
a case that has not run yet, and the set-up sketch."""

import numpy as np
import pytest

from lagoonviz.core import Session
from lagoonviz.core.data.setup import COLORS, read_setup

#: a 3D wave basin: waves from x min, a beach at x max, a monopile, gauges and probes
CONTROL = """\
C 11 6     // waves
C 14 7     // beach
C 16 3
B 1 0.25
B 10 0.0 20.0 0.0 5.0 0.0 1.5
S 33 10.0 2.5 0.4   // monopile
M 10 4
"""
CTRL = """\
A 10 5
B 90 1
B 92 5
B 93 0.1 2.0
B 96 4.0 6.0
B 98 2
B 99 1
F 60 1.0
P 51 2.0 2.5
P 51 8.0 2.5
P 52 2.5
P 61 9.0 2.5 0.5
P 81 9.0 11.0 1.5 3.5 0.0 1.5
X 110 14.0 15.0 2.0 3.0 0.8 1.2
X 311 13.0 14.0 2.5 2.5 0.0 0.8 1.0 1000 1e6
M 10 2
"""


@pytest.fixture
def basin(tmp_path):
    case = tmp_path / "basin"
    case.mkdir()
    (case / "control.txt").write_text(CONTROL)
    (case / "ctrl.txt").write_text(CTRL)
    return case


def test_the_facts_of_the_input_files(basin):
    setup = read_setup(basin)
    assert setup.domain == (0.0, 20.0, 0.0, 5.0, 0.0, 1.5)
    assert setup.cells == (80, 20, 6) and setup.spacing == pytest.approx((0.25, 0.25, 0.25))
    assert setup.sides[1] == 6 and setup.sides[4] == 7 and setup.sides[2] == 21
    assert setup.water_level == 1.0 and setup.waves and setup.solver == "NHFLOW"
    text = setup.describe()
    assert "80 x 20 x 6 = 9,600 cells" in text and "x min wave generation" in text
    assert "5th-order Stokes Waves (Fenton) (B 92 5), H 0.1 m, T 2 s" in text
    assert "M 10 differs: 4 in control.txt, 2 in ctrl.txt" in text


def test_the_marks_of_the_set_up(basin):
    marks = read_setup(basin).marks()
    legends = {m.legend: m for m in marks}
    assert {"domain (B 10)", "wave generation side (C 11)", "numerical beach side (C 14)"} <= set(
        legends
    )
    generation = legends["wave generation zone (B 96)"]
    xs = generation.segments[..., 0]
    assert xs.min() == 0.0 and xs.max() == 4.0  # 4 m from x min, at the water
    assert np.allclose(generation.segments[..., 2], 1.0)
    beach = legends["numerical beach zone (B 96)"].segments[..., 0]
    assert beach.min() == 14.0 and beach.max() == 20.0
    gauges = [m for m in marks if m.code == "P 51"]
    assert [m.labels[0][0] for m in gauges] == ["1", "2"]
    staff = gauges[1].segments[0]
    assert staff[0].tolist() == [8.0, 2.5, 0.0] and staff[1].tolist() == [8.0, 2.5, 1.5]
    line = legends["free surface line in x (P 52)"].segments[0]
    assert line[:, 0].tolist() == [0.0, 20.0] and line[0, 1] == 2.5
    assert legends["point probe (P 61)"].points.tolist() == [[9.0, 2.5, 0.5]]
    assert len(legends["force box (P 81)"].segments) == 12
    assert len(legends["floating body (X 110)"].segments) == 12
    mooring = legends["mooring line (X 311)"].segments[0]
    assert mooring.tolist() == [[13.0, 2.5, 0.0], [14.0, 2.5, 0.8]]
    pile = legends["solid (S 33)"]
    assert pile.group == "solids" and np.isclose(pile.segments[..., 2].max(), 1.5)
    assert "wave direction (B 105)" in legends  # along x from the middle of x min


def test_a_2d_case_has_symmetry_planes_in_y(tmp_path):
    (tmp_path / "control.txt").write_text("B 2 200 1 20\nB 10 0 10 0 0.05 0 1\nC 12 21\n")
    setup = read_setup(tmp_path)
    assert setup.two_d and setup.sides[2] == setup.sides[3] == 3
    assert setup.spacing == pytest.approx((0.05, 0.05, 0.05))


def test_edits_change_the_marks(basin):
    setup = read_setup(basin)
    assert len([m for m in setup.marks() if m.code == "P 51"]) == 2
    setup.ctrl.add("P 51", [12.0, 1.0])
    setup.control.set("B 10", [0, 30, 0, 5, 0, 1.5])
    gauges = [m for m in setup.marks() if m.code == "P 51"]
    assert len(gauges) == 3 and gauges[2].labels[0][0] == "3"
    assert setup.domain[1] == 30.0 and setup.cells == (120, 20, 6)


def test_a_case_that_has_not_run_opens_as_its_set_up(basin):
    session = Session()
    case_id = session.open_case(basin)
    kinds = sorted(i.kind for i in session.scene)
    assert kinds == ["case", "grid", "setup", "structure"]  # the monopile from control.txt
    assert session.data.case(case_id).setup_only
    item = next(i for i in session.scene if i.kind == "setup")
    marks = session.data.setup_marks(item.id)
    assert marks.visible and any(text == "1" for text, _, _ in marks.labels)
    assert all(segments.dtype == np.float32 for _, segments in marks.lines)
    session.set_property(item.id, "show_gauges", False)
    labels = session.data.setup_marks(item.id).labels
    assert labels and not any(color == COLORS["gauges"] for _, _, color in labels)
    # exaggerated about the still water level, as the other items
    session.set_property(case_id, "exaggeration", 2.0)
    marks = session.data.setup_marks(item.id)
    top = max(float(s[..., 2].max()) for _, s in marks.lines)
    assert top == pytest.approx(1.0 + 2.0 * 0.5)


def test_a_folder_without_input_or_output_is_refused(tmp_path):
    with pytest.raises(ValueError, match="input files"):
        Session().open_case(tmp_path)


def test_setup_item_on_a_case_with_output(reef3d_run):
    (reef3d_run / "control.txt").write_text("B 10 0.0 8.0 0.0 4.0 -0.5 3.0\nB 1 0.5\n")
    (reef3d_run / "ctrl.txt").write_text("A 10 5\nF 60 2.0\nP 51 4 2\n")
    session = Session()
    session.open_case(reef3d_run)
    # results with input files: the set-up is there, switched off
    item = next(i for i in session.scene if i.kind == "setup")
    assert not item.visible
    marks = session.data.setup_marks(item.id)
    assert not marks.visible and any(text == "1" for text, _, _ in marks.labels)
    session.set_visible(item.id, True)
    assert session.data.setup_marks(item.id).visible
    # Add > Case set-up gives another one, shown
    added = session.add_item("setup")
    assert session.data.setup_marks(added).visible
    # without input files: none
    other = Session()
    (reef3d_run / "control.txt").unlink()
    (reef3d_run / "ctrl.txt").unlink()
    other.open_case(reef3d_run)
    assert not any(i.kind == "setup" for i in other.scene)


def test_results_fill_in_what_the_inputs_lack(reef3d_run):
    """No control.txt and no ctrl.txt beside the results: the domain comes from the
    results, ctrl.txt from the copy REEF3D kept in REEF3D_Case."""
    (reef3d_run / "REEF3D_Case").mkdir()
    (reef3d_run / "REEF3D_Case" / "ctrl-20260101T000000Z-aaaa.txt").write_text("F 60 9\n")
    (reef3d_run / "REEF3D_Case" / "ctrl-20261003T192135Z-8b14.txt").write_text(
        "A 10 5\nF 60 2.0\nP 51 4 2\n"
    )
    session = Session()
    session.open_case(reef3d_run)
    item = next(i for i in session.scene if i.kind == "setup")
    setup = session.data.case_setup(item.id)
    assert setup.water_level == 2.0  # the newest copy
    assert setup.domain_from == "the results"
    xs, xe, ys, ye, _, _ = setup.domain
    assert xs == pytest.approx(0.0, abs=0.5) and xe == pytest.approx(8.0, abs=0.5)
    assert any(m.legend == "wave gauge (P 51)" for m in setup.marks())
    assert "REEF3D_Case/ctrl-20261003T192135Z-8b14.txt" in setup.describe()


def test_the_set_up_sketch(basin, tmp_path):
    from lagoonviz.render.figure import FigureSpec, save_figure
    from lagoonviz.render.setup_sketch import main, setup_figure

    figure = setup_figure(basin, FigureSpec(width_mm=120))
    assert len(figure.axes) == 3  # plan, side, legend
    legend = figure.axes[2].get_legend()
    labels = [t.get_text() for t in legend.get_texts()]
    assert "wave gauge (P 51)" in labels and "force box (P 81)" in labels
    path = save_figure(figure, tmp_path / "sketch.png", FigureSpec(width_mm=120, dpi=80))
    assert path.stat().st_size > 1000
    assert main([str(basin), str(tmp_path / "cli.pdf")]) == 0
    assert (tmp_path / "cli.pdf").stat().st_size > 1000


def test_lagoon_setup_describes(basin, capsys):
    from lagoonviz.render.setup_sketch import main

    assert main([str(basin), "--describe"]) == 0
    assert "domain: x 0 .. 20, y 0 .. 5, z 0 .. 1.5 m" in capsys.readouterr().out


def test_editing_the_input_files(basin):
    session = Session()
    case_id = session.open_case(basin)
    session.set_input("ctrl.txt", "P 51", [3.0, 2.5], occurrence=1)  # move gauge 2
    session.add_input("ctrl.txt", "P 51", [12.0, 1.0])
    session.set_input("control.txt", "B 1", [0.5])
    session.remove_input("ctrl.txt", "X 311")
    assert session.inputs_modified() == ["control.txt", "ctrl.txt"]
    ctrl = session.input_file("ctrl.txt")
    assert ctrl.all("P 51") == [[2.0, 2.5], [3.0, 2.5], [12.0, 1.0]] and "X 311" not in ctrl
    setup = session.data.case_setup(case_id)
    assert setup.cells == (40, 10, 3)
    script = session.to_script()
    assert "session.set_input('ctrl.txt', 'P 51', [3.0, 2.5], occurrence=1)" in script
    assert "session.remove_input('ctrl.txt', 'X 311')" in script
    for _ in range(4):
        session.undo()
    assert session.input_file("ctrl.txt").text() == CTRL and not session.inputs_modified()
    session.redo()
    written = session.save_inputs()
    assert written == [basin / "ctrl.txt"] and (basin / "ctrl.txt").read_text().count("P 51") == 2
    assert "P 51 3.0 2.5" in (basin / "ctrl.txt").read_text()
    # the text editor: the whole file at once, undone as one step
    session.set_input_text("control.txt", "B 10 0 10 0 5 0 1.5\nB 1 0.5\n")
    assert session.data.case_setup(case_id).domain[1] == 10.0
    session.undo()
    assert session.data.case_setup(case_id).domain[1] == 20.0


def test_a_case_without_input_files_starts_them(reef3d_run):
    session = Session()
    session.open_case(reef3d_run)
    assert not (reef3d_run / "ctrl.txt").exists()
    session.set_input("ctrl.txt", "P 51", [4.0, 2.0])
    assert session.save_inputs() == [reef3d_run / "ctrl.txt"]
    assert (reef3d_run / "ctrl.txt").read_text() == "P 51 4.0 2.0\n"


def test_the_selected_command_is_highlighted(basin):
    session = Session()
    case_id = session.open_case(basin)
    item = next(i for i in session.scene if i.kind == "setup")
    assert session.data.setup_marks(item.id).highlight is None
    session.data.set_highlight(case_id, "ctrl.txt", "P 51", 1)
    segments, points = session.data.setup_marks(item.id).highlight
    assert np.allclose(segments[0][:, :2], [[8.0, 2.5], [8.0, 2.5]])  # the second gauge
    session.data.set_highlight(case_id, "control.txt", "B 10", 0)
    assert len(session.data.setup_marks(item.id).highlight[0]) == 12  # the domain's box
    session.data.set_highlight(case_id, "ctrl.txt", "")
    assert session.data.setup_marks(item.id).highlight is None
