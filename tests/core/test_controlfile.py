"""The input files (ctrl.txt, control.txt) read and edited line by line, and the
catalogues of their commands from the user guides and the programs' readers."""

from lagoonviz.core.data.controlfile import Catalog, ControlFile, catalog_for, format_value

CTRL = """\
// a wave flume
A 10 6          // CFD
F 60 0.5

P 51 1.0 0.5    // first gauge
P 51 2.0 0.5
N 41 20.0
"""


def test_a_file_reads_and_writes_back_unchanged(tmp_path):
    path = tmp_path / "ctrl.txt"
    path.write_text(CTRL)
    ctrl = ControlFile.read(path)
    assert ctrl.exists and ctrl.program == "reef3d"
    assert ctrl.codes() == ["A 10", "F 60", "P 51", "N 41"]
    assert ctrl.int("A 10") == 6 and ctrl.value("F 60") == 0.5
    assert ctrl.all("P 51") == [[1.0, 0.5], [2.0, 0.5]]
    assert ctrl.values("P 51", 1) == [2.0, 0.5] and ctrl.values("P 51", 2) is None
    assert ctrl.text() == CTRL and not ctrl.modified
    assert ControlFile.read(tmp_path / "control.txt").exists is False


def test_edits_keep_comments_order_and_other_lines(tmp_path):
    path = tmp_path / "ctrl.txt"
    path.write_text(CTRL)
    ctrl = ControlFile.read(path)
    ctrl.set("P 51", [1.5, 0.5])  # the first gauge moves, its comment stays
    ctrl.add("P 51", [3.0, 0.5])  # a third, after the other two
    ctrl.set("P 30", [0.1])  # a new command: after the last P
    ctrl.set("B 90", [1])  # a new letter: at the end
    ctrl.remove("N 41")
    assert ctrl.modified and ctrl.revision == 5
    lines = ctrl.text().splitlines()
    assert lines[4] == "P 51 1.5 0.5    // first gauge"
    assert lines[6:9] == ["P 51 3.0 0.5", "P 30 0.1", "B 90 1"]
    assert "N 41" not in ctrl.text() and lines[0] == "// a wave flume"
    ctrl.write()
    assert not ctrl.modified and ControlFile.read(path).all("P 51")[2] == [3.0, 0.5]
    assert ctrl.remove("P 51") == 3 and "P 51" not in ctrl


def test_values_are_written_as_the_command_wants(tmp_path):
    control = ControlFile(tmp_path / "control.txt")
    control.set("B 2", [40, 1.0, 10.4])  # cells: integers
    control.set("B 10", [0, 20, 0, 1, 0, 1.25])  # the domain: decimals
    assert control.text() == "B 2 40 1 10\nB 10 0.0 20.0 0.0 1.0 0.0 1.25\n"
    assert format_value(0.1 + 0.2) == "0.3" and format_value(1e-7) == "1e-07"


def test_the_catalogues_know_both_programs():
    reef3d, divemesh = Catalog.load("reef3d"), Catalog.load("divemesh")
    assert len(reef3d) > 800 and len(divemesh) > 150
    gauge = reef3d.get("P 51")
    assert gauge.title == "Wave gauge" and gauge.repeat and gauge.count == 2
    assert gauge.names() == ["x", "y"] and gauge.section == "Printer"
    model = reef3d.get("A 10")
    assert model.options["5"] == "NHFLOW" and model.default == ("6",)
    domain = divemesh.get("B 10")
    assert domain.count == 6 and domain.names()[0] == "x_start"
    assert divemesh.get("S 84").count == 24  # 8 corners
    assert catalog_for("control.txt") is divemesh
    assert reef3d.search("wave gauge")[0].code in ("P 50", "P 51")
    assert reef3d.search("P51")[0].code == "P 51"
    assert all(c.used_by("SFLOW") for c in reef3d.search("probe", "SFLOW"))
    assert "P 61" not in [c.code for c in reef3d.search("probe", "SFLOW")]  # CFD only


def test_check_finds_what_the_programs_stumble_over(tmp_path):
    ctrl = ControlFile(
        tmp_path / "ctrl.txt",
        "A 10 6\nA 10 5\nP 51 1.0\nP 99999 1\nN 41 x\nThis line has no slashes\nP 52 0.5 // ok\n",
    )
    problems = catalog_for("ctrl.txt").check(ctrl, "CFD")
    found = {(p.line, p.level) for p in problems}
    assert (2, "warning") in found  # A 10 again: the last counts
    assert (3, "error") in found  # P 51 needs 2 values
    assert (4, "warning") in found  # not a command
    assert (5, "error") in found  # not a number
    assert (6, "error") in found  # text with a capital letter
    assert not [p for p in problems if p.line == 7]
