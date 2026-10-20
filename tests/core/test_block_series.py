"""The open faces of the blocks 1.. over time (REEF3D's G 54 types 3 and 4): a river's
discharge from GloFAS or a file as discharge-<b>.dat, a water level as
waterlevel-<b>.dat, the G 54 lines, and blocks.dat's G 54 read back."""

import json

import numpy as np
import pytest
from openmeteo_fake import FakeOpenMeteo

from lagoonviz.core import Session
from lagoonviz.core import multiblock as M
from lagoonviz.core.data.controlfile import ControlFile
from lagoonviz.core.geo import openmeteo as om
from lagoonviz.core.geo.crs import transformer

LAT, LON = 38.95, -8.85  # the Tagus near Muge


@pytest.fixture
def river(tmp_path, monkeypatch):
    monkeypatch.setattr(om, "FETCH", FakeOpenMeteo())
    folder = tmp_path / "river"
    folder.mkdir()
    (folder / "control.txt").write_text("B 1 50\nB 10 0 2000 0 1000 -20 10\nC 11 21\n")
    (folder / "ctrl.txt").write_text("A 10 5\nF 60 0.0\nM 10 2\n")
    x, y = transformer("EPSG:4326", "EPSG:32629")(LON, LAT)
    georef = {"version": 1, "crs": "EPSG:32629", "origin": [float(x), float(y)],
              "rotation_deg": 0.0}  # fmt: skip
    (folder / "georef.json").write_text(json.dumps(georef))
    s = Session()
    s.open_case(folder)
    s.start_multiblock([0, 1000, 0, 500], cells=(20, 10))
    s.extrude_block("b0_e", 500.0, cells=10)  # block 1: its -i and +i faces are free
    return s, folder


def test_a_series_file():
    text = M.series_text([0.0, 3600.0], [336.0, 2000.5], "from a test")
    assert text.splitlines() == [
        "# from a test",
        "# t (s), discharge (m3/s)",
        "0 336",
        "3600 2000.5",
    ]
    assert "water level (m)" in M.series_text([0], [1], unit="m")


def test_discharge_from_glofas_and_a_level_from_a_file(river, tmp_path):
    s, folder = river
    found = s.write_block_series(1, "2023-03-10T00:00", days=2, glofas=True)
    rows = np.loadtxt(folder / "discharge-1.dat")
    assert rows[0, 0] == 0.0 and rows[-1, 0] == 48 * 3600.0 and len(rows) == 49
    assert rows[0, 1] == pytest.approx(130.0) and rows[24, 1] == pytest.approx(140.0)
    assert (folder / "discharge-1.dat").read_text().startswith("# GloFAS river discharge")
    assert found.file == "discharge-1.dat" and "130 - 150 m3/s" in found.text()
    block = s.multiblock().blocks[1]
    assert block.bc_type("i0") == "inflow" and block.bc_series("i0") == 1
    assert block.bc_value("i0") == pytest.approx(140.0)  # the mean, shown
    assert M.settings(s.multiblock()).g54 == [[1, 3, 1]]
    assert not s.input_file("ctrl.txt").all("G 54")  # no grid yet: Make grid writes it
    # a level from a file: dates from the start, cut to the run, scaled
    levels = tmp_path / "levels.csv"
    levels.write_text("date,level\n2023-03-09T00:00,0.0\n2023-03-10T12:00,1.0\n2023-03-13,2.0\n")
    found = s.write_block_series(
        1, "2023-03-10T00:00", days=1, file=levels, outflow=True, scale=0.5
    )
    t, z = np.loadtxt(folder / "waterlevel-1.dat").T
    assert list(t) == [0.0, 43200.0, 86400.0]
    assert z == pytest.approx([0.5 * 2 / 3, 0.5, 0.5 * 1.2])
    assert found.g54 == 4 and found.unit == "m"
    s.make_multiblock_grid()
    ctrl = s.input_file("ctrl.txt")
    assert ctrl.all("G 54") == [[1, 3, 1], [1, 4, 1]]
    assert "G 54 1 3 1\n" in ctrl.text()  # the file's number an int
    assert "session.write_block_series(1, start='2023-03-10T00:00', days=2, glofas=True)" in (
        s.to_script()
    )
    # a constant again: the series goes
    s.set_block_face(1, "i0", "inflow", 300.0)
    assert s.multiblock().blocks[1].bc_series("i0") is None
    # with the grid made, the G 54 lines follow at once
    q = tmp_path / "q.txt"
    q.write_text("0 100\n7200 300\n")
    s.write_block_series(1, file=q)
    assert s.input_file("ctrl.txt").all("G 54") == [[1, 3, 1], [1, 4, 1]]
    assert np.loadtxt(folder / "discharge-1.dat")[-1, 1] == 300.0
    # undo: the face constant again, the file as it was
    s.undo()
    assert s.multiblock().blocks[1].bc_series("i0") is None
    assert np.loadtxt(folder / "discharge-1.dat")[0, 1] == pytest.approx(130.0)


def test_refused(river):
    s, _ = river
    with pytest.raises(ValueError, match="blocks 1"):
        s.write_block_series(0, file="x")
    with pytest.raises(ValueError, match="water level"):
        s.write_block_series(1, "2023-03-10", days=1, glofas=True, outflow=True)
    with pytest.raises(ValueError, match="start"):
        s.write_block_series(1, glofas=True)
    topology = s.multiblock().with_bc(0, "i0", "inflow", 10.0, series=0)
    assert any("over time only on the blocks 1" in i.text for i in M.check(topology))
    with pytest.raises(ValueError, match="inflow or outflow"):
        s.multiblock().with_bc(1, "i0", "tide", series=1)


def test_read_back_from_the_input_files():
    ctrl = ControlFile("ctrl.txt", "G 50 1\nG 54 5 3 1\nG 54 2 4 7\nG 54 3 1 336.0\n")
    bcs = M.bcs_from_inputs(None, ctrl)
    assert bcs[(5, "i0")] == {"type": "inflow", "series": 1}
    assert bcs[(2, "i1")] == {"type": "outflow", "series": 7}
    assert bcs[(3, "i0")] == {"type": "inflow", "value": 336.0}
