"""Adaptive mesh refinement surfaces: grids that change from output to output (SPEC 11)."""

from __future__ import annotations

import numpy as np
import pytest
from lagoon_format import amr as module
from lagoon_format.amr import AmrGrid
from lagoon_format.store import Store, StoreWriter


def _grids(t: int) -> list[AmrGrid]:
    """Level 0 on two ranks, patches that move and come and go (regridding)."""
    rng = np.random.default_rng(t)

    def grid(level, rank, x0, x1, y0, y1, d):
        x = np.arange(x0, x1 + 0.5 * d, d)
        y = np.arange(y0, y1 + 0.5 * d, d)
        nx, ny = len(x) - 1, len(y) - 1
        fields = {
            "eta": rng.normal(size=(ny, nx)).astype(np.float32),
            "velocity": rng.normal(size=(ny, nx, 2)).astype(np.float32),
            "wet": (rng.uniform(size=(ny, nx)) > 0.2).astype(np.int8),
        }
        return AmrGrid(level, x, y, fields, rank)

    grids = [grid(0, 0, 0.0, 4.0, 0.0, 3.0, 0.5), grid(0, 1, 4.0, 8.0, 0.0, 3.0, 0.5)]
    shift = 0.5 * (t % 4)
    grids.append(grid(1, 0, 1.0 + shift, 3.0 + shift, 0.5, 2.5, 0.25))
    if t % 3:
        grids.append(grid(2, 0, 1.5 + shift, 2.5 + shift, 1.0, 2.0, 0.125))
    if t >= 2:
        grids.append(grid(1, 1, 6.0, 7.5, 0.0, 1.5, 0.25))
    return grids


FIELDS = {"eta": (1, "float32"), "velocity": (2, "float32"), "wet": (1, "int8")}


def _same(got: list[AmrGrid], expected: list[AmrGrid], fields=None) -> None:
    assert len(got) == len(expected)
    for a, b in zip(got, expected, strict=True):
        assert (a.level, a.rank, a.nx, a.ny) == (b.level, b.rank, b.nx, b.ny)
        assert np.array_equal(a.x, b.x) and np.array_equal(a.y, b.y)
        for name in fields or b.fields:
            assert a.fields[name].dtype == b.fields[name].dtype
            assert np.array_equal(a.fields[name], b.fields[name])


@pytest.mark.parametrize("small_buffer", [False, True])
def test_an_amr_surface(tmp_path, monkeypatch, small_buffer):
    if small_buffer:  # flushes in the middle of the run
        monkeypatch.setattr(module, "BUFFER_ROWS", 200)
    path = tmp_path / "REEF3D_NHFLOW.lagoon"
    writer = StoreWriter.create(path, "NHFLOW")
    amr = writer.amr_writer(
        "nhflow_amr",
        {
            "fields": FIELDS,
            "source": "REEF3D_NHFLOW_AMR",
            "attributes": {"dataset": "nhflow_amr", "role": "amr", "refinement": 2},
        },
    )
    for t in range(9):
        amr.append(0.5 * t, t, _grids(t))
    writer.close()

    stored = Store(path).amr["nhflow_amr"]
    assert len(stored) == 9 and stored.info["refinement"] == 2
    assert stored.fields == {name: (c, np.dtype(d)) for name, (c, d) in FIELDS.items()}
    np.testing.assert_allclose(stored.times, 0.5 * np.arange(9))
    assert stored.steps.tolist() == list(range(9))
    for t in range(9):
        expected = _grids(t)
        assert stored.count(t) == len(expected)
        _same(stored.grids(t), expected)
        _same(stored.grids(t, ["eta"]), expected, ["eta"])
        assert list(stored.grids(t, ["eta"])[0].fields) == ["eta"]
        table = stored.table(t)
        assert table["level"].tolist() == [g.level for g in expected]

    # appending later goes on where it stopped
    writer = StoreWriter(path)
    amr = writer.amr_writer("nhflow_amr")
    for t in range(9, 12):
        amr.append(0.5 * t, t, _grids(t))
    writer.close()
    stored = Store(path).amr["nhflow_amr"]
    assert len(stored) == 12
    for t in (0, 8, 9, 11):
        _same(stored.grids(t), _grids(t))


def test_root_lists_amr_sets(tmp_path):
    path = tmp_path / "REEF3D_FNPF.lagoon"
    writer = StoreWriter.create(path, "FNPF")
    writer.amr_writer("fnpf_amr", {"fields": {"eta": (1, "float32")}}).append(
        0.0, 0, [AmrGrid(0, np.arange(3.0), np.arange(2.0), {"eta": np.zeros((1, 2))})]
    )
    writer.close()
    store = Store(path)
    assert store.info["amr"] == ["fnpf_amr"] and store.info["version"] == "0.4"
    assert list(store.amr) == ["fnpf_amr"]


def test_bad_grids_are_refused(tmp_path):
    writer = StoreWriter.create(tmp_path / "s.lagoon", "FNPF")
    amr = writer.amr_writer("fnpf_amr", {"fields": {"eta": (1, "float32")}})
    x, y = np.arange(4.0), np.arange(3.0)
    with pytest.raises(ValueError, match="without 'eta'"):
        amr.append(0.0, 0, [AmrGrid(0, x, y, {})])
    with pytest.raises(ValueError, match="values for 3 x 2 cells"):
        amr.append(0.0, 0, [AmrGrid(0, x, y, {"eta": np.zeros(5)})])


def test_zarr_python_reads_an_amr_set(tmp_path):
    zarr = pytest.importorskip("zarr")
    path = tmp_path / "REEF3D_SFLOW.lagoon"
    writer = StoreWriter.create(path, "SFLOW")
    amr = writer.amr_writer("sflow_amr", {"fields": FIELDS})
    for t in range(3):
        amr.append(float(t), t, _grids(t))
    writer.close()
    group = zarr.open_group(str(path / "amr" / "sflow_amr"), mode="r")
    ends = group["grid_end"][:]
    assert ends.tolist() == np.cumsum([len(_grids(t)) for t in range(3)]).tolist()
    levels = group["grids/level"][:]
    assert levels.tolist() == [g.level for t in range(3) for g in _grids(t)]
    eta = group["cell_data/eta"][:]
    first = _grids(0)[0]
    assert np.array_equal(eta[: first.nx * first.ny], first.fields["eta"].ravel())


def test_paraview_reader_amr(tmp_path):
    """The plugin's AMR grids (without ParaView): the cells finer grids cover are hidden."""
    import importlib.util
    from pathlib import Path

    plugin = Path(__file__).parents[2] / "lagoon-format" / "paraview" / "lagoon_reader.py"
    spec = importlib.util.spec_from_file_location("lagoon_reader", plugin)
    reader = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(reader)
    path = tmp_path / "REEF3D_NHFLOW.lagoon"
    writer = StoreWriter.create(path, "NHFLOW")
    amr = writer.amr_writer("nhflow_amr", {"fields": FIELDS})
    for t in range(3):
        amr.append(1.0 + t, t, _grids(t))
    writer.close()
    store = Store(path)
    assert reader.all_times(store) == [1.0, 2.0, 3.0]
    grids = reader.amr_grids(store.amr["nhflow_amr"], 1)
    assert [g.level for g, _, _ in grids] == [g.level for g in _grids(1)]
    area = 0.0
    for grid, cells, ghosts in grids:
        assert cells["level"].tolist() == [grid.level] * (grid.nx * grid.ny)
        assert cells["velocity"].shape == (grid.nx * grid.ny, 2)
        widths = np.outer(np.diff(grid.y), np.diff(grid.x)).ravel()
        area += widths[ghosts == 0].sum()
    assert area == pytest.approx(8.0 * 3.0)  # the leaf cells cover the domain once
