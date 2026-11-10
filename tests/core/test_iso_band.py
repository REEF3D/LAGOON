# LAGOON — iso-surfaces of large volumes: read near the surface, cut in processes
# Architect: Hans Bihs
"""A CFD free surface lies in a few of the volume's levels: a LAGOON store's volume
reads the level set, finds the levels it crosses (down to where the water-side
colours look) and reads the other arrays only there. The surface is the one the
whole volume gives. Large volumes are cut in processes of their own, with the
same result."""

from __future__ import annotations

import numpy as np
import pytest
from lagoon_format.store import StoreWriter

from lagoonviz.core.data import cutters
from lagoonviz.core.data.lagoon import StoreDataset
from lagoonviz.core.data.reef3d import Dataset, open_case

LEVEL = 1.3


def _surface(x, y, t):
    return LEVEL + 0.15 * np.sin(x[None, :] / 1.5 + t) * np.cos(y[:, None] / 2.0)


def _cfd_store(root, frames: int = 2):
    """A CFD store: a cartesian volume of 30 levels in two blocks, phi the distance
    to a wavy free surface (positive in the water), velocity and pressure."""
    root.mkdir()
    (root / "ctrl.txt").write_text("N 40 3\nF 45 1.6\n")
    x, y = np.linspace(0.0, 6.0, 25), np.linspace(0.0, 3.0, 9)
    z = np.linspace(0.0, 3.0, 30)
    blocks = [((0, 0), (13, 9)), ((12, 0), (13, 9))]
    writer = StoreWriter.create(root / "REEF3D_CFD.lagoon", solver="CFD", run={"run": "R1"})
    volume = writer.add_output(
        "volume",
        "cartesian",
        x,
        y,
        {"phi": 1, "velocity": 3, "pressure": 1},
        blocks,
        z=z,
        shard_time=2,
        tile=8,
    )
    for t in range(frames):
        values = []
        for (i0, _), (nx, _) in blocks:
            bx = x[i0 : i0 + nx]
            top = _surface(bx, y, t)  # (ny, nx)
            depth = top[None] - z[:, None, None]  # (nz, ny, nx)
            u = np.zeros((*depth.shape, 3))
            u[..., 0] = 0.3 + 0.1 * z[:, None, None] + 0.05 * bx[None, None, :]
            u[..., 2] = 0.02 * np.cos(bx)[None, None, :]
            values.append(
                {
                    "phi": depth.astype(np.float32),
                    "velocity": u.astype(np.float32),
                    "pressure": (9810.0 * depth).astype(np.float32),
                }
            )
        volume.append(0.5 * t, t, values)
    writer.close()
    return root


def _whole(monkeypatch):
    """Read every block whole, as before the band reads."""
    monkeypatch.setattr(StoreDataset, "_iso_piece", Dataset._iso_piece)
    monkeypatch.setattr(StoreDataset, "_water_margin", lambda self: None)


def _same(a, b):
    np.testing.assert_array_equal(a.points, b.points)
    np.testing.assert_array_equal(a.triangles, b.triangles)
    assert sorted(a.point_data) == sorted(b.point_data)
    for name in a.point_data:
        np.testing.assert_allclose(a.point_data[name], b.point_data[name], rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize("water_side", [False, True])
def test_the_band_read_gives_the_whole_volumes_surface(tmp_path, monkeypatch, water_side):
    root = _cfd_store(tmp_path / "cfd")
    volume = open_case(root).datasets["cfd_volume"]
    assert isinstance(volume, StoreDataset) and volume.output.grid == "cartesian"
    names = ["velocity", "pressure"]
    band = volume.read_iso(1, "phi", 0.0, names, water_side=water_side)
    # read only near the surface: a few of the 30 levels
    piece = volume._iso_piece(0, volume.frames[1].pieces[0], "phi", 0.0, names)[0]
    k0, k1 = piece.band
    assert 0 < k0 and k1 - k0 <= 4 and len(piece.points) == (k1 - k0) * 9 * 13
    deep = volume._iso_piece(0, volume.frames[1].pieces[0], "phi", 0.0, names, 0.6)[0]
    assert deep.band[0] < k0 and deep.band[1] == k1  # down to phi = 0.6 too
    assert band.n_triangles and abs(float(np.median(band.points[:, 2])) - LEVEL) < 0.1
    _whole(monkeypatch)
    whole = volume.read_iso(1, "phi", 0.0, names, water_side=water_side)
    _same(band, whole)
    if water_side:  # the colours of the water under the band, not its own
        plain = volume.read_iso(1, "phi", 0.0, names)
        assert not np.allclose(whole.point_data["pressure"], plain.point_data["pressure"])


def test_the_water_side_reads_deep_enough(tmp_path):
    """The band holds the levels the water-side colours are cut at (F 45 of 1.6 and
    0.3 cells, twice, at the tallest cells, and a quarter more)."""
    root = _cfd_store(tmp_path / "cfd", frames=1)
    volume = open_case(root).datasets["cfd_volume"]
    height = float(np.max(np.diff(volume.output.z)))
    assert volume._water_margin() == pytest.approx(2.5 * (1.6 + 0.3) * height)


def test_no_crossing_no_piece(tmp_path):
    root = _cfd_store(tmp_path / "cfd", frames=1)
    volume = open_case(root).datasets["cfd_volume"]
    path = volume.frames[0].pieces[0]
    assert volume._iso_piece(0, path, "phi", 5.0, ["velocity"]) is None  # above the domain
    empty = volume.read_iso(0, "phi", 5.0, ["velocity"])
    assert empty.n_points == 0 and empty.n_triangles == 0


def test_large_volumes_are_cut_in_processes(tmp_path, monkeypatch):
    """The data manager cuts a large volume's iso-surface in a process of its own,
    the same surface; a frame the process cannot cut is cut in the thread."""
    from lagoonviz.core import Session

    root = _cfd_store(tmp_path / "cfd")
    monkeypatch.setenv("LAGOON_CUT_PROCESSES", "0")
    s = Session()
    s.open_case(root)
    free = next(i for i in s.scene if i.kind == "iso_surface")
    key = s.data.frame_key(free.id)
    here = s.data._read_key(key)
    monkeypatch.setenv("LAGOON_CUT_PROCESSES", "2")
    monkeypatch.setattr(cutters, "MIN_POINTS", 1)
    try:
        options = {"grid_lines": bool(key.extra[3]), "water_side": bool(key.extra[4])}
        apart = cutters.cut(key.case, key.dataset, "read_iso", *_iso_args(key), **options)
        assert apart is not None and apart.normals is not None
        _same(apart, here)
        through = s.data._read_key(key)  # the manager's way
        _same(through, here)
        # a frame the process's case does not have (yet): cut here
        assert cutters.cut(key.case, key.dataset, "read_iso", 99, "phi", 0.0, []) is None
        assert cutters.cut(key.case, "nothing", "read_iso", 0, "phi", 0.0, []) is None
    finally:
        cutters.shutdown()


def _iso_args(key):
    return (key.index, key.extra[1], key.extra[2], list(key.arrays))


def test_how_many_processes(monkeypatch):
    from lagoonviz.core import memory

    monkeypatch.setenv("LAGOON_CUT_PROCESSES", "3")
    assert cutters.processes() == 3
    monkeypatch.delenv("LAGOON_CUT_PROCESSES")
    monkeypatch.setattr("os.cpu_count", lambda: 10)
    memory.set_mode("balanced")
    try:
        assert cutters.processes() == memory.current().loader_workers
        memory.set_mode("low memory")
        assert cutters.processes() == 0
        monkeypatch.setattr("os.cpu_count", lambda: 2)
        memory.set_mode("balanced")
        assert cutters.processes() == 0  # one process: the threads do as well
    finally:
        memory.set_mode("balanced")
