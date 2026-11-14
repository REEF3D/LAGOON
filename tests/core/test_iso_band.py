# LAGOON — iso-surfaces of large volumes: read near the surface, cut in processes
# Architect: Hans Bihs
"""A CFD free surface lies in a few of the volume's levels: a LAGOON store's volume
reads the level set, finds the levels it crosses (down to where the water-side
colours look) and reads the other arrays only there. The surface is the one the
whole volume gives. Large volumes are cut in processes of their own, with the
same result."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
from lagoon_format.store import StoreWriter
from scipy.spatial import cKDTree

from lagoonviz.core.data import cutters
from lagoonviz.core.data.lagoon import StoreDataset
from lagoonviz.core.data.reef3d import Dataset, open_case

LEVEL = 1.3


def _surface(x, y, t):
    return LEVEL + 0.15 * np.sin(x[None, :] / 1.5 + t) * np.cos(y[:, None] / 2.0)


def _cfd_store(root, frames: int = 2, rise: float = 0.0, drop: int | None = None):
    """A CFD store: a cartesian volume of 30 levels in two blocks, phi the distance
    to a wavy free surface (positive in the water), velocity and pressure. ``rise``:
    the surface rises so much from frame to frame; ``drop``: the frame with a drop of
    water flying high above the surface."""
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
            top = _surface(bx, y, t) + rise * t  # (ny, nx)
            depth = top[None] - z[:, None, None]  # (nz, ny, nx)
            if t == drop:  # a ball of water of 0.25 m at (3, 1.5, 2.6)
                gz, gy, gx = np.meshgrid(z, y, bx, indexing="ij")
                ball = 0.25 - np.sqrt((gx - 3.0) ** 2 + (gy - 1.5) ** 2 + (gz - 2.6) ** 2)
                depth = np.maximum(depth, ball)
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


def test_near_the_surface_follows_it(tmp_path):
    """Searched near the surface: only the levels round where it crossed in the frame
    read before are read; a surface that rose beyond them is followed (more is read
    where it reaches the edge), in any order of the frames; the surface is the one
    the whole volume gives."""
    root = _cfd_store(tmp_path / "cfd", frames=6, rise=0.25)  # 2.5 levels a frame
    whole = open_case(root).datasets["cfd_volume"]
    near = open_case(root).datasets["cfd_volume"]
    names = ["velocity", "pressure"]
    reads = []
    original = near.output.read_block

    def counted(name, t, block):
        reads.append((name, t))
        return original(name, t, block)

    near.output.read_block = counted
    for k in (0, 1, 2, 5, 3, 0, 4):
        for side in (False, True):
            a = whole.read_iso(k, "phi", 0.0, names, water_side=side)
            b = near.read_iso(k, "phi", 0.0, names, water_side=side, near_surface=True)
            _same(a, b)
    # whole blocks for the first frame and after the jumps (2 -> 5, 3 -> 0, 0 -> 4: the
    # surface 7.5 levels or more away, beyond what was read: found in the whole block
    # then); a frame next to the one before is read near the surface
    assert {t for _, t in reads} == {0, 5, 4}
    band = near._bands[(0, "phi", 0.0, 0.0)]
    assert band[2] == 30 and band[1] - band[0] <= 5


def test_near_the_surface_misses_a_drop_far_from_it(tmp_path):
    """What searching near the surface leaves out: a drop flying high above it is
    found in the whole volume only."""
    root = _cfd_store(tmp_path / "cfd", frames=2, drop=1)
    near = open_case(root).datasets["cfd_volume"]
    whole = near.read_iso(1, "phi", 0.0, ["velocity"])
    near.read_iso(0, "phi", 0.0, ["velocity"], near_surface=True)  # the band of the frame before
    found = near.read_iso(1, "phi", 0.0, ["velocity"], near_surface=True)
    high = whole.points[:, 2] > 2.2
    assert high.sum() > 20  # the drop
    assert not (found.points[:, 2] > 2.2).any()
    assert found.n_points == whole.n_points - high.sum()


def test_the_free_surface_item_searches_near_it(tmp_path):
    from lagoonviz.core import Session

    root = _cfd_store(tmp_path / "cfd")
    s = Session()
    s.open_case(root)
    free = next(i for i in s.scene if i.kind == "iso_surface")
    assert free.properties.get("search", "near the surface") == "near the surface"
    assert s.data.frame_key(free.id).extra[5] is True
    s.set_property(free.id, "search", "whole volume")
    assert s.data.frame_key(free.id).extra[5] is False


def test_coarser_while_the_time_moves(tmp_path, monkeypatch):
    """While the time moves, the iso-surface of a large volume is cut from every 2nd
    (3rd, ...) grid line across, all levels of its band: the surface where the
    whole one is, a fraction of its points, no gap between the blocks; at full
    detail once the time stands still."""
    from lagoonviz.core import Session
    from lagoonviz.core.data import manager

    root = _cfd_store(tmp_path / "cfd", frames=3)
    volume = open_case(root).datasets["cfd_volume"]
    full = volume.read_iso(1, "phi", 0.0, ["velocity"], water_side=True)
    coarse = volume.read_iso(1, "phi", 0.0, ["velocity"], water_side=True, stride=2)
    assert 0 < coarse.n_points < 0.6 * full.n_points
    gap, _ = cKDTree(full.points).query(coarse.points)
    assert gap.max() < 0.25  # within a cell (0.25 m across) of the whole surface
    assert _open_edges(coarse) <= _open_edges(full)  # its rim only: the blocks join
    s = Session()
    s.open_case(root)
    free = next(i for i in s.scene if i.kind == "iso_surface")
    monkeypatch.setattr(manager, "ISO_PREVIEW_POINTS", 500)  # the volume counts as large
    assert s.data.frame_key(free.id).detail == 1  # the time stands still
    s.data.moving = True
    key = s.data.frame_key(free.id)
    assert key.detail >= 2 and s.data.coarse
    assert s.data._read_key(key).n_points < s.data._read_key(replace(key, detail=1)).n_points
    s.data.moving = False
    assert s.data.frame_key(free.id).detail == 1


def _open_edges(mesh) -> int:
    """Edges of a surface that only one triangle has (its rim and its gaps)."""
    points = np.round(mesh.points.astype(np.float64), 5)
    _, ids = np.unique(points, axis=0, return_inverse=True)
    t = ids.ravel()[mesh.triangles.astype(np.int64)]
    edges = np.sort(np.concatenate([t[:, [0, 1]], t[:, [1, 2]], t[:, [2, 0]]]), axis=1)
    _, count = np.unique(edges, axis=0, return_counts=True)
    return int((count == 1).sum())


def test_the_cells_of_a_band_are_kept_once(tmp_path):
    """Bands of any depth share one list of cells per block (the first cells of the
    deepest band cut so far): a list was kept for every depth a band ever had, tens
    of MB each in a large volume and never dropped."""
    root = _cfd_store(tmp_path / "cfd", frames=4, rise=0.1)
    volume = open_case(root).datasets["cfd_volume"]
    surfaces = []
    for extra in (0.0, 0.3, 0.9, 0.5):  # bands of several depths
        for frame in range(4):
            cut = volume._iso_piece(0, volume.frames[frame].pieces[0], "phi", 0.0, [], extra)
            piece, connectivity, offsets, types = cut
            nz = piece.band[1] - piece.band[0]
            assert len(offsets) == len(types) == (nz - 1) * 8 * 12
            assert int(connectivity.max()) == len(piece.points) - 1
            surfaces.append(volume.read_iso(frame, "phi", 0.0, ["pressure"], water_side=True))
    kept = [key for key in volume._cells_cache if key[0] == "band"]
    assert len(kept) == 2 == len(volume._cells_cache)  # one per block
    again = open_case(root).datasets["cfd_volume"]  # the same surfaces from new lists
    _same(surfaces[-1], again.read_iso(3, "phi", 0.0, ["pressure"], water_side=True))


def test_the_same_edges_once_packed_or_not():
    """The cut points on the same edge are found by one sort of edge and position
    packed into a number, or (where they do not fit) as before: the same either way."""
    from lagoonviz.core.data.mesh import _same_edges

    rng = np.random.default_rng(5)

    def as_before(a, b):
        pairs = np.stack([np.minimum(a, b), np.maximum(a, b)], axis=1)
        _, first, inverse = np.unique(pairs, axis=0, return_index=True, return_inverse=True)
        return first, inverse.ravel()

    for points, far in ((500, 40), (2**40, 2**21), (2**40, 40), (1, 1)):
        a = rng.integers(0, points, 3000)
        b = a + rng.integers(0, far + 1, 3000)
        swap = rng.random(3000) < 0.5
        a, b = np.where(swap, b, a), np.where(swap, a, b)
        a, b = np.concatenate([a, b[:700]]), np.concatenate([b, a[:700]])  # some twice
        first, inverse = _same_edges(a.copy(), b.copy())
        expected = as_before(a, b)
        np.testing.assert_array_equal(first, expected[0])
        np.testing.assert_array_equal(inverse, expected[1])


def test_a_horizontal_slice_reads_two_levels(tmp_path, monkeypatch):
    """A horizontal slice of a Cartesian volume in a store: the values between the
    two levels round the plane on the grid's points, as cutting the hexahedra gives
    on their upright edges, with only those levels read."""
    root = _cfd_store(tmp_path / "cfd")
    volume = open_case(root).datasets["cfd_volume"]
    names = ["velocity", "pressure", "phi"]
    at = 1.37
    read = []
    get = type(volume.output.array(0, "phi")).get
    monkeypatch.setattr(
        type(volume.output.array(0, "phi")),
        "get",
        lambda self, key, parallel=True: read.append(key) or get(self, key, parallel),
    )
    flat = volume.read_slice(1, 2, at, names)
    assert read and all(key[1] == slice(13, 15) for key in read)  # 30 levels over 3 m
    monkeypatch.undo()
    assert flat.n_points == 2 * 13 * 9 and flat.n_triangles == 2 * 2 * 12 * 8
    assert np.all(flat.points[:, 2] == np.float32(at)) and flat.normals[:, 2].min() > 0.99
    assert flat.outlines is not None and len(flat.outlines)
    cut = Dataset.read_slice(volume, 1, 2, at, names)  # the hexahedra cut
    tree = cKDTree(cut.points)
    distance, nearest = tree.query(flat.points)
    assert distance.max() < 1e-5  # every grid point is a point of the cut
    for name in names:
        np.testing.assert_allclose(
            flat.point_data[name], cut.point_data[name][nearest], rtol=1e-5, atol=1e-3
        )
    # outside the levels: nothing, as before
    assert volume.read_slice(1, 2, 9.0, names).n_points == 0
