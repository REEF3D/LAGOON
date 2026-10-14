"""Large cases: lean surfaces from LAGOON stores, constant fields read once, colour
ranges from samples, streaming at a coarser detail while playing, reading ahead
within the frame budget, switched-off items not read, AMR with many patches."""

import math

import numpy as np
import pytest
from lagoon_format.store import Store, StoreWriter

from lagoonviz.core import Session, memory
from lagoonviz.core.data import lagoon
from lagoonviz.core.data.manager import BackgroundLoader, FrameCache, FrameKey
from lagoonviz.core.data.mesh import grid_normals, grid_points, grid_triangles

NX, NY, FRAMES = 91, 67, 5


@pytest.fixture
def restore():
    before = memory.mode()
    yield
    memory.set_mode(before)


@pytest.fixture
def store(tmp_path):
    """An FNPF-like store: a free surface on four blocks over a bed, its still water
    depth the same in every output."""
    path = tmp_path / "case" / "REEF3D_FNPF.lagoon"
    path.parent.mkdir()
    run = {"run": "r1", "started": "2026-10-14T06:00:00Z", "solver": "FNPF"}
    writer = StoreWriter.create(path, solver="FNPF", run=run)
    x = np.linspace(0.0, 90.0, NX)
    y = np.linspace(0.0, 66.0, NY)
    blocks = [((0, 0), (46, 34)), ((45, 0), (46, 34)), ((0, 33), (46, 34)), ((45, 33), (46, 34))]
    surface = writer.add_output(
        "free_surface", "surface", x, y, {"eta": 1, "depth": 1, "velocity": 3}, blocks
    )
    bed = writer.add_output("bed", "surface", x, y, {"depth": 1}, blocks)
    gx, gy = np.meshgrid(x, y)
    bottom = (-5.0 + 0.1 * gx).astype(np.float32)
    for frame in range(FRAMES):
        eta = (0.5 * np.sin(gx / 7.0 - frame) * np.cos(gy / 9.0)).astype(np.float32)
        fields = {
            "z": eta,
            "eta": eta,
            "depth": -bottom,
            "velocity": np.stack([np.cos(gx / 7.0 - frame), 0 * gx, 0 * gx], -1).astype(np.float32),
        }
        surface.append(float(frame), frame, [_part(fields, b) for b in blocks])
        if frame == 0:
            bed.append(0.0, 0, [_part({"z": bottom, "depth": -bottom}, b) for b in blocks])
    writer.close()
    return path


def _part(fields, block):
    (i0, j0), (w, h) = block
    return {name: v[j0 : j0 + h, i0 : i0 + w] for name, v in fields.items()}


def _surface(store_path):
    s = Session()
    s.open_case(store_path)
    item = next(i for i in s.scene if i.kind == "free_surface")
    return s, item, s.data.dataset(item.id)


# ----------------------------------------------------------------- grids
def test_grid_helpers_match_the_slow_ways():
    x = np.linspace(0.0, 30.0, 31)
    y = np.cumsum(np.linspace(1.0, 2.0, 21))  # uneven rows
    gx, gy = np.meshgrid(x, y)
    z = (np.sin(gx / 5.0) + 0.1 * gy).astype(np.float32)
    z[4:6, 7:9] = np.nan
    # normals: as numpy's gradient made them (to rounding), dry points straight up
    dz_dy, dz_dx = np.gradient(z.astype(np.float64), y, x)
    slow = np.stack([-dz_dx, -dz_dy, np.ones_like(dz_dx)], -1)
    slow /= np.linalg.norm(slow, axis=-1, keepdims=True)
    fast = grid_normals(z, x, y)
    good = np.isfinite(slow).all(axis=-1)
    assert np.allclose(fast[good], slow[good], atol=1e-5)
    assert np.allclose(fast[~good], (0.0, 0.0, 1.0))
    # a large grid is worked on in bands on several threads: the same result
    big = np.tile(z, (60, 40))
    bx, by = np.arange(big.shape[1], dtype=float), np.arange(big.shape[0], dtype=float)
    banded = grid_normals(big, bx, by)
    from lagoonviz.core.data import mesh

    whole = np.empty_like(banded)
    inv_x = np.full(len(bx), 0.5, np.float32)
    inv_x[[0, -1]] = 1.0
    inv_y = np.full(len(by), 0.5, np.float32)
    inv_y[[0, -1]] = 1.0
    mesh._normals_band(big, inv_x, inv_y, whole, 0, big.shape[0])
    assert big.size >= mesh.BAND_POINTS
    np.testing.assert_array_equal(banded, whole)
    # points and triangles: the grid row by row, two triangles per whole cell
    points = grid_points(x, y, z)
    assert points.shape == (z.size, 3) and points.dtype == np.float32
    np.testing.assert_allclose(points[:, 0], gx.ravel())
    np.testing.assert_allclose(points[:, 1], gy.ravel(), rtol=1e-6)
    triangles = grid_triangles(*z.shape, ~np.isfinite(z))
    assert triangles.dtype == np.uint32
    assert np.isfinite(points[triangles]).all()
    whole_cells = (z.shape[0] - 1) * (z.shape[1] - 1)
    assert len(triangles) == 2 * (whole_cells - 9)  # 2 x 2 points gone: 9 cells
    a, b, c = (points[triangles[:, i]] for i in range(3))
    assert (np.cross(b - a, c - a)[:, 2] > 0).all()  # all facing up


# ----------------------------------------------------------------- stores
def test_constant_fields_are_read_once(store):
    s, item, dataset = _surface(store)
    first = s.data.load(item.id)
    s.set_time(2.0)
    second = s.data.load(item.id)
    assert first.point_data["depth"] is second.point_data["depth"]  # read once
    assert not second.point_data["depth"].flags.writeable
    assert not np.array_equal(first.point_data["eta"], second.point_data["eta"])
    # the dry-land triangles follow it without being worked out again
    shown = s.data.display(item.id, first)
    assert s.data.display(item.id, second).triangles is shown.triangles


def test_colour_range_samples_a_large_surface(store, monkeypatch):
    output = Store(store).outputs["free_surface"]
    sample = output.sample("eta", 3, chunks=8)
    full = output.read("eta", 3)
    assert sample.size and np.isin(sample, full).all()  # (small blocks: one tile each)
    vectors = output.sample("velocity", 3, chunks=8)
    assert vectors.ndim == 2 and vectors.shape[1] == 3
    s, item, dataset = _surface(store)
    monkeypatch.setattr(lagoon, "SAMPLE_ALL", 100)  # "large"
    values = dataset.sample_values(3, "eta")
    assert values.size == output.sample("eta", 3).size != NX * NY
    monkeypatch.setattr(lagoon, "SAMPLE_ALL", 10**9)
    assert dataset.sample_values(3, "eta").size == NX * NY


def test_streaming_shows_large_surfaces_coarser_while_playing(store, restore, monkeypatch):
    s, item, dataset = _surface(store)
    bed = next(i for i in s.scene if i.kind == "bathymetry")
    memory.set_mode("streaming")
    assert memory.current().detail_points
    full = s.data.load(item.id)
    s.data.moving = True  # playing
    assert s.data.frame_key(item.id).detail == 1  # under 2.5 million points: as it is
    monkeypatch.setattr(memory, "STREAM_POINTS", 1000)  # "large"
    stride = dataset.detail_stride(1000)
    assert stride == math.ceil(math.sqrt(NX * NY / 1000))
    key = s.data.frame_key(item.id)
    assert key.detail == stride
    assert s.data.frame_key(bed.id).detail == 1  # one output: as it is
    coarse = s.data.load(item.id)
    rows = len(range(0, NY, stride)) + ((NY - 1) % stride != 0)
    cols = len(range(0, NX, stride)) + ((NX - 1) % stride != 0)
    assert len(coarse.points) == rows * cols
    # the same extent: the last row and column always kept
    np.testing.assert_allclose(coarse.points.min(axis=0)[:2], full.points.min(axis=0)[:2])
    np.testing.assert_allclose(coarse.points.max(axis=0)[:2], full.points.max(axis=0)[:2])
    assert coarse.triangles.max() < len(coarse.points)
    assert coarse.point_data["eta"].shape == (rows * cols,)
    s.data.display(item.id, coarse)  # shows
    s.data.moving = False  # stands still: full detail
    assert s.data.frame_key(item.id).detail == 1
    memory.set_mode("balanced")
    s.data.moving = True
    assert s.data.frame_key(item.id).detail == 1  # only when streaming


def test_the_four_modes():
    fast, middle, small, stream = (memory.profile(m) for m in memory.MODES)
    assert fast.play_all and not middle.play_all
    assert stream.detail_points and not fast.detail_points
    assert stream.cache_bytes < middle.cache_bytes
    assert "coarser while playing" in stream.describe()
    assert "the play range" in fast.describe()


def test_reading_ahead_stays_within_the_budget(store):
    s, item, dataset = _surface(store)
    s.data.cache = FrameCache(max_bytes=1)  # nothing fits
    loader = BackgroundLoader(s.data, workers=1)
    try:
        s.data.load(item.id)  # the size of a frame is known now
        size = s.data.cache.size_hint(s.data.frame_key(item.id))
        assert size
        times = s.data.timeline()[0]
        assert loader.prefetch([item.id], 0.0, frames_ahead=4, times=times) == 1
        s.data.cache.max_bytes = 5 * size
        assert loader.prefetch([item.id], 0.0, frames_ahead=4, times=times) == 3
        s.data.cache.max_bytes = 100 * size
        assert loader.prefetch([item.id], 0.0, frames_ahead=4, times=times) == 4
    finally:
        loader.shutdown()


def test_switched_off_items_are_not_read(reef3d_run):
    s = Session()
    s.open_case(reef3d_run)
    volume = next(i for i in s.scene if i.kind == "volume")
    water = next(i for i in s.scene if i.kind == "free_surface")
    assert not volume.visible  # opened switched off
    assert volume.id not in s.data.shown_items() and water.id in s.data.shown_items()
    case = next(i for i in s.scene if i.kind == "case")
    s.set_visible(case.id, False)
    assert not s.data.shown_items()


def test_film_reads_only_what_is_shown(reef3d_run, monkeypatch, tmp_path):
    from lagoonviz.render import movie

    pytest.importorskip("pygfx")
    s = Session()
    s.open_case(reef3d_run)
    volume = next(i for i in s.scene if i.kind == "volume")
    read = []
    original = s.data.load_key
    monkeypatch.setattr(s.data, "load_key", lambda key: read.append(key) or original(key))
    movie.write_movie(s, tmp_path / "f", times=[0.0, 1.0], size=(64, 48))
    assert read and all(k.dataset != s.data.dataset(volume.id).key for k in read)


# ----------------------------------------------------------------- AMR
def test_many_patches_find_what_covers_them():
    from lagoon_format.amr import Boxes, covered

    from lagoonviz.core.data.amr import AmrGrid, leaf_mesh

    rng = np.random.default_rng(3)
    base = AmrGrid(0, np.linspace(0, 100, 51), np.linspace(0, 60, 31))
    base.cell_data = {"eta": np.zeros((30, 50), np.float32)}
    grids = [base]
    for k in range(60):
        x0, y0 = rng.uniform(0, 90), rng.uniform(0, 50)
        level = 1 + k % 3
        patch = AmrGrid(level, x0 + np.arange(9) * 0.5**level, y0 + np.arange(7) * 0.5**level)
        patch.cell_data = {"eta": np.full((6, 8), float(level), np.float32)}
        grids.append(patch)
    boxes = Boxes(grids)

    def brute(grid):  # as covered() worked before: every grid looked at
        xc, yc = 0.5 * (grid.x[:-1] + grid.x[1:]), 0.5 * (grid.y[:-1] + grid.y[1:])
        mask = np.zeros((len(yc), len(xc)), bool)
        for other in grids:
            if other.level > grid.level:
                i0, i1 = np.searchsorted(xc, other.x[0]), np.searchsorted(xc, other.x[-1], "right")
                j0, j1 = np.searchsorted(yc, other.y[0]), np.searchsorted(yc, other.y[-1], "right")
                mask[j0:j1, i0:i1] |= i0 < i1 and j0 < j1
        return mask

    for grid in grids:
        np.testing.assert_array_equal(covered(grid, grids, boxes), brute(grid))
        np.testing.assert_array_equal(covered(grid, grids), brute(grid))
    mesh = leaf_mesh(grids)
    assert mesh.normals is not None and mesh.normals.shape == mesh.points.shape
    assert np.allclose(np.linalg.norm(mesh.normals, axis=1), 1.0, atol=1e-5)


def test_arrays_read_alike_on_one_thread_or_many(store):
    output = Store(store).outputs["free_surface"]
    array = output.array(0, "eta")
    np.testing.assert_array_equal(array.get(2, parallel=False), array[2])
    np.testing.assert_array_equal(output.read("eta", 2, threads=1), output.read("eta", 2))


def test_frame_keys_tell_their_detail_apart():
    a = FrameKey("c", "d", 1, ("eta",))
    assert a.detail == 1 and a != FrameKey("c", "d", 1, ("eta",), (), 3)
