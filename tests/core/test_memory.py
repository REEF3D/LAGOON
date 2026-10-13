"""Speed or memory: the memory use modes."""

import numpy as np
import pytest

from lagoonviz.core import memory
from lagoonviz.core.data.manager import FrameCache, FrameKey
from lagoonviz.core.data.mesh import SurfaceMesh


@pytest.fixture
def restore():
    before = memory.mode()
    yield
    memory.set_mode(before)


def _mesh(n: int) -> SurfaceMesh:
    return SurfaceMesh(np.zeros((n, 3), np.float32), np.zeros((0, 3), np.uint32), {})


def test_the_three_modes():
    fast, middle, small = (memory.profile(m) for m in memory.MODES)
    assert fast.cache_bytes >= 2 * memory.GB > middle.cache_bytes == memory.GB > small.cache_bytes
    assert fast.play_ahead > middle.play_ahead > small.play_ahead
    assert fast.loader_workers >= middle.loader_workers > small.loader_workers
    assert fast.keep_cells and middle.keep_cells and not small.keep_cells
    assert "GB of frames" in small.describe()
    with pytest.raises(ValueError, match="memory use"):
        memory.profile("huge")


def test_the_frame_cache_follows_the_mode(restore):
    memory.set_mode("balanced")
    cache = FrameCache()  # no budget of its own: the mode's
    assert cache.max_bytes == memory.GB
    for i in range(30):  # 30 frames of 12 MB: 360 MB, all kept
        cache.put(FrameKey("c", "d", i, ()), _mesh(1_000_000))
    assert len(cache) == 30
    memory.set_mode("low memory")
    cache.trim()
    assert cache.nbytes <= 256 << 20 and len(cache) == 22  # the oldest went first
    assert FrameKey("c", "d", 29, ()) in cache and FrameKey("c", "d", 0, ()) not in cache
    assert FrameCache(max_bytes=1).max_bytes == 1  # a budget of its own stays


def test_low_memory_reads_a_volumes_cells_again(reef3d_run, restore):
    from lagoonviz.core import Session

    s = Session()
    s.open_case(reef3d_run)
    item = s.add_slice("y", 2.0)
    memory.set_mode("balanced")
    s.data.load(item)
    dataset = s.data.dataset(item)
    assert dataset._cells  # kept for the next frame
    memory.set_mode("low memory")
    s.set_time(1)
    mesh = s.data.load(item)
    assert mesh.n_triangles and not dataset._cells  # read, used, not kept
