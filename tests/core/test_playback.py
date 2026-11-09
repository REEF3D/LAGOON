# LAGOON — playing as fast as the frames are read, steadily
# Architect: Hans Bihs
"""CFD's free surface is cut from the volume for every frame. Playing raced through
the few frames read ahead and then stood still for the next batch; the pacer plays
no faster than they come while few are ready. The cutting keeps what stays from
output to output (the cells of each piece, where the nodes lie on the lattice)."""

import numpy as np
import pytest

from lagoonviz.core.playback import Pacer


class Clock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now


def test_the_pacer_plays_as_fast_as_frames_come():
    clock = Clock()
    pacer = Pacer(window=3.0, enough=6, clock=clock)
    assert pacer.interval(1 / 12, 0) == pytest.approx(1 / 12)  # nothing known yet
    # a frame becomes ready every 0.5 s; playing shows them as they come
    for step in range(8):
        clock.now = 0.5 * step
        pacer.note(shown=step, ready_ahead=1)
    assert pacer.read_interval() == pytest.approx(0.5)
    assert pacer.interval(1 / 12, 1) == pytest.approx(0.5)
    # enough read ahead: the frame rate asked for
    assert pacer.interval(1 / 12, 6) == pytest.approx(1 / 12)
    # nothing comes: waits at most 2 s between frames
    for step in range(8, 14):
        clock.now = 0.5 * step
        pacer.note(shown=8, ready_ahead=0)
    assert pacer.interval(1 / 12, 0) == pytest.approx(2.0)
    pacer.reset()
    assert pacer.read_interval() is None


def test_ready_frames_ahead_are_counted_in_order():
    from lagoonviz.gui.panels.timeline import _ready_count

    class Data:
        def __init__(self, ready):
            self.cache = set(ready)

        def frame_key(self, item, when):
            return (item, when)

    times = np.arange(10.0)
    data = Data({("a", 3.0), ("b", 3.0), ("a", 4.0), ("b", 4.0), ("a", 6.0), ("b", 6.0)})
    assert _ready_count(data, ["a", "b"], times, 3, 6) == 2  # 3 and 4; 5 is missing
    assert _ready_count(data, ["a", "b"], times, 5, 6) == 0
    assert _ready_count(data, ["a"], times, 9, 6) == 0


def test_cut_cells_are_gathered_once():
    from lagoonviz.core.data import mesh

    # two hexahedra in a row
    connectivity = np.array([0, 1, 3, 2, 4, 5, 7, 6, 1, 8, 9, 3, 5, 10, 11, 7], np.int64)
    offsets = np.array([8, 16], np.int64)
    types = np.array([12, 12], np.uint8)
    first = mesh._cells_by_type(connectivity, offsets, types)
    assert mesh._cells_by_type(connectivity, offsets, types) is first  # kept
    assert first[0][2].shape == (2, 8)
    distance = np.array([-1, 1, -1, 1, -1, 1, -1, 1, 2, 2, 2, 2], float)
    plan = mesh.plan_level(distance, connectivity, offsets, types, True)
    assert len(plan.triangles) > 0  # the first cell is cut, the second is not
    assert np.allclose(plan.t, 0.5)


def test_the_lattice_of_a_run_is_found_once():
    from lagoonviz.core.data import amr3d
    from lagoonviz.core.data.vtkxml import Piece

    x, y, z = np.meshgrid(np.arange(3.0), np.arange(2.0), np.arange(2.0), indexing="ij")
    points = np.c_[x.ravel(), y.ravel(), z.ravel()].astype(np.float32)
    empty = np.zeros(0, np.int64)

    def piece(t):
        return Piece("UnstructuredGrid", points, empty, empty, None, {"phi": points[:, 0] + t})

    amr3d._LATTICES.clear()
    first = amr3d.volume_lattice([piece(0.0)], ["phi"])
    assert len(amr3d._LATTICES) == 1
    second = amr3d.volume_lattice([piece(1.0)], ["phi"])
    assert len(amr3d._LATTICES) == 1  # the same nodes: found once
    np.testing.assert_allclose(second[3]["phi"], first[3]["phi"] + 1.0)
