"""A line traced point by point in the 2D top view (a river drawn on the map)."""

import numpy as np

from lagoonviz.core.line_trace import LineTrace


def test_a_line_clicked_point_by_point():
    trace = LineTrace()
    assert trace.sketch((1, 1)) == (None, None)  # nothing yet
    assert not trace.click((0, 0), reach=0.5)
    assert not trace.done
    assert not trace.click((0.2, 0.1), reach=0.5)  # on the only point: not done yet
    assert trace.points == [(0.0, 0.0)]
    assert not trace.click((10, 0), reach=0.5)
    segments, dots = trace.sketch((10, 5), z=4.0)
    assert segments.shape == (2, 2, 3) and dots.shape == (2, 3)
    np.testing.assert_allclose(segments[-1], [[10, 0, 4], [10, 5, 4]])  # to the pointer
    assert trace.length((10, 5)) == 15.0
    assert trace.back() and trace.points == [(0.0, 0.0)]
    assert not trace.click((10, 0), reach=0.5)
    assert not trace.click((20, 0), reach=0.5)
    assert trace.click((20.3, 0.1), reach=0.5)  # the last point again: done
    assert trace.points == [(0.0, 0.0), (10.0, 0.0), (20.0, 0.0)]
    assert trace.done and trace.length() == 20.0
    for _ in range(3):
        assert trace.back()
    assert not trace.back()


def test_a_line_taken_up_again_and_changed():
    trace = LineTrace([(0, 0), (10, 0), (20, 0)])
    assert trace.done
    assert not trace.click((0.1, 0.2), reach=0.5)  # on the first point: nothing
    assert len(trace.points) == 3
    assert not trace.click((5, 0.3), reach=0.5)  # on a segment: a point put in
    assert trace.points[1] == (5.0, 0.3)
    assert not trace.click((30, 5), reach=0.5)  # elsewhere: added at the end
    assert trace.points[-1] == (30.0, 5.0)
    k = trace.point_at((10.2, 0.0), reach=0.5)
    assert k == 2 and trace.point_at((12, 0), reach=0.5) is None
    trace.move(k, (11, 2))
    assert trace.points[2] == (11.0, 2.0)
    trace.remove(1)
    assert trace.points == [(0.0, 0.0), (11.0, 2.0), (20.0, 0.0), (30.0, 5.0)]
    assert trace.segment_at((25, 2.5), reach=0.5) == 2
    assert trace.segment_at((20, 0), reach=0.5) is None  # at a point, not on a segment
    assert trace.click((30, 5.1), reach=0.5)  # the last point: done
