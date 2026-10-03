"""Contour lines on triangle surfaces."""

import numpy as np

from lagoonviz.core import Session
from lagoonviz.core.contours import contour_levels, contour_segments


def test_levels_are_round_and_inside():
    levels = contour_levels(-1.93, 2.07, 8)
    assert levels and all(-1.93 < v < 2.07 for v in levels)
    assert all(abs(v * 2 - round(v * 2)) < 1e-9 for v in levels)  # steps of 0.5
    assert contour_levels(1.0, 1.0, 5) == [] and contour_levels(0, float("nan"), 5) == []


def test_a_plane_gives_straight_lines_at_the_levels():
    # a ramp f = x on a 3 x 3 grid of two triangles per cell
    xs, ys = np.meshgrid(np.arange(3.0), np.arange(3.0), indexing="ij")
    points = np.stack([xs.ravel(), ys.ravel(), np.zeros(9)], axis=1)
    triangles = []
    for i in range(2):
        for j in range(2):
            a = i * 3 + j
            triangles += [(a, a + 3, a + 4), (a, a + 4, a + 1)]
    triangles = np.array(triangles)
    values = points[:, 0]
    segments = contour_segments(points, triangles, values, [0.5, 1.5])
    assert segments.shape[1:] == (2, 3)
    np.testing.assert_allclose(np.sort(np.unique(segments[..., 0].round(9))), [0.5, 1.5])
    # the lines run the whole width in y, without gaps
    for level in (0.5, 1.5):
        line = segments[np.isclose(segments[:, 0, 0], level)]
        length = np.linalg.norm(line[:, 1] - line[:, 0], axis=1).sum()
        assert abs(length - 2.0) < 1e-9
    # missing values and hidden (degenerate) triangles draw nothing
    holes = values.copy()
    holes[4] = np.nan
    assert len(contour_segments(points, triangles, holes, [0.5])) < len(
        contour_segments(points, triangles, values, [0.5])
    )
    assert contour_segments(points, np.zeros((4, 3), int), values, [0.5]).shape == (0, 2, 3)


def test_bed_contours_follow_the_shown_surface(reef3d_run):
    s = Session()
    s.open_case(reef3d_run)
    bed = next(i.id for i in s.scene if i.kind == "bathymetry")
    mesh = s.data.load(bed)
    shown = s.data.display(bed, mesh)
    assert s.data.contour_lines(bed, mesh, shown) is None  # off by default
    s.set_property(bed, "contour_by", "elevation")
    s.set_property(bed, "contour_levels", 4)
    mesh = s.data.load(bed)  # now with the elevation array
    lines = s.data.contour_lines(bed, mesh, s.data.display(bed, mesh))
    assert lines is not None and lines.segments.shape[0] > 0 and len(lines.levels) >= 2
    # the bed rises with x (z = 0.1 x): each level is a line at its height, lifted a little
    z = lines.segments[..., 2]
    assert np.all(z > mesh.points[:, 2].min()) and np.all(z <= mesh.points[:, 2].max() + 0.1)
    s.set_property("case-1", "exaggeration", 3.0)
    taller = s.data.contour_lines(bed, mesh, s.data.display(bed, mesh))
    assert np.ptp(taller.segments[..., 2]) > np.ptp(z)  # drawn on the exaggerated bed
    s.set_plan_view(True)
    flat = s.data.contour_lines(bed, mesh, s.data.display(bed, mesh))
    assert np.ptp(flat.segments[..., 2]) < 1e-6  # on the flat 2D layer
