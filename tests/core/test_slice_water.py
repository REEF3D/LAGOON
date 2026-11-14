# LAGOON — slices of CFD: the water alone, the line of an iso-surface
# Architect: Hans Bihs
"""A slice of a CFD volume can be shown only in the water (phi >= 0), cut at the
level set's zero; and any slice can draw the line where an iso-surface of a variable
crosses it (phi at 0: the free surface), lying on the slice."""

from __future__ import annotations

import numpy as np
from test_iso_band import _cfd_store, _surface

from lagoonviz.core import Session
from lagoonviz.core.data.mesh import SurfaceMesh, clip_surface


def _slice(tmp_path, **properties):
    root = _cfd_store(tmp_path / "cfd", frames=2)
    s = Session()
    s.open_case(root)
    for item in s.scene:
        if item.kind == "iso_surface":
            s.set_visible(item.id, False)
    item = s.add_slice("y", position=1.5, **properties)
    return s, item


def test_a_clipped_surface_keeps_its_grid_lines_up_to_the_cut():
    """clip_surface: the outlines (a slice's grid lines, points of their own) where
    the field is at least the level, a line across it ending where it crosses."""
    x, z = np.meshgrid([0.0, 1.0, 2.0], [0.0, 1.0, 2.0])
    points = np.stack([x.ravel(), 0 * x.ravel(), z.ravel()], 1).astype(np.float32)
    k = np.arange(9).reshape(3, 3)
    a, b, c, d = k[:-1, :-1].ravel(), k[:-1, 1:].ravel(), k[1:, 1:].ravel(), k[1:, :-1].ravel()
    triangles = np.concatenate([np.stack([a, b, c], 1), np.stack([a, c, d], 1)]).astype(np.uint32)
    # the vertical grid lines as points of their own, after the surface's
    ends = np.array([[i, 0, 0, i, 0, 2] for i in (0.0, 1.0, 2.0)], np.float32).reshape(-1, 3)
    all_points = np.concatenate([points, ends])
    phi = 1.5 - all_points[:, 2]  # water below z = 1.5
    mesh = SurfaceMesh(
        all_points,
        triangles,
        {"phi": phi.astype(np.float32)},
        outlines=(9 + np.arange(6, dtype=np.uint32)).reshape(-1, 2),
    )
    water = clip_surface(mesh, phi, 0.0, "above")
    assert water.points[water.triangles.ravel(), 2].max() <= 1.5 + 1e-6
    lines = water.points[water.outlines.astype(np.int64)]
    assert lines.shape == (3, 2, 3)
    np.testing.assert_allclose(np.sort(lines[:, :, 2], axis=1), [[0.0, 1.5]] * 3, atol=1e-6)


def test_a_cfd_slice_in_the_water_alone(tmp_path):
    s, item = _slice(tmp_path)
    whole = s.data.load(item)
    assert whole.points[:, 2].max() > 2.9  # up to the top of the volume, the air too
    s.set_property(item, "water_only", True)
    s.set_property(item, "grid_lines", "dark")
    key = s.data.frame_key(item)
    assert key.extra[3] is True and "phi" in key.arrays
    water = s.data.load(item)
    cells = water.points[water.triangles.astype(np.int64).ravel()]
    assert np.nanmin(water.point_data["phi"][water.triangles.ravel()]) >= -1e-5
    # up to the free surface at the slice (y = 1.5), not above it
    level = _surface(np.unique(cells[:, 0].astype(np.float64)), np.array([1.5]), 0)[0]
    highest = cells[:, 2].max()
    assert level.max() - 0.15 < highest <= level.max() + 1e-3
    # its grid lines end at the surface too
    lines = water.points[water.outlines.astype(np.int64)]
    assert len(lines) and lines[..., 2].max() <= level.max() + 1e-3


def test_water_only_needs_a_level_set(tmp_path, reef3d_run):
    """A slice of a volume without phi (NHFLOW): the setting does nothing."""
    s = Session()
    s.open_case(reef3d_run)
    item = s.add_slice("y", position=2.5, water_only=True)
    assert len(s.data.frame_key(item).extra) == 3
    assert s.data.load(item).n_triangles


def test_the_line_of_an_iso_surface_on_a_slice(tmp_path):
    """ "Iso-line of" phi at 0: the free surface on the slice, lying on it (drawn
    towards the camera, not lifted), with the slice's contour lines if it has any."""
    s, item = _slice(tmp_path, variable="velocity")
    mesh = s.data.load(item)
    shown = s.data.display(item, mesh)
    assert s.data.contour_lines(item, mesh, shown) is None
    s.set_property(item, "iso_line_of", "phi")
    mesh = s.data.load(item)
    shown = s.data.display(item, mesh)
    lines = s.data.contour_lines(item, mesh, shown)
    assert lines is not None and lines.facing and len(lines.segments)
    ends = lines.segments.reshape(-1, 3).astype(np.float64)
    np.testing.assert_allclose(ends[:, 1], 1.5, atol=1e-5)  # on the slice
    expected = _surface(ends[:, 0], np.array([1.5]), 0)[0]
    assert np.abs(ends[:, 2] - expected).max() < 0.05  # where the free surface is
    s.set_property(item, "iso_line_value", 0.5)  # half a metre under it
    mesh = s.data.load(item)
    deeper = s.data.contour_lines(item, mesh, s.data.display(item, mesh)).segments
    assert abs(float(np.median(ends[:, 2]) - np.median(deeper[..., 2])) - 0.5) < 0.05
    s.set_property(item, "contour_by", "pressure")  # contour lines and the iso-line
    mesh = s.data.load(item)
    both = s.data.contour_lines(item, mesh, s.data.display(item, mesh))
    assert len(both.segments) > len(deeper) and both.levels
