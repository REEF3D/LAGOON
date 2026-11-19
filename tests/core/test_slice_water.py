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
    s, item = _slice(tmp_path, water_only=False)
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
    towards the camera, not lifted), in its own colour, beside contour lines."""
    s, item = _slice(tmp_path, variable="velocity")
    mesh = s.data.load(item)
    shown = s.data.display(item, mesh)
    assert s.data.iso_line(item, mesh, shown) is None
    s.set_property(item, "iso_line_of", "phi")
    mesh = s.data.load(item)
    shown = s.data.display(item, mesh)
    line = s.data.iso_line(item, mesh, shown)
    assert line is not None and line.facing and len(line.segments)
    assert line.color == "#ffffff"
    assert s.data.contour_lines(item, mesh, shown) is None  # no contour lines asked for
    ends = line.segments.reshape(-1, 3).astype(np.float64)
    np.testing.assert_allclose(ends[:, 1], 1.5, atol=1e-5)  # on the slice
    expected = _surface(ends[:, 0], np.array([1.5]), 0)[0]
    assert np.abs(ends[:, 2] - expected).max() < 0.05  # where the free surface is
    s.set_property(item, "iso_line_value", 0.5)  # half a metre under it
    s.set_property(item, "iso_line_colour", "red")
    mesh = s.data.load(item)
    deeper = s.data.iso_line(item, mesh, s.data.display(item, mesh))
    assert deeper.color == "#e0302a"
    assert abs(float(np.median(ends[:, 2]) - np.median(deeper.segments[..., 2])) - 0.5) < 0.05
    # with Water only: the line along the slice's top
    s.set_property(item, "iso_line_value", 0.0)
    s.set_property(item, "water_only", True)
    mesh = s.data.load(item)
    top = s.data.iso_line(item, mesh, s.data.display(item, mesh))
    assert top is not None and len(top.segments)
    # drawn in the 3D view, kept in its colour by another look
    from lagoonviz.render.scene3d import SceneRenderer

    view = SceneRenderer()
    view.update_iso_line(item, top)
    drawn = view._contours[f"{item}#isoline"]
    view.set_style("ocean")
    assert tuple(round(c, 3) for c in tuple(drawn.material.color)[:3]) == (0.878, 0.188, 0.165)


def test_a_slice_within_the_iso_volume(tmp_path):
    """Kelley: a CFD section only in the water. A new slice of a CFD run is in the
    water (phi >= 0); with an iso-volume shown, within it (its variable's range, a
    calculated field's too), following its range as it changes."""
    root = _cfd_store(tmp_path / "cfd", frames=2)
    s = Session()
    case = s.open_case(root)
    first = s.add_slice("y", position=1.5)
    assert s.scene.get(first).properties["water_only"] is True
    # an iso-volume of a calculated field: the water deeper than 0.5 m
    s.set_property(case, "calculated", "depth = phi")
    deep = s.add_iso_volume("depth", 0.5, 10.0)
    item = s.add_slice("y", position=1.5, variable="velocity")
    assert s.scene.get(item).properties["within"] == deep
    assert s.data.choices(item, "iso_volumes") == ["", deep]
    mesh = s.data.load(item)
    cells = mesh.points[mesh.triangles.astype(np.int64)]
    level = _surface(np.unique(cells[..., 0].astype(np.float64)), np.array([1.5]), 0)[0]
    assert cells[..., 2].max() <= level.max() - 0.5 + 1e-3  # half a metre under the surface
    assert np.nanmin(mesh.point_data["depth"][mesh.triangles.ravel()]) >= 0.5 - 1e-4
    # the iso-volume's range changed: the slice follows (and is redrawn with it)
    s.set_property(deep, "iso_min", 1.0)
    assert s.data.slices_within(deep) == [item]
    deeper = s.data.load(item)
    assert (
        deeper.points[deeper.triangles.astype(np.int64)][..., 2].max() <= level.max() - 1.0 + 1e-3
    )
    # the iso-volume gone: everywhere
    s.remove_item(deep)
    assert s.data.within(item) is None and s.data.load(item).n_triangles
