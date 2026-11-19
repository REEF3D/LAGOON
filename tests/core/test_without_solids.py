# LAGOON — the volume, its slices and iso-surfaces without the solids in them
# Architect: Hans Bihs
"""REEF3D writes its volume in solids too (DIVEMesh's, NHFLOW's direct forcing
solids, under floating bodies): "Without solids" cuts the volume's outside, its
slices and its iso-surfaces where they enter the case's solids, along their faces.
Porous structures stay (the flow in the stones is real)."""

from __future__ import annotations

import numpy as np
from test_iso_band import _cfd_store

from lagoonviz.core import Session
from lagoonviz.core.data.solidclip import Solids, without_solids

# a box of DIVEMesh's (control.txt S 10) across the flume, a vertical cylinder
CONTROL = "B 10 0.0 6.0 0.0 3.0 0.0 3.0\nS 10 2.0 3.0 0.5 2.5 0.0 1.2\nS 33 4.5 1.5 0.4\n"


def _box(x0, x1, y0, y1, z0, z1):
    """A box's twelve triangles, facing out."""
    c = np.array([(x, y, z) for x in (x0, x1) for y in (y0, y1) for z in (z0, z1)], float)
    faces = [(0, 1, 3, 2), (4, 6, 7, 5), (0, 4, 5, 1), (2, 3, 7, 6), (0, 2, 6, 4), (1, 5, 7, 3)]
    out = []
    for a, b, cc, d in faces:
        out += [(c[a], c[b], c[cc]), (c[a], c[cc], c[d])]
    return np.array(out)


def test_inside_two_solids_that_overlap():
    solids = Solids(np.concatenate([_box(0, 2, 0, 2, 0, 2), _box(1, 3, 1, 3, 1, 3)]))
    points = np.array(
        [
            [0.5, 0.5, 0.5],  # in the first
            [1.5, 1.5, 1.5],  # in both
            [2.5, 2.5, 2.5],  # in the second
            [2.5, 0.5, 0.5],  # in neither
            [1.0, 1.0, 1.0],  # on a corner of the second: in the first
            [5.0, 5.0, 5.0],
            [0.5, 0.5, 2.5],  # above the first
        ]
    )
    np.testing.assert_array_equal(
        solids.inside(points), [True, True, True, False, True, False, False]
    )


def test_a_surface_cut_along_the_solid():
    """A plane across a box: cut where its edges enter the box (not half way along
    them), so along the box's faces; a corner of the box in a cell cut off it."""
    from lagoonviz.core.data.mesh import grid_triangles

    x, z = np.meshgrid(np.linspace(0, 4, 9), np.linspace(0, 2, 5))
    points = np.stack([x.ravel(), np.full(x.size, 0.5), z.ravel()], 1).astype(np.float32)
    from lagoonviz.core.data.mesh import SurfaceMesh

    plane = SurfaceMesh(points, grid_triangles(5, 9), {"u": points[:, 0].copy()})
    cut = without_solids(plane, Solids(_box(1.2, 2.7, 0, 1, -1, 1.3)))
    t = cut.points[cut.triangles.astype(np.int64)].astype(np.float64)
    centre = t.mean(axis=1)
    deep = (centre[:, 0] > 1.45) & (centre[:, 0] < 2.45) & (centre[:, 2] < 1.05)
    assert not deep.any()
    area = 0.5 * np.abs(np.cross(t[:, 1] - t[:, 0], t[:, 2] - t[:, 0])[:, 1]).sum()
    # the box's part (1.5 x 1.3), but for the two corners in a cell (at most half a cell)
    assert 4 * 2 - 1.5 * 1.3 <= area <= 4 * 2 - 1.5 * 1.3 + 2 * 0.5 * 0.25
    # on the box's sides exactly (x = 1.2, 2.7 below the corners' cells)
    sides = cut.points[(cut.points[:, 2] < 0.99)][:, 0]
    assert np.isclose(sides, 1.2, atol=1e-6).any() and np.isclose(sides, 2.7, atol=1e-6).any()
    assert not ((sides > 1.2 + 1e-6) & (sides < 2.7 - 1e-6)).any()
    np.testing.assert_allclose(cut.point_data["u"], cut.points[:, 0], atol=1e-5)


def test_without_solids_of_a_case(tmp_path):
    root = _cfd_store(tmp_path / "cfd", frames=1)
    (root / "control.txt").write_text(CONTROL)
    s = Session()
    s.open_case(root)
    item = s.add_slice("y", position=1.5)
    whole = s.data.load(item)
    assert s.data.frame_key(item).clip == ()
    s.set_property(item, "without_solids", True)
    key = s.data.frame_key(item)
    assert [name for name, _ in key.clip] == ["divemesh_solids"]
    cut = s.data.load(item)
    centre = cut.points[cut.triangles.astype(np.int64)].mean(axis=1)
    # (but for the corners of a solid in a cell: a cell inside its sides)
    in_box = (centre[:, 0] > 2.25) & (centre[:, 0] < 2.75) & (centre[:, 2] < 1.0)
    in_cylinder = (np.abs(centre[:, 0] - 4.5) < 0.3) & (centre[:, 2] < 3.0)
    assert not in_box.any() and not in_cylinder.any()
    assert cut.n_triangles < whole.n_triangles
    # the volume's outside and an iso-surface too
    volume = next(i for i in s.scene if i.kind == "volume")
    s.set_property(volume.id, "without_solids", True)
    outside = s.data.load(volume.id)
    p = outside.points[outside.triangles.astype(np.int64)].mean(axis=1)
    on_box = (p[:, 0] > 2.25) & (p[:, 0] < 2.75) & (p[:, 2] < 1.19)
    on_box &= (p[:, 1] > 0.9) & (p[:, 1] < 2.1)
    assert not on_box.any()
    free = next(i for i in s.scene if i.kind == "iso_surface")
    s.set_property(free.id, "iso_value", 0.0)
    s.set_property(free.id, "without_solids", True)
    assert s.data.load(free.id).n_triangles  # the free surface lies above the box


def test_the_floating_bodies_at_the_time(tmp_path):
    """A floating body cuts the volume where it is at the time shown."""
    from reef3d_fixtures import body_offset, make_run

    run = make_run(tmp_path / "moored", frames=4, objects=True)
    s = Session()
    s.open_case(run)
    volume = next(i for i in s.scene if i.kind == "volume")
    s.set_property(volume.id, "without_solids", True)
    body = next(i for i in s.scene if i.kind == "body")
    for frame in (1, 3):
        s.set_time(float(frame))
        key = s.data.frame_key(volume.id)
        assert (s.data.dataset(body.id).key, s.data.frame_index(body.id)) in key.clip
        solids = s.data._solids(key)
        assert np.allclose(solids.low, body_offset(s.data.frame_index(body.id)), atol=1e-5)


def test_on_a_side_face_at_either_side():
    """Bug: the ray up started a little aside, to one side only: a point on a solid's
    side face at its low x or y was in it, one at its high x or y not. A beach across
    a flume (from one side wall to the other) was cut from the volume's outside at
    one wall and stood there at the other."""
    solids = Solids(_box(1.0, 2.0, 0.0, 3.0, 0.0, 1.0))
    points = np.array(
        [
            [1.5, 0.0, 0.5],  # on the side at y = 0
            [1.5, 3.0, 0.5],  # on the side at y = 3
            [1.0, 1.5, 0.5],  # on the side at x = 1
            [2.0, 1.5, 0.5],  # on the side at x = 2
            [2.0, 3.0, 0.5],  # on an edge
            [1.5, 3.0 + 1e-3, 0.5],  # beside it
            [2.0 + 1e-3, 1.5, 0.5],
        ]
    )
    np.testing.assert_array_equal(
        solids.inside(points), [True, True, True, True, True, False, False]
    )


def _turned_up(triangles):
    """A solid's triangles as DIVEMesh's REEF3D_Solid.vtp has them: each flat face
    turned up (its bottom as its top), each its own three points."""
    t = np.array(triangles, float)
    normal = np.cross(t[:, 1] - t[:, 0], t[:, 2] - t[:, 0])
    down = normal[:, 2] < 0
    t[down] = t[down][:, ::-1]
    return t


def test_solids_that_are_not_oriented():
    """Bug: DIVEMesh's solids (REEF3D_Solid.vtp) have every flat face turned up, a
    box's bottom as its top: counted by the winding, all under a solid was in it.
    The free surface under a pier's deck was cut away (and the water's sides of
    the stand hung round the gap). A solid whose faces are not oriented counts by
    the number of its faces crossed, odd inside; its faces may touch another's
    (a girder under the deck slab)."""
    slab = _turned_up(_box(0.0, 4.0, 1.0, 2.0, 0.70, 0.74))
    girder = _turned_up(_box(0.0, 4.0, 1.4, 1.6, 0.68, 0.70))  # its top on the slab's bottom
    cap = _turned_up(_box(1.9, 2.1, 1.0, 2.0, 0.66, 0.70))
    solids = Solids(np.concatenate([slab, girder, cap]))
    points = np.array(
        [
            [1.0, 1.5, 0.60],  # under the deck: the water there
            [1.0, 1.5, 0.72],  # in the slab
            [1.0, 1.5, 0.69],  # in the girder
            [2.0, 1.2, 0.67],  # in the cap
            [2.0, 1.5, 0.69],  # in the girder and the cap
            [1.0, 1.2, 0.69],  # beside the girder, under the slab
            [1.0, 1.5, 0.80],  # over the deck
        ]
    )
    np.testing.assert_array_equal(
        solids.inside(points), [False, True, True, True, True, False, False]
    )
    # oriented, the same (by the winding)
    oriented = Solids(np.concatenate([_box(0.0, 4.0, 1.0, 2.0, 0.70, 0.74),
                                      _box(0.0, 4.0, 1.4, 1.6, 0.68, 0.70),
                                      _box(1.9, 2.1, 1.0, 2.0, 0.66, 0.70)]))  # fmt: skip
    np.testing.assert_array_equal(oriented.inside(points), solids.inside(points))


def test_a_solid_across_the_domain_is_cut_at_both_walls(tmp_path):
    root = _cfd_store(tmp_path / "cfd", frames=1)
    (root / "control.txt").write_text(
        "B 10 0.0 6.0 0.0 3.0 0.0 3.0\nS 10 2.0 3.0 0.0 3.0 0.0 1.2\n"
    )
    s = Session()
    s.open_case(root)
    volume = next(i for i in s.scene if i.kind == "volume")
    s.set_property(volume.id, "without_solids", True)
    outside = s.data.load(volume.id)
    p = outside.points[outside.triangles.astype(np.int64)].mean(axis=1)
    in_solid = (p[:, 0] > 2.05) & (p[:, 0] < 2.95) & (p[:, 2] < 1.15)
    for wall in (0.0, 3.0):
        on_wall = np.abs(p[:, 1] - wall) < 1e-6
        assert on_wall.any() and not (on_wall & in_solid).any(), wall


def _sheet(x0, x1, y0, y1, z, n=9):
    from lagoonviz.core.data.mesh import grid_triangles

    x, y = np.meshgrid(np.linspace(x0, x1, n), np.linspace(y0, y1, n))
    points = np.stack([x.ravel(), y.ravel(), np.full(x.size, z)], 1)
    return points, grid_triangles(n, n)


def _joined(*sheets):
    from lagoonviz.core.data.mesh import SurfaceMesh

    points, triangles, offset = [], [], 0
    for p, t in sheets:
        points.append(p)
        triangles.append(np.asarray(t, np.int64) + offset)
        offset += len(p)
    points = np.concatenate(points).astype(np.float32)
    return SurfaceMesh(points, np.concatenate(triangles).astype(np.uint32), {"z": points[:, 2]})


def test_a_sheet_on_a_solid_s_top_is_not_water():
    """REEF3D keeps the level set in a solid's cells (a deck over the waves holds the
    water of an earlier wave), so the free surface had a sheet lying on the deck, in
    it or a cell over it: water on the deck where none was. Without the solids such
    a sheet goes when all of it lies within the interface band over a solid. Water
    joined to the rest of the surface stays, and so does a thick body of water."""
    from lagoonviz.core.data.solidclip import without_films

    deck = Solids(_box(4.0, 6.0, 0.0, 1.0, 1.5, 2.0))
    sea = _sheet(0.0, 10.0, 0.0, 1.0, 1.0, n=21)
    film = _sheet(4.2, 5.0, 0.2, 0.8, 2.004)  # a cell over the deck
    pool = _sheet(5.2, 5.8, 0.2, 0.8, 2.3)  # water over the deck, deeper than the band
    # a wave going over the deck: one surface from the sea up onto it (corners shared)
    over_x = np.array([3.6, 3.8, 4.0, 4.1])
    over_z = np.array([1.0, 1.6, 2.004, 2.004])
    rows = np.stack([np.repeat(over_x, 2), np.tile([0.0, 0.2], 4), np.repeat(over_z, 2)], 1)
    wave = (rows, np.array([[0, 2, 1], [1, 2, 3], [2, 4, 3], [3, 4, 5], [4, 6, 5], [5, 6, 7]]))
    mesh = _joined(sea, film, pool, wave)
    kept = without_films(mesh, deck, band=0.02)
    z = kept.points[:, 2]
    # the film gone, the wave's two rows on the deck not, the pool not
    assert np.isclose(z, 2.004).sum() == 4 and np.isclose(z, 2.3).sum() == 81
    assert np.isclose(z, 1.0).sum() == len(sea[0]) + 2
    np.testing.assert_allclose(kept.point_data["z"], z)
    assert kept.triangles.max() < kept.n_points
    # nothing on a solid: the surface as it was
    assert without_films(_joined(sea), deck, band=0.02).n_triangles == len(sea[1])
