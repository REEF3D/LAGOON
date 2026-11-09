"""CFD's free surface (phi at 0) coloured from the water side: inside the level set's
interface band the density is smoothed over F 45 cells, so the pressure at phi = 0
carries part of the band's hydrostatic pressure and the velocity a blend of water and
air, both set by the cell size: with AMR a step at every block edge. The colours are
taken at phi = depth and 2 depth below the band and extrapolated up to the surface."""

import numpy as np
from test_cfd_amr import cfd_amr_case

from lagoonviz.core import Session
from lagoonviz.core.data.mesh import SurfaceMesh
from lagoonviz.core.data.reef3d import _cell_height, _interface_cells, water_side_values

RHO_G = 1000.0 * 9.81


def _plane(z: float, n: int = 21, down: bool = True, **fields) -> SurfaceMesh:
    """A square of 2 x 2 m at height z, its normals down (into the water) or up."""
    x, y = np.meshgrid(np.linspace(0, 2, n), np.linspace(0, 2, n), indexing="ij")
    points = np.stack([x.ravel(), y.ravel(), np.full(x.size, z)], axis=1).astype(np.float32)
    k = np.arange(n * n).reshape(n, n)
    a, b, c, d = k[:-1, :-1].ravel(), k[1:, :-1].ravel(), k[1:, 1:].ravel(), k[:-1, 1:].ravel()
    triangles = np.concatenate([np.stack([a, b, c], 1), np.stack([a, c, d], 1)])
    data = {name: f(points).astype(np.float32) for name, f in fields.items()}
    normals = np.tile(np.float32([0, 0, -1 if down else 1]), (len(points), 1))
    return SurfaceMesh(points, triangles.astype(np.uint32), data, normals=normals)


def _flow(level: float):
    """Hydrostatic pressure from a surface at ``level``, a velocity linear in depth."""
    return {
        "pressure": lambda p: RHO_G * (level - p[:, 2]) + 50.0,
        "velocity": lambda p: np.stack([0.2 + 0.5 * p[:, 2], 0 * p[:, 2], p[:, 0]], 1),
        "level": lambda p: np.full(len(p), 7.0),
    }


def test_the_water_side_takes_out_the_band_and_the_layer():
    depth = 0.06
    # on the surface: the smoothed band's pressure (a step of 38 Pa between two grids)
    surface = _plane(
        0.5,
        pressure=lambda p: np.where(p[:, 0] < 1.0, 77.0, 38.0) + 50.0,
        velocity=lambda p: np.zeros((len(p), 3)),
        level=lambda p: (p[:, 0] >= 1.0).astype(float),
    )
    below = _plane(0.5 - depth, **_flow(0.5))
    deeper = _plane(0.5 - 2 * depth, **_flow(0.5))
    got = water_side_values(surface, below, deeper, depth)
    np.testing.assert_allclose(got.point_data["pressure"], 50.0, atol=0.05)  # atmospheric
    expected = 0.2 + 0.5 * 0.5  # the velocity extrapolated to the surface
    np.testing.assert_allclose(got.point_data["velocity"][:, 0], expected, atol=1e-5)
    np.testing.assert_allclose(got.point_data["velocity"][:, 2], surface.points[:, 0], atol=1e-4)
    # where the surface is and which grid it is on stay its own
    np.testing.assert_array_equal(got.point_data["level"], surface.point_data["level"])
    np.testing.assert_array_equal(got.points, surface.points)


def test_points_without_water_under_them_keep_their_own_values():
    depth = 0.06
    surface = _plane(0.5, pressure=lambda p: np.full(len(p), 9.0))
    far = _plane(-3.0, pressure=lambda p: np.zeros(len(p)))  # nowhere near
    got = water_side_values(surface, far, far, depth)
    np.testing.assert_array_equal(got.point_data["pressure"], 9.0)


def test_the_band_from_the_case(tmp_path):
    points = np.array([[0, 0, z] for z in (0.0, 0.4, 0.45, 0.5, 0.55, 1.0)], float)
    assert abs(_cell_height(points, 0.52) - 0.05) < 1e-9
    assert _cell_height(points, 2.0) == 0.0
    assert _interface_cells(tmp_path) == 2.1
    case = tmp_path / "case"
    case.mkdir()
    (case / "ctrl.txt").write_text("N 40 5\nF 45 1.6\n")
    assert _interface_cells(case) == 1.6


def test_the_free_surface_item_reads_the_water_side(tmp_path):
    root = cfd_amr_case(tmp_path)
    s = Session()
    case_id = s.open_case(root)
    volume = s.data.case(case_id).datasets["cfd_volume"]
    plain = volume.read_iso(0, "phi", 0.0, ["level", "velocity"])
    water = volume.read_iso(0, "phi", 0.0, ["level", "velocity"], water_side=True)
    np.testing.assert_array_equal(water.points, plain.points)
    np.testing.assert_array_equal(water.point_data["level"], plain.point_data["level"])
    free = next(i for i in s.scene if i.kind == "iso_surface")
    assert free.properties.get("colours_from", "water side") == "water side"
    key = s.data.frame_key(free.id)
    assert key.extra[0] == "iso" and key.extra[4] is True
    assert s.data.load(free.id).n_triangles
