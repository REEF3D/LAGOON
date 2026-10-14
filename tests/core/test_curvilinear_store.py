"""Curvilinear grids in LAGOON stores (SPEC 2.2, REEF3D's G 50): each grid point
has its own x and y (xc, yc); the surfaces, volume faces, pieces, slices and probes
are drawn where the points lie, not on the index lines."""

import numpy as np
import pytest
from lagoon_format.store import Store, StoreError, StoreWriter, sigma_heights

from lagoonviz.core.data.lagoon import store_datasets

NX, NY = 13, 7
SIGMA = np.array([0.0, 0.5, 1.0])
#: two blocks overlapping at index line 6, as two MPI ranks write them
BLOCKS = [((0, 0), (7, NY)), ((6, 0), (7, NY))]


def bend():
    """A quarter of a river bend: radius 100 to 160 m across, 90 degrees along."""
    i = np.arange(NX, dtype=float)
    j = np.arange(NY, dtype=float)
    angle = np.radians(90.0 * i / (NX - 1))
    radius = 100.0 + 10.0 * j
    xc = radius[:, None] * np.cos(angle)[None, :]
    yc = radius[:, None] * np.sin(angle)[None, :]
    return i, j, xc, yc


@pytest.fixture
def store(tmp_path):
    i, j, xc, yc = bend()
    writer = StoreWriter.create(tmp_path / "REEF3D_SFLOW.lagoon", solver="SFLOW")
    surface = writer.add_output("free_surface", "surface", i, j, {"eta": 1}, BLOCKS, xc=xc, yc=yc)
    volume = writer.add_output(
        "volume", "sigma", i, j, {"pressure": 1}, BLOCKS, sigma=SIGMA, xc=xc, yc=yc
    )
    for t in range(2):
        eta = 0.1 * t + 0.01 * xc
        bed = -2.0 - 0.001 * yc
        z = sigma_heights(bed, eta, SIGMA)
        parts, columns = [], []
        for (i0, _), (nx_b, _) in BLOCKS:
            w = slice(i0, i0 + nx_b)
            parts.append({"z": eta[:, w], "eta": eta[:, w]})
            columns.append({"z_bed": bed[:, w], "z_surface": eta[:, w], "pressure": z[:, :, w]})
        surface.append(0.5 * t, t, parts)
        volume.append(0.5 * t, t, columns)
    writer.close()
    return tmp_path / "REEF3D_SFLOW.lagoon"


def test_the_store_keeps_where_each_point_lies(store):
    _, _, xc, yc = bend()
    output = Store(store)["free_surface"]
    assert output.curvilinear and output.info["horizontal"] == "curvilinear"
    np.testing.assert_allclose(output.xc, xc)
    np.testing.assert_allclose(output.yc, yc)
    assert output.nearest(0.0, 130.0) == (NX - 1, 3)  # the end of the bend, mid-river
    assert output.point_xy(0, NY - 1) == pytest.approx((160.0, 0.0))
    px, py = output.plan(2, 5, 1, 3)
    np.testing.assert_allclose(px, xc[1:3, 2:5])


def test_xc_and_yc_must_fit_the_grid(tmp_path):
    writer = StoreWriter.create(tmp_path / "bad.lagoon", solver="SFLOW")
    with pytest.raises(StoreError, match="xc and yc"):
        writer.add_output(
            "bed",
            "surface",
            np.arange(3.0),
            np.arange(2.0),
            {},
            [((0, 0), (3, 2))],
            xc=np.zeros((3, 2)),
            yc=np.zeros((3, 2)),
        )


def test_the_surface_and_volume_drawn_where_the_points_lie(store):
    _, _, xc, yc = bend()
    datasets = {d.key: d for d in store_datasets(store)}
    surface = datasets["sflow_free_surface"].read(1)
    np.testing.assert_allclose(surface.points[:, 0], xc.ravel(), atol=1e-4)
    np.testing.assert_allclose(surface.points[:, 1], yc.ravel(), atol=1e-4)
    np.testing.assert_allclose(surface.points[:, 2], (0.1 + 0.01 * xc).ravel(), atol=1e-5)
    assert (surface.normals[:, 2] > 0.99).all()  # nearly flat, pointing up
    low, high = datasets["sflow_free_surface"].bounds(0)
    assert low[:2] == pytest.approx((0.0, 0.0), abs=1e-4)
    assert high[:2] == pytest.approx((160.0, 160.0), abs=1e-3)
    volume = datasets["sflow_volume"]
    faces = volume.read(0, ["pressure"])
    radius = np.hypot(faces.points[:, 0], faces.points[:, 1])
    assert radius.min() == pytest.approx(100.0, abs=1e-3)
    assert radius.max() == pytest.approx(160.0, abs=1e-3)
    # a slice across the river at x = 50 m: the hexahedra cut where they lie
    cut = volume.read_slice(0, 0, 50.0, ["pressure"])
    assert len(cut.points) and np.allclose(cut.points[:, 0], 50.0, atol=1e-3)
    where, series = volume.point_series("pressure", 0.0, 130.0)
    assert where[:2] == pytest.approx((xc[3, NX - 1], yc[3, NX - 1]))
    assert len(series) == 2


def test_a_curvilinear_run_is_not_converted_from_its_vtk():
    """lagoon-convert takes grid lines from the points' distinct x and y: a
    curvilinear grid has none, its output is left out (REEF3D's P 18 writes it)."""
    from lagoonviz.core.convert import Unstructured, _grid_lines

    _, _, xc, yc = bend()
    points = np.stack([xc.ravel(), yc.ravel(), np.zeros(xc.size)], axis=1)
    with pytest.raises(Unstructured, match="curvilinear"):
        _grid_lines([points])
    gx, gy = np.meshgrid(np.arange(NX, dtype=float), np.arange(NY, dtype=float))
    xs, ys = _grid_lines([np.stack([gx.ravel(), gy.ravel(), gx.ravel() * 0], axis=1)])
    assert len(xs) == NX and len(ys) == NY
