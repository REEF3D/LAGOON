"""REEF3D::SEASTATE's maps of the sea state, and its AMR output."""

import math

import numpy as np
import pytest
from reef3d_fixtures import (
    DX,
    NX,
    NY,
    SEASTATE_DIR,
    SEASTATE_WD,
    seastate_hs,
    write_seastate,
    write_seastate_amr,
)

from lagoonviz.core import Session


def test_a_seastate_run_shows_its_sea_state(tmp_path):
    case = write_seastate(tmp_path / "bay")
    s = Session()
    s.open_case(case)
    item = next(i for i in s.scene if i.kind == "seastate")
    props = item.properties
    assert props["variable"] == "Hs" and props["colormap"] == "rainbow"
    assert props["vectors_of"] == "wave_direction"
    dataset = s.data.dataset(item.id)
    assert dataset.solver == "SEASTATE" and len(dataset) == 3
    assert {"Hs", "Tp", "dir", "wave_direction", "wave_vector", "wind"} <= set(dataset.variables())
    s.set_time(2)
    mesh = s.data.load(item.id)
    assert mesh.n_points == NX * NY  # the two ranks' shared column merged
    np.testing.assert_allclose(mesh.points[:, 2], SEASTATE_WD)
    wet = mesh.point_data["wet"] > 0
    np.testing.assert_allclose(
        mesh.point_data["Hs"][wet], seastate_hs(mesh.points[wet, 0], 2), rtol=1e-6
    )
    a = math.radians(SEASTATE_DIR)
    np.testing.assert_allclose(
        mesh.point_data["wave_direction"][0], (math.cos(a), math.sin(a), 0.0), atol=1e-6
    )
    every = dataset.read(2)  # all arrays: the wind too
    np.testing.assert_allclose(every.point_data["wind"][0], (10.0, 0.0, 0.0))
    shown = s.data.display(item.id, mesh)
    # dry land (the last column) left out
    kept = np.unique(shown.triangles)
    assert mesh.points[kept, 0].max() < (NX - 1) * DX - 1e-6
    arrows = s.data.vector_arrows(item.id, mesh, shown)
    assert arrows is not None and len(arrows.segments)
    direction = arrows.segments[0, 1] - arrows.segments[0, 0]
    assert math.degrees(math.atan2(direction[1], direction[0])) == pytest.approx(
        SEASTATE_DIR, abs=1e-3
    )
    # its colour range from the run, Hs on Turbo
    low, high = s.data.automatic_range(item.id)
    assert 0 <= low < high


def test_a_seastate_amr_run(tmp_path):
    case = write_seastate_amr(tmp_path / "amr")
    s = Session()
    s.open_case(case)
    item = next(i for i in s.scene if i.kind == "amr")
    assert item.properties["variable"] == "Hs" and item.properties["colormap"] == "rainbow"
    mesh = s.data.load(item.id)
    assert mesh.n_triangles and "Hs" in mesh.point_data and "level" in mesh.point_data
    assert mesh.point_data["level"].max() == 1  # the refined patch
    times = s.data.dataset(item.id).times  # from SEASTATE's TIME
    assert times is not None and list(times) == [0.0, 10.0]
