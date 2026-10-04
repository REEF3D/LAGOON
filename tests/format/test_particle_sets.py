"""Particles and objects with changing numbers of points (SPEC 10)."""

from __future__ import annotations

import numpy as np
import pytest
from lagoon_format import particles as module
from lagoon_format.store import Store, StoreWriter


def _cloud(t: int) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    rng = np.random.default_rng(t)
    n = 50 + 7 * t  # particles are added over time
    positions = rng.uniform(0, 10, (n, 3)).astype(np.float32)
    return positions, {
        "velocity": rng.normal(size=(n, 3)).astype(np.float32),
        "fluid velocity": rng.normal(size=(n, 3)).astype(np.float32),
        "id": np.arange(n, dtype=np.int32) + 1000,
        "radius": np.full(n, 0.001, np.float32),
    }


@pytest.mark.parametrize("small_buffer", [False, True])
def test_a_particle_cloud(tmp_path, monkeypatch, small_buffer):
    if small_buffer:  # flushes in the middle of the run
        monkeypatch.setattr(module, "BUFFER_ROWS", 120)
    path = tmp_path / "REEF3D_CFD.lagoon"
    writer = StoreWriter.create(path, "CFD")
    fields = {
        "velocity": (3, "float32"),
        "fluid velocity": (3, "float32"),
        "id": (1, "int32"),
        "radius": (1, "float32"),
    }
    cloud = writer.particle_writer(
        "cfd_particles",
        {
            "point_fields": fields,
            "source": "REEF3D_CFD_CPM_Particle",
            "attributes": {"dataset": "cfd_particles"},
        },
    )
    for t in range(12):
        positions, data = _cloud(t)
        cloud.append(0.25 * t, t, positions, data)
    writer.close()

    stored = Store(path).particles["cfd_particles"]
    assert len(stored) == 12 and stored.cells is None
    assert stored.info["point_fields"]["fluid velocity"]["array"] == "fluid_velocity"
    np.testing.assert_allclose(stored.times, 0.25 * np.arange(12))
    for t in (0, 5, 11):
        positions, data = _cloud(t)
        assert stored.count(t) == len(positions)
        assert np.array_equal(stored.positions(t), positions)
        for name, values in data.items():
            got = stored.point_field(name, t)
            assert got.dtype == values.dtype and np.array_equal(got, values)

    # appending later goes on where it stopped
    writer = StoreWriter.open(path)
    cloud = writer.particle_writer("cfd_particles")
    positions, data = _cloud(12)
    cloud.append(3.0, 12, positions, data)
    writer.close()
    stored = Store(path).particles["cfd_particles"]
    assert len(stored) == 13 and np.array_equal(stored.positions(12), positions)
    assert np.array_equal(stored.positions(11), _cloud(11)[0])


def test_objects_keep_their_cells_once(tmp_path):
    """Two boxes (DEM-like): the same polygons every output, one value per cell."""
    path = tmp_path / "s.lagoon"
    writer = StoreWriter.create(path, "DEM")
    sets = writer.particle_writer(
        "dem_dem",
        {
            "point_fields": {"velocity": (3, "float32")},
            "cells": "polys",
            "cell_fields": {"id": (1, "int32")},
        },
    )
    quads = np.arange(16, dtype=np.int32)
    offsets = np.array([4, 8, 12, 16], np.int32)
    for t in range(10):
        points = np.random.default_rng(t).uniform(size=(16, 3)).astype(np.float32)
        conn, offs = (quads, offsets) if t < 6 else (quads[:12], offsets[:3])  # one breaks off
        sets.append(
            t,
            t,
            points,
            {"velocity": points},
            conn,
            offs,
            {"id": np.arange(len(offs), dtype=np.int32)},
        )
    writer.close()
    stored = Store(path).particles["dem_dem"]
    assert stored.cells == "polys"
    assert list(stored.steps) == list(range(10))
    # two cell sets were stored for ten outputs
    assert stored.group.array("cell_sets/cell_end").shape[0] == 2
    conn, offs = stored.cell_set(3)
    assert np.array_equal(conn, quads) and np.array_equal(offs, offsets)
    conn, offs = stored.cell_set(8)
    assert np.array_equal(conn, quads[:12]) and len(offs) == 3
    assert np.array_equal(stored.cell_field("id", 8), [0, 1, 2])
    assert np.array_equal(stored.cell_field("id", 2), [0, 1, 2, 3])


def test_wrong_sizes_are_refused(tmp_path):
    writer = StoreWriter.create(tmp_path / "s.lagoon", "CFD")
    cloud = writer.particle_writer("p", {"point_fields": {"radius": (1, "float32")}})
    with pytest.raises(ValueError):
        cloud.append(0.0, 0, np.zeros((3, 3)), {"radius": np.zeros(2)})
    with pytest.raises(ValueError):
        cloud.append(0.0, 0, np.zeros((3, 3)), {})
