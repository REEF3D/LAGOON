"""Floating bodies in a LAGOON store (SPEC 9): mesh once, a rigid motion per output."""

from __future__ import annotations

import numpy as np
import pytest
from lagoon_format.bodies import (
    NotRigid,
    check_motion,
    moved,
    quaternion,
    rigid_tolerance,
    rotation_matrix,
)
from lagoon_format.store import Store, StoreWriter


def _rotation(angle_z: float, angle_x: float) -> np.ndarray:
    cz, sz, cx, sx = np.cos(angle_z), np.sin(angle_z), np.cos(angle_x), np.sin(angle_x)
    rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    return rz @ rx


def _ship(offset=(700.0, 40.0, 5.0)) -> tuple[np.ndarray, np.ndarray]:
    """A triangle soup (as REEF3D writes it) far from the origin."""
    rng = np.random.default_rng(1)
    corners = rng.uniform(-20, 20, (30000, 3)) * [3, 1, 0.5] + offset
    return corners.astype(np.float32), np.arange(30000).reshape(-1, 3)


def _motion(t: int, centre: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    r = _rotation(0.02 * t, 0.01 * np.sin(t))
    shift = np.array([0.5 * t, 0.1 * t, np.sin(t)])
    return r, centre + shift - r @ centre


def test_quaternions_round_trip():
    for angles in ((0.0, 0.0), (3.0, 0.2), (np.pi, 0.0), (-2.5, 3.1), (0.1, np.pi)):
        r = _rotation(*angles)
        q = quaternion(r)
        assert q[0] >= 0 and np.isclose(np.linalg.norm(q), 1.0)
        assert np.allclose(rotation_matrix(q), r, atol=1e-12)


def test_rigid_motion_is_found_to_float32_precision():
    vertices, _ = _ship()
    r, shift = _motion(7, vertices.astype(np.float64).mean(axis=0))
    written = (vertices.astype(np.float64) @ r.T + shift).astype(np.float32)
    translation, q, error, rigid = check_motion(vertices, written)
    assert rigid and error <= rigid_tolerance(written)
    assert np.abs(moved(vertices, translation, q) - written).max() <= rigid_tolerance(written)
    bent = written.copy()
    bent[0] += 0.01  # one vertex 1 cm off
    assert not check_motion(vertices, bent)[3]


def test_bodies_in_a_store(tmp_path):
    vertices, triangles = _ship()
    centre = vertices.astype(np.float64).mean(axis=0)
    path = tmp_path / "REEF3D_NHFLOW.lagoon"
    writer = StoreWriter.create(path, "NHFLOW")
    bodies = writer.body_writer(
        "nhflow_body",
        {0: (vertices, triangles, {}, True)},
        source="REEF3D_NHFLOW_6DOF_VTP",
        attributes={"dataset": "nhflow_body"},
    )
    written = []
    for t in range(40):
        r, shift = _motion(t, centre)
        points = (vertices.astype(np.float64) @ r.T + shift).astype(np.float32)
        written.append(points)
        bodies.append(0.5 * t, t, {0: (points, {})})
    writer.close()

    store = Store(path)
    body_set = store.bodies["nhflow_body"]
    assert len(body_set) == 40 and body_set.info["dataset"] == "nhflow_body"
    assert list(body_set.steps) == list(range(40)) and body_set.times[-1] == 19.5
    body = body_set.bodies[0]
    assert body.rigid and body.info["max_error"] <= rigid_tolerance(written[-1])
    for t in (0, 13, 39):
        points, tris = body_set.mesh(t)
        assert np.abs(points - written[t]).max() <= rigid_tolerance(written[t])
        assert np.array_equal(tris, triangles)
    # the mesh once and seven numbers per output: far less than the vertices
    size = sum(p.stat().st_size for p in path.rglob("*") if p.is_file())
    assert size < 0.1 * 40 * vertices.nbytes

    # appended later, and a body that bends is refused
    writer = StoreWriter.open(path)
    bodies = writer.body_writer("nhflow_body")
    r, shift = _motion(40, centre)
    bodies.append(
        20.0, 40, {0: ((vertices.astype(np.float64) @ r.T + shift).astype(np.float32), {})}
    )
    bent = written[0].copy()
    bent[:10] *= 1.01
    with pytest.raises(NotRigid):
        bodies.append(20.5, 41, {0: (bent, {})})
    writer.close()
    assert len(Store(path).bodies["nhflow_body"]) == 41


def test_deforming_bodies_keep_their_vertices(tmp_path):
    vertices, triangles = _ship()
    path = tmp_path / "s.lagoon"
    writer = StoreWriter.create(path, "CFD")
    bodies = writer.body_writer("cfd_body", {0: (vertices, triangles, {"pressure": 1}, False)})
    frames = []
    for t in range(5):
        points = vertices * np.float32(1 + 0.01 * t)
        pressure = np.full(len(vertices), t, np.float32)
        frames.append(points)
        bodies.append(float(t), t, {0: (points, {"pressure": pressure})})
    writer.close()
    body = Store(path).bodies["cfd_body"].bodies[0]
    assert not body.rigid and body.fields == {"pressure": 1}
    assert np.array_equal(body.vertices(3), frames[3])
    assert np.array_equal(body.field("pressure", 4), np.full(len(vertices), 4, np.float32))
