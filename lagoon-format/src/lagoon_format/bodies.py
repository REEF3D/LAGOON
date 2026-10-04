# SPDX-License-Identifier: MIT
# Architect: Hans Bihs
"""Floating bodies in a LAGOON store (SPEC.md, section 9).

REEF3D writes the whole triangle mesh of every 6DOF body at every output. A body
moves as a rigid body, so the store keeps its mesh once and, per output, the
rigid motion that takes the stored vertices to where they are: a translation and
a rotation (unit quaternion). The motion is fitted to the written vertices and
checked: a body whose vertices are not where a rigid motion puts them (to a few
float32 steps) keeps its vertices per output instead.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from lagoon_format.zarr3 import Array, Group, codecs_for

#: outputs per chunk of the motion arrays
MOTION_ROWS = 4096
#: a fitted motion is accepted when no vertex is further than this many float32
#: steps (of the largest coordinate) from where it was written
RIGID_ULPS = 4.0


# ======================================================================== rigid motion
def fit_motion(reference: np.ndarray, points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The rigid motion (rotation matrix, translation) that moves ``reference`` onto
    ``points`` best in the least squares sense (Kabsch), in float64."""
    a = np.asarray(reference, np.float64)
    b = np.asarray(points, np.float64)
    ca, cb = a.mean(axis=0), b.mean(axis=0)
    h = (a - ca).T @ (b - cb)
    u, _s, vt = np.linalg.svd(h)
    d = 1.0 if np.linalg.det(vt.T @ u.T) >= 0 else -1.0
    rotation = vt.T @ np.diag([1.0, 1.0, d]) @ u.T
    return rotation, cb - rotation @ ca


def quaternion(matrix: np.ndarray) -> np.ndarray:
    """Unit quaternion (w, x, y, z) of a rotation matrix, w >= 0."""
    m = np.asarray(matrix, np.float64)
    trace = m[0, 0] + m[1, 1] + m[2, 2]
    if trace > 0:
        s = 2.0 * np.sqrt(trace + 1.0)
        q = [0.25 * s, (m[2, 1] - m[1, 2]) / s, (m[0, 2] - m[2, 0]) / s, (m[1, 0] - m[0, 1]) / s]
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = 2.0 * np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2])
        q = [(m[2, 1] - m[1, 2]) / s, 0.25 * s, (m[0, 1] + m[1, 0]) / s, (m[0, 2] + m[2, 0]) / s]
    elif m[1, 1] > m[2, 2]:
        s = 2.0 * np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2])
        q = [(m[0, 2] - m[2, 0]) / s, (m[0, 1] + m[1, 0]) / s, 0.25 * s, (m[1, 2] + m[2, 1]) / s]
    else:
        s = 2.0 * np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1])
        q = [(m[1, 0] - m[0, 1]) / s, (m[0, 2] + m[2, 0]) / s, (m[1, 2] + m[2, 1]) / s, 0.25 * s]
    q = np.asarray(q, np.float64)
    q /= np.linalg.norm(q)
    return -q if q[0] < 0 else q


def rotation_matrix(q: np.ndarray) -> np.ndarray:
    """Rotation matrix of a quaternion (w, x, y, z); normalised first."""
    w, x, y, z = np.asarray(q, np.float64) / np.linalg.norm(q)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ]
    )


def moved(vertices: np.ndarray, translation: np.ndarray, rotation: np.ndarray) -> np.ndarray:
    """The vertices after the motion (quaternion ``rotation``), as float32."""
    matrix = rotation_matrix(rotation)
    return (np.asarray(vertices, np.float64) @ matrix.T + translation).astype(np.float32)


def rigid_tolerance(points: np.ndarray) -> float:
    """How far a vertex may be from the fitted motion: a few float32 steps of the
    largest coordinate (REEF3D writes float32)."""
    largest = np.float32(np.abs(points).max()) if points.size else np.float32(1.0)
    return float(RIGID_ULPS * np.spacing(max(largest, np.float32(1.0))))


def check_motion(
    reference: np.ndarray, points: np.ndarray
) -> tuple[np.ndarray, np.ndarray, float, bool]:
    """Fit the motion of ``points`` from ``reference``: (translation, quaternion,
    largest vertex error, rigid?)."""
    matrix, translation = fit_motion(reference, points)
    q = quaternion(matrix)
    error = float(np.abs(moved(reference, translation, q) - points).max()) if points.size else 0.0
    return translation, q, error, error <= rigid_tolerance(points)


# ======================================================================== reading
@dataclass
class Body:
    """One body of a body set: its mesh and how it moves."""

    group: Group
    number: int
    rigid: bool
    info: dict[str, Any] = field(default_factory=dict)
    _vertices: np.ndarray | None = None
    _triangles: np.ndarray | None = None
    _motion: tuple[np.ndarray, np.ndarray] | None = None

    @property
    def triangles(self) -> np.ndarray:
        if self._triangles is None:
            self._triangles = self.group.array("triangles")[:]
        return self._triangles

    @property
    def fields(self) -> dict[str, int]:
        """Point arrays stored per output: name -> components."""
        return {k: int(v) for k, v in self.info.get("fields", {}).items()}

    def reference(self) -> np.ndarray:
        """The stored vertices (of the first output), rigid bodies only."""
        if self._vertices is None:
            self._vertices = self.group.array("vertices")[:]
        return self._vertices

    def motion(self) -> tuple[np.ndarray, np.ndarray]:
        """Translation (nt, 3) and rotation (nt, 4) of all outputs."""
        if self._motion is None:
            self._motion = (self.group.array("translation")[:], self.group.array("rotation")[:])
        return self._motion

    def vertices(self, t: int) -> np.ndarray:
        """The vertices at output t (float32, as REEF3D wrote them to a few float32 steps)."""
        if not self.rigid:
            return self.group.array("vertices")[t]
        translation, rotation = self.motion()
        if t >= len(translation):  # appended since: read again
            self._motion = None
            translation, rotation = self.motion()
        return moved(self.reference(), translation[t], rotation[t])

    def field(self, name: str, t: int) -> np.ndarray:
        return self.group.array(name)[t]


@dataclass
class BodySet:
    """The bodies of one 6DOF output (one group ``bodies/<key>``)."""

    group: Group
    info: dict[str, Any]
    bodies: dict[int, Body]

    @classmethod
    def open(cls, group: Group) -> BodySet:
        info = group.attrs.get("lagoon", {})
        bodies = {}
        for number in info.get("bodies", []):
            body_group = group.group(f"body_{int(number)}")
            body_info = body_group.attrs.get("lagoon", {})
            bodies[int(number)] = Body(
                body_group, int(number), bool(body_info.get("rigid", True)), body_info
            )
        return cls(group, info, bodies)

    @property
    def key(self) -> str:
        return self.group.path.name

    def _count(self) -> int:
        return self.group.array("time").shape[0]

    @property
    def times(self) -> np.ndarray:
        time = self.group.array("time")
        return time[: time.shape[0]]

    @property
    def steps(self) -> np.ndarray:
        n = self._count()  # time is extended last: it counts the complete outputs
        return self.group.array("step")[:n]

    def __len__(self) -> int:
        return self._count()

    def mesh(self, t: int) -> tuple[np.ndarray, np.ndarray]:
        """All bodies at output t as one mesh: vertices (n, 3) float32, triangles (m, 3)."""
        points, triangles = [], []
        offset = 0
        for number in sorted(self.bodies):
            body = self.bodies[number]
            vertices = body.vertices(t)
            points.append(vertices)
            triangles.append(body.triangles.astype(np.int64) + offset)
            offset += vertices.shape[0]
        if not points:
            return np.zeros((0, 3), np.float32), np.zeros((0, 3), np.int64)
        return np.concatenate(points), np.concatenate(triangles)


# ======================================================================== writing
def _codecs(itemsize: int):
    return codecs_for("gzip", 1, itemsize)


class BodyWriter:
    """Appends outputs to a body set (one group ``bodies/<key>``)."""

    @classmethod
    def create(
        cls,
        parent: Group,
        key: str,
        bodies: dict[int, tuple[np.ndarray, np.ndarray, dict[str, int], bool]],
        source: str = "",
        attributes: dict[str, Any] | None = None,
    ) -> BodyWriter:
        """``bodies``: number -> (vertices of the first output (n, 3), triangles (m, 3),
        point arrays (name -> components), rigid?). Rigid bodies store these vertices
        as their mesh; the others get their vertices per output."""
        info: dict[str, Any] = {"kind": "bodies", "bodies": sorted(int(n) for n in bodies)}
        if source:
            info["source"] = source
        info.update(attributes or {})
        group = Group.create(parent.path / key, {"lagoon": info})
        for name, dtype in (("time", "float64"), ("step", "int64")):
            Array.create(
                group.path / name,
                [0],
                dtype,
                [MOTION_ROWS],
                dimension_names=["time"],
                attributes={"units": "s"} if name == "time" else {},
            )
        for number, (vertices, triangles, fields, rigid) in bodies.items():
            body = group.path / f"body_{int(number)}"
            vertices = np.asarray(vertices, np.float32)
            n = vertices.shape[0]
            Group.create(
                body,
                {"lagoon": {"rigid": bool(rigid), "vertices": n, "fields": dict(fields)}},
            )
            triangles = np.asarray(triangles, np.int32).reshape(-1, 3)
            Array.create(
                body / "triangles",
                list(triangles.shape),
                "int32",
                [max(triangles.shape[0], 1), 3],
                codecs=_codecs(4),
                dimension_names=["triangle", "corner"],
            )[:] = triangles
            if rigid:
                Array.create(
                    body / "vertices",
                    [n, 3],
                    "float32",
                    [max(n, 1), 3],
                    codecs=_codecs(4),
                    dimension_names=["vertex", "xyz"],
                    attributes={"units": "m"},
                )[:] = vertices
                for name, width in (("translation", 3), ("rotation", 4)):
                    Array.create(
                        body / name,
                        [0, width],
                        "float64",
                        [MOTION_ROWS, width],
                        codecs=_codecs(8),
                        dimension_names=["time", "xyz" if width == 3 else "wxyz"],
                        attributes={"units": "m"} if width == 3 else {},
                    )
            else:
                Array.create(
                    body / "vertices",
                    [0, n, 3],
                    "float32",
                    [1, max(n, 1), 3],
                    codecs=_codecs(4),
                    dimension_names=["time", "vertex", "xyz"],
                    attributes={"units": "m"},
                )
            for name, components in fields.items():
                shape = [0, n, components] if components > 1 else [0, n]
                Array.create(
                    body / name,
                    shape,
                    "float32",
                    [1, *shape[1:]] if n else [1, 1],
                    codecs=_codecs(4),
                    dimension_names=["time", "vertex", "component"][: len(shape)],
                )
        return cls(group)

    def __init__(self, group: Group):
        self.group = group
        self.stored = BodySet.open(group)
        self._count = len(self.stored)
        self._rows: list[tuple[float, int, dict[int, tuple[np.ndarray, np.ndarray]]]] = []
        self._errors: dict[int, float] = {}

    def append(
        self,
        time: float,
        step: int,
        bodies: dict[int, tuple[np.ndarray, dict[str, np.ndarray]]],
    ) -> float:
        """One output: number -> (vertices (n, 3), point arrays). Returns the largest
        vertex error of the rigid bodies' fitted motions. Raises :class:`NotRigid`
        when a rigid body's vertices are not where a rigid motion puts them, and
        ValueError when a body is missing or has another number of vertices.

        Motions are kept until :meth:`flush` (or ``flush_every`` outputs); vertices
        of non-rigid bodies and point arrays are written at once."""
        t = self._count + len(self._rows)
        motions = {}
        errors = {}
        for number, body in self.stored.bodies.items():
            if number not in bodies:
                raise ValueError(f"output {step}: body {number} is missing")
            vertices = np.asarray(bodies[number][0], np.float32)
            if vertices.shape[0] != int(body.info.get("vertices", -1)):
                raise ValueError(f"output {step}: body {number} has another mesh")
            if body.rigid:
                translation, rotation, error, rigid = check_motion(body.reference(), vertices)
                if not rigid:
                    raise NotRigid(number, step, error)
                motions[number] = (translation, rotation)
                errors[number] = error
        for number, body in self.stored.bodies.items():
            vertices, fields = bodies[number]
            if not body.rigid:
                _put(body.group.array("vertices"), t, np.asarray(vertices, np.float32))
            for name in body.fields:
                array = body.group.array(name)
                _put(array, t, np.asarray(fields[name], np.float32).reshape(array.shape[1:]))
        for number, error in errors.items():
            self._errors[number] = max(self._errors.get(number, 0.0), error)
        self._rows.append((float(time), int(step), motions))
        if len(self._rows) >= MOTION_ROWS:
            self.flush()
        return max(errors.values(), default=0.0)

    def flush(self) -> None:
        """Write the kept motions, then the times (which count the outputs)."""
        if not self._rows:
            return
        start, stop = self._count, self._count + len(self._rows)
        for number, body in self.stored.bodies.items():
            if not body.rigid:
                continue
            for k, name in enumerate(("translation", "rotation")):
                array = body.group.array(name)
                array.resize([stop, array.shape[1]])
                array[start:stop] = np.stack([row[2][number][k] for row in self._rows])
            info = dict(body.info)
            info["max_error"] = max(
                float(info.get("max_error", 0.0)), self._errors.get(number, 0.0)
            )
            body.info = info
            body.group.set_attrs({"lagoon": info})
        for name, column, dtype in (("step", 1, np.int64), ("time", 0, np.float64)):
            array = self.group.array(name)  # time last: readers count by it
            array.resize([stop])
            array[start:stop] = np.asarray([row[column] for row in self._rows], dtype)
        self._count = stop
        self._rows = []
        self._errors = {}

    def discard(self) -> None:
        """Forget the motions not written yet (the set is written again)."""
        self._rows = []
        self._errors = {}

    def close(self) -> None:
        self.flush()


def _put(array: Array, t: int, values: np.ndarray) -> None:
    """Row t of an array that grows by one output at a time."""
    if array.shape[0] <= t:
        array.resize([t + 1, *array.shape[1:]])
    array[t : t + 1] = values[None]


class NotRigid(ValueError):
    """A body declared rigid moved in a way no rigid motion describes."""

    def __init__(self, number: int, step: int, error: float):
        super().__init__(
            f"body {number} at output {step}: vertices {error:.3g} m off a rigid motion"
        )
        self.number = number
        self.step = step
        self.error = error
