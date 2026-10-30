# SPDX-License-Identifier: GPL-3.0-or-later
# Architect: Hans Bihs
"""Particles drawn as spheres: a small round mesh at every particle.

The spheres are subdivided icosahedra, as fine as their number allows (a few
thousand particles: 162 corners each, round even close up; hundreds of thousands:
12), with normals pointing out so that they shade smoothly. Being triangles like
any surface, they are lit, cast shadows and are coloured by the particles' values.
"""

from __future__ import annotations

from functools import cache

import numpy as np

#: about the most corners all spheres together get: the finest sphere whose
#: corners times the particles stay below this is used (12 corners at least)
CORNER_BUDGET = 4_000_000


def _icosahedron() -> tuple[np.ndarray, np.ndarray]:
    t = (1 + 5**0.5) / 2
    v = np.array(
        [
            (-1, t, 0), (1, t, 0), (-1, -t, 0), (1, -t, 0),
            (0, -1, t), (0, 1, t), (0, -1, -t), (0, 1, -t),
            (t, 0, -1), (t, 0, 1), (-t, 0, -1), (-t, 0, 1),
        ],
        float,
    )  # fmt: skip
    f = np.array(
        [
            (0, 11, 5), (0, 5, 1), (0, 1, 7), (0, 7, 10), (0, 10, 11),
            (1, 5, 9), (5, 11, 4), (11, 10, 2), (10, 7, 6), (7, 1, 8),
            (3, 9, 4), (3, 4, 2), (3, 2, 6), (3, 6, 8), (3, 8, 9),
            (4, 9, 5), (2, 4, 11), (6, 2, 10), (8, 6, 7), (9, 8, 1),
        ],
        np.int64,
    )  # fmt: skip
    return v / np.linalg.norm(v, axis=1, keepdims=True), f


@cache
def icosphere(subdivisions: int) -> tuple[np.ndarray, np.ndarray]:
    """A unit sphere: corners (n, 3) on it and triangles (m, 3) facing out; each
    subdivision splits every triangle into four (12, 42, 162, 642 corners)."""
    corners, faces = _icosahedron()
    corners = list(corners)

    def between(a: int, b: int, middle: dict[tuple[int, int], int]) -> int:
        """The corner halfway between two, on the sphere (made once per edge)."""
        key = (min(a, b), max(a, b))
        if key not in middle:
            point = corners[a] + corners[b]
            corners.append(point / np.linalg.norm(point))
            middle[key] = len(corners) - 1
        return middle[key]

    for _ in range(int(subdivisions)):
        middle: dict[tuple[int, int], int] = {}
        finer = []
        for a, b, c in faces:
            ab, bc, ca = between(a, b, middle), between(b, c, middle), between(c, a, middle)
            finer += [(a, ab, ca), (b, bc, ab), (c, ca, bc), (ab, bc, ca)]
        faces = np.array(finer, np.int64)
    out = np.array(corners, np.float64)
    out.setflags(write=False)
    faces = np.asarray(faces, np.int64)
    faces.setflags(write=False)
    return out, faces


def subdivisions_for(count: int, budget: int = CORNER_BUDGET) -> int:
    """How finely spheres for ``count`` particles are made: 2 (162 corners) while
    they fit the budget, else 1 (42), else 0 (12)."""
    for level in (2, 1):
        if count * len(icosphere(level)[0]) <= budget:
            return level
    return 0


def sphere_mesh(
    centres: np.ndarray,
    radii,
    values: np.ndarray | None = None,
    subdivisions: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray | None]:
    """Spheres at points (n, 3) with a radius (one, or one per point): their corners
    (float32), triangles (uint32), normals (float32) and each particle's value at
    its corners (None without values)."""
    centres = np.asarray(centres, np.float64).reshape(-1, 3)
    n = len(centres)
    level = subdivisions_for(n) if subdivisions is None else int(subdivisions)
    unit, faces = icosphere(level)
    k = len(unit)
    r = np.broadcast_to(np.asarray(radii, np.float64).reshape(-1), (n,)) if n else np.zeros(0)
    positions = centres[:, None, :] + unit[None] * r[:, None, None]
    triangles = faces[None] + (np.arange(n, dtype=np.int64) * k)[:, None, None]
    normals = np.broadcast_to(unit[None], (n, k, 3))
    corner_values = None
    if values is not None:
        corner_values = np.repeat(np.asarray(values, np.float32).reshape(n, -1), k, axis=0)
        if corner_values.shape[1] == 1:
            corner_values = corner_values[:, 0]
    return (
        np.ascontiguousarray(positions.reshape(-1, 3), dtype=np.float32),
        np.ascontiguousarray(triangles.reshape(-1, 3), dtype=np.uint32),
        np.ascontiguousarray(normals.reshape(-1, 3), dtype=np.float32),
        corner_values,
    )
