# SPDX-License-Identifier: GPL-3.0-or-later
# Architect: Hans Bihs
"""Path-traced images and films: the scene handed to headless Blender (Cycles).

    lagoon-render run/ waves.mp4 --engine cycles --style open_ocean --samples 128
    path_trace(session, "showcase.mp4", size=(1920, 1080), look="open_ocean")

LAGOON writes what the 3D view shows into a folder: ``scene.json`` (size,
samples, the look's world, sun and materials, and per frame the camera and the
objects), a ``frame_NNNNN.npz`` of triangles per frame (lines become thin tubes,
particles small spheres, colours already mapped) and the surroundings as an
equirectangular ``environment.hdr``. Blender then runs as a separate process,
``blender -b --factory-startup -P blender_cycles.py -- <folder>``, builds each
frame and renders it with Cycles into ``frame_NNNNN.png`` beside the data; LAGOON
encodes the PNGs into the film. LAGOON never imports Blender, and Blender needs
nothing of LAGOON (the script uses its own Python and NumPy).

The camera is the 3D view's: the camera path at each frame's time when it is on,
else the saved view, else LAGOON's default perspective. Blender 4.2 or newer is
found through ``$LAGOON_BLENDER``, the ``PATH`` or the usual install places.

Everything here is NumPy only; :func:`render_folder` starts Blender.
"""

from __future__ import annotations

import functools
import hashlib
import json
import logging
import math
import os
import shutil
import subprocess
import sys
import threading
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

from lagoonviz.core.colormaps import colors
from lagoonviz.render.cinematic import (
    PRESETS,
    SURFACES,
    Preset,
    environment_panorama,
    look_colour,
    panorama_directions,
    surface_of,
    with_sun,
)

log = logging.getLogger("lagoon.pathtrace")

#: the script Blender runs (it is not imported here)
BLENDER_SCRIPT = Path(__file__).with_name("blender_cycles.py")
#: the version of the folder layout (scene.json "version")
VERSION = 2
#: LAGOON's own looks, path traced in the cinematic look nearest to them
LOOK_FOR = {"studio": "wave_flume", "ocean": "open_ocean"}
#: the default view: the direction the camera looks (as LAGOON's "perspective" view)
PERSPECTIVE = (-1.0, 1.2, -0.7)
#: the field of view [degrees] when the session has no camera
DEFAULT_FOV = 50.0
#: line widths [pixels] of the overlays, as the 3D view draws them
CONTOUR_WIDTH = 1.2
VECTOR_WIDTH = 1.4
TRAIL_WIDTH = 1.0
OUTLINE_WIDTH = 1.3
STAFF_WIDTH = 2.5
DOT_SIZE = 8.0
#: the sea around the domain reaches this many domain sizes out (to the horizon)
SEA_REACH = 400.0
#: the name of the sea's object
SEA = "LAGOON sea"
#: surfaces up to this many triangles are subdivided once in Blender (smooth crests)
#: materials whose numbers differ less than this (relatively) are taken as one: the
#: water's depth-scaled values change a little from frame to frame, and every new
#: material is a new shader for Cycles to set up
MATERIAL_TOLERANCE = 0.03
#: the haze inside the water (ÆgirFrames' ocean water): a principled volume that
#: mostly absorbs, a little blue-green, per metre
AEGIR_HAZE = {"color": [0.0428, 0.0658, 0.0732], "density": 0.01, "absorption": [0.0, 0.0, 0.0]}
#: materials of white water, as ÆgirFrames' diffuse particles: light grey, half
#: see-through
WHITE_WATER: dict[str, dict[str, Any]] = {
    kind: {
        "color": [0.8, 0.8, 0.8],
        "roughness": 0.5,
        "metallic": 0.0,
        "ior": 1.33,
        "transmission": 0.5,
        "coat": 0.0,
        "alpha": 0.7,
        "grain": None,
        "white_water": kind,
    }
    for kind in ("foam", "spray", "bubbles")
}
#: sky pictures (HDR panoramas) for path tracing
SKY_FOLDER = Path(__file__).with_name("assets")
#: the sun lamp's strength under a sky picture (its own sun lights the scene)
SKY_PICTURE_SUN = 1.0
#: ``progress(done, total) -> keep going``
Progress = Callable[[int, int], bool]


# ======================================================================== colours
def _hex(color: str) -> np.ndarray:
    color = color.lstrip("#")
    return np.array([int(color[i : i + 2], 16) / 255 for i in (0, 2, 4)], float)


def linear(rgb) -> np.ndarray:
    """sRGB values (0..1) as physical (linear) light, as Blender wants colours."""
    c = np.clip(np.asarray(rgb, float), 0.0, 1.0)
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def linear_hex(color: str) -> list[float]:
    return [round(float(v), 6) for v in linear(_hex(color))]


def write_hdr(path: str | Path, rgb: np.ndarray) -> None:
    """Write (h, w, 3) linear light as a Radiance .hdr (RGBE, run-length scanlines;
    widths 8 to 32767)."""
    rgb = np.maximum(np.asarray(rgb, np.float64), 0.0)
    height, width, _ = rgb.shape
    if not 8 <= width <= 32767:
        raise ValueError("Radiance scanlines need a width of 8 to 32767")
    brightest = rgb.max(axis=-1)
    mantissa, exponent = np.frexp(brightest)
    scale = np.where(brightest > 1e-32, 256.0 / np.exp2(exponent), 0.0)
    rgbe = np.zeros((height, width, 4), np.uint8)
    rgbe[..., :3] = np.clip(np.floor(rgb * scale[..., None]), 0, 255).astype(np.uint8)
    rgbe[..., 3] = np.where(brightest > 1e-32, exponent + 128, 0).astype(np.uint8)
    header = f"#?RADIANCE\nFORMAT=32-bit_rle_rgbe\n\n-Y {height} +X {width}\n".encode()
    start = bytes((2, 2, width >> 8, width & 255))
    chunks = [header]
    for row in rgbe:
        chunks.append(start)
        for channel in range(4):
            chunks.append(_rle(row[:, channel]))
    Path(path).write_bytes(b"".join(chunks))


def _rle(values: np.ndarray) -> bytes:
    """One channel of a scanline, run-length coded as Radiance does: runs of one
    value (3 to 127 long) as count + 128 and the value, the rest as literal pieces of
    up to 128 bytes."""
    values = np.asarray(values, np.uint8)
    n = len(values)
    # where runs of equal values start, and how long they are
    change = np.flatnonzero(np.diff(values.astype(np.int16))) + 1
    starts = np.concatenate([[0], change])
    lengths = np.diff(np.concatenate([starts, [n]]))
    out = bytearray()
    literal_from = 0
    for begin, length in zip(starts.tolist(), lengths.tolist(), strict=True):
        if length < 3:
            continue
        _literal(out, values, literal_from, begin)
        value = int(values[begin])
        while length >= 3:
            count = min(length, 127)
            out += bytes((128 + count, value))
            begin += count
            length -= count
        literal_from = begin  # a tail of 1 or 2 joins the next literal piece
    _literal(out, values, literal_from, n)
    return bytes(out)


def _literal(out: bytearray, values: np.ndarray, start: int, end: int) -> None:
    for at in range(start, end, 128):
        piece = values[at : min(at + 128, end)].tobytes()
        out += bytes((len(piece),)) + piece


def read_hdr(path: str | Path) -> np.ndarray:
    """Read a Radiance .hdr written by :func:`write_hdr` (and other run-length ones)
    as (h, w, 3) float32 linear light."""
    data = Path(path).read_bytes()
    end = data.index(b"\n\n") + 2
    size_line, _, rest = data[end:].partition(b"\n")
    _, height, _, width = size_line.split()
    height, width = int(height), int(width)
    rgbe = np.zeros((height, width, 4), np.uint8)
    at = 0
    for y in range(height):
        at += 4  # 2, 2, width
        for channel in range(4):
            x = 0
            while x < width:
                count = rest[at]
                at += 1
                if count > 128:  # a run of one value
                    count -= 128
                    rgbe[y, x : x + count, channel] = rest[at]
                    at += 1
                else:
                    rgbe[y, x : x + count, channel] = np.frombuffer(rest[at : at + count], np.uint8)
                    at += count
                x += count
    exponent = rgbe[..., 3].astype(np.float64)
    scale = np.where(exponent > 0, np.exp2(exponent - 136.0), 0.0)
    return (rgbe[..., :3] * scale[..., None]).astype(np.float32)


def subdivision_levels(triangles: int, pixels: int, most: int = 2) -> int:
    """How often to subdivide a smooth surface (each time a triangle becomes six), at
    most ``most`` times, as long as it stays at no more than one triangle for each
    pixel of the image: finer is not seen. A coarse lab mesh gets two levels (as
    ÆgirFrames uses), a fine coast mesh one or none."""
    levels = 0
    while levels < most and triangles * 6 ** (levels + 1) <= pixels:
        levels += 1
    return levels


def same_material(a: dict, b: dict, tolerance: float = MATERIAL_TOLERANCE) -> bool:
    """Two material specs alike: the same keys and texts, numbers within a tolerance."""
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(same_material(a[k], b[k], tolerance) for k in a)
    if isinstance(a, list | tuple) and isinstance(b, list | tuple):
        if len(a) != len(b):
            return False
        return all(same_material(x, y, tolerance) for x, y in zip(a, b, strict=True))
    if isinstance(a, bool) or isinstance(b, bool) or a is None or b is None:
        return type(a) is type(b) and a == b
    if isinstance(a, int | float) and isinstance(b, int | float):
        return abs(a - b) <= tolerance * max(abs(a), abs(b)) + 1e-12
    return a == b


# ======================================================================== geometry
def tubes(segments: np.ndarray, radius: float, sides: int = 6) -> tuple[np.ndarray, np.ndarray]:
    """Thin open tubes around line segments (k, 2, 3): (positions (k * 2 * sides, 3),
    triangles (k * 2 * sides, 3)). Zero-length segments give flat slivers."""
    seg = np.asarray(segments, np.float64).reshape(-1, 2, 3)
    k = len(seg)
    if not k:
        return np.zeros((0, 3), np.float32), np.zeros((0, 3), np.uint32)
    axis = seg[:, 1] - seg[:, 0]
    length = np.linalg.norm(axis, axis=1, keepdims=True)
    axis = np.where(length > 1e-12, axis / np.maximum(length, 1e-12), [0.0, 0.0, 1.0])
    # any direction across the axis, then a second one across both
    helper = np.where(np.abs(axis[:, 2:3]) < 0.9, [[0.0, 0.0, 1.0]], [[1.0, 0.0, 0.0]])
    u = np.cross(axis, helper)
    u /= np.linalg.norm(u, axis=1, keepdims=True)
    v = np.cross(axis, u)
    angles = np.arange(sides) * 2 * np.pi / sides
    ring = np.cos(angles)[None, :, None] * u[:, None] + np.sin(angles)[None, :, None] * v[:, None]
    ring *= radius
    positions = np.concatenate([seg[:, 0:1] + ring, seg[:, 1:2] + ring], axis=1)  # (k, 2s, 3)
    a = np.arange(sides)
    b = (a + 1) % sides
    local = np.concatenate(
        [np.stack([a, b, a + sides], 1), np.stack([b, b + sides, a + sides], 1)]
    )  # (2s, 3)
    triangles = local[None] + (np.arange(k) * 2 * sides)[:, None, None]
    return (
        positions.reshape(-1, 3).astype(np.float32),
        triangles.reshape(-1, 3).astype(np.uint32),
    )


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
        np.uint32,
    )  # fmt: skip
    return v / np.linalg.norm(v, axis=1, keepdims=True), f


def spheres(centres: np.ndarray, radius) -> tuple[np.ndarray, np.ndarray]:
    """Small spheres (icosahedra, 12 corners each) at points (n, 3) with a radius
    (one, or one per point)."""
    centres = np.asarray(centres, np.float64).reshape(-1, 3)
    corners, faces = _icosahedron()
    r = np.broadcast_to(np.asarray(radius, float), (len(centres),))[:, None, None]
    positions = centres[:, None] + corners[None] * r
    triangles = faces[None] + (np.arange(len(centres)) * len(corners))[:, None, None]
    return (
        positions.reshape(-1, 3).astype(np.float32),
        triangles.reshape(-1, 3).astype(np.uint32),
    )


class BedHeights:
    """The bed's height under any point, never above the bed: the bed vertex at the
    same place where there is one (REEF3D writes the free surface and the bed on the
    same horizontal grid), else the lowest bed vertex near the point (binned on a
    fine grid, empty cells filled from their neighbours). A body of water closed on
    these heights stays under the land everywhere, so it never pokes through it."""

    def __init__(self, points: np.ndarray, cells: int | None = None) -> None:
        p = np.asarray(points, float).reshape(-1, 3)
        p = p[np.isfinite(p).all(axis=1)]
        if not len(p):
            raise ValueError("no bed points")
        self.low = p[:, :2].min(axis=0)
        span = np.maximum(p[:, :2].max(axis=0) - self.low, 1e-9)
        # exact: the vertices by their place (several ranks share seams: the lowest)
        self.quantum = float(span.max()) * 1e-7
        keys = self._keys(p[:, :2])
        order = np.lexsort((p[:, 2], keys[:, 1], keys[:, 0]))
        keys, z = keys[order], p[order, 2]
        first = np.ones(len(keys), bool)
        first[1:] = (keys[1:] != keys[:-1]).any(axis=1)
        self.keys, self.z = keys[first], z[first]  # sorted, lowest z per place
        # near: the lowest bed vertex of a fine grid's cell and its neighbours, about
        # one vertex per cell (cells: at most this many along an axis)
        aspect = float(span[0] / span[1])
        nx = np.sqrt(len(p) * aspect)
        top = 2048 if cells is None else int(cells)
        self.shape = (
            int(np.clip(nx, 16, top)),
            int(np.clip(len(p) / max(nx, 1.0), 16, top)),
        )
        self.size = span / (np.array(self.shape) - 1)
        ij = self._cells(p[:, :2])
        flat = ij[:, 0] * self.shape[1] + ij[:, 1]
        grid = np.full(self.shape[0] * self.shape[1], np.inf)
        np.minimum.at(grid, flat, p[:, 2])
        grid = grid.reshape(self.shape)
        for _ in range(64):  # fill empty cells from their neighbours
            empty = ~np.isfinite(grid)
            if not empty.any():
                break
            grid = np.where(empty, self._lowest_around(grid), grid)
        grid = np.where(np.isfinite(grid), grid, p[:, 2].min())
        self.grid = self._lowest_around(grid)

    @staticmethod
    def _lowest_around(grid: np.ndarray) -> np.ndarray:
        padded = np.pad(grid, 1, constant_values=np.inf)
        rows, cols = grid.shape
        return np.min(
            [padded[i : i + rows, j : j + cols] for i in range(3) for j in range(3)], axis=0
        )

    def _keys(self, xy: np.ndarray) -> np.ndarray:
        return np.round((np.asarray(xy, float) - self.low) / self.quantum).astype(np.int64)

    def _cells(self, xy: np.ndarray) -> np.ndarray:
        ij = np.floor((np.asarray(xy, float) - self.low) / self.size + 0.5).astype(int)
        return np.clip(ij, 0, np.array(self.shape) - 1)

    def __call__(self, xy: np.ndarray) -> np.ndarray:
        xy = np.asarray(xy, float).reshape(-1, 2)
        ij = self._cells(xy)
        heights = self.grid[ij[:, 0], ij[:, 1]]
        # where a bed vertex is at the same place: its height
        keys = self._keys(xy)
        combined = self.keys[:, 0] * (1 << 31) + self.keys[:, 1]
        wanted = keys[:, 0] * (1 << 31) + keys[:, 1]
        at = np.searchsorted(combined, wanted)
        at = np.clip(at, 0, len(combined) - 1)
        exact = combined[at] == wanted
        heights[exact] = self.z[at[exact]]
        return heights


def water_body(
    positions: np.ndarray, triangles: np.ndarray, bottom
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """A closed body of water under a free surface: the surface (facing up), a floor
    (facing down) and walls along the surface's open edges (the sides of the domain
    and the shore of dry land), all facing out. ``bottom`` is the floor's height: a
    number, or a function of (n, 2) points giving heights (the bed, so the water
    ends on it), or a height per corner. Returns positions, triangles and which
    triangles are smooth (the surface; walls and floor flat).

    Path tracing needs the water as a volume: light refracts into it, is absorbed
    on its way through (the deeper, the bluer) and refracts out again; a surface
    alone can only tint."""
    p = np.asarray(positions, np.float64).reshape(-1, 3)
    t = np.asarray(triangles, np.int64).reshape(-1, 3)
    finite = np.isfinite(p).all(axis=1)
    t = t[finite[t].all(axis=1)]
    # hidden triangles stay in LAGOON's lists as degenerate ones (all corners 0)
    t = t[(t[:, 0] != t[:, 1]) & (t[:, 1] != t[:, 2]) & (t[:, 0] != t[:, 2])]
    if not len(t):
        return np.zeros((0, 3), np.float32), np.zeros((0, 3), np.uint32), np.zeros(0, bool)
    t, rim = _facing_up_and_rim(p, t)
    n = len(p)
    base = np.where(finite[:, None], p, 0.0)  # corners of dry land: unused
    depth = max(float(np.ptp(base[finite, 2])) if finite.any() else 1.0, 1e-6)
    if callable(bottom):
        # on the bed, a little under it (the bed hides the floor), never above the water
        heights = bottom(base[:, :2]) - 1e-3 * depth
        heights = np.minimum(heights, base[:, 2] - 1e-4 * depth)
    elif np.ndim(bottom) == 1:  # a height per corner, never above the water
        heights = np.minimum(np.asarray(bottom, float), base[:, 2] - 1e-4 * depth)
    else:
        heights = np.full(n, float(bottom))
    floor = base.copy()
    floor[:, 2] = heights
    # walls: their own corners, so the surface's smooth normals stay its own
    top_a, top_b = p[rim[:, 0]], p[rim[:, 1]]
    low_a, low_b = top_a.copy(), top_b.copy()
    low_a[:, 2] = heights[rim[:, 0]]
    low_b[:, 2] = heights[rim[:, 1]]
    k = len(rim)
    wall_points = np.concatenate([top_a, top_b, low_b, low_a])  # 4 k corners
    i = np.arange(k) + 2 * n
    wall = np.concatenate(
        [np.stack([i, i + k, i + 2 * k], 1), np.stack([i, i + 2 * k, i + 3 * k], 1)]
    )
    # outward: to the right of the edge seen from above (the surface turns
    # counter-clockwise, its water on the left)
    d = top_b - top_a
    out = np.stack([d[:, 1], -d[:, 0], np.zeros(k)], 1)
    all_points = np.concatenate([base, floor, wall_points])
    w0, w1, w2 = all_points[wall[:, 0]], all_points[wall[:, 1]], all_points[wall[:, 2]]
    facing = (np.cross(w1 - w0, w2 - w0) * np.concatenate([out, out])).sum(axis=1)
    wall = np.where(facing[:, None] < 0, wall[:, ::-1], wall)
    floor_triangles = t[:, ::-1] + n  # facing down
    triangles_out = np.concatenate([t, floor_triangles, wall])
    smooth = np.concatenate([np.ones(len(t), bool), np.zeros(len(t) + len(wall), bool)])
    return all_points.astype(np.float32), triangles_out.astype(np.uint32), smooth


def _facing_up_and_rim(p: np.ndarray, t: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The triangles turned to face up, and their open edges (used by one triangle
    only, directed as in their triangle: the surface on their left seen from above)."""
    a, b, c = p[t[:, 0]], p[t[:, 1]], p[t[:, 2]]
    up = np.cross(b - a, c - a)[:, 2] >= 0
    t = np.where(up[:, None], t, t[:, ::-1])
    edges = np.concatenate([t[:, [0, 1]], t[:, [1, 2]], t[:, [2, 0]]])
    keys = np.sort(edges, axis=1)
    _, inverse, counts = np.unique(keys, axis=0, return_inverse=True, return_counts=True)
    return t, edges[counts[inverse.ravel()] == 1]


def skirt(positions: np.ndarray, triangles: np.ndarray, bottom: float):
    """Walls hanging from a surface's open edges down to ``bottom``, facing out: the
    land of a bed ends in a cliff at the side of the domain, not in a paper edge
    over nothing. Returns positions and triangles of the walls alone."""
    p = np.asarray(positions, np.float64).reshape(-1, 3)
    t = np.asarray(triangles, np.int64).reshape(-1, 3)
    finite = np.isfinite(p).all(axis=1)
    t = t[finite[t].all(axis=1)]
    t = t[(t[:, 0] != t[:, 1]) & (t[:, 1] != t[:, 2]) & (t[:, 0] != t[:, 2])]
    if not len(t):
        return np.zeros((0, 3), np.float32), np.zeros((0, 3), np.uint32)
    _, rim = _facing_up_and_rim(p, t)
    top_a, top_b = p[rim[:, 0]], p[rim[:, 1]]
    keep = (top_a[:, 2] > bottom) | (top_b[:, 2] > bottom)
    top_a, top_b = top_a[keep], top_b[keep]
    low_a, low_b = top_a.copy(), top_b.copy()
    low_a[:, 2] = low_b[:, 2] = bottom
    k = len(top_a)
    points = np.concatenate([top_a, top_b, low_b, low_a])
    i = np.arange(k)
    walls = np.concatenate(
        [np.stack([i, i + 2 * k, i + k], 1), np.stack([i, i + 3 * k, i + 2 * k], 1)]
    )
    # outward: to the right of the edge seen from above
    d = top_b - top_a
    out = np.tile(np.stack([d[:, 1], -d[:, 0], np.zeros(k)], 1), (2, 1))
    w0, w1, w2 = points[walls[:, 0]], points[walls[:, 1]], points[walls[:, 2]]
    facing = (np.cross(w1 - w0, w2 - w0) * out).sum(axis=1)
    walls = np.where(facing[:, None] < 0, walls[:, ::-1], walls)
    return points.astype(np.float32), walls.astype(np.uint32)


def ocean_ring(low, high, level: float, reach: float):
    """The sea around the domain out to the horizon: a flat ring at the still water
    level from the box ``low``-``high`` (x, y) out to ``reach`` from its centre,
    facing up. Returns positions and triangles."""
    (x0, y0), (x1, y1) = np.asarray(low, float)[:2], np.asarray(high, float)[:2]
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    r = max(float(reach), 2 * max(x1 - x0, y1 - y0))
    inner = [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]
    outer = [(cx - r, cy - r), (cx + r, cy - r), (cx + r, cy + r), (cx - r, cy + r)]
    positions = np.array([(x, y, level) for x, y in inner + outer], np.float32)
    triangles = []
    for j in range(4):  # a band per side: inner j, j+1 and outer j, j+1 (counter-clockwise)
        a, b = j, (j + 1) % 4
        triangles += [(a, 4 + a, 4 + b), (a, 4 + b, b)]
    return positions, np.array(triangles, np.uint32)


def rim_loop(positions: np.ndarray, triangles: np.ndarray) -> np.ndarray | None:
    """The outer edge of a surface as one closed loop of its corners, counter-clockwise
    seen from above (the surface on its left); None when it has no single clear one
    (the loop must enclose at least half of the surface's box)."""
    p = np.asarray(positions, np.float64).reshape(-1, 3)
    t = np.asarray(triangles, np.int64).reshape(-1, 3)
    finite = np.isfinite(p).all(axis=1)
    t = t[finite[t].all(axis=1)]
    t = t[(t[:, 0] != t[:, 1]) & (t[:, 1] != t[:, 2]) & (t[:, 0] != t[:, 2])]
    if not len(t):
        return None
    _, rim = _facing_up_and_rim(p, t)
    following: dict[int, list[int]] = {}
    for a, b in rim.tolist():
        following.setdefault(a, []).append(b)
    used: set[int] = set()
    best, best_area = None, 0.0
    for start in following:
        if start in used:
            continue
        loop, v = [start], start
        used.add(start)
        while True:
            nexts = [w for w in following.get(v, ()) if w == start or w not in used]
            if not nexts:
                loop = None  # open: a pinch or a broken edge
                break
            v = nexts[0]
            if v == start:
                break
            loop.append(v)
            used.add(v)
        if loop is None or len(loop) < 3:
            continue
        x, y = p[loop, 0], p[loop, 1]
        area = 0.5 * float(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))
        if area > best_area:
            best, best_area = loop, area
    if best is None:
        return None
    box = np.ptp(p[np.unique(t), :2], axis=0)
    if best_area < 0.5 * float(box[0] * box[1]):
        return None
    return np.asarray(best, np.int64)


def ring_distances(step: float, reach: float, growth: float = 1.45) -> np.ndarray:
    """Distances out from a rim for rings of corners: the first a cell out, then each
    gap ``growth`` times the last, up to ``reach``."""
    out, d, gap = [], 0.0, max(float(step), 1e-9)
    while d < reach:
        d = min(d + gap, float(reach))
        out.append(d)
        gap *= growth
    return np.asarray(out)


def outward_rings(xy: np.ndarray, distances: np.ndarray, extent: float) -> np.ndarray:
    """Rings around a counter-clockwise loop ``xy`` (m, 2) at ``distances`` out
    (k, m, 2): near ones parallel to the loop (its corners kept), far ones rounding
    into circles, so they never cross."""
    xy = np.asarray(xy, float)
    edge = np.roll(xy, -1, axis=0) - xy
    length = np.maximum(np.linalg.norm(edge, axis=1), 1e-12)
    normal = np.stack([edge[:, 1], -edge[:, 0]], 1) / length[:, None]  # to the right: out
    before = np.roll(normal, 1, axis=0)
    # mitred: each side moves out by the distance, corners stay corners
    miter = (before + normal) / np.maximum(1.0 + (before * normal).sum(axis=1), 0.5)[:, None]
    centre = xy.mean(axis=0)
    radial = xy - centre
    radius = np.maximum(np.linalg.norm(radial, axis=1), 1e-12)
    unit = radial / radius[:, None]
    rings = []
    for d in distances:
        s = np.clip((d - 0.2 * extent) / (2.0 * extent), 0.0, 1.0)
        w = s * s * (3 - 2 * s)
        rings.append((1 - w) * (xy + miter * d) + w * (centre + unit * (radius + d)[:, None]))
    return np.asarray(rings)


def ring_triangles(loop: np.ndarray, start: int, rings: int) -> np.ndarray:
    """Triangles from a loop of corners (their indices) out through ``rings`` rings of
    len(loop) new corners each, numbered from ``start`` ring by ring; facing up."""
    m = len(loop)
    index = np.concatenate([loop[None, :], start + np.arange(rings * m).reshape(rings, m)])
    inner, outer = index[:-1], index[1:]
    a, b = inner, np.roll(inner, -1, axis=1)
    c, d = np.roll(outer, -1, axis=1), outer
    return np.concatenate(
        [np.stack([a, c, b], -1).reshape(-1, 3), np.stack([a, d, c], -1).reshape(-1, 3)]
    )


def loop_mean(values: np.ndarray, half: int) -> np.ndarray:
    """A moving mean over ``2 half + 1`` neighbours along a closed loop."""
    values = np.asarray(values, float)
    m = len(values)
    half = max(int(half), 0)
    if half == 0:
        return values
    if 2 * half + 1 >= m:  # the whole loop
        return np.full(m, values.mean())
    total = np.concatenate([[0.0], np.cumsum(np.concatenate([values, values, values]))])
    i = np.arange(m) + m
    return (total[i + half + 1] - total[i - half]) / (2 * half + 1)


def apron(rim_heights: np.ndarray, distances: np.ndarray, step: float, slope: float, deep):
    """Heights of the bed beyond the domain (k, m): each ring continues the rim's
    heights, ever more smoothed along it, and sinks to ``deep`` over about ``slope``
    out, so the land runs down into the sea and the sea floor falls away."""
    out = []
    for d in distances:
        smooth = loop_mean(rim_heights, round(0.5 * d / max(step, 1e-12)))
        out.append(deep + (smooth - deep) * math.exp(-((d / slope) ** 2)))
    return np.asarray(out)


# ======================================================================== cameras
def look_rotation(forward, up=(0.0, 0.0, 1.0)) -> list[float]:
    """The quaternion (x, y, z, w) of a camera looking along ``forward`` with ``up``
    up (cameras look along their -z with +y up, in pygfx and in Blender alike)."""
    f = np.asarray(forward, float)
    f /= np.linalg.norm(f)
    x = np.cross(f, np.asarray(up, float))
    if np.linalg.norm(x) < 1e-9:  # looking straight up or down
        x = np.cross(f, [0.0, 1.0, 0.0])
    x /= np.linalg.norm(x)
    y = np.cross(x, f)
    return _quaternion(np.stack([x, y, -f], axis=1))


def _quaternion(m: np.ndarray) -> list[float]:
    """The unit quaternion (x, y, z, w) of a rotation matrix (columns: the axes)."""
    trace = m[0, 0] + m[1, 1] + m[2, 2]
    if trace > 0:
        s = 2.0 * math.sqrt(trace + 1.0)
        q = [(m[2, 1] - m[1, 2]) / s, (m[0, 2] - m[2, 0]) / s, (m[1, 0] - m[0, 1]) / s, s / 4]
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = 2.0 * math.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2])
        q = [s / 4, (m[0, 1] + m[1, 0]) / s, (m[0, 2] + m[2, 0]) / s, (m[2, 1] - m[1, 2]) / s]
    elif m[1, 1] > m[2, 2]:
        s = 2.0 * math.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2])
        q = [(m[0, 1] + m[1, 0]) / s, s / 4, (m[1, 2] + m[2, 1]) / s, (m[0, 2] - m[2, 0]) / s]
    else:
        s = 2.0 * math.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1])
        q = [(m[0, 2] + m[2, 0]) / s, (m[1, 2] + m[2, 1]) / s, s / 4, (m[1, 0] - m[0, 1]) / s]
    q = np.asarray(q)
    return (q / np.linalg.norm(q)).tolist()


def rotate(q, v) -> np.ndarray:
    """A vector rotated by a quaternion (x, y, z, w)."""
    x, y, z, w = (float(c) for c in q)
    u = np.array([x, y, z])
    v = np.asarray(v, float)
    return v + 2.0 * np.cross(u, np.cross(u, v) + w * v)


def default_camera(low, high, fov: float = DEFAULT_FOV) -> dict[str, Any]:
    """A camera state (as the 3D view keeps one) looking at a box from LAGOON's
    perspective direction, the box's sphere filling the view."""
    low, high = np.asarray(low, float), np.asarray(high, float)
    centre = (low + high) / 2
    radius = max(float(np.linalg.norm(high - low)) / 2, 1e-6)
    forward = np.asarray(PERSPECTIVE, float)
    forward /= np.linalg.norm(forward)
    distance = 1.08 * radius / math.sin(math.radians(fov) / 2)
    return {
        "position": (centre - forward * distance).tolist(),
        "rotation": look_rotation(forward),
        "fov": fov,
        "zoom": 1.0,
        "width": 2 * radius,
        "height": 2 * radius,
    }


def blender_camera(state: dict[str, Any], size: tuple[int, int]) -> dict[str, Any]:
    """A 3D view camera state as Blender's camera for an image of ``size``: what
    pygfx shows, it shows. pygfx fits the field of view (over the zoom) into the mean
    of the camera's width and height at its aspect, then widens whichever side the
    image's aspect needs; Blender gets the vertical extent that results."""
    width, height = (float(v) for v in size)
    image_aspect = width / height
    w = float(state.get("width") or 1.0)
    h = float(state.get("height") or 1.0)
    aspect = w / h if w > 0 and h > 0 else 1.0
    zoom = float(state.get("zoom") or 1.0)
    fov = float(state.get("fov") or 0.0)
    rotation = list(state.get("rotation", (0.0, 0.0, 0.0, 1.0)))
    camera = {
        "position": [float(v) for v in state.get("position", (0.0, 0.0, 1.0))],
        # Blender's order: w, x, y, z
        "rotation_wxyz": [float(rotation[3]), *(float(v) for v in rotation[:3])],
    }
    if fov > 0:
        mean = 2.0 * math.tan(math.radians(fov) / 2) / zoom  # per unit of distance
        vertical = 2 * mean / (1 + aspect)
        if aspect >= image_aspect:  # the camera's width holds: the height grows
            vertical = vertical * aspect / image_aspect
        camera.update(type="PERSP", angle_y=2.0 * math.atan(vertical / 2))
    else:
        vertical = h / zoom
        if w / h >= image_aspect:
            vertical = (w / zoom) / image_aspect
        camera.update(type="ORTHO", ortho_scale=vertical)
    return camera


def pixel_size(camera: dict[str, Any], point, image_height: int) -> float:
    """The size [m] of a pixel at a point, seen by a Blender camera."""
    if camera["type"] == "ORTHO":
        return camera["ortho_scale"] / image_height
    distance = float(np.linalg.norm(np.asarray(point, float) - camera["position"]))
    return 2.0 * distance * math.tan(camera["angle_y"] / 2) / image_height


# ======================================================================== export
def look_preset(look: str | None) -> Preset:
    """The cinematic preset a look is path traced in."""
    name = LOOK_FOR.get(look or "", look or "open_ocean")
    if name not in PRESETS:
        raise ValueError(f"unknown look {look!r}; path tracing knows {', '.join(PRESETS)}")
    return PRESETS[name]


class _Collector:
    """The objects and materials of a frame, as arrays and plain values."""

    def __init__(self, preset: Preset, pixel: float, bounds=None, pixels: int = 1920 * 1080):
        self.preset = preset
        self.pixel = pixel
        #: the image's size in pixels: how finely surfaces need subdividing
        self.pixels = int(pixels)
        #: (low, high) of the frame's data: the floor and the depth of water bodies
        self.bounds = bounds
        self.water_bodies = 0
        #: the bed's heights (BedHeights), where water bodies end
        self.bed: BedHeights | None = None
        #: the still water level (as drawn), where the land begins and the sea around
        #: the domain lies; None when not known
        self.level: float | None = None
        #: how far out the land and the sea reach beyond the domain (0: they end there)
        self.far = 0.0
        self.sea_done = False
        self.objects: list[dict[str, Any]] = []
        self.arrays: dict[str, np.ndarray] = {}
        self.materials: dict[str, dict[str, Any]] = {}

    def material(self, spec: dict[str, Any]) -> str:
        for key, known in self.materials.items():
            if same_material(known, spec):
                return key
        key = f"m{len(self.materials)}"
        self.materials[key] = spec
        return key

    def add(
        self,
        name: str,
        positions,
        triangles,
        material: dict,
        rgb=None,
        smooth=None,
        attributes: dict[str, np.ndarray] | None = None,
        subdivide: bool = False,
    ) -> None:
        positions = np.asarray(positions, np.float32).reshape(-1, 3)
        triangles = np.asarray(triangles, np.uint32).reshape(-1, 3)
        if not len(triangles):
            return
        # only proper triangles: none at dry land (outside the record) and none with a
        # corner twice (hidden ones), so Blender takes the mesh as it is (no validating)
        finite = np.isfinite(positions).all(axis=1)
        keep = finite[triangles].all(axis=1)
        keep &= (triangles[:, 0] != triangles[:, 1]) & (triangles[:, 1] != triangles[:, 2])
        keep &= triangles[:, 0] != triangles[:, 2]
        if not keep.all():
            triangles = triangles[keep]
            if smooth is not None and len(smooth) == len(keep):
                smooth = np.asarray(smooth, bool)[keep]
            if not len(triangles):
                return
        if not finite.all():
            positions = np.where(finite[:, None], positions, 0.0).astype(np.float32)
        index = len(self.objects)
        self.arrays[f"positions_{index}"] = positions
        self.arrays[f"triangles_{index}"] = triangles
        if smooth is not None:
            self.arrays[f"smooth_{index}"] = np.asarray(smooth, bool)
        if rgb is not None:
            self.arrays[f"colors_{index}"] = np.asarray(rgb, np.float32).reshape(-1, 3)
        entry = {"name": name, "material": self.material(material), "colors": rgb is not None}
        if attributes:  # per corner values the material reads (by name)
            for key, values in attributes.items():
                self.arrays[f"attribute_{index}_{key}"] = np.asarray(values, np.float32)
            entry["attributes"] = sorted(attributes)
        levels = subdivision_levels(len(triangles), self.pixels) if subdivide else 0
        if levels:  # smoothed in Blender (its flat faces kept)
            entry["subdivide"] = levels
        self.objects.append(entry)

    def points(self, name: str, positions, radius, material: dict[str, Any]) -> None:
        """Particles drawn as small spheres (Blender's point clouds): foam, spray."""
        positions = np.asarray(positions, np.float32).reshape(-1, 3)
        if not len(positions):
            return
        index = len(self.objects)
        self.arrays[f"points_{index}"] = positions
        self.arrays[f"radius_{index}"] = np.asarray(radius, np.float32).reshape(-1)
        self.objects.append(
            {"name": name, "material": self.material(material), "colors": False, "points": True}
        )

    def white_water(self, particles: dict[str, tuple[np.ndarray, np.ndarray]]) -> None:
        """Foam, spray and bubbles (:mod:`lagoonviz.core.foam`), each with its material."""
        for kind, (positions, radius) in particles.items():
            self.points(f"LAGOON {kind}", positions, radius, WHITE_WATER[kind])

    @staticmethod
    def plain(color: str | None, roughness: float = 0.5) -> dict[str, Any]:
        return {
            "color": linear_hex(color) if color else None,
            "roughness": roughness,
            "metallic": 0.0,
            "ior": 1.5,
            "transmission": 0.0,
            "coat": 0.0,
            "alpha": 1.0,
            "grain": None,
        }

    def lines(self, name: str, segments, width_px: float, color: str, rgb=None) -> None:
        segments = np.asarray(segments, np.float64).reshape(-1, 2, 3)
        keep = np.isfinite(segments).all(axis=(1, 2))
        segments = segments[keep]
        if not len(segments):
            return
        positions, triangles = tubes(segments, 0.5 * width_px * self.pixel)
        if rgb is not None:
            rgb = np.repeat(np.asarray(rgb, np.float32)[keep], len(positions) // len(segments), 0)
        material = self.plain(None if rgb is not None else color)
        self.add(name, positions, triangles, material, rgb)

    def surface(self, kind: str, shown, all_triangles=None) -> None:
        if not shown.visible or shown.opacity <= 0:
            return
        values = shown.values
        rgb = None
        if values is not None:
            rgb = linear(colors(values, shown.colormap, shown.vmin, shown.vmax)[:, :3])
        if shown.point_size is not None:  # particles
            keep = np.isfinite(shown.positions).all(axis=1)
            radius = 0.5 * shown.point_size * self.pixel
            if shown.point_sizes is not None and len(shown.point_sizes) == len(keep):
                radius = 0.5 * np.asarray(shown.point_sizes, float)[keep]  # their own size
            positions, triangles = spheres(shown.positions[keep], radius)
            if rgb is not None:
                rgb = np.repeat(rgb[keep], 12, axis=0)
            self.add(shown.item_id, positions, triangles, self.plain(shown.color, 0.6), rgb)
            return
        if shown.segments is not None and shown.triangles.size == 0:  # mooring lines
            ends = shown.positions[np.asarray(shown.segments, int)]
            seg_rgb = None if rgb is None else rgb[np.asarray(shown.segments, int)].mean(axis=1)
            self.lines(shown.item_id, ends, shown.line_width, shown.color, seg_rgb)
            return
        surface = surface_of(kind, shown.material)
        s = SURFACES.get(surface, SURFACES["plain"])
        water = surface == "water"
        if water and rgb is None and shown.opacity >= 1.0:
            self.water(shown, all_triangles)
            return
        spec = {
            "color": None
            if rgb is not None
            else linear_hex(self.preset.water_color if water else shown.color),
            "roughness": (self.preset.water_roughness if water else s.roughness)
            if rgb is None
            else max(0.3, s.roughness if not water else 0.3),
            "metallic": s.metalness,
            "ior": s.ior,
            # clear water is a body of its own (water()); a see-through sheet here
            # (opacity < 1) tints what is behind it; coloured by a variable, water is
            # opaque, so the colours read
            "transmission": 0.3 if water and rgb is None else 0.0,
            "water": bool(water and rgb is None),
            "coat": s.clearcoat,
            "alpha": round(float(shown.opacity), 4),
            "grain": None,
        }
        finite = shown.positions[np.isfinite(shown.positions).all(axis=1)]
        span = float(np.ptp(finite[:, :2], axis=0).max()) if len(finite) else 0.0
        land = kind == "bathymetry" and rgb is None and self.dressed and len(finite)
        # a look's own colours (ÆgirFrames' dark rock bed, red solids) for an item in
        # its kind's default colour
        given = look_colour(self.preset, kind, shown.color, shown.material)
        if given is not None and rgb is None and not land:
            color, surface = given
            s = SURFACES[surface]
            spec.update(color=linear_hex(color), roughness=s.roughness, metallic=s.metalness)
        if s.grain and rgb is None:
            spec["grain"] = {
                "kind": s.grain,
                "tile": round(max(s.tile, span / 3), 6),
                "relief": s.relief,
            }
        if land:
            spec["terrain"] = self.terrain(finite, span)
            spec["grain"] = None  # the terrain has its own fine relief
            extended = self.beyond(shown.positions, shown.triangles)
            if extended is not None:  # the land and the sea floor run on
                self.add(shown.item_id, *extended, spec, None)
                return
        self.add(shown.item_id, shown.positions, shown.triangles, spec, rgb, subdivide=water)
        if kind == "bathymetry" and self.stand and len(finite):
            # a model on a stand: the bed a solid block, its sides down to a base (land
            # cut off in rock)
            color = spec["terrain"]["rock"] if land else spec["color"] or linear_hex(shown.color)
            sides = dict(spec, color=color, terrain=None)
            positions, triangles = skirt(shown.positions, shown.triangles, self.base(finite))
            flat = np.zeros(len(triangles), bool)
            self.add(f"{shown.item_id}#sides", positions, triangles, sides, None, flat)
        elif land:
            # the land ends in cliffs at the sides of the domain, down under the sea
            zspan = max(float(np.ptp(finite[:, 2])), 1e-6)
            bottom = min(float(finite[:, 2].min()), float(self.level)) - 0.05 * zspan
            positions, triangles = skirt(shown.positions, shown.triangles, bottom)
            self.add(
                f"{shown.item_id}#sides",
                positions,
                triangles,
                spec,
                None,
                np.zeros(len(triangles), bool),
            )

    def base(self, bed: np.ndarray) -> float:
        """The height of the diorama's base: under the bed by a quarter of the water's
        greatest depth (a block of sand, not a sheet)."""
        low = float(bed[:, 2].min())
        span = float(np.ptp(bed[:, :2], axis=0).max()) if len(bed) else 1.0
        depth = float(self.level) - low if self.level is not None else float(np.ptp(bed[:, 2]))
        return low - max(0.25 * depth, 0.01 * span, 1e-6)

    @property
    def dressed(self) -> bool:
        """The bed dressed as land by height above the still water level and slope,
        the water with ripples and foam at the shore (under the open sky)."""
        return self.preset.world in ("coast", "model") and self.level is not None

    @property
    def stand(self) -> bool:
        """The domain as a model on a stand: the bed a block down to a base."""
        return self.preset.world in ("diorama", "model")

    @property
    def coast(self) -> bool:
        """Land and sea under the open sky: the bed is dressed as land above the still
        water level, and the sea runs on beyond the domain."""
        return self.preset.world == "coast" and self.level is not None

    def world(self) -> dict[str, float] | None:
        """Sizes of the world beyond the domain under the open sky: the domain's size
        (extent), how far the sea reaches (to the horizon), over what distance the land
        and the sea floor sink away (slope), and how deep the far sea floor (deep_bed)
        and the water's own floor (deep_water, under everything) lie."""
        if not self.coast or self.bounds is None:
            return None
        low, high = (np.asarray(v, float) for v in self.bounds)
        extent = max(float(np.ptp([low[:2], high[:2]], axis=0).max()), 1e-6)
        level = float(self.level)
        bottom = float(self.bed.z.min()) if self.bed is not None else float(low[2])
        depth = max(level - bottom, 0.02 * extent)
        return {
            "extent": extent,
            "reach": SEA_REACH * extent,
            "slope": 0.4 * extent,
            "deep_bed": level - 1.5 * depth,
            "deep_water": level - 3.0 * depth,
        }

    def beyond(self, positions, triangles):
        """The bed with its apron: rings of corners out from its outer edge to the
        horizon, continuing its heights and sinking into the sea (positions and
        triangles), or None when it has no clear outer edge."""
        world = self.world()
        p = np.asarray(positions, float).reshape(-1, 3)
        loop = rim_loop(p, triangles) if world is not None else None
        if loop is None:
            return None
        xy = p[loop, :2]
        step = float(np.median(np.linalg.norm(np.roll(xy, -1, 0) - xy, axis=1)))
        distances = ring_distances(step, world["reach"])
        rings = outward_rings(xy, distances, world["extent"])
        heights = apron(p[loop, 2], distances, step, world["slope"], world["deep_bed"])
        new = np.concatenate([rings.reshape(-1, 2), heights.reshape(-1, 1)], axis=1)
        self.far = max(self.far, world["reach"])
        return (
            np.concatenate([p, new]),
            np.concatenate(
                [
                    np.asarray(triangles, np.int64).reshape(-1, 3),
                    ring_triangles(loop, len(p), len(distances)),
                ]
            ),
        )

    def terrain(self, points: np.ndarray, span: float) -> dict[str, Any]:
        """Colours of a coast by the height above the still water level and the slope:
        wet sand at the water line, dry sand above it, then grass and heath, rock where
        it is steep (heights as fractions of how high the land rises)."""
        level = float(self.level)
        top = float(points[:, 2].max())
        depth = max(level - float(points[:, 2].min()), 0.0)
        return {
            "level": round(level, 6),
            # at least a tenth of the depth: low land is beach, not hills
            "relief": round(max(top - level, 0.1 * depth, 1e-6), 6),
            "scale": round(6.0 / max(span, 1e-6), 6),  # patches of colour, a few per side
            "detail": round(250.0 / max(span, 1e-6), 6),  # fine relief, a few pixels each
            # natural, muted colours (as seen from the air, not a map's)
            "wet": linear_hex("#6b604e"),
            "sand": linear_hex("#a99878"),
            "grass": linear_hex("#5d6540"),
            "heath": linear_hex("#6e6449"),
            "rock": linear_hex("#6f6860"),
        }

    def water(self, shown, all_triangles=None) -> None:
        """Clear water as path tracing sees it: a closed body (surface, walls, floor).

        Above water as ÆgirFrames renders it: glass in the look's water colour, smooth
        (ripples only on the open sea), refracting and reflecting what the camera
        sees, while light, shadows and reflections pass through it (it is there for
        camera rays only), with a faint dark haze inside. Under water a body that
        absorbs light the deeper it goes and scatters it (haze)."""
        points = np.asarray(shown.positions, float)
        finite = points[np.isfinite(points).all(axis=1)]
        if not len(finite):
            return
        if self.bounds is not None:
            low, high = (np.asarray(v, float) for v in self.bounds)
        else:
            low, high = finite.min(axis=0), finite.max(axis=0)
        extent = max(float(np.ptp(finite[:, :2], axis=0).max()), 1e-6)
        if self.bed is not None:
            bottom = self.bed
            floor = self.bed(finite[:, :2])
            depth = float(np.mean(np.maximum(finite[:, 2] - floor, 0.0)))
        else:
            bottom = float(low[2]) - 0.01 * max(float(np.linalg.norm(high - low)), 1e-6)
            depth = float(np.mean(finite[:, 2])) - float(low[2])
        depth = max(depth, 0.02 * extent)
        top = np.asarray(shown.positions, float).copy()
        triangles = shown.triangles
        if self.bed is not None and all_triangles is not None:
            # the whole surface, dry land too, but there pushed just under the bed: the
            # land hides it, and the shore is where water and land meet, not a wall
            # along the edge of the wet cells
            triangles = all_triangles
            span = max(float(np.ptp(finite[:, 2])), 1e-6)
            ok = np.isfinite(top).all(axis=1)
            bed = np.full(len(top), np.nan)
            bed[ok] = self.bed(top[ok, :2])
            # dry: no wet triangle of the 3D view uses the corner, or a film of water
            # thinner than the bed and the surface meshes agree on (they are different
            # grids: a thin film pokes through the land in a comb of specks)
            shown_t = np.asarray(shown.triangles, np.int64).reshape(-1, 3)
            shown_t = shown_t[(shown_t[:, 0] != shown_t[:, 1]) & (shown_t[:, 1] != shown_t[:, 2])]
            wet = np.zeros(len(top), bool)
            wet[shown_t.ravel()] = True
            dry = ok & (~wet | (top[:, 2] - bed < 5e-3 * span))
            # well under the land: the surface dives under it within one cell, so the
            # shore is a clean line where water meets land
            top[dry, 2] = bed[dry] - 0.02 * span
        # how deep the water is at each corner of the surface
        film_top = np.full(len(top), 1e6)
        if self.bed is not None:
            ok = np.isfinite(top).all(axis=1)
            film_top[ok] = top[ok, 2] - self.bed(top[ok, :2])
        extended = self.sea_beyond(top, triangles) if self.bed is not None else None
        if extended is not None:  # one body of water with the sea out to the horizon
            top, triangles, bottom, film_top = extended
        positions, triangles, smooth = water_body(top, triangles, bottom)
        if not len(triangles):
            return
        film = np.full(len(positions), 1e6, np.float32)  # walls and floor: deep
        film[: len(top)] = film_top
        tint = linear(_hex(self.preset.water_color))
        under = self.preset.fog is not None
        if not under:
            self._glass_water(shown, positions, triangles, smooth, film, extent, depth, extended)
            return
        tint = 0.3 + 0.7 * tint / max(float(tint.max()), 1e-6)  # what the water lets through
        spec = {
            "color": [1.0, 1.0, 1.0],
            "roughness": 0.02,
            "metallic": 0.0,
            "ior": 1.333,
            "transmission": 1.0,
            "water": True,
            "coat": 0.0,
            "alpha": 1.0,
            "grain": None,
            "volume": {
                "color": [round(float(v), 4) for v in tint],
                # straight through the water's mean depth, light keeps exp(-absorb)
                # (Beer-Lambert; about two thirds in most looks): the bed shows,
                # deep water turns blue
                "density": round(
                    self.preset.absorb / (depth * max(1.0 - float(tint.mean()), 0.05)), 6
                ),
                "scatter": 0.1 if under else 0.0,
            },
            # wind ripples, much smaller than the waves: they break up the reflections
            "ripples": {"scale": round(120.0 / extent, 6), "strength": 0.2},
            # at the shore the water thins out to nothing (no sheet of glass over the
            # sand); under the open sky broken foam where it breaks on the beach
            "shore": {
                "fade": round(0.03 * depth, 6),
                "foam": round(0.05 * depth, 6) if self.preset.world == "coast" else 0.0,
                "scale": round(60.0 / extent, 6),
            }
            if self.bed is not None
            else None,
        }
        if extended is not None:  # a long, low swell under the ripples, out to sea
            spec["swell"] = {"scale": round(4.0 / extent, 6), "strength": 0.15}
        attributes = {"lagoon_depth": film} if self.bed is not None else None
        self.add(shown.item_id, positions, triangles, spec, None, smooth, attributes, True)
        self.water_bodies += 1

    def _glass_water(self, shown, positions, triangles, smooth, film, extent, depth, extended):
        """The water body as ÆgirFrames' ocean water (see :meth:`water`)."""
        coast = self.dressed
        spec = {
            "color": linear_hex(self.preset.water_color),
            "roughness": 0.0,
            "metallic": 0.0,
            "ior": 1.33,
            "transmission": 1.0,
            "water": True,
            "coat": 0.0,
            "alpha": 1.0,
            "grain": None,
            "camera_only": True,
            "volume": dict(AEGIR_HAZE),
            "ripples": self._far_fade({"scale": round(120.0 / extent, 6), "strength": 0.2})
            if coast
            else None,
            "shore": {
                "fade": round(0.03 * depth, 6),
                "foam": round(0.05 * depth, 6) if coast else 0.0,
                "scale": round(60.0 / extent, 6),
            }
            if self.bed is not None
            else None,
        }
        if extended is not None:  # a long, low swell under the ripples, out to sea
            spec["swell"] = self._far_fade({"scale": round(4.0 / extent, 6), "strength": 0.15})
        attributes = {"lagoon_depth": film} if self.bed is not None else None
        self.add(shown.item_id, positions, triangles, spec, None, smooth, attributes, True)
        self.water_bodies += 1

    def _far_fade(self, waves: dict[str, Any]) -> dict[str, Any]:
        """Small waves drawn as bumps fade out far from the domain: there they are
        smaller than a pixel (they would shimmer in stripes), and the sea far off looks
        smooth anyway. Their noise is measured from the domain's middle."""
        if self.bounds is None:
            return waves
        low, high = (np.asarray(v, float) for v in self.bounds)
        extent = max(float(np.ptp([low[:2], high[:2]], axis=0).max()), 1e-6)
        centre = (low + high) / 2
        return {
            **waves,
            "centre": [round(float(v), 6) for v in centre],
            "fade": [round(1.5 * extent, 6), round(6.0 * extent, 6)],
        }

    def sea_beyond(self, top: np.ndarray, triangles):
        """The water's surface with the sea around it: rings out from its outer edge to
        the horizon, the waves at the edge dying out into the still water level, under
        the land where the bed's apron (:meth:`beyond`) stays above it. Returns the
        surface, its triangles, the floor's height per corner (the bed in the domain,
        far down beyond it) and the water's depth per corner; None without an edge."""
        world = self.world()
        loop = rim_loop(top, triangles) if world is not None else None
        if loop is None:
            return None
        n = len(top)
        xy = top[loop, :2]
        step = float(np.median(np.linalg.norm(np.roll(xy, -1, 0) - xy, axis=1)))
        distances = ring_distances(step, world["reach"])
        rings = outward_rings(xy, distances, world["extent"])
        land = apron(self.bed(xy), distances, step, world["slope"], world["deep_bed"])
        level = float(self.level)
        # the waves at the edge run on and die out gently (no line where they stop)
        decay = np.exp(-distances / max(3 * step, 0.1 * world["extent"]))[:, None]
        z = level + (top[loop, 2] - level)[None, :] * decay
        finite = top[np.isfinite(top).all(axis=1)]
        span = max(float(np.ptp(finite[:, 2])), 1e-6) if len(finite) else 1.0
        z = np.where(land > level, np.minimum(z, land - 0.02 * span), z)
        surface = np.concatenate(
            [top, np.concatenate([rings.reshape(-1, 2), z.reshape(-1, 1)], axis=1)]
        )
        faces = np.concatenate(
            [
                np.asarray(triangles, np.int64).reshape(-1, 3),
                ring_triangles(loop, n, len(distances)),
            ]
        )
        ok = np.isfinite(top).all(axis=1)
        floor = np.full(len(surface), world["deep_water"])
        floor[:n][ok] = self.bed(top[ok, :2]) - 1e-3 * span
        film = np.full(len(surface), 1e6)
        film[:n][ok] = top[ok, 2] - self.bed(top[ok, :2])
        film[n:] = (z - land).ravel()
        self.far = max(self.far, world["reach"])
        self.sea_done = True
        return surface, faces, floor, film

    def sea(self) -> None:
        """The sea beyond the domain, out to the horizon, when the water could not run
        on itself (:meth:`sea_beyond`): deep water at the still water level all around
        (only with a body of water under the open sky)."""
        if self.sea_done or not (self.coast and self.water_bodies and self.bounds is not None):
            return
        low, high = (np.asarray(v, float) for v in self.bounds)
        extent = max(float(np.ptp([low[:2], high[:2]], axis=0).max()), 1e-6)
        positions, triangles = ocean_ring(low, high, float(self.level), SEA_REACH * extent)
        deep = linear(_hex(self.preset.water_color)) * 0.35
        spec = {
            "color": [round(float(v), 6) for v in deep],
            "roughness": 0.03,
            "metallic": 0.0,
            "ior": 1.333,
            "transmission": 0.0,
            "coat": 0.0,
            "alpha": 1.0,
            "grain": None,
            "ripples": self._far_fade({"scale": round(120.0 / extent, 6), "strength": 0.2}),
            "swell": self._far_fade({"scale": round(4.0 / extent, 6), "strength": 0.25}),
        }
        self.add(SEA, positions, triangles, spec, None, np.zeros(len(triangles), bool))
        self.far = max(self.far, SEA_REACH * extent)


def _frame_objects(
    session, preset: Preset, pixel: float, bounds=None, pixels: int = 1920 * 1080
) -> _Collector:
    """The objects of the session's current time, as the 3D view shows them."""
    data = session.data
    out = _Collector(preset, pixel, bounds, pixels)
    shown_items = []
    for item_id in data.data_items():
        item = session.scene.get(item_id)
        if not item.visible:
            continue
        mesh = data.load(item_id)
        if mesh is None:
            continue
        shown_items.append((item_id, item, mesh, data.display(item_id, mesh)))
    beds = [
        shown.positions
        for _, item, _, shown in shown_items
        if item.kind == "bathymetry" and len(shown.positions)
    ]
    out.level = _still_water(session, shown_items)
    if beds:  # bodies of water end on the bed
        try:
            out.bed = BedHeights(np.concatenate([np.asarray(b, float) for b in beds]))
        except ValueError:
            out.bed = None
    for item_id, item, mesh, shown in shown_items:
        out.surface(item.kind, shown, mesh.triangles)
        if not shown.visible:
            continue
        overlays = (
            ("contours", data.contour_lines(item_id, mesh, shown), CONTOUR_WIDTH, None),
            ("vectors", data.vector_arrows(item_id, mesh, shown), VECTOR_WIDTH, "color"),
            ("trails", data.particle_trails(item_id, mesh), TRAIL_WIDTH, "color"),
            ("grid", data.amr_grid_lines(item_id, mesh, shown), OUTLINE_WIDTH * 0.6, "color"),
            ("outlines", data.patch_outlines(item_id, mesh, shown), OUTLINE_WIDTH, "color"),
        )
        for label, lines, width, own in overlays:
            if lines is None or not lines.visible or not lines.segments.size:
                continue
            color = (getattr(lines, own, None) if own else None) or preset.contours
            out.lines(f"{item_id}#{label}", lines.segments, width, color)
    out.sea()
    profiles = set(data.profile_items())
    for item_id in data.series_items() + sorted(profiles):
        if not session.scene.get(item_id).visible:
            continue
        marks = data.profile_marks(item_id) if item_id in profiles else data.gauge_marks(item_id)
        if marks is None or not marks.visible:
            continue
        rgb = linear(np.array([_hex(c) for c in marks.colors]).reshape(-1, 3))
        out.lines(f"{item_id}#staffs", marks.staffs, STAFF_WIDTH, "#000000", rgb)
        dots = np.asarray(marks.dots, float).reshape(-1, 3)
        wet = np.isfinite(dots).all(axis=1)
        if wet.any():
            positions, triangles = spheres(dots[wet], 0.5 * DOT_SIZE * pixel)
            out.add(
                f"{item_id}#dots",
                positions,
                triangles,
                out.plain(None, 0.4),
                np.repeat(rgb[wet], 12, axis=0),
            )
    return out


def _still_water(session, shown_items) -> float | None:
    """The still water level of the first shown water (or bed) as drawn: its case's,
    else the middle of its surface; None in the plan view or with neither."""
    if session.scene.flat:
        return None
    data = session.data
    for kinds in (("free_surface", "amr"), ("bathymetry",)):
        for item_id, item, _, shown in shown_items:
            if item.kind not in kinds:
                continue
            try:
                if data.case(item_id) is not None:
                    return float(data.case_water_level(item_id))
            except (AttributeError, OSError, ValueError):
                pass
            if item.kind in ("free_surface", "amr"):
                z = np.asarray(shown.positions, float).reshape(-1, 3)[:, 2]
                z = z[np.isfinite(z)]
                if len(z):
                    return float(np.median(z))
    return None


def _bounds(session) -> tuple[np.ndarray, np.ndarray] | None:
    """The box of the visible data at the session's time."""
    data = session.data
    lows, highs = [], []
    for item_id in data.data_items():
        if not session.scene.get(item_id).visible:
            continue
        mesh = data.load(item_id)
        if mesh is None:
            continue
        points = np.asarray(data.display(item_id, mesh).positions, float).reshape(-1, 3)
        points = points[np.isfinite(points).all(axis=1)]
        if len(points):
            lows.append(points.min(axis=0))
            highs.append(points.max(axis=0))
    if not lows:
        return None
    return np.min(lows, axis=0), np.max(highs, axis=0)


def camera_state_at(session, time: float, bounds) -> dict[str, Any]:
    """The 3D view's camera at a time: the camera path's, the saved view, or the
    default view of the data's box."""
    state = session.camera_at(time)
    if state is None and session.scene.camera:
        state = dict(session.scene.camera)
    if state is None:
        low, high = bounds if bounds is not None else ((-1, -1, -1), (1, 1, 1))
        state = default_camera(low, high)
    return state


class WhiteWater:
    """Foam, spray and bubbles for the frames of a film (:mod:`lagoonviz.core.foam`):
    the simulation runs through REEF3D's outputs up to each frame's time, from a
    warm-up before the first one, so a still shows foam that has had time to form.
    Off (``ok`` False) without a free surface with velocity on a grid, or without
    times in seconds."""

    def __init__(self, session, settings=None) -> None:
        from lagoonviz.core.foam import FoamSettings

        self.session = session
        self.settings = settings or FoamSettings()
        self.item = next(
            (
                item_id
                for item_id in session.data.data_items()
                if session.scene.get(item_id).kind == "free_surface"
                and session.scene.get(item_id).visible
            ),
            None,
        )
        times, _, unit = session.data.timeline()
        self.outputs = np.asarray(times, float) if unit == "s" else np.zeros(0)
        self.sim = None
        self.grid = None
        self.ok = self.item is not None and len(self.outputs) > 0

    def _surface(self, time: float):
        """The free surface's grid at a time (heights as drawn, velocity, breaking)."""
        from lagoonviz.core.foam import SurfaceGrid

        data = self.session.data  # read directly: the session's time stays
        dataset = data.dataset(self.item)
        index = data.frame_index(self.item, time)
        if dataset is None or index is None or "velocity" not in dataset.components:
            return None, None
        wanted = [n for n in ("velocity", "breaking", "WL", "depth") if n in dataset.components]
        raw = dataset.read(index, wanted)
        points = data.placed(self.item, raw.points)  # exaggerated as drawn
        grid = SurfaceGrid.from_points(
            points, raw.point_data["velocity"], raw.point_data.get("breaking")
        )
        depth = None
        for name in ("WL", "depth"):  # the water depth (NHFLOW: WL, FNPF: depth)
            if name in raw.point_data and depth is None:
                values = np.asarray(raw.point_data[name], float)
                values = values[np.isfinite(values) & (values > 0)]
                depth = float(np.median(values)) if len(values) else None
        return grid, depth

    def at(self, time: float) -> dict[str, tuple[np.ndarray, np.ndarray]]:
        """The white water at a time (per kind: positions and radii)."""
        from lagoonviz.core.foam import FoamSimulation

        if not self.ok:
            return {}
        if self.sim is None or self.sim.time is None or time < self.sim.time - 1e-9:
            grid, depth = self._surface(time)
            if grid is None:
                self.ok = False
                return {}
            if depth is None:
                depth = max(float(np.nanmax(grid.z) - np.nanmin(grid.z)) * 5.0, 10 * grid.h)
            self.sim = FoamSimulation(depth, self.settings)
            warm = 10.0 * self.sim.unit_time  # long enough for foam to form and fade
            before = self.outputs[(self.outputs >= time - warm) & (self.outputs < time)]
            start = float(before[0]) if len(before) else time
            self.grid, _ = self._surface(start)
            self.sim.time = start
        steps = self.outputs[(self.outputs > self.sim.time + 1e-9) & (self.outputs < time)]
        for t in [*steps.tolist(), time]:
            if t <= self.sim.time + 1e-9:
                continue
            grid, _ = self._surface(t)
            if grid is None or self.grid is None or grid.z.shape != self.grid.z.shape:
                self.grid, self.sim.time = grid, t  # a jump: start afresh from here
                continue
            self.sim.advance(self.grid, grid, self.sim.time, t)
            self.grid = grid
        return self.sim.particles()


def export_scene(
    session,
    folder: str | Path,
    times: list[float],
    size: tuple[int, int] = (1920, 1080),
    look: str = "open_ocean",
    samples: int = 64,
    progress: Progress | None = None,
    foam: bool | str | float = True,
) -> Path:
    """Write the session at ``times`` for Blender: ``scene.json``, a ``frame_NNNNN.npz``
    per time and ``environment.hdr``. With ``foam``, white water (foam, spray and
    bubbles) where the free surface breaks: True the default amount ("medium"), or
    "none", "light", "medium", "heavy" or a number (1: "heavy"; see
    :func:`lagoonviz.core.foam.white_water_amount`). Returns the path of ``scene.json``."""
    folder = Path(folder).expanduser()
    folder.mkdir(parents=True, exist_ok=True)
    preset = look_preset(look)
    times = [float(t) for t in times]
    if not times:
        raise ValueError("nothing to render: no times")
    size = (int(size[0]), int(size[1]))
    original_time = session.scene.time
    from lagoonviz.core.foam import FoamSettings, white_water_amount

    amount = white_water_amount(foam)
    white = WhiteWater(session, FoamSettings().scaled(amount)) if amount > 0 else None
    frames = []
    materials: dict[str, dict[str, Any]] = {}
    water_bodies = 0
    fars: list[float] = []  # how far each frame's world reaches beyond the domain
    low = high = None
    try:
        for number, time in enumerate(times):
            particles = white.at(time) if white is not None else {}
            session.set_time(time)
            bounds = _bounds(session)
            if bounds is not None:
                low = bounds[0] if low is None else np.minimum(low, bounds[0])
                high = bounds[1] if high is None else np.maximum(high, bounds[1])
            camera = blender_camera(camera_state_at(session, time, bounds), size)
            centre = (bounds[0] + bounds[1]) / 2 if bounds is not None else np.zeros(3)
            collector = _frame_objects(
                session, preset, pixel_size(camera, centre, size[1]), bounds, size[0] * size[1]
            )
            collector.white_water(particles)
            water_bodies += collector.water_bodies
            fars.append(collector.far)
            # one material table for the film: the frame's keys renamed into it
            rename = {}
            for key, spec in collector.materials.items():
                found = next((k for k, v in materials.items() if same_material(v, spec)), None)
                if found is None:
                    found = f"m{len(materials)}"
                    materials[found] = spec
                rename[key] = found
            for obj in collector.objects:
                obj["material"] = rename[obj["material"]]
            name = f"frame_{number:05d}"
            np.savez(folder / f"{name}.npz", **collector.arrays)
            digest = hashlib.sha1()
            for key in sorted(collector.arrays):
                array = np.ascontiguousarray(collector.arrays[key])
                digest.update(f"{key}:{array.dtype.str}:{array.shape}".encode())
                digest.update(array.tobytes())
            frames.append(
                {
                    "time": time,
                    "data": f"{name}.npz",
                    "image": f"{name}.png",
                    "camera": camera,
                    "objects": collector.objects,
                    "key": digest.hexdigest(),  # the data's; completed below
                }
            )
            if progress is not None and not progress(number + 1, len(times)):
                break
    finally:
        session.set_time(original_time)
    if low is None:
        low, high = np.full(3, -1.0), np.full(3, 1.0)
    extent = max(float(np.linalg.norm(high - low)), 1e-6)
    for frame, far in zip(frames, fars, strict=False):  # clip planes around everything
        distance = float(np.linalg.norm(np.asarray(frame["camera"]["position"]) - (low + high) / 2))
        frame["camera"]["clip_start"] = max(1e-4 * extent, (distance - extent) * 1e-3, 1e-5)
        frame["camera"]["clip_end"] = (distance + extent) * 4.0
        if far:  # out to the horizon
            frame["camera"]["clip_end"] = max(frame["camera"]["clip_end"], 3.0 * far + distance)
    own = session.sun_direction()  # the session's sun, else the look's
    sun = np.asarray(own if own is not None else preset.sun_direction, float)
    sun /= np.linalg.norm(sun)
    if preset.sky_picture:
        world, sun = _sky_picture_world(preset, folder, sun, own is not None)
    else:
        world, sun = _made_world(preset, folder, sun), sun
    if preset.fog is not None and not water_bodies:  # a water body has its own haze
        color, half = preset.fog
        margin = 0.1 * (high - low) + 0.01 * extent
        world["fog"] = {
            "color": linear_hex(color),
            "density": math.log(2.0) / (max(half, 1e-6) * extent),
            "low": (low - margin).tolist(),
            "high": (high + margin).tolist(),
        }
    picture = bool(preset.sky_picture)
    scene = {
        "version": VERSION,
        "look": preset.name,
        "size": list(size),
        "samples": int(samples),
        "view_transform": "AgX",
        # under a sky picture as ÆgirFrames renders: AgX as it is, the picture's own
        # light; under LAGOON's own sky, AgX's contrast and a little less exposure
        "look_contrast": None if picture else "Medium High Contrast",
        "exposure": 0.0 if picture else -0.6,
        "world": world,
        "sun": {
            "rotation_wxyz": _wxyz(_track_z(sun)),
            "direction": sun.tolist(),
            "color": linear_hex(preset.sun_color),
            # the picture's own sun lights the scene; the lamp only adds a little
            "strength": SKY_PICTURE_SUN if picture else preset.sun,
            "angle": 0.53 if picture else preset.sun_angle,  # degrees: soft edges
            "shadows": preset.shadows,
        },
        "materials": materials,
        "frames": frames,
    }
    # a frame's key: everything its image depends on. Blender keeps an image whose
    # key it wrote beside it (frame_NNNNN.key) when asked to resume.
    keys = ("version", "look", "size", "samples", "view_transform", "look_contrast", "exposure")
    common = {k: scene[k] for k in keys}
    common.update(world=world, sun=scene["sun"])
    for frame in frames:
        used = {o["material"]: materials[o["material"]] for o in frame["objects"]}
        text = json.dumps(
            [common, frame["key"], frame["camera"], frame["objects"], used], sort_keys=True
        )
        frame["key"] = hashlib.sha1(text.encode()).hexdigest()
    path = folder / "scene.json"
    path.write_text(json.dumps(scene, indent=1), encoding="utf-8")
    return path


def _made_world(preset: Preset, folder: Path, sun: np.ndarray) -> dict[str, Any]:
    """LAGOON's own surroundings for a look, with their sun glow where the sun is."""
    write_hdr(folder / "environment.hdr", environment_panorama(with_sun(preset, sun), 1024))
    world: dict[str, Any] = {
        "environment": "environment.hdr",
        "strength": preset.env_intensity,
        "background": [linear_hex(c) for c in preset.background] if preset.background else None,
        "fog": None,
        # under the open sky: Blender's physical sky (Nishita) when it has one; the
        # sun itself is the sun lamp, so the sky's own sun disc stays off
        "sky": {
            "elevation": math.asin(float(np.clip(sun[2], -1.0, 1.0))),
            # Cycles puts the sky's sun at azimuth rotation + 90 degrees from +x
            "rotation": math.atan2(-float(sun[0]), float(sun[1])),
            "strength": 0.35,
        }
        if preset.environment == "sky"
        else None,
    }
    return world


def _sky_picture_world(preset: Preset, folder: Path, sun: np.ndarray, own: bool):
    """A sky picture as the surroundings (ÆgirFrames' way): it lights the scene and is
    seen behind it, turned so its sun stands in the sun's direction. Its sun's height
    is the picture's; the lamp, weak, comes from where the picture's sun is (or from
    the session's sun when it has one). Returns the world and the lamp's direction."""
    source = SKY_FOLDER / str(preset.sky_picture)
    shutil.copyfile(source, folder / "environment.hdr")
    azimuth, elevation = sky_sun(str(preset.sky_picture))
    wanted = math.atan2(float(sun[1]), float(sun[0]))
    if not own:
        lamp = np.array(
            [
                math.cos(elevation) * math.cos(wanted),
                math.cos(elevation) * math.sin(wanted),
                math.sin(elevation),
            ]
        )
    else:
        lamp = sun
    world = {
        "environment": "environment.hdr",
        "strength": 1.0,
        # the picture turned about z: its sun from ``azimuth`` to ``wanted``
        "rotation": round(azimuth - wanted, 6),
        "background": None,
        "fog": None,
        "sky": None,
        "picture": preset.sky_picture,
    }
    return world, lamp


@functools.lru_cache(maxsize=2)
def sky_picture(name: str) -> np.ndarray:
    """A sky picture of render/assets as (h, w, 3) linear light (read once)."""
    sky = read_hdr(SKY_FOLDER / name)
    sky.flags.writeable = False
    return sky


@functools.lru_cache(maxsize=4)
def sky_sun(name: str) -> tuple[float, float]:
    """Azimuth and elevation (radians, as LAGOON's directions) of a sky picture's sun:
    its brightest point."""
    sky = sky_picture(name)
    brightness = sky @ np.array([0.2126, 0.7152, 0.0722])
    row, column = np.unravel_index(int(np.argmax(brightness)), brightness.shape)
    d = panorama_directions(sky.shape[1])[row, column]
    return math.atan2(float(d[1]), float(d[0])), math.asin(float(np.clip(d[2], -1.0, 1.0)))


def _track_z(direction) -> list[float]:
    """The quaternion (x, y, z, w) turning +z onto a direction (a sun lamp shines
    along its -z, so its +z points at the sun)."""
    d = np.asarray(direction, float)
    d /= np.linalg.norm(d)
    up = np.array([0.0, 1.0, 0.0]) if abs(d[1]) < 0.9 else np.array([1.0, 0.0, 0.0])
    x = np.cross(up, d)
    x /= np.linalg.norm(x)
    y = np.cross(d, x)
    return _quaternion(np.stack([x, y, d], axis=1))


def _wxyz(q) -> list[float]:
    return [float(q[3]), float(q[0]), float(q[1]), float(q[2])]


# ======================================================================== Blender
def find_blender() -> str | None:
    """The Blender executable: ``$LAGOON_BLENDER``, ``blender`` on the PATH, or the
    usual install places (macOS, Windows, Linux); None if there is none."""
    given = os.environ.get("LAGOON_BLENDER")
    if given:
        return given if Path(given).exists() else None
    found = shutil.which("blender")
    if found:
        return found
    places = [
        Path("/Applications/Blender.app/Contents/MacOS/Blender"),
        Path.home() / "Applications/Blender.app/Contents/MacOS/Blender",
        Path("/snap/bin/blender"),
        Path("/usr/bin/blender"),
    ]
    for root in (os.environ.get("ProgramFiles"), r"C:\Program Files"):
        if root:
            folder = Path(root) / "Blender Foundation"
            if folder.is_dir():
                places += sorted(folder.glob("Blender*/blender.exe"), reverse=True)
    return next((str(p) for p in places if p.exists()), None)


def render_folder(
    folder: str | Path,
    blender: str | None = None,
    progress: Progress | None = None,
    resume: bool = False,
    cancel: Callable[[], bool] | None = None,
    status: Callable[[str], None] | None = None,
) -> list[Path]:
    """Render an exported folder with Blender (Cycles), frame by frame into PNGs
    beside the data. Returns the images written. ``progress(done, total)`` hears of
    each frame; returning False stops Blender (the frames done are kept), as does
    ``cancel()`` turning True (asked twice a second, for frames that take long).
    Blender's output goes to ``blender.log`` in the folder, line by line, and to
    ``status(line)``."""
    folder = Path(folder).expanduser()
    blender = blender or find_blender()
    if blender is None:
        raise RuntimeError(
            "Blender not found: install Blender 4.2 or newer (blender.org), or set "
            "LAGOON_BLENDER to its executable"
        )
    command = [
        blender,
        "-b",
        "--factory-startup",
        "--python-exit-code",
        "1",
        "-P",
        str(BLENDER_SCRIPT),
        "--",
        str(folder),
    ]
    if resume:
        command.append("--resume")
    log.info("path tracing: %s", " ".join(command))
    environment = {**os.environ, "PYTHONUNBUFFERED": "1"}
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        errors="replace",
        env=environment,
    )
    log_file = (folder / "blender.log").open("w", encoding="utf-8", buffering=1)
    log_file.write(" ".join(command) + "\n")
    tail: list[str] = []
    stopped = False
    watching = threading.Event()
    if cancel is not None:

        def watch() -> None:
            while not watching.wait(0.5):
                if cancel():
                    process.terminate()
                    return

        threading.Thread(target=watch, daemon=True, name="lagoon-cycles-cancel").start()
    assert process.stdout is not None
    for line in process.stdout:
        tail = (tail + [line.rstrip()])[-40:]
        log_file.write(line)
        if status is not None:
            status(line.rstrip())
        if line.startswith("LAGOON frame "):
            done, _, total = line.split()[2].partition("/")
            if progress is not None and not progress(int(done), int(total)):
                stopped = True
                process.terminate()
                break
        elif line.startswith("LAGOON "):
            log.info(line.rstrip())
    code = process.wait()
    log_file.write(f"exit code {code}\n")
    log_file.close()
    watching.set()
    stopped = stopped or (cancel is not None and cancel())
    if code != 0 and not stopped:
        raise RuntimeError("Blender failed:\n" + "\n".join(tail))
    scene = json.loads((folder / "scene.json").read_text(encoding="utf-8"))
    images = [folder / f["image"] for f in scene["frames"]]
    return [p for p in images if p.exists()]


def blender_progress(line: str) -> str | None:
    """What a line of Blender's output says about how far it is, briefly ("sample
    40/128, 00:45 left", "compiling GPU kernels", LAGOON's own steps), or None."""
    import re

    line = line.strip()
    if line.startswith("LAGOON "):
        return line[len("LAGOON ") :]
    sample = re.search(r"Sample (\d+)/(\d+)", line)
    if sample:
        left = re.search(r"Remaining:\s*([0-9:.]+)", line)
        text = f"sample {sample.group(1)}/{sample.group(2)}"
        return f"{text}, {left.group(1)} left" if left else text
    lowered = line.lower()
    if "compil" in lowered and ("kernel" in lowered or "shader" in lowered):
        return "compiling Cycles' GPU kernels (the first time with new features: minutes)"
    if "loading render kernels" in lowered:
        return "loading Cycles' kernels"
    if "synchroniz" in lowered or ("updating" in lowered and "mesh" in lowered):
        return "preparing the scene in Cycles"
    return None


def encode_images(images: list[Path], path: Path, fps: float) -> Path:
    """Numbered PNGs into an H.264 film with imageio-ffmpeg's ffmpeg."""
    import imageio_ffmpeg

    folder = images[0].parent
    pattern = folder / "frame_%05d.png"
    command = [
        imageio_ffmpeg.get_ffmpeg_exe(),
        "-y",
        "-loglevel",
        "error",
        "-framerate",
        f"{fps:g}",
        "-start_number",
        images[0].stem.rpartition("_")[2],
        "-i",
        str(pattern),
        "-frames:v",
        str(len(images)),
        "-c:v",
        "libx264",
        "-crf",
        "16",
        "-preset",
        "slow",
        "-pix_fmt",
        "yuv420p",
        "-vf",
        "pad=ceil(iw/2)*2:ceil(ih/2)*2",
        "-movflags",
        "+faststart",
        str(path),
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(command, check=True, capture_output=True)
    return path


def path_trace(
    session,
    output: str | Path,
    times: list[float] | None = None,
    size: tuple[int, int] = (1920, 1080),
    look: str = "open_ocean",
    samples: int = 64,
    fps: float = 25.0,
    blender: str | None = None,
    progress: Progress | None = None,
    work: str | Path | None = None,
    resume: bool = True,
    cancel: Callable[[], bool] | None = None,
    status: Callable[[str], None] | None = None,
    foam: bool | str | float = True,
) -> Path:
    """Path trace the session with Blender: a still (``.png``, at ``times[0]`` or the
    session's time), a film (``.mp4``/``.mov``) or numbered PNGs (a folder). The
    exported scene is kept in ``work`` (default: :func:`work_folder`; for a still it
    is removed once the image is there, and kept with Blender's log if it fails).
    With ``resume``, frames rendered before from the same data, camera and settings
    are kept (a film stopped earlier goes on where it stopped). ``status`` hears
    each line Blender prints. Returns what was written."""
    from lagoonviz.render.movie import frame_times

    output = Path(output).expanduser()
    still = output.suffix.lower() == ".png"
    if times is None:
        times = [session.scene.time] if still else frame_times(session)
    times = list(times)[:1] if still else list(times)
    keep = work is not None or not still
    work = Path(work) if work is not None else work_folder(output)
    export_scene(session, work, times, size, look, samples, foam=foam)
    written = finish(work, output, fps, blender, progress, resume, cancel, status)
    if not keep:  # a still keeps no frames (on failure, the folder and its log stay)
        shutil.rmtree(work, ignore_errors=True)
    return written


def work_folder(output: str | Path) -> Path:
    """Where the exported scene and the rendered frames of an output are kept: beside
    a film or a still, ``<name>.cycles``; for a folder of PNGs, the folder."""
    from lagoonviz.render.movie import MOVIE_SUFFIXES

    output = Path(output).expanduser()
    suffix = output.suffix.lower()
    return output.with_suffix(".cycles") if suffix in (*MOVIE_SUFFIXES, ".png") else output


def finish(
    work: str | Path,
    output: str | Path,
    fps: float = 25.0,
    blender: str | None = None,
    progress: Progress | None = None,
    resume: bool = True,
    cancel: Callable[[], bool] | None = None,
    status: Callable[[str], None] | None = None,
) -> Path:
    """Render an exported folder with Blender and deliver it: the still copied to
    ``output`` (``.png``), the frames encoded into the film (``.mp4``/``.mov``), or
    the folder of PNGs. Takes long; touches no session (safe in a thread)."""
    from lagoonviz.render.movie import MOVIE_SUFFIXES, movie_available

    work, output = Path(work), Path(output).expanduser()
    suffix = output.suffix.lower()
    images = render_folder(work, blender, progress, resume=resume, cancel=cancel, status=status)
    if not images:
        raise RuntimeError("Blender wrote no images")
    if suffix == ".png":
        output.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(images[0], output)
        return output
    if suffix in MOVIE_SUFFIXES:
        if not movie_available():
            log.warning("imageio-ffmpeg missing: the frames are in %s", work)
            return work
        return encode_images(images, output, fps)
    return work


def blender_version(blender: str | None = None) -> str | None:
    """Blender's version line ("Blender 4.2.3 LTS"), or None without Blender."""
    blender = blender or find_blender()
    if blender is None:
        return None
    try:
        result = subprocess.run(
            [blender, "-b", "--version"], capture_output=True, text=True, timeout=60
        )
    except (OSError, subprocess.SubprocessError):
        return None
    line = next((x for x in result.stdout.splitlines() if x.startswith("Blender")), "")
    return line.strip() or None


if __name__ == "__main__":  # pragma: no cover
    sys.exit(print(find_blender() or "Blender not found"))
