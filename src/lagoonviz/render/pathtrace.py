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
    surface_of,
)

log = logging.getLogger("lagoon.pathtrace")

#: the script Blender runs (it is not imported here)
BLENDER_SCRIPT = Path(__file__).with_name("blender_cycles.py")
#: the version of the folder layout (scene.json "version")
VERSION = 1
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
            values = row[:, channel].tobytes()
            for at in range(0, width, 128):  # literal runs of up to 128 bytes
                piece = values[at : at + 128]
                chunks.append(bytes((len(piece),)) + piece)
    Path(path).write_bytes(b"".join(chunks))


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

    def __init__(self, preset: Preset, pixel: float) -> None:
        self.preset = preset
        self.pixel = pixel
        self.objects: list[dict[str, Any]] = []
        self.arrays: dict[str, np.ndarray] = {}
        self.materials: dict[str, dict[str, Any]] = {}

    def material(self, spec: dict[str, Any]) -> str:
        for key, known in self.materials.items():
            if known == spec:
                return key
        key = f"m{len(self.materials)}"
        self.materials[key] = spec
        return key

    def add(self, name: str, positions, triangles, material: dict, rgb=None) -> None:
        positions = np.asarray(positions, np.float32).reshape(-1, 3)
        triangles = np.asarray(triangles, np.uint32).reshape(-1, 3)
        if not len(triangles):
            return
        finite = np.isfinite(positions).all(axis=1)
        if not finite.all():  # dry land, outside the record: no triangles there
            triangles = triangles[finite[triangles].all(axis=1)]
            positions = np.where(finite[:, None], positions, 0.0).astype(np.float32)
            if not len(triangles):
                return
        index = len(self.objects)
        self.arrays[f"positions_{index}"] = positions
        self.arrays[f"triangles_{index}"] = triangles
        if rgb is not None:
            self.arrays[f"colors_{index}"] = np.asarray(rgb, np.float32).reshape(-1, 3)
        self.objects.append(
            {"name": name, "material": self.material(material), "colors": rgb is not None}
        )

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

    def surface(self, kind: str, shown) -> None:
        if not shown.visible or shown.opacity <= 0:
            return
        values = shown.values
        rgb = None
        if values is not None:
            rgb = linear(colors(values, shown.colormap, shown.vmin, shown.vmax)[:, :3])
        if shown.point_size is not None:  # particles
            keep = np.isfinite(shown.positions).all(axis=1)
            positions, triangles = spheres(
                shown.positions[keep], 0.5 * shown.point_size * self.pixel
            )
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
        spec = {
            "color": None
            if rgb is not None
            else linear_hex(self.preset.water_color if water else shown.color),
            "roughness": (self.preset.water_roughness if water else s.roughness)
            if rgb is None
            else max(0.3, s.roughness if not water else 0.3),
            "metallic": s.metalness,
            "ior": s.ior,
            # clear water: a sheet that refracts, tinted by its colour; coloured by a
            # variable it is opaque, so the colours read
            "transmission": 0.9 if water and rgb is None else 0.0,
            "coat": s.clearcoat,
            "alpha": round(float(shown.opacity), 4),
            "grain": None,
        }
        if s.grain and rgb is None:
            finite = shown.positions[np.isfinite(shown.positions).all(axis=1)]
            span = float(np.ptp(finite[:, :2], axis=0).max()) if len(finite) else 0.0
            spec["grain"] = {
                "kind": s.grain,
                "tile": round(max(s.tile, span / 3), 6),
                "relief": s.relief,
            }
        self.add(shown.item_id, shown.positions, shown.triangles, spec, rgb)


def _frame_objects(session, preset: Preset, pixel: float) -> _Collector:
    """The objects of the session's current time, as the 3D view shows them."""
    data = session.data
    out = _Collector(preset, pixel)
    for item_id in data.data_items():
        item = session.scene.get(item_id)
        if not item.visible:
            continue
        mesh = data.load(item_id)
        if mesh is None:
            continue
        shown = data.display(item_id, mesh)
        out.surface(item.kind, shown)
        if not shown.visible:
            continue
        overlays = (
            ("contours", data.contour_lines(item_id, mesh, shown), CONTOUR_WIDTH, None),
            ("vectors", data.vector_arrows(item_id, mesh, shown), VECTOR_WIDTH, None),
            ("trails", data.particle_trails(item_id, mesh), TRAIL_WIDTH, "color"),
            ("outlines", data.patch_outlines(item_id, mesh, shown), OUTLINE_WIDTH, "color"),
        )
        for label, lines, width, own in overlays:
            if lines is None or not lines.visible or not lines.segments.size:
                continue
            color = getattr(lines, own) if own else preset.contours
            out.lines(f"{item_id}#{label}", lines.segments, width, color)
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


def export_scene(
    session,
    folder: str | Path,
    times: list[float],
    size: tuple[int, int] = (1920, 1080),
    look: str = "open_ocean",
    samples: int = 128,
    progress: Progress | None = None,
) -> Path:
    """Write the session at ``times`` for Blender: ``scene.json``, a ``frame_NNNNN.npz``
    per time and ``environment.hdr``. Returns the path of ``scene.json``."""
    folder = Path(folder).expanduser()
    folder.mkdir(parents=True, exist_ok=True)
    preset = look_preset(look)
    times = [float(t) for t in times]
    if not times:
        raise ValueError("nothing to render: no times")
    size = (int(size[0]), int(size[1]))
    original_time = session.scene.time
    frames = []
    materials: dict[str, dict[str, Any]] = {}
    low = high = None
    try:
        for number, time in enumerate(times):
            session.set_time(time)
            bounds = _bounds(session)
            if bounds is not None:
                low = bounds[0] if low is None else np.minimum(low, bounds[0])
                high = bounds[1] if high is None else np.maximum(high, bounds[1])
            camera = blender_camera(camera_state_at(session, time, bounds), size)
            centre = (bounds[0] + bounds[1]) / 2 if bounds is not None else np.zeros(3)
            collector = _frame_objects(session, preset, pixel_size(camera, centre, size[1]))
            # one material table for the film: the frame's keys renamed into it
            rename = {}
            for key, spec in collector.materials.items():
                found = next((k for k, v in materials.items() if v == spec), None)
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
    for frame in frames:  # clip planes around everything, from wherever the camera is
        distance = float(np.linalg.norm(np.asarray(frame["camera"]["position"]) - (low + high) / 2))
        frame["camera"]["clip_start"] = max(1e-4 * extent, (distance - extent) * 1e-3, 1e-5)
        frame["camera"]["clip_end"] = (distance + extent) * 4.0
    write_hdr(folder / "environment.hdr", environment_panorama(preset, 1024))
    sun = np.asarray(preset.sun_direction, float)
    sun /= np.linalg.norm(sun)
    world: dict[str, Any] = {
        "environment": "environment.hdr",
        "strength": preset.env_intensity,
        "background": [linear_hex(c) for c in preset.background] if preset.background else None,
        "fog": None,
    }
    if preset.fog is not None:
        color, half = preset.fog
        margin = 0.1 * (high - low) + 0.01 * extent
        world["fog"] = {
            "color": linear_hex(color),
            "density": math.log(2.0) / (max(half, 1e-6) * extent),
            "low": (low - margin).tolist(),
            "high": (high + margin).tolist(),
        }
    scene = {
        "version": VERSION,
        "look": preset.name,
        "size": list(size),
        "samples": int(samples),
        "view_transform": "AgX",
        "world": world,
        "sun": {
            "rotation_wxyz": _wxyz(_track_z(sun)),
            "direction": sun.tolist(),
            "color": linear_hex(preset.sun_color),
            "strength": preset.sun,
            "angle": 1.0,  # degrees: soft edges, as the sky's sun
            "shadows": preset.shadows,
        },
        "materials": materials,
        "frames": frames,
    }
    # a frame's key: everything its image depends on. Blender keeps an image whose
    # key it wrote beside it (frame_NNNNN.key) when asked to resume.
    common = {k: scene[k] for k in ("version", "look", "size", "samples", "view_transform")}
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
    samples: int = 128,
    fps: float = 25.0,
    blender: str | None = None,
    progress: Progress | None = None,
    work: str | Path | None = None,
    resume: bool = True,
    cancel: Callable[[], bool] | None = None,
    status: Callable[[str], None] | None = None,
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
    export_scene(session, work, times, size, look, samples)
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
