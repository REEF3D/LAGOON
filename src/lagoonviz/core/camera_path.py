# SPDX-License-Identifier: GPL-3.0-or-later
# Architect: Hans Bihs
"""Camera keyframes: a camera path over the timeline, for animations.

A key holds the camera as the 3D view had it (``SceneRenderer.camera_state``:
position, rotation as a quaternion (x, y, z, w), field of view, zoom, ...) at a
time. Between keys the camera moves on a smooth path: the position on a cubic
Hermite curve through the keys whose tangents are the finite differences in time
(Catmull-Rom with the keys' own spacing), so its speed changes gently as it passes
a key; the rotation the same way on the quaternions (signs aligned, normalised),
the field of view and zoom alike. At the first and last key the camera eases in
and out (no speed there: the easing is in time, the path keeps its shape); before
the first and after the last it stays put.

    keys = [CameraKey(0.0, state_a), CameraKey(10.0, state_b), CameraKey(20.0, state_c)]
    state = camera_at(keys, 12.5)
    keys = orbit_keys(state_a, centre=(50, 10, 0), start=0.0, end=60.0)   # once around

NumPy only; the session keeps the keys (``Session.add_camera_key``), the views and
the film export follow them when the camera path is on.
"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass
from typing import Any

import numpy as np

#: state entries that move between keys (the others are taken from the key before)
VECTORS = ("position",)
QUATERNIONS = ("rotation",)
SCALARS = ("fov", "zoom", "width", "height")


@dataclass
class CameraKey:
    time: float
    state: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {"time": float(self.time), "state": copy.deepcopy(self.state)}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CameraKey:
        return cls(float(data["time"]), dict(data["state"]))


def sorted_keys(keys: list[CameraKey]) -> list[CameraKey]:
    """Keys in time order, one per time (the later given wins)."""
    by_time: dict[float, CameraKey] = {}
    for key in keys:
        by_time[round(float(key.time), 9)] = key
    return [by_time[t] for t in sorted(by_time)]


def _hermite(t: float, t0: float, t1: float, p0, p1, m0, m1) -> np.ndarray:
    h = t1 - t0
    s = (t - t0) / h
    s2, s3 = s * s, s * s * s
    return (
        (2 * s3 - 3 * s2 + 1) * p0
        + (s3 - 2 * s2 + s) * h * m0
        + (-2 * s3 + 3 * s2) * p1
        + (s3 - s2) * h * m1
    )


def _tangents(times: np.ndarray, values: np.ndarray) -> np.ndarray:
    """Finite differences in time at the inner keys; at the first and last the
    tangent of a parabola through the end and its neighbour (so the path keeps its
    shape there: an orbit stays round; the easing is done in time, :func:`_eased`)."""
    m = np.zeros_like(values)
    n = len(times)
    if n == 2:
        m[:] = (values[1] - values[0]) / (times[1] - times[0])
    elif n > 2:
        m[1:-1] = (values[2:] - values[:-2]) / (times[2:] - times[:-2])[:, None]
        m[0] = 2 * (values[1] - values[0]) / (times[1] - times[0]) - m[1]
        m[-1] = 2 * (values[-1] - values[-2]) / (times[-1] - times[-2]) - m[-2]
    return m


def _eased(s: float, first: bool, last: bool) -> float:
    """Where along a segment (0..1) the camera is at the share ``s`` of its time: it
    starts from rest in the first segment and comes to rest in the last, at full
    speed where it meets the next (2 s^2 - s^3 and its mirror; one segment:
    smoothstep)."""
    if first and last:
        return s * s * (3 - 2 * s)
    if first:
        return s * s * (2 - s)
    if last:
        r = 1 - s
        return 1 - r * r * (2 - r)
    return s


def _aligned(quaternions: np.ndarray) -> np.ndarray:
    """Quaternions with signs chosen so each is on the near side of the one before
    (q and -q are the same rotation; the path must not take the long way)."""
    q = np.array(quaternions, float)
    for i in range(1, len(q)):
        if q[i] @ q[i - 1] < 0:
            q[i] = -q[i]
    return q


def camera_at(keys: list[CameraKey], time: float) -> dict[str, Any] | None:
    """The camera state at a time on the path through the keys; None without keys."""
    keys = sorted_keys(keys)
    if not keys:
        return None
    if len(keys) == 1 or time <= keys[0].time:
        return copy.deepcopy(keys[0].state)
    if time >= keys[-1].time:
        return copy.deepcopy(keys[-1].state)
    times = np.array([k.time for k in keys], float)
    i = int(np.searchsorted(times, time, side="right")) - 1
    state = copy.deepcopy(keys[i].state)

    share = (time - times[i]) / (times[i + 1] - times[i])
    when = times[i] + _eased(share, i == 0, i == len(keys) - 2) * (times[i + 1] - times[i])

    def path(name: str, values: np.ndarray) -> np.ndarray:
        m = _tangents(times, values)
        return _hermite(when, times[i], times[i + 1], values[i], values[i + 1], m[i], m[i + 1])

    for name in VECTORS:
        if all(name in k.state for k in keys):
            values = np.array([k.state[name] for k in keys], float)
            state[name] = path(name, values).tolist()
    for name in QUATERNIONS:
        if all(name in k.state for k in keys):
            q = path(name, _aligned([k.state[name] for k in keys]))
            state[name] = (q / max(np.linalg.norm(q), 1e-12)).tolist()
    for name in SCALARS:
        if all(isinstance(k.state.get(name), int | float) for k in keys):
            values = np.array([[k.state[name]] for k in keys], float)
            state[name] = float(path(name, values)[0])
    return state


def multiply(a, b) -> np.ndarray:
    """The quaternion product a b (x, y, z, w): b's rotation, then a's."""
    ax, ay, az, aw = (float(v) for v in a)
    bx, by, bz, bw = (float(v) for v in b)
    return np.array(
        [
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
            aw * bw - ax * bx - ay * by - az * bz,
        ]
    )


def orbit_keys(
    state: dict[str, Any],
    centre,
    start: float,
    end: float,
    degrees: float = 360.0,
    step: float = 30.0,
) -> list[CameraKey]:
    """Keys that take the camera from ``state`` around the vertical axis through
    ``centre`` by ``degrees`` (counter-clockwise seen from above) between the times
    ``start`` and ``end``: one key every ``step`` degrees or less, so the smooth path
    stays on the circle; it eases in and out at the ends."""
    centre = np.asarray(centre, float)
    count = max(2, math.ceil(abs(float(degrees)) / max(float(step), 1e-6)) + 1)
    position = np.asarray(state.get("position", (0.0, 0.0, 0.0)), float)
    rotation = np.asarray(state.get("rotation", (0.0, 0.0, 0.0, 1.0)), float)
    keys = []
    for i in range(count):
        share = i / (count - 1)
        angle = math.radians(float(degrees) * share)
        c, s = math.cos(angle), math.sin(angle)
        offset = position - centre
        moved = centre + [c * offset[0] - s * offset[1], s * offset[0] + c * offset[1], offset[2]]
        turn = (0.0, 0.0, math.sin(angle / 2), math.cos(angle / 2))
        key_state = copy.deepcopy(state)
        key_state["position"] = moved.tolist()
        key_state["rotation"] = multiply(turn, rotation).tolist()
        keys.append(CameraKey(float(start) + share * (float(end) - float(start)), key_state))
    return keys
