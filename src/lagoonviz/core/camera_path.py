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
and out (no speed there); before the first and after the last it stays put.

    keys = [CameraKey(0.0, state_a), CameraKey(10.0, state_b), CameraKey(20.0, state_c)]
    state = camera_at(keys, 12.5)

NumPy only; the session keeps the keys (``Session.add_camera_key``), the views and
the film export follow them when the camera path is on.
"""

from __future__ import annotations

import copy
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
    """Finite differences in time at the inner keys, zero at the first and last
    (the camera eases in and out there)."""
    m = np.zeros_like(values)
    if len(times) > 2:
        m[1:-1] = (values[2:] - values[:-2]) / (times[2:] - times[:-2])[:, None]
    return m


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

    def path(name: str, values: np.ndarray) -> np.ndarray:
        m = _tangents(times, values)
        return _hermite(time, times[i], times[i + 1], values[i], values[i + 1], m[i], m[i + 1])

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
