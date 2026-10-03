# SPDX-License-Identifier: GPL-3.0-or-later
"""Mouse controls of the 3D view: orbiting without a lock at the poles, and a 2D mode.

pygfx's OrbitController stops the camera 1 degree short of straight up and straight
down, so the view cannot be turned over. This one turns freely, all the way round,
about the centre of the view (where "Frame all" and the named views put the middle
of the data; zooming keeps it, panning moves it with the view).

In 2D mode the view only pans and zooms: no rotation at all.
"""

from __future__ import annotations

import numpy as np
import pygfx as gfx
import pylinalg as la


def orbit_rotation(
    rotation: np.ndarray, up: np.ndarray, delta_azimuth: float, delta_elevation: float
) -> np.ndarray:
    """New camera rotation (quaternion x, y, z, w) after turning about the world's up
    axis and the camera's own horizontal axis, with no limit on the elevation.

    Upside down (past a pole), the turn about the up axis is reversed, so a drag to
    the right still moves the foreground to the right.
    """
    camera_up = la.vec_transform_quat((0.0, 1.0, 0.0), rotation)
    if float(np.dot(camera_up, up)) < 0.0:
        delta_azimuth = -delta_azimuth
    r_azimuth = la.quat_from_axis_angle(up, -delta_azimuth)
    r_elevation = la.quat_from_axis_angle((1.0, 0.0, 0.0), -delta_elevation)
    return la.quat_mul(r_azimuth, la.quat_mul(rotation, r_elevation))


class FreeOrbitController(gfx.OrbitController):
    """Orbit about the centre of the view, through the poles; or, in 2D mode, only
    pan and zoom."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.locked = False  # 2D: no rotation

    def _update_rotate(self, delta):
        assert isinstance(delta, tuple) and len(delta) == 2
        if self.locked:
            return
        if self._custom_target is not None:  # not used by LAGOON: keep pygfx's way
            super()._update_rotate(delta)
            return
        state = self._get_camera_state()
        position = np.asarray(state["position"], dtype=float)
        rotation = np.asarray(state["rotation"], dtype=float)
        up = np.asarray(state.get("reference_up", (0.0, 0.0, 1.0)), dtype=float)
        new_rotation = orbit_rotation(rotation, up, *delta)
        # keep the point at the centre of the view where it is
        before = self._get_target_vec(state, rotation=rotation)
        after = self._get_target_vec(state, rotation=new_rotation)
        self._set_camera_state({"position": position + before - after, "rotation": new_rotation})
