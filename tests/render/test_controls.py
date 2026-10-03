"""Mouse controls of the 3D view: free orbit through the poles, 2D lock."""

import numpy as np
import pytest

gfx = pytest.importorskip("pygfx")
la = pytest.importorskip("pylinalg")

from lagoonviz.render.controls import FreeOrbitController, orbit_rotation  # noqa: E402

RECT = (0, 0, 400, 300)


def _camera():
    camera = gfx.PerspectiveCamera(45, 4 / 3)
    camera.show_object((0.0, 0.0, 0.0, 10.0), view_dir=(0.0, 1.0, 0.0), up=(0.0, 0.0, 1.0))
    return camera


def _forward(rotation):
    return np.asarray(la.vec_transform_quat((0.0, 0.0, -1.0), rotation))


def _centre(controller, camera):
    state = camera.get_state()
    return np.asarray(state["position"]) + controller._get_target_vec(state)


def test_turns_over_the_pole():
    """pygfx stops 1 degree short of straight down; LAGOON turns on."""
    up = np.array((0.0, 0.0, 1.0))
    rotation = np.asarray(_camera().local.rotation)
    over = orbit_rotation(rotation, up, 0.0, np.radians(100))
    # past the pole: the camera now looks slightly backwards, steeply down (or up)
    forward = _forward(over)
    assert abs(forward[2]) == pytest.approx(np.sin(np.radians(80)), abs=1e-6)
    assert forward[1] < 0  # it started looking along +y
    full = rotation
    for _ in range(36):
        full = orbit_rotation(full, up, 0.0, np.radians(10))
    np.testing.assert_allclose(_forward(full), _forward(rotation), atol=1e-6)


def test_orbits_about_the_centre_of_the_view():
    camera = _camera()
    controller = FreeOrbitController(camera)
    before = _centre(controller, camera)
    controller.rotate((0.7, 1.9), RECT)  # past the pole
    np.testing.assert_allclose(_centre(controller, camera), before, atol=1e-4)
    controller.zoom((0.5, 0.5), RECT)
    controller.rotate((-0.3, 0.4), RECT)
    np.testing.assert_allclose(_centre(controller, camera), before, atol=1e-4)


def test_2d_view_does_not_rotate():
    camera = _camera()
    controller = FreeOrbitController(camera)
    controller.locked = True
    rotation = np.asarray(camera.local.rotation).copy()
    controller.rotate((0.5, 0.5), RECT)
    np.testing.assert_allclose(camera.local.rotation, rotation)
