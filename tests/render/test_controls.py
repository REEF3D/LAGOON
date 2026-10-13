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


def test_frame_all_fits_a_tall_scene(reef3d_run):
    """With strong vertical exaggeration the scene is tall: framing used to cut it off
    at the top in a wide window. Every corner of the data must be in view."""
    from lagoonviz.core import Session
    from lagoonviz.render.scene3d import SceneRenderer

    session = Session()
    session.open_case(reef3d_run)
    session.set_property("case-1", "exaggeration", 5.0)
    view = SceneRenderer()
    for item_id in session.data.data_items():
        mesh = session.data.load(item_id)
        if mesh is not None:
            view.update(session.scene.get(item_id).kind, session.data.display(item_id, mesh))
    for name in ("perspective", "front", "top"):
        view.frame(name)
        view.camera.set_view_size(320, 180)
        low, high = view.bounds()
        corners = np.array(
            [
                (x, y, z)
                for x in (low[0], high[0])
                for y in (low[1], high[1])
                for z in (low[2], high[2])
            ]
        )
        matrix = np.asarray(view.camera.camera_matrix, dtype=float)
        clip = np.c_[corners, np.ones(8)] @ matrix.T
        ndc = clip[:, :2] / clip[:, 3:4]
        assert np.abs(ndc).max() <= 1.0, (name, np.abs(ndc).max(axis=0))


def test_an_empty_surface_shows_nothing(reef3d_run):
    """An iso-value the field does not reach gives no triangles: nothing is drawn
    (pygfx refused empty buffers, and the item failed to show)."""
    from lagoonviz.core import Session
    from lagoonviz.render.scene3d import SceneRenderer

    session = Session()
    session.open_case(reef3d_run)
    iso = session.add_iso_surface("pressure", 1e9)
    view = SceneRenderer()
    mesh = session.data.load(iso)
    assert mesh.triangles.size == 0
    view.update("iso_surface", session.data.display(iso, mesh))
    assert iso not in view.item_ids


def test_north_turns_the_3d_view_about_the_model(reef3d_run):
    """A georeferenced model turned on the map: in 3D the camera turns with north
    about the middle of the model, so the map stays where it was on the screen."""
    import math

    from lagoonviz.core import Session
    from lagoonviz.render.scene3d import SceneRenderer

    session = Session()
    session.open_case(reef3d_run)
    view = SceneRenderer()
    for item_id in session.data.data_items():
        mesh = session.data.load(item_id)
        if mesh is not None:
            view.update(session.scene.get(item_id).kind, session.data.display(item_id, mesh))
    view.frame("perspective")
    low, high = view.bounds()
    pivot = (low + high) / 2
    view.set_north(None)  # what the camera knows
    before = np.asarray(view.camera.local.position, float)
    forward = np.asarray(view.camera.world.forward, float)
    angle = math.radians(30.0)
    view.set_north((math.sin(angle), math.cos(angle)))  # the model turned 30° on the map
    after = np.asarray(view.camera.local.position, float)
    # about the vertical through the middle: the same height and distance from it
    assert after[2] == pytest.approx(before[2])
    assert np.linalg.norm((after - pivot)[:2]) == pytest.approx(
        np.linalg.norm((before - pivot)[:2])
    )
    turned = np.asarray(view.camera.world.forward, float)
    assert turned[2] == pytest.approx(forward[2], abs=1e-6)
    heading = math.atan2(turned[1], turned[0]) - math.atan2(forward[1], forward[0])
    # the map turned back by those 30° in the model's frame: the camera turns with it
    assert math.remainder(heading + angle, 2 * math.pi) == pytest.approx(0.0, abs=1e-6)
    # a camera set from a saved session already holds its north: nothing turns
    state = view.camera_state()
    view.set_camera_state(state)
    view.set_north((0.5, math.sqrt(0.75)))
    assert np.asarray(view.camera.local.position, float) == pytest.approx(after)


def test_north_up_turns_any_view_so_north_points_up(reef3d_run):
    import math

    from lagoonviz.core import Session
    from lagoonviz.render.scene3d import SceneRenderer

    session = Session()
    session.open_case(reef3d_run)
    view = SceneRenderer()
    for item_id in session.data.data_items():
        mesh = session.data.load(item_id)
        if mesh is not None:
            view.update(session.scene.get(item_id).kind, session.data.display(item_id, mesh))
    angle = math.radians(40.0)
    north = (math.sin(angle), math.cos(angle))  # the model turned 40° on the map
    for name in ("perspective", "front", "top"):
        view.frame(name)
        view.set_north(north)
        view.point_north()
        up = np.asarray(view.camera.world.up, float)
        forward = np.asarray(view.camera.world.forward, float)
        heading = up[:2] if np.hypot(*up[:2]) > 1e-3 else forward[:2]
        heading = heading / np.linalg.norm(heading)
        assert heading == pytest.approx(north, abs=1e-6), name
    # without a georeference: the model's y
    view.frame("perspective")
    view.set_north(None)
    view.point_north()
    up = np.asarray(view.camera.world.up, float)
    assert (up[:2] / np.linalg.norm(up[:2])) == pytest.approx((0.0, 1.0), abs=1e-6)
