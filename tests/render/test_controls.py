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


def test_the_top_view_shows_the_models_axes(reef3d_run):
    """A model turned on the map (georeferenced): from the top the view shows the
    model's x to the right and its y up the screen (asked for, 2026-11-02), not north
    up; turning the model on the map then turns the camera with it (the map stays
    where it was); North up turns it to north."""
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

    def heading():
        up = np.asarray(view.camera.world.up, float)[:2]
        return math.atan2(up[1], up[0])

    angle = math.radians(40.0)
    view.set_north((math.sin(angle), math.cos(angle)))  # the case opened: north known
    view.frame("top")
    assert heading() == pytest.approx(math.pi / 2, abs=1e-6)  # the model's y up
    right = np.asarray(view.camera.world.right, float)
    assert right[:2] == pytest.approx((1.0, 0.0), abs=1e-6)  # its x to the right
    # turned 10° more on the map: the camera turns with north
    turned = math.radians(50.0)
    view.set_north((math.sin(turned), math.cos(turned)))
    change = math.atan2(math.cos(turned), math.sin(turned)) - math.atan2(
        math.cos(angle), math.sin(angle)
    )
    assert math.remainder(heading() - math.pi / 2 - change, 2 * math.pi) == pytest.approx(
        0.0, abs=1e-6
    )
    # North up: north up the screen; Top again: the model's axes
    view.point_north()
    assert heading() == pytest.approx(math.atan2(math.cos(turned), math.sin(turned)), abs=1e-6)
    view.frame("top")
    assert heading() == pytest.approx(math.pi / 2, abs=1e-6)


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


def test_the_view_turns_to_a_heading(reef3d_run):
    """The view's turn about the vertical (the slider and box by Probe): from the top
    the model's y up is 0°, its x up 90°; in 3D where it looks; turning keeps the
    tilt, the distance from the middle of the view and the point there."""
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
    view.frame("top")
    assert math.remainder(view.heading(), 360.0) == pytest.approx(0.0, abs=1e-6)
    centre = view._view_centre()
    view.set_heading(90.0)
    assert view.heading() == pytest.approx(90.0, abs=1e-6)
    up = np.asarray(view.camera.world.up, float)
    assert up[:2] == pytest.approx((1.0, 0.0), abs=1e-6)  # the model's x up the screen
    assert view._view_centre()[:2] == pytest.approx(centre[:2], abs=1e-6)
    view.set_heading(-45.0)  # the same as 315°
    assert view.heading() == pytest.approx(315.0, abs=1e-6)
    view.frame("perspective")
    forward = np.asarray(view.camera.world.forward, float)
    position = np.asarray(view.camera.local.position, float)
    centre = view._view_centre()
    start = view.heading()
    expected = math.degrees(math.atan2(forward[0], forward[1])) % 360.0
    assert math.remainder(start - expected, 360.0) == pytest.approx(0.0, abs=1e-6)
    for step in range(1, 13):  # gradually all the way round
        view.set_heading(start + 30.0 * step)
        assert math.remainder(view.heading() - start - 30.0 * step, 360.0) == pytest.approx(
            0.0, abs=1e-6
        )
        turned = np.asarray(view.camera.world.forward, float)
        assert turned[2] == pytest.approx(forward[2], abs=1e-9)  # the tilt kept
        moved = np.asarray(view.camera.local.position, float)
        assert moved[2] == pytest.approx(position[2])
        assert np.linalg.norm((moved - centre)[:2]) == pytest.approx(
            np.linalg.norm((position - centre)[:2])
        )
    assert np.asarray(view.camera.local.position, float) == pytest.approx(position, abs=1e-6)


def test_2d_lines_stay_over_their_surface(reef3d_run):
    """2D from the top: the camera's depth range spans the flat layers alone, so the
    depth buffer tells a surface (the free surface's layer at 3) from its grid lines
    5 cm over it, however the view is zoomed or moved; in 3D its own again."""
    from lagoonviz.core import Session
    from lagoonviz.render.scene3d import SceneRenderer

    session = Session()
    session.open_case(reef3d_run)
    view = SceneRenderer()
    for item_id in session.data.data_items():
        mesh = session.data.load(item_id)
        if mesh is not None:
            view.update(session.scene.get(item_id).kind, session.data.display(item_id, mesh))
    view.set_plan(True)
    view.set_orthographic(True)
    view.frame("top")
    view.camera.width *= 1e4  # zoomed far out: before, the range grew with it
    view.camera.height *= 1e4
    view.fit_plan_depth()
    near, far = view.camera.near, view.camera.far
    height = float(view.camera.world.position[2])
    assert near <= height - 4.5 and far >= height + 0.5  # the marks and the maps inside
    assert far - near <= 50.0  # 24-bit depth: steps of 3 µm, not of metres

    def depth(z):
        matrix = np.asarray(view.camera.projection_matrix) @ np.asarray(
            view.camera.world.inverse_matrix
        )
        point = matrix @ np.array([0.0, 0.0, z, 1.0])
        return point[2] / point[3]

    assert abs(depth(3.05) - depth(3.0)) > 1e-4
    view.set_plan(False)
    view.set_orthographic(False)
    view.fit_plan_depth()
    assert view.camera.depth_range is None


def test_shadows_without_diagonal_bands(reef3d_run):
    """Bug: solids, slices and the volume's sides were shaded in diagonal bands (the
    shadow map's texels on faces the sun grazes). The shadow map covers the scene's
    box as the sun sees it, its bias is in texels of the map, the views into the
    water cast no shadow, and no 2D view has shadows."""
    import pygfx as gfx

    from lagoonviz.core import Session
    from lagoonviz.render.scene3d import (
        NO_SHADOW_CAST,
        SHADOW_BIAS_TEXELS,
        SHADOW_MAP_TEXELS,
        SceneRenderer,
        _seen_from,
    )

    # a long, narrow flume seen from the sun straight above: the map is as long and
    # as narrow as the flume (a sphere around it was 8640 m both ways)
    width, height, depth = _seen_from((0.0, 0.0, 1.0), (0, 0, 0), (8640, 1.5, 600))
    assert sorted([width, height]) == pytest.approx([1.5, 8640])
    assert depth == pytest.approx(600)
    session = Session()
    session.open_case(reef3d_run)
    view = SceneRenderer()
    for item_id in session.data.data_items():
        mesh = session.data.load(item_id)
        if mesh is not None:
            view.update(session.scene.get(item_id).kind, session.data.display(item_id, mesh))
    assert view.sun.cast_shadow
    shadow = view.sun.shadow
    low, high = view.bounds(visible_only=False)
    radius = float(np.linalg.norm(high - low)) / 2
    assert max(shadow.camera.width, shadow.camera.height) < 2.05 * radius
    texel = max(shadow.camera.width, shadow.camera.height) / SHADOW_MAP_TEXELS
    reach = shadow.camera.depth_range[1] - shadow.camera.depth_range[0]
    assert shadow.bias * reach == pytest.approx(SHADOW_BIAS_TEXELS * texel)
    # the camera's frame is the one the extents were reckoned in
    shadow._update_matrix(view.sun)
    forward = np.asarray(shadow.camera.world.forward)
    assert forward == pytest.approx(-np.asarray(view.sun_direction), abs=1e-6)
    view.set_two_d(True)  # 2D from the side
    assert not view.sun.cast_shadow
    view.set_two_d(False)
    assert view.sun.cast_shadow
    for kind in ("slice", "volume", "iso_volume"):
        assert kind in NO_SHADOW_CAST
    surface = next(s for s in view._shown.values() if isinstance(s.mesh, gfx.Mesh))
    assert surface.mesh.cast_shadow  # the free surface and bed still do


def test_a_slices_grid_lines_lie_on_it_as_seen(reef3d_run):
    """A slice's grid lines are drawn a little towards the camera: in front of the
    slice for the depth buffer, on the screen exactly where they belong, from either
    side and however far away (lifted to both sides by a share of the slice's
    length, they floated clear of a long flume's slice: two grids beside it)."""
    from lagoonviz.core import Session
    from lagoonviz.render.scene3d import FACING_SHARE, SceneRenderer

    session = Session()
    session.open_case(reef3d_run)
    view = SceneRenderer()
    slice_id = session.add_slice("y", position=2.5)
    session.set_property(slice_id, "grid_lines", "dark")
    mesh = session.data.load(slice_id)
    shown = session.data.display(slice_id, mesh)
    view.update("slice", shown)
    lines = session.data.patch_outlines(slice_id, mesh, shown)
    assert lines.facing
    view.update_outlines(slice_id, lines)
    drawn = view._contours[f"{slice_id}#outlines"]
    ends = lines.segments.reshape(-1, 3).astype(np.float64)

    def clip(points):
        matrix = np.asarray(view.camera.projection_matrix) @ np.asarray(
            view.camera.world.inverse_matrix
        )
        h = np.c_[points, np.ones(len(points))] @ matrix.T
        return h[:, :3] / h[:, 3:]

    for side in (-1.0, 1.0):
        view.camera.show_object((5.0, 2.5, 1.0, 6.0), view_dir=(0.3, -side, -0.4), up=(0, 0, 1))
        view.face_camera()
        moved = np.asarray(drawn.geometry.positions.data, np.float64)
        eye = np.asarray(view.camera.world.position, float)
        np.testing.assert_allclose(
            np.linalg.norm(moved - ends, axis=1),
            FACING_SHARE * np.linalg.norm(eye - ends, axis=1),
            rtol=1e-3,
        )
        assert np.all(np.sign(moved[:, 1] - 2.5) == np.sign(eye[1] - 2.5))  # the near side
        a, b = clip(ends), clip(moved)
        np.testing.assert_allclose(b[:, :2], a[:, :2], atol=1e-5)  # where they were seen
        assert np.all(b[:, 2] < a[:, 2])  # and in front
