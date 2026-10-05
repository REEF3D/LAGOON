"""Headless rendering with pygfx: the design test for 3D views.

Skipped where pygfx or a GPU adapter is missing. CI installs Mesa's software
Vulkan driver on Linux so these run there too. With ``LAGOON_TEST_OUTPUT`` set,
the rendered images are written to that folder for a visual check.
"""

import os
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("pygfx")

from lagoonviz.core import Session  # noqa: E402
from lagoonviz.render.offscreen import main, render_session, write_png  # noqa: E402


def _has_adapter() -> bool:
    try:
        import wgpu

        request = getattr(wgpu.gpu, "request_adapter_sync", None) or wgpu.gpu.request_adapter
        return request(power_preference="high-performance") is not None
    except Exception:  # noqa: BLE001
        return False


pytestmark = pytest.mark.skipif(not _has_adapter(), reason="no GPU adapter")


def _keep(name: str, image: np.ndarray) -> None:
    folder = os.environ.get("LAGOON_TEST_OUTPUT")
    if folder:
        Path(folder).mkdir(parents=True, exist_ok=True)
        write_png(Path(folder) / name, image)


def _colours(image: np.ndarray) -> int:
    return len(np.unique(image[..., :3].reshape(-1, 3), axis=0))


def test_render_case(reef3d_run):
    session = Session()
    session.open_case(reef3d_run)
    image = render_session(session, (480, 270))
    _keep("render_case.png", image)
    assert image.shape == (270, 480, 4)
    # the surfaces add many colours to the two-colour background gradient
    assert _colours(image) > 200


def test_saved_session_renders_the_same(reef3d_run, tmp_path):
    session = Session()
    session.open_case(reef3d_run)
    session.set_property("free_surface-1", "exaggeration", 5.0)
    first = render_session(session, (320, 180), time=1)
    path = tmp_path / "view.lagoon-session"
    session.save(path)
    again = render_session(Session.open(path), (320, 180))
    _keep("render_session.png", again)
    assert np.array_equal(first, again) or np.mean(np.abs(first.astype(int) - again)) < 1.0


def test_hidden_items_are_not_drawn(reef3d_run):
    session = Session()
    session.open_case(reef3d_run)
    shown = render_session(session, (320, 180), view="top")
    for item in ("free_surface-1", "bathymetry-1"):
        session.set_visible(item, False)
    hidden = render_session(session, (320, 180), view="top")
    assert _colours(hidden) < _colours(shown)


def test_scene_renderer_updates_in_place(reef3d_run):
    from lagoonviz.render.scene3d import SceneRenderer

    session = Session()
    session.open_case(reef3d_run)
    view = SceneRenderer()
    first = session.data.display("free_surface-1", session.data.load("free_surface-1"))
    view.update("free_surface", first)
    mesh = view._shown["free_surface-1"].mesh
    session.set_time(2)
    later = session.data.display("free_surface-1", session.data.load("free_surface-1"))
    view.update("free_surface", later)
    assert view._shown["free_surface-1"].mesh is mesh  # buffers updated, not rebuilt
    np.testing.assert_allclose(mesh.geometry.positions.data, later.positions)


def test_command_line(reef3d_run, tmp_path, capsys):
    out = tmp_path / "frame.png"
    assert main([str(reef3d_run), str(out), "--size", "200x120", "--time", "2"]) == 0
    assert out.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
    assert "Free surface" in capsys.readouterr().out


def test_render_slice(reef3d_run):
    session = Session()
    session.open_case(reef3d_run)
    for item in ("free_surface-1", "bathymetry-1"):
        session.set_visible(item, False)
    session.add_slice("y", variable="pressure")
    image = render_session(session, (320, 180), view="front")
    _keep("render_slice.png", image)
    assert _colours(image) > 50


def test_render_body_and_mooring(tmp_path):
    from reef3d_fixtures import make_run

    run = make_run(tmp_path / "moored", frames=3, volume=False, objects=True)
    session = Session()
    session.open_case(run)
    session.set_visible("free_surface-1", False)
    image = render_session(session, (320, 180), time=1, view="front")
    _keep("render_body_mooring.png", image)
    assert _colours(image) > 50


def test_render_particles(tmp_path):
    from reef3d_fixtures import make_run, write_cpm

    from lagoonviz.render.scene3d import SceneRenderer

    run = make_run(tmp_path / "cpm", solver="CFD", frames=4, print_interval=0.5)
    write_cpm(run, end=1.5, interval=0.25)
    session = Session()
    session.open_case(run)
    session.set_visible("free_surface-1", False)
    item = next(i for i in session.scene if i.kind == "particles")
    session.set_property(item.id, "point_size", 10.0)
    image = render_session(session, (320, 180), time=1, view="front")
    _keep("render_particles.png", image)
    assert _colours(image) > 20
    # a later output has more particles: the points are made again
    view = SceneRenderer()
    for time in (0.25, 1.5):
        session.set_time(time)
        view.update("particles", session.data.display(item.id, session.data.load(item.id)))
    import pygfx as gfx

    points = view._shown[item.id].mesh
    assert isinstance(points, gfx.Points) and points.material.size == 10.0
    assert len(points.geometry.positions.data) == len(session.data.load(item.id).points)


def test_marker(reef3d_run):
    from lagoonviz.render.scene3d import SceneRenderer

    view = SceneRenderer()
    view.set_marker((1.0, 2.0, 3.0))
    view.set_marker((2.0, 2.0, 3.0))
    np.testing.assert_allclose(view._marker.geometry.positions.data[0], (2, 2, 3))
    view.set_marker(None)
    assert not view._marker.visible


def test_styles(reef3d_run):
    from lagoonviz.render.scene3d import LIGHTING, SceneRenderer

    session = Session()
    session.open_case(reef3d_run)
    studio = render_session(session, (160, 90), style="studio")
    ocean = render_session(session, (160, 90), style="ocean")
    assert studio[0, 0, :3].mean() > 200 > ocean[0, 0, :3].mean()  # white vs dark blue sky
    view = SceneRenderer("ocean")
    view.set_style("studio")
    assert view.style_name == "studio"
    # both looks share the lights; only the background differs
    assert abs(view.ambient.intensity - LIGHTING["ambient"]) < 1e-6
    assert abs(view.headlamp.intensity - LIGHTING["headlamp"]) < 1e-6


def test_gauges_in_plot_colours(tmp_path):
    from reef3d_fixtures import make_run

    from lagoonviz.render.scene3d import SceneRenderer

    run = make_run(tmp_path / "gauged", frames=5, volume=False, print_interval=0.5, series=True)
    session = Session()
    session.open_case(run)
    image = render_session(session, (480, 270), time=1.0, view="perspective")
    _keep("render_gauges.png", image)
    green = np.array([0x3A, 0x9A, 0x5B])  # third plot colour: gauge P3
    near = np.abs(image[..., :3].astype(int) - green).sum(axis=-1) < 40
    assert near.sum() > 5

    view = SceneRenderer()
    marks = session.data.gauge_marks(session.data.series_items()[0])
    view.update_gauges(marks)
    view.update_gauges(marks)  # replaced, not added twice
    assert view.gauge_ids == [marks.item_id]
    view.clear()
    assert view.gauge_ids == []


def test_colour_map_ends_do_not_wrap():
    from lagoonviz.core.colormaps import lut
    from lagoonviz.render.scene3d import _texel_coords

    n = len(lut("reef3d"))
    coords = _texel_coords(np.array([0.0, 1.0, np.nan], np.float32), "reef3d")
    np.testing.assert_allclose(coords, [0.5 / n, 1 - 0.5 / n, 0.5 / n], rtol=1e-6)


def test_sun_casts_shadows_over_the_whole_scene(reef3d_run):
    from lagoonviz.render.scene3d import SceneRenderer

    session = Session()
    session.open_case(reef3d_run)
    view = SceneRenderer()
    for item_id in session.data.data_items():
        mesh = session.data.load(item_id)
        view.update(session.scene.get(item_id).kind, session.data.display(item_id, mesh))
    assert view.sun.cast_shadow
    assert all(shown.mesh.receive_shadow for shown in view._shown.values())
    camera = view.sun.shadow.camera
    low, high = (np.asarray(b) for b in view.content.get_world_bounding_box())
    assert camera.width >= np.linalg.norm(high - low) * 0.99
    image = render_session(session, (320, 180), style="studio")
    _keep("render_shadows.png", image)


def test_renderer_never_writes_into_cached_frames(reef3d_run):
    """pygfx buffers keep the array they get: they must not be the cached frame's."""
    from lagoonviz.render.scene3d import SceneRenderer

    session = Session()
    session.open_case(reef3d_run)
    view = SceneRenderer()
    item_id = "free_surface-1"
    first = session.data.load(item_id)
    original = first.points.copy()
    for exaggeration in (1.0, 2.0, 3.0, 2.0, 1.0):
        session.set_property(item_id, "exaggeration", exaggeration)
        view.update("free_surface", session.data.display(item_id, first))
    np.testing.assert_array_equal(first.points, original)
    z = view._shown[item_id].mesh.geometry.positions.data[:, 2]
    np.testing.assert_allclose(z, original[:, 2], rtol=1e-6)  # back at 1: as read
    session.set_time(2)
    view.update("free_surface", session.data.display(item_id, session.data.load(item_id)))
    np.testing.assert_array_equal(first.points, original)  # playing leaves frame 0 alone


def test_movie_mp4_and_png_frames(reef3d_run, tmp_path):
    from lagoonviz.render.movie import frame_times, movie_available, write_movie

    session = Session()
    session.open_case(reef3d_run)
    times = frame_times(session)
    assert times == [0.0, 1.0, 2.0]
    assert frame_times(session, start=1, every=2) == [1.0]
    seen = []

    def progress(done, total):
        seen.append((done, total))
        return True

    folder = write_movie(session, tmp_path / "frames", times, (160, 90), progress=progress)
    assert sorted(p.name for p in folder.iterdir()) == [
        "frame_00000.png",
        "frame_00001.png",
        "frame_00002.png",
    ]
    assert seen == [(1, 3), (2, 3), (3, 3)]
    stopped = write_movie(
        session, tmp_path / "stopped", times, (160, 90), progress=lambda d, t: False
    )
    assert len(list(stopped.iterdir())) == 1  # stopped after the first frame
    if movie_available():
        movie = write_movie(session, tmp_path / "waves.mp4", times, (161, 91), fps=10)
        assert movie.suffix == ".mp4" and movie.stat().st_size > 1000


def test_command_line_movie(reef3d_run, tmp_path, capsys):
    from lagoonviz.render.movie import movie_available

    if not movie_available():
        pytest.skip("imageio-ffmpeg not installed")
    out = tmp_path / "cli.mp4"
    assert main([str(reef3d_run), str(out), "--size", "160x90", "--fps", "5"]) == 0
    assert out.stat().st_size > 1000 and "3 frames" in capsys.readouterr().out


def test_pan_moves_the_view_sideways(reef3d_run):
    from lagoonviz.render.scene3d import SceneRenderer

    session = Session()
    session.open_case(reef3d_run)
    view = SceneRenderer()
    for item_id in session.data.data_items():
        mesh = session.data.load(item_id)
        view.update(session.scene.get(item_id).kind, session.data.display(item_id, mesh))
    view.frame("top")  # looking down: right is +x, up is +y
    before = np.asarray(view.camera.world.position).copy()
    width = view.camera.get_state()["width"]
    view.pan(0.1, 0.0)
    moved = np.asarray(view.camera.world.position) - before
    np.testing.assert_allclose(moved, (0.1 * width, 0, 0), atol=1e-6 * width)
    # the top view's up (y) does not become the axis the view orbits about
    np.testing.assert_allclose(view.camera.world.reference_up, (0, 0, 1))


def _plane_and_box():
    from lagoonviz.core.data.manager import DisplaySurface

    def surface(item_id, positions, triangles, color):
        positions = np.asarray(positions, np.float32)
        normals = np.zeros_like(positions)
        normals[:, 2] = 1.0
        return DisplaySurface(
            item_id=item_id,
            positions=positions,
            triangles=np.asarray(triangles, np.uint32),
            normals=normals,
            values=None,
            vmin=0.0,
            vmax=1.0,
            colormap="greys",
            opacity=1.0,
            visible=True,
            variable="none",
            frame=0,
            color=color,
        )

    plane = surface(
        "plane",
        [(0, 0, 0), (100, 0, 0), (100, 100, 0), (0, 100, 0)],
        [(0, 1, 2), (0, 2, 3)],
        "#d0d0d0",
    )
    # a flat roof 20 m above the plane: it must throw a shadow
    roof = surface(
        "roof",
        [(40, 40, 20), (60, 40, 20), (60, 60, 20), (40, 60, 20)],
        [(0, 1, 2), (0, 2, 3)],
        "#d0d0d0",
    )
    return plane, roof


def test_sun_shadow_darkens_the_ground():
    import pygfx as gfx

    from lagoonviz.render.movie import _offscreen_canvas
    from lagoonviz.render.scene3d import SceneRenderer

    images = {}
    for shadows in (True, False):
        canvas = _offscreen_canvas((240, 240))
        renderer = gfx.WgpuRenderer(canvas)
        view = SceneRenderer()
        for shown in _plane_and_box():
            view.update("bathymetry", shown)
        view.sun.cast_shadow = shadows
        view.frame("top")
        canvas.request_draw(lambda v=view, r=renderer: v.render(r))
        images[shadows] = np.asarray(canvas.draw())[..., :3].astype(int)
    _keep("render_shadow_on.png", images[True].astype(np.uint8))
    _keep("render_shadow_off.png", images[False].astype(np.uint8))
    darker = (images[False] - images[True]).sum(axis=-1) > 30
    assert darker.sum() > 200, f"no visible shadow ({darker.sum()} darker pixels)"


def test_render_wave_profiles(tmp_path):
    from reef3d_fixtures import make_run

    run = make_run(tmp_path / "lines", frames=4, volume=False, print_interval=0.5, profiles=True)
    session = Session()
    session.open_case(run)
    image = render_session(session, (320, 180), time=1.0, view="perspective")
    _keep("render_profiles.png", image)
    blue = np.array([0x2B, 0x4C, 0x7E])  # first plot colour: line 1
    assert (np.abs(image[..., :3].astype(int) - blue).sum(axis=-1) < 40).sum() > 5


def test_render_amr_surface(tmp_path):
    from reef3d_fixtures import make_amr

    from lagoonviz.render.scene3d import SceneRenderer

    make_amr(tmp_path, "NHFLOW", frames=3)
    session = Session()
    session.open_case(tmp_path)
    item = next(i for i in session.scene if i.kind == "amr")
    session.set_property(item.id, "variable", "level")
    session.set_property(item.id, "colormap", "rainbow")
    image = render_session(session, (320, 180), time=1, view="top")
    _keep("render_amr.png", image)
    assert _colours(image) > 20
    # the patch borders are drawn over the surface, and go with the item
    view = SceneRenderer()
    mesh = session.data.load(item.id)
    shown = session.data.display(item.id, mesh)
    view.update("amr", shown)
    view.update_outlines(item.id, session.data.patch_outlines(item.id, mesh, shown))
    assert f"{item.id}#outlines" in view._contours
    view.remove(item.id)
    assert f"{item.id}#outlines" not in view._contours


@pytest.mark.parametrize("look", ["open_ocean", "wave_flume", "underwater"])
def test_cinematic_looks(tmp_path, look):
    """Each cinematic look renders: water, sand and a steel body in their surroundings."""
    from reef3d_fixtures import make_run, write_objects

    run = make_run(tmp_path / "run", frames=3, print_interval=0.5, objects=True)
    write_objects(run, "NHFLOW", 3)
    session = Session()
    session.open_case(run)
    for item in session.scene:
        if item.kind == "free_surface":
            session.set_property(item.id, "variable", "none")  # the water's own colour
    session.set_property("case-1", "exaggeration", 3.0)
    image = render_session(session, (640, 360), time=1.0, view="perspective", style=look)
    _keep(f"render_{look}.png", image)
    assert _colours(image) > 200  # sky, reflections, grain: many shades
    # level with the scene: the horizon, the background, the fog across the domain
    front = render_session(session, (640, 360), time=1.0, view="front", style=look)
    _keep(f"render_{look}_front.png", front)


def test_switching_looks_swaps_materials(reef3d_run):
    import pygfx as gfx

    from lagoonviz.render.scene3d import SceneRenderer

    session = Session()
    session.open_case(reef3d_run)
    view = SceneRenderer()
    for item in session.scene:
        mesh = session.data.load(item.id)
        if mesh is not None and item.kind in ("free_surface", "bathymetry"):
            view.update(item.kind, session.data.display(item.id, mesh))
    bed = next(s for s in view._shown.values() if s.kind == "bathymetry")
    water = next(s for s in view._shown.values() if s.kind == "free_surface")
    assert isinstance(water.mesh.material, gfx.MeshPhongMaterial)
    view.set_style("open_ocean")
    assert isinstance(water.mesh.material, gfx.MeshPhysicalMaterial)
    assert water.mesh.material.env_map is not None
    assert bed.mesh.material.normal_map is not None  # sand grain
    assert bed.mesh.geometry.texcoords1.data.shape == (bed.mesh.geometry.positions.data.shape[0], 2)
    assert isinstance(view.background.material, gfx.BackgroundSkyboxMaterial)

    class Renderer:  # what the looks need of a renderer
        effect_passes = ()

    renderer = Renderer()
    view.set_style("underwater")
    view.apply_effects(renderer)
    assert len(renderer.effect_passes) == 1  # the depth fog
    view.set_style("studio")
    view.apply_effects(renderer)
    assert renderer.effect_passes == ()
    assert isinstance(water.mesh.material, gfx.MeshPhongMaterial)
    assert not isinstance(view.background.material, gfx.BackgroundSkyboxMaterial)


def test_film_follows_the_camera_path(reef3d_run, tmp_path):
    from lagoonviz.render.movie import OffscreenView

    session = Session()
    session.open_case(reef3d_run)
    view = OffscreenView(session, (160, 90))
    first = view.render(0.0)
    start = view.scene.camera_state()
    end = dict(start)
    end["position"] = (np.asarray(start["position"]) + [0.0, 0.0, 2.0]).tolist()
    session.add_camera_key(0.0, start)
    session.add_camera_key(2.0, end)
    view.render(1.0)
    middle = np.asarray(view.scene.camera_state()["position"])
    # two keys, eased at both: halfway in time is halfway along
    np.testing.assert_allclose(
        middle, (np.asarray(start["position"]) + end["position"]) / 2, atol=1e-5
    )
    last = view.render(2.0)
    np.testing.assert_allclose(view.scene.camera_state()["position"], end["position"], atol=1e-5)
    assert np.abs(last.astype(int) - first.astype(int)).sum() > 0
    session.set_camera_path(False)  # off: the camera stays where it is
    view.render(0.0)
    np.testing.assert_allclose(view.scene.camera_state()["position"], end["position"], atol=1e-5)
    _keep("camera-path-end.png", last)
