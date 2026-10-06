"""Path tracing with Blender: the export (NumPy only) and the Blender script.

Blender itself is only needed by the last test (skipped without it). The others
run the Blender script against a stand-in ``bpy`` that writes a small PNG for each
render, so the whole way from a session to a film is tested anywhere.
"""

import json
import math
import os
import stat
import sys
from pathlib import Path

import numpy as np
import pytest

from lagoonviz.core import Session
from lagoonviz.render.cinematic import OPEN_OCEAN, UNDERWATER, WAVE_FLUME, panorama_directions
from lagoonviz.render.pathtrace import (
    SEA,
    apron,
    blender_camera,
    default_camera,
    environment_panorama,
    export_scene,
    find_blender,
    look_rotation,
    loop_mean,
    ocean_ring,
    outward_rings,
    path_trace,
    pixel_size,
    read_hdr,
    render_folder,
    rim_loop,
    ring_distances,
    ring_triangles,
    rotate,
    skirt,
    spheres,
    tubes,
    work_folder,
    write_hdr,
)


def test_panorama_directions_as_blender_maps_them():
    d = panorama_directions(64)
    assert d.shape == (32, 64, 3)
    np.testing.assert_allclose(np.linalg.norm(d, axis=-1), 1.0, atol=1e-12)
    assert d[0, :, 2].min() > 0.99  # the top row looks up
    assert d[-1, :, 2].max() < -0.99
    middle = d[16, 32]  # just right of and below the centre: about +x
    assert middle[0] > 0.99
    assert d[16, 16, 1] > 0.99  # a quarter in from the left: +y (u = 0.5 - atan2 / 2 pi)


def test_panorama_lights_are_brighter_than_white():
    sky = environment_panorama(OPEN_OCEAN, 256)
    assert sky.shape == (128, 256, 3) and sky.dtype == np.float32
    # the sun's glow is where the sun is, and brighter than anything a PNG holds
    d = panorama_directions(256)
    brightest = np.unravel_index(np.argmax(sky.sum(axis=-1)), sky.shape[:2])
    sun = np.asarray(OPEN_OCEAN.sun_direction) / np.linalg.norm(OPEN_OCEAN.sun_direction)
    assert d[brightest] @ sun > 0.995
    assert sky.max() > 1.5
    hall = environment_panorama(WAVE_FLUME, 256)
    assert hall.max() > 3.0  # the ceiling panels light the flume
    assert hall[-1].mean() < hall[0].mean()  # the floor darker than the ceiling


def test_hdr_round_trip(tmp_path):
    rng = np.random.default_rng(1)
    rgb = rng.uniform(0, 1, (5, 300, 3)) ** 3 * np.array([1, 10, 100])
    rgb[0, 0] = 0.0
    write_hdr(tmp_path / "x.hdr", rgb)
    back = read_hdr(tmp_path / "x.hdr")
    assert back.shape == rgb.shape
    assert (tmp_path / "x.hdr").read_bytes().startswith(b"#?RADIANCE\n")
    big = rgb.max(axis=-1, keepdims=True)  # RGBE: 8 bits relative to the brightest
    assert (np.abs(back - rgb) < big / 128 + 1e-30).all()


def test_tubes_and_spheres():
    seg = np.array([[[0, 0, 0], [0, 0, 2]], [[1, 0, 0], [3, 0, 0]]], float)
    positions, triangles = tubes(seg, 0.1, sides=6)
    assert positions.shape == (24, 3) and triangles.shape == (24, 3)
    assert triangles.max() == 23
    first = positions[:12]  # around the z axis, radius 0.1
    np.testing.assert_allclose(np.hypot(first[:, 0], first[:, 1]), 0.1, rtol=1e-6)
    np.testing.assert_allclose(sorted(set(first[:, 2].round(6))), [0, 2])
    second = positions[12:]
    np.testing.assert_allclose(np.hypot(second[:, 1], second[:, 2]), 0.1, rtol=1e-6)
    p, t = spheres(np.array([[0, 0, 0], [5, 5, 5]]), [1.0, 0.5])
    assert p.shape == (24, 3) and t.shape == (40, 3)
    np.testing.assert_allclose(np.linalg.norm(p[:12], axis=1), 1.0, rtol=1e-6)
    np.testing.assert_allclose(np.linalg.norm(p[12:] - 5, axis=1), 0.5, rtol=1e-6)
    assert tubes(np.zeros((0, 2, 3)), 1.0)[0].shape == (0, 3)


def test_look_rotation_points_the_camera():
    forward = np.array([-1.0, 1.2, -0.7])
    q = look_rotation(forward)
    np.testing.assert_allclose(rotate(q, [0, 0, -1]), forward / np.linalg.norm(forward), atol=1e-9)
    up = rotate(q, [0, 1, 0])
    assert up[2] > 0 and abs(up @ forward) < 1e-9  # upright
    down = look_rotation([0, 0, -1])  # straight down still works
    np.testing.assert_allclose(rotate(down, [0, 0, -1]), [0, 0, -1], atol=1e-9)


def test_blender_camera_matches_pygfx():
    state = {
        "position": [1.0, 2.0, 3.0],
        "rotation": [0.1, 0.2, 0.3, 0.927],
        "fov": 50.0,
        "zoom": 1.0,
        "width": 16.0,
        "height": 9.0,
    }
    cam = blender_camera(state, (1600, 900))
    assert cam["type"] == "PERSP"
    assert cam["rotation_wxyz"] == [0.927, 0.1, 0.2, 0.3]
    # same aspect: pygfx puts 2 tan(fov / 2) into the mean of width and height
    mean = 2 * math.tan(math.radians(25))
    vertical = 2 * mean / (1 + 16 / 9)
    assert cam["angle_y"] == pytest.approx(2 * math.atan(vertical / 2))
    # a taller image keeps the width: more is seen vertically
    tall = blender_camera(state, (900, 900))
    assert tall["angle_y"] == pytest.approx(2 * math.atan(vertical * (16 / 9) / 2))
    # a wider image keeps the height
    wide = blender_camera(state, (3200, 900))
    assert wide["angle_y"] == pytest.approx(cam["angle_y"])
    zoomed = blender_camera({**state, "zoom": 2.0}, (1600, 900))
    assert math.tan(zoomed["angle_y"] / 2) == pytest.approx(math.tan(cam["angle_y"] / 2) / 2)
    ortho = blender_camera({**state, "fov": 0.0, "zoom": 2.0}, (1600, 900))
    assert ortho["type"] == "ORTHO" and ortho["ortho_scale"] == pytest.approx(4.5)
    assert blender_camera({**state, "fov": 0.0}, (900, 900))["ortho_scale"] == pytest.approx(16)
    assert pixel_size(ortho, [0, 0, 0], 900) == pytest.approx(4.5 / 900)
    far = pixel_size(cam, np.array(state["position"]) + [0, 0, 10], 900)
    assert far == pytest.approx(2 * 10 * math.tan(cam["angle_y"] / 2) / 900)


def test_default_camera_sees_the_box():
    low, high = np.array([0.0, 0.0, -1.0]), np.array([10.0, 4.0, 1.0])
    state = default_camera(low, high)
    cam = blender_camera(state, (1600, 900))
    centre = (low + high) / 2
    q = state["rotation"]
    to_centre = centre - state["position"]
    np.testing.assert_allclose(
        rotate(q, [0, 0, -1]), to_centre / np.linalg.norm(to_centre), atol=1e-9
    )
    # every corner within the view (camera space, perspective divide)
    inverse = np.array(q) * [-1, -1, -1, 1]
    for corner in np.array(np.meshgrid(*zip(low, high, strict=True))).T.reshape(-1, 3):
        x, y, z = rotate(inverse, corner - state["position"])
        assert z < 0
        assert abs(y / -z) <= math.tan(cam["angle_y"] / 2)
        assert abs(x / -z) <= math.tan(cam["angle_y"] / 2) * 16 / 9


def test_export_scene(reef3d_run, tmp_path):
    session = Session()
    session.open_case(reef3d_run)
    path = export_scene(session, tmp_path / "out", [0.0, 2.0], (320, 180), "open_ocean", 16)
    scene = json.loads(path.read_text())
    assert scene["size"] == [320, 180] and scene["samples"] == 16
    assert scene["look"] == "open_ocean" and scene["world"]["fog"] is None
    assert (tmp_path / "out" / "environment.hdr").exists()
    assert [f["time"] for f in scene["frames"]] == [0.0, 2.0]
    assert session.scene.time == 0.0  # the session's time is put back
    frame = scene["frames"][0]
    names = [o["name"] for o in frame["objects"]]
    kinds = {session.scene.get(n).kind for n in names if "#" not in n}
    assert {"free_surface", "bathymetry"} <= kinds
    with np.load(tmp_path / "out" / frame["data"]) as arrays:
        for index, entry in enumerate(frame["objects"]):
            positions = arrays[f"positions_{index}"]
            triangles = arrays[f"triangles_{index}"]
            assert np.isfinite(positions).all()
            assert triangles.max() < len(positions)
            if entry["colors"]:
                assert arrays[f"colors_{index}"].shape == positions.shape
    materials = scene["materials"]
    items = [o for o in frame["objects"] if "#" not in o["name"] and o["name"] != "LAGOON sea"]
    water = next(
        materials[o["material"]]
        for o in items
        if session.scene.get(o["name"]).kind == "free_surface"
    )
    bed = next(
        materials[o["material"]] for o in items if session.scene.get(o["name"]).kind == "bathymetry"
    )
    if water["color"] is not None:  # clear water unless coloured by a variable
        assert water["transmission"] > 0.2 and water["ior"] == pytest.approx(1.33)
    assert bed["grain"] is None or bed["grain"]["kind"] == "sand"
    cam = frame["camera"]
    assert cam["type"] == "PERSP" and 0 < cam["clip_start"] < cam["clip_end"]
    sun = scene["sun"]
    np.testing.assert_allclose(
        rotate([*sun["rotation_wxyz"][1:], sun["rotation_wxyz"][0]], [0, 0, 1]),
        sun["direction"],
        atol=1e-9,
    )


def test_export_follows_the_camera_path(reef3d_run, tmp_path):
    session = Session()
    session.open_case(reef3d_run)
    start = default_camera([0, 0, -1], [10, 4, 1])
    end = {**start, "position": (np.asarray(start["position"]) + [0, 0, 4]).tolist()}
    session.add_camera_key(0.0, start)
    session.add_camera_key(2.0, end)
    path = export_scene(session, tmp_path / "out", [0.0, 1.0, 2.0], (64, 36), "underwater")
    scene = json.loads(path.read_text())
    z = [f["camera"]["position"][2] for f in scene["frames"]]
    assert z == pytest.approx([start["position"][2], start["position"][2] + 2, end["position"][2]])
    fog = scene["world"]["fog"]
    assert fog is not None and fog["density"] > 0  # under water: fog around the scene
    assert scene["world"]["background"] is not None
    assert scene["look"] == UNDERWATER.name


FAKE_BLENDER = """\
#!{python}
# A stand-in for Blender: runs the -P script with a mocked bpy whose render writes
# a small PNG of noise (each render its own) to the scene's file path.
import runpy, struct, sys, zlib
from unittest import mock

import numpy as np

bpy = mock.MagicMock(name="bpy")
rendered = []

def render(write_still=False):
    import os, time
    time.sleep(float(os.environ.get("FAKE_BLENDER_SECONDS", "0")))
    scene = bpy.context.scene
    path = str(scene.render.filepath)
    w, h = int(scene.render.resolution_x), int(scene.render.resolution_y)
    image = np.random.default_rng().integers(0, 256, (h, w, 3), dtype=np.uint8)  # noise
    rows = np.concatenate([np.zeros((h, 1), np.uint8), image.reshape(h, -1)], 1)
    def chunk(tag, data):
        return struct.pack(">I", len(data)) + tag + data + struct.pack(
            ">I", zlib.crc32(tag + data))
    with open(path, "wb") as f:
        header = struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0)
        f.write(b"\\x89PNG\\r\\n\\x1a\\n" + chunk(b"IHDR", header)
                + chunk(b"IDAT", zlib.compress(rows.tobytes())) + chunk(b"IEND", b""))
    rendered.append(path)

bpy.ops.render.render.side_effect = render
sys.modules["bpy"] = bpy
script = sys.argv[sys.argv.index("-P") + 1]
sys.argv = [sys.argv[0]] + sys.argv[1:]
runpy.run_path(script, run_name="__main__")
"""


@pytest.fixture
def fake_blender(tmp_path):
    path = tmp_path / "fake-blender"
    path.write_text(FAKE_BLENDER.format(python=sys.executable))
    path.chmod(path.stat().st_mode | stat.S_IXUSR)
    return str(path)


def test_blender_script_renders_each_frame(reef3d_run, tmp_path, fake_blender):
    session = Session()
    session.open_case(reef3d_run)
    view = default_camera([0, 0, -1], [10, 4, 1])
    session.set_camera(view)
    folder = tmp_path / "film.cycles"
    export_scene(session, folder, [0.0, 1.0, 2.0], (64, 36), "wave_flume", 4)
    seen = []
    images = render_folder(folder, fake_blender, lambda d, t: seen.append((d, t)) or True)
    assert [p.name for p in images] == ["frame_00000.png", "frame_00001.png", "frame_00002.png"]
    assert seen == [(1, 3), (2, 3), (3, 3)]
    keys = sorted(p.name for p in folder.glob("frame_*.key"))
    assert keys == ["frame_00000.key", "frame_00001.key", "frame_00002.key"]
    # again with --resume: nothing is rendered twice
    before = [p.read_bytes() for p in images]
    render_folder(folder, fake_blender, resume=True)
    assert [p.read_bytes() for p in images] == before
    # exported again, the camera moved in the middle frame only: only that one is new
    moved = {**view, "position": (np.asarray(view["position"]) + [0, 0, 3]).tolist()}
    for time, state in ((0.0, view), (1.0, moved), (2.0, view)):
        session.add_camera_key(time, state)
    export_scene(session, folder, [0.0, 1.0, 2.0], (64, 36), "wave_flume", 4)
    render_folder(folder, fake_blender, resume=True)
    after = [p.read_bytes() for p in images]
    assert after[0] == before[0] and after[2] == before[2] and after[1] != before[1]
    render_folder(folder, fake_blender)  # without resume: all again
    assert all(a != b for a, b in zip(after, (p.read_bytes() for p in images), strict=True))


def test_stopping_blender(reef3d_run, tmp_path, fake_blender, monkeypatch):
    session = Session()
    session.open_case(reef3d_run)
    folder = tmp_path / "stopped.cycles"
    export_scene(session, folder, [0.0, 1.0, 2.0], (64, 36), "open_ocean", 4)
    monkeypatch.setenv("FAKE_BLENDER_SECONDS", "3")  # each frame takes 3 s
    import time

    started = time.monotonic()
    images = render_folder(folder, fake_blender, cancel=lambda: time.monotonic() > started + 1)
    assert time.monotonic() - started < 6 and len(images) <= 1


def test_blender_failure_is_reported(tmp_path):
    folder = tmp_path / "empty"
    folder.mkdir()
    broken = tmp_path / "broken-blender"
    broken.write_text(f"#!{sys.executable}\nimport sys\nprint('Error: no GPU')\nsys.exit(1)\n")
    broken.chmod(broken.stat().st_mode | stat.S_IXUSR)
    with pytest.raises(RuntimeError, match="no GPU"):
        render_folder(folder, str(broken))


def test_path_trace_still_and_film(reef3d_run, tmp_path, fake_blender):
    from lagoonviz.render.movie import movie_available

    session = Session()
    session.open_case(reef3d_run)
    still = path_trace(session, tmp_path / "still.png", size=(64, 36), blender=fake_blender)
    assert still.read_bytes().startswith(b"\x89PNG")
    out = path_trace(
        session, tmp_path / "film.mp4", size=(64, 36), samples=4, fps=5, blender=fake_blender
    )
    if movie_available():
        assert out == tmp_path / "film.mp4" and out.stat().st_size > 500
    assert len(list((tmp_path / "film.cycles").glob("frame_*.png"))) == 3
    assert work_folder(tmp_path / "a.png") == tmp_path / "a.cycles"
    assert work_folder(tmp_path / "frames") == tmp_path / "frames"


def test_command_line_cycles(reef3d_run, tmp_path, fake_blender, capsys):
    from lagoonviz.render.offscreen import main

    out = tmp_path / "cli.png"
    code = main(
        [str(reef3d_run), str(out), "--engine", "cycles", "--blender", fake_blender]
        + ["--size", "64x36", "--samples", "4", "--style", "underwater"]
    )
    assert code == 0 and out.exists()
    assert "Cycles" in capsys.readouterr().out


@pytest.mark.skipif(find_blender() is None, reason="Blender not installed")
def test_real_blender(reef3d_run, tmp_path, monkeypatch):
    """Blender itself: one small frame per look. The first run compiles Cycles'
    GPU kernels (some minutes); later ones take seconds. With LAGOON_TEST_OUTPUT
    set, the images are kept there; while one renders, Blender's output is in
    <image>.cycles/blender.log beside it (kept if it fails)."""
    import time

    session = Session()
    session.open_case(reef3d_run)
    folder = Path(os.environ.get("LAGOON_TEST_OUTPUT") or tmp_path)
    for look in ("open_ocean", "wave_flume", "underwater", "diorama"):
        deadline = time.monotonic() + 900
        target = folder / f"cycles_{look}.png"
        out = path_trace(
            session,
            target,
            size=(480, 270),
            look=look,
            samples=24,
            cancel=lambda deadline=deadline: time.monotonic() > deadline,
        )
        data = out.read_bytes()
        assert data.startswith(b"\x89PNG") and len(data) > 5000, look
    # clear water (coloured by nothing): the water must show, not only the bed
    surface = next(i.id for i in session.scene if i.kind == "free_surface")
    session.set_property(surface, "variable", "none")
    for look in ("open_ocean", "underwater", "diorama"):
        out = path_trace(
            session, folder / f"cycles_clear_{look}.png", size=(480, 270), look=look, samples=24
        )
        assert out.read_bytes().startswith(b"\x89PNG"), look
    # white water: point clouds by geometry nodes (here a band of foam and spray)
    from lagoonviz.render import pathtrace

    def particles(self, time):
        rng = np.random.default_rng(0)
        foam = np.c_[rng.uniform(2, 6, 4000), rng.uniform(0, 4, 4000), np.full(4000, 2.08)]
        spray = foam[:600] + [0.0, 0.0, 0.3]
        return {
            "foam": (foam.astype(np.float32), np.full(4000, 0.03, np.float32)),
            "spray": (spray.astype(np.float32), np.full(600, 0.015, np.float32)),
        }

    monkeypatch.setattr(pathtrace.WhiteWater, "at", particles)
    out = path_trace(
        session, folder / "cycles_foam_diorama.png", size=(480, 270), look="diorama", samples=24
    )
    assert out.read_bytes().startswith(b"\x89PNG")


def test_blender_progress_lines():
    from lagoonviz.render.pathtrace import blender_progress

    line = (
        "Fra:1 Mem:120.5M (Peak 130.1M) | Time:00:03.21 | Remaining:00:12.40 | "
        "Mem:80M, Peak:90M | Scene, ViewLayer | Sample 40/128"
    )
    assert blender_progress(line) == "sample 40/128, 00:12.40 left"
    assert blender_progress("LAGOON rendering frame 1/3 (200 triangles)").startswith("rendering")
    assert "compiling" in blender_progress("Compiling render kernels")
    assert blender_progress("Blender 4.2.3 LTS") is None


def _grid(nx=6, ny=4, nan=None):
    x, y = np.meshgrid(np.linspace(0, 5, nx), np.linspace(0, 3, ny))
    z = 1.0 + 0.1 * np.sin(x)
    points = np.stack([x.ravel(), y.ravel(), z.ravel()], 1)
    if nan is not None:
        points[nan] = np.nan  # dry land: a hole in the surface
    quads = [(j * nx + i, j * nx + i + 1, (j + 1) * nx + i + 1, (j + 1) * nx + i)
             for j in range(ny - 1) for i in range(nx - 1)]  # fmt: skip
    triangles = np.array([t for a, b, c, d in quads for t in ((a, b, c), (a, c, d))])
    triangles[::3] = triangles[::3, ::-1]  # some turned over, as meshes come
    return points, triangles


def _closed_and_outward(positions, triangles):
    # merge corners at the same place; then every edge has exactly two triangles,
    # met in opposite directions, and the volume enclosed is positive
    keys = {tuple(np.round(p, 9)) for p in positions}
    index = {k: i for i, k in enumerate(sorted(keys))}
    merged = np.array([index[tuple(np.round(p, 9))] for p in positions])[triangles]
    directed = {}
    for a, b, c in merged:
        for e in ((a, b), (b, c), (c, a)):
            if e[0] != e[1]:
                directed[e] = directed.get(e, 0) + 1
    assert all(n == 1 for n in directed.values())  # no edge twice the same way
    assert all((b, a) in directed for a, b in directed)  # each one met back
    p = positions.astype(float)
    t = triangles
    return float(np.einsum("ij,ij->i", p[t[:, 0]], np.cross(p[t[:, 1]], p[t[:, 2]])).sum() / 6)


def test_water_body_is_closed_and_faces_out():
    from lagoonviz.render.pathtrace import water_body

    points, triangles = _grid()
    positions, body, smooth = water_body(points, triangles, bottom=0.0)
    volume = _closed_and_outward(positions, body)
    assert volume == pytest.approx(5 * 3 * 1.0, rel=0.05)  # about the box under it
    assert smooth.sum() == len(triangles) and len(smooth) == len(body)
    assert positions[:, 2].min() == 0.0
    # a dry patch: still closed, with walls around the hole
    points, triangles = _grid(nan=[8, 9])
    positions, body, _ = water_body(points, triangles, bottom=0.0)
    assert 0 < _closed_and_outward(positions, body) < 15.0


def test_clear_water_goes_to_blender_as_a_body(reef3d_run, tmp_path):
    session = Session()
    session.open_case(reef3d_run)
    surface = next(i.id for i in session.scene if i.kind == "free_surface")
    session.set_property(surface, "variable", "none")
    path = export_scene(session, tmp_path / "out", [0.0], (64, 36), "underwater", 4)
    scene = json.loads(path.read_text())
    frame = scene["frames"][0]
    index, entry = next((i, o) for i, o in enumerate(frame["objects"]) if o["name"] == surface)
    water = scene["materials"][entry["material"]]
    assert water["transmission"] == 1.0 and water["ior"] == pytest.approx(1.333)
    assert water["volume"]["density"] > 0 and water["volume"]["scatter"] > 0  # under water
    assert water["ripples"]["scale"] > 0
    assert scene["world"]["fog"] is None  # the water's own haze instead of the fog box
    with np.load(tmp_path / "out" / frame["data"]) as arrays:
        assert arrays[f"smooth_{index}"].dtype == bool
        _closed_and_outward(arrays[f"positions_{index}"], arrays[f"triangles_{index}"])


def test_water_body_ends_on_the_bed():
    from lagoonviz.render.pathtrace import BedHeights, water_body

    x, y = np.meshgrid(np.linspace(0, 5, 26), np.linspace(0, 3, 16))
    bed = np.stack([x.ravel(), y.ravel(), (0.1 * x).ravel()], 1)  # sloping up to 0.5
    heights = BedHeights(bed, cells=32)
    np.testing.assert_allclose(heights(np.array([[0.0, 1.0], [5.0, 2.0]])), [0.0, 0.5], atol=0.02)
    points, triangles = _grid()
    positions, body, smooth = water_body(points, triangles, heights)
    floor = positions[len(points) : 2 * len(points)]
    np.testing.assert_allclose(floor[:, 2], 0.1 * points[:, 0], atol=0.03)  # on the bed
    volume = _closed_and_outward(positions, body)
    assert volume == pytest.approx(5 * 3 * (1.0 - 0.25), rel=0.06)  # water above the bed only


def test_water_body_ignores_hidden_triangles():
    from lagoonviz.render.pathtrace import water_body

    points, triangles = _grid()
    hidden = triangles.copy()
    hidden[6:8] = 0  # a dry cell, hidden as LAGOON does: degenerate, all corners 0
    positions, body, smooth = water_body(points, hidden, bottom=0.0)
    assert smooth.sum() == len(triangles) - 2  # the hidden ones are not on top
    assert (body[:, 0] != body[:, 1]).all()  # and no degenerate ones anywhere
    _closed_and_outward(positions, body)  # walls around the hole close it
    _, outer, _ = water_body(points, triangles, bottom=0.0)
    walls = len(outer) - 2 * len(triangles)
    assert walls == 2 * (2 * 5 + 2 * 3)  # two triangles per edge of the outer rim only


def test_skirt_hangs_from_the_rim_and_faces_out():
    positions, triangles = _grid(5, 4)
    walls, faces = skirt(positions, triangles, -2.0)
    assert len(faces) == 2 * 2 * (4 + 3)  # two triangles per rim edge
    assert walls[:, 2].min() == pytest.approx(-2.0)
    centre = positions[:, :2].mean(axis=0)
    a, b, c = (walls[faces[:, i]] for i in range(3))
    normal = np.cross(b - a, c - a)
    middle = (a + b + c)[:, :2] / 3 - centre
    assert ((normal[:, :2] * middle).sum(axis=1) > 0).all()
    assert np.allclose(normal[:, 2], 0, atol=1e-6)


def test_ocean_ring_surrounds_the_domain():
    positions, triangles = ocean_ring((0, 0, -5), (10, 4, 1), 0.5, 1000.0)
    assert np.allclose(positions[:, 2], 0.5)
    a, b, c = (positions[triangles[:, i]].astype(float) for i in range(3))
    area = np.cross(b - a, c - a)[:, 2] / 2
    assert (area > 0).all()  # facing up
    assert area.sum() == pytest.approx(2000.0**2 - 40.0)  # all but the domain


def test_coast_under_the_open_sky(reef3d_run, tmp_path):
    """Open ocean, clear water: the bed dressed as land by height and slope, running
    on beyond the domain and sinking into the sea; the water one body with the sea
    out to the horizon (no seam, no cliffs, no separate sea); the shore thinning out."""
    session = Session()
    session.open_case(reef3d_run)
    surface = next(i.id for i in session.scene if i.kind == "free_surface")
    bed_id = next(i.id for i in session.scene if i.kind == "bathymetry")
    session.set_property(surface, "variable", "none")
    session.set_property(bed_id, "variable", "none")
    level = session.data.case_water_level(surface)
    path = export_scene(session, tmp_path / "out", [0.0], (64, 36), "open_ocean", 4)
    scene = json.loads(path.read_text())
    frame = scene["frames"][0]
    by_name = {o["name"]: (i, o) for i, o in enumerate(frame["objects"])}
    assert SEA not in by_name and f"{bed_id}#sides" not in by_name
    bed = scene["materials"][by_name[bed_id][1]["material"]]
    assert bed["terrain"]["level"] == pytest.approx(level, abs=1e-5)
    assert bed["terrain"]["relief"] > 0 and bed["terrain"]["detail"] > 0
    assert bed["grain"] is None
    index, entry = by_name[surface]
    water = scene["materials"][entry["material"]]
    assert water["shore"]["fade"] > 0 and water["shore"]["foam"] > water["shore"]["fade"]
    assert water["swell"]["scale"] > 0
    # small waves fade out far off (no shimmering stripes on the far sea)
    near, far = water["ripples"]["fade"]
    assert 0 < near < far and len(water["ripples"]["centre"]) == 3
    assert entry["attributes"] == ["lagoon_depth"]
    with np.load(tmp_path / "out" / frame["data"]) as arrays:
        ground = arrays[f"positions_{by_name[bed_id][0]}"].astype(float)
        positions = arrays[f"positions_{index}"]
        depth = arrays[f"attribute_{index}_lagoon_depth"]
        assert depth.shape == (len(positions),)
        volume = _closed_and_outward(positions, arrays[f"triangles_{index}"])
        assert volume > 0
    session.set_time(0.0)
    shown = session.data.display(bed_id, session.data.load(bed_id))
    low, high = shown.positions.min(axis=0), shown.positions.max(axis=0)
    extent = float(np.ptp([low[:2], high[:2]], axis=0).max())
    centre = (low[:2] + high[:2]) / 2
    reach = np.linalg.norm(ground[:, :2] - centre, axis=1)
    assert reach.max() > 100 * extent  # out to the horizon
    far = ground[reach > 10 * extent]
    assert (far[:, 2] < level).all()  # the far sea floor is under the sea
    water_far = positions[np.linalg.norm(positions[:, :2] - centre, axis=1) > 10 * extent]
    assert water_far[:, 2].max() == pytest.approx(level, abs=1e-3)  # the still sea
    assert water_far[:, 2].min() < far[:, 2].min()  # its floor deeper than the sea floor
    assert frame["camera"]["clip_end"] > 100 * extent
    # in the wave flume: neither land nor sea, and no foam
    path = export_scene(session, tmp_path / "flume", [0.0], (64, 36), "wave_flume", 4)
    scene = json.loads(path.read_text())
    names = [o["name"] for o in scene["frames"][0]["objects"]]
    assert SEA not in names and f"{bed_id}#sides" not in names
    assert all("terrain" not in m for m in scene["materials"].values())


def test_rim_loop_and_rings():
    positions, triangles = _grid(6, 4)
    loop = rim_loop(positions, triangles)
    assert len(loop) == 2 * 5 + 2 * 3
    xy = positions[loop, :2]
    area = 0.5 * np.sum(xy[:, 0] * np.roll(xy[:, 1], -1) - np.roll(xy[:, 0], -1) * xy[:, 1])
    assert area == pytest.approx(15.0)  # counter-clockwise: the whole grid
    distances = ring_distances(1.0, 1000.0)
    assert distances[0] == 1.0 and distances[-1] == 1000.0 and len(distances) < 30
    rings = outward_rings(xy, distances, 5.0)
    # the first ring a cell out all round (the corners mitred), the last nearly round
    assert np.allclose(rings[0].min(axis=0), xy.min(axis=0) - 1.0)
    radius = np.linalg.norm(rings[-1] - xy.mean(axis=0), axis=1)
    assert radius.min() > 0.99 * 1000.0
    points = np.concatenate([positions, np.c_[rings.reshape(-1, 2), np.zeros(rings[..., 0].size)]])
    faces = ring_triangles(loop, len(positions), len(distances))
    a, b, c = (points[faces[:, i]] for i in range(3))
    assert (np.cross(b - a, c - a)[:, 2] > 0).all()  # facing up, none folded over
    # two patches far apart: no single outer edge
    second = positions + [100.0, 0.0, 0.0]
    both = np.concatenate([positions, second])
    assert rim_loop(both, np.concatenate([triangles, triangles + len(positions)])) is None


def test_apron_continues_the_rim_and_sinks():
    rim = np.array([5.0, 5.0, -2.0, -2.0, 5.0, 5.0, -2.0, -2.0])
    out = apron(rim, np.array([0.0, 1.0, 10.0, 100.0]), 1.0, 10.0, -20.0)
    assert np.allclose(out[0], rim)  # at the rim: the rim
    assert out[2].max() < rim.max() and out[2].min() > -20.0
    assert np.allclose(out[3], -20.0, atol=1e-6)  # far out: the deep floor
    assert np.allclose(loop_mean(rim, 100), rim.mean())


def test_diorama_is_a_model_on_a_stand(reef3d_run, tmp_path):
    """The bed a solid block (its sides down to a base under it), the water a closed
    body with clear sides, smoothed when rendered; no land dressing, no sea around."""
    session = Session()
    session.open_case(reef3d_run)
    surface = next(i.id for i in session.scene if i.kind == "free_surface")
    bed_id = next(i.id for i in session.scene if i.kind == "bathymetry")
    session.set_property(surface, "variable", "none")
    path = export_scene(session, tmp_path / "out", [0.0], (64, 36), "diorama", 4)
    scene = json.loads(path.read_text())
    frame = scene["frames"][0]
    by_name = {o["name"]: (i, o) for i, o in enumerate(frame["objects"])}
    assert SEA not in by_name and f"{bed_id}#sides" in by_name
    assert all("terrain" not in m or m["terrain"] is None for m in scene["materials"].values())
    assert by_name[surface][1]["subdivide"] == 2  # smooth crests, as ÆgirFrames
    water = scene["materials"][by_name[surface][1]["material"]]
    assert water["shore"]["foam"] == 0.0
    # ÆgirFrames' ocean water: tinted glass seen by the camera only, a faint haze
    assert water["camera_only"] and water["transmission"] == 1.0 and water["roughness"] == 0
    assert water["color"] == pytest.approx([0.0144, 0.1384, 0.2016], abs=2e-3)
    assert water["volume"]["density"] == pytest.approx(0.01)
    assert water["ripples"] is None  # smooth glass in the diorama
    with np.load(tmp_path / "out" / frame["data"]) as arrays:
        bed = arrays[f"positions_{by_name[bed_id][0]}"]
        sides = arrays[f"positions_{by_name[f'{bed_id}#sides'][0]}"]
        index = by_name[surface][0]
        assert _closed_and_outward(arrays[f"positions_{index}"], arrays[f"triangles_{index}"]) > 0
    assert sides[:, 2].min() < bed[:, 2].min()  # down to a base under the bed


def test_white_water_goes_to_blender(reef3d_run, tmp_path, fake_blender, monkeypatch):
    """Foam, spray and bubbles become point clouds with their own materials; Blender
    builds them (here: the stand-in) beside the meshes."""
    from lagoonviz.render import pathtrace

    calls = []

    def particles(self, time):
        calls.append(time)
        rng = np.random.default_rng(int(time * 10))
        return {
            "foam": (rng.random((50, 3)).astype(np.float32), np.full(50, 0.01, np.float32)),
            "spray": (rng.random((20, 3)).astype(np.float32), np.full(20, 0.005, np.float32)),
        }

    monkeypatch.setattr(pathtrace.WhiteWater, "at", particles)
    session = Session()
    session.open_case(reef3d_run)
    folder = tmp_path / "film.cycles"
    path = export_scene(session, folder, [0.0, 1.0], (64, 36), "diorama", 4)
    assert calls == [0.0, 1.0]
    scene = json.loads(path.read_text())
    frame = scene["frames"][1]
    by_name = {o["name"]: (i, o) for i, o in enumerate(frame["objects"])}
    index, foam = by_name["LAGOON foam"]
    assert foam["points"] and scene["materials"][foam["material"]]["white_water"] == "foam"
    with np.load(folder / frame["data"]) as arrays:
        assert arrays[f"points_{index}"].shape == (50, 3)
        assert arrays[f"radius_{index}"].shape == (50,)
    images = render_folder(folder, fake_blender)
    assert len(images) == 2
    # without foam: no particles
    calls.clear()
    path = export_scene(session, tmp_path / "plain", [0.0], (64, 36), "diorama", 4, foam=False)
    assert not calls
    names = [o["name"] for o in json.loads(path.read_text())["frames"][0]["objects"]]
    assert "LAGOON foam" not in names


def test_white_water_follows_the_outputs(tmp_path):
    """The simulation reads the free surface output by output, from a warm-up before
    the first time, without moving the session's time."""
    from reef3d_fixtures import make_run

    from lagoonviz.render.pathtrace import WhiteWater

    run = make_run(tmp_path / "run", frames=6, print_interval=0.5)
    session = Session()
    session.open_case(run)
    session.set_time(1.0)
    white = WhiteWater(session)
    assert white.ok
    read = []
    original = WhiteWater._surface

    def spy(self, time):
        read.append(time)
        return original(self, time)

    WhiteWater._surface = spy
    try:
        white.at(2.0)
        assert read[0] == 2.0 and min(read) < 2.0 and max(read) == 2.0
        read.clear()
        white.at(2.5)  # on from where it was
        assert read == [2.5]
    finally:
        WhiteWater._surface = original
    assert session.scene.time == 1.0


def test_sky_picture_is_turned_to_the_sun(reef3d_run, tmp_path):
    """Open ocean and Diorama: ÆgirFrames' sky picture lights the scene and is seen
    behind it, turned so its sun stands in the sun's direction; plain AgX, a weak
    lamp from where the picture's sun is."""
    from lagoonviz.render.pathtrace import SKY_FOLDER, read_hdr, sky_sun

    azimuth, elevation = sky_sun("sea_clear_sky.hdr")
    assert math.degrees(elevation) == pytest.approx(29, abs=2)
    session = Session()
    session.open_case(reef3d_run)
    path = export_scene(session, tmp_path / "a", [0.0], (64, 36), "diorama", 4)
    scene = json.loads(path.read_text())
    world = scene["world"]
    assert world["sky"] is None and world["background"] is None
    assert (tmp_path / "a" / "environment.hdr").read_bytes() == (
        SKY_FOLDER / "sea_clear_sky.hdr"
    ).read_bytes()
    assert scene["exposure"] == 0.0 and scene["look_contrast"] is None
    assert scene["sun"]["strength"] == pytest.approx(1.0)
    lamp = np.asarray(scene["sun"]["direction"])
    wanted = math.atan2(lamp[1], lamp[0])
    assert math.asin(lamp[2]) == pytest.approx(elevation, abs=1e-6)  # the picture's height
    # turning the picture by the rotation brings its sun to the lamp's direction
    assert (azimuth - world["rotation"] - wanted + math.pi) % (2 * math.pi) == pytest.approx(
        math.pi, abs=1e-6
    )
    # the session's sun: the picture turns with it, the lamp comes from it
    session.set_sun(200.0, 40.0)
    scene = json.loads(
        export_scene(session, tmp_path / "b", [0.0], (64, 36), "diorama", 4).read_text()
    )
    turned = (azimuth - scene["world"]["rotation"]) % (2 * math.pi)
    assert math.degrees(turned) == pytest.approx(200.0, abs=1e-4)
    assert math.degrees(math.asin(scene["sun"]["direction"][2])) == pytest.approx(40.0, abs=1e-4)
    # LAGOON's own sky in the other looks
    scene = json.loads(
        export_scene(session, tmp_path / "c", [0.0], (64, 36), "wave_flume", 4).read_text()
    )
    assert "rotation" not in scene["world"] and scene["look_contrast"]
    assert read_hdr(tmp_path / "c" / "environment.hdr").shape == (512, 1024, 3)


def test_hdr_run_lengths():
    from lagoonviz.render.pathtrace import _rle

    row = np.array([5, 5, 5, 5, 1, 2, 7, 7, 7] + [3] * 300 + [9], np.uint8)
    coded = _rle(row)
    assert len(coded) < 20
    decoded = []
    at = 0
    while at < len(coded):
        count = coded[at]
        if count > 128:
            decoded += [coded[at + 1]] * (count - 128)
            at += 2
        else:
            decoded += list(coded[at + 1 : at + 1 + count])
            at += 1 + count
    assert decoded == row.tolist()
