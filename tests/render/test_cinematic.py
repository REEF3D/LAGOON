"""Cinematic looks: generated surroundings, grain and materials (NumPy part; the
rendering is in test_offscreen)."""

import numpy as np
import pytest

from lagoonviz.render.cinematic import (
    OPEN_OCEAN,
    PRESETS,
    SURFACES,
    UNDERWATER,
    WAVE_FLUME,
    cube_directions,
    deep_faces,
    environment_faces,
    grain_coords,
    grain_heights,
    grain_normals,
    hall_faces,
    sky_faces,
    surface_of,
)

#: the cube map texel table of WebGPU (as OpenGL): face -> (sc, tc) from c = (x, y, z)
_FACE = {
    0: lambda x, y, z: (-z, -y),
    1: lambda x, y, z: (z, -y),
    2: lambda x, y, z: (x, z),
    3: lambda x, y, z: (x, -z),
    4: lambda x, y, z: (x, -y),
    5: lambda x, y, z: (-x, -y),
}


def sample(faces: np.ndarray, world: np.ndarray) -> np.ndarray:
    """Look up world directions as pygfx's shaders do: the cube at (-x, y, z)."""
    world = np.atleast_2d(np.asarray(world, float))
    c = world.copy()
    c[:, 0] = -c[:, 0]
    size = faces.shape[1]
    out = []
    for x, y, z in c:
        axis = int(np.argmax(np.abs((x, y, z))))
        value = (x, y, z)[axis]
        face = 2 * axis + (0 if value > 0 else 1)
        sc, tc = _FACE[face](x, y, z)
        s = int(np.clip((sc / abs(value) + 1) / 2 * size, 0, size - 1))
        t = int(np.clip((tc / abs(value) + 1) / 2 * size, 0, size - 1))
        out.append(faces[face, t, s])
    return np.array(out)


def _brightness(rgba: np.ndarray) -> np.ndarray:
    return rgba[..., :3].astype(float).mean(axis=-1)


def test_cube_directions_are_where_the_shader_looks():
    directions = cube_directions(64)
    assert directions.shape == (6, 64, 64, 3)
    assert np.allclose(np.linalg.norm(directions, axis=-1), 1.0)
    # a cube holding its own directions: sampling any world direction finds it again
    encoded = np.round((directions + 1) / 2 * 255).astype(np.uint8)
    faces = np.concatenate([encoded, np.full((6, 64, 64, 1), 255, np.uint8)], -1)
    rng = np.random.default_rng(1)
    for d in rng.normal(size=(200, 3)):
        d /= np.linalg.norm(d)
        found = sample(faces, d)[0, :3] / 255 * 2 - 1
        found /= np.linalg.norm(found)
        assert np.degrees(np.arccos(np.clip(found @ d, -1, 1))) < 3.0  # within a texel
    # up is the middle of +Z, +x the middle of -X (pygfx flips x)
    assert np.allclose(directions[4, 32, 32], (0, 0, 1), atol=0.05)
    assert np.allclose(directions[1, 32, 32], (1, 0, 0), atol=0.05)


def test_open_ocean_sky():
    faces = sky_faces(128, OPEN_OCEAN)
    assert faces.shape == (6, 128, 128, 4) and faces.dtype == np.uint8
    sun = np.asarray(OPEN_OCEAN.sun_direction) / np.linalg.norm(OPEN_OCEAN.sun_direction)
    up, horizon, down = sample(faces, [(0, 0, 1), (1, 0, 0.05), (0, 0, -1)])
    assert _brightness(sample(faces, sun)[0]) > 250  # the sun
    assert up[2] > up[0]  # blue overhead
    assert _brightness(horizon) > _brightness(up) > _brightness(down)  # hazy, deep below
    # the sun's glow lights the sky on its side, not the opposite one
    away = -sun * np.array([1, 1, 0]) + np.array([0, 0, sun[2]])
    toward = sun + np.array([0.15, 0, 0])
    assert _brightness(sample(faces, toward)[0]) > _brightness(sample(faces, away)[0])


def test_flume_hall_and_deep_water():
    hall = hall_faces(128, WAVE_FLUME)
    up = hall[4]  # the ceiling face
    assert (up[..., :3] == 255).all(axis=-1).mean() > 0.05  # rows of light panels
    floor = sample(hall, (0, 0, -1))[0]
    assert _brightness(floor) < _brightness(sample(hall, (1, 0, 0))[0])  # walls lighter
    deep = deep_faces(128, UNDERWATER)
    overhead, side, below = sample(deep, [(0, 0, 1), (1, 0, 0), (0, 0, -1)])
    assert _brightness(overhead) > _brightness(side) > _brightness(below)
    # Snell's window: bright within about 48.6 degrees of overhead, not beyond
    tilt = np.radians(30), np.radians(65)
    inside = sample(deep, (np.sin(tilt[0]), 0, np.cos(tilt[0])))[0]
    outside = sample(deep, (np.sin(tilt[1]), 0, np.cos(tilt[1])))[0]
    assert _brightness(inside) > _brightness(outside) + 40
    for preset in PRESETS.values():
        assert environment_faces(preset, 32).shape == (6, 32, 32, 4)


@pytest.mark.parametrize("grain", ["sand", "rock", "concrete"])
def test_grain_tiles(grain):
    heights = grain_heights(128, grain)
    assert heights.min() == 0.0 and heights.max() == 1.0
    # it tiles: the step across the seam is like any other step
    seam = np.abs(heights[:, 0] - heights[:, -1]).mean()
    inside = np.abs(np.diff(heights, axis=1)).mean()
    assert seam < 3 * inside
    normals = grain_normals(128, grain, relief=20.0)
    assert normals.shape == (128, 128, 4) and normals.dtype == np.uint8
    n = normals[..., :3] / 255 * 2 - 1
    assert (n[..., 2] > 0.2).all()  # facing out
    tilt = np.degrees(np.arctan2(np.hypot(n[..., 0], n[..., 1]), n[..., 2]))
    assert np.percentile(tilt, 95) == pytest.approx(20.0, abs=1.5)  # the relief asked for


def test_materials():
    assert surface_of("free_surface") == surface_of("amr") == "water"
    assert surface_of("bathymetry") == "sand"
    assert surface_of("body", "rock") == "rock" and surface_of("body") == "steel"
    assert surface_of("slice") == "plain"
    assert SURFACES["steel"].metalness > 0.5 and SURFACES["water"].roughness < 0.1
    assert SURFACES["sand"].grain == "sand"
    uv = grain_coords(np.array([[0.0, 0.0, 0.0], [0.6, 1.2, 0.0]], np.float32), 0.6)
    assert uv.dtype == np.float32 and np.allclose(uv[1], (1.0, 2.0))


def test_the_looks_are_listed():
    pytest.importorskip("pygfx")
    from lagoonviz.render.scene3d import STYLES

    for name, preset in PRESETS.items():
        assert STYLES[name]["cinematic"] == name
        assert STYLES[name]["sun"] == preset.sun
    assert "cinematic" not in STYLES["studio"]


def test_display_carries_the_material(tmp_path):
    from reef3d_fixtures import make_run, write_objects

    from lagoonviz.core import Session

    run = make_run(tmp_path / "run", frames=3, print_interval=0.5, objects=True)
    write_objects(run, "NHFLOW", 3)
    session = Session()
    session.open_case(run)
    body = next((i for i in session.scene if i.kind == "body"), None)
    if body is None:
        pytest.skip("no bodies in the fixture")
    session.set_property(body.id, "material", "rock")
    shown = session.data.display(body.id, session.data.load(body.id))
    assert shown.material == "rock"


def test_fog_colour_is_physical():
    from lagoonviz.render.cinematic import physical

    # effect passes mix in linear light: #0b4a5e (sRGB) is much darker there
    r, g, b, a = physical((0x0B / 255, 0x4A / 255, 0x5E / 255, 1.0))
    assert (round(r, 4), round(g, 3), round(b, 3), a) == (0.0033, 0.068, 0.112, 1.0)
    assert physical((1.0, 1.0, 1.0, 1.0)) == (1.0, 1.0, 1.0, 1.0)


def test_fog_starts_at_the_scene():
    from lagoonviz.render.cinematic import fog_settings

    low, high = np.zeros(3), np.array([8.0, 3.0, 1.0])
    size = np.linalg.norm(high)
    camera = (low + high) / 2 + np.array([0.0, -3 * size, 0.0])
    start, density = fog_settings(0.8, camera, low, high)
    assert start == pytest.approx(2.5 * size)  # clear up to the near side
    # half thick 0.8 scene sizes beyond it
    assert 1 - np.exp(-density * 0.8 * size) == pytest.approx(0.5)
    inside, _ = fog_settings(0.8, (low + high) / 2, low, high)
    assert inside == 0.0  # the camera inside the scene: fog from the camera on


def test_the_sky_glows_where_the_sun_is():
    from lagoonviz.render.cinematic import with_sun

    moved = with_sun(OPEN_OCEAN, (0.6, 0.6, 0.3))
    assert moved.name == OPEN_OCEAN.name and moved.water_color == OPEN_OCEAN.water_color
    assert with_sun(OPEN_OCEAN, None) is OPEN_OCEAN
    faces = sky_faces(128, moved)
    sun = np.asarray(moved.sun_direction)
    assert _brightness(sample(faces, sun / np.linalg.norm(sun))[0]) > 250
    old = np.asarray(OPEN_OCEAN.sun_direction) / np.linalg.norm(OPEN_OCEAN.sun_direction)
    assert _brightness(sample(faces, old)[0]) < 250
