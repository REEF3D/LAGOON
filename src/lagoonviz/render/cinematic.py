# SPDX-License-Identifier: GPL-3.0-or-later
# Architect: Hans Bihs
"""Cinematic looks (phase 5): physically based materials and the surroundings they
reflect, for showcase images and animations rather than for reading values.

Three scene presets join the engineering looks ("studio", "ocean"):

* **open ocean**: a sky with a low sun, reflected by the water; sand and rock
  with a fine grain;
* **wave flume**: a lab hall with rows of ceiling lights, neutral and bright;
* **underwater**: the camera below the surface, blue depth fog, light from above.

The environments are made here (no image files): cube maps computed from the
direction of each texel. pygfx samples cube maps at (-x, y, z) of a world
direction (its cube maps are left-handed, the world right-handed), so the faces
are made the same way. Materials follow the metallic-roughness model: water is
smooth (its reflections come from the environment, Fresnel from the shader),
sand and rock rough with a tiled normal map of grain, steel metallic.

    faces = sky_faces(256, OPEN_OCEAN)       # (6, 256, 256, 4) uint8, NumPy only
    normals = grain_normals(256, "sand")     # (256, 256, 4) uint8 normal map

The NumPy part has no pygfx; :func:`environment_texture`, :func:`grain_texture`
and :func:`pbr_material` make the pygfx objects.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache

import numpy as np

# ======================================================================== presets


@dataclass(frozen=True)
class Preset:
    """A cinematic look: its surroundings, light and water."""

    name: str
    label: str
    environment: str  # "sky", "hall" or "deep": what is reflected and seen behind
    zenith: str  # sky colours from overhead down to the horizon, and below it
    horizon: str
    ground: str
    sun_direction: tuple[float, float, float]  # towards the sun, z up
    sun_color: str = "#fff4e0"
    sun: float = 3.0  # light intensities (pygfx: physical, diffuse = colour / pi)
    ambient: float = 0.25
    headlamp: float = 0.5
    env_intensity: float = 1.0  # how strongly materials reflect the surroundings
    water_color: str = "#0f4c5c"  # the water's own colour, when coloured by nothing
    water_roughness: float = 0.06
    fog: tuple[str, float] | None = None  # colour and power of the depth fog
    background: tuple[str, str] | None = None  # a gradient instead of the cube (bottom, top)
    contours: str = "#1e2833"
    shadows: bool = True
    extra: dict = field(default_factory=dict)


OPEN_OCEAN = Preset(
    "open_ocean",
    "Open ocean",
    "sky",
    zenith="#3a6fb0",
    horizon="#cfe0ee",
    ground="#123848",
    sun_direction=(-0.75, -0.35, 0.56),
    sun=3.2,
    ambient=0.2,
    headlamp=0.35,
    water_color="#0d4a5e",
    contours="#e8eef5",
)
WAVE_FLUME = Preset(
    "wave_flume",
    "Wave flume",
    "hall",
    zenith="#f4f2ee",  # ceiling between the lights
    horizon="#c9c7c2",  # walls
    ground="#6d6b67",  # floor
    sun_direction=(0.0, 0.25, 1.0),  # the ceiling lights, nearly overhead
    sun_color="#ffffff",
    sun=2.6,
    ambient=0.35,
    headlamp=0.6,
    env_intensity=0.9,
    water_color="#5f8f9a",  # clear tap water over a grey floor
    water_roughness=0.04,
)
UNDERWATER = Preset(
    "underwater",
    "Underwater",
    "deep",
    zenith="#9fd6e6",  # Snell's window: the bright sky seen through the surface
    horizon="#1f6f86",
    ground="#04202c",
    sun_direction=(-0.2, -0.1, 1.0),
    sun_color="#d8f4ff",
    sun=2.2,
    ambient=0.45,
    headlamp=0.3,
    env_intensity=0.7,
    water_color="#2a8aa0",
    water_roughness=0.1,
    fog=("#0b4a5e", 60.0),
    background=("#03202b", "#1d6f86"),
    contours="#e8eef5",
    shadows=False,
)
PRESETS = {p.name: p for p in (OPEN_OCEAN, WAVE_FLUME, UNDERWATER)}


# ======================================================================== materials
@dataclass(frozen=True)
class Surface:
    """Metallic-roughness parameters of a material; ``grain``: its normal map."""

    roughness: float
    metalness: float = 0.0
    grain: str = ""  # "sand", "rock", "concrete" or none
    grain_scale: float = 0.0  # strength of the normal map
    tile: float = 1.0  # metres per repeat of the normal map
    ior: float = 1.5
    clearcoat: float = 0.0


SURFACES = {
    "water": Surface(0.06, ior=1.33),
    "sand": Surface(0.95, grain="sand", grain_scale=0.55, tile=0.6),
    "rock": Surface(0.85, grain="rock", grain_scale=1.0, tile=2.0),
    "concrete": Surface(0.8, grain="concrete", grain_scale=0.3, tile=1.0),
    "steel": Surface(0.35, metalness=0.9),
    "ice": Surface(0.18, ior=1.31, clearcoat=0.6),
    "wood": Surface(0.7),
    "plain": Surface(0.6),
}
#: the material of an item kind that names none
KIND_SURFACES = {
    "free_surface": "water",
    "amr": "water",
    "bathymetry": "sand",
    "ice": "ice",
    "rods": "wood",
    "dem": "rock",
    "body": "steel",
    "structure": "concrete",
}


def surface_of(kind: str, material: str = "") -> str:
    """The material an item is drawn with: its own, else its kind's."""
    if material in SURFACES:
        return material
    return KIND_SURFACES.get(kind, "plain")


# ======================================================================== environments
def _rgb(color: str) -> np.ndarray:
    color = color.lstrip("#")
    return np.array([int(color[i : i + 2], 16) / 255 for i in (0, 2, 4)])


def cube_directions(size: int) -> np.ndarray:
    """(6, size, size, 3) world directions (unit, z up) of the texels of a pygfx cube
    map: faces +X, -X, +Y, -Y, +Z, -Z, rows top to bottom. pygfx samples a world
    direction d at (-d.x, d.y, d.z), so a texel at cube direction c shows d = (-c.x,
    c.y, c.z)."""
    t = (np.arange(size) + 0.5) / size * 2.0 - 1.0
    u, v = np.meshgrid(t, t)  # u along a row (s), v down the rows (t)
    one = np.ones_like(u)
    faces = [
        np.stack([one, -v, -u], -1),  # +X
        np.stack([-one, -v, u], -1),  # -X
        np.stack([u, one, v], -1),  # +Y
        np.stack([u, -one, -v], -1),  # -Y
        np.stack([u, -v, one], -1),  # +Z
        np.stack([-u, -v, -one], -1),  # -Z
    ]
    cube = np.stack(faces)
    cube /= np.linalg.norm(cube, axis=-1, keepdims=True)
    world = cube.copy()
    world[..., 0] = -cube[..., 0]
    return world


def _to_uint8(rgb: np.ndarray) -> np.ndarray:
    rgba = np.concatenate([np.clip(rgb, 0.0, 1.0), np.ones((*rgb.shape[:-1], 1))], -1)
    return np.ascontiguousarray(np.round(rgba * 255).astype(np.uint8))


def sky_faces(size: int, preset: Preset) -> np.ndarray:
    """A sky over the sea: overhead ``zenith`` to a hazy ``horizon``, a sun glow, and
    below the horizon the far sea (``ground``) brightening towards the horizon."""
    d = cube_directions(size)
    z = d[..., 2:3]
    zenith, horizon, ground = _rgb(preset.zenith), _rgb(preset.horizon), _rgb(preset.ground)
    up = np.clip(z, 0.0, 1.0) ** 0.45  # a broad hazy band over the horizon
    sky = horizon + (zenith - horizon) * up
    down = np.clip(-z, 0.0, 1.0) ** 0.35
    sea = horizon * 0.55 + (ground - horizon * 0.55) * down
    rgb = np.where(z >= 0.0, sky, sea)
    sun = np.asarray(preset.sun_direction, float)
    sun /= np.linalg.norm(sun)
    facing = np.clip((d @ sun)[..., None], 0.0, 1.0)
    glow = 0.35 * facing**12 + 0.9 * facing**600  # haze around it, the disc itself
    rgb = rgb + glow * _rgb(preset.sun_color) * (z >= -0.02)
    return _to_uint8(rgb)


def hall_faces(size: int, preset: Preset, rows: int = 6) -> np.ndarray:
    """A lab hall: a light ceiling with rows of light panels, walls, a darker floor."""
    d = cube_directions(size)
    z = d[..., 2]
    ceiling, walls, floor = _rgb(preset.zenith), _rgb(preset.horizon), _rgb(preset.ground)
    rgb = np.empty(d.shape)
    rgb[...] = walls
    up = z > 0.45
    down = z < -0.45
    rgb[down] = floor
    rgb[up] = ceiling * 0.85
    # light panels: on the ceiling plane z = 1, strips along x every 1/rows
    with np.errstate(divide="ignore", invalid="ignore"):
        px = d[..., 0] / z
        py = d[..., 1] / z
    strips = (np.abs((py * rows / 2.0) % 1.0 - 0.5) < 0.12) & (np.abs(px) < 3.0) & up
    rgb[strips] = 1.0
    # walls darker towards the floor
    band = (~up) & (~down)
    rgb[band] = walls * (0.75 + 0.25 * (z[band] + 0.45) / 0.9)[..., None]
    return _to_uint8(rgb)


def deep_faces(size: int, preset: Preset) -> np.ndarray:
    """Under water: the bright window of the sky overhead, blue all around, dark below."""
    d = cube_directions(size)
    z = d[..., 2:3]
    window, side, deep = _rgb(preset.zenith), _rgb(preset.horizon), _rgb(preset.ground)
    above = np.clip(z, 0.0, 1.0)
    # Snell's window: the sky squeezed into a cone of about 48.6 degrees from overhead
    cone = np.clip((z - np.cos(np.radians(48.6))) / 0.12, 0.0, 1.0)
    rgb = side + (window - side) * cone + (side * 0.3) * above
    below = np.clip(-z, 0.0, 1.0) ** 0.6
    rgb = np.where(z >= 0.0, rgb, side + (deep - side) * below)
    return _to_uint8(rgb)


ENVIRONMENTS = {"sky": sky_faces, "hall": hall_faces, "deep": deep_faces}


def environment_faces(preset: Preset, size: int = 256) -> np.ndarray:
    """The cube map faces of a preset's surroundings."""
    return ENVIRONMENTS[preset.environment](size, preset)


# ======================================================================== grain
def _tileable_noise(size: int, cells: int, rng: np.random.Generator) -> np.ndarray:
    """Smooth value noise that tiles: random values on a cells x cells lattice,
    interpolated with a smooth step, wrapping at the edges."""
    lattice = rng.random((cells, cells))
    t = np.arange(size) / size * cells
    i0 = np.floor(t).astype(int)
    f = t - i0
    f = f * f * (3 - 2 * f)
    i1 = (i0 + 1) % cells
    a = lattice[np.ix_(i0, i0)]
    b = lattice[np.ix_(i0, i1)]
    c = lattice[np.ix_(i1, i0)]
    e = lattice[np.ix_(i1, i1)]
    fy, fx = np.meshgrid(f, f, indexing="ij")
    return (a * (1 - fx) + b * fx) * (1 - fy) + (c * (1 - fx) + e * fx) * fy


GRAINS = {
    # octaves: (cells across the tile, weight)
    "sand": ((64, 0.6), (128, 0.4), (16, 0.25)),
    "rock": ((4, 1.0), (8, 0.6), (16, 0.35), (32, 0.2), (64, 0.1)),
    "concrete": ((32, 0.5), (128, 0.5)),
}


def grain_heights(size: int, grain: str, seed: int = 7) -> np.ndarray:
    """(size, size) heights in [0, 1] of a tiling grain."""
    rng = np.random.default_rng(seed)
    height = np.zeros((size, size))
    for cells, weight in GRAINS[grain]:
        height += weight * _tileable_noise(size, min(cells, size), rng)
    height -= height.min()
    return height / max(height.max(), 1e-12)


def grain_normals(size: int, grain: str, seed: int = 7) -> np.ndarray:
    """(size, size, 4) uint8 tangent-space normal map of a grain, tiling (encoded as
    (n + 1) / 2, alpha 255)."""
    height = grain_heights(size, grain, seed) * 4.0
    dx = (np.roll(height, -1, 1) - np.roll(height, 1, 1)) * 0.5 * size / 64
    dy = (np.roll(height, -1, 0) - np.roll(height, 1, 0)) * 0.5 * size / 64
    normal = np.stack([-dx, -dy, np.ones_like(height)], -1)
    normal /= np.linalg.norm(normal, axis=-1, keepdims=True)
    rgba = np.concatenate([(normal + 1.0) / 2.0, np.ones((size, size, 1))], -1)
    return np.ascontiguousarray(np.round(rgba * 255).astype(np.uint8))


def grain_coords(positions: np.ndarray, tile: float) -> np.ndarray:
    """(n, 2) texture coordinates of the grain: x and y over the tile size, with some
    of z, so steep faces do not smear it into streaks."""
    p = np.asarray(positions, np.float32)
    uv = np.stack([p[:, 0] + 0.5 * p[:, 2], p[:, 1] + 0.5 * p[:, 2]], -1) / max(tile, 1e-6)
    return np.ascontiguousarray(uv, dtype=np.float32)


# ======================================================================== pygfx
@lru_cache(maxsize=8)
def environment_texture(name: str, size: int = 256):
    """The preset's surroundings as a pygfx cube texture (mipmapped: rough materials
    reflect it blurred)."""
    import pygfx as gfx

    faces = environment_faces(PRESETS[name], size)
    return gfx.Texture(faces, dim=2, size=(size, size, 6), generate_mipmaps=True)


@lru_cache(maxsize=8)
def grain_texture(grain: str, size: int = 256):
    """A grain's normal map as a pygfx texture map on the second texture coordinates,
    repeating."""
    import pygfx as gfx

    texture = gfx.Texture(
        grain_normals(size, grain), dim=2, colorspace="physical", generate_mipmaps=True
    )
    return gfx.TextureMap(texture, uv_channel=1, filter="linear", wrap="repeat")


def pbr_material(surface: str, preset: Preset):
    """A pygfx physical material for a surface in a preset's surroundings (colour,
    colour map and opacity are set by the caller, as for the other looks)."""
    import pygfx as gfx

    s = SURFACES.get(surface, SURFACES["plain"])
    roughness = preset.water_roughness if surface == "water" else s.roughness
    material = gfx.MeshPhysicalMaterial(
        side="both", roughness=roughness, metalness=s.metalness, ior=s.ior
    )
    if s.clearcoat:
        material.clearcoat = s.clearcoat
        material.clearcoat_roughness = 0.1
    material.env_map = environment_texture(preset.name)
    material.env_map_intensity = preset.env_intensity
    if s.grain:
        material.normal_map = grain_texture(s.grain)
        material.normal_scale = (s.grain_scale, s.grain_scale)
    return material
