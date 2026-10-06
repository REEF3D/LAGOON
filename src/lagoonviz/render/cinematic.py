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

from dataclasses import dataclass, field, replace
from functools import lru_cache

import numpy as np

from lagoonviz.core.looks import LOOK_SUNS

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
    #: depth fog: its colour, and how far beyond the scene's near side it is half
    #: thick, in sizes of the scene (the diagonal of what is shown)
    fog: tuple[str, float] | None = None
    background: tuple[str, str] | None = None  # a gradient instead of the cube (bottom, top)
    contours: str = "#1e2833"
    shadows: bool = True
    #: what path tracing builds around the data: "box" (the domain as it is), "coast"
    #: (land and sea run on to the horizon), "diorama" (a model on a stand: the bed
    #: a solid block, the water a body with clear sides) or "model" (a diorama whose
    #: bed is dressed as land, as in "coast", floating in the sky picture as
    #: ÆgirFrames shows a case)
    world: str = "box"
    #: how strongly the water absorbs light: over its mean depth it keeps exp(-absorb)
    absorb: float = 0.45
    sun_angle: float = 1.0  # the sun's size [degrees]: larger, softer shadows
    #: a sky picture (an HDR panorama in render/assets) that lights the scene and is
    #: seen behind it, its sun turned to where the look's sun is; path traced always,
    #: in the 3D view unless the look has a ``background`` of its own
    sky_picture: str | None = None
    #: path tracing and the 3D view: the bed's colour when it has the bed's own
    #: default (a dark rock, as ÆgirFrames' terrain), and that of structures and bodies
    #: with theirs (ÆgirFrames' red, half metallic and matte); None: their own colours
    bed: str | None = None
    structure: str | None = None
    extra: dict = field(default_factory=dict)


#: ÆgirFrames' colours: its terrain's dark rock and its solids' red (linear 0.443,
#: 0.014, 0.017, half metallic, roughness 1)
AEGIR_ROCK = "#57524d"
AEGIR_RED = "#b21f24"

OPEN_OCEAN = Preset(
    "open_ocean",
    "Open ocean",
    "sky",
    zenith="#3a6fb0",
    horizon="#cfe0ee",
    ground="#123848",
    sun_direction=LOOK_SUNS["open_ocean"],
    sun=3.2,
    ambient=0.2,
    headlamp=0.35,
    water_color="#20687c",  # ÆgirFrames' ocean water (linear 0.014, 0.138, 0.202)
    contours="#e8eef5",
    world="model",
    sky_picture="sea_clear_sky.hdr",
    structure=AEGIR_RED,
)
WAVE_FLUME = Preset(
    "wave_flume",
    "Wave flume",
    "hall",
    zenith="#f4f2ee",  # ceiling between the lights
    horizon="#c9c7c2",  # walls
    ground="#6d6b67",  # floor
    sun_direction=LOOK_SUNS["wave_flume"],  # the ceiling lights, nearly overhead
    sun_color="#ffffff",
    sun=2.6,
    ambient=0.35,
    headlamp=0.6,
    env_intensity=0.9,
    water_color="#20687c",  # ÆgirFrames' ocean water, as in its wave tank renders
    water_roughness=0.04,
    # as ÆgirFrames shows a tank: in its sky picture (also in the 3D view), on a
    # dark floor
    sky_picture="sea_clear_sky.hdr",
    bed=AEGIR_ROCK,
    structure=AEGIR_RED,
)
UNDERWATER = Preset(
    "underwater",
    "Underwater",
    "deep",
    zenith="#9fd6e6",  # Snell's window: the bright sky seen through the surface
    horizon="#1f6f86",
    ground="#04202c",
    sun_direction=LOOK_SUNS["underwater"],
    sun_color="#d8f4ff",
    sun=2.2,
    ambient=0.45,
    headlamp=0.3,
    env_intensity=0.7,
    water_color="#2a8aa0",
    water_roughness=0.1,
    fog=("#0b4a5e", 0.8),
    background=("#03202b", "#1d6f86"),
    contours="#e8eef5",
    shadows=False,
)
DIORAMA = Preset(
    "diorama",
    "Diorama",
    "sky",
    zenith="#8fb0d0",  # a soft, even sky: the model lit as in a light tent
    horizon="#e4ecf3",
    ground="#5b7491",
    sun_direction=LOOK_SUNS["diorama"],
    sun_color="#fff8ec",
    sun=2.0,
    ambient=0.35,
    headlamp=0.45,
    env_intensity=1.1,
    water_color="#20687c",  # ÆgirFrames' ocean water (linear 0.014, 0.138, 0.202)
    water_roughness=0.04,
    contours="#e8eef5",
    world="diorama",
    absorb=1.1,
    sun_angle=8.0,
    sky_picture="sea_clear_sky.hdr",
    bed=AEGIR_ROCK,
    structure=AEGIR_RED,
)
#: the Diorama look as it was before it took ÆgirFrames' sky picture and colours into
#: the 3D view: a blue backdrop, LAGOON's own soft sky reflected, a block of sand
DIORAMA_CLASSIC = replace(
    DIORAMA,
    name="diorama_classic",
    label="Diorama classic",
    background=("#3e5f86", "#a9bfd3"),  # a blue backdrop, lighter at the top
    bed=None,
    structure=None,
)
PRESETS = {p.name: p for p in (OPEN_OCEAN, WAVE_FLUME, UNDERWATER, DIORAMA, DIORAMA_CLASSIC)}


# ======================================================================== materials
@dataclass(frozen=True)
class Surface:
    """Metallic-roughness parameters of a material; ``grain``: its normal map."""

    roughness: float
    metalness: float = 0.0
    grain: str = ""  # "sand", "rock", "concrete" or none
    relief: float = 0.0  # the grain's slope [degrees] that 95 % of the surface stays under
    tile: float = 1.0  # metres per repeat of the normal map
    ior: float = 1.5
    clearcoat: float = 0.0


SURFACES = {
    "water": Surface(0.06, ior=1.33),
    "sand": Surface(0.95, grain="sand", relief=22.0, tile=0.6),
    "rock": Surface(0.85, grain="rock", relief=35.0, tile=2.0),
    "concrete": Surface(0.8, grain="concrete", relief=8.0, tile=1.0),
    "steel": Surface(0.35, metalness=0.9),
    "ice": Surface(0.18, ior=1.31, clearcoat=0.6),
    "wood": Surface(0.7),
    "plain": Surface(0.6),
    "painted": Surface(1.0, metalness=0.5),  # ÆgirFrames' solids: half metallic, matte
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


def look_colour(preset: Preset | None, kind: str, color, material: str = ""):
    """The colour and surface a look gives an item shown in its kind's default colour
    (no colour, colour map or material of its own chosen): ÆgirFrames' dark rock for
    a bed, its red for structures and bodies; None when the item keeps its own."""
    from lagoonviz.core.colormaps import PLAIN_COLORS

    if preset is None or material or not isinstance(color, str):
        return None
    if color.lower() != PLAIN_COLORS.get(kind, "").lower():
        return None
    if kind == "bathymetry" and preset.bed:
        return preset.bed, "rock"
    if kind in ("structure", "body") and preset.structure:
        return preset.structure, "painted"
    return None


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


def sky_colours(d: np.ndarray, preset: Preset, light: float = 1.0) -> np.ndarray:
    """A sky over the sea seen in directions ``d`` (..., 3), as sRGB values: overhead
    ``zenith`` to a hazy ``horizon``, a sun glow (``light`` times as bright), and below
    the horizon the far sea (``ground``) brightening towards the horizon."""
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
    return rgb + light * glow * _rgb(preset.sun_color) * (z >= -0.02)


def sky_faces(size: int, preset: Preset) -> np.ndarray:
    """The sky (:func:`sky_colours`) as cube map faces."""
    return _to_uint8(sky_colours(cube_directions(size), preset))


def hall_colours(d: np.ndarray, preset: Preset, light: float = 1.0, rows: int = 6) -> np.ndarray:
    """A lab hall seen in directions ``d``: a light ceiling with rows of light panels
    (``light``: their brightness), walls, a darker floor."""
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
    rgb[strips] = light
    # walls darker towards the floor
    band = (~up) & (~down)
    rgb[band] = walls * (0.75 + 0.25 * (z[band] + 0.45) / 0.9)[..., None]
    return rgb


def hall_faces(size: int, preset: Preset, rows: int = 6) -> np.ndarray:
    """The hall (:func:`hall_colours`) as cube map faces."""
    return _to_uint8(hall_colours(cube_directions(size), preset, rows=rows))


def deep_colours(d: np.ndarray, preset: Preset, light: float = 1.0) -> np.ndarray:
    """Under water, seen in directions ``d``: the bright window of the sky overhead
    (``light`` times as bright), blue all around, dark below."""
    z = d[..., 2:3]
    window, side, deep = _rgb(preset.zenith), _rgb(preset.horizon), _rgb(preset.ground)
    above = np.clip(z, 0.0, 1.0)
    # Snell's window: the sky squeezed into a cone of about 48.6 degrees from overhead
    cone = np.clip((z - np.cos(np.radians(48.6))) / 0.12, 0.0, 1.0)
    rgb = side + (light * window - side) * cone + (side * 0.3) * above
    below = np.clip(-z, 0.0, 1.0) ** 0.6
    return np.where(z >= 0.0, rgb, side + (deep - side) * below)


def deep_faces(size: int, preset: Preset) -> np.ndarray:
    """Under water (:func:`deep_colours`) as cube map faces."""
    return _to_uint8(deep_colours(cube_directions(size), preset))


#: the surroundings: colours (sRGB values, the lights may exceed 1) in directions
ENVIRONMENTS = {"sky": sky_colours, "hall": hall_colours, "deep": deep_colours}


def environment_faces(preset: Preset, size: int = 256) -> np.ndarray:
    """The cube map faces of a preset's surroundings: its sky picture (as Cycles shows
    it) or LAGOON's own."""
    d = cube_directions(size)
    if uses_picture(preset):
        return _to_uint8(picture_colours(d, preset))
    return _to_uint8(ENVIRONMENTS[preset.environment](d, preset))


def uses_picture(preset: Preset) -> bool:
    """The 3D view shows the look's sky picture (a look with a background of its own
    keeps that and LAGOON's surroundings)."""
    return bool(preset.sky_picture) and preset.background is None


#: Blender's AgX view transform, as the minimal fit of its curve (Wrensch 2023): the
#: inset into AgX's space, the log range in stops, the outset back
AGX_INSET = np.array(
    [
        [0.842479062253094, 0.0423282422610123, 0.0423756549057051],
        [0.0784335999999992, 0.878468636469772, 0.0784336],
        [0.0792237451477643, 0.0791661274605434, 0.879142973793104],
    ]
)
AGX_OUTSET = np.array(
    [
        [1.19687900512017, -0.0528968517574562, -0.0529716355144438],
        [-0.0980208811401368, 1.15190312990417, -0.0980434501171241],
        [-0.0990297440797205, -0.0989611768448433, 1.15107367264116],
    ]
)
AGX_STOPS = (-12.47393, 4.026069)


def agx(linear: np.ndarray) -> np.ndarray:
    """Linear light (..., 3) as Blender's AgX shows it: display values in [0, 1]."""
    x = np.asarray(linear, np.float64) @ AGX_INSET
    low, high = AGX_STOPS
    x = (np.clip(np.log2(np.maximum(x, 1e-10)), low, high) - low) / (high - low)
    x2 = x * x
    x4 = x2 * x2
    y = (
        15.5 * x4 * x2
        - 40.14 * x4 * x
        + 31.96 * x4
        - 6.868 * x2 * x
        + 0.4298 * x2
        + 0.1191 * x
        - 0.00232
    )
    return np.clip(y @ AGX_OUTSET, 0.0, 1.0)


def picture_turn(preset: Preset) -> float:
    """How far a look's sky picture is turned about z (radians) so its sun stands at
    the azimuth of the look's sun, as Cycles turns it."""
    from lagoonviz.render.pathtrace import sky_sun

    azimuth, _ = sky_sun(str(preset.sky_picture))
    sun = np.asarray(preset.sun_direction, float)
    return azimuth - float(np.arctan2(sun[1], sun[0]))


def picture_colours(d: np.ndarray, preset: Preset) -> np.ndarray:
    """The look's sky picture seen in directions ``d`` (..., 3), turned to its sun and
    shown as Cycles shows it (AgX): display values."""
    from lagoonviz.render.pathtrace import sky_picture

    sky = sky_picture(str(preset.sky_picture))
    turn = picture_turn(preset)
    c, s = np.cos(turn), np.sin(turn)
    x = c * d[..., 0] - s * d[..., 1]
    y = s * d[..., 0] + c * d[..., 1]
    z = np.clip(d[..., 2], -1.0, 1.0)
    height, width = sky.shape[:2]
    # Blender's equirectangular map (see panorama_directions), sampled bilinearly
    u = (0.5 - np.arctan2(y, x) / (2 * np.pi)) * width - 0.5
    v = (0.5 - np.arcsin(z) / np.pi) * height - 0.5
    u0, v0 = np.floor(u).astype(int), np.floor(v).astype(int)
    fu, fv = (u - u0)[..., None], (v - v0)[..., None]
    rows = np.clip([v0, v0 + 1], 0, height - 1)
    cols = np.mod([u0, u0 + 1], width)
    linear = (
        sky[rows[0], cols[0]] * (1 - fu) * (1 - fv)
        + sky[rows[0], cols[1]] * fu * (1 - fv)
        + sky[rows[1], cols[0]] * (1 - fu) * fv
        + sky[rows[1], cols[1]] * fu * fv
    )
    return agx(linear)


def panorama_directions(width: int) -> np.ndarray:
    """(width / 2, width, 3) world directions (unit, z up) of an equirectangular
    panorama as Blender maps one (rows top to bottom): the middle column looks along
    +x, longitude grows to the left (u = 0.5 - atan2(y, x) / 2 pi), the top row
    looks up."""
    height = width // 2
    u = (np.arange(width) + 0.5) / width
    v = (np.arange(height) + 0.5) / height
    lon = np.pi * (1.0 - 2.0 * u)
    lat = np.pi * (0.5 - v)
    lon, lat = np.meshgrid(lon, lat)
    return np.stack([np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)], -1)


def environment_panorama(preset: Preset, width: int = 1024, light: float = 4.0) -> np.ndarray:
    """A preset's surroundings as an equirectangular panorama of physical (linear)
    light, (width / 2, width, 3) float32, for path tracing: the lights (sun glow,
    ceiling panels, Snell's window) ``light`` times as bright as in the cube map."""
    colours = ENVIRONMENTS[preset.environment]
    d = panorama_directions(width)
    rgb = np.clip(colours(d, preset), 0.0, 1.0)
    linear = np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)
    # the lights' extra brightness on top, as light (sRGB stops at white)
    brighter = np.maximum(colours(d, preset, light=light) - colours(d, preset), 0.0)
    return (linear + brighter).astype(np.float32)


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
    # octaves: (cells across the tile, weight); sand: ripples a few to a tile (they show
    # at the scale of a whole domain), the grains on them
    "sand": ((4, 0.9), (8, 0.45), (32, 0.15), (128, 0.08)),
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


def grain_normals(size: int, grain: str, relief: float = 20.0, seed: int = 7) -> np.ndarray:
    """(size, size, 4) uint8 tangent-space normal map of a grain, tiling (encoded as
    (n + 1) / 2, alpha 255). ``relief``: the slope [degrees] that 95 % of the surface
    stays under; the grain's heights are scaled to it."""
    height = grain_heights(size, grain, seed)
    dx = (np.roll(height, -1, 1) - np.roll(height, 1, 1)) * 0.5
    dy = (np.roll(height, -1, 0) - np.roll(height, 1, 0)) * 0.5
    steep = float(np.percentile(np.hypot(dx, dy), 95))
    gain = np.tan(np.radians(relief)) / max(steep, 1e-12)
    normal = np.stack([-gain * dx, -gain * dy, np.ones_like(height)], -1)
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
def sun_key(direction) -> tuple[float, float, float]:
    """A sun direction as a short hashable key (unit vector, 3 decimals)."""
    d = np.asarray(direction, float)
    d = d / max(float(np.linalg.norm(d)), 1e-12)
    return tuple(round(float(v), 3) for v in d)


def with_sun(preset: Preset, direction=None) -> Preset:
    """The preset with its sun towards ``direction`` (None: its own)."""
    if direction is None:
        return preset
    return replace(preset, sun_direction=sun_key(direction))


@lru_cache(maxsize=12)
def environment_texture(name: str, size: int = 256, sun: tuple | None = None):
    """The preset's surroundings as a pygfx cube texture (mipmapped: rough materials
    reflect it blurred), with the sun's glow towards ``sun`` (None: the preset's)."""
    import pygfx as gfx

    faces = environment_faces(with_sun(PRESETS[name], sun), size)
    return gfx.Texture(faces, dim=2, size=(size, size, 6), generate_mipmaps=True)


@lru_cache(maxsize=8)
def grain_texture(grain: str, relief: float = 20.0, size: int = 256):
    """A grain's normal map as a pygfx texture map on the second texture coordinates,
    repeating."""
    import pygfx as gfx

    texture = gfx.Texture(
        grain_normals(size, grain, relief), dim=2, colorspace="physical", generate_mipmaps=True
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
    material.env_map = environment_texture(preset.name, sun=sun_key(preset.sun_direction))
    material.env_map_intensity = preset.env_intensity
    if s.grain:
        material.normal_map = grain_texture(s.grain, s.relief)
        material.normal_scale = (1.0, 1.0)
    return material


_FOG_WGSL = """
    @fragment
    fn fs_main(varyings: Varyings) -> @location(0) vec4<f32> {
        let texIndex = vec2i(varyings.position.xy);
        let d = textureLoad(depthTex, texIndex, 0);
        // the distance along the view: standard depth, 0 at the near and 1 at the far plane
        let n = u_effect.near;
        let f = u_effect.far;
        let z = n * f / max(f - d * (f - n), 1e-9);
        let amount = 1.0 - exp(-u_effect.density * max(z - u_effect.start, 0.0));
        let color = textureLoad(colorTex, texIndex, 0);
        return vec4<f32>(mix(color.rgb, u_effect.color.rgb, amount), color.a);
    }
"""


@lru_cache(maxsize=1)
def depth_fog_class():
    """A pygfx effect pass: fog that thickens with the distance from the camera,
    1 - exp(-density (z - start)) beyond ``start``. pygfx's own fog pass works on the
    raw depth, which hardly changes across a scene; this one turns it back into the
    distance (with the camera's near and far planes)."""
    import pygfx as gfx
    from pygfx.renderers.wgpu import EffectPass

    class DepthFogPass(EffectPass):
        USES_DEPTH = True
        uniform_type = dict(
            EffectPass.uniform_type,
            color="4xf4",
            near="f4",
            far="f4",
            start="f4",
            density="f4",
        )
        wgsl = _FOG_WGSL

        def __init__(self, color="#0b4a5e"):
            super().__init__()
            self.set(color, 0.1, 1000.0, 0.0, 0.0)

        def set(self, color, near, far, start, density) -> None:
            # effect passes work on physical (linear) colours; the colour is sRGB
            self._uniform_data["color"] = physical(tuple(gfx.Color(color)))
            self._uniform_data["near"] = float(near)
            self._uniform_data["far"] = float(far)
            self._uniform_data["start"] = float(start)
            self._uniform_data["density"] = float(density)

    return DepthFogPass


def physical(rgba) -> tuple[float, float, float, float]:
    """An sRGB colour (0..1 floats) as physical (linear) light, alpha kept."""
    c = np.asarray(rgba, float)
    rgb = np.where(c[:3] <= 0.04045, c[:3] / 12.92, ((c[:3] + 0.055) / 1.055) ** 2.4)
    return (*map(float, rgb), float(c[3]) if c.size > 3 else 1.0)


def fog_settings(half: float, camera_position, low, high) -> tuple[float, float]:
    """(start, density) of the depth fog: clear up to the scene's near side (its
    bounding sphere seen from the camera), half thick ``half`` scene sizes beyond."""
    low, high = np.asarray(low, float), np.asarray(high, float)
    size = max(float(np.linalg.norm(high - low)), 1e-9)
    centre = (low + high) / 2
    distance = float(np.linalg.norm(np.asarray(camera_position, float) - centre))
    start = max(distance - size / 2, 0.0)
    return start, float(np.log(2.0) / (max(half, 1e-6) * size))
