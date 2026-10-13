# SPDX-License-Identifier: GPL-3.0-or-later
"""The 3D scene: LAGOON items drawn with pygfx (WebGPU: Metal, Vulkan or Direct3D 12).

This module knows nothing about Qt. The GUI gives it a canvas to draw on; the
offscreen renderer gives it an image canvas. Both draw the same scene, so a view
on screen and an image made headless on a cluster look the same.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pygfx as gfx

from lagoonviz.core.colormaps import colors, lut, normalize
from lagoonviz.core.data.manager import (
    ContourLines,
    DisplaySurface,
    GaugeMarks,
    Trails,
    VectorArrows,
)
from lagoonviz.core.looks import DEFAULT_SUN
from lagoonviz.render.cinematic import (
    PRESETS,
    grain_coords,
    look_colour,
    surface_of,
    uses_picture,
)

#: The lights and gloss of every look: a soft sun from above for the shadows, most of
#: the light from a headlamp that moves with the camera, so slopes keep their shading
#: from any side, and some sheen on the water. pygfx lights are physical (diffuse =
#: colour / pi).
LIGHTING = {
    "ambient": 0.45,
    # fixed light from above: casts the shadows. Strong enough that they show (at 0.6
    # they darkened the ground by 1 %), with the headlamp still modelling the waves
    "sun": 2.4,
    "headlamp": 1.1,  # light that moves with the camera
    "shininess": 20,
    "specular": "#494949",
    # water is glossier: a soft sheen on the slopes that face the light
    "water_shininess": 40,
    "water_specular": "#5a5a5a",
    "shadows": True,
}

#: Viewport looks: the same light on a white ("studio", as in the REEF3D result
#: animations) or a dark blue ("ocean") background.
STYLES = {
    # background: bottom, top (pygfx order); contour lines dark on white, light on blue
    "studio": {"background": ("#e6eaef", "#ffffff"), "contours": "#1e2833", **LIGHTING},
    "ocean": {"background": ("#1c3354", "#3b5f8f"), "contours": "#e8eef5", **LIGHTING},
}
#: the cinematic looks (``lagoonviz.render.cinematic``): physically based materials in
#: surroundings they reflect; the engineering looks above stay as they are
for _name, _preset in PRESETS.items():
    STYLES[_name] = {
        **LIGHTING,
        "background": _preset.background or (_preset.ground, _preset.zenith),
        "contours": _preset.contours,
        "ambient": _preset.ambient,
        "sun": _preset.sun,
        "headlamp": _preset.headlamp,
        "shadows": _preset.shadows,
        "cinematic": _name,
    }
DEFAULT_STYLE = "studio"

#: the Diorama look in 3D: how see-through the water's top and sides are
DIORAMA_WATER_OPACITY = 0.8
DIORAMA_SIDE_OPACITY = 0.7
#: item kinds that are water
WATER_KINDS = ("free_surface", "amr")

#: where the sun stands, seen from the scene: about 31 degrees above the horizon and
#: mostly from -x, across the crests of waves that travel along x (REEF3D's usual
#: set-up), so the slopes facing it and facing away differ clearly
SUN_DIRECTION = np.array(DEFAULT_SUN) / np.linalg.norm(DEFAULT_SUN)
#: shadow map depth bias, as a fraction of the shadow camera's depth range
SHADOW_BIAS = 0.0008

#: room around the data when framing (perspective makes the near side look larger)
FRAME_MARGIN = 1.05


def _fov_distance(fov: float) -> float:
    """Camera distance per unit of view extent, as pygfx computes it."""
    try:
        from pygfx.cameras._perspective import fov_distance_factor

        return float(fov_distance_factor(fov))
    except ImportError:  # pragma: no cover (older or newer pygfx)
        fov = max(float(fov), 0.1)
        return 0.5 / np.tan(0.5 * np.radians(fov))


#: named views: (view direction, up). z is up, as in REEF3D.
VIEWS = {
    "front": ((0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),  # looking along +y
    "back": ((0.0, -1.0, 0.0), (0.0, 0.0, 1.0)),
    "right": ((-1.0, 0.0, 0.0), (0.0, 0.0, 1.0)),  # looking along -x
    "left": ((1.0, 0.0, 0.0), (0.0, 0.0, 1.0)),
    "top": ((0.0, 0.0, -1.0), (0.0, 1.0, 0.0)),
    "bottom": ((0.0, 0.0, 1.0), (0.0, 1.0, 0.0)),
    "perspective": ((-1.0, 1.2, -0.7), (0.0, 0.0, 1.0)),
}

#: the headlamp's place on the camera (pygfx: a light on a camera shines along the view)
HEADLAMP = (0.4, -0.6, 1.0)
#: 2D from the top: the headlamp, fixed in the scene, shines as it does in the 3D view
#: (along the perspective view's direction); straight down it would light every flat
#: face fully and put a glare over them, a washed-out picture
PLAN_HEADLAMP = tuple(
    -np.asarray(VIEWS["perspective"][0]) / np.linalg.norm(VIEWS["perspective"][0])
)


def _background(colors: tuple[str, str]) -> gfx.Background:
    if hasattr(gfx.Background, "from_color"):
        return gfx.Background.from_color(*colors)
    return gfx.Background(None, gfx.BackgroundMaterial(*colors))


def _look_background(look: dict, sun=None) -> gfx.Background:
    """A look's background: its colour gradient, or a cinematic look's surroundings
    (with the sun's glow towards ``sun``, None: the look's own)."""
    name = look.get("cinematic")
    preset = PRESETS.get(name) if name else None
    if preset is not None and preset.background is None:
        from lagoonviz.render.cinematic import environment_texture, sun_key

        key = sun_key(sun if sun is not None else preset.sun_direction)
        texture = environment_texture(name, sun=key)
        return gfx.Background(None, gfx.BackgroundSkyboxMaterial(map=texture))
    return _background(look["background"])


def _sun_direction(look: dict) -> np.ndarray:
    """Towards a look's own sun; under a sky picture at the picture's sun's height (as
    Cycles lights it), in the direction of the look's sun."""
    preset = PRESETS.get(look.get("cinematic") or "")
    if preset is None:
        return SUN_DIRECTION
    direction = np.asarray(preset.sun_direction, float)
    direction = direction / np.linalg.norm(direction)
    if uses_picture(preset):
        from lagoonviz.render.pathtrace import sky_sun

        _, elevation = sky_sun(str(preset.sky_picture))
        azimuth = np.arctan2(direction[1], direction[0])
        direction = np.array(
            [
                np.cos(elevation) * np.cos(azimuth),
                np.cos(elevation) * np.sin(azimuth),
                np.sin(elevation),
            ]
        )
    return direction


#: the set-up editor's selected command in the 3D view
HIGHLIGHT = "#ffd23f"


def _label(text: str, where, color: str):
    """A label at a point: its size fixed on screen, drawn over everything, with a
    dark outline so it reads on water and on sand (None if pygfx cannot draw text)."""
    try:
        material = gfx.TextMaterial(color=color, outline_color="#10161d", outline_thickness=0.25)
        label = gfx.Text(
            material=material,
            text=text,
            font_size=13,
            screen_space=True,
            anchor="bottom-left",
        )
    except (AttributeError, TypeError, ValueError):  # an older pygfx
        return None
    for name, value in (("depth_test", False), ("render_queue", 4000)):
        try:
            setattr(material, name, value)
        except (AttributeError, ValueError, TypeError):
            pass
    label.local.position = tuple(float(v) for v in where)
    return label


def _rgba(color: str) -> tuple[float, float, float, float]:
    color = color.lstrip("#")
    return tuple(int(color[i : i + 2], 16) / 255 for i in (0, 2, 4)) + (1.0,)  # type: ignore[return-value]


def _corners(positions: np.ndarray) -> tuple[np.ndarray | None, np.ndarray | None]:
    """(low, high) of finite vertex positions; (None, None) if there are none."""
    low, high = positions.min(axis=0), positions.max(axis=0)
    if not (np.isfinite(low).all() and np.isfinite(high).all()):
        finite = positions[np.isfinite(positions).all(axis=1)]
        if not finite.size:
            return None, None
        low, high = finite.min(axis=0), finite.max(axis=0)
    return low.astype(float), high.astype(float)


def _texel_coords(unit: np.ndarray, colormap: str) -> np.ndarray:
    """[0, 1] values to texture coordinates between the first and last texel centre."""
    n = len(lut(colormap))
    coords = (0.5 + np.nan_to_num(unit) * (n - 1)) / n
    return np.ascontiguousarray(coords, dtype=np.float32)


def _update_buffer(geometry: gfx.Geometry, name: str, data: np.ndarray) -> bool:
    """Write new data into an existing buffer. False if the size changed."""
    buffer = getattr(geometry, name, None)
    if buffer is None or buffer.data.shape != data.shape:
        return False
    buffer.data[...] = data
    if hasattr(buffer, "update_full"):
        buffer.update_full()
    else:  # older pygfx
        buffer.update_range()
    return True


@dataclass
class _Shown:
    mesh: gfx.Mesh
    kind: str
    colormap: str
    #: the triangle array last uploaded (kept, so a new array is never mistaken for it)
    triangles: np.ndarray
    has_values: bool
    #: corners of the vertex positions, kept with each update (bounds without a scan)
    low: np.ndarray | None = None
    high: np.ndarray | None = None
    #: the display data last shown (to make the material again for another look)
    display: DisplaySurface | None = None


def _north_angle(north) -> float:
    """The angle about z that turns the model's y to north (None: 0)."""
    nx, ny = north if north is not None else (0.0, 1.0)
    return math.atan2(-nx, ny)


def _quaternion_product(a, b) -> np.ndarray:
    """a * b of two (x, y, z, w) quaternions: b, then a."""
    ax, ay, az, aw = (float(v) for v in a)
    bx, by, bz, bw = (float(v) for v in b)
    return np.array(
        [
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
            aw * bw - ax * bx - ay * by - az * bz,
        ]
    )


class SceneRenderer:
    """Keeps a pygfx scene in sync with the display data of LAGOON items."""

    def __init__(self, style: str = DEFAULT_STYLE) -> None:
        self.style_name = style if style in STYLES else DEFAULT_STYLE
        look = STYLES[self.style_name]
        self.scene = gfx.Scene()
        self.background = _look_background(look)
        self.scene.add(self.background)
        self.ambient = gfx.AmbientLight(intensity=look["ambient"])
        self.scene.add(self.ambient)
        # a sun from above, fixed in the scene: wave crests read the same from any side
        self.sun = gfx.DirectionalLight(intensity=look["sun"])
        #: the session's sun (a unit vector towards it), or None for the look's own
        self.sun_override: np.ndarray | None = None
        self.sun_direction = _sun_direction(look)
        self._sun_colour(look)
        self.sun.local.position = tuple(self.sun_direction)
        self._fog = None
        self.sun.cast_shadow = bool(look.get("shadows", False))
        self.plan = False
        self.scene.add(self.sun)
        self._shadow_fit: tuple | None = None
        self.camera = gfx.PerspectiveCamera(45, 16 / 9)
        try:
            self.camera.world.reference_up = (0, 0, 1)
        except AttributeError:  # older pygfx: up is given per call
            pass
        self.headlamp = gfx.DirectionalLight(intensity=look["headlamp"])
        self.headlamp.local.position = HEADLAMP
        self.camera.add(self.headlamp)  # moves with the camera
        self.scene.add(self.camera)
        self.content = gfx.Group()
        self.scene.add(self.content)
        self._shown: dict[str, _Shown] = {}
        self._textures: dict[str, object] = {}
        self._framed = False
        self._marker: gfx.Points | None = None
        # gauges sit outside ``content`` so they do not change the framed bounds
        self._gauges: dict[str, gfx.Group] = {}
        #: contour lines of an item; outside ``content``, like the gauges
        self._contours: dict[str, gfx.Line] = {}
        #: the Diorama look's stand: the sides of the bed and of the water, per item
        self._stand: dict[str, gfx.Mesh] = {}
        self._rims: dict[str, tuple[np.ndarray, np.ndarray]] = {}  # triangles -> open edges
        #: "Case set-up" items: lines, dots, areas and labels, with their extent
        self._setup: dict[str, gfx.Group] = {}
        self._setup_bounds: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        self._bed_heights: tuple[np.ndarray, object] | None = None  # bed positions -> heights
        #: background maps: a textured rectangle (or frame round the domain) per item,
        #: outside ``content`` so they do not change the framed bounds
        self._backgrounds: dict[str, tuple[gfx.Mesh, int]] = {}
        self._outline: gfx.Line | None = None
        #: north in model coordinates (set_north), None: the model's y
        self.north: tuple[float, float] | None = None
        #: whether the camera has seen north (a camera set from a session has not: the
        #: first north then turns nothing)
        self._north_known = False

    # ------------------------------------------------------------- items
    def _texture(self, colormap: str):
        """The colour map as a 1D texture, clamped at both ends.

        With pygfx's default (repeat), values at the top of the range wrapped round
        to the bottom colour and showed as dark spots on the crests.
        """
        if colormap not in self._textures:
            texture = gfx.Texture(lut(colormap), dim=1)
            try:
                texture = gfx.TextureMap(texture, filter="linear", wrap="clamp")
            except (AttributeError, TypeError):  # older pygfx: texel centres still help
                pass
            self._textures[colormap] = texture
        return self._textures[colormap]

    @property
    def preset(self):
        """The cinematic preset of the look (with the session's sun, if it has one), or
        None for an engineering look."""
        from lagoonviz.render.cinematic import with_sun

        preset = PRESETS.get(STYLES[self.style_name].get("cinematic") or "")
        if preset is None:
            return None
        return with_sun(preset, getattr(self, "sun_override", None))

    def _sun_colour(self, look: dict) -> None:
        preset = PRESETS.get(look.get("cinematic") or "")
        try:
            self.sun.color = preset.sun_color if preset is not None else "#ffffff"
        except (AttributeError, ValueError):
            pass

    def _surface(self, kind: str, shown: DisplaySurface) -> tuple[str, str]:
        """The colour and surface an item is drawn with in the look: the look's own
        for an item in its kind's default colour (ÆgirFrames' red; the bed keeps its
        light sand in the 3D view), else its own."""
        given = look_colour(self.preset, kind, shown.color, shown.material, view=True)
        if given is not None and shown.values is None:
            return given
        return shown.color, surface_of(kind, shown.material)

    def _material(self, kind: str, shown: DisplaySurface) -> gfx.Material:
        preset = self.preset
        if preset is not None:
            from lagoonviz.render.cinematic import pbr_material

            material = pbr_material(self._surface(kind, shown)[1], preset)
        else:
            material = gfx.MeshPhongMaterial(side="both")
            self._gloss(material, kind)
        try:
            material.pick_write = True  # clicks report which triangle was hit
        except AttributeError:
            pass
        self._style(material, kind, shown)
        return material

    def _gloss(self, material: gfx.Material, kind: str) -> None:
        """Shininess and specular colour of the look; water is glossier."""
        look = STYLES[self.style_name]
        water = kind in ("free_surface", "amr")
        material.shininess = look["water_shininess" if water else "shininess"]
        try:
            material.specular = look["water_specular" if water else "specular"]
        except (AttributeError, ValueError):
            pass

    def _style(self, material: gfx.Material, kind: str, shown: DisplaySurface) -> None:
        if shown.values is not None:
            material.color_mode = "vertex_map"
            material.map = self._texture(shown.colormap)
            material.color = "#ffffff"
        else:
            material.color_mode = "uniform"
            material.map = None
            preset = self.preset
            water = preset is not None and surface_of(kind, shown.material) == "water"
            material.color = preset.water_color if water else self._surface(kind, shown)[0]
        opacity = shown.opacity
        if self.diorama and kind in WATER_KINDS and shown.values is None:
            opacity = min(opacity, DIORAMA_WATER_OPACITY)  # the bed shows through
        material.opacity = opacity
        # transparent surfaces must not hide what lies behind them
        if hasattr(material, "alpha_mode"):
            try:
                material.alpha_mode = "blend" if opacity < 1.0 else "solid"
            except (ValueError, TypeError):
                pass

    def update(self, kind: str, shown: DisplaySurface) -> None:
        """Create or update the mesh (or lines) of an item from its display data."""
        if shown.point_size is not None:
            self._update_points(kind, shown)
            return
        if shown.segments is not None and shown.triangles.size == 0:
            self._update_lines(kind, shown)
            return
        if shown.positions.shape[0] == 0 or shown.triangles.size == 0:
            # nothing at this time (an iso-value the field does not reach, a slice
            # outside the volume): show nothing rather than fail
            self.remove(shown.item_id)
            return
        normals = shown.normals
        texcoords = (
            _texel_coords(normalize(shown.values, shown.vmin, shown.vmax), shown.colormap)
            if shown.values is not None
            else np.zeros(shown.positions.shape[0], np.float32)
        )
        indices = np.ascontiguousarray(shown.triangles, dtype=np.uint32)
        current = self._shown.get(shown.item_id)
        same_sizes = (
            current is not None
            and isinstance(current.mesh, gfx.Mesh)
            and current.mesh.geometry.positions.data.shape == shown.positions.shape
            and current.mesh.geometry.indices.data.shape == indices.shape
        )
        if same_sizes:
            geometry = current.mesh.geometry
            same_sizes = (
                _update_buffer(geometry, "positions", shown.positions)
                and _update_buffer(geometry, "normals", normals)
                and _update_buffer(geometry, "texcoords", texcoords)
                and (
                    current.triangles is shown.triangles
                    or _update_buffer(geometry, "indices", indices)
                )
            )
            if same_sizes:
                self._grain(current.mesh, shown.positions, kind, shown.material)
        if not same_sizes:
            if current is not None:
                self.content.remove(current.mesh)
            # pygfx buffers keep the array they are given and later updates write into
            # it: give them copies, never the cached frame's (or dataset's) own arrays
            geometry = gfx.Geometry(
                positions=np.array(shown.positions, np.float32),
                indices=np.array(indices, np.uint32),
                normals=np.array(normals, np.float32),
                texcoords=np.array(texcoords, np.float32),
            )
            mesh = gfx.Mesh(geometry, self._material(kind, shown))
            self._grain(mesh, shown.positions, kind, shown.material)
            mesh.cast_shadow = True
            mesh.receive_shadow = True
            self.content.add(mesh)
            current = _Shown(mesh, kind, shown.colormap, shown.triangles, shown.values is not None)
            self._shown[shown.item_id] = current
        else:
            self._style(current.mesh.material, kind, shown)
        current.triangles = shown.triangles
        current.low, current.high = _corners(shown.positions)
        current.mesh.visible = shown.visible
        current.colormap = shown.colormap
        current.has_values = shown.values is not None
        current.display = shown
        self._update_stand(shown.item_id)
        self._fit_shadow()

    # ------------------------------------------------------------- the diorama's stand
    @property
    def diorama(self) -> bool:
        """A look that shows the model on a stand (the Diorama and, as Cycles renders
        it, Open ocean), in 3D: the sides of bed and water."""
        preset = self.preset
        return preset is not None and preset.world in ("diorama", "model") and not self.plan

    def _update_stand(self, item_id: str) -> None:
        """The sides of an item on a stand: the bed a solid block down to a base, the
        water's sides down to the bed (see-through, in its colour)."""
        shown = self._shown.get(item_id)
        if not self.diorama or shown is None or shown.kind not in ("bathymetry", *WATER_KINDS):
            self._remove_stand(item_id)
            if item_id in self._shown and shown.kind == "bathymetry":
                self._bed_heights = None
            return
        display = shown.display
        positions = np.asarray(display.positions, np.float64)
        rim = self._rim(item_id, positions, display.triangles)
        if rim is None or not len(rim):
            self._remove_stand(item_id)
            return
        top_a, top_b = positions[rim[:, 0]], positions[rim[:, 1]]
        if shown.kind == "bathymetry":
            low_a = low_b = np.full(len(rim), self._base())
        else:
            heights = self._bed_under()
            if heights is not None:
                low_a, low_b = heights(top_a[:, :2]), heights(top_b[:, :2])
            else:
                low_a = low_b = np.full(len(rim), self._base())
        keep = np.isfinite(top_a).all(axis=1) & np.isfinite(top_b).all(axis=1)
        keep &= np.isfinite(low_a) & np.isfinite(low_b)
        keep &= (top_a[:, 2] > low_a) | (top_b[:, 2] > low_b)
        top_a, top_b, low_a, low_b = top_a[keep], top_b[keep], low_a[keep], low_b[keep]
        if not len(top_a):
            self._remove_stand(item_id)
            return
        bottom_a, bottom_b = top_a.copy(), top_b.copy()
        bottom_a[:, 2], bottom_b[:, 2] = (
            np.minimum(low_a, top_a[:, 2]),
            np.minimum(low_b, top_b[:, 2]),
        )
        k = len(top_a)
        points = np.concatenate([top_a, top_b, bottom_b, bottom_a]).astype(np.float32)
        i = np.arange(k)
        faces = np.concatenate(
            [np.stack([i, i + k, i + 2 * k], 1), np.stack([i, i + 2 * k, i + 3 * k], 1)]
        )
        # flat faces, facing out (to the right of the rim seen from above)
        d = top_b - top_a
        out = np.stack([d[:, 1], -d[:, 0], np.zeros(k)], 1)
        out /= np.maximum(np.linalg.norm(out, axis=1, keepdims=True), 1e-12)
        normals = np.concatenate([out, out, out, out]).astype(np.float32)
        indices = faces.astype(np.uint32)
        mesh = self._stand.get(item_id)
        if (
            mesh is not None
            and mesh.geometry.positions.data.shape == points.shape
            and _update_buffer(mesh.geometry, "positions", points)
            and _update_buffer(mesh.geometry, "normals", normals)
        ):
            mesh.visible = shown.mesh.visible
            return
        self._remove_stand(item_id)
        geometry = gfx.Geometry(positions=points, indices=indices, normals=normals)
        mesh = gfx.Mesh(geometry, self._stand_material(shown))
        mesh.cast_shadow = shown.kind == "bathymetry"
        mesh.receive_shadow = True
        mesh.visible = shown.mesh.visible
        self.scene.add(mesh)
        self._stand[item_id] = mesh
        if shown.kind == "bathymetry":  # the water's sides end on this bed
            for other, item in list(self._shown.items()):
                if item.kind in WATER_KINDS:
                    self._update_stand(other)

    def _stand_material(self, shown: _Shown) -> gfx.Material:
        from lagoonviz.render.cinematic import pbr_material

        preset = self.preset
        if shown.kind == "bathymetry":
            color, surface = self._surface(shown.kind, shown.display)
            material = pbr_material("rock" if surface == "rock" else "sand", preset)
            material.normal_map = None  # the grain needs texture coordinates the sides lack
            material.color = color
            return material
        material = pbr_material("water", preset)
        material.color = preset.water_color
        material.opacity = DIORAMA_SIDE_OPACITY
        try:
            material.alpha_mode = "blend"
        except (AttributeError, ValueError, TypeError):
            pass
        return material

    def _rim(self, item_id: str, positions: np.ndarray, triangles: np.ndarray):
        """The open edges of an item's surface (kept while its triangles stay the same)."""
        from lagoonviz.render.pathtrace import _facing_up_and_rim

        known = self._rims.get(item_id)
        if known is not None and known[0] is triangles:
            return known[1]
        t = np.asarray(triangles, np.int64).reshape(-1, 3)
        finite = np.isfinite(positions).all(axis=1)
        t = t[finite[t].all(axis=1)]
        t = t[(t[:, 0] != t[:, 1]) & (t[:, 1] != t[:, 2]) & (t[:, 0] != t[:, 2])]
        rim = _facing_up_and_rim(positions, t)[1] if len(t) else None
        self._rims[item_id] = (triangles, rim)
        return rim

    def _bed_under(self):
        """The bed's heights under any point (the first bed shown), or None."""
        from lagoonviz.render.pathtrace import BedHeights

        bed = next((s for s in self._shown.values() if s.kind == "bathymetry"), None)
        if bed is None or bed.display is None:
            return None
        positions = bed.display.positions
        if self._bed_heights is None or self._bed_heights[0] is not positions:
            try:
                heights = BedHeights(np.asarray(positions, float))
            except ValueError:
                return None
            self._bed_heights = (positions, heights)
        return self._bed_heights[1]

    def _base(self) -> float:
        """The stand's base: under the lowest bed by a quarter of the model's height."""
        corners = self.bounds(visible_only=False)
        if corners is None:
            return 0.0
        low, high = corners
        bed = [s.low for s in self._shown.values() if s.kind == "bathymetry" and s.low is not None]
        bottom = float(min(b[2] for b in bed)) if bed else float(low[2])
        size = float(np.ptp([low[:2], high[:2]], axis=0).max())
        return bottom - max(0.25 * (float(high[2]) - bottom), 0.01 * size, 1e-6)

    def _remove_stand(self, item_id: str) -> None:
        mesh = self._stand.pop(item_id, None)
        if mesh is not None:
            self.scene.remove(mesh)

    def _refresh_stand(self) -> None:
        """The stand again for every item (the look or the 2D view changed)."""
        for item_id in list(self._stand):
            self._remove_stand(item_id)
        self._bed_heights = None
        if self.diorama:
            for item_id in list(self._shown):
                self._update_stand(item_id)

    def _grain(self, mesh: gfx.Mesh, positions: np.ndarray, kind: str, material: str) -> None:
        """Texture coordinates of a material's grain (its normal map), from the positions."""
        if getattr(mesh.material, "normal_map", None) is None:
            return
        from lagoonviz.render.cinematic import SURFACES

        # three tiles across the domain at least (some 12 sand ripples): a finer grain
        # turns into an even speckle when the whole domain is in view
        span = np.ptp(np.asarray(positions)[:, :2], axis=0).max() if len(positions) else 0.0
        tile = max(SURFACES[surface_of(kind, material)].tile, float(span) / 3.0)
        uv = grain_coords(positions, tile)
        if not _update_buffer(mesh.geometry, "texcoords1", uv):
            mesh.geometry.texcoords1 = gfx.Buffer(uv)

    def set_sun_direction(self, direction, surroundings: bool = True) -> None:
        """Put the sun towards ``direction`` (a vector, z up), or back where the look
        puts it (None). ``surroundings``: the sky's sun glow moves along now (it takes
        a moment; while a slider moves, :meth:`refresh_surroundings` once it rests)."""
        if direction is None:
            self.sun_override = None
            self.sun_direction = _sun_direction(STYLES[self.style_name])
        else:
            d = np.asarray(direction, float)
            self.sun_override = d / max(float(np.linalg.norm(d)), 1e-12)
            self.sun_direction = self.sun_override
        self._shadow_fit = None
        self._fit_shadow()
        if surroundings:
            self.refresh_surroundings()

    def refresh_surroundings(self) -> None:
        """A cinematic look's surroundings again, with the sun's glow where the sun is
        now: the background and what the materials reflect (the glint on the water)."""
        preset = self.preset
        if preset is None or preset.background is not None and not self._shown:
            return
        from lagoonviz.render.cinematic import environment_texture, sun_key

        texture = environment_texture(preset.name, sun=sun_key(self.sun_direction))
        if preset.background is None:
            self.scene.remove(self.background)
            self.background = _look_background(STYLES[self.style_name], self.sun_direction)
            self.scene.add(self.background)
        for shown in self._shown.values():
            material = getattr(shown.mesh, "material", None)
            if getattr(material, "env_map", None) is not None:
                material.env_map = texture

    def _fit_shadow(self) -> None:
        """Point the sun along ``sun_direction``. With shadows, its shadow map is aimed
        at what is shown: it covers the scene's bounding sphere, seen from the sun, so
        any domain size gets the full map resolution."""
        corners = self.bounds(visible_only=False) if self.sun.cast_shadow else None
        if corners is None:
            # no shadow map to aim: only the direction counts (from the target to the
            # light), so the target goes back to the origin
            self._shadow_fit = None
            self.sun.target.local.position = (0.0, 0.0, 0.0)
            self.sun.local.position = tuple(self.sun_direction)
            return
        low, high = corners
        centre = (low + high) / 2
        radius = max(float(np.linalg.norm(high - low)) / 2, 1e-6)
        fit = (*np.round(centre, 3), round(radius, 3), *np.round(self.sun_direction, 6))
        if fit == self._shadow_fit:
            return
        self._shadow_fit = fit
        self.sun.target.local.position = tuple(centre)
        self.sun.local.position = tuple(centre + self.sun_direction * 2 * radius)
        shadow = getattr(self.sun, "shadow", None)
        if shadow is None:
            return
        camera = shadow.camera
        camera.width = camera.height = 2 * radius
        camera.depth_range = (0.0, 4 * radius)
        shadow.bias = SHADOW_BIAS
        shadow.cull_mode = "none"  # open surfaces (water, bed) cast shadows from both sides

    def _update_lines(self, kind: str, shown: DisplaySurface) -> None:
        """Mooring lines and other segment data, drawn as screen-space lines.

        Line colours are computed here (one RGBA per vertex), which works the same
        on every pygfx version.
        """
        if not len(shown.segments):
            self.remove(shown.item_id)  # no lines at this time
            return
        order = shown.segments.astype(np.int64).ravel()
        positions = np.ascontiguousarray(shown.positions[order], dtype=np.float32)
        if shown.values is not None:
            rgba = colors(shown.values[order], shown.colormap, shown.vmin, shown.vmax)
        else:
            rgba = np.tile(np.array(_rgba(shown.color), np.float32), (order.size, 1))
        rgba = np.ascontiguousarray(rgba, dtype=np.float32)
        rgba[:, 3] = shown.opacity
        current = self._shown.get(shown.item_id)
        if current is not None and isinstance(current.mesh, gfx.Line):
            geometry = current.mesh.geometry
            if not (
                _update_buffer(geometry, "positions", positions)
                and _update_buffer(geometry, "colors", rgba)
            ):
                self.content.remove(current.mesh)
                current = None
        elif current is not None:
            self.content.remove(current.mesh)
            current = None
        if current is None:
            material = gfx.LineSegmentMaterial(thickness=shown.line_width, color_mode="vertex")
            line = gfx.Line(gfx.Geometry(positions=positions, colors=rgba), material)
            self.content.add(line)
            current = _Shown(line, kind, shown.colormap, shown.segments, shown.values is not None)
            self._shown[shown.item_id] = current
        current.mesh.material.thickness = shown.line_width
        current.low, current.high = _corners(positions)
        current.mesh.visible = shown.visible

    def _update_points(self, kind: str, shown: DisplaySurface) -> None:
        """Particles: one screen-space point each, coloured per point."""
        positions = np.ascontiguousarray(shown.positions, dtype=np.float32)
        if not len(positions):
            self.remove(shown.item_id)  # no particles at this time
            return
        if shown.values is not None:
            rgba = colors(shown.values, shown.colormap, shown.vmin, shown.vmax)
        else:
            rgba = np.tile(np.array(_rgba(shown.color), np.float32), (len(positions), 1))
        rgba = np.ascontiguousarray(rgba, dtype=np.float32)
        rgba[:, 3] = shown.opacity
        current = self._shown.get(shown.item_id)
        if current is not None and isinstance(current.mesh, gfx.Points):
            geometry = current.mesh.geometry
            if not (
                _update_buffer(geometry, "positions", positions)
                and _update_buffer(geometry, "colors", rgba)
            ):
                self.content.remove(current.mesh)
                current = None
        elif current is not None:
            self.content.remove(current.mesh)
            current = None
        if current is None:
            material = gfx.PointsMaterial(size=shown.point_size, color_mode="vertex")
            try:
                material.pick_write = True
            except AttributeError:
                pass
            points = gfx.Points(
                gfx.Geometry(positions=np.array(positions), colors=np.array(rgba)), material
            )
            self.content.add(points)
            current = _Shown(points, kind, shown.colormap, None, shown.values is not None)
            self._shown[shown.item_id] = current
        current.mesh.material.size = shown.point_size
        current.low, current.high = _corners(positions)
        current.mesh.visible = shown.visible

    def set_visible(self, item_id: str, visible: bool) -> None:
        if item_id in self._shown:
            self._shown[item_id].mesh.visible = visible
        if item_id in self._stand:
            self._stand[item_id].visible = visible

    def set_style(self, style: str) -> None:
        """Switch the look (background and lights) of the viewport."""
        self.style_name = style if style in STYLES else DEFAULT_STYLE
        look = STYLES[self.style_name]
        self.scene.remove(self.background)
        self.background = _look_background(look, self.sun_override)
        self.scene.add(self.background)
        self.ambient.intensity = look["ambient"]
        self.sun.intensity = look["sun"]
        self.sun_direction = (
            self.sun_override if self.sun_override is not None else _sun_direction(look)
        )
        self._sun_colour(look)
        self.sun.cast_shadow = bool(look.get("shadows", False)) and not self.plan
        self._shadow_fit = None
        self._fit_shadow()
        self.headlamp.intensity = look["headlamp"]
        for key, line in self._contours.items():
            if not key.endswith("#trails"):  # trails keep their item's colour
                line.material.color = look.get("contours", "#1e2833")
        cinematic = self.preset is not None
        for shown in self._shown.values():
            material = shown.mesh.material
            if not isinstance(shown.mesh, gfx.Mesh):
                continue
            physical = isinstance(material, gfx.MeshStandardMaterial)
            if shown.display is not None and (physical or cinematic):
                # another family of material (or other surroundings): made again
                shown.mesh.material = self._material(shown.kind, shown.display)
                self._grain(
                    shown.mesh,
                    shown.mesh.geometry.positions.data,
                    shown.kind,
                    shown.display.material,
                )
            elif hasattr(material, "shininess"):
                self._gloss(material, shown.kind)
        self._refresh_stand()

    def item_at(self, world_object) -> str | None:
        """The item a picked pygfx object belongs to."""
        for item_id, shown in self._shown.items():
            if shown.mesh is world_object:
                return item_id
        return None

    def set_marker(self, point: tuple[float, float, float] | None) -> None:
        """Show (or hide) the orange probe marker at a point; it keeps a fixed size on
        screen and is drawn on top, so it stays visible at any zoom."""
        if point is None:
            if self._marker is not None:
                self._marker.visible = False
            return
        if self._marker is None:
            material = gfx.PointsMaterial(size=14, color="#e07b39")  # as PROBE_COLOR
            for name, value in (("depth_test", False), ("render_queue", 4000)):
                try:
                    setattr(material, name, value)
                except (AttributeError, ValueError, TypeError):
                    pass
            self._marker = gfx.Points(
                gfx.Geometry(positions=np.zeros((1, 3), np.float32)), material
            )
            self.scene.add(self._marker)
        _update_buffer(self._marker.geometry, "positions", np.array([point], np.float32))
        self._marker.visible = True

    def update_gauges(self, marks: GaugeMarks) -> None:
        """Wave gauges as thin staffs with a dot at the water, in their plot colours.

        The dots are drawn on top, like the probe marker, so a gauge behind a wave
        crest is still found.
        """
        rgba = np.array([_rgba(c) for c in marks.colors], np.float32).reshape(-1, 4)
        staff_positions = np.array(marks.staffs.reshape(-1, 3), np.float32)
        staff_colors = np.array(np.repeat(rgba, 2, axis=0), np.float32)
        wet = np.isfinite(marks.dots[:, 2])
        dot_positions = np.array(marks.dots[wet], np.float32)
        dot_colors = np.array(rgba[wet], np.float32)
        group = self._gauges.get(marks.item_id)
        if group is not None and len(group.children) == (2 if wet.any() else 1):
            # while playing only the dots move: update the buffers in place
            staffs, *dots = group.children
            if (
                _update_buffer(staffs.geometry, "positions", staff_positions)
                and _update_buffer(staffs.geometry, "colors", staff_colors)
                and all(
                    _update_buffer(d.geometry, "positions", dot_positions)
                    and _update_buffer(d.geometry, "colors", dot_colors)
                    for d in dots
                )
            ):
                group.visible = marks.visible
                return
        self.remove_gauges(marks.item_id)
        group = gfx.Group()
        group.visible = marks.visible
        line_material = gfx.LineSegmentMaterial(thickness=2.5, color_mode="vertex")
        if not len(marks.dots):
            # a wave profile lies on the water it describes: draw it on top
            for name, value in (("depth_test", False), ("render_queue", 3900)):
                try:
                    setattr(line_material, name, value)
                except (AttributeError, ValueError, TypeError):
                    pass
        staffs = gfx.Line(
            gfx.Geometry(positions=staff_positions, colors=staff_colors), line_material
        )
        group.add(staffs)
        if wet.any():
            material = gfx.PointsMaterial(size=11, color_mode="vertex")
            for name, value in (("depth_test", False), ("render_queue", 3900)):
                try:
                    setattr(material, name, value)
                except (AttributeError, ValueError, TypeError):
                    pass
            dots = gfx.Points(gfx.Geometry(positions=dot_positions, colors=dot_colors), material)
            group.add(dots)
        self.scene.add(group)
        self._gauges[marks.item_id] = group

    def update_setup(self, marks) -> None:
        """The signs of a "Case set-up" item (:class:`SetupMarks`): thin lines and dots in
        their colours, faint areas (the still water, relaxation zones), and labels
        that keep their size on screen. Drawn again whenever the set-up changes."""
        self.remove_setup(marks.item_id)
        group = gfx.Group()
        group.visible = marks.visible
        corners = []
        for color, triangles in marks.faces:
            positions = np.ascontiguousarray(triangles.reshape(-1, 3), np.float32)
            indices = np.arange(len(positions), dtype=np.uint32).reshape(-1, 3)
            material = gfx.MeshBasicMaterial(color=color, opacity=0.16, side="both")
            for name, value in (("alpha_mode", "blend"), ("depth_write", False)):
                try:
                    setattr(material, name, value)
                except (AttributeError, ValueError, TypeError):
                    pass
            group.add(gfx.Mesh(gfx.Geometry(positions=positions, indices=indices), material))
        for color, segments in marks.lines:
            positions = np.ascontiguousarray(segments.reshape(-1, 3), np.float32)
            corners.append(positions)
            material = gfx.LineSegmentMaterial(thickness=1.8, color=color)
            group.add(gfx.Line(gfx.Geometry(positions=positions), material))
        for color, points in marks.points:
            positions = np.ascontiguousarray(points.reshape(-1, 3), np.float32)
            corners.append(positions)
            group.add(
                gfx.Points(
                    gfx.Geometry(positions=positions), gfx.PointsMaterial(size=7, color=color)
                )
            )
        for text, where, color in marks.labels:
            label = _label(text, where, color)
            if label is not None:
                group.add(label)
        if marks.highlight is not None:  # the command selected in the editor, on top
            segments, points = marks.highlight
            for name, positions in (("lines", segments), ("points", points)):
                if not len(positions):
                    continue
                positions = np.ascontiguousarray(positions.reshape(-1, 3), np.float32)
                corners.append(positions)
                if name == "lines":
                    material = gfx.LineSegmentMaterial(thickness=4.0, color=HIGHLIGHT)
                    shape = gfx.Line(gfx.Geometry(positions=positions), material)
                else:
                    material = gfx.PointsMaterial(size=13, color=HIGHLIGHT)
                    shape = gfx.Points(gfx.Geometry(positions=positions), material)
                for key, value in (("depth_test", False), ("render_queue", 3900)):
                    try:
                        setattr(material, key, value)
                    except (AttributeError, ValueError, TypeError):
                        pass
                group.add(shape)
        self.scene.add(group)
        self._setup[marks.item_id] = group
        if corners:
            every = np.concatenate(corners)
            every = every[np.isfinite(every).all(axis=1)]
            if len(every):
                self._setup_bounds[marks.item_id] = (every.min(axis=0), every.max(axis=0))

    def remove_setup(self, item_id: str) -> None:
        group = self._setup.pop(item_id, None)
        self._setup_bounds.pop(item_id, None)
        if group is not None:
            self.scene.remove(group)

    @property
    def setup_ids(self) -> list[str]:
        return list(self._setup)

    def update_background(self, shown) -> None:
        """A background map item (:class:`BackgroundMap`): its image on a horizontal
        rectangle in model coordinates, unlit, under the results."""
        positions, triangles, uv = shown.mesh()
        current = self._backgrounds.get(shown.item_id)
        if current is not None and current[1] == id(shown.image):
            mesh = current[0]
            if (
                _update_buffer(mesh.geometry, "positions", positions)
                and _update_buffer(mesh.geometry, "texcoords", uv)
                and _update_buffer(mesh.geometry, "indices", triangles)
            ):
                self._background_material(mesh.material, shown.opacity)
                mesh.visible = shown.visible
                return
        self.remove_background(shown.item_id)
        image = np.ascontiguousarray(shown.image, np.uint8)
        try:
            texture = gfx.Texture(image, dim=2, generate_mipmaps=True)
        except TypeError:  # older pygfx
            texture = gfx.Texture(image, dim=2)
        try:
            texture_map = gfx.TextureMap(
                texture, filter="linear", mipmap_filter="linear", wrap="clamp"
            )
        except (AttributeError, TypeError):
            texture_map = texture
        material = gfx.MeshBasicMaterial(map=texture_map, side="both")
        self._background_material(material, shown.opacity)
        geometry = gfx.Geometry(
            positions=np.array(positions, np.float32),
            indices=np.array(triangles, np.uint32),
            texcoords=np.array(uv, np.float32),
        )
        mesh = gfx.Mesh(geometry, material)
        mesh.visible = shown.visible
        mesh.receive_shadow = False
        mesh.cast_shadow = False
        self.scene.add(mesh)
        self._backgrounds[shown.item_id] = (mesh, id(shown.image))

    @staticmethod
    def _background_material(material, opacity: float) -> None:
        material.opacity = float(opacity)
        for name, value in (("alpha_mode", "blend" if opacity < 1.0 else "solid"),):
            try:
                setattr(material, name, value)
            except (AttributeError, ValueError, TypeError):
                pass

    def remove_background(self, item_id: str) -> None:
        found = self._backgrounds.pop(item_id, None)
        if found is not None:
            self.scene.remove(found[0])

    @property
    def background_ids(self) -> list[str]:
        return list(self._backgrounds)

    def set_outline(self, corners: np.ndarray | None, color: str = HIGHLIGHT) -> None:
        """A closed outline of four corners drawn over everything (the domain being
        drawn in the set-up), or None for none."""
        if corners is None or len(corners) < 2:
            if self._outline is not None:
                self.scene.remove(self._outline)
                self._outline = None
            return
        ring = np.asarray(corners, np.float32).reshape(-1, 3)
        ring = np.concatenate([ring, ring[:1]])
        segments = np.ascontiguousarray(np.repeat(ring, 2, axis=0)[1:-1])  # a, b, b, c, ...
        if self._outline is not None and _update_buffer(
            self._outline.geometry, "positions", segments
        ):
            return
        if self._outline is not None:
            self.scene.remove(self._outline)
        material = gfx.LineSegmentMaterial(thickness=2.5, color=color)
        for name, value in (("depth_test", False), ("render_queue", 3950)):
            try:
                setattr(material, name, value)
            except (AttributeError, ValueError, TypeError):
                pass
        self._outline = gfx.Line(gfx.Geometry(positions=segments), material)
        self.scene.add(self._outline)

    def plan_point(
        self, x: float, y: float, size: tuple[float, float]
    ) -> tuple[float, float] | None:
        """The model point (x, y) under a point of the view (pixels from the top left)
        in the 2D top view; None in any other view."""
        camera = self.camera
        if float(getattr(camera, "fov", 0.0) or 0.0) > 0:
            return None  # perspective: no plan to map to (orthographic is fov 0)
        direction = np.asarray(camera.world.forward, dtype=float)
        if abs(abs(direction[2]) - 1.0) > 1e-6:
            return None
        width, height = float(camera.width), float(camera.height)
        aspect = size[0] / max(size[1], 1)
        if width / max(height, 1e-12) < aspect:
            width = height * aspect
        else:
            height = width / aspect
        zoom = float(getattr(camera, "zoom", 1.0) or 1.0)
        width, height = width / zoom, height / zoom
        # the camera's own right and up (a 2D view may be turned about the vertical)
        try:
            right = np.asarray(camera.world.right, dtype=float)[:2]
            up = np.asarray(camera.world.up, dtype=float)[:2]
        except AttributeError:  # older pygfx: the 2D top view is not turned
            right, up = np.array([1.0, 0.0]), np.array([0.0, 1.0])
        centre = np.asarray(camera.world.position, dtype=float)[:2]
        u = (x / max(size[0], 1) - 0.5) * width
        v = (0.5 - y / max(size[1], 1)) * height
        point = centre + u * right + v * up
        return float(point[0]), float(point[1])

    def remove_gauges(self, item_id: str) -> None:
        group = self._gauges.pop(item_id, None)
        if group is not None:
            self.scene.remove(group)

    @property
    def gauge_ids(self) -> list[str]:
        return list(self._gauges)

    def remove(self, item_id: str) -> None:
        shown = self._shown.pop(item_id, None)
        if shown is not None:
            self.content.remove(shown.mesh)
        self._remove_stand(item_id)
        self._rims.pop(item_id, None)
        if shown is not None and shown.kind == "bathymetry":
            self._bed_heights = None
        self.update_contours(item_id, None)
        self.update_vectors(item_id, None)
        self.update_trails(item_id, None)
        self.update_outlines(item_id, None)

    def update_contours(self, item_id: str, lines: ContourLines | None) -> None:
        """Contour lines of an item (None or no segments: none), thin and dark."""
        self._update_overlay_lines(item_id, lines, 1.2)

    def update_vectors(self, item_id: str, arrows: VectorArrows | None) -> None:
        """Arrows of an item (None or no segments: none), in the contour colour."""
        self._update_overlay_lines(f"{item_id}#vectors", arrows, 1.4)

    def update_trails(self, item_id: str, trails: Trails | None) -> None:
        """Paths of an item's particles (None or no segments: none), thin, in its colour."""
        self._update_overlay_lines(
            f"{item_id}#trails", trails, 1.0, trails.color if trails is not None else None
        )

    def update_outlines(self, item_id: str, outlines) -> None:
        """Borders of an AMR surface's refined patches (None: none), in their colour."""
        self._update_overlay_lines(
            f"{item_id}#outlines", outlines, 1.3, outlines.color if outlines is not None else None
        )

    def _update_overlay_lines(
        self, key: str, lines, thickness: float, color: str | None = None
    ) -> None:
        line = self._contours.get(key)
        if lines is None or not lines.segments.size:
            if line is not None:
                self.scene.remove(line)
                del self._contours[key]
            return
        positions = np.array(lines.segments.reshape(-1, 3), np.float32)
        if line is not None and _update_buffer(line.geometry, "positions", positions):
            line.visible = lines.visible
            if color is not None:
                line.material.color = color
            return
        if line is not None:
            self.scene.remove(line)
        look = STYLES[self.style_name]
        material = gfx.LineSegmentMaterial(
            thickness=thickness, color=color or look.get("contours", "#1e2833")
        )
        line = gfx.Line(gfx.Geometry(positions=positions), material)
        line.visible = lines.visible
        self.scene.add(line)
        self._contours[key] = line

    def clear(self) -> None:
        for item_id in list(self._shown):
            self.remove(item_id)
        for item_id in list(self._gauges):
            self.remove_gauges(item_id)
        for item_id in list(self._setup):
            self.remove_setup(item_id)
        for item_id in list(self._backgrounds):
            self.remove_background(item_id)
        self.set_outline(None)
        for key in list(self._contours):
            self.scene.remove(self._contours.pop(key))
        self._framed = False
        self.set_marker(None)

    @property
    def item_ids(self) -> list[str]:
        return list(self._shown)

    @property
    def is_empty(self) -> bool:
        return not any(s.mesh.visible for s in self._shown.values()) and not any(
            g.visible for g in self._setup.values()
        )

    # ------------------------------------------------------------- camera
    def bounds(self, visible_only: bool = True) -> tuple[np.ndarray, np.ndarray] | None:
        """(low, high) corners of the shown data, from the vertex arrays (finite values).

        Computed here rather than by pygfx, so hidden items, markers, gauges and the
        lights never change what "frame all" shows.
        """
        lows, highs = [], []
        for shown in self._shown.values():
            if visible_only and not shown.mesh.visible:
                continue
            if shown.low is None:
                shown.low, shown.high = _corners(shown.mesh.geometry.positions.data)
            if shown.low is not None:
                lows.append(shown.low)
                highs.append(shown.high)
        for item_id, (low, high) in self._setup_bounds.items():
            group = self._setup.get(item_id)
            if group is not None and (group.visible or not visible_only):
                lows.append(low)
                highs.append(high)
        if not lows:
            return None
        return np.min(lows, axis=0).astype(float), np.max(highs, axis=0).astype(float)

    def frame(self, view: str = "perspective") -> None:
        """Point the camera at everything visible from a named direction (from the
        top: north up, when :meth:`set_north` gave north)."""
        direction, up = VIEWS[view]
        if view == "top" and self.north is not None:
            up = (self.north[0], self.north[1], 0.0)
        corners = self.bounds() or self.bounds(visible_only=False)
        if corners is None:
            return  # nothing to frame
        low, high = corners
        centre = (low + high) / 2
        radius = max(float(np.linalg.norm(high - low)) / 2, 1e-6)
        try:
            self.camera.show_object((*centre, radius), view_dir=direction, up=up)
        except (ValueError, TypeError):
            return
        self._fit_box(low, high, centre, np.asarray(direction, dtype=float))
        self._keep_z_up()
        self._framed = True

    def _fit_box(self, low, high, centre, direction) -> None:
        """Widen the view where the data's box does not fit.

        pygfx frames the bounding sphere into the mean of width and height, which in a
        wide window leaves the height short: a tall scene (strong vertical
        exaggeration) was cut off at the top and bottom. Flat scenes keep the view
        they had. Perspective is taken into account (the near side looks larger).
        """
        corners = np.array(
            [
                (x, y, z)
                for x in (low[0], high[0])
                for y in (low[1], high[1])
                for z in (low[2], high[2])
            ]
        )
        centre = np.asarray(centre, dtype=float)
        forward = direction / max(float(np.linalg.norm(direction)), 1e-12)
        inverse = np.asarray(self.camera.world.inverse_matrix, dtype=float)
        flat = np.abs(corners @ inverse[:3, :3].T + inverse[:3, 3])[:, :2].max(axis=0)
        if (
            2 * FRAME_MARGIN * flat[0] <= self.camera.width
            and 2 * FRAME_MARGIN * flat[1] <= self.camera.height
        ):
            return  # the box fits as pygfx framed it: flat scenes keep their view
        perspective = float(self.camera.fov) > 0
        for _ in range(6):  # moving back changes the perspective a little: repeat
            inverse = np.asarray(self.camera.world.inverse_matrix, dtype=float)
            seen = corners @ inverse[:3, :3].T + inverse[:3, 3]
            if perspective:
                to_centre = float(np.linalg.norm(np.asarray(self.camera.world.position) - centre))
                depth = np.maximum(-seen[:, 2], 1e-3 * max(to_centre, 1e-9))
                seen = seen[:, :2] * (to_centre / depth)[:, None]
            half = np.abs(seen[:, :2]).max(axis=0)
            width = max(float(self.camera.width), 2 * FRAME_MARGIN * float(half[0]))
            height = max(float(self.camera.height), 2 * FRAME_MARGIN * float(half[1]))
            if width <= self.camera.width * 1.0001 and height <= self.camera.height * 1.0001:
                return  # fits
            self.camera.width, self.camera.height = width, height
            extent = 0.5 * (width + height)
            self.camera.world.position = tuple(
                centre - forward * _fov_distance(self.camera.fov) * extent
            )
            self.camera.depth = extent

    def _keep_z_up(self) -> None:
        """pygfx makes the ``up`` of the last named view the axis the view orbits
        about; after the top view that was y, and turning then tipped the scene over.
        Orbiting is always about z, the vertical."""
        try:
            self.camera.world.reference_up = (0.0, 0.0, 1.0)
        except AttributeError:
            pass

    def diagnostics(self) -> dict:
        """What the renderer holds, for the developer remote: items, bounds, camera, sun."""
        items = {}
        for item_id, shown in self._shown.items():
            positions = shown.mesh.geometry.positions.data
            finite = np.isfinite(positions).all(axis=1)
            items[item_id] = {
                "type": type(shown.mesh).__name__,
                "visible": bool(shown.mesh.visible),
                "vertices": int(positions.shape[0]),
                "not_finite": int((~finite).sum()),
                "low": positions[finite].min(axis=0).round(3).tolist() if finite.any() else None,
                "high": positions[finite].max(axis=0).round(3).tolist() if finite.any() else None,
                "pygfx_bounds": np.asarray(shown.mesh.get_world_bounding_box()).round(3).tolist(),
            }
        shadow = getattr(self.sun, "shadow", None)
        return {
            "style": self.style_name,
            "items": items,
            "gauges": self.gauge_ids,
            "bounds": [b.round(3).tolist() for b in self.bounds() or ()],
            "camera": {
                k: v
                for k, v in self.camera_state().items()
                if k in ("position", "width", "height", "fov")
            },
            "sun": {
                "position": np.asarray(self.sun.local.position).round(3).tolist(),
                "target": np.asarray(self.sun.target.local.position).round(3).tolist(),
                "cast_shadow": bool(self.sun.cast_shadow),
                "shadow_width": getattr(getattr(shadow, "camera", None), "width", None),
                "shadow_depth": list(getattr(getattr(shadow, "camera", None), "depth_range", ())),
            },
        }

    def pan(self, right: float, up: float) -> None:
        """Move the view sideways and up/down by fractions of what it shows."""
        state = self.camera.get_state()
        extent = float(state.get("width") or 1.0)
        matrix = np.asarray(self.camera.world.matrix, dtype=float)
        axes = matrix[:3, :2] / np.maximum(np.linalg.norm(matrix[:3, :2], axis=0), 1e-12)
        shift = axes @ np.array([right, up]) * extent
        self.camera.world.position = tuple(np.asarray(self.camera.world.position) + shift)

    def frame_once(self, view: str = "perspective") -> None:
        """Frame the scene the first time something is shown."""
        if not self._framed and (self._shown or self._setup_bounds):
            self.frame(view)

    def set_north(self, north) -> bool:
        """Which way north is in model coordinates (a unit (x, y) vector; None: the
        model's y), so the view from the top shows the map north up and a turned
        model turned. Turns a camera that looks straight down at once. True if it
        changed."""
        north = None if north is None else (float(north[0]), float(north[1]))
        if north is not None and abs(north[0]) < 1e-9 and north[1] > 0:
            north = None  # the model's y is north: as without
        if north == self.north:
            self._north_known = True
            return False
        before = _north_angle(self.north)
        self.north = north
        forward = np.asarray(self.camera.world.forward, dtype=float)
        if abs(abs(forward[2]) - 1.0) < 1e-6:  # looking straight down: turn about z
            angle = _north_angle(north)
            self.camera.local.rotation = (0.0, 0.0, math.sin(angle / 2), math.cos(angle / 2))
        elif self._north_known:
            # in 3D the camera turns with north about the vertical through the middle of
            # the model: the map stays where it was, a turned model turns
            self._turn_camera(_north_angle(north) - before)
        self._north_known = True
        return True

    def point_north(self) -> None:
        """Turn the view so north (the model's y without a georeference) points up the
        screen: from the top straight up, in 3D away from the viewer, the tilt kept;
        about the point at the middle of the view."""
        nx, ny = self.north if self.north is not None else (0.0, 1.0)
        forward = np.asarray(self.camera.world.forward, dtype=float)
        if abs(abs(forward[2]) - 1.0) < 1e-6:  # straight down: north up the screen
            angle = _north_angle(self.north)
            self.camera.local.rotation = (0.0, 0.0, math.sin(angle / 2), math.cos(angle / 2))
            return
        # the screen's up on the ground (for a view from above, where it looks; for a
        # view along the ground, where it looks)
        up = np.asarray(self.camera.world.up, dtype=float)
        heading = up[:2] if np.hypot(*up[:2]) > 1e-3 else forward[:2]
        if np.hypot(*heading) < 1e-9:
            return
        angle = math.atan2(ny, nx) - math.atan2(heading[1], heading[0])
        self._turn_camera(math.remainder(angle, 2 * math.pi), self._view_centre())

    def _view_centre(self) -> np.ndarray:
        """Where the middle of the view meets the level of the middle of the model."""
        corners = self.bounds() or self.bounds(visible_only=False)
        middle = (corners[0] + corners[1]) / 2 if corners is not None else np.zeros(3)
        position = np.asarray(self.camera.local.position, dtype=float)
        forward = np.asarray(self.camera.world.forward, dtype=float)
        if abs(forward[2]) < 1e-6:
            return middle
        reach = (middle[2] - position[2]) / forward[2]
        return position + max(reach, 0.0) * forward if reach > 0 else middle

    def _turn_camera(self, angle: float, pivot=None) -> None:
        """Turn the camera about the vertical through ``pivot`` (default: the middle
        of what is shown)."""
        if abs(angle) < 1e-12:
            return
        if pivot is None:
            corners = self.bounds() or self.bounds(visible_only=False)
            pivot = (corners[0] + corners[1]) / 2 if corners is not None else np.zeros(3)
        pivot = np.asarray(pivot, dtype=float)
        c, s = math.cos(angle), math.sin(angle)
        position = np.asarray(self.camera.local.position, dtype=float) - pivot
        position[:2] = (c * position[0] - s * position[1], s * position[0] + c * position[1])
        self.camera.local.position = tuple(position + pivot)
        turn = np.array([0.0, 0.0, math.sin(angle / 2), math.cos(angle / 2)])
        self.camera.local.rotation = tuple(_quaternion_product(turn, self.camera.local.rotation))

    def set_plan(self, on: bool) -> None:
        """2D: the items are flat layers seen from above; the sun's shadows would only
        darken the lower layers, so they are off. The headlamp shines as in the 3D
        view, not straight down."""
        self.plan = bool(on)
        look = STYLES[self.style_name]
        self.sun.cast_shadow = bool(look.get("shadows", False)) and not self.plan
        self._shadow_fit = None
        self._fit_shadow()
        self._place_headlamp()
        self._refresh_stand()  # no stand in 2D

    def _place_headlamp(self) -> None:
        """On the camera in 3D (it shines along the view); in 2D from the top fixed in
        the scene, shining as in the 3D view, so the light is the same."""
        lamp = self.headlamp
        home = self.scene if self.plan else self.camera
        if lamp.parent is not home:
            if lamp.parent is not None:
                lamp.parent.remove(lamp)
            home.add(lamp)
        lamp.local.position = PLAN_HEADLAMP if self.plan else HEADLAMP

    def set_orthographic(self, orthographic: bool) -> None:
        self.camera.fov = 0 if orthographic else 45

    @property
    def orthographic(self) -> bool:
        return self.camera.fov == 0

    # ------------------------------------------------------------- camera state
    def camera_state(self) -> dict:
        """The camera as plain JSON-ready values (saved with the session)."""
        state = self.camera.get_state()
        return {
            key: (np.asarray(value).tolist() if not isinstance(value, bool | str) else value)
            for key, value in state.items()
            if isinstance(value, int | float | bool | str | tuple | list | np.ndarray)
        }

    def set_camera_state(self, state: dict) -> None:
        clean = {k: (tuple(v) if isinstance(v, list) else v) for k, v in state.items()}
        self._north_known = False  # the camera as it was saved, north and all
        try:
            self.camera.set_state(clean)
            self._keep_z_up()
            self._framed = True
        except (KeyError, TypeError, ValueError):
            self.frame()

    def render(self, renderer: gfx.WgpuRenderer) -> None:
        self.apply_effects(renderer)
        renderer.render(self.scene, self.camera)

    def _set_fog(self, fog: tuple[str, float]) -> None:
        """The fog for the camera as it is now: clear up to the scene, half thick
        ``fog[1]`` scene sizes into it."""
        from lagoonviz.render.cinematic import fog_settings

        corners = self.bounds(visible_only=True)
        if corners is None:
            corners = (np.zeros(3), np.ones(3))
        position = np.asarray(self.camera.world.position, float)
        start, density = fog_settings(fog[1], position, *corners)
        near = float(getattr(self.camera, "near", 0.1) or 0.1)
        far = float(getattr(self.camera, "far", 1000.0) or 1000.0)
        self._fog.set(fog[0], near, far, start, density)

    def apply_effects(self, renderer) -> None:
        """The look's post effects on a renderer: the depth fog of the underwater look,
        before the renderer's anti-aliasing; nothing for the other looks."""
        passes = getattr(renderer, "effect_passes", None)
        if passes is None:
            return
        fog = self.preset.fog if self.preset is not None else None
        kept = tuple(p for p in passes if p is not self._fog)
        if fog is None:
            if len(kept) != len(passes):
                renderer.effect_passes = kept
            return
        if self._fog is None:
            try:
                from lagoonviz.render.cinematic import depth_fog_class

                self._fog = depth_fog_class()(fog[0])
            except (AttributeError, ImportError):  # an older pygfx: no fog
                return
        self._set_fog(fog)
        if passes[:1] != (self._fog,):
            renderer.effect_passes = (self._fog, *kept)
