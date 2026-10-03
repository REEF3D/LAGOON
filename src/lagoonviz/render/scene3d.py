# SPDX-License-Identifier: GPL-3.0-or-later
"""The 3D scene: LAGOON items drawn with pygfx (WebGPU: Metal, Vulkan or Direct3D 12).

This module knows nothing about Qt. The GUI gives it a canvas to draw on; the
offscreen renderer gives it an image canvas. Both draw the same scene, so a view
on screen and an image made headless on a cluster look the same.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pygfx as gfx

from lagoonviz.core.colormaps import colors, lut, normalize
from lagoonviz.core.data.manager import DisplaySurface, GaugeMarks

#: Viewport looks. "studio" is the light look of the REEF3D result animations:
#: white background, soft light from above, matt surfaces. "ocean" is dark blue.
STYLES = {
    "studio": {
        "background": ("#e6eaef", "#ffffff"),  # bottom, top (pygfx order)
        # pygfx lights are physical (diffuse = colour / pi): this mix shows flat water
        # close to its colour map colour, shades slopes and lets the sun cast shadows
        "ambient": 1.0,
        "sun": 3.0,  # fixed light from above, low enough to model the waves
        "headlamp": 0.2,  # light that moves with the camera (more of it flattens)
        "shininess": 16,
        "specular": "#2a2a2a",
        # water is glossier: a soft sun sheen on the slopes that face it
        "water_shininess": 40,
        "water_specular": "#3a3a3a",
        "shadows": True,
    },
    "ocean": {
        "background": ("#1c3354", "#3b5f8f"),
        "ambient": 0.55,
        "sun": 0.6,
        "headlamp": 2.0,
        "shininess": 20,
        "specular": "#494949",
        "water_shininess": 40,
        "water_specular": "#5a5a5a",
        "shadows": True,
    },
}
DEFAULT_STYLE = "studio"

#: where the sun stands, seen from the scene: about 31 degrees above the horizon and
#: mostly from -x, across the crests of waves that travel along x (REEF3D's usual
#: set-up), so the slopes facing it and facing away differ clearly
SUN_DIRECTION = np.array((-0.9, -0.4, 0.6)) / np.linalg.norm((-0.9, -0.4, 0.6))
#: shadow map depth bias, as a fraction of the shadow camera's depth range
SHADOW_BIAS = 0.0008

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


def _background(colors: tuple[str, str]) -> gfx.Background:
    if hasattr(gfx.Background, "from_color"):
        return gfx.Background.from_color(*colors)
    return gfx.Background(None, gfx.BackgroundMaterial(*colors))


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


class SceneRenderer:
    """Keeps a pygfx scene in sync with the display data of LAGOON items."""

    def __init__(self, style: str = DEFAULT_STYLE) -> None:
        self.style_name = style if style in STYLES else DEFAULT_STYLE
        look = STYLES[self.style_name]
        self.scene = gfx.Scene()
        self.background = _background(look["background"])
        self.scene.add(self.background)
        self.ambient = gfx.AmbientLight(intensity=look["ambient"])
        self.scene.add(self.ambient)
        # a sun from above, fixed in the scene: wave crests read the same from any side
        self.sun = gfx.DirectionalLight(intensity=look["sun"])
        self.sun.local.position = tuple(SUN_DIRECTION)
        self.sun.cast_shadow = bool(look.get("shadows", False))
        self.scene.add(self.sun)
        self._shadow_fit: tuple | None = None
        self.camera = gfx.PerspectiveCamera(45, 16 / 9)
        try:
            self.camera.world.reference_up = (0, 0, 1)
        except AttributeError:  # older pygfx: up is given per call
            pass
        self.headlamp = gfx.DirectionalLight(intensity=look["headlamp"])
        self.headlamp.local.position = (0.4, -0.6, 1.0)
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

    def _material(self, kind: str, shown: DisplaySurface) -> gfx.Material:
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
        water = kind == "free_surface"
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
            material.color = shown.color
        material.opacity = shown.opacity
        # transparent surfaces must not hide what lies behind them
        if hasattr(material, "alpha_mode"):
            try:
                material.alpha_mode = "blend" if shown.opacity < 1.0 else "solid"
            except (ValueError, TypeError):
                pass

    def update(self, kind: str, shown: DisplaySurface) -> None:
        """Create or update the mesh (or lines) of an item from its display data."""
        if shown.segments is not None and shown.triangles.size == 0:
            self._update_lines(kind, shown)
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
        self._fit_shadow()

    def _fit_shadow(self) -> None:
        """Aim the sun's shadow map at what is shown: it covers the scene's bounding
        sphere, seen from the sun, so any domain size gets the full map resolution."""
        if not self.sun.cast_shadow:
            return
        corners = self.bounds(visible_only=False)
        if corners is None:
            return
        low, high = corners
        centre = (low + high) / 2
        radius = max(float(np.linalg.norm(high - low)) / 2, 1e-6)
        fit = (*np.round(centre, 3), round(radius, 3))
        if fit == self._shadow_fit:
            return
        self._shadow_fit = fit
        self.sun.target.local.position = tuple(centre)
        self.sun.local.position = tuple(centre + SUN_DIRECTION * 2 * radius)
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

    def set_visible(self, item_id: str, visible: bool) -> None:
        if item_id in self._shown:
            self._shown[item_id].mesh.visible = visible

    def set_style(self, style: str) -> None:
        """Switch the look (background and lights) of the viewport."""
        self.style_name = style if style in STYLES else DEFAULT_STYLE
        look = STYLES[self.style_name]
        self.scene.remove(self.background)
        self.background = _background(look["background"])
        self.scene.add(self.background)
        self.ambient.intensity = look["ambient"]
        self.sun.intensity = look["sun"]
        self.sun.cast_shadow = bool(look.get("shadows", False))
        self._shadow_fit = None
        self._fit_shadow()
        self.headlamp.intensity = look["headlamp"]
        for shown in self._shown.values():
            material = shown.mesh.material
            if hasattr(material, "shininess"):
                self._gloss(material, shown.kind)

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
            material = gfx.PointsMaterial(size=14, color="#e07b39")
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
        staffs = gfx.Line(
            gfx.Geometry(positions=staff_positions, colors=staff_colors),
            gfx.LineSegmentMaterial(thickness=2.5, color_mode="vertex"),
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

    def clear(self) -> None:
        for item_id in list(self._shown):
            self.remove(item_id)
        for item_id in list(self._gauges):
            self.remove_gauges(item_id)
        self._framed = False
        self.set_marker(None)

    @property
    def item_ids(self) -> list[str]:
        return list(self._shown)

    @property
    def is_empty(self) -> bool:
        return not any(s.mesh.visible for s in self._shown.values())

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
        if not lows:
            return None
        return np.min(lows, axis=0).astype(float), np.max(highs, axis=0).astype(float)

    def frame(self, view: str = "perspective") -> None:
        """Point the camera at everything visible from a named direction."""
        direction, up = VIEWS[view]
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
        self._framed = True

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

    def frame_once(self) -> None:
        """Frame the scene the first time something is shown."""
        if not self._framed and self._shown:
            self.frame("perspective")

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
        try:
            self.camera.set_state(clean)
            self._framed = True
        except (KeyError, TypeError, ValueError):
            self.frame()

    def render(self, renderer: gfx.WgpuRenderer) -> None:
        renderer.render(self.scene, self.camera)
