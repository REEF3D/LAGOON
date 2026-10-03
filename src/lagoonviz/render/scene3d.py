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
from lagoonviz.core.data.manager import DisplaySurface

#: Viewport looks. "studio" is the light look of the REEF3D result animations:
#: white background, soft light from above, matt surfaces. "ocean" is dark blue.
STYLES = {
    "studio": {
        "background": ("#e6eaef", "#ffffff"),  # bottom, top (pygfx order)
        "ambient": 0.75,
        "sun": 1.5,  # fixed light from above
        "headlamp": 0.9,  # light that moves with the camera
        "shininess": 12,
        "specular": "#2a2a2a",
    },
    "ocean": {
        "background": ("#1c3354", "#3b5f8f"),
        "ambient": 0.55,
        "sun": 0.6,
        "headlamp": 2.0,
        "shininess": 20,
        "specular": "#494949",
    },
}
DEFAULT_STYLE = "studio"

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
    colormap: str
    triangles_id: int
    has_values: bool


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
        self.sun.local.position = (-0.4, -0.7, 1.0)
        self.scene.add(self.sun)
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
        self._textures: dict[str, gfx.Texture] = {}
        self._framed = False
        self._marker: gfx.Points | None = None

    # ------------------------------------------------------------- items
    def _texture(self, colormap: str) -> gfx.Texture:
        if colormap not in self._textures:
            self._textures[colormap] = gfx.Texture(lut(colormap), dim=1)
        return self._textures[colormap]

    def _material(self, kind: str, shown: DisplaySurface) -> gfx.Material:
        look = STYLES[self.style_name]
        material = gfx.MeshPhongMaterial(side="both", shininess=look["shininess"])
        try:
            material.specular = look["specular"]
        except (AttributeError, ValueError):
            pass
        try:
            material.pick_write = True  # clicks report which triangle was hit
        except AttributeError:
            pass
        self._style(material, kind, shown)
        return material

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
            normalize(shown.values, shown.vmin, shown.vmax)
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
                    current.triangles_id == id(shown.triangles)
                    or _update_buffer(geometry, "indices", indices)
                )
            )
        if not same_sizes:
            if current is not None:
                self.content.remove(current.mesh)
            geometry = gfx.Geometry(
                positions=shown.positions, indices=indices, normals=normals, texcoords=texcoords
            )
            mesh = gfx.Mesh(geometry, self._material(kind, shown))
            self.content.add(mesh)
            current = _Shown(mesh, shown.colormap, id(shown.triangles), shown.values is not None)
            self._shown[shown.item_id] = current
        else:
            self._style(current.mesh.material, kind, shown)
        current.triangles_id = id(shown.triangles)
        current.mesh.visible = shown.visible
        current.colormap = shown.colormap
        current.has_values = shown.values is not None

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
            current = _Shown(line, shown.colormap, id(shown.segments), shown.values is not None)
            self._shown[shown.item_id] = current
        current.mesh.material.thickness = shown.line_width
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
        self.headlamp.intensity = look["headlamp"]
        for shown in self._shown.values():
            material = shown.mesh.material
            if hasattr(material, "shininess"):
                material.shininess = look["shininess"]

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

    def remove(self, item_id: str) -> None:
        shown = self._shown.pop(item_id, None)
        if shown is not None:
            self.content.remove(shown.mesh)

    def clear(self) -> None:
        for item_id in list(self._shown):
            self.remove(item_id)
        self._framed = False
        self.set_marker(None)

    @property
    def item_ids(self) -> list[str]:
        return list(self._shown)

    @property
    def is_empty(self) -> bool:
        return not any(s.mesh.visible for s in self._shown.values())

    # ------------------------------------------------------------- camera
    def frame(self, view: str = "perspective") -> None:
        """Point the camera at everything visible from a named direction."""
        direction, up = VIEWS[view]
        target = self.content if self._shown else self.scene
        try:
            self.camera.show_object(target, view_dir=direction, up=up)
        except (ValueError, TypeError):  # empty scene: nothing to frame
            return
        self._framed = True

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
