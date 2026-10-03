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

from lagoonviz.core.colormaps import lut, normalize
from lagoonviz.core.data.manager import DisplaySurface

#: viewport background as pygfx expects it: bottom colour, then top (LAGOON blue)
BACKGROUND = ("#1c3354", "#3b5f8f")
#: colour of surfaces shown without a variable
PLAIN = {"free_surface": "#4f8fcf", "bathymetry": "#b39c72", "volume": "#9aa7b4"}

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


def _background() -> gfx.Background:
    if hasattr(gfx.Background, "from_color"):
        return gfx.Background.from_color(*BACKGROUND)
    return gfx.Background(None, gfx.BackgroundMaterial(*BACKGROUND))


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

    def __init__(self) -> None:
        self.scene = gfx.Scene()
        self.scene.add(_background())
        self.scene.add(gfx.AmbientLight(intensity=0.55))
        self.camera = gfx.PerspectiveCamera(45, 16 / 9)
        try:
            self.camera.world.reference_up = (0, 0, 1)
        except AttributeError:  # older pygfx: up is given per call
            pass
        light = gfx.DirectionalLight(intensity=2.2)
        light.local.position = (0.4, -0.6, 1.0)
        self.camera.add(light)  # the light moves with the camera, like a headlamp
        self.scene.add(self.camera)
        self.content = gfx.Group()
        self.scene.add(self.content)
        self._shown: dict[str, _Shown] = {}
        self._textures: dict[str, gfx.Texture] = {}
        self._framed = False
        self._marker: gfx.Mesh | None = None

    # ------------------------------------------------------------- items
    def _texture(self, colormap: str) -> gfx.Texture:
        if colormap not in self._textures:
            self._textures[colormap] = gfx.Texture(lut(colormap), dim=1)
        return self._textures[colormap]

    def _material(self, kind: str, shown: DisplaySurface) -> gfx.Material:
        material = gfx.MeshPhongMaterial(side="both", shininess=20)
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
            material.color = PLAIN.get(kind, "#cccccc")
        material.opacity = shown.opacity
        # transparent surfaces must not hide what lies behind them
        if hasattr(material, "alpha_mode"):
            try:
                material.alpha_mode = "blend" if shown.opacity < 1.0 else "solid"
            except (ValueError, TypeError):
                pass

    def update(self, kind: str, shown: DisplaySurface) -> None:
        """Create or update the mesh of an item from its display data."""
        normals = shown.normals
        texcoords = (
            normalize(shown.values, shown.vmin, shown.vmax)
            if shown.values is not None
            else np.zeros(shown.positions.shape[0], np.float32)
        )
        current = self._shown.get(shown.item_id)
        same_topology = current is not None and current.triangles_id == id(shown.triangles)
        if same_topology:
            geometry = current.mesh.geometry
            ok = (
                _update_buffer(geometry, "positions", shown.positions)
                and _update_buffer(geometry, "normals", normals)
                and _update_buffer(geometry, "texcoords", texcoords)
            )
            same_topology = ok
        if not same_topology:
            if current is not None:
                self.content.remove(current.mesh)
            geometry = gfx.Geometry(
                positions=shown.positions,
                indices=np.ascontiguousarray(shown.triangles, dtype=np.uint32),
                normals=normals,
                texcoords=texcoords,
            )
            mesh = gfx.Mesh(geometry, self._material(kind, shown))
            self.content.add(mesh)
            current = _Shown(mesh, shown.colormap, id(shown.triangles), shown.values is not None)
            self._shown[shown.item_id] = current
        else:
            self._style(current.mesh.material, kind, shown)
        current.mesh.visible = shown.visible
        current.colormap = shown.colormap
        current.has_values = shown.values is not None

    def set_visible(self, item_id: str, visible: bool) -> None:
        if item_id in self._shown:
            self._shown[item_id].mesh.visible = visible

    def item_at(self, world_object) -> str | None:
        """The item a picked pygfx object belongs to."""
        for item_id, shown in self._shown.items():
            if shown.mesh is world_object:
                return item_id
        return None

    def set_marker(self, point: tuple[float, float, float] | None) -> None:
        """Show (or hide) the orange probe marker at a point."""
        if point is None:
            if self._marker is not None:
                self._marker.visible = False
            return
        if self._marker is None:
            radius = 1.0
            box = self.content.get_world_bounding_box()
            if box is not None:
                radius = max(float(np.linalg.norm(np.asarray(box[1]) - np.asarray(box[0]))), 1e-6)
                radius *= 0.006
            self._marker = gfx.Mesh(
                gfx.sphere_geometry(radius, 16, 12), gfx.MeshBasicMaterial(color="#e07b39")
            )
            self.scene.add(self._marker)
        self._marker.local.position = tuple(point)
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
