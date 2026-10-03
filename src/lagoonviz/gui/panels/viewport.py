# SPDX-License-Identifier: GPL-3.0-or-later
"""Central 3D viewport: a pygfx canvas inside Qt, with Blender-style navigation.

Navigation (mouse and trackpad):

* rotate: left drag, or middle drag (Blender)
* pan: right drag, or Shift + middle drag (Blender)
* zoom: wheel or pinch; Ctrl + middle drag
* views: 1 front, 3 right, 7 top (Ctrl for the opposite side), 5 perspective or
  orthographic, Home or . to frame everything

Frames are read in background threads; the viewport shows the newest frame that
has arrived, so scrubbing never blocks the window. Without a usable GPU (or in
tests, which run without a screen) a placeholder takes the viewport's place.
"""

from __future__ import annotations

import logging
import os
import traceback

import numpy as np
from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtGui import QColor, QLinearGradient, QPainter, QPixmap
from PySide6.QtWidgets import QHBoxLayout, QLabel, QToolButton, QVBoxLayout, QWidget

from lagoonviz.core.data.manager import BackgroundLoader, FrameKey
from lagoonviz.gui.bridge import SessionBridge
from lagoonviz.gui.panels.colorbar import ColorBarEntry, ColorBarStrip
from lagoonviz.gui.perf import PERF
from lagoonviz.gui.resources import resource_path
from lagoonviz.gui.theme import COLORS, SERIES_COLORS

log = logging.getLogger("lagoon.viewport")

TOP = QColor(COLORS["viewport_top"])
BOTTOM = QColor(COLORS["viewport_bottom"])


class ViewportPlaceholder(QWidget):
    """Shown when no GPU canvas is available (tests, remote sessions without GPU)."""

    statusMessage = Signal(str)

    def __init__(self, parent: QWidget | None = None, reason: str = ""):
        super().__init__(parent)
        self.setObjectName("viewport")
        self.setMinimumSize(320, 240)
        self._icon = QPixmap(str(resource_path("viewport_mark.png")))
        self.reason = reason
        self.message = "3D viewport" + (f"\n{reason}" if reason else "")

    # the same small interface as Viewport3D
    @property
    def busy(self) -> bool:
        return False

    def frame(self, view: str = "perspective") -> None:
        pass

    def reframe(self) -> None:
        pass

    def refresh(self, item_ids=None) -> None:
        self.update()

    def set_style(self, style: str) -> None:
        pass

    def canvas_snapshot(self):
        return None

    probe = None

    def probe_at(self, item_id: str, triangle: int, weights: tuple[float, ...]) -> None:
        pass

    def clear_probe(self) -> None:
        pass

    def paintEvent(self, event) -> None:  # noqa: N802 (Qt naming)
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        gradient = QLinearGradient(0, 0, 0, self.height())
        gradient.setColorAt(0.0, TOP)
        gradient.setColorAt(1.0, BOTTOM)
        painter.fillRect(self.rect(), gradient)
        size = min(160, self.height() // 3, self.width() // 3)
        if not self._icon.isNull() and size > 16:
            icon = self._icon.scaled(size, size, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            painter.setOpacity(0.45)
            painter.drawPixmap((self.width() - size) // 2, self.height() // 2 - size, icon)
            painter.setOpacity(1.0)
        painter.setPen(QColor(COLORS["dim"]))
        text_rect = self.rect().adjusted(20, self.height() // 2 + 12, -20, 0)
        painter.drawText(text_rect, Qt.AlignHCenter | Qt.AlignTop | Qt.TextWordWrap, self.message)
        painter.end()


def gpu_unavailable_reason() -> str | None:
    """Why the GPU viewport cannot be used here, or None if it can be tried."""
    if os.environ.get("LAGOON_NO_GPU"):
        return "GPU viewport switched off (LAGOON_NO_GPU)"
    if os.environ.get("QT_QPA_PLATFORM", "") in ("offscreen", "minimal"):
        return "no screen (offscreen Qt platform)"
    try:
        import pygfx  # noqa: F401
        import rendercanvas.qt  # noqa: F401
    except Exception as error:  # noqa: BLE001 (any import problem means: no viewport)
        return f"pygfx is not available: {error}"
    return None


def create_viewport(
    bridge: SessionBridge, loader: BackgroundLoader, parent: QWidget, style: str = "studio"
) -> QWidget:
    """The GPU viewport, or a placeholder that says why there is none."""
    reason = gpu_unavailable_reason()
    if reason is None:
        try:
            return Viewport3D(bridge, loader, parent, style)
        except Exception as error:  # noqa: BLE001
            log.error("could not create the 3D viewport:\n%s", traceback.format_exc())
            reason = f"3D viewport failed to start: {error}"
    log.info("viewport placeholder: %s", reason)
    return ViewportPlaceholder(parent, reason)


#: Blender-like controls on top of the trackpad-friendly defaults
BLENDER_CONTROLS = {
    "mouse1": ("rotate", "drag", (0.005, 0.005)),
    "mouse2": ("pan", "drag", (1.0, 1.0)),
    "mouse3": ("rotate", "drag", (0.005, 0.005)),
    "shift+mouse3": ("pan", "drag", (1.0, 1.0)),
    "shift+mouse1": ("pan", "drag", (1.0, 1.0)),
    # Cmd (Ctrl elsewhere) + drag moves the view; on a Mac, Ctrl + click is a right
    # click, which pans too
    "control+mouse1": ("pan", "drag", (1.0, 1.0)),
    "control+mouse3": ("zoom", "drag", (0.005, 0.005)),
    "wheel": ("zoom", "push", -0.001),
}

KEY_VIEWS = {"1": "front", "3": "right", "7": "top"}
#: arrow keys move the view (a tenth of it; with Shift a third)
KEY_PANS = {"ArrowLeft": (-1, 0), "ArrowRight": (1, 0), "ArrowUp": (0, 1), "ArrowDown": (0, -1)}
OPPOSITE = {"front": "back", "right": "left", "top": "bottom"}


class Viewport3D(QWidget):
    """pygfx canvas plus a small view toolbar and the colour bars."""

    statusMessage = Signal(str)
    #: (key, mesh, error) from a loader thread, delivered on the GUI thread
    _loaded = Signal(object, object, object)

    def __init__(
        self,
        bridge: SessionBridge,
        loader: BackgroundLoader,
        parent: QWidget | None,
        style: str = "studio",
    ):
        super().__init__(parent)
        import pygfx as gfx
        from rendercanvas.qt import QRenderWidget

        from lagoonviz.render.scene3d import SceneRenderer

        self.setObjectName("viewport")
        self.setMinimumSize(320, 240)
        self.bridge = bridge
        self.session = bridge.session
        self.loader = loader
        self._wanted: dict[str, FrameKey] = {}
        self._shown_keys: dict[str, FrameKey] = {}

        try:
            self.canvas = QRenderWidget(self, update_mode="ondemand")
        except TypeError:  # rendercanvas without update modes
            self.canvas = QRenderWidget(self)
        self.renderer = gfx.WgpuRenderer(self.canvas)
        self.view = SceneRenderer(style)
        self.controller = gfx.OrbitController(self.view.camera, register_events=self.renderer)
        for control, action in BLENDER_CONTROLS.items():
            try:
                self.controller.controls[control] = action
            except Exception as error:  # noqa: BLE001 (keep the pygfx default for this one)
                log.warning("control %s not set: %s", control, error)
        self.canvas.request_draw(self._draw)
        self.renderer.add_event_handler(self._on_key, "key_down")
        self.renderer.add_event_handler(self._on_interaction, "pointer_up", "wheel")
        self.renderer.add_event_handler(self._on_pointer, "pointer_down", "pointer_up")
        self._press: tuple[float, float, object, dict] | None = None
        self._probe_at: tuple[str, int, tuple[float, ...]] | None = None
        #: the last probe (values where the user clicked), shown in the header
        self.probe = None

        self.colorbars = ColorBarStrip(self)
        self.info = QLabel()
        bar = QHBoxLayout()
        bar.setContentsMargins(6, 2, 6, 2)
        bar.setSpacing(2)
        self.view_buttons: dict[str, QToolButton] = {}
        for key, text, tip in (
            ("frame", "Frame all", "Frame everything (Home)"),
            ("perspective", "3D", "Perspective view"),
            ("top", "Top", "Top view (7)"),
            ("front", "Front", "Front view (1)"),
            ("right", "Side", "Side view (3)"),
        ):
            button = QToolButton()
            button.setText(text)
            button.setToolTip(tip)
            button.setAutoRaise(True)
            button.clicked.connect(
                lambda _=False, k=key: self.frame("perspective" if k == "frame" else k)
            )
            bar.addWidget(button)
            self.view_buttons[key] = button
        self.ortho_button = QToolButton()
        self.ortho_button.setText("Ortho")
        self.ortho_button.setCheckable(True)
        self.ortho_button.setAutoRaise(True)
        self.ortho_button.setToolTip("Orthographic projection (5)")
        self.ortho_button.toggled.connect(self._set_ortho)
        bar.addWidget(self.ortho_button)
        bar.addStretch(1)
        bar.addWidget(self.info)
        header = QWidget()
        header.setLayout(bar)
        header.setObjectName("viewportHeader")
        self.header = header
        self.apply_theme_colors()

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(header)
        layout.addWidget(self.canvas, 1)
        layout.addWidget(self.colorbars)

        self._save_camera = QTimer(self)
        self._save_camera.setSingleShot(True)
        self._save_camera.setInterval(400)
        self._save_camera.timeout.connect(self._store_camera)

        self._loaded.connect(self._on_loaded)
        bridge.sceneChanged.connect(self._on_scene)
        bridge.timeChanged.connect(lambda _t: self.refresh())
        if self.session.scene.camera:
            self.view.set_camera_state(self.session.scene.camera)
        self.refresh()

    # ------------------------------------------------------------- state
    @property
    def busy(self) -> bool:
        """True while frames for the current time are still loading."""
        return any(self._wanted.get(i) != self._shown_keys.get(i) for i in self._wanted)

    def _draw(self) -> None:
        with PERF.track("render"):
            self.view.render(self.renderer)

    def redraw(self) -> None:
        self.canvas.request_draw()

    # ------------------------------------------------------------- data
    def refresh(self, item_ids: list[str] | None = None) -> None:
        """Bring the shown meshes up to date with the scene and the current time."""
        data = self.session.data
        current = set(data.data_items())
        for item_id in list(self.view.item_ids):
            if item_id not in current:
                self.view.remove(item_id)
                self._shown_keys.pop(item_id, None)
                self._wanted.pop(item_id, None)
        for item_id in item_ids if item_ids is not None else sorted(current):
            if item_id not in current:
                continue
            key = data.frame_key(item_id)
            if key is None:
                self.view.remove(item_id)
                self._wanted.pop(item_id, None)
                self._shown_keys.pop(item_id, None)
                error = data.error(item_id)
                if error:
                    self.statusMessage.emit(f"{item_id}: {error}")
                continue
            self._wanted[item_id] = key
            mesh = data.cache.get(key)
            if mesh is not None:
                self._show(item_id, key, mesh)
            else:
                self.loader.request(key, self._loaded.emit)
        self._update_gauges()
        self._update_overlays()
        self.redraw()

    def _update_gauges(self) -> None:
        with PERF.track("update gauges"):
            self.__update_gauges_now()

    def __update_gauges_now(self) -> None:
        """Wave gauge positions, in the colours of their plot lines."""
        data = self.session.data
        wanted = set(data.series_items())
        for item_id in self.view.gauge_ids:
            if item_id not in wanted:
                self.view.remove_gauges(item_id)
        for item_id in wanted:
            try:
                marks = data.gauge_marks(item_id, SERIES_COLORS)
            except Exception:  # noqa: BLE001 (a bad file must not take the window down)
                log.error("gauges of %s:\n%s", item_id, traceback.format_exc())
                marks = None
            if marks is None:
                self.view.remove_gauges(item_id)
            else:
                self.view.update_gauges(marks)

    def _on_loaded(self, key: FrameKey, mesh, error) -> None:
        if error is not None:
            log.error("loading %s failed: %s", key, error)
            self.statusMessage.emit(f"Could not read frame {key.index} of {key.dataset}: {error}")
            return
        for item_id, wanted in list(self._wanted.items()):
            if wanted == key and item_id in self.session.scene:
                self._show(item_id, key, mesh)
        self._update_gauges()  # the still water level may be known now
        self._update_overlays()
        self.redraw()

    def apply_theme_colors(self) -> None:
        """Header bar colours from the current theme (again after a theme switch)."""
        self.header.setStyleSheet(
            f"#viewportHeader {{ background: {COLORS['panel']}; }}"
            f"QToolButton {{ color: {COLORS['text']}; padding: 3px 8px; }}"
        )
        self.info.setStyleSheet(f"color: {COLORS['dim']};")
        self.colorbars.update()

    def _show(self, item_id: str, key: FrameKey, mesh) -> None:
        item = self.session.scene.get(item_id)
        try:
            with PERF.track(f"display {item.kind}"):
                shown = self.session.data.display(item_id, mesh)
            with PERF.track(f"upload {item.kind}"):
                self.view.update(item.kind, shown)
        except Exception:  # noqa: BLE001 (a bad frame must not take the window down)
            log.error("showing %s failed:\n%s", item_id, traceback.format_exc())
            self.statusMessage.emit(f"Could not show {item.name}; see the log")
            return
        self._shown_keys[item_id] = key
        if item.kind == "free_surface":
            self._reshow_exaggerated_beds()
        if not self.session.scene.camera:
            self.view.frame_once()

    def _reshow_exaggerated_beds(self) -> None:
        """A bed without its own still water depth is exaggerated about the free
        surface's level, which is known only once a free surface frame has arrived."""
        for other in self.session.scene:
            if other.kind != "bathymetry" or float(other.properties.get("exaggeration", 1)) == 1:
                continue
            key = self._shown_keys.get(other.id)
            mesh = self.session.data.cache.get(key) if key is not None else None
            if mesh is not None and "depth" not in mesh.point_data:
                self.view.update(other.kind, self.session.data.display(other.id, mesh))

    def _on_scene(self, event: str, item_id: str) -> None:
        if event == "reset":
            self.view.clear()
            self._wanted.clear()
            self._shown_keys.clear()
            if self.session.scene.camera:
                self.view.set_camera_state(self.session.scene.camera)
            self.refresh()
        elif event == "changed" and item_id in self.session.scene:
            item = self.session.scene.get(item_id)
            if item.kind == "case":
                self.refresh()  # path or output interval changed
            else:
                self.refresh([item_id])
        else:
            self.refresh()

    def _update_overlays(self) -> None:
        with PERF.track("update overlays"):
            self.__update_overlays_now()

    def __update_overlays_now(self) -> None:
        self._update_probe()  # a probe follows the timeline
        entries = []
        frames = []
        for item_id in self.view.item_ids:
            if item_id not in self.session.scene:
                continue
            item = self.session.scene.get(item_id)
            key = self._shown_keys.get(item_id)
            if item.visible and key is not None:
                dataset = self.session.data.dataset(item_id)
                if dataset is not None and key.index < len(dataset.frames):
                    frames.append(dataset.frames[key.index].step)  # REEF3D output number
                mesh = self.session.data.cache.get(key)
                if mesh is None:
                    continue
                variable = str(item.properties.get("variable", "none"))
                if variable != "none":
                    try:
                        vmin, vmax = self.session.data.value_range(item_id, mesh)
                    except KeyError:
                        continue  # a new variable is still loading
                    entries.append(
                        ColorBarEntry(
                            f"{item.name} · {variable}",
                            str(item.properties.get("colormap", "viridis")),
                            vmin,
                            vmax,
                        )
                    )
        self.colorbars.set_entries(entries)
        loading = " · loading…" if self.busy else ""
        if self.probe is not None:
            self.info.setText(self.probe.summary() + loading)
        elif frames:
            self.info.setText(f"output {max(frames)}{loading}")
        else:
            self.info.setText("Open a REEF3D case: File → Open case…" if not loading else loading)

    # ------------------------------------------------------------- camera
    def frame(self, view: str = "perspective") -> None:
        self.view.frame(view)
        self._store_camera()
        self.redraw()

    def canvas_snapshot(self):
        """(widget, RGBA image) of the 3D view as it is now, for screenshots."""
        try:
            self.canvas.force_draw()  # the last frame may be older than the latest change
        except Exception:  # noqa: BLE001 (older rendercanvas: the last frame will do)
            pass
        try:
            image = self.renderer.snapshot()
        except Exception:  # noqa: BLE001
            log.warning("no snapshot of the 3D view:\n%s", traceback.format_exc())
            return None
        return self.canvas, image

    def set_style(self, style: str) -> None:
        """Switch the viewport look ("studio" or "ocean")."""
        self.view.set_style(style)
        self.redraw()

    def reframe(self) -> None:
        """Frame everything again once the next meshes arrive (after opening a case)."""
        self.session.set_camera(None)
        self.view._framed = False
        if not self.view.is_empty:
            self.frame("perspective")

    def _set_ortho(self, on: bool) -> None:
        self.view.set_orthographic(on)
        self._store_camera()
        self.redraw()

    def _on_key(self, event) -> None:
        if isinstance(event, dict):
            key, modifiers = event.get("key", ""), event.get("modifiers", ())
        else:
            key, modifiers = getattr(event, "key", ""), getattr(event, "modifiers", ())
        if key in KEY_VIEWS:
            view = KEY_VIEWS[key]
            self.frame(OPPOSITE[view] if "Control" in modifiers else view)
        elif key == "5":
            self.ortho_button.toggle()
        elif key in ("Home", "."):
            self.frame("perspective")
        elif key in KEY_PANS:
            step = 0.33 if "Shift" in modifiers else 0.1
            dx, dy = KEY_PANS[key]
            self.view.pan(-dx * step, -dy * step)  # the scene follows the arrow
            self._store_camera()
            self.redraw()

    def _on_pointer(self, event) -> None:
        """A click (press and release without dragging) on a surface probes its values."""
        kind = getattr(event, "type", "")
        x, y = getattr(event, "x", 0.0), getattr(event, "y", 0.0)
        if kind == "pointer_down":
            if getattr(event, "button", 0) == 1:
                self._press = (
                    x,
                    y,
                    getattr(event, "target", None),
                    getattr(event, "pick_info", {}),
                )
            return
        press, self._press = self._press, None
        if press is None or abs(x - press[0]) + abs(y - press[1]) > 4:
            return  # a drag rotates the view; only clicks probe
        _, _, target, info = press
        item_id = self.view.item_at(target) if target is not None else None
        if item_id is None or not info or "face_index" not in info:
            self.clear_probe()
            return
        weights = tuple(info.get("face_coord", (1 / 3, 1 / 3, 1 / 3)))[:3]
        self.probe_at(item_id, int(info["face_index"]), weights)

    def probe_at(self, item_id: str, triangle: int, weights: tuple[float, ...]) -> None:
        """Show the values at a point of an item's surface; they follow the timeline."""
        self._probe_at = (item_id, int(triangle), tuple(float(w) for w in weights))
        self._update_probe()
        if self.probe is None:
            self.clear_probe()
            return
        self.statusMessage.emit(self.probe.summary())
        self._update_overlays()
        self.redraw()

    def _update_probe(self) -> None:
        if self._probe_at is None:
            return
        item_id, triangle, weights = self._probe_at
        data = self.session.data
        try:
            self.probe = data.probe(item_id, triangle, weights)
        except Exception:  # noqa: BLE001
            log.warning("probe failed:\n%s", traceback.format_exc())
            self.probe = None
        if self.probe is None:
            return
        marker = self.probe.point
        key = data.frame_key(item_id)
        mesh = data.cache.get(key) if key is not None else None
        if mesh is not None:  # the marker goes where the surface is drawn (exaggeration)
            shown = data.display(item_id, mesh)
            nodes = shown.triangles[triangle].astype(int)
            marker = tuple((shown.positions[nodes].T @ np.asarray(weights, float)).tolist())
        self.view.set_marker(marker)

    def clear_probe(self) -> None:
        self._probe_at = None
        if self.probe is not None:
            self.probe = None
            self.view.set_marker(None)
            self._update_overlays()
            self.redraw()

    def _on_interaction(self, _event) -> None:
        self._save_camera.start()

    def _store_camera(self) -> None:
        try:
            self.session.set_camera(self.view.camera_state())
        except Exception:  # noqa: BLE001
            log.warning("could not store the camera:\n%s", traceback.format_exc())
