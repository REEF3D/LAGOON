# SPDX-License-Identifier: GPL-3.0-or-later
"""Central 3D viewport: a pygfx canvas inside Qt, with Blender-style navigation.

Navigation (mouse and trackpad):

* rotate: left drag, or middle drag (Blender), all the way round, about the centre
  of the view
* pan: right drag, or Shift + middle drag (Blender)
* zoom: wheel or pinch; Ctrl + middle drag
* views: 1 front, 3 right, 7 top (Ctrl for the opposite side), 5 perspective or
  orthographic, Home or . to frame everything, 2 the 2D view (orthographic, no
  rotation: any drag pans; from the top everything is drawn flat, front and side
  give 2D elevations; 3D leaves it)

Frames are read in background threads; the viewport shows the newest frame that
has arrived, so scrubbing never blocks the window. Without a usable GPU (or in
tests, which run without a screen) a placeholder takes the viewport's place.
"""

from __future__ import annotations

import html
import logging
import os
import threading
import traceback

import numpy as np
from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtGui import QColor, QLinearGradient, QPainter, QPixmap
from PySide6.QtWidgets import (
    QComboBox,
    QHBoxLayout,
    QLabel,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from lagoonviz.core.data.manager import BackgroundLoader, FrameKey
from lagoonviz.core.scene import VIEW_EVENTS
from lagoonviz.gui.bridge import SessionBridge
from lagoonviz.gui.panels.colorbar import ColorBarEntry, ColorBarStrip
from lagoonviz.gui.perf import PERF
from lagoonviz.gui.resources import resource_path
from lagoonviz.gui.theme import COLORS, SERIES_COLORS

log = logging.getLogger("lagoon.viewport")
#: the probe marker's colour (the dot in the 3D view and before its values in the header)
PROBE_COLOR = "#e07b39"

TOP = QColor(COLORS["viewport_top"])
BOTTOM = QColor(COLORS["viewport_bottom"])


class ViewportPlaceholder(QWidget):
    """Shown when no GPU canvas is available (tests, remote sessions without GPU)."""

    statusMessage = Signal(str)
    #: the probe (orange dot) placed or moved: a Probe, or None when removed
    probeChanged = Signal(object)
    #: the domain drawn in the 2D top view (x0, x1, y0, y1); never sent here
    domainDrawn = Signal(object)
    #: the model turned on the map (degrees, about the model point); never sent here
    modelTurned = Signal(float, object)

    def start_domain_drawing(self, case_id: str = "") -> None:
        self.statusMessage.emit("Drawing the domain needs the 3D view (no GPU here)")

    def stop_domain_drawing(self) -> None:
        pass

    def __init__(self, parent: QWidget | None = None, reason: str = ""):
        super().__init__(parent)
        self.setObjectName("viewport")
        self.setMinimumSize(320, 240)
        self._icon = QPixmap(str(resource_path("viewport_mark.png")))
        self.reason = reason
        self.message = "3D viewport" + (f"\n{reason}" if reason else "")

    # the same small interface as Viewport3D
    case_filter = ""
    comparing = False
    partner = None

    def items(self) -> list[str]:
        return []

    def set_case_filter(self, case_id: str | None) -> None:
        self.case_filter = case_id or ""

    def set_compare(self, on: bool) -> None:
        self.comparing = on
        if not on:
            self.case_filter = ""

    @property
    def busy(self) -> bool:
        return False

    two_d = False

    def frame(self, view: str = "perspective") -> None:
        pass

    def set_2d(self, on: bool, side: str | None = None) -> None:
        self.two_d = on

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

#: 2D view: from above, no rotation; every drag pans
CONTROLS_2D = {
    "mouse1": ("pan", "drag", (1.0, 1.0)),
    "mouse2": ("pan", "drag", (1.0, 1.0)),
    "mouse3": ("pan", "drag", (1.0, 1.0)),
    "shift+mouse1": ("pan", "drag", (1.0, 1.0)),
    "shift+mouse3": ("pan", "drag", (1.0, 1.0)),
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
    #: the probe (orange dot) placed or moved: a Probe, or None when removed
    probeChanged = Signal(object)
    #: (key, mesh, error) from a loader thread, delivered on the GUI thread
    _loaded = Signal(object, object, object)
    #: a background map item's map was made in a worker thread
    _background_made = Signal(str)
    #: the domain drawn in the 2D top view: (x0, x1, y0, y1) in model coordinates
    domainDrawn = Signal(object)
    #: the model turned on the map with Shift + drag: degrees (counter-clockwise),
    #: about the model point (x, y)
    modelTurned = Signal(float, object)

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

        from lagoonviz.render.controls import FreeOrbitController
        from lagoonviz.render.scene3d import SceneRenderer

        self.setObjectName("viewport")
        self.setMinimumSize(320, 240)
        self.bridge = bridge
        self.session = bridge.session
        self.loader = loader
        self._wanted: dict[str, FrameKey] = {}
        self._shown_keys: dict[str, FrameKey] = {}
        #: what each "Case set-up" item was last drawn from (see DataManager.setup_signature)
        self._setup_shown: dict[str, tuple] = {}
        #: the errors of background maps told in the status bar
        self._background_told: dict[str, str] = {}
        #: compare mode: the case this view shows ("" = all), and the view whose camera
        #: moves with this one
        self.case_filter = ""
        self.partner: Viewport3D | None = None
        self.comparing = False
        self._camera_seen: dict | None = None

        try:
            self.canvas = QRenderWidget(self, update_mode="ondemand")
        except TypeError:  # rendercanvas without update modes
            self.canvas = QRenderWidget(self)
        self.renderer = gfx.WgpuRenderer(self.canvas)
        self.view = SceneRenderer(style)
        self.controller = FreeOrbitController(self.view.camera, register_events=self.renderer)
        self.two_d = False
        self.two_d_side = "top"
        self._use_controls(BLENDER_CONTROLS)
        self.canvas.request_draw(self._draw)
        self.renderer.add_event_handler(self._on_key, "key_down")
        self.renderer.add_event_handler(self._on_interaction, "pointer_up", "wheel")
        self.renderer.add_event_handler(self._on_pointer, "pointer_down", "pointer_up")
        self._press: tuple[float, float, object, dict] | None = None
        #: drawing the domain (start_domain_drawing): where the drag began, or None
        self._drawing: dict | None = None
        self.renderer.add_event_handler(self._on_move, "pointer_move")
        self._probe_at: tuple[str, int, tuple[float, ...]] | None = None
        #: the last probe (values where the user clicked), shown in the header
        self.probe = None

        self.colorbars = ColorBarStrip(self)
        self.info = QLabel()
        self.info.setTextFormat(Qt.RichText)  # the probe's orange dot
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
            button.clicked.connect(lambda _=False, k=key: self._view_button(k))
            bar.addWidget(button)
            self.view_buttons[key] = button
            if key == "frame":
                self.two_d_button = QToolButton()
                self.two_d_button.setText("2D")
                self.two_d_button.setCheckable(True)
                self.two_d_button.setAutoRaise(True)
                self.two_d_button.setToolTip(
                    "2D: orthographic, no rotation; any drag pans, the wheel zooms. "
                    "Top draws everything flat, Front and Side are elevations (2)"
                )
                self.two_d_button.toggled.connect(self.set_2d)
                bar.addWidget(self.two_d_button)
        self.ortho_button = QToolButton()
        self.ortho_button.setText("Ortho")
        self.ortho_button.setCheckable(True)
        self.ortho_button.setAutoRaise(True)
        self.ortho_button.setToolTip("Orthographic projection (5)")
        self.ortho_button.toggled.connect(self._set_ortho)
        bar.addWidget(self.ortho_button)
        self.case_box = QComboBox()
        self.case_box.setToolTip("The case this view shows")
        self.case_box.setSizeAdjustPolicy(QComboBox.AdjustToContents)
        self.case_box.activated.connect(
            lambda _i: self.set_case_filter(self.case_box.currentData())
        )
        self.case_box.hide()  # only when comparing
        bar.addSpacing(8)
        bar.addWidget(self.case_box)
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
        #: the attribution the background maps' sources ask for (© Kartverket, ...)
        self.attribution = QLabel()
        self.attribution.setObjectName("mapAttribution")
        self.attribution.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.attribution.setWordWrap(True)
        self.attribution.setContentsMargins(6, 0, 6, 0)
        self.attribution.hide()
        self.apply_theme_colors()
        layout.addWidget(self.attribution)
        layout.addWidget(self.colorbars)

        self._save_camera = QTimer(self)
        self._save_camera.setSingleShot(True)
        self._save_camera.setInterval(400)
        self._save_camera.timeout.connect(self._store_camera)
        self._sun_timer = QTimer(self)  # the sky's sun glow follows once the sun rests
        self._sun_timer.setSingleShot(True)
        self._sun_timer.setInterval(150)
        self._sun_timer.timeout.connect(self._sun_rests)

        self._loaded.connect(self._on_loaded)
        self._background_made.connect(self._on_background_made)
        bridge.sceneChanged.connect(self._on_scene)
        bridge.timeChanged.connect(lambda _t: self._on_time())
        self._apply_2d(*self._plan_state, frame=False)
        if self.session.scene.camera:
            self.view.set_camera_state(self.session.scene.camera)
        if self.session.scene.sun is not None:
            self.view.set_sun_direction(self.session.sun_direction())
        self.refresh()

    # ------------------------------------------------------------- state
    @property
    def busy(self) -> bool:
        """True while frames for the current time are still loading."""
        return any(self._wanted.get(i) != self._shown_keys.get(i) for i in self._wanted)

    def _draw(self) -> None:
        self._follow_camera()
        with PERF.track("render"):
            self.view.render(self.renderer)

    def _follow_camera(self) -> None:
        """Compare mode: when this camera moved, move the partner's the same way."""
        if self.partner is None or not self.partner.isVisible() or not self.isVisible():
            return
        state = self.view.camera.get_state()
        if self._camera_seen is not None and _same_camera(state, self._camera_seen):
            return
        self._camera_seen = state
        partner = self.partner
        partner.view.camera.set_state(state)
        partner._camera_seen = partner.view.camera.get_state()  # no echo back
        partner.redraw()

    # ------------------------------------------------------------- compare
    def items(self) -> list[str]:
        """The data items this view shows: all, or those of its case."""
        data = self.session.data
        ids = data.data_items()
        return [i for i in ids if self._in_view(i)]

    def _in_view(self, item_id: str) -> bool:
        if not self.session.data.drawn(item_id):
            return False  # the set-up leaves the results out
        if not self.case_filter:
            return True
        case_item = self.session.data.case_item(item_id)
        return case_item is not None and case_item.id == self.case_filter

    def set_case_filter(self, case_id: str | None) -> None:
        """Show only one case ("" or None: everything)."""
        self.case_filter = case_id or ""
        self._fill_cases()
        self.refresh()

    def set_compare(self, on: bool) -> None:
        """Show the case selector (compare mode) or not."""
        self.comparing = on
        self.case_box.setVisible(on)
        if not on:
            self.case_filter = ""
        self._fill_cases()
        self.refresh()

    def _fill_cases(self) -> None:
        box = self.case_box
        box.blockSignals(True)
        box.clear()
        box.addItem("All cases", "")
        for item in self.session.scene:
            if item.kind == "case":
                box.addItem(item.name, item.id)
        index = box.findData(self.case_filter)
        box.setCurrentIndex(max(index, 0))
        box.blockSignals(False)

    def redraw(self) -> None:
        self.canvas.request_draw()

    # ------------------------------------------------------------- data
    def refresh(self, item_ids: list[str] | None = None) -> None:
        """Bring the shown meshes up to date with the scene and the current time."""
        data = self.session.data
        if self.case_filter and self.case_filter not in self.session.scene:
            self.case_filter = ""  # its case was removed
        current = set(self.items())
        for item_id in list(self.view.item_ids):
            if item_id not in current:
                self.view.remove(item_id)
                self._shown_keys.pop(item_id, None)
                self._wanted.pop(item_id, None)
                self.loader.want((id(self), item_id), None)
        for item_id in item_ids if item_ids is not None else sorted(current):
            if item_id not in current:
                continue
            key = data.frame_key(item_id)
            # a frame asked for before and not started is dropped (dragging the slider)
            owner = (id(self), item_id)
            if key is None:
                self.loader.want(owner, None)
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
                self.loader.want(owner, key)
                self._show(item_id, key, mesh)
            else:
                self.loader.request(key, self._loaded.emit, owner=owner)
        self._update_gauges()
        self._update_setup()
        self._update_backgrounds()
        self._update_overlays()
        self.redraw()

    def _update_backgrounds(self) -> None:
        """The background map items: drawn from the map made for them, which is made
        in a worker thread (it may fetch map tiles) when what it shows changed."""
        backgrounds = self.session.data.backgrounds
        wanted = [i for i in backgrounds.items() if self._in_view(i)]
        for item_id in self.view.background_ids:
            if item_id not in wanted:
                self.view.remove_background(item_id)
        credits = []
        for item_id in wanted:
            item = self.session.scene.get(item_id)
            try:
                # a hidden map is made only once it is shown
                job = backgrounds.job(item_id) if item.visible else None
                if job is not None:
                    # a daemon thread: closing LAGOON does not wait for the tile servers
                    threading.Thread(
                        target=self._make_background,
                        args=(item_id, job),
                        name="lagoon-map",
                        daemon=True,
                    ).start()
                made = backgrounds.cached(item_id)
                if made is None:
                    self.view.remove_background(item_id)
                else:
                    shown = backgrounds.shown(item_id, made)
                    self.view.update_background(shown)
                    if shown.visible and made.attribution:
                        credits.append(made.attribution)
            except Exception:  # noqa: BLE001 (a bad file must not take the window down)
                log.error("background map %s:\n%s", item_id, traceback.format_exc())
                self.statusMessage.emit(f"Could not show {item.name}; see the log")
                continue
            error = backgrounds.error(item_id)
            if error and self._background_told.get(item_id) != error:
                if not backgrounds.making(item_id):
                    self.statusMessage.emit(f"{item.name}: {error}")
                    self._background_told[item_id] = error
            elif not error:
                self._background_told.pop(item_id, None)
        text = " · ".join(dict.fromkeys(credits))
        self.attribution.setText(text)
        self.attribution.setVisible(bool(text))

    def _make_background(self, item_id: str, job) -> None:
        try:
            job()
        except Exception:  # noqa: BLE001 (the worker must always report back)
            log.error("background map %s:\n%s", item_id, traceback.format_exc())
        try:
            self._background_made.emit(item_id)
        except RuntimeError:  # the view was closed meanwhile
            pass

    def _on_background_made(self, item_id: str) -> None:
        if item_id in self.session.scene:
            self._update_backgrounds()
            self.redraw()

    def _update_setup(self) -> None:
        """The "Case set-up" items: drawn again when what they show has changed."""
        data = self.session.data
        wanted = {i for i in data.setup_items() if self._in_view(i)}
        for item_id in self.view.setup_ids:
            if item_id not in wanted:
                self.view.remove_setup(item_id)
                self._setup_shown.pop(item_id, None)
        framed = False
        for item_id in wanted:
            try:
                signature = data.setup_signature(item_id)
                if signature is None:
                    self.view.remove_setup(item_id)
                    self._setup_shown.pop(item_id, None)
                    continue
                if self._setup_shown.get(item_id) == signature:
                    continue
                marks = data.setup_marks(item_id)
            except Exception:  # noqa: BLE001 (a bad input file must not take the window down)
                log.error("set-up of %s:\n%s", item_id, traceback.format_exc())
                self.statusMessage.emit(f"Could not draw the set-up of {item_id}; see the log")
                continue
            if marks is None:
                continue
            self.view.update_setup(marks)
            self._setup_shown[item_id] = signature
            framed = True
        if framed and not self.session.scene.camera:
            self.view.frame_once(self._home_view)

    def _update_gauges(self) -> None:
        with PERF.track("update gauges"):
            self._update_gauges_now()

    def _update_gauges_now(self) -> None:
        """Wave gauge positions and wave profiles, in the colours of their plot lines."""
        data = self.session.data
        profiles = {i for i in data.profile_items() if self._in_view(i)}
        wanted = {i for i in data.series_items() if self._in_view(i)} | profiles
        for item_id in self.view.gauge_ids:
            if item_id not in wanted:
                self.view.remove_gauges(item_id)
        for item_id in wanted:
            try:
                if item_id in profiles:
                    marks = data.profile_marks(item_id, SERIES_COLORS)
                else:
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
        if hasattr(self, "attribution"):
            self.attribution.setStyleSheet(
                f"color: {COLORS['dim']}; background: {COLORS['panel']}; font-size: 10px;"
            )
        self.colorbars.update()

    def _show(self, item_id: str, key: FrameKey, mesh) -> None:
        item = self.session.scene.get(item_id)
        try:
            with PERF.track(f"display {item.kind}"):
                shown = self.session.data.display(item_id, mesh)
            with PERF.track(f"upload {item.kind}"):
                self.view.update(item.kind, shown)
            with PERF.track("contours and arrows"):
                data = self.session.data
                self.view.update_contours(item_id, data.contour_lines(item_id, mesh, shown))
                self.view.update_vectors(item_id, data.vector_arrows(item_id, mesh, shown))
                self.view.update_trails(item_id, data.particle_trails(item_id, mesh))
                self.view.update_outlines(item_id, data.patch_outlines(item_id, mesh, shown))
        except Exception:  # noqa: BLE001 (a bad frame must not take the window down)
            log.error("showing %s failed:\n%s", item_id, traceback.format_exc())
            self.statusMessage.emit(f"Could not show {item.name}; see the log")
            return
        self._shown_keys[item_id] = key
        if item.kind in ("free_surface", "amr"):
            self._reshow_exaggerated_beds()
        if not self.session.scene.camera:
            self.view.frame_once(self._home_view)

    def _reshow_exaggerated_beds(self) -> None:
        """A bed without its own still water depth is exaggerated about the free
        surface's level, which is known only once a free surface frame has arrived."""
        for other in self.session.scene:
            if other.kind != "bathymetry" or self.session.data.exaggeration(other.id) == 1:
                continue
            key = self._shown_keys.get(other.id)
            mesh = self.session.data.cache.get(key) if key is not None else None
            if mesh is not None and "depth" not in mesh.point_data:
                shown = self.session.data.display(other.id, mesh)
                self.view.update(other.kind, shown)
                self.view.update_contours(
                    other.id, self.session.data.contour_lines(other.id, mesh, shown)
                )

    def _on_time(self) -> None:
        self.follow_path()
        self.refresh()

    def follow_path(self) -> bool:
        """With the camera path on, put the camera where the path is at this time."""
        state = self.session.camera_at()
        if state is None:
            return False
        self.view.set_camera_state(state)
        self.redraw()
        return True

    def _sun_rests(self) -> None:
        self.view.refresh_surroundings()
        self.redraw()

    def _on_scene(self, event: str, item_id: str) -> None:
        if event == "highlight":  # a command selected in the set-up panel: light it
            self._update_setup()
            self.redraw()
            return
        if event == "camera_keys":
            self.follow_path()
            return
        if event == "sun":
            # the light at once; the sky's sun glow (a new environment map) once the
            # sliders rest
            self.view.set_sun_direction(self.session.sun_direction(), surroundings=False)
            self._sun_timer.start()
            self.redraw()
            return
        if event in VIEW_EVENTS:
            return  # e.g. the play range: nothing to draw again
        if self.comparing:
            self._fill_cases()
        if event == "reset":
            self.view.clear()
            self._wanted.clear()
            self._shown_keys.clear()
            self._setup_shown.clear()
            self._apply_2d(*self._plan_state, frame=False)  # a loaded session's
            if self.session.scene.camera:
                self.view.set_camera_state(self.session.scene.camera)
            self.view.set_sun_direction(self.session.sun_direction())  # a loaded session's
            self.refresh()
        elif event == "view":  # 2D or 3D, or the 2D side: everything is drawn again
            self.refresh()
            self._apply_2d(*self._plan_state)
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
            self._update_overlays_now()

    def _update_overlays_now(self) -> None:
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
            # the orange dot of the marker in the view, so the values read as its values
            self.info.setText(
                f'<span style="color:{PROBE_COLOR}">\u25cf</span> '
                + html.escape(self.probe.summary() + loading)
            )
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
            self.frame(self._home_view)

    @property
    def _plan_state(self) -> tuple[bool, str]:
        scene = self.session.scene
        return scene.plan_view, scene.plan_side

    @property
    def _home_view(self) -> str:
        """The view Frame all goes to: the 2D side, or the 3D view."""
        return self.two_d_side if self.two_d else "perspective"

    def _view_button(self, key: str) -> None:
        """Frame all keeps the view; in 2D, Top, Front and Side turn the 2D view to
        that side and 3D leaves 2D."""
        if key == "frame":
            self.frame(self._home_view)
        elif self.two_d and key == "perspective":
            self.set_2d(False)
        elif self.two_d:
            self.set_2d(True, key)
            self._mark_2d_side()  # a click on the pressed side would unpress it
        else:
            self.frame(key)

    def _use_controls(self, controls: dict) -> None:
        self.controller.controls.clear()
        for control, action in controls.items():
            try:
                self.controller.controls[control] = action
            except Exception as error:  # noqa: BLE001 (keep the pygfx default for this one)
                log.warning("control %s not set: %s", control, error)

    def set_2d(self, on: bool, side: str | None = None) -> None:
        """2D: orthographic, no rotation (drags pan), seen from a side: from the top
        every item is drawn flat; front, back, right and left are elevations of the
        shapes as they are (with the vertical scale). ``side=None`` keeps the side.
        The state is the session's (saved with it, filmed by the export, shared by
        both views when comparing); leaving goes back to the 3D view."""
        on = bool(on)
        if side is None and self.two_d_button.isChecked() != on:
            self.two_d_button.setChecked(on)  # comes back here through toggled
            return
        self.session.set_plan_view(on, side)  # every view hears it ("view")
        self._apply_2d(*self._plan_state)

    def _apply_2d(self, on: bool, side: str = "top", frame: bool = True) -> None:
        """Controls, buttons and projection for 2D (from a side) or 3D;
        ``frame=False`` keeps the camera (a session that was saved in 2D brings its
        own)."""
        side = side if on else self.two_d_side
        if on == self.two_d and side == self.two_d_side:
            return
        self.two_d, self.two_d_side = on, side
        if not (on and side == "top"):
            self.stop_domain_drawing()  # the domain is drawn in the 2D top view only
        self.two_d_button.blockSignals(True)
        self.two_d_button.setChecked(on)
        self.two_d_button.blockSignals(False)
        self.controller.locked = on
        self._use_controls(CONTROLS_2D if on else BLENDER_CONTROLS)
        self.ortho_button.setEnabled(not on)
        self.view.set_plan(on and side == "top")
        self.view.set_orthographic(on or self.ortho_button.isChecked())
        self._mark_2d_side()
        if frame:
            self.frame(self._home_view)

    def _mark_2d_side(self) -> None:
        """In 2D the button of the side it looks from is shown pressed."""
        sides = {"top": "top", "bottom": "top", "front": "front", "back": "front"}
        pressed = sides.get(self.two_d_side, "right") if self.two_d else None
        for key in ("top", "front", "right"):
            button = self.view_buttons[key]
            button.setCheckable(self.two_d)
            button.setChecked(key == pressed)

    def _set_ortho(self, on: bool) -> None:
        self.view.set_orthographic(on or self.two_d)
        self._store_camera()
        self.redraw()

    def _on_key(self, event) -> None:
        if isinstance(event, dict):
            key, modifiers = event.get("key", ""), event.get("modifiers", ())
        else:
            key, modifiers = getattr(event, "key", ""), getattr(event, "modifiers", ())
        if key == "Escape" and self._drawing is not None:
            self.stop_domain_drawing()
            self.statusMessage.emit("Stopped drawing the domain")
            return
        if key in KEY_VIEWS:
            view = KEY_VIEWS[key]
            if "Control" in modifiers and not (self.two_d and view == "top"):
                view = OPPOSITE[view]  # 2D from the bottom would show only the bed
            if self.two_d:
                self.set_2d(True, view)
            else:
                self.frame(view)
        elif key == "2":
            self.two_d_button.toggle()
        elif key == "5":
            if not self.two_d:
                self.ortho_button.toggle()
        elif key in ("Home", "."):
            self.frame(self._home_view)
        elif key in KEY_PANS:
            step = 0.33 if "Shift" in modifiers else 0.1
            dx, dy = KEY_PANS[key]
            self.view.pan(-dx * step, -dy * step)  # the scene follows the arrow
            self._store_camera()
            self.redraw()

    # ------------------------------------------------------------- drawing the domain
    def start_domain_drawing(self, case_id: str = "") -> None:
        """The next drag in the 2D top view draws the domain (domainDrawn); with Shift
        it turns the model on the map about the domain's centre (modelTurned). Esc
        stops."""
        if not (self.two_d and self.two_d_side == "top"):
            self.set_2d(True, "top")
        self._drawing = {"start": None, "case": case_id}
        try:
            self.controller.enabled = False  # the drag draws, it does not pan
        except AttributeError:
            pass
        self.canvas.setCursor(Qt.CrossCursor)
        self.statusMessage.emit(
            "Drag the domain in the 2D view · Shift + drag turns the model on the map · Esc stops"
        )

    def stop_domain_drawing(self) -> None:
        if self._drawing is None:
            return
        self._drawing = None
        try:
            self.controller.enabled = True
        except AttributeError:
            pass
        self.canvas.unsetCursor()
        self.view.set_outline(None)
        self.redraw()

    def _plan_point(self, event) -> tuple[float, float] | None:
        try:
            size = tuple(self.canvas.get_logical_size())
        except AttributeError:
            size = (self.canvas.width(), self.canvas.height())
        return self.view.plan_point(getattr(event, "x", 0.0), getattr(event, "y", 0.0), size)

    def _domain_box(self, case_id: str = "") -> tuple[float, float, float, float] | None:
        cases = [i.id for i in self.session.scene if i.kind == "case"]
        if case_id and case_id in self.session.scene:
            cases = [case_id]
        elif self.case_filter:
            cases = [self.case_filter]
        domain = self.session.data.backgrounds.domain(cases[0]) if cases else None
        return tuple(domain[:4]) if domain is not None else None

    def _draw_domain(self, kind: str, event) -> None:
        point = self._plan_point(event)
        drawing = self._drawing
        if point is None or drawing is None:
            return
        level = 4.0  # over everything in the flat 2D view (PLAN_MARKS)
        if kind == "pointer_down":
            if getattr(event, "button", 1) != 1:
                return
            modifiers = getattr(event, "modifiers", ()) or ()
            box = self._domain_box(drawing.get("case", ""))
            drawing["start"] = point
            drawing["turn"] = "Shift" in modifiers and box is not None
            drawing["box"] = box
            return
        start = drawing.get("start")
        if start is None:
            return
        if drawing.get("turn"):
            x0, x1, y0, y1 = drawing["box"]
            centre = np.array([(x0 + x1) / 2, (y0 + y1) / 2])
            a = np.arctan2(*(np.asarray(start) - centre)[::-1])
            b = np.arctan2(*(np.asarray(point) - centre)[::-1])
            angle = float(np.degrees(b - a))
            angle = (angle + 180.0) % 360.0 - 180.0
            angle = round(angle * 2) / 2  # half degrees
            corners = np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]]) - centre
            c, s_ = np.cos(np.radians(angle)), np.sin(np.radians(angle))
            turned = corners @ np.array([[c, s_], [-s_, c]]) + centre
            outline = np.column_stack([turned, np.full(4, level)])
            drawing["angle"] = angle
            drawing["centre"] = (float(centre[0]), float(centre[1]))
            self.statusMessage.emit(f"Turn the model by {angle:+.1f}°")
        else:
            (sx, sy), (px, py) = start, point
            outline = np.array(
                [[sx, sy, level], [px, sy, level], [px, py, level], [sx, py, level]], float
            )
            self.statusMessage.emit(
                f"Domain x {min(sx, px):.1f} … {max(sx, px):.1f} m, "
                f"y {min(sy, py):.1f} … {max(sy, py):.1f} m"
            )
        if kind == "pointer_move":
            self.view.set_outline(outline)
            self.redraw()
            return
        # released: the domain, or the turn
        turn, angle, centre = drawing.get("turn"), drawing.get("angle", 0.0), drawing.get("centre")
        self.stop_domain_drawing()
        if turn:
            if angle:
                self.modelTurned.emit(angle, centre)
            return
        (sx, sy), (px, py) = start, point
        if abs(px - sx) > 0 and abs(py - sy) > 0:
            self.domainDrawn.emit((min(sx, px), max(sx, px), min(sy, py), max(sy, py)))

    def _on_move(self, event) -> None:
        if self._drawing is not None and self._drawing.get("start") is not None:
            self._draw_domain("pointer_move", event)

    def _on_pointer(self, event) -> None:
        """A click (press and release without dragging) on a surface probes its values."""
        kind = getattr(event, "type", "")
        if self._drawing is not None:
            self._draw_domain(kind, event)
            return
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
        self.probeChanged.emit(self.probe)
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
            self.probeChanged.emit(None)
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


def _same_camera(a: dict, b: dict) -> bool:
    """True if two camera states match (positions, rotations, sizes)."""
    for key in ("position", "rotation", "width", "height", "fov", "zoom"):
        if key in a or key in b:
            if not np.allclose(np.asarray(a.get(key, 0.0)), np.asarray(b.get(key, 0.0))):
                return False
    return True
