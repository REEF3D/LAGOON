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
import math
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
    QProgressBar,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from lagoonviz.core.data.manager import BackgroundLoader, FrameKey
from lagoonviz.core.legend import colour_bars, plain
from lagoonviz.core.scene import DATA_RESET, VIEW_EVENTS
from lagoonviz.gui.bridge import SessionBridge
from lagoonviz.gui.panels.colorbar import ColorBarEntry, ColorBarStrip
from lagoonviz.gui.perf import PERF
from lagoonviz.gui.resources import resource_path
from lagoonviz.gui.theme import COLORS, SERIES_COLORS

log = logging.getLogger("lagoon.viewport")
#: the probe marker's colour (the dot in the 3D view and before its values in the header)
PROBE_COLOR = "#e07b39"
#: the river corridor being edited, drawn over everything
HIGHLIGHT_SKETCH = "#ffd84a"

TOP = QColor(COLORS["viewport_top"])
BOTTOM = QColor(COLORS["viewport_bottom"])


def _edges_near(box, point, reach: float) -> set[str]:
    """The edges of a rectangle (x0, x1, y0, y1) within ``reach`` of a point (along
    their length, and a little beyond their ends): one, or two at a corner."""
    x0, x1, y0, y1 = box
    px, py = point
    edges = set()
    if y0 - reach <= py <= y1 + reach:
        near = {"x0": abs(px - x0), "x1": abs(px - x1)}
        side = min(near, key=near.get)
        if near[side] <= reach:
            edges.add(side)
    if x0 - reach <= px <= x1 + reach:
        near = {"y0": abs(py - y0), "y1": abs(py - y1)}
        side = min(near, key=near.get)
        if near[side] <= reach:
            edges.add(side)
    return edges


def _moved_edges(box, edges, point) -> tuple[float, float, float, float]:
    """A rectangle with its grabbed edges at the pointer: still a rectangle, its
    sides in order (an edge dragged past the other one swaps them)."""
    x0, x1, y0, y1 = box
    px, py = point
    if "x0" in edges:
        x0 = px
    if "x1" in edges:
        x1 = px
    if "y0" in edges:
        y0 = py
    if "y1" in edges:
        y1 = py
    return min(x0, x1), max(x0, x1), min(y0, y1), max(y0, y1)


class ViewportPlaceholder(QWidget):
    """Shown when no GPU canvas is available (tests, remote sessions without GPU)."""

    statusMessage = Signal(str)
    #: a frame of an item was drawn (its automatic colour range is that frame's)
    frameShown = Signal(str)
    #: the probe (orange dot) placed or moved: a Probe, or None when removed
    probeChanged = Signal(object)
    #: the domain drawn in the 2D top view (x0, x1, y0, y1); never sent here
    domainDrawn = Signal(object)
    #: another area drawn there: (purpose, (x0, x1, y0, y1)); never sent here
    areaDrawn = Signal(str, object)
    #: a line traced there: (purpose, [(x, y), ...]), or (purpose, None) when the
    #: tracing stopped without a line; never sent here
    lineDrawn = Signal(str, object)
    #: the model turned on the map (degrees, about the model point); never sent here
    modelTurned = Signal(float, object)

    #: a river corridor's point dragged: (case, side, index, x, y); never sent here
    corridorPointMoved = Signal(str, str, int, float, float)
    #: editing the river corridor stopped (Esc, another view); never sent here
    corridorEditingStopped = Signal()
    #: the blocks edited: a corner, a point of an edge, an edge picked; never sent here
    blockVertexMoved = Signal(str, str, float, float)
    blockPointMoved = Signal(str, str, int, float, float)
    blockEdgePicked = Signal(str, str, str, float, float)
    blocksEditingStopped = Signal()

    def start_blocks_editing(self, case_id: str = "", mode: str = "drag") -> None:
        self.statusMessage.emit("Editing the blocks needs the 3D view (no GPU here)")

    def stop_blocks_editing(self) -> None:
        pass

    def point_north(self) -> None:
        pass

    def start_domain_drawing(self, case_id: str = "", purpose: str = "domain") -> None:
        self.statusMessage.emit(f"Drawing the {purpose} needs the 3D view (no GPU here)")

    def start_line_drawing(self, case_id: str = "", purpose: str = "river", points=None) -> None:
        self.statusMessage.emit(f"Drawing the {purpose} needs the 3D view (no GPU here)")
        self.lineDrawn.emit(purpose, None)

    def stop_line_drawing(self) -> None:
        pass

    def start_corridor_editing(self, case_id: str = "") -> None:
        self.statusMessage.emit("Dragging the corridor's points needs the 3D view (no GPU here)")

    def view_centre(self) -> tuple[float, float] | None:
        return None

    def stop_corridor_editing(self) -> None:
        pass

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
    #: a frame of an item was drawn (its automatic colour range is that frame's)
    frameShown = Signal(str)
    #: the probe (orange dot) placed or moved: a Probe, or None when removed
    probeChanged = Signal(object)
    #: (key, mesh, error) from a loader thread, delivered on the GUI thread
    _loaded = Signal(object, object, object)
    #: a background map item's map was made in a worker thread
    _background_made = Signal(str)
    #: its tiles: (item, had, of)
    _background_progress = Signal(str, int, int)
    #: the domain drawn in the 2D top view: (x0, x1, y0, y1) in model coordinates
    domainDrawn = Signal(object)
    #: another area drawn there (start_domain_drawing with a purpose other than
    #: "domain", e.g. "porous structure"): (purpose, (x0, x1, y0, y1))
    areaDrawn = Signal(str, object)
    #: a line traced there point by point (start_line_drawing): (purpose, [(x, y),
    #: ...]), or (purpose, None) when the tracing stopped without a line
    lineDrawn = Signal(str, object)
    #: the model turned on the map with Shift + drag: degrees (counter-clockwise),
    #: about the model point (x, y)
    modelTurned = Signal(float, object)
    #: a river corridor's point dragged in the 2D top view: (case, side, index, x, y)
    corridorPointMoved = Signal(str, str, int, float, float)
    #: editing the river corridor stopped (Esc, another view)
    corridorEditingStopped = Signal()
    #: a corner of the blocks dragged in the 2D top view: (case, vertex, x, y)
    blockVertexMoved = Signal(str, str, float, float)
    #: a point shaping an edge of the blocks dragged: (case, edge, index, x, y)
    blockPointMoved = Signal(str, str, int, float, float)
    #: an edge of the blocks clicked ("cut" or "extend" mode): (case, mode, edge, x, y)
    blockEdgePicked = Signal(str, str, str, float, float)
    #: editing the blocks stopped (Esc, another view)
    blocksEditingStopped = Signal()

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
        #: editing a river corridor (start_corridor_editing): the case, the point being
        #: dragged, or None
        self._corridor_edit: dict | None = None
        #: tracing a line point by point (start_line_drawing): the case, the purpose,
        #: the LineTrace, the press of a click, or None
        self._tracing: dict | None = None
        #: editing the blocks (start_blocks_editing): the case, the mode ("drag", "cut",
        #: "extend") and what is being dragged
        self._blocks_edit: dict | None = None
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
        self.north_button = QToolButton()
        self.north_button.setText("North up")
        self.north_button.setAutoRaise(True)
        self.north_button.clicked.connect(self.point_north)
        self.north_button.hide()
        bar.addWidget(self.north_button)
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
        #: the background map's tiles being fetched
        self.map_progress = QProgressBar()
        self.map_progress.setMaximumWidth(170)
        self.map_progress.setMaximumHeight(16)
        self.map_progress.setTextVisible(True)
        self.map_progress.setFormat("Map: %v of %m tiles")
        self.map_progress.setToolTip("The background map's tiles (fetched once, then cached)")
        self.map_progress.hide()
        bar.addWidget(self.map_progress)
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
        self._view_timer = QTimer(self)  # the background maps follow the 2D view at rest
        self._view_timer.setSingleShot(True)
        self._view_timer.setInterval(300)
        self._view_timer.timeout.connect(self._follow_view)
        self._sun_timer = QTimer(self)  # the sky's sun glow follows once the sun rests
        self._sun_timer.setSingleShot(True)
        self._sun_timer.setInterval(150)
        self._sun_timer.timeout.connect(self._sun_rests)

        self._loaded.connect(self._on_loaded)
        self._background_made.connect(self._on_background_made)
        self._background_progress.connect(self._on_background_progress)
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
    def refresh(self, item_ids: list[str] | None = None, time_only: bool = False) -> None:
        """Bring the shown meshes up to date with the scene and the current time.
        ``time_only``: only the time moved (the timeline played or dragged): what does
        not change with time (north, the set-up, the background maps) is left as it
        is, so dragging the timeline stays smooth."""
        data = self.session.data
        if not time_only:
            self._apply_north()
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
            # a frame asked for before and not started is dropped (dragging the slider)
            owner = (id(self), item_id)
            if not data.shown(item_id):
                # switched off: its frames are not read (a large volume costs nothing)
                self.loader.want(owner, None)
                self.view.remove(item_id)
                self._wanted.pop(item_id, None)
                self._shown_keys.pop(item_id, None)
                continue
            key = data.frame_key(item_id)
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
            if time_only and self._shown_keys.get(item_id) == key:
                # the same frame as before (a bed, solids, geo.dat's points: shown at
                # any time): not displayed and uploaded again at every step
                self.loader.want(owner, key)
                continue
            mesh = data.cache.get(key)
            if mesh is not None:
                self.loader.want(owner, key)
                self._show(item_id, key, mesh)
            else:
                self.loader.request(key, self._loaded.emit, owner=owner)
        self._update_gauges()
        if not time_only:
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
                    self.map_progress.setRange(0, 0)  # busy until the tiles are counted
                    self.map_progress.setFormat("Map: %v of %m tiles")
                    self.map_progress.show()
                    # a daemon thread: closing LAGOON does not wait for the tile servers
                    threading.Thread(
                        target=self._make_background,
                        args=(item_id, job),
                        name="lagoon-map",
                        daemon=True,
                    ).start()
                # the map for the view, else the one before while that is being made
                made = backgrounds.cached(item_id) or backgrounds.latest(item_id)
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
        def progress(done: int, total: int) -> None:
            try:
                self._background_progress.emit(item_id, done, total)
            except RuntimeError:  # the view was closed meanwhile
                pass

        try:
            job(progress)
        except Exception:  # noqa: BLE001 (the worker must always report back)
            log.error("background map %s:\n%s", item_id, traceback.format_exc())
        try:
            self._background_made.emit(item_id)
        except RuntimeError:  # the view was closed meanwhile
            pass

    def _on_background_progress(self, _item_id: str, done: int, total: int) -> None:
        bar = self.map_progress
        bar.setRange(0, max(total, 1))
        bar.setValue(done)
        bar.setVisible(done < total)

    def _on_background_made(self, item_id: str) -> None:
        backgrounds = self.session.data.backgrounds
        if not any(backgrounds.making(i) for i in backgrounds.items()):
            self.map_progress.hide()
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
        """Wave gauge positions and free surface lines, in the colours of their plot lines."""
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
                self.view.update_grid_lines(item_id, data.amr_grid_lines(item_id, mesh, shown))
            self.frameShown.emit(item_id)
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
        self.refresh(time_only=True)

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
        if event == "legend":  # the bars under the view on or off
            self._update_overlays()
            return
        if event == "georef_tried":  # a rotation slider dragged: the maps and north follow
            self._update_backgrounds()
            self._apply_north()
            self.redraw()
            return
        if event in VIEW_EVENTS:
            return  # e.g. the play range: nothing to draw again
        if self.comparing:
            self._fill_cases()
        if event == "reset" and item_id == DATA_RESET:
            # the cases read again (Reload, following a run): what is shown stays
            # until each item's new frame has been read (a large run takes a while;
            # cleared, the view stayed empty all that time, and for good if a read
            # failed)
            self.refresh()
        elif event == "reset":
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
        frames = []
        loaded = {}
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
                if mesh is not None:
                    loaded[item_id] = mesh
        entries = []
        if self.session.legend_layout()["view"] != "off":
            entries = [
                ColorBarEntry(plain(bar.label), bar.colormap, bar.vmin, bar.vmax, bar.ticks)
                for bar in colour_bars(self.session, loaded)
            ]
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
            self.stop_line_drawing()
            self.stop_corridor_editing()
            self.stop_blocks_editing()
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
        self._view_moved()  # the maps: the 2D view's area, or the domain's in 3D

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
            purpose = self._drawing.get("purpose", "domain")
            self.stop_domain_drawing()
            self.statusMessage.emit(f"Stopped drawing the {purpose}")
            return
        if self._tracing is not None and self._trace_key(key):
            return
        if key == "Escape" and self._corridor_edit is not None:
            self.stop_corridor_editing()
            self.statusMessage.emit("Stopped editing the river corridor")
            return
        if key == "Escape" and self._blocks_edit is not None:
            self.stop_blocks_editing()
            self.statusMessage.emit("Stopped editing the blocks")
            return
        if key in KEY_VIEWS:
            view = KEY_VIEWS[key]
            if "Control" in modifiers and not (self.two_d and view == "top"):
                view = OPPOSITE[view]  # 2D from the bottom would show only the bed
            if self.two_d:
                self.set_2d(True, view)
            else:
                self.frame(view)
        elif key in ("n", "N"):
            self.point_north()
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
    def start_domain_drawing(self, case_id: str = "", purpose: str = "domain") -> None:
        """The next drag in the 2D top view draws the domain (domainDrawn); with Shift
        it turns the model on the map about the domain's centre (modelTurned). Esc
        stops. With another ``purpose`` ("porous structure") the drag draws an area
        for it (areaDrawn), Shift does nothing more."""
        if not (self.two_d and self.two_d_side == "top"):
            self.set_2d(True, "top")
        self.stop_line_drawing()
        self._drawing = {"start": None, "case": case_id, "purpose": purpose}
        try:
            self.controller.enabled = False  # the drag draws, it does not pan
        except AttributeError:
            pass
        self.canvas.setCursor(Qt.CrossCursor)
        if purpose == "domain":
            self.statusMessage.emit(
                "Drag the domain in the 2D view, or its edges and corners · Shift + drag "
                "turns the model on the map · Esc stops"
            )
        else:
            self.statusMessage.emit(f"Drag the {purpose} in the 2D view · Esc stops")

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

    # ------------------------------------------------------------- tracing a line
    #: how near (pixels) a click must be to a point or a segment to be on it
    TRACE_PICK = 8.0
    #: what the status bar says while tracing
    TRACE_HELP = (
        "click to add · click on a segment to put a point in · drag a point to move it · "
        "right click (or Alt + click) removes it · click the last point again or Enter "
        "to end · Backspace · Esc"
    )

    def start_line_drawing(self, case_id: str = "", purpose: str = "river", points=None) -> None:
        """Clicks in the 2D top view trace a line point by point (upstream first for a
        river; ``points``: a line drawn before, taken up again): see
        :mod:`lagoonviz.core.line_trace`. Ended, it is sent (lineDrawn); Esc stops;
        dragging away from the points pans the view as always."""
        from lagoonviz.core.line_trace import LineTrace

        if not (self.two_d and self.two_d_side == "top"):
            self.set_2d(True, "top")
        self.stop_domain_drawing()
        self.stop_corridor_editing()
        self.stop_line_drawing(quiet=True)
        self._tracing = {
            "case": case_id,
            "purpose": purpose,
            "trace": LineTrace(points),
            "press": None,
            "grab": None,
        }
        self.canvas.setCursor(Qt.CrossCursor)
        if points:
            self._trace_sketch()
        else:
            self.statusMessage.emit(
                f"Click the {purpose} point by point in the 2D view, upstream first · "
                + self.TRACE_HELP
            )

    def stop_line_drawing(self, quiet: bool = False) -> None:
        """Stop tracing (no line); ``lineDrawn(purpose, None)`` unless ``quiet``."""
        tracing = self._tracing
        if tracing is None:
            return
        self._tracing = None
        self._trace_pan(True)
        self.canvas.unsetCursor()
        self.view.set_sketch(None)
        self.redraw()
        if not quiet:
            self.lineDrawn.emit(tracing["purpose"], None)

    def _finish_line(self) -> None:
        tracing = self._tracing
        if tracing is None or not tracing["trace"].done:
            return
        points = list(tracing["trace"].points)
        self._tracing = None
        self._trace_pan(True)
        self.canvas.unsetCursor()
        self.view.set_sketch(None)
        self.redraw()
        self.lineDrawn.emit(tracing["purpose"], points)

    def _trace_pan(self, on: bool) -> None:
        try:
            self.controller.enabled = on
        except AttributeError:
            pass

    def _trace_sketch(self, pointer=None) -> None:
        trace = self._tracing["trace"]
        segments, dots = trace.sketch(pointer, self.SKETCH_Z)
        self.view.set_sketch(segments, dots, color=HIGHLIGHT_SKETCH, point_size=8.0)
        count = len(trace.points)
        if count:
            self.statusMessage.emit(
                f"{self._tracing['purpose'].capitalize()}: {count} point"
                f"{'s' if count != 1 else ''}, {trace.length(pointer):.1f} m · " + self.TRACE_HELP
            )
        self.redraw()

    def _trace_key(self, key: str) -> bool:
        """Esc, Enter and Backspace while tracing; True if the key was the trace's."""
        if key == "Escape":
            purpose = self._tracing["purpose"]
            self.stop_line_drawing()
            self.statusMessage.emit(f"Stopped drawing the {purpose}")
            return True
        if key in ("Enter", "Return"):
            if self._tracing["trace"].done:
                self._finish_line()
            else:
                self.statusMessage.emit("Click two points at least, then Enter")
            return True
        if key in ("Backspace", "Delete"):
            self._tracing["trace"].back()
            self._trace_sketch()
            return True
        return False

    def _trace_pointer(self, kind: str, event) -> None:
        """Press on a point: drag it (a click on the last one ends the line); right
        click (or Alt + click) on a point: removed; a click elsewhere adds a point
        (or puts one into a segment); a drag elsewhere pans."""
        tracing = self._tracing
        trace = tracing["trace"]
        x, y = getattr(event, "x", 0.0), getattr(event, "y", 0.0)
        point = self._plan_point(event)
        reach = self._reach(event, self.TRACE_PICK) or 0.0
        if kind == "pointer_down":
            button = getattr(event, "button", 1)
            modifiers = getattr(event, "modifiers", ()) or ()
            on = trace.point_at(point, reach) if point is not None else None
            tracing["press"] = tracing["grab"] = None
            # a right click, Alt + click, or Ctrl + click on a Mac (Qt's Meta there)
            removing = button == 2 or (button == 1 and ({"Alt", "Meta"} & set(modifiers)))
            if on is not None and removing:
                trace.remove(on)
                self._trace_pan(False)  # this press does not pan
                tracing["grab"] = {"index": None, "moved": False}
                self._trace_sketch()
                return
            if button != 1:
                return
            if on is not None:
                tracing["grab"] = {"index": on, "moved": False}
                self._trace_pan(False)  # the drag moves the point, not the view
                self.canvas.setCursor(Qt.ClosedHandCursor)
                return
            tracing["press"] = (x, y)
            return
        if kind != "pointer_up":
            return
        grab, tracing["grab"] = tracing["grab"], None
        press, tracing["press"] = tracing["press"], None
        if grab is not None:
            self._trace_pan(True)
            self.canvas.setCursor(Qt.CrossCursor)
            if grab["index"] is None or grab["moved"]:
                self._trace_sketch()
                return
            # pressed and let go on a point without moving it: a click on it
            if point is not None and trace.click(point, reach):
                self._finish_line()
            return
        if press is None or abs(x - press[0]) + abs(y - press[1]) > 4 or point is None:
            return  # a drag: the view panned
        if trace.click(point, reach):
            self._finish_line()
            return
        self._trace_sketch(point)

    def _trace_move(self, event) -> None:
        tracing = self._tracing
        trace = tracing["trace"]
        point = self._plan_point(event)
        grab = tracing["grab"]
        if grab is not None:
            if grab["index"] is not None and point is not None:
                trace.move(grab["index"], point)
                grab["moved"] = True
                self._trace_sketch()
            return
        if tracing["press"] is not None or point is None:
            return  # panning
        reach = self._reach(event, self.TRACE_PICK) or 0.0
        over = trace.point_at(point, reach) is not None
        self.canvas.setCursor(Qt.OpenHandCursor if over else Qt.CrossCursor)
        if trace.points:
            self._trace_sketch(None if over else point)

    # ------------------------------------------------------------- the river corridor
    #: how near (pixels) a press must be to a point of a bank to drag it
    CORRIDOR_PICK = 12.0
    #: the sketch's height in the flat 2D view (over the set-up's marks)
    SKETCH_Z = 5.0

    def start_corridor_editing(self, case_id: str = "") -> None:
        """Drag the points of a case's river corridor in the 2D top view (a press near
        a point takes it, elsewhere the view pans as always); Esc stops."""
        if not (self.two_d and self.two_d_side == "top"):
            self.set_2d(True, "top")
        self.stop_line_drawing()
        self._corridor_edit = {"case": case_id, "drag": None}
        self.statusMessage.emit(
            "Drag the points of the banks in the 2D view · elsewhere the view pans · Esc stops"
        )

    def stop_corridor_editing(self) -> None:
        if self._corridor_edit is None:
            return
        self._corridor_edit = None
        try:
            self.controller.enabled = True
        except AttributeError:
            pass
        self.canvas.unsetCursor()
        self.view.set_sketch(None)
        self.redraw()
        self.corridorEditingStopped.emit()

    def _corridor(self):
        edit = self._corridor_edit
        try:
            return self.session.corridor(edit["case"] or None) if edit else None
        except ValueError:
            return None

    def _corridor_pick(self, event):
        """The point of a bank under the pointer: (side, index), or None."""
        corridor = self._corridor()
        point = self._plan_point(event)
        if corridor is None or point is None:
            return None
        x, y = getattr(event, "x", 0.0), getattr(event, "y", 0.0)
        try:
            size = tuple(self.canvas.get_logical_size())
        except AttributeError:
            size = (self.canvas.width(), self.canvas.height())
        beside = self.view.plan_point(x + self.CORRIDOR_PICK, y, size)
        if beside is None:
            return None
        reach = math.hypot(beside[0] - point[0], beside[1] - point[1])
        best = None
        for side in ("left", "right"):
            bank = corridor.bank(side)
            distance = np.hypot(bank[:, 0] - point[0], bank[:, 1] - point[1])
            index = int(np.argmin(distance))
            if distance[index] <= reach and (best is None or distance[index] < best[0]):
                best = (float(distance[index]), side, index)
        return None if best is None else best[1:]

    def _corridor_sketch(self, corridor, side: str | None = None) -> None:
        """The corridor drawn over everything (the bank being dragged in full), or the
        point under the pointer, larger."""
        z = self.SKETCH_Z
        if corridor is None:
            self.view.set_sketch(None)
            return
        sides = (side,) if side else ()
        lines = []
        for name in sides:
            line = corridor.bank_line(name)
            points = np.column_stack([line, np.full(len(line), z)])
            lines.append(np.stack([points[:-1], points[1:]], axis=1))
        handles = np.vstack([corridor.left, corridor.right])
        self.view.set_sketch(
            np.concatenate(lines) if lines else None,
            np.column_stack([handles, np.full(len(handles), z)]),
            color=HIGHLIGHT_SKETCH,
            point_size=9.0,
        )

    def _edit_corridor(self, kind: str, event) -> bool:
        """Press, drag and release on a point of a bank; True if the event was the
        corridor's (not the view's)."""
        edit = self._corridor_edit
        if edit is None:
            return False
        drag = edit.get("drag")
        if kind == "pointer_down":
            if getattr(event, "button", 1) != 1:
                return False
            picked = self._corridor_pick(event)
            if picked is None:
                return False  # the view pans
            corridor = self._corridor()
            edit["drag"] = {
                "side": picked[0],
                "index": picked[1],
                "corridor": corridor,
                "moved": None,
            }
            try:
                self.controller.enabled = False  # cancels the pan just begun
            except AttributeError:
                pass
            self.canvas.setCursor(Qt.ClosedHandCursor)
            self._corridor_sketch(corridor, picked[0])
            self.redraw()
            return True
        if kind == "pointer_move":
            if drag is None:  # hovering: the hand over a point
                over = self._corridor_pick(event) is not None
                if over:
                    self.canvas.setCursor(Qt.OpenHandCursor)
                else:
                    self.canvas.unsetCursor()
                return False
            point = self._plan_point(event)
            if point is None:
                return True
            moved = drag["corridor"].moved(drag["side"], drag["index"], *point)
            drag["moved"] = point
            self._corridor_sketch(moved, drag["side"])
            self.statusMessage.emit(
                f"{drag['side']} bank, point {drag['index'] + 1}: "
                f"x {point[0]:.2f}, y {point[1]:.2f} m"
            )
            self.redraw()
            return True
        if kind == "pointer_up" and drag is not None:
            edit["drag"] = None
            try:
                self.controller.enabled = True
            except AttributeError:
                pass
            self.canvas.setCursor(Qt.OpenHandCursor)
            self.view.set_sketch(None)
            self.redraw()
            if drag["moved"] is not None:
                x, y = drag["moved"]
                self.corridorPointMoved.emit(edit["case"], drag["side"], drag["index"], x, y)
            return True
        return False

    # ------------------------------------------------------------- the blocks
    #: what each mode of editing the blocks says in the status bar
    BLOCK_MODES = {
        "drag": "Drag the corners and the points of the edges in the 2D view",
        "cut": "Click an edge to cut it in two there",
        "extend": "Click an edge on the outline to add a block outside it",
    }

    def start_blocks_editing(self, case_id: str = "", mode: str = "drag") -> None:
        """Edit a case's blocks in the 2D top view: drag their corners and the points
        of their edges (``mode`` "drag"), or click an edge to cut it ("cut") or to add a
        block outside it ("extend"). Elsewhere the view pans as always; Esc stops."""
        if not (self.two_d and self.two_d_side == "top"):
            self.set_2d(True, "top")
        self.stop_corridor_editing()
        self._blocks_edit = {"case": case_id, "mode": mode, "drag": None}
        self.statusMessage.emit(
            self.BLOCK_MODES.get(mode, "") + " · elsewhere the view pans · Esc stops"
        )

    def stop_blocks_editing(self) -> None:
        if self._blocks_edit is None:
            return
        self._blocks_edit = None
        try:
            self.controller.enabled = True
        except AttributeError:
            pass
        self.canvas.unsetCursor()
        self.view.set_sketch(None)
        self.redraw()
        self.blocksEditingStopped.emit()

    def _blocks(self):
        edit = self._blocks_edit
        try:
            return self.session.multiblock(edit["case"] or None) if edit else None
        except ValueError:
            return None

    def _blocks_pick(self, event):
        """The handle under the pointer: ("vertex", name) or ("point", edge, index); in
        the "cut" and "extend" modes the edge under it: ("edge", name, x, y)."""
        topology = self._blocks()
        point = self._plan_point(event)
        reach = self._reach(event, self.CORRIDOR_PICK) if point is not None else None
        if topology is None or point is None or reach is None:
            return None
        if self._blocks_edit.get("mode", "drag") != "drag":
            outline = (
                set(topology.boundary_edges())
                if self._blocks_edit["mode"] == "extend"
                else set(topology.edges)
            )
            best = None
            for name in outline:
                line = topology.edges[name].line(topology.vertices)
                d = np.hypot(line[:, 0] - point[0], line[:, 1] - point[1])
                k = int(np.argmin(d))
                if d[k] <= 2 * reach and (best is None or d[k] < best[0]):
                    best = (float(d[k]), name)
            return None if best is None else ("edge", best[1], *point)
        best = None
        for name, (vx, vy) in topology.vertices.items():
            d = math.hypot(vx - point[0], vy - point[1])
            if d <= reach and (best is None or d < best[0]):
                best = (d, ("vertex", name))
        for name, edge in topology.edges.items():
            if not len(edge.points):
                continue
            d = np.hypot(edge.points[:, 0] - point[0], edge.points[:, 1] - point[1])
            k = int(np.argmin(d))
            if d[k] <= reach and (best is None or d[k] < best[0]):
                best = (float(d[k]), ("point", name, k))
        return None if best is None else best[1]

    def _blocks_sketch(self, topology, picked=None) -> None:
        """The blocks' edges touching what is dragged, and every handle, over all."""
        z = self.SKETCH_Z
        if topology is None:
            self.view.set_sketch(None)
            return
        lines = []
        for name, edge in topology.edges.items():
            touched = picked is not None and (
                (picked[0] == "vertex" and picked[1] in (edge.start, edge.end))
                or (picked[0] in ("point", "edge") and picked[1] == name)
            )
            if touched:
                line = edge.line(topology.vertices)
                p = np.column_stack([line, np.full(len(line), z)])
                lines.append(np.stack([p[:-1], p[1:]], axis=1))
        handles = [np.asarray(v, float) for v in topology.vertices.values()]
        for edge in topology.edges.values():
            handles += list(edge.points)
        handles = np.asarray(handles, float).reshape(-1, 2)
        self.view.set_sketch(
            np.concatenate(lines) if lines else None,
            np.column_stack([handles, np.full(len(handles), z)]),
            color=HIGHLIGHT_SKETCH,
            point_size=9.0,
        )

    def _edit_blocks(self, kind: str, event) -> bool:
        """Press, drag and release on a handle (or a click on an edge); True if the
        event was the blocks' (not the view's)."""
        edit = self._blocks_edit
        if edit is None:
            return False
        drag = edit.get("drag")
        if kind == "pointer_down":
            if getattr(event, "button", 1) != 1:
                return False
            picked = self._blocks_pick(event)
            if picked is None:
                return False  # the view pans
            if picked[0] == "edge":
                self.blockEdgePicked.emit(edit["case"], edit["mode"], picked[1], *picked[2:])
                return True
            edit["drag"] = {"picked": picked, "topology": self._blocks(), "moved": None}
            try:
                self.controller.enabled = False  # cancels the pan just begun
            except AttributeError:
                pass
            self.canvas.setCursor(Qt.ClosedHandCursor)
            self._blocks_sketch(edit["drag"]["topology"], picked)
            self.redraw()
            return True
        if kind == "pointer_move":
            if drag is None:  # hovering: the hand over a handle, a cross over an edge
                over = self._blocks_pick(event)
                if over is None:
                    self.canvas.unsetCursor()
                elif over[0] == "edge":
                    self.canvas.setCursor(Qt.PointingHandCursor)
                else:
                    self.canvas.setCursor(Qt.OpenHandCursor)
                return False
            point = self._plan_point(event)
            if point is None:
                return True
            picked = drag["picked"]
            try:
                if picked[0] == "vertex":
                    moved = drag["topology"].moved_vertex(picked[1], *point)
                    what = f"corner {picked[1]}"
                else:
                    moved = drag["topology"].moved_point(picked[1], picked[2], *point)
                    what = f"edge {picked[1]}, point {picked[2] + 1}"
            except (ValueError, IndexError):
                return True
            drag["moved"] = point
            self._blocks_sketch(moved, picked)
            self.statusMessage.emit(f"{what}: x {point[0]:.2f}, y {point[1]:.2f} m")
            self.redraw()
            return True
        if kind == "pointer_up" and drag is not None:
            edit["drag"] = None
            try:
                self.controller.enabled = True
            except AttributeError:
                pass
            self.canvas.setCursor(Qt.OpenHandCursor)
            self.view.set_sketch(None)
            self.redraw()
            if drag["moved"] is not None:
                x, y = drag["moved"]
                picked = drag["picked"]
                if picked[0] == "vertex":
                    self.blockVertexMoved.emit(edit["case"], picked[1], x, y)
                else:
                    self.blockPointMoved.emit(edit["case"], picked[1], picked[2], x, y)
            return True
        return False

    def view_centre(self) -> tuple[float, float] | None:
        """The model point (x, y) in the middle of the view: from the top, the one
        under the middle of the screen; else where the line of sight through it meets
        the level of the middle of the model."""
        try:
            size = tuple(self.canvas.get_logical_size())
        except AttributeError:
            size = (self.canvas.width(), self.canvas.height())
        point = self.view.plan_point(size[0] / 2, size[1] / 2, size)
        if point is not None:
            return point
        try:
            centre = self.view._view_centre()
        except Exception:  # noqa: BLE001 (nothing shown yet)
            return None
        return float(centre[0]), float(centre[1])

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
            drawing["turn"] = (
                "Shift" in modifiers and box is not None and drawing.get("purpose") == "domain"
            )
            drawing["box"] = box
            # pressed on an edge (or a corner) of the domain: that edge is dragged
            drawing["edges"] = (
                self._domain_edges(event, point, box)
                if box is not None and drawing.get("purpose") == "domain" and not drawing["turn"]
                else set()
            )
            return
        start = drawing.get("start")
        if start is None:
            return
        if drawing.get("edges"):
            x0, x1, y0, y1 = _moved_edges(drawing["box"], drawing["edges"], point)
            outline = np.array(
                [[x0, y0, level], [x1, y0, level], [x1, y1, level], [x0, y1, level]], float
            )
            self.statusMessage.emit(
                f"Domain x {x0:.2f} … {x1:.2f} m, y {y0:.2f} … {y1:.2f} m "
                f"({x1 - x0:.2f} x {y1 - y0:.2f} m)"
            )
            if kind == "pointer_move":
                self.view.set_outline(outline)
                self.redraw()
                return
            self.stop_domain_drawing()
            if x1 > x0 and y1 > y0:
                self.domainDrawn.emit((x0, x1, y0, y1))
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
            what = str(drawing.get("purpose", "domain"))
            self.statusMessage.emit(
                f"{what[0].upper()}{what[1:]} x {min(sx, px):.2f} … {max(sx, px):.2f} m, "
                f"y {min(sy, py):.2f} … {max(sy, py):.2f} m"
            )
        if kind == "pointer_move":
            self.view.set_outline(outline)
            self.redraw()
            return
        # released: the domain, or the turn
        turn, angle, centre = drawing.get("turn"), drawing.get("angle", 0.0), drawing.get("centre")
        purpose = drawing.get("purpose", "domain")
        self.stop_domain_drawing()
        if turn:
            if angle:
                self.modelTurned.emit(angle, centre)
            return
        (sx, sy), (px, py) = start, point
        if abs(px - sx) > 0 and abs(py - sy) > 0:
            area = (min(sx, px), max(sx, px), min(sy, py), max(sy, py))
            if purpose == "domain":
                self.domainDrawn.emit(area)
            else:
                self.areaDrawn.emit(purpose, area)

    #: how near (pixels) a press must be to an edge of the domain to drag it
    EDGE_PICK = 10.0
    #: the cursor over the edges grabbed: x, y, both (corners)
    EDGE_CURSORS = {
        frozenset({"x0"}): Qt.SizeHorCursor,
        frozenset({"x1"}): Qt.SizeHorCursor,
        frozenset({"y0"}): Qt.SizeVerCursor,
        frozenset({"y1"}): Qt.SizeVerCursor,
        frozenset({"x0", "y0"}): Qt.SizeBDiagCursor,
        frozenset({"x1", "y1"}): Qt.SizeBDiagCursor,
        frozenset({"x0", "y1"}): Qt.SizeFDiagCursor,
        frozenset({"x1", "y0"}): Qt.SizeFDiagCursor,
    }

    def _reach(self, event, pixels: float) -> float | None:
        """How far ``pixels`` on the screen reach in the model, at the pointer."""
        point = self._plan_point(event)
        if point is None:
            return None
        x, y = getattr(event, "x", 0.0), getattr(event, "y", 0.0)
        try:
            size = tuple(self.canvas.get_logical_size())
        except AttributeError:
            size = (self.canvas.width(), self.canvas.height())
        beside = self.view.plan_point(x + pixels, y, size)
        return None if beside is None else math.hypot(beside[0] - point[0], beside[1] - point[1])

    def _domain_edges(self, event, point, box) -> set[str]:
        """The edges of the domain (x0, x1, y0, y1) under the pointer: one, or two at a
        corner; none away from them."""
        reach = self._reach(event, self.EDGE_PICK)
        if reach is None or box is None:
            return set()
        return _edges_near(box, point, reach)

    def _on_move(self, event) -> None:
        if self._tracing is not None:
            self._trace_move(event)
            return
        if self._drawing is not None and self._drawing.get("start") is not None:
            self._draw_domain("pointer_move", event)
        elif self._drawing is not None and self._drawing.get("purpose") == "domain":
            point = self._plan_point(event)  # over an edge: the cursor says it can be dragged
            box = self._domain_box(self._drawing.get("case", ""))
            edges = self._domain_edges(event, point, box) if point is not None else set()
            self.canvas.setCursor(self.EDGE_CURSORS.get(frozenset(edges), Qt.CrossCursor))
        elif self._corridor_edit is not None:
            self._edit_corridor("pointer_move", event)
        elif self._blocks_edit is not None:
            self._edit_blocks("pointer_move", event)

    def _on_pointer(self, event) -> None:
        """A click (press and release without dragging) on a surface probes its values."""
        kind = getattr(event, "type", "")
        if self._drawing is not None:
            self._draw_domain(kind, event)
            return
        if self._tracing is not None:
            self._trace_pointer(kind, event)
            return
        if self._corridor_edit is not None and self._edit_corridor(kind, event):
            return
        if self._blocks_edit is not None and self._edit_blocks(kind, event):
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

    def point_north(self) -> None:
        """North up (the button, N): the view turns so north points up the screen, from
        the top and in 3D; the model's y without a georeference."""
        self._apply_north()
        self.view.point_north()
        self._store_camera()
        self.redraw()

    def _store_camera(self) -> None:
        try:
            self.session.set_camera(self.view.camera_state())
        except Exception:  # noqa: BLE001
            log.warning("could not store the camera:\n%s", traceback.format_exc())
        self._view_moved()

    def _north(self):
        """North in model coordinates for the case shown (its georeference), or None."""
        cases = [i.id for i in self.session.scene if i.kind == "case"]
        if self.case_filter in cases:
            cases = [self.case_filter]
        for case_id in cases:
            north = self.session.data.backgrounds.north(case_id)
            if north is not None:
                return north
        return None

    def _apply_north(self) -> None:
        """From the top, north up (the map as on a map, a turned model turned), when
        the case is georeferenced and "North up" is on."""
        try:
            north = self._north()
        except Exception:  # noqa: BLE001 (a bad georef.json must not stop the drawing)
            log.warning("north of the case:\n%s", traceback.format_exc())
            north = None
        cases = any(i.kind == "case" for i in self.session.scene)
        self.north_button.setVisible(cases)
        self.north_button.setToolTip(
            "Turn the view so north points up the screen (N); the map stays north up when "
            "the model is turned"
            if north is not None
            else "Turn the view so the model's y points up the screen (N): the case has no "
            "georeference (Set-up → Map) to know north by"
        )
        if self.view.set_north(north):
            self._view_moved()
            self.redraw()

    def _view_moved(self) -> None:
        """The view was zoomed, panned or framed: the background maps follow the 2D
        view once it rests."""
        timer = getattr(self, "_view_timer", None)
        if timer is not None:
            timer.start()

    def _follow_view(self) -> None:
        """Tell the background maps what the 2D top view shows (in 3D: nothing, they
        cover the domain and its margin) and draw them again (made anew in a worker
        thread when the area or the zoom changed)."""
        backgrounds = self.session.data.backgrounds
        if not backgrounds.items():
            return
        box = None
        pixels = 0
        if self.two_d and self.two_d_side == "top":
            try:
                size = tuple(self.canvas.get_logical_size())
            except AttributeError:
                size = (self.canvas.width(), self.canvas.height())
            w, h = float(size[0]), float(size[1])
            corners = [
                self.view.plan_point(x, y, size) for x, y in ((0, 0), (w, 0), (0, h), (w, h))
            ]
            if all(c is not None for c in corners) and w > 0 and h > 0:
                xs = [c[0] for c in corners]
                ys = [c[1] for c in corners]
                box = (min(xs), max(xs), min(ys), max(ys))
                pixels = int(max(w, h) * self.devicePixelRatioF())
        backgrounds.set_view(box, pixels)
        self._update_backgrounds()
        self.redraw()


def _same_camera(a: dict, b: dict) -> bool:
    """True if two camera states match (positions, rotations, sizes)."""
    for key in ("position", "rotation", "width", "height", "fov", "zoom"):
        if key in a or key in b:
            if not np.allclose(np.asarray(a.get(key, 0.0)), np.asarray(b.get(key, 0.0))):
                return False
    return True
