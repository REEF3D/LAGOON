# SPDX-License-Identifier: GPL-3.0-or-later
"""LAGOON main window: Blender-style layout with dockable panels.

Scene tree on the left, viewport in the centre, properties on the right, plots,
timeline and Python console along the bottom.
"""

from __future__ import annotations

import logging
import os
import time as time_module
import traceback
from pathlib import Path

import numpy as np
from PySide6.QtCore import QByteArray, QLocale, QPoint, QRect, QSettings, QSize, Qt, QTimer
from PySide6.QtGui import (
    QAction,
    QActionGroup,
    QCloseEvent,
    QIcon,
    QImage,
    QKeySequence,
    QPainter,
)
from PySide6.QtWidgets import (
    QApplication,
    QDockWidget,
    QFileDialog,
    QInputDialog,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QSizePolicy,
    QSplitter,
    QWidget,
)

import lagoonviz
from lagoonviz.core import SESSION_EXTENSION, Session
from lagoonviz.core.data.manager import BackgroundLoader
from lagoonviz.core.kinds import KINDS
from lagoonviz.gui.bridge import SessionBridge
from lagoonviz.gui.icons import icon
from lagoonviz.gui.panels import (
    ConsolePanel,
    PlotsPanel,
    ProfilesPanel,
    PropertiesPanel,
    SceneTreePanel,
    SetupPanel,
    TimelinePanel,
    create_viewport,
)
from lagoonviz.gui.resources import resource_path
from lagoonviz.gui.theme import DEFAULT_THEME, THEME_LOOKS, apply_theme

log = logging.getLogger(__name__)
SESSION_FILTER = f"LAGOON session (*{SESSION_EXTENSION});;All files (*)"
#: bump when the dock layout changes, so stale saved layouts are ignored
LAYOUT_VERSION = 4
#: the items the toolbar adds (the Add menu has them all)
#: items that cut a case's volume (made for the case of the selected item)
VOLUME_TOOLS = ("slice", "iso_surface", "iso_volume", "streamlines", "volume")
#: the window setting of the memory use mode (lagoonviz.core.memory)
MEMORY_KEY = "performance/memory"
TOOLBAR_KINDS = (
    "free_surface",
    "bathymetry",
    "volume",
    "slice",
    "iso_surface",
    "iso_volume",
    "series",
)
#: how often "Follow running simulation" looks for new output
FOLLOW_SECONDS = 5
QWIDGETSIZE_MAX = (1 << 24) - 1  # Qt's "no maximum size"


class MainWindow(QMainWindow):
    def __init__(self, session: Session | None = None, settings: QSettings | None = None):
        super().__init__()
        # numbers as in scripts and papers: decimal point, no digit grouping
        locale = QLocale(QLocale.English, QLocale.UnitedKingdom)
        locale.setNumberOptions(QLocale.OmitGroupSeparator)
        QLocale.setDefault(locale)
        self.session = session or Session()
        self.bridge = SessionBridge(self.session, self)
        self.settings = settings if settings is not None else QSettings()
        self.dirty = False
        #: ask before discarding unsaved changes (tests and batch tools switch this off)
        self.ask_before_discard = True

        self.setObjectName("LagoonMainWindow")
        self.setWindowIcon(QIcon(str(resource_path("icon.png"))))
        self.setDockNestingEnabled(True)
        self.resize(1400, 860)

        # frames are read in background threads and kept in the session's cache
        from lagoonviz.core import memory

        if not os.environ.get("LAGOON_MEMORY"):  # (the environment wins: scripts, tests)
            saved = str(self.settings.value(MEMORY_KEY, memory.DEFAULT_MODE))
            memory.set_mode(saved if saved in memory.MODES else memory.DEFAULT_MODE)
        self.loader = BackgroundLoader(self.session.data, workers=memory.current().loader_workers)

        # panels
        self.viewport = create_viewport(
            self.bridge, self.loader, self, str(self.settings.value("view/look", "studio"))
        )
        self.viewport.statusMessage.connect(lambda text: self.statusBar().showMessage(text, 8000))
        self.viewport.probeChanged.connect(lambda probe: self.plots.set_probe(probe))
        # the 3D views sit in a splitter: one normally, two when comparing cases
        self.viewports = QSplitter(Qt.Horizontal)
        self.viewports.setObjectName("viewports")
        self.viewports.setChildrenCollapsible(False)
        self.viewports.addWidget(self.viewport)
        self.setCentralWidget(self.viewports)
        self.viewport2 = None
        self.scene_tree = SceneTreePanel(self.bridge)
        self.properties = PropertiesPanel(self.bridge)
        self.timeline = TimelinePanel(self.bridge, loader=self.loader, busy=self._views_busy)
        self.timeline.exit_full_screen.clicked.connect(lambda: self.toggle_full_screen(False))
        self.timeline.keyRequested.connect(self.add_camera_key)
        self.plots = PlotsPanel(self.bridge)
        self.profiles = ProfilesPanel(self.bridge)
        self.console = ConsolePanel(self.bridge)
        self.setup_panel = SetupPanel(self.bridge, self.settings)
        # the Map tab draws the domain in the 3D view's 2D top view
        self.setup_panel.map_tab.drawDomain.connect(self.viewport.start_domain_drawing)
        self.viewport.domainDrawn.connect(self.setup_panel.map_tab.domain_drawn)
        self.viewport.modelTurned.connect(self.setup_panel.map_tab.model_turned)
        self.setup_panel.map_tab.northUp.connect(self.viewport.point_north)
        self.setup_panel.map_tab.view_centre = self.viewport.view_centre
        # the Porous and Solids tabs draw a structure's plan there too
        for tab in (self.setup_panel.porous_tab, self.setup_panel.solids_tab):
            tab.drawArea.connect(self.viewport.start_domain_drawing)
            self.viewport.areaDrawn.connect(tab.area_drawn)
        # the River tab: an area drawn, the banks' points dragged there
        river = self.setup_panel.river_tab
        river.drawArea.connect(self.viewport.start_domain_drawing)
        self.viewport.areaDrawn.connect(river.area_drawn)
        river.editPoints.connect(self.viewport.start_corridor_editing)
        river.stopEditing.connect(self.viewport.stop_corridor_editing)
        self.viewport.corridorPointMoved.connect(river.point_moved)
        self.viewport.corridorEditingStopped.connect(river.editing_stopped)
        # the Blocks tab: an area drawn, corners and points dragged, edges clicked there
        blocks = self.setup_panel.blocks_tab
        blocks.drawArea.connect(self.viewport.start_domain_drawing)
        self.viewport.areaDrawn.connect(blocks.area_drawn)
        blocks.editBlocks.connect(self.viewport.start_blocks_editing)
        blocks.stopEditing.connect(self.viewport.stop_blocks_editing)
        self.viewport.blockVertexMoved.connect(blocks.vertex_moved)
        self.viewport.blockPointMoved.connect(blocks.point_moved)
        self.viewport.blockEdgePicked.connect(blocks.edge_picked)
        self.viewport.blocksEditingStopped.connect(blocks.editing_stopped)
        self.setup_panel.map_tab.newCase.connect(self.new_case_from_map_dialog)

        self.docks: dict[str, QDockWidget] = {}
        self._dock("scene", "Scene", self.scene_tree, Qt.LeftDockWidgetArea)
        self._dock("properties", "Properties", self.properties, Qt.RightDockWidgetArea)
        # the set-up's input files beside the properties, as a tab next to them
        self._dock("setup", "Set-up", self.setup_panel, Qt.RightDockWidgetArea)
        self.tabifyDockWidget(self.docks["properties"], self.docks["setup"])
        self.docks["properties"].raise_()
        self._dock("plots", "Plots", self.plots, Qt.BottomDockWidgetArea)
        self._dock("timeline", "Timeline", self.timeline, Qt.BottomDockWidgetArea)
        self._dock("console", "Python console", self.console, Qt.BottomDockWidgetArea)
        self.splitDockWidget(self.docks["plots"], self.docks["timeline"], Qt.Vertical)
        self.splitDockWidget(self.docks["timeline"], self.docks["console"], Qt.Horizontal)
        # free surface lines share the plots' place, as a tab next to them
        self._dock("profiles", "Free surface lines", self.profiles, Qt.BottomDockWidgetArea)
        self.tabifyDockWidget(self.docks["plots"], self.docks["profiles"])
        self.docks["plots"].raise_()
        self._default_state = self.saveState(LAYOUT_VERSION)
        self._sized = False

        self.scene_tree.selectionChanged.connect(self.properties.show_item)
        self.scene_tree.itemPicked.connect(self._item_picked)
        self.bridge.historyChanged.connect(self._on_history)
        self.bridge.sceneChanged.connect(self._on_mode_event)

        self._build_actions()
        self._build_menus()
        self._build_toolbar()
        self._restore_layout()
        self._on_history()
        self._apply_mode()
        self.dirty = False
        self._update_title()
        self.statusBar().showMessage("Ready", 3000)

    # ------------------------------------------------------------- building
    def _dock(self, key: str, title: str, widget: QWidget, area: Qt.DockWidgetArea) -> None:
        dock = QDockWidget(title, self)
        dock.setObjectName(f"dock_{key}")
        dock.setWidget(widget)
        self.addDockWidget(area, dock)
        self.docks[key] = dock

    def _action(self, text: str, slot, shortcut=None, tip: str = "") -> QAction:
        action = QAction(text, self)
        if shortcut is not None:
            action.setShortcut(QKeySequence(shortcut))
        if tip:
            action.setStatusTip(tip)
        action.triggered.connect(slot)
        return action

    def _build_actions(self) -> None:
        self.act_open_case = self._action(
            "Open &case...",
            self.open_case_dialog,
            "Ctrl+Shift+O",
            tip="Open a REEF3D case folder (or one of its REEF3D_* output folders)",
        )
        self.act_open_case.setIconText("Open case")
        self.act_new_case = self._action(
            "&New case...",
            self.new_case_dialog,
            tip="Start a case in a folder: control.txt and ctrl.txt, edited in the Set-up panel",
        )
        self.act_new_case_map = self._action(
            "New case from &map...",
            self.new_case_from_map_dialog,
            tip="Start a case on the map: find the place, draw the domain, add the "
            "bathymetry, then Create case puts it in a folder",
        )
        self.act_open_file = self._action(
            "Open output &file...",
            self.open_output_dialog,
            tip="Open a REEF3D .pvtu/.pvtp/.vtu/.vtp file and the series it belongs to",
        )
        self.act_reload = self._action(
            "&Reload case",
            self.reload_cases,
            QKeySequence.Refresh,
            tip="Read the case folders again: new outputs of a running simulation",
        )
        self.act_reload.setIconText("Reload")
        self.act_follow = self._action(
            "&Follow running simulation",
            self.set_follow,
            tip="Reload by itself when the simulation writes new output, and stay at the end",
        )
        self.act_follow.setCheckable(True)
        # the two modes: the set-up of a case (input files, DIVEMesh) and its results
        self.act_mode_setup = self._action(
            "Set-&up",
            lambda: self.set_mode("setup"),
            "Ctrl+Shift+U",
            tip="Work on the set-up: the input files, DIVEMesh's grid (the results are hidden)",
        )
        self.act_mode_results = self._action(
            "&Results",
            lambda: self.set_mode("results"),
            "Ctrl+Shift+R",
            tip="Look at the results of the runs (the set-up only where ticked)",
        )
        self.act_run_divemesh = self._action(
            "Run &DIVEMesh",
            lambda: self.setup_panel.run_divemesh(),
            tip="Save control.txt and run DIVEMesh in the case folder: its grid, solids and bed",
        )
        self.act_run_divemesh.setIconText("Run DIVEMesh")
        self.act_save_inputs = self._action(
            "Save &input files",
            lambda: self.setup_panel.save(),
            tip="Write the edited ctrl.txt and control.txt of the case shown in the Set-up panel",
        )
        self.mode_group = QActionGroup(self)
        self.mode_group.setExclusive(True)
        for action in (self.act_mode_setup, self.act_mode_results):
            action.setCheckable(True)
            self.mode_group.addAction(action)
        self.act_mode_results.setChecked(True)
        self.act_new = self._action("&New session", self.new_session, QKeySequence.New)
        self.act_open = self._action(
            "&Open session...", self.open_session_dialog, QKeySequence.Open
        )
        self.act_save = self._action("&Save session", self.save_session, QKeySequence.Save)
        self.act_save_as = self._action(
            "Save session &as...", self.save_session_as, QKeySequence.SaveAs
        )
        self.act_export_script = self._action(
            "&Export script...",
            self.export_script_dialog,
            tip="Write the session history as a Python script",
        )
        self.act_quit = self._action("&Quit", self.close, QKeySequence.Quit)
        self.act_undo = self._action("&Undo", self.session.undo, QKeySequence.Undo)
        self.act_redo = self._action("&Redo", self.session.redo, QKeySequence.Redo)
        self.act_undo.setIconText("Undo")
        self.act_redo.setIconText("Redo")
        self.act_remove = self.scene_tree.remove_action
        self.add_actions = {
            kind: self._action(
                spec.label,
                lambda _=False, k=kind: self.add_item(k),
                tip=f"Add a {spec.label.lower()} to the scene",
            )
            for kind, spec in KINDS.items()
        }
        self.act_screenshot = self._action(
            "Save &screenshot...",
            self.save_screenshot_dialog,
            "F12",
            tip="Save a picture of the LAGOON window, including the 3D view",
        )
        self.act_export_image = self._action(
            "Export &image...",
            lambda _checked=False: self.export_image(),  # triggered(checked): no path
            "Ctrl+Shift+I",
            tip="The 3D view alone as an image at a size of your choice (as on the screen, "
            "4K, 8K, a width in cm for print): PNG, JPEG or TIFF",
        )
        self.act_convert = self._action(
            "Convert to LAGOON &format...",
            self.convert_case,
            tip="Write the case's VTU/VTP output as a LAGOON store: several times smaller "
            "and faster to read (again: add the outputs written since)",
        )
        self.act_export_movie = self._action(
            "Export &animation...",
            self.export_animation,
            "Ctrl+Shift+E",
            tip="Film the 3D view over the timeline into an MP4 video",
        )
        self.act_cycles_still = self._action(
            "Render still with &Cycles...",
            self.render_still,
            tip="The 3D view path traced by Blender (reflections, refraction, soft shadows)",
        )
        self.act_export_figure = self._action(
            "Export &figure...",
            self.export_figure,
            "Ctrl+Shift+P",
            tip="The plots, the free surface lines or the 3D view as a paper figure "
            "(PDF, SVG, PNG)",
        )
        self.act_figure_editor = self._action(
            "Figure &editor...",
            self.open_figure_editor,
            tip="Build a figure by dragging time series onto panels; keep it as a template",
        )
        self.act_compare = self._action(
            "&Compare side by side",
            self.toggle_compare,
            "Ctrl+Shift+2",
            tip="Two 3D views, one case each; cameras and time move together",
        )
        self.act_compare.setCheckable(True)
        self.act_full_screen = self._action(
            "&Full screen viewer",
            self.toggle_full_screen,
            # not the system's full screen key (Ctrl+Cmd+F on a Mac): macOS takes that
            # for its own full screen, which keeps all panels
            "Ctrl+Shift+F",
            tip="Only the 3D view and the player, on the whole screen (Esc to leave)",
        )
        self.act_full_screen.setCheckable(True)
        # Esc leaves the full screen viewer (and does nothing otherwise)
        self.act_leave_full_screen = self._action(
            "Leave full screen", lambda: self.toggle_full_screen(False), "Esc"
        )
        self.act_leave_full_screen.setEnabled(False)
        self.act_leave_full_screen.setShortcutContext(Qt.ApplicationShortcut)
        self.addAction(self.act_leave_full_screen)
        self.act_play = self._action(
            "&Play / pause", lambda: self.timeline.play_button.toggle(), "Space"
        )
        self.act_add_key = self._action(
            "Add camera &keyframe",
            self.add_camera_key,
            "Ctrl+K",
            tip="Keep the 3D view at this time; between keyframes the camera moves smoothly",
        )
        self.act_remove_key = self._action(
            "Remove camera keyframe",
            self.remove_camera_key,
            tip="Remove the camera keyframe at this time",
        )
        self.act_camera_path = self._action(
            "Follow camera &path",
            lambda on: self.session.set_camera_path(on),
            tip="The 3D view and films follow the camera path through the keyframes",
        )
        self.act_camera_path.setCheckable(True)
        self.act_orbit = self._action(
            "&Orbit around the scene",
            lambda: self.orbit_camera(),  # not triggered's "checked" as the degrees
            tip="Keyframes that take the 3D view once around the scene over the timeline",
        )
        self.act_clear_keys = self._action(
            "Clear camera keyframes",
            self.session.clear_camera_keys,
            tip="Remove all camera keyframes",
        )
        self.bridge.sceneChanged.connect(self._on_camera_keys)
        self._on_camera_keys("reset", "")
        self.act_sun = self._action(
            "S&un...",
            self.show_sun_dialog,
            tip="Where the light comes from: the 3D view, films and path tracing",
        )
        self.sun_dialog = None
        self.act_legend = self._action(
            "&Legend...",
            self.show_legend_dialog,
            tip="Where films put the time and the colour bars, and the 3D view's bars",
        )
        self.legend_dialog = None
        self.act_reset_layout = self._action("&Reset layout", self.reset_layout)
        self.act_about = self._action("&About LAGOON", self.about)
        self.act_profile = self._action(
            "Record a &performance report",
            self.toggle_profile,
            tip="Record where LAGOON spends its time while you use it; stop to save the report",
        )
        self.act_profile.setCheckable(True)
        self.profiler = None

    def _build_menus(self) -> None:
        bar = self.menuBar()
        file_menu = bar.addMenu("&File")
        file_menu.addAction(self.act_open_case)
        file_menu.addAction(self.act_new_case)
        file_menu.addAction(self.act_new_case_map)
        file_menu.addAction(self.act_open_file)
        file_menu.addAction(self.act_reload)
        file_menu.addAction(self.act_follow)
        file_menu.addSeparator()
        file_menu.addAction(self.act_save_inputs)
        file_menu.addAction(self.act_run_divemesh)
        file_menu.addSeparator()
        for action in (self.act_new, self.act_open, self.act_save, self.act_save_as):
            file_menu.addAction(action)
        file_menu.addSeparator()
        file_menu.addAction(self.act_convert)
        file_menu.addSeparator()
        file_menu.addAction(self.act_export_script)
        file_menu.addAction(self.act_export_figure)
        file_menu.addAction(self.act_figure_editor)
        file_menu.addAction(self.act_export_image)
        file_menu.addAction(self.act_export_movie)
        file_menu.addAction(self.act_cycles_still)
        file_menu.addAction(self.act_screenshot)
        file_menu.addSeparator()
        file_menu.addAction(self.act_quit)

        edit_menu = bar.addMenu("&Edit")
        edit_menu.addAction(self.act_undo)
        edit_menu.addAction(self.act_redo)
        edit_menu.addSeparator()
        edit_menu.addAction(self.act_remove)

        add_menu = bar.addMenu("&Add")
        for action in self.add_actions.values():
            add_menu.addAction(action)

        view_menu = bar.addMenu("&View")
        view_menu.addAction(self.act_mode_setup)
        view_menu.addAction(self.act_mode_results)
        view_menu.addSeparator()
        for dock in self.docks.values():
            view_menu.addAction(dock.toggleViewAction())
        theme_menu = view_menu.addMenu("&Theme")
        current = str(self.settings.value("view/theme", DEFAULT_THEME))
        self.theme_actions = {}
        for name, label in (("light", "&Light"), ("dark", "&Dark")):
            action = theme_menu.addAction(label)
            action.setCheckable(True)
            action.setChecked(name == current)
            action.triggered.connect(lambda _=False, n=name: self.set_theme(n))
            self.theme_actions[name] = action
        look_menu = view_menu.addMenu("&Viewport look")
        current_look = str(self.settings.value("view/look", "studio"))
        self.look_actions = {}
        from lagoonviz.render.cinematic import PRESETS

        looks = [("studio", "&Studio (light)"), ("ocean", "&Ocean (dark blue)"), None]
        looks += [(name, f"{preset.label} (cinematic)") for name, preset in PRESETS.items()]
        for entry in looks:
            if entry is None:  # the engineering looks, then the cinematic ones
                look_menu.addSeparator()
                continue
            name, label = entry
            action = look_menu.addAction(label)
            action.setCheckable(True)
            action.setChecked(name == current_look)
            action.triggered.connect(lambda _=False, n=name: self.set_viewport_look(n))
            self.look_actions[name] = action
        memory_menu = view_menu.addMenu("&Memory use")
        from lagoonviz.core import memory

        self.memory_actions = {}
        for name in memory.MODES:
            action = memory_menu.addAction(memory.LABELS.get(name, name.capitalize()))
            action.setCheckable(True)
            action.setChecked(name == memory.mode())
            action.setToolTip(memory.profile(name).describe())
            action.triggered.connect(lambda _=False, n=name: self.set_memory_use(n))
            self.memory_actions[name] = action
        water_menu = view_menu.addMenu("&White water in Cycles")
        from lagoonviz.gui.export_movie import set_white_water, white_water_setting

        chosen = white_water_setting(self)
        self.white_water_actions = {}
        for name in ("none", "light", "medium", "heavy"):
            action = water_menu.addAction(name.capitalize())
            action.setCheckable(True)
            action.setChecked(name == chosen)
            action.setToolTip("Foam, spray and bubbles in path-traced stills and films")
            action.triggered.connect(lambda _=False, n=name: set_white_water(self, n))
            self.white_water_actions[name] = action
        view_menu.addAction(self.act_sun)
        view_menu.addAction(self.act_legend)
        view_menu.addSeparator()
        view_menu.addAction(self.act_compare)
        view_menu.addAction(self.act_full_screen)
        view_menu.addAction(self.act_play)
        camera_menu = view_menu.addMenu("&Camera path")
        camera_menu.addAction(self.act_add_key)
        camera_menu.addAction(self.act_remove_key)
        camera_menu.addAction(self.act_camera_path)
        camera_menu.addAction(self.act_orbit)
        camera_menu.addSeparator()
        camera_menu.addAction(self.act_clear_keys)
        view_menu.addAction(self.act_screenshot)
        view_menu.addAction(self.act_reset_layout)

        help_menu = bar.addMenu("&Help")
        help_menu.addAction(self.act_profile)
        help_menu.addAction(self.act_about)

    def _build_toolbar(self) -> None:
        self._set_icons()
        self.act_open.setIconText("Open session")
        self.act_save.setIconText("Save")
        self.act_screenshot.setIconText("Screenshot")
        bar = self.toolbar = self.addToolBar("Main")
        bar.setObjectName("toolbar_main")
        bar.setMovable(False)
        bar.setIconSize(QSize(18, 18))
        bar.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        bar.addAction(self.act_open_case)
        bar.addAction(self.act_reload)
        bar.addSeparator()
        bar.addAction(self.act_mode_setup)
        bar.addAction(self.act_mode_results)
        # only in the set-up: QToolBar hides the action's button with the action
        self.run_divemesh_button = bar.addAction(self.act_run_divemesh)
        bar.addSeparator()
        bar.addAction(self.act_open)
        bar.addAction(self.act_save)
        bar.addSeparator()
        bar.addAction(self.act_undo)
        bar.addAction(self.act_redo)
        bar.addSeparator()
        for kind in TOOLBAR_KINDS:
            bar.addAction(self.add_actions[kind])
        spacer = QWidget()
        spacer.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        bar.addWidget(spacer)
        bar.addAction(self.act_screenshot)

    # ------------------------------------------------------------- layout
    def _restore_layout(self) -> None:
        geometry = self.settings.value("window/geometry")
        state = self.settings.value("window/state")
        if isinstance(geometry, QByteArray):
            self.restoreGeometry(geometry)
        self._restored = isinstance(state, QByteArray) and self.restoreState(state, LAYOUT_VERSION)

    def _apply_default_sizes(self) -> None:
        self.resizeDocks([self.docks["scene"], self.docks["properties"]], [260, 300], Qt.Horizontal)
        self.resizeDocks([self.docks["plots"], self.docks["timeline"]], [240, 80], Qt.Vertical)
        self.resizeDocks([self.docks["timeline"], self.docks["console"]], [700, 500], Qt.Horizontal)

    def showEvent(self, event) -> None:  # noqa: N802 (Qt naming)
        super().showEvent(event)
        # dock sizes only apply once the window has a size; saved layouts keep their own
        if not self._sized and not self._restored:
            self._apply_default_sizes()
        self._sized = True

    def _set_icons(self) -> None:
        for action, name in (
            (self.act_open_case, "open_case"),
            (self.act_open, "open"),
            (self.act_save, "save"),
            (self.act_undo, "undo"),
            (self.act_redo, "redo"),
            (self.act_screenshot, "screenshot"),
            (self.act_reload, "reload"),
            (self.act_mode_setup, "setup"),
            (self.act_mode_results, "results"),
        ):
            action.setIcon(icon(name))
        for kind in TOOLBAR_KINDS:
            self.add_actions[kind].setIcon(icon(kind))

    def _streamline_start(self) -> dict:
        """New streamlines where the user is looking: in the plane of the selected
        slice, or on the selected free surface; else through the volume."""
        selected = self.scene_tree.selected_id()
        if selected and selected in self.session.scene:
            item = self.session.scene.get(selected)
            if item.kind == "slice":
                return {
                    "seed_normal": item.properties["normal"],
                    "seed_position": item.properties["position"],
                    "lines_in": "seed plane",
                }
            if item.kind == "free_surface":
                return {"lines_in": "free surface"}
        return {}

    # ------------------------------------------------------------- modes
    def set_mode(self, mode: str) -> None:
        """Set-up or results (the toolbar's two buttons, View menu)."""
        if mode != self.session.mode:
            self.session.set_mode(mode)
        self._apply_mode()

    def _on_mode_event(self, event: str, _item_id: str) -> None:
        if event in ("mode", "reset"):
            self._apply_mode()

    def _item_picked(self, item_id: str) -> None:
        """An item clicked in the scene: its properties come to the front (the Set-up
        panel shares their place). In the set-up, a case and its set-up items stay
        with the Set-up panel, which edits them."""
        if not item_id or item_id not in self.session.scene or self.full_screen:
            return
        kind = self.session.scene.get(item_id).kind
        if self.session.mode == "setup" and kind in ("case", "setup", "grid"):
            return
        if self.session.mode == "setup" and kind in ("corridor", "multiblock"):
            panel = self.setup_panel  # edited in the River or the Blocks tab
            panel.tabs.setCurrentWidget(panel.river_tab if kind == "corridor" else panel.blocks_tab)
            if "setup" in self.docks:
                self.docks["setup"].show()
                self.docks["setup"].raise_()
            return
        dock = self.docks["properties"]
        dock.show()
        dock.raise_()

    def _apply_mode(self) -> None:
        """The window for the mode: the timeline rests in the set-up."""
        setup = self.session.mode == "setup"
        self.act_mode_setup.setChecked(setup)
        self.act_mode_results.setChecked(not setup)
        if setup and self.timeline.play_button.isChecked():
            self.timeline.play_button.setChecked(False)
        self.timeline.setEnabled(not setup)
        self.act_play.setEnabled(not setup)
        self.act_run_divemesh.setVisible(setup)
        switched = getattr(self, "_shown_mode", "results") != self.session.mode
        if setup and not self.full_screen:
            dock = self.docks["setup"]
            dock.show()
            dock.raise_()
        elif switched and not self.full_screen:
            # back to the results: the Set-up panel makes way for Properties (View →
            # Set-up shows it again, e.g. to edit the input files of a case that has run)
            self.docks["setup"].hide()
            dock = self.docks["properties"]
            dock.show()
            dock.raise_()
        if switched:
            self._switch_view(getattr(self, "_shown_mode", "results"), self.session.mode)
            self.statusBar().showMessage(
                "Set-up: the input files and DIVEMesh's grid; the results are hidden"
                if setup
                else "Results: what the runs wrote",
                6000,
            )
        self._shown_mode = self.session.mode

    def _switch_view(self, old: str, new: str) -> None:
        """Each mode keeps its own view: the set-up its 2D map, the results their 3D
        view. The view of the mode left is kept; the one of the mode entered comes
        back (the results start in 3D, framed)."""
        import copy

        scene = self.session.scene
        views = self.__dict__.setdefault("_mode_views", {})
        views[old] = (scene.plan_view, scene.plan_side, copy.deepcopy(scene.camera))
        saved = views.get(new)
        try:
            if saved is not None:
                plan, side, camera = saved
                self.viewport.set_2d(plan, side if plan else None)
                if camera:
                    self.session.set_camera(camera)
                    view = getattr(self.viewport, "view", None)
                    if view is not None:
                        view.set_camera_state(camera)
                        self.viewport.redraw()
            elif new == "results" and scene.plan_view:
                self.viewport.set_2d(False)  # the 3D view, framed
                self.viewport.frame("perspective")
        except Exception:  # noqa: BLE001 (the mode switches all the same)
            log.warning("view of the %s:\n%s", new, traceback.format_exc())

    def _suggest_streaming(self, case) -> None:
        """A very large case (millions of points a frame) opened while not streaming:
        say that streaming plays it smoothly."""
        from lagoonviz.core import memory

        if memory.mode() == "streaming":
            return
        points = max((d.points_per_frame() or 0 for d in case.main_datasets().values()), default=0)
        if points >= memory.LARGE_POINTS:
            self.statusBar().showMessage(
                f"A large case: {points / 1e6:.0f} million points a frame. View → Memory use "
                "→ Streaming plays it at a coarser detail (full detail when it stops)",
                20000,
            )

    def set_memory_use(self, name: str) -> None:
        """View → Memory use: performance (in memory, fast), balanced, low memory
        (large cases fit, slower) or streaming (very large cases: coarser while
        playing); kept for the next start."""
        from lagoonviz.core import memory

        use = memory.set_mode(name)
        self.settings.setValue(MEMORY_KEY, name)
        self.session.data.cache.trim()
        self.loader.set_workers(use.loader_workers)
        for key, action in self.memory_actions.items():
            action.setChecked(key == name)
        self.statusBar().showMessage(f"Memory use {use.describe()}", 8000)

    def add_item(self, kind: str) -> str:
        """Add an item from the menu or toolbar (slices start in the middle of the volume).
        With several cases open, slices, iso-surfaces and streamlines cut the case of
        the selected item (or the one the active view shows), if it has a volume."""
        case = self._target_case() if kind in VOLUME_TOOLS else {}
        if kind in ("iso_surface", "iso_volume", "slice", "streamlines"):
            case = {**case, **self._target_dataset(kind)}
        if kind == "slice":
            return self.session.add_slice("y", **case)
        if kind == "iso_surface":
            return self.session.add_iso_surface(**case)
        if kind == "iso_volume":
            return self.session.add_iso_volume(**case)
        if kind == "streamlines":
            return self.session.add_streamlines(**self._streamline_start(), **case)
        if kind == "volume":  # the volume a case opened with, switched off: show it
            wanted = case.get("case")
            hidden = [
                i
                for i in self.session.scene
                if i.kind == "volume"
                and not i.visible
                and (not wanted or self.session.data.case_item(i.id).id == wanted)
            ]
            if hidden:
                self.session.set_visible(hidden[0].id, True)
                return hidden[0].id
            return self.session.add_item(kind, **case)
        return self.session.add_item(kind)

    def _target_dataset(self, kind: str) -> dict:
        """``{"case": id, "dataset": key}`` of the data the selected item shows, when
        a new item of this kind can be made of it: iso-surfaces and iso-volumes of a
        volume or a surface (free surface, bed: iso-lines, iso-patches), slices and
        streamlines of a volume. Selected: a volume, a surface, or a slice or iso item
        of a volume (the same volume). Else nothing: the case's own volume."""
        data = self.session.data
        selected = self.scene_tree.selected_id()
        if not selected or selected not in self.session.scene:
            return {}
        dataset = data.dataset(selected)
        if dataset is None:
            return {}
        volume = dataset.kind == "UnstructuredGrid"
        surface = getattr(dataset, "surface_like", False) or dataset.role == "amr"
        wanted = volume or (surface and kind in ("iso_surface", "iso_volume"))
        if not wanted:
            return {}
        case_item = data.case_item(selected)
        found = {"dataset": dataset.key}
        if case_item is not None and sum(i.kind == "case" for i in self.session.scene) > 1:
            found["case"] = case_item.id
        return found

    def _target_case(self) -> dict:
        """``{"case": id}`` for an item that cuts a volume, when several cases are
        open: the selected item's case, else the case the second view shows when it
        has the focus, if that case has a volume (else the first with one)."""
        data = self.session.data
        if sum(i.kind == "case" for i in self.session.scene) < 2:
            return {}
        prefer = self.scene_tree.selected_id()
        second = self.viewport2
        if second is not None and not second.isHidden() and second.case_filter:
            if second.hasFocus() or second.isAncestorOf(self.focusWidget()):
                prefer = second.case_filter
        if prefer and prefer not in self.session.scene:
            prefer = None
        case_id = data.volume_case(prefer)
        return {"case": case_id} if case_id else {}

    # ------------------------------------------------------------- compare
    def _views_busy(self) -> bool:
        second = self.viewport2
        return self.viewport.busy or (second is not None and not second.isHidden() and second.busy)

    def toggle_compare(self, on: bool | None = None) -> None:
        """Two 3D views side by side, one case each, cameras linked (or back to one)."""
        on = self.act_compare.isChecked() if on is None else bool(on)
        self.act_compare.setChecked(on)
        if on and self.viewport2 is None:
            self.viewport2 = create_viewport(
                self.bridge, self.loader, self, str(self.settings.value("view/look", "studio"))
            )
            self.viewport2.setObjectName("viewport2")
            self.viewport2.statusMessage.connect(
                lambda text: self.statusBar().showMessage(text, 8000)
            )
            self.viewport2.probeChanged.connect(lambda probe: self.plots.set_probe(probe))
            self.viewports.addWidget(self.viewport2)
            self.viewport.partner = self.viewport2
            self.viewport2.partner = self.viewport
        if self.viewport2 is None:
            return
        cases = [item.id for item in self.session.scene if item.kind == "case"]
        if on:
            self.viewport2.show()
            self.viewport.set_compare(True)
            self.viewport2.set_compare(True)
            if len(cases) >= 2:
                self.viewport.set_case_filter(cases[0])
                self.viewport2.set_case_filter(cases[1])
            camera = getattr(self.viewport, "view", None)
            if camera is not None and hasattr(self.viewport2, "view"):
                self.viewport2.view.camera.set_state(camera.camera.get_state())
            width = max(self.viewports.width(), 2)
            self.viewports.setSizes([width // 2, width - width // 2])
        else:
            self.viewport.set_compare(False)
            self.viewport2.set_compare(False)
            self.viewport2.hide()
        self.statusBar().showMessage(
            "Comparing: pick the case of each view in its header" if on else "One 3D view",
            8000,
        )

    # ------------------------------------------------------------- camera path
    def add_camera_key(self) -> int:
        """A camera keyframe: the 3D view as it is now, at the current time.
        Returns the number of keys (0 without a 3D view)."""
        view = getattr(self.viewport, "view", None)
        if view is not None:
            self.session.set_camera(view.camera_state())
        try:
            count = self.session.add_camera_key()
        except ValueError:
            self.statusBar().showMessage("No 3D view to take a camera keyframe from", 5000)
            return 0
        self.statusBar().showMessage(
            f"Camera keyframe at t = {self.session.scene.time:.2f} s ({count} in all)", 5000
        )
        return count

    def orbit_camera(self, degrees: float = 360.0) -> int:
        """Keyframes that take the 3D view as it is now once around the middle of what
        it shows, over the whole timeline (they replace the keyframes there were).
        Returns the number of keys (0 if there is nothing to go around)."""
        if self.session.scene.camera_keys and self.ask_before_discard:
            answer = QMessageBox.question(
                self,
                "Orbit around the scene",
                "The orbit replaces the camera keyframes there are. Go on?",
                QMessageBox.Ok | QMessageBox.Cancel,
                QMessageBox.Cancel,
            )
            if answer != QMessageBox.Ok:
                return 0
        view = getattr(self.viewport, "view", None)
        centre = None
        if view is not None:
            self.session.set_camera(view.camera_state())
            corners = view.bounds()
            if corners is not None:
                centre = ((corners[0] + corners[1]) / 2).tolist()
        try:
            count = self.session.orbit_camera(degrees, centre=centre)
        except ValueError as error:
            self.statusBar().showMessage(f"No orbit: {error}", 8000)
            return 0
        self.statusBar().showMessage(
            f"Camera orbit: {count} keyframes over the timeline; play to see it", 8000
        )
        return count

    def remove_camera_key(self) -> bool:
        """Remove the camera keyframe on the current frame (if there is one)."""
        when = self.timeline.key_here()
        if when is None:
            self.statusBar().showMessage("No camera keyframe at this time", 5000)
            return False
        return self.session.remove_camera_key(when)

    def _on_camera_keys(self, event: str, _item: str) -> None:
        if event not in ("camera_keys", "reset"):
            return
        keys = bool(self.session.scene.camera_keys)
        self.act_camera_path.blockSignals(True)
        self.act_camera_path.setChecked(keys and self.session.scene.camera_path)
        self.act_camera_path.blockSignals(False)
        for action in (self.act_remove_key, self.act_camera_path, self.act_clear_keys):
            action.setEnabled(keys)
        if event == "camera_keys":  # keyframes are work worth saving (not undo steps)
            self.dirty = True
            self._update_title()

    def show_sun_dialog(self):
        """View → Sun: sliders for where the sun is; the 3D view follows."""
        from lagoonviz.gui.sun_dialog import SunDialog

        if self.sun_dialog is None:
            self.sun_dialog = SunDialog(
                self,
                self.session,
                lambda: getattr(getattr(self.viewport, "view", None), "style_name", None),
            )
        self.sun_dialog.show_sun()
        self.sun_dialog.show()
        self.sun_dialog.raise_()
        return self.sun_dialog

    def show_legend_dialog(self):
        """View → Legend: the films' time and colour bars, the 3D view's bars."""
        from lagoonviz.gui.legend_dialog import LegendDialog

        if self.legend_dialog is None:
            self.legend_dialog = LegendDialog(self, self.session)
        self.legend_dialog.show_layout()
        self.legend_dialog.show()
        self.legend_dialog.raise_()
        return self.legend_dialog

    def set_viewport_look(self, name: str) -> None:
        """Background and lighting of the 3D view; remembered for the next start."""
        self.settings.setValue("view/look", name)
        for key, action in self.look_actions.items():
            action.setChecked(key == name)
        for view in (self.viewport, self.viewport2):
            if view is not None:
                view.set_style(name)

    def set_theme(self, name: str) -> None:
        """Switch the whole window to the light or dark theme, viewport included."""
        self.settings.setValue("view/theme", name)
        for key, action in self.theme_actions.items():
            action.setChecked(key == name)
        apply_theme(QApplication.instance(), name)
        self._set_icons()
        self.timeline.refresh_icons()
        self.set_viewport_look(THEME_LOOKS.get(name, "studio"))
        for view in (self.viewport, self.viewport2):
            if view is None:
                continue
            if hasattr(view, "apply_theme_colors"):
                view.apply_theme_colors()
            view.refresh()  # gauges take the new plot colours
        # only this window's own widgets (floating panels included): allWidgets()
        # also hands out widgets of closed windows that Qt is deleting, and
        # touching those crashed (segfault in the test suite)
        self.update()
        for widget in self.findChildren(QWidget):
            widget.update()

    def reset_layout(self) -> None:
        self.restoreState(self._default_state, LAYOUT_VERSION)
        for dock in self.docks.values():
            dock.show()
        self._apply_default_sizes()

    def closeEvent(self, event: QCloseEvent) -> None:  # noqa: N802 (Qt naming)
        if not self.confirm_discard():
            event.ignore()
            return
        if self.ask_before_discard and not self.setup_panel.ask_to_save(self):
            event.ignore()
            return
        if self.full_screen:
            # closed in the full screen viewer: keep the window as it was before it,
            # panels and toolbar included, for the next start
            state, _maximized, geometry, _shown = self._before_full_screen
        else:
            state, geometry = self.saveState(LAYOUT_VERSION), self.saveGeometry()
        self.settings.setValue("window/geometry", geometry)
        self.settings.setValue("window/state", state)
        self.timeline.timer.stop()
        self.loader.shutdown()
        event.accept()

    # ------------------------------------------------------------- history and title
    def _on_history(self) -> None:
        stack = self.session.stack
        self.act_undo.setEnabled(stack.can_undo)
        self.act_redo.setEnabled(stack.can_redo)
        self.act_undo.setText(f"&Undo {stack.undo_label}" if stack.undo_label else "&Undo")
        self.act_redo.setText(f"&Redo {stack.redo_label}" if stack.redo_label else "&Redo")
        self.act_undo.setIconText("Undo")
        self.act_redo.setIconText("Redo")
        self.act_undo.setToolTip(self.act_undo.text().replace("&", ""))
        self.act_redo.setToolTip(self.act_redo.text().replace("&", ""))
        self.dirty = True
        self._update_title()

    def _update_title(self) -> None:
        name = self.session.path.name if self.session.path else "Untitled"
        self.setWindowTitle(f"{name}{' *' if self.dirty else ''} - LAGOON")

    # ------------------------------------------------------------- files
    def confirm_discard(self) -> bool:
        """Ask before throwing away unsaved changes. Returns True to go ahead."""
        changed = self.session.stack.can_undo or self.session.scene.camera_keys
        if not self.ask_before_discard or not self.dirty or not changed:
            return True
        answer = QMessageBox.question(
            self,
            "Unsaved changes",
            "The session has unsaved changes. Discard them?",
            QMessageBox.Discard | QMessageBox.Cancel,
            QMessageBox.Cancel,
        )
        return answer == QMessageBox.Discard

    def new_session(self) -> None:
        if self.confirm_discard():
            self.session.reset()
            self.dirty = False
            self._update_title()

    def open_case_dialog(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "Open REEF3D case", self._last_dir())
        if path:
            self.open_case(path)

    def new_case_dialog(self) -> str | None:
        """File → New case: a folder (made in the dialog) for a case's input files."""
        path = QFileDialog.getExistingDirectory(
            self, "Folder of the new case (make one with New Folder)", self._last_dir()
        )
        if not path:
            return None
        try:
            case_id = self.session.new_case(path)
        except (OSError, ValueError) as error:
            QMessageBox.warning(self, "Could not start the case", str(error))
            return None
        self._remember_dir(path)
        self.setup_panel.refresh()
        dock = self.docks["setup"]
        dock.show()
        dock.raise_()
        self.statusBar().showMessage(
            f"New case {Path(path).name}: add DIVEMesh's commands (control.txt: domain B 10, "
            "cells B 1) and REEF3D's (ctrl.txt) in the Set-up panel, then Save",
            15000,
        )
        return case_id

    def new_case_from_map_dialog(self) -> str | None:
        """File → New case from map: where (a place name or latitude, longitude), then
        a draft case there, with a map, in the set-up's 2D view and the Map tab."""
        from lagoonviz.core.geo.places import search

        text, ok = QInputDialog.getText(
            self,
            "New case from map",
            "Where? A place name (Nidelva; Nidelva, Arendal: name, municipality) or "
            "latitude, longitude (63.44, 10.40). Leave it empty to find the place later "
            "in the Map tab.",
            QLineEdit.EchoMode.Normal,
            "Trondheim",
        )
        if not ok:
            return None
        place = None
        if text.strip():
            QApplication.setOverrideCursor(Qt.WaitCursor)
            try:
                found = search(text, near=self.session.location() or (63.4365, 10.398))
            except OSError as error:
                found = []
                QMessageBox.warning(self, "Place search", f"{error}\nType latitude, longitude.")
            finally:
                QApplication.restoreOverrideCursor()
            if len(found) > 1:
                labels = [p.label() for p in found]
                chosen, ok = QInputDialog.getItem(
                    self, "New case from map", "Which one?", labels, 0, False
                )
                if not ok:
                    return None
                found = [found[labels.index(chosen)]]
            place = found[0] if found else None
            if place is None:
                self.statusBar().showMessage(f"Nothing found for {text!r}", 8000)
        try:
            case_id = self.session.new_case_from_map(place)
        except (OSError, ValueError) as error:
            QMessageBox.warning(self, "Could not start the case", str(error))
            return None
        self.setup_panel.refresh()
        dock = self.docks["setup"]
        dock.show()
        dock.raise_()
        self.setup_panel.tabs.setCurrentWidget(self.setup_panel.map_tab)
        self.statusBar().showMessage(
            "A draft case on the map: Draw on the map (the domain), Bathymetry (geo.dat), "
            "then Create case… puts it in a folder",
            15000,
        )
        return case_id

    def open_output_dialog(self) -> None:
        """File → Open output file: the folder's files grouped into series (as
        ParaView lists them); a series opens with its case."""
        from lagoonviz.gui.open_output import choose_output

        path = choose_output(self._last_dir(), self)
        if path:
            self.open_case(path)

    def open_case(self, path: str | Path) -> str | None:
        """Open a REEF3D case into the session; returns the case item id."""
        self.statusBar().showMessage(f"Opening {path} ...")
        try:
            case_id = self.session.open_case(path)
        except (OSError, ValueError) as error:
            QMessageBox.warning(self, "Could not open case", str(error))
            self.statusBar().clearMessage()
            return None
        self._remember_dir(path)
        self.timeline.update_range()  # the whole case is in now: first output, end
        case = self.session.data.case(case_id)
        summary = ", ".join(f"{d.label} ({len(d)} frames)" for d in case.datasets.values())
        self.statusBar().showMessage(f"Opened {Path(path).name}: {summary}", 10000)
        self._suggest_streaming(case)
        self.viewport.reframe()
        if self.viewport2 is not None:
            self.viewport2.reframe()
        self.scene_tree.select(self._main_item(case_id))
        return case_id

    def _main_item(self, case_id: str) -> str:
        """The item to show in the properties after opening a case: its free surface,
        else its bed, else the case itself (not the last item added, a gauge plot)."""
        data = self.session.data
        for kind in ("amr", "free_surface", "bathymetry"):
            for item in self.session.scene:
                if item.kind == kind and getattr(data.case_item(item.id), "id", None) == case_id:
                    return item.id
        return case_id

    def open_session_dialog(self) -> None:
        if not self.confirm_discard():
            return
        path, _ = QFileDialog.getOpenFileName(
            self, "Open session", self._last_dir(), SESSION_FILTER
        )
        if path:
            self.open_session(path)

    def open_session(self, path: str | Path) -> None:
        try:
            self.session.load(path)
        except (OSError, ValueError, KeyError) as error:
            QMessageBox.warning(self, "Could not open session", str(error))
            return
        self._remember_dir(path)
        self.dirty = False
        self._update_title()
        self.statusBar().showMessage(f"Opened {path}", 5000)

    def save_session(self) -> bool:
        if self.session.path is None:
            return self.save_session_as()
        return self.save_session_to(self.session.path)

    def save_session_as(self) -> bool:
        path, _ = QFileDialog.getSaveFileName(
            self, "Save session", self._last_dir(), SESSION_FILTER
        )
        if not path:
            return False
        if not Path(path).suffix:
            path += SESSION_EXTENSION
        return self.save_session_to(path)

    def save_session_to(self, path: str | Path) -> bool:
        try:
            self.session.save(path)
        except OSError as error:
            QMessageBox.warning(self, "Could not save session", str(error))
            return False
        self._remember_dir(path)
        self.dirty = False
        self._update_title()
        self.statusBar().showMessage(f"Saved {path}", 5000)
        return True

    def export_script_dialog(self) -> None:
        path, _ = QFileDialog.getSaveFileName(
            self, "Export script", self._last_dir(), "Python script (*.py)"
        )
        if path:
            self.export_script(path)

    def export_script(self, path: str | Path) -> None:
        Path(path).write_text(self.session.to_script(), encoding="utf-8")
        self.statusBar().showMessage(f"Script written to {path}", 5000)

    def export_figure(self) -> str | None:
        """File → Export figure: a paper figure of the plots, profiles or 3D view."""
        from lagoonviz.gui.export_figure import export_figure

        return export_figure(self)

    def open_figure_editor(self):
        """File → Figure editor: a figure built by hand, kept as a template."""
        from lagoonviz.gui.figure_editor import open_figure_editor

        return open_figure_editor(self)

    def export_image(self, path: str | Path | None = None, **settings) -> str | None:
        """File → Export image: the 3D view alone at a size of one's choice."""
        from lagoonviz.gui.export_image import export_image

        return export_image(self, path, **settings)

    def export_animation(self) -> str | None:
        """File → Export animation: an MP4 of the 3D view over the timeline."""
        from lagoonviz.gui.export_movie import export_animation

        return export_animation(self)

    def render_still(self) -> str | None:
        """File → Render still with Cycles: the 3D view path traced by Blender."""
        from lagoonviz.gui.export_movie import render_still

        return render_still(self)

    # ------------------------------------------------------------- full screen
    @property
    def full_screen(self) -> bool:
        return getattr(self, "_before_full_screen", None) is not None

    def toggle_full_screen(self, on: bool | None = None) -> None:
        """The 3D view and the player alone on the whole screen, or back to the
        window as it was (panels, toolbar, size)."""
        on = (not self.full_screen) if on is None else bool(on)
        if on == self.full_screen:
            self.act_full_screen.setChecked(on)
            return
        if on:
            self._before_full_screen = (
                self.saveState(LAYOUT_VERSION),
                self.isMaximized(),
                self.saveGeometry(),
                [name for name, dock in self.docks.items() if not dock.isHidden()],
            )
            for name, dock in self.docks.items():
                if name != "timeline":
                    dock.hide()
            self.docks["timeline"].show()
            self.toolbar.hide()
            self.statusBar().hide()
            self.timeline.exit_full_screen.show()
            # the player takes only the height it needs, without its title bar
            dock = self.docks["timeline"]
            dock.setTitleBarWidget(QWidget())
            dock.setFixedHeight(self.timeline.sizeHint().height())
            self.showFullScreen()
        else:
            state, maximized, geometry, shown = self._before_full_screen
            self._before_full_screen = None
            dock = self.docks["timeline"]
            dock.setTitleBarWidget(None)
            dock.setMinimumHeight(0)
            dock.setMaximumHeight(QWIDGETSIZE_MAX)
            self.timeline.exit_full_screen.hide()
            self.toolbar.show()
            self.statusBar().show()
            for name in shown:
                self.docks[name].show()
            self.showNormal()
            if maximized:
                self.showMaximized()
            else:
                self.restoreGeometry(geometry)

            def restore() -> None:
                # sizes and tabs as before; after macOS has finished leaving full screen
                # (while it animates, the window does not have its size yet)
                if self.full_screen:
                    return  # back in full screen meanwhile
                self.restoreState(state, LAYOUT_VERSION)
                for name in shown:
                    self.docks[name].show()

            restore()
            QTimer.singleShot(700, self, restore)  # dropped if the window goes first
        self.act_full_screen.setChecked(on)
        self.act_leave_full_screen.setEnabled(on)

    # ------------------------------------------------------------- reload
    def convert_case(self, ask: bool = True):
        """Convert the selected case to a LAGOON store (in the background)."""
        from lagoonviz.gui.convert_dialog import convert_case

        return convert_case(self, ask=ask)

    def reload_cases(self) -> list[str]:
        """Read the case folders again, keeping items, time and camera. At the last
        frame, it stays at the (new) last frame, so a running simulation can be
        followed."""
        timeline = self.timeline
        at_end = timeline.has_frames and timeline.frame_of(self.session.scene.time) >= (
            timeline.n_frames - 1
        )
        before = timeline.n_frames
        paths = self.session.reload()
        self._signatures = self._case_signatures()
        timeline.update_range()
        if at_end and timeline.n_frames > before:
            timeline.go_to_frame(timeline.n_frames - 1)
        added = timeline.n_frames - before
        news = f", {added} new output{'s' if added != 1 else ''}" if added > 0 else ""
        self.statusBar().showMessage(f"Reloaded {len(paths)} case(s){news}", 6000)
        return paths

    def _case_signatures(self) -> dict[str, tuple]:
        return {
            item.id: self.session.data.signature(item.id)
            for item in self.session.scene
            if item.kind == "case"
        }

    def set_follow(self, on: bool) -> None:
        """Watch the case folders every few seconds and reload when they change."""
        if not hasattr(self, "_follow_timer"):
            self._follow_timer = QTimer(self)
            self._follow_timer.setInterval(FOLLOW_SECONDS * 1000)
            self._follow_timer.timeout.connect(self._check_for_new_output)
        self.act_follow.setChecked(bool(on))
        if on:
            self._signatures = self._case_signatures()
            self._follow_timer.start()
        else:
            self._follow_timer.stop()

    def _check_for_new_output(self) -> None:
        if self.timeline.play_button.isChecked():
            return  # not in the middle of playing
        current = self._case_signatures()
        if current != getattr(self, "_signatures", None):
            self.reload_cases()

    def save_screenshot_dialog(self) -> None:
        path, _ = QFileDialog.getSaveFileName(
            self, "Save screenshot", self._last_dir(), "PNG image (*.png)"
        )
        if path:
            if not Path(path).suffix:
                path += ".png"
            self.save_screenshot(path)

    def screenshot(self) -> QImage:
        """The window as an image, with the GPU-drawn 3D view pasted in.

        Qt cannot read back the GPU canvas, so the last frame pygfx rendered is
        drawn over the canvas area of the window grab.
        """
        image = self.grab().toImage()
        views = [self.viewport]
        if self.viewport2 is not None and not self.viewport2.isHidden():
            views.append(self.viewport2)
        for view in views:
            snapshot = view.canvas_snapshot()
            if snapshot is None:
                continue
            widget, pixels = snapshot
            pixels = np.ascontiguousarray(np.asarray(pixels)[..., :4], dtype=np.uint8)
            height, width = pixels.shape[:2]
            frame = QImage(pixels.data, width, height, 4 * width, QImage.Format_RGBA8888)
            top_left = widget.mapTo(self, QPoint(0, 0))
            painter = QPainter(image)
            painter.drawImage(QRect(top_left, widget.size()), frame)
            painter.end()
        return image

    def save_screenshot(self, path: str | Path) -> bool:
        ok = self.screenshot().save(str(path), "PNG")
        if ok:
            self._record_screenshot(path)
        self.statusBar().showMessage(
            f"Screenshot saved to {path}" if ok else f"Could not save {path}", 5000
        )
        return ok

    def _record_screenshot(self, path: str | Path) -> None:
        """The screenshot into the recorded script: there, the 3D view as it is now."""
        from lagoonviz.render.script import screenshot_code

        view = getattr(self.viewport, "view", None)
        canvas = getattr(self.viewport, "canvas", self.viewport)
        try:
            camera = view.camera_state() if view is not None else self.session.scene.camera
            scene = self.session.scene
            code = screenshot_code(
                path,
                size=(canvas.width(), canvas.height()),
                time=scene.time,
                style=getattr(view, "style_name", "studio"),
                camera=camera,
                plan=(scene.plan_view, scene.plan_side),
            )
        except Exception:  # noqa: BLE001 (the screenshot is saved; the script misses it)
            return
        self.session.record_output(code)

    def _last_dir(self) -> str:
        return str(self.settings.value("files/last_dir", str(Path.home())))

    def _remember_dir(self, path: str | Path) -> None:
        self.settings.setValue("files/last_dir", str(Path(path).resolve().parent))

    # ------------------------------------------------------------- help
    def toggle_profile(self, on: bool | None = None, path: str | Path | None = None) -> str | None:
        """Help → Record a performance report: on, every thread's doings are counted;
        off, the report is saved (asked where, or ``path``). Returns where it went."""
        from lagoonviz.gui.profiler import Recorder

        on = self.act_profile.isChecked() if on is None else bool(on)
        self.act_profile.setChecked(on)
        if on:
            self.profiler = Recorder()
            self.profiler.start()
            self.statusBar().showMessage(
                "Recording a performance report: use LAGOON as when it is slow, then "
                "Help → Record a performance report again to stop and save it",
                15000,
            )
            return None
        recorder, self.profiler = self.profiler, None
        if recorder is None:
            return None
        recorder.stop()
        if path is None:
            stamp = time_module.strftime("%Y%m%d-%H%M%S")
            path, _ = QFileDialog.getSaveFileName(
                self,
                "Save the performance report",
                str(Path(self._last_dir()) / f"lagoon-performance-{stamp}.txt"),
                "Text (*.txt)",
            )
        if not path:
            return None
        saved = recorder.save(path)
        self.statusBar().showMessage(f"Performance report saved to {saved}", 10000)
        return str(saved)

    def about(self) -> None:
        QMessageBox.about(
            self,
            "About LAGOON",
            f"<b>LAGOON {lagoonviz.__version__}</b><br>"
            "Lightweight Animation and Graphics for Open Ocean Numerics<br><br>"
            "Part of the REEF3D family. Licensed under the GNU GPL v3 or later; "
            "the LAGOON data format library is MIT-licensed.",
        )
