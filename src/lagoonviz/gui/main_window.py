# SPDX-License-Identifier: GPL-3.0-or-later
"""LAGOON main window: Blender-style layout with dockable panels.

Scene tree on the left, viewport in the centre, properties on the right, plots,
timeline and Python console along the bottom.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PySide6.QtCore import QByteArray, QLocale, QPoint, QRect, QSettings, Qt
from PySide6.QtGui import QAction, QCloseEvent, QIcon, QImage, QKeySequence, QPainter
from PySide6.QtWidgets import (
    QDockWidget,
    QFileDialog,
    QMainWindow,
    QMessageBox,
    QWidget,
)

import lagoonviz
from lagoonviz.core import SESSION_EXTENSION, Session
from lagoonviz.core.data.manager import BackgroundLoader
from lagoonviz.core.kinds import KINDS
from lagoonviz.gui.bridge import SessionBridge
from lagoonviz.gui.panels import (
    ConsolePanel,
    PlotsPanel,
    PropertiesPanel,
    SceneTreePanel,
    TimelinePanel,
    create_viewport,
)
from lagoonviz.gui.resources import resource_path

SESSION_FILTER = f"LAGOON session (*{SESSION_EXTENSION});;All files (*)"
OUTPUT_FILTER = "REEF3D output (*.pvtu *.pvtp *.vtu *.vtp);;All files (*)"
#: bump when the dock layout changes, so stale saved layouts are ignored
LAYOUT_VERSION = 1


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
        self.loader = BackgroundLoader(self.session.data)

        # panels
        self.viewport = create_viewport(self.bridge, self.loader, self)
        self.viewport.statusMessage.connect(lambda text: self.statusBar().showMessage(text, 8000))
        self.setCentralWidget(self.viewport)
        self.scene_tree = SceneTreePanel(self.bridge)
        self.properties = PropertiesPanel(self.bridge)
        self.timeline = TimelinePanel(
            self.bridge, loader=self.loader, busy=lambda: self.viewport.busy
        )
        self.plots = PlotsPanel()
        self.console = ConsolePanel(self.bridge)

        self.docks: dict[str, QDockWidget] = {}
        self._dock("scene", "Scene", self.scene_tree, Qt.LeftDockWidgetArea)
        self._dock("properties", "Properties", self.properties, Qt.RightDockWidgetArea)
        self._dock("plots", "Plots", self.plots, Qt.BottomDockWidgetArea)
        self._dock("timeline", "Timeline", self.timeline, Qt.BottomDockWidgetArea)
        self._dock("console", "Python console", self.console, Qt.BottomDockWidgetArea)
        self.splitDockWidget(self.docks["plots"], self.docks["timeline"], Qt.Vertical)
        self.splitDockWidget(self.docks["timeline"], self.docks["console"], Qt.Horizontal)
        self._default_state = self.saveState(LAYOUT_VERSION)
        self._sized = False

        self.scene_tree.selectionChanged.connect(self.properties.show_item)
        self.bridge.historyChanged.connect(self._on_history)

        self._build_actions()
        self._build_menus()
        self._build_toolbar()
        self._restore_layout()
        self._on_history()
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
        self.act_open_file = self._action(
            "Open output &file...",
            self.open_output_dialog,
            tip="Open a REEF3D .pvtu/.pvtp/.vtu/.vtp file and the series it belongs to",
        )
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
                lambda _=False, k=kind: self.session.add_item(k),
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
        self.act_reset_layout = self._action("&Reset layout", self.reset_layout)
        self.act_about = self._action("&About LAGOON", self.about)

    def _build_menus(self) -> None:
        bar = self.menuBar()
        file_menu = bar.addMenu("&File")
        file_menu.addAction(self.act_open_case)
        file_menu.addAction(self.act_open_file)
        file_menu.addSeparator()
        for action in (self.act_new, self.act_open, self.act_save, self.act_save_as):
            file_menu.addAction(action)
        file_menu.addSeparator()
        file_menu.addAction(self.act_export_script)
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
        for dock in self.docks.values():
            view_menu.addAction(dock.toggleViewAction())
        view_menu.addSeparator()
        view_menu.addAction(self.act_screenshot)
        view_menu.addAction(self.act_reset_layout)

        help_menu = bar.addMenu("&Help")
        help_menu.addAction(self.act_about)

    def _build_toolbar(self) -> None:
        bar = self.addToolBar("Main")
        bar.setObjectName("toolbar_main")
        bar.addAction(self.act_open_case)
        bar.addSeparator()
        bar.addAction(self.act_open)
        bar.addAction(self.act_save)
        bar.addSeparator()
        bar.addAction(self.act_undo)
        bar.addAction(self.act_redo)
        bar.addSeparator()
        for kind in ("free_surface", "bathymetry", "volume", "slice", "gauge"):
            bar.addAction(self.add_actions[kind])

    # ------------------------------------------------------------- layout
    def _restore_layout(self) -> None:
        geometry = self.settings.value("window/geometry")
        state = self.settings.value("window/state")
        if isinstance(geometry, QByteArray):
            self.restoreGeometry(geometry)
        self._restored = isinstance(state, QByteArray) and self.restoreState(state, LAYOUT_VERSION)

    def _apply_default_sizes(self) -> None:
        self.resizeDocks([self.docks["scene"], self.docks["properties"]], [260, 300], Qt.Horizontal)
        self.resizeDocks([self.docks["plots"], self.docks["timeline"]], [160, 90], Qt.Vertical)
        self.resizeDocks([self.docks["timeline"], self.docks["console"]], [700, 500], Qt.Horizontal)

    def showEvent(self, event) -> None:  # noqa: N802 (Qt naming)
        super().showEvent(event)
        # dock sizes only apply once the window has a size; saved layouts keep their own
        if not self._sized and not self._restored:
            self._apply_default_sizes()
        self._sized = True

    def reset_layout(self) -> None:
        self.restoreState(self._default_state, LAYOUT_VERSION)
        for dock in self.docks.values():
            dock.show()
        self._apply_default_sizes()

    def closeEvent(self, event: QCloseEvent) -> None:  # noqa: N802 (Qt naming)
        if not self.confirm_discard():
            event.ignore()
            return
        self.settings.setValue("window/geometry", self.saveGeometry())
        self.settings.setValue("window/state", self.saveState(LAYOUT_VERSION))
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
        if not self.ask_before_discard or not self.dirty or not self.session.stack.can_undo:
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

    def open_output_dialog(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Open REEF3D output file", self._last_dir(), OUTPUT_FILTER
        )
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
        case = self.session.data.case(case_id)
        summary = ", ".join(f"{d.label} ({len(d)} frames)" for d in case.datasets.values())
        self.statusBar().showMessage(f"Opened {Path(path).name}: {summary}", 10000)
        self.viewport.reframe()
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
        snapshot = self.viewport.canvas_snapshot()
        if snapshot is not None:
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
        self.statusBar().showMessage(
            f"Screenshot saved to {path}" if ok else f"Could not save {path}", 5000
        )
        return ok

    def _last_dir(self) -> str:
        return str(self.settings.value("files/last_dir", str(Path.home())))

    def _remember_dir(self, path: str | Path) -> None:
        self.settings.setValue("files/last_dir", str(Path(path).resolve().parent))

    # ------------------------------------------------------------- help
    def about(self) -> None:
        QMessageBox.about(
            self,
            "About LAGOON",
            f"<b>LAGOON {lagoonviz.__version__}</b><br>"
            "Lightweight Animation and Graphics for Open Ocean Numerics<br><br>"
            "Part of the REEF3D family. Licensed under the GNU GPL v3 or later; "
            "the LAGOON data format library is MIT-licensed.",
        )
