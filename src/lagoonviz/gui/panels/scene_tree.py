# SPDX-License-Identifier: GPL-3.0-or-later
"""Scene tree (like Blender's outliner): items, visibility, selection, rename, remove.

Items that show data are listed under their case; hiding a case hides them all.
"""

from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QAction, QKeySequence
from PySide6.QtWidgets import (
    QAbstractItemView,
    QTreeWidget,
    QTreeWidgetItem,
    QTreeWidgetItemIterator,
    QVBoxLayout,
    QWidget,
)

from lagoonviz.core.kinds import KINDS
from lagoonviz.gui.bridge import SessionBridge

ID_ROLE = Qt.UserRole


class SceneTreePanel(QWidget):
    #: selected item id, or "" for none
    selectionChanged = Signal(str)

    def __init__(self, bridge: SessionBridge, parent: QWidget | None = None):
        super().__init__(parent)
        self.bridge = bridge
        self.session = bridge.session
        self._updating = False

        self.tree = QTreeWidget(self)
        self.tree.setObjectName("sceneTree")
        self.tree.setHeaderLabels(["Name", "Kind"])
        self.tree.setRootIsDecorated(True)
        self.tree.setSelectionMode(QAbstractItemView.SingleSelection)
        self.tree.setEditTriggers(
            QAbstractItemView.DoubleClicked | QAbstractItemView.EditKeyPressed
        )
        self.tree.setContextMenuPolicy(Qt.ActionsContextMenu)
        self.tree.itemChanged.connect(self._on_item_changed)
        self.tree.itemSelectionChanged.connect(self._on_selection)

        self.remove_action = QAction("Remove", self.tree)
        self.remove_action.setShortcut(QKeySequence.Delete)
        self.remove_action.setShortcutContext(Qt.WidgetShortcut)
        self.remove_action.triggered.connect(self.remove_selected)
        self.tree.addAction(self.remove_action)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.tree)

        bridge.sceneChanged.connect(self._on_scene)
        self.rebuild()

    # ------------------------------------------------------------- state
    def selected_id(self) -> str:
        items = self.tree.selectedItems()
        return items[0].data(0, ID_ROLE) if items else ""

    def entries(self) -> list[QTreeWidgetItem]:
        """All entries, parents before their children."""
        found = []
        iterator = QTreeWidgetItemIterator(self.tree)
        while iterator.value():
            found.append(iterator.value())
            iterator += 1
        return found

    def select(self, item_id: str) -> None:
        entry = self._entry(item_id)
        if entry is not None:
            self.tree.setCurrentItem(entry)
        else:
            self.tree.clearSelection()

    def rebuild(self) -> None:
        selected = self.selected_id()
        self._updating = True
        self.tree.blockSignals(True)
        self.tree.clear()
        cases: dict[str, QTreeWidgetItem] = {}
        for item in self.session.scene:
            entry = QTreeWidgetItem([item.name, KINDS[item.kind].label])
            entry.setData(0, ID_ROLE, item.id)
            entry.setFlags(entry.flags() | Qt.ItemIsEditable | Qt.ItemIsUserCheckable)
            entry.setCheckState(0, Qt.Checked if item.visible else Qt.Unchecked)
            entry.setToolTip(0, item.id)
            parent = None
            if item.kind != "case":
                case_item = self.session.data.case_item(item.id)
                parent = cases.get(case_item.id) if case_item is not None else None
            if parent is not None:
                parent.addChild(entry)
            else:
                self.tree.addTopLevelItem(entry)
            if item.kind == "case":
                cases[item.id] = entry
        self.tree.expandAll()
        self.tree.blockSignals(False)
        self._updating = False
        self.tree.resizeColumnToContents(0)
        if selected in self.session.scene:
            self.select(selected)
        else:
            self.selectionChanged.emit(self.selected_id())

    # ------------------------------------------------------------- events
    def _entry(self, item_id: str) -> QTreeWidgetItem | None:
        for entry in self.entries():
            if entry.data(0, ID_ROLE) == item_id:
                return entry
        return None

    def _on_scene(self, event: str, item_id: str) -> None:
        if event == "camera_keys":
            return  # nothing in the tree changes
        if event == "changed":
            # update in place: rebuilding here would delete the entry that is
            # currently emitting itemChanged (e.g. its visibility checkbox)
            entry = self._entry(item_id)
            if entry is not None:
                item = self.session.scene.get(item_id)
                self._updating = True
                self.tree.blockSignals(True)
                entry.setText(0, item.name)
                entry.setCheckState(0, Qt.Checked if item.visible else Qt.Unchecked)
                self.tree.blockSignals(False)
                self._updating = False
                return
        self.rebuild()
        if event == "added":
            self.select(item_id)

    def _on_selection(self) -> None:
        if not self._updating:
            self.selectionChanged.emit(self.selected_id())

    def _on_item_changed(self, entry: QTreeWidgetItem, column: int) -> None:
        if self._updating or column != 0:
            return
        item_id = entry.data(0, ID_ROLE)
        item = self.session.scene.get(item_id)
        visible = entry.checkState(0) == Qt.Checked
        if visible != item.visible:
            self.session.set_visible(item_id, visible)
        name = entry.text(0).strip()
        if name and name != item.name:
            self.session.rename(item_id, name)

    def remove_selected(self) -> None:
        item_id = self.selected_id()
        if item_id:
            self.session.remove_item(item_id)
