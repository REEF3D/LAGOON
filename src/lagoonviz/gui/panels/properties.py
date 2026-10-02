# SPDX-License-Identifier: GPL-3.0-or-later
"""Properties panel: settings of the selected item, generated from the kind's property specs."""

from __future__ import annotations

from typing import Any

from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QLabel,
    QLineEdit,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from lagoonviz.core.kinds import PropertySpec, kind_spec
from lagoonviz.gui.bridge import SessionBridge


class PropertiesPanel(QWidget):
    def __init__(self, bridge: SessionBridge, parent: QWidget | None = None):
        super().__init__(parent)
        self.setObjectName("properties")
        self.bridge = bridge
        self.session = bridge.session
        self.item_id = ""
        self.editors: dict[str, QWidget] = {}

        self._scroll = QScrollArea(self)
        self._scroll.setWidgetResizable(True)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._scroll)
        bridge.sceneChanged.connect(self._on_scene)
        self.show_item("")

    # ------------------------------------------------------------- building
    def show_item(self, item_id: str) -> None:
        self.item_id = item_id if item_id in self.session.scene else ""
        self.editors = {}
        body = QWidget()
        form = QFormLayout(body)
        if not self.item_id:
            hint = QLabel("Select an item in the scene tree.")
            hint.setEnabled(False)
            form.addRow(hint)
            self._scroll.setWidget(body)
            return

        item = self.session.scene.get(self.item_id)
        spec = kind_spec(item.kind)

        name = QLineEdit(item.name)
        name.editingFinished.connect(lambda: self._rename(name.text()))
        form.addRow("Name", name)
        self.editors["name"] = name
        form.addRow("Kind", QLabel(spec.label))
        visible = QCheckBox()
        visible.setChecked(item.visible)
        visible.toggled.connect(lambda on: self.session.set_visible(self.item_id, on))
        form.addRow("Visible", visible)
        self.editors["visible"] = visible

        for key, prop in spec.properties.items():
            editor = self._editor(key, prop, item.properties[key])
            self.editors[key] = editor
            form.addRow(prop.label, editor)
        self._scroll.setWidget(body)

    def _editor(self, key: str, prop: PropertySpec, value: Any) -> QWidget:
        default = prop.default
        if prop.choices is not None:
            box = QComboBox()
            box.addItems(list(prop.choices))
            box.setCurrentText(str(value))
            box.currentTextChanged.connect(lambda text: self._set(key, text))
            return box
        if isinstance(default, bool):
            check = QCheckBox()
            check.setChecked(bool(value))
            check.toggled.connect(lambda on: self._set(key, on))
            return check
        if isinstance(default, int | float):
            spin = QDoubleSpinBox()
            spin.setDecimals(3)
            spin.setRange(
                prop.minimum if prop.minimum is not None else -1e9,
                prop.maximum if prop.maximum is not None else 1e9,
            )
            spin.setSingleStep(0.1 if (prop.maximum or 10) <= 1 else 1.0)
            spin.setValue(float(value))
            spin.setKeyboardTracking(False)
            # dragging or stepping folds into one undo step until editing finishes
            spin.valueChanged.connect(lambda v: self._set(key, v, merge=True))
            spin.editingFinished.connect(self.session.stack.end_merge)
            return spin
        line = QLineEdit(str(value))
        line.editingFinished.connect(lambda: self._set(key, line.text()))
        return line

    # ------------------------------------------------------------- actions
    def _set(self, key: str, value: Any, merge: bool = False) -> None:
        if not self.item_id:
            return
        if self.session.scene.get(self.item_id).properties[key] == value:
            return
        self.session.set_property(self.item_id, key, value, merge=merge)

    def _rename(self, text: str) -> None:
        text = text.strip()
        if self.item_id and text and text != self.session.scene.get(self.item_id).name:
            self.session.rename(self.item_id, text)

    # ------------------------------------------------------------- updates
    def _on_scene(self, event: str, item_id: str) -> None:
        if event in ("removed", "reset") and self.item_id not in self.session.scene:
            self.show_item("")
        elif event == "changed" and item_id == self.item_id:
            self._refresh()

    def _refresh(self) -> None:
        """Update editor values in place (no rebuild, so focus and cursor stay)."""
        item = self.session.scene.get(self.item_id)
        values: dict[str, Any] = {"name": item.name, "visible": item.visible, **item.properties}
        for key, editor in self.editors.items():
            value = values[key]
            editor.blockSignals(True)
            if isinstance(editor, QComboBox):
                editor.setCurrentText(str(value))
            elif isinstance(editor, QCheckBox):
                editor.setChecked(bool(value))
            elif isinstance(editor, QDoubleSpinBox):
                editor.setValue(float(value))
            elif isinstance(editor, QLineEdit) and not editor.hasFocus():
                editor.setText(str(value))
            editor.blockSignals(False)
