# SPDX-License-Identifier: GPL-3.0-or-later
"""Properties panel: settings of the selected item, generated from the kind's property specs."""

from __future__ import annotations

from typing import Any

from PySide6.QtCore import QLocale
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QLabel,
    QLineEdit,
    QPushButton,
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
        self.info = QLabel()

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
            if prop.tip:
                editor.setToolTip(prop.tip)
            self.editors[key] = editor
            form.addRow(prop.label, editor)

        if "auto_range" in spec.properties:
            fit = QPushButton("Fit range to this frame")
            fit.setToolTip("Fix the colour range to the values of the current frame")
            fit.clicked.connect(self._fit_range)
            form.addRow("", fit)
            self.fit_button = fit
        self.info = QLabel(self._data_info(item.id))
        self.info.setWordWrap(True)
        self.info.setEnabled(False)
        form.addRow(self.info)
        self._scroll.setWidget(body)

    def _choices(self, prop: PropertySpec, value: Any) -> list[tuple[str, str]]:
        """(label, value) pairs for a choice property; the current value is always listed."""
        if prop.provider is None:
            choices = [(c, c) for c in prop.choices or ()]
        else:
            choices = self.session.data.labelled_choices(self.item_id, prop.provider)
        if str(value) not in [v for _, v in choices]:
            choices.append((str(value), str(value)))
        return choices

    @staticmethod
    def _fill(box: QComboBox, choices: list[tuple[str, str]], value: Any) -> None:
        current = [(box.itemText(i), box.itemData(i)) for i in range(box.count())]
        if current != choices:
            box.clear()
            for label, data in choices:
                box.addItem(label, data)
        box.setCurrentIndex(max(box.findData(str(value)), 0))

    def _editor(self, key: str, prop: PropertySpec, value: Any) -> QWidget:
        default = prop.default
        if prop.choices is not None or prop.provider is not None:
            box = QComboBox()
            # long file and dataset names must not widen the panel
            box.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLengthWithIcon)
            box.setMinimumContentsLength(10)
            self._fill(box, self._choices(prop, value), value)
            box.currentIndexChanged.connect(lambda _i, b=box: self._set(key, b.currentData()))
            return box
        if isinstance(default, bool):
            check = QCheckBox()
            check.setChecked(bool(value))
            check.toggled.connect(lambda on: self._set(key, on))
            return check
        if isinstance(default, int | float):
            spin = QDoubleSpinBox()
            # a decimal point whatever the system language, as in scripts and papers
            number_locale = QLocale.c()
            number_locale.setNumberOptions(QLocale.OmitGroupSeparator)
            spin.setLocale(number_locale)
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

    def _fit_range(self) -> None:
        if not self.item_id:
            return
        try:
            self.session.fit_range(self.item_id)
        except (ValueError, KeyError) as error:
            self.info.setText(str(error))

    def _data_info(self, item_id: str) -> str:
        """A line about the data behind an item: frames and points, or why there is none."""
        data = self.session.data
        item = self.session.scene.get(item_id)
        if item.kind == "case":
            case = data.case(item_id)
            if case is None:
                return data.error(item_id) or ""
            parts = [f"{d.label}: {len(d)} frames" for d in case.datasets.values()]
            if case.series:
                parts.append(f"{len(case.series)} gauge, 6DOF or force files")
            return "\n".join(parts)
        if item.kind == "series":
            found = data.series_file(item_id)
            table = data.series(item_id)
            if found is None or table is None:
                return "No time series: open a case with gauge, 6DOF or force output"
            span = f"{table.time[0]:g}–{table.time[-1]:g} s" if len(table) else "empty"
            return f"{found.label}\n{len(table)} rows, {span}\nColumns: {', '.join(table.names)}"
        dataset = data.dataset(item_id)
        if dataset is None:
            return (
                "No data: open a case (File → Open case…)"
                if item.kind
                in (
                    "free_surface",
                    "bathymetry",
                    "volume",
                )
                else ""
            )
        return f"{dataset.label} · {len(dataset)} frames · {dataset.folder.name}"

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
        spec = kind_spec(item.kind)
        for key, editor in self.editors.items():
            value = values[key]
            editor.blockSignals(True)
            if isinstance(editor, QComboBox):
                prop = spec.properties.get(key)
                if prop is not None:
                    self._fill(editor, self._choices(prop, value), value)
            elif isinstance(editor, QCheckBox):
                editor.setChecked(bool(value))
            elif isinstance(editor, QDoubleSpinBox):
                editor.setValue(float(value))
            elif isinstance(editor, QLineEdit) and not editor.hasFocus():
                editor.setText(str(value))
            editor.blockSignals(False)
        self.info.setText(self._data_info(self.item_id))
