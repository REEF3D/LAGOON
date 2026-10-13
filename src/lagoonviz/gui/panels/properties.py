# SPDX-License-Identifier: GPL-3.0-or-later
"""Properties panel: settings of the selected item, generated from the kind's property specs."""

from __future__ import annotations

import math
from typing import Any

from PySide6.QtCore import QLocale, Qt, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QScrollArea,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from lagoonviz.core.kinds import PropertySpec, kind_spec
from lagoonviz.gui.bridge import SessionBridge
from lagoonviz.gui.panels.timeline import ClickSlider


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
        self.fit_button = None
        self.sync_button: QToolButton | None = None
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
            if key == "sync_exaggeration":
                continue  # the Sync button beside the exaggeration slider
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
        self._show_range()
        self._show_synced()

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
            spin = _number_box(prop)
            spin.setValue(float(value))
            if prop.slider == "log" and prop.minimum and prop.maximum:
                both = _LogSlider(spin, prop.minimum, prop.maximum)
                if key == "exaggeration":
                    both.valueChanged.connect(self._set_exaggeration)
                    self.sync_button = self._sync_button()
                    both.add_below(self.sync_button)  # the slider keeps the whole width
                else:
                    both.valueChanged.connect(lambda v: self._set(key, v, merge=True))
                both.released.connect(self.session.stack.end_merge)
                spin.editingFinished.connect(self.session.stack.end_merge)
                return both
            if key in ("vmin", "vmax"):
                # typing a bound fixes the range (automatic range off), the other
                # bound stays where the colours are now
                spin.valueChanged.connect(lambda v: self._set_bound(key, v))
                spin.editingFinished.connect(self.session.stack.end_merge)
                return spin
            # dragging or stepping folds into one undo step until editing finishes
            spin.valueChanged.connect(lambda v: self._set(key, v, merge=True))
            spin.editingFinished.connect(self.session.stack.end_merge)
            return spin
        line = QLineEdit(str(value))
        line.editingFinished.connect(lambda: self._set(key, line.text()))
        return line

    def _sync_button(self) -> QToolButton:
        button = QToolButton()
        button.setText("Sync")
        button.setCheckable(True)
        button.setToolTip(
            "On: one vertical scale for the whole case (every item follows "
            "this slider). Off: each item has its own."
        )
        button.toggled.connect(self._sync_exaggeration)
        return button

    # ------------------------------------------------------------- actions
    def _set_exaggeration(self, value: float) -> None:
        """The slider: the case's exaggeration while synced, else the item's own."""
        if not self.item_id or self.session.data.exaggeration(self.item_id) == value:
            return
        self.session.set_exaggeration(self.item_id, value, merge=True)

    def _sync_exaggeration(self, on: bool) -> None:
        if self.item_id and self._case_synced() != on:
            self.session.sync_exaggeration(self.item_id, on)

    def _set(self, key: str, value: Any, merge: bool = False) -> None:
        if not self.item_id:
            return
        if self.session.scene.get(self.item_id).properties[key] == value:
            return
        self.session.set_property(self.item_id, key, value, merge=merge)

    def _set_bound(self, key: str, value: float) -> None:
        """Range min or max typed: with the automatic range on, it is switched off and
        the other bound keeps the value shown (the automatic one)."""
        if not self.item_id:
            return
        props = self.session.scene.get(self.item_id).properties
        if not props.get("auto_range", True):
            self._set(key, value, merge=True)
            return
        other = "vmax" if key == "vmin" else "vmin"
        shown = float(self.editors[other].value())
        self.session.set_properties(self.item_id, auto_range=False, **{key: value, other: shown})

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
        if item.kind == "setup":
            setup = data.case_setup(item_id)
            if setup is None:
                return "No input files (control.txt, ctrl.txt) in the case folder"
            return setup.describe().split("\n", 1)[-1].replace("  ", "")
        if item.kind == "grid":
            grid = data.grid(item_id)
            if grid is None:
                setup = data.case_setup(item_id)
                if setup is not None and setup.stretched:
                    return "A stretched grid (B 101 - B 103): run DIVEMesh to see it"
                return "No grid: control.txt has no domain (B 10) and cells (B 1, B 2)"
            nx, ny, nz = grid.cells
            sizes = ", ".join(
                f"d{a} {lo:.4g}" + (f"–{hi:.4g}" if hi > lo * 1.0001 else "")
                for a, (lo, hi) in zip("xyz", grid.spacing(), strict=True)
            )
            source = (
                "DIVEMesh's grid" if grid.source == "DIVEMesh" else "from control.txt (not run)"
            )
            return f"{source}\n{nx} × {ny} × {nz} = {nx * ny * nz:,} cells\n{sizes} m"
        if item.kind == "geodat":
            dataset = data.dataset(item_id)
            if dataset is None:
                return "No geo.dat in the case folder: Set-up → Map → Make geo.dat"
            try:
                facts = dataset.facts()
            except (OSError, ValueError) as error:
                return f"geo.dat: {error}"
            lines = [f"{k}: {v}" for k, v in facts.items()]
            setup = data.case_setup(item_id)
            if setup is not None and setup.control.int("G 10", 0) <= 0:
                lines.append("Not read by DIVEMesh without G 10 1 (control.txt)")
            return "\n".join(lines)
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
        elif event == "inputs" and self.item_id in self.session.scene:
            if self.session.scene.get(self.item_id).kind in ("setup", "grid", "geodat"):
                self.info.setText(self._data_info(self.item_id))  # the set-up's facts
        elif event == "changed" and item_id == self.item_id:
            self._refresh()
        elif (
            event == "changed"
            and self.item_id in self.session.scene
            and item_id in self.session.scene
            and self.session.scene.get(item_id).kind == "case"
        ):
            self._show_synced()  # the case's exaggeration may apply to this item

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
            elif isinstance(editor, QDoubleSpinBox) and editor.hasFocus():
                pass  # being typed in: an update elsewhere must not overwrite it
            elif isinstance(editor, QDoubleSpinBox | _LogSlider):
                editor.setValue(float(value))
            elif isinstance(editor, QLineEdit) and not editor.hasFocus():
                editor.setText(str(value))
            editor.blockSignals(False)
        self.info.setText(self._data_info(self.item_id))
        self._show_range()
        self._show_synced()

    def _case_synced(self) -> bool | None:
        """Whether the item's case has one exaggeration for all items (None: no case)."""
        case = self.session.data.case_item(self.item_id)
        if case is None:
            return None
        return bool(case.properties.get("sync_exaggeration", True))

    def _show_synced(self) -> None:
        """The exaggeration slider shows what is drawn (the case's value while synced),
        and the Sync button the case's setting."""
        editor = self.editors.get("exaggeration")
        if not isinstance(editor, _LogSlider) or self.item_id not in self.session.scene:
            return
        item = self.session.scene.get(self.item_id)
        synced = self._case_synced()
        editor.setValue(self.session.data.exaggeration(self.item_id))
        if self.sync_button is not None:
            self.sync_button.blockSignals(True)
            self.sync_button.setChecked(bool(synced))
            self.sync_button.blockSignals(False)
            self.sync_button.setEnabled(synced is not None)
        if item.kind == "case":
            # the case's own value is only used while synced
            editor.slider.setEnabled(bool(synced))
            editor.spin.setEnabled(bool(synced))
            text = (
                "Scales heights about the still water level, for every item of the case"
                if synced
                else "Sync is off: each item has its own vertical scale"
            )
        elif synced:
            text = "Sync is on: this sets the vertical scale of the whole case"
        else:
            text = "Scales heights about the still water level, for this item only"
        editor.slider.setToolTip(text)
        editor.spin.setToolTip(text)

    def _show_range(self) -> None:
        """With an automatic range, the min and max boxes show it; typing in them fixes
        the range. Coloured by nothing (a plain colour), the colour map and range are
        greyed out."""
        low_box, high_box = self.editors.get("vmin"), self.editors.get("vmax")
        if low_box is None or high_box is None or self.item_id not in self.session.scene:
            return
        props = self.session.scene.get(self.item_id).properties
        coloured = str(props.get("variable", "none")) != "none"
        hint = 'Coloured by nothing: choose "Colour by" to use a colour range'
        auto_box = self.editors.get("auto_range")
        if auto_box is not None:
            auto_box.setEnabled(coloured)
            auto_box.setToolTip("" if coloured else hint)
        colormap_box = self.editors.get("colormap")
        if colormap_box is not None:
            # coloured by nothing, a surface takes one colour of the map (white: white);
            # bodies and structures take their material's
            uses_material = "material" in props
            colormap_box.setEnabled(coloured or not uses_material)
            colormap_box.setToolTip(
                ""
                if coloured
                else 'Coloured by nothing: the material sets the colour; choose "Colour by"'
                " to use a colour map"
                if uses_material
                else "Coloured by nothing: the surface takes one colour of this map"
            )
        if getattr(self, "fit_button", None) is not None:
            self.fit_button.setEnabled(coloured)
        percent_box = self.editors.get("range_percent")
        if percent_box is not None:
            percent_box.setEnabled(coloured)
            if not coloured:
                percent_box.setToolTip(hint)
            else:
                percent_box.setToolTip(kind_spec_tip(self.session, self.item_id, "range_percent"))
        auto = bool(props.get("auto_range", True))
        for box in (low_box, high_box):
            # always editable while coloured: typing a bound fixes the range
            box.setEnabled(coloured)
            box.setToolTip(
                hint
                if not coloured
                else "Automatic now: type a value to fix the range"
                if auto
                else ""
            )
        if not auto or not coloured:
            return
        try:
            shown = self.session.data.automatic_range(self.item_id)
        except Exception:  # noqa: BLE001 (no data yet: the boxes stay as they are)
            shown = None
        if shown is None:
            return
        for box, value in zip((low_box, high_box), shown, strict=True):
            box.blockSignals(True)
            box.setValue(float(value))
            box.blockSignals(False)


class _SpinBox(QDoubleSpinBox):
    """A spin box that leaves the mouse wheel to the panel unless it has focus, so
    scrolling the property panel never changes a value by accident."""

    def __init__(self) -> None:
        super().__init__()
        self.setFocusPolicy(Qt.StrongFocus)

    def wheelEvent(self, event) -> None:  # noqa: N802 (Qt naming)
        if self.hasFocus():
            super().wheelEvent(event)
        else:
            event.ignore()


def _number_box(prop: PropertySpec) -> _SpinBox:
    spin = _SpinBox()
    # a decimal point whatever the system language, as in scripts and papers
    number_locale = QLocale.c()
    number_locale.setNumberOptions(QLocale.OmitGroupSeparator)
    spin.setLocale(number_locale)
    spin.setDecimals(0 if isinstance(prop.default, int) else 3)
    spin.setRange(
        prop.minimum if prop.minimum is not None else -1e9,
        prop.maximum if prop.maximum is not None else 1e9,
    )
    spin.setSingleStep(0.1 if (prop.maximum or 10) <= 1 else 1.0)
    spin.setKeyboardTracking(False)
    return spin


class _LogSlider(QWidget):
    """A slider on a log scale with the spin box beside it, for factors like the
    vertical exaggeration (0.1 to 100: the middle of the slider is 3.16)."""

    STEPS = 1000
    valueChanged = Signal(float)
    released = Signal()

    def __init__(self, spin: _SpinBox, minimum: float, maximum: float) -> None:
        super().__init__()
        self.spin = spin
        self._low, self._high = math.log10(minimum), math.log10(maximum)
        # goes where it is clicked and follows the mouse, like the timeline
        self.slider = ClickSlider(Qt.Horizontal, ranged=False)  # no play range stripes
        self.slider.setRange(0, self.STEPS)
        self.slider.setFocusPolicy(Qt.StrongFocus)
        self._rows = QVBoxLayout(self)
        self._rows.setContentsMargins(0, 0, 0, 0)
        self._rows.setSpacing(2)
        layout = QHBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.slider, 1)
        layout.addWidget(spin)
        self._rows.addLayout(layout)
        spin.setMaximumWidth(80)
        self._place(spin.value())
        self.slider.valueChanged.connect(self._slid)
        self.slider.sliderReleased.connect(self.released)
        spin.valueChanged.connect(self._typed)

    def add_below(self, widget: QWidget) -> None:
        """A widget (the Sync button) on a row under the slider."""
        row = QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        row.addWidget(widget)
        row.addStretch(1)
        self._rows.addLayout(row)

    def to_value(self, position: int) -> float:
        value = 10 ** (self._low + position / self.STEPS * (self._high - self._low))
        return float(f"{value:.3g}")  # 1.0, 2.5, 10: round numbers while dragging

    def to_position(self, value: float) -> int:
        if value <= 0:
            return 0
        share = (math.log10(value) - self._low) / (self._high - self._low)
        return round(min(max(share, 0.0), 1.0) * self.STEPS)

    def value(self) -> float:
        return self.spin.value()

    def setValue(self, value: float) -> None:  # noqa: N802 (Qt naming)
        self.spin.blockSignals(True)
        self.spin.setValue(value)
        self.spin.blockSignals(False)
        self._place(value)

    def _place(self, value: float) -> None:
        self.slider.blockSignals(True)
        self.slider.setValue(self.to_position(value))
        self.slider.blockSignals(False)

    def _slid(self, position: int) -> None:
        value = self.to_value(position)
        self.spin.blockSignals(True)
        self.spin.setValue(value)
        self.spin.blockSignals(False)
        self.valueChanged.emit(self.spin.value())

    def _typed(self, value: float) -> None:
        self._place(value)
        self.valueChanged.emit(value)


def kind_spec_tip(session, item_id: str, key: str) -> str:
    """The tooltip of an item's property, as its kind gives it."""
    from lagoonviz.core.kinds import kind_spec

    prop = kind_spec(session.scene.get(item_id).kind).properties.get(key)
    return prop.tip if prop is not None else ""
