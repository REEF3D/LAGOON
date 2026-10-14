# SPDX-License-Identifier: GPL-3.0-or-later
# Architect: Hans Bihs
"""The figure editor (File → Figure editor): build a paper figure by dragging time
series and free surface lines onto panels, change its look live, keep it as a template.

Left, what can be drawn: the time-series files and free surface lines of the open cases,
and measured files added with "Add measured file". In the middle, the figure: its
panels (time series, spectra, free surface lines, 3D views) and their lines; drag a
column onto a panel (or double-click it) to add it, onto empty space for a new
panel. Under it the settings of what is selected: of the figure (style, size, a
grid of columns, panel letters), of a panel (title, columns it spans, notes, its
moment; a 3D view takes the view as it is when asked), of a line. Right, the
figure as it will be written, drawn again after each change. Templates
(``*.lagoonfig``) keep the figure for other runs of the same set-up.
"""

from __future__ import annotations

import copy
import logging
import traceback
from pathlib import Path

from PySide6.QtCore import QMimeData, Qt, QTimer
from PySide6.QtGui import QColor, QImage, QPixmap
from PySide6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QColorDialog,
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMenu,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QSpinBox,
    QSplitter,
    QStackedWidget,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from lagoonviz.core.figure_template import (
    LINE_STYLES,
    MARKERS,
    SUFFIX,
    FigureTemplate,
    Line,
    Panel,
    fits,
    notes_text,
    parse_notes,
    profile_columns,
    series_columns,
)

log = logging.getLogger("lagoon.gui")
MIME = "application/x-lagoon-line"
ROLE = Qt.UserRole
PANEL_LABELS = {
    "time": "Time series",
    "spectrum": "Spectrum",
    "profile": "Free surface lines",
    "view": "3D view",
}

STYLE_NAMES = {"-": "solid", "--": "dashed", ":": "dotted", "-.": "dash-dot", "none": "no line"}


def _encode(line: Line) -> bytes:
    return "\x1f".join((line.source, line.column, line.file, line.data, line.linestyle)).encode()


def _decode(data: bytes) -> Line:
    source, column, file, kind, linestyle = data.decode().split("\x1f")
    return Line(source, column, file=file, data=kind, linestyle=linestyle)


class SourceTree(QTreeWidget):
    """What can be drawn: files and their columns; columns drag as lines."""

    def __init__(self) -> None:
        super().__init__()
        self.setHeaderLabel("Data")
        self.setDragEnabled(True)
        self.setDragDropMode(QAbstractItemView.DragOnly)

    def mimeData(self, items):  # noqa: N802 (Qt naming)
        lines = [item.data(0, ROLE) for item in items if isinstance(item.data(0, ROLE), Line)]
        if not lines:
            return None
        mime = QMimeData()
        mime.setData(MIME, _encode(lines[0]))
        return mime


class FigureTree(QTreeWidget):
    """The figure: panels and their lines; lines are dropped onto it."""

    def __init__(self, editor: FigureEditor) -> None:
        super().__init__()
        self.editor = editor
        self.setHeaderLabel("Figure")
        self.setAcceptDrops(True)
        self.setDragDropMode(QAbstractItemView.DropOnly)

    def dragEnterEvent(self, event) -> None:  # noqa: N802 (Qt naming)
        if event.mimeData().hasFormat(MIME):
            event.acceptProposedAction()
        else:
            event.ignore()

    def dragMoveEvent(self, event) -> None:  # noqa: N802 (Qt naming)
        self.dragEnterEvent(event)

    def dropEvent(self, event) -> None:  # noqa: N802 (Qt naming)
        if not event.mimeData().hasFormat(MIME):
            event.ignore()
            return
        line = _decode(bytes(event.mimeData().data(MIME)))
        target = self.itemAt(event.position().toPoint())
        panel = self.editor.panel_of(target)
        self.editor.add_line(line, panel)
        event.acceptProposedAction()


class FigureEditor(QDialog):
    """A non-modal window: one figure, edited live."""

    def __init__(self, window, template: FigureTemplate | None = None) -> None:
        super().__init__(window)
        from lagoonviz.render.figure import STYLES, WIDTHS

        self.main_window = window
        self.session = window.session
        self.template = template or FigureTemplate()
        self.setWindowTitle("Figure editor")
        self.resize(1100, 640)
        self._updating = False
        self._measured: list[str] = []

        # what can be drawn
        self.sources = SourceTree()
        self.sources.itemDoubleClicked.connect(self._source_double_clicked)
        add_file = QPushButton("Add measured file…")
        add_file.clicked.connect(self.add_measured_file)
        left = QWidget()
        column = QVBoxLayout(left)
        column.setContentsMargins(0, 0, 0, 0)
        column.addWidget(self.sources, 1)
        column.addWidget(add_file)

        # the figure and the settings of what is selected
        self.tree = FigureTree(self)
        self.tree.currentItemChanged.connect(lambda *_: self._show_selected())
        buttons = QHBoxLayout()
        add = QPushButton("Add panel")
        menu = QMenu(add)
        for kind, label in PANEL_LABELS.items():
            menu.addAction(label, lambda kind=kind: self.add_panel(kind))
        add.setMenu(menu)
        buttons.addWidget(add)
        for text, slot in (
            ("Up", lambda: self.move_selected(-1)),
            ("Down", lambda: self.move_selected(1)),
            ("Remove", self.remove_selected),
        ):
            button = QPushButton(text)
            button.clicked.connect(slot)
            buttons.addWidget(button)
        self.settings_stack = QStackedWidget()
        self.figure_form = self._figure_form(STYLES, WIDTHS)
        self.panel_form = self._panel_form()
        self.line_form = self._line_form()
        for form in (self.figure_form, self.panel_form, self.line_form):
            self.settings_stack.addWidget(form)
        middle = QWidget()
        column = QVBoxLayout(middle)
        column.setContentsMargins(0, 0, 0, 0)
        column.addWidget(self.tree, 1)
        column.addLayout(buttons)
        column.addWidget(self.settings_stack)

        # the figure as it will be written
        self.preview = QLabel("Drag a time series onto the figure")
        self.preview.setAlignment(Qt.AlignCenter)
        self.preview.setMinimumWidth(360)
        self.problems = QLabel()
        self.problems.setWordWrap(True)
        self.problems.setStyleSheet("color: #c4302b;")
        right = QWidget()
        column = QVBoxLayout(right)
        column.setContentsMargins(0, 0, 0, 0)
        column.addWidget(self.preview, 1)
        column.addWidget(self.problems)
        actions = QHBoxLayout()
        for text, slot in (
            ("Open template…", self.open_template),
            ("Save template…", self.save_template),
            ("Export figure…", self.export),
        ):
            button = QPushButton(text)
            button.clicked.connect(slot)
            actions.addWidget(button)
        column.addLayout(actions)

        splitter = QSplitter()
        for part in (left, middle, right):
            splitter.addWidget(part)
        splitter.setSizes([230, 380, 490])
        layout = QVBoxLayout(self)
        layout.addWidget(splitter)

        self._redraw = QTimer(self)
        self._redraw.setSingleShot(True)
        self._redraw.setInterval(250)
        self._redraw.timeout.connect(self.draw_preview)
        self.fill_sources()
        self.rebuild()

    # ------------------------------------------------------------- forms
    def _figure_form(self, styles, widths) -> QWidget:
        form_widget = QWidget()
        form = QFormLayout(form_widget)
        self.style_box = QComboBox()
        self.style_box.addItems(list(styles))
        self.width_box = QComboBox()
        for name, width in widths.items():
            self.width_box.addItem(f"{name} ({width:g} mm)", width)
        self.height_box = _number(0, 1000, " mm", "automatic")
        self.font_box = _number(0, 30, " pt", "style's")
        self.title_box = QLineEdit()
        self.columns_box = QSpinBox()
        self.columns_box.setRange(1, 4)
        self.columns_box.setToolTip("Panels fill a grid of this many columns, row by row")
        self.letters_box = QCheckBox("(a), (b), ... before the panel titles")
        form.addRow("Style", self.style_box)
        form.addRow("Width", self.width_box)
        form.addRow("Height", self.height_box)
        form.addRow("Font size", self.font_box)
        form.addRow("Title", self.title_box)
        form.addRow("Columns", self.columns_box)
        form.addRow("Letters", self.letters_box)
        for box in (self.style_box, self.width_box):
            box.currentIndexChanged.connect(self._figure_changed)
        for box in (self.height_box, self.font_box, self.columns_box):
            box.valueChanged.connect(self._figure_changed)
        self.title_box.editingFinished.connect(self._figure_changed)
        self.letters_box.toggled.connect(self._figure_changed)
        return form_widget

    def _panel_form(self) -> QWidget:
        form_widget = QWidget()
        form = self.panel_layout = QFormLayout(form_widget)
        self.kind_box = QComboBox()
        for kind, label in PANEL_LABELS.items():
            self.kind_box.addItem(label, kind)
        self.panel_title_box = QLineEdit()
        self.panel_title_box.setPlaceholderText("none (profiles and views: their time)")
        self.span_box = QSpinBox()
        self.span_box.setRange(1, 4)
        self.span_box.setToolTip("Columns of the figure's grid the panel takes")
        self.moment_box = _number(-1, 1e7, " s", "the scene's time")
        self.moment_box.setDecimals(3)
        self.moment_box.setToolTip("The moment a free surface line or 3D view shows")
        self.notes_box = QPlainTextEdit()
        self.notes_box.setPlaceholderText(
            r"one note per line, $LaTeX$ allowed; '0.6, 0.9: text' places it"
        )
        self.notes_box.setMaximumHeight(60)
        self.view_box = QComboBox()
        self.view_box.addItem("as captured (3D)", "camera")
        self.view_box.addItem("from the top (2D, axes in m)", "top")
        self.bars_box = QCheckBox("Colour bars under the image")
        self.take_view_button = QPushButton("Take the 3D view as it is now")
        self.take_view_button.setToolTip("The camera, time and look of the 3D view, for this panel")
        self.take_view_button.clicked.connect(self.take_view)
        self.ylabel_box = QLineEdit()
        self.ylabel_box.setPlaceholderText(r"automatic; $\eta$ [m] for symbols")
        self.ymin_box = QLineEdit()
        self.ymax_box = QLineEdit()
        for box in (self.ymin_box, self.ymax_box):
            box.setPlaceholderText("automatic")
        self.logy_box = QCheckBox("Logarithmic y axis")
        self.start_box = _number(0, 1e7, " s", "start")
        self.end_box = _number(0, 1e7, " s", "end")
        self.compare_box = QCheckBox("Statistics: first line against the measured one")
        form.addRow("Panel", self.kind_box)
        form.addRow("Title", self.panel_title_box)
        form.addRow("Columns", self.span_box)
        form.addRow("Moment", self.moment_box)
        form.addRow("", self.take_view_button)
        form.addRow("View", self.view_box)
        form.addRow("", self.bars_box)
        form.addRow("y label", self.ylabel_box)
        form.addRow("y min", self.ymin_box)
        form.addRow("y max", self.ymax_box)
        form.addRow("", self.logy_box)
        form.addRow("From", self.start_box)
        form.addRow("To", self.end_box)
        form.addRow("", self.compare_box)
        form.addRow("Notes", self.notes_box)
        for box in (self.kind_box, self.view_box):
            box.currentIndexChanged.connect(self._panel_changed)
        for box in (self.ylabel_box, self.ymin_box, self.ymax_box, self.panel_title_box):
            box.editingFinished.connect(self._panel_changed)
        for box in (self.logy_box, self.compare_box, self.bars_box):
            box.toggled.connect(self._panel_changed)
        for box in (self.start_box, self.end_box, self.moment_box, self.span_box):
            box.valueChanged.connect(self._panel_changed)
        self.notes_box.textChanged.connect(self._panel_changed)
        return form_widget

    def _panel_rows(self, kind: str) -> None:
        """Only the settings a kind of panel has."""
        lines = kind != "view"
        shown = {
            self.moment_box: kind in ("profile", "view"),
            self.take_view_button: kind == "view",
            self.view_box: kind == "view",
            self.bars_box: kind == "view",
            self.ylabel_box: lines,
            self.ymin_box: lines,
            self.ymax_box: lines,
            self.logy_box: lines,
            self.start_box: lines,
            self.end_box: lines,
            self.compare_box: kind == "time",
        }
        for widget, visible in shown.items():
            self.panel_layout.setRowVisible(widget, visible)
        suffix = " m" if kind == "profile" else " s"
        for box in (self.start_box, self.end_box):
            box.setSuffix(suffix)

    def _line_form(self) -> QWidget:
        form_widget = QWidget()
        form = QFormLayout(form_widget)
        self.label_box = QLineEdit()
        self.label_box.setPlaceholderText("the column's name")
        self.color_button = QPushButton("Style's colour")
        self.color_button.clicked.connect(self._choose_colour)
        self.linestyle_box = QComboBox()
        for style in LINE_STYLES:
            self.linestyle_box.addItem(STYLE_NAMES[style], style)
        self.marker_box = QComboBox()
        for marker in MARKERS:
            self.marker_box.addItem(marker or "no marker", marker)
        self.width_line_box = _number(0, 10, " pt", "style's")
        form.addRow("Label", self.label_box)
        form.addRow("Colour", self.color_button)
        form.addRow("Line", self.linestyle_box)
        form.addRow("Marker", self.marker_box)
        form.addRow("Width", self.width_line_box)
        self.label_box.editingFinished.connect(self._line_changed)
        for box in (self.linestyle_box, self.marker_box):
            box.currentIndexChanged.connect(self._line_changed)
        self.width_line_box.valueChanged.connect(self._line_changed)
        return form_widget

    # ------------------------------------------------------------- sources
    def fill_sources(self) -> None:
        self.sources.clear()
        for key, (name, columns) in series_columns(self.session).items():
            top = QTreeWidgetItem([name])
            top.setToolTip(0, key)
            top.setFlags(top.flags() & ~Qt.ItemIsDragEnabled)
            for column in columns:
                child = QTreeWidgetItem([column])
                child.setData(0, ROLE, Line(key, column))
                top.addChild(child)
            self.sources.addTopLevelItem(top)
        for key, (name, columns) in profile_columns(self.session).items():
            top = QTreeWidgetItem([name])
            top.setToolTip(0, key)
            top.setFlags(top.flags() & ~Qt.ItemIsDragEnabled)
            for column, label in columns:
                child = QTreeWidgetItem([label])
                theory = column.endswith("theory")
                child.setData(
                    0, ROLE, Line(key, column, data="profile", linestyle="--" if theory else "-")
                )
                top.addChild(child)
            self.sources.addTopLevelItem(top)
        for path in self._measured:
            self._add_measured_item(path)
        self.sources.expandAll()

    def add_measured_file(self, path: str | None = None) -> None:
        if not path:
            path, _ = QFileDialog.getOpenFileName(
                self, "Measured time series", "", "Time series (*.dat *.txt *.csv);;All (*)"
            )
        if not path:
            return
        if path not in self._measured:
            self._measured.append(path)
        self._add_measured_item(path)

    def _add_measured_item(self, path: str) -> None:
        from lagoonviz.core.analysis import read_measurements

        try:
            _time, columns = read_measurements(path)
        except (OSError, ValueError) as error:
            QMessageBox.warning(self, "Measured file", str(error))
            return
        top = QTreeWidgetItem([f"{Path(path).name} (measured)"])
        top.setToolTip(0, path)
        for column in columns:
            child = QTreeWidgetItem([column])
            child.setData(0, ROLE, Line("", column, file=path))
            top.addChild(child)
        self.sources.addTopLevelItem(top)
        top.setExpanded(True)

    def _source_double_clicked(self, item: QTreeWidgetItem) -> None:
        line = item.data(0, ROLE)
        if isinstance(line, Line):
            self.add_line(copy.copy(line), self.panel_of(self.tree.currentItem()))

    # ------------------------------------------------------------- editing
    def panel_of(self, item: QTreeWidgetItem | None) -> int | None:
        """The panel index of a tree item (a panel, or a line in it); None: none."""
        if item is None:
            return None
        if item.parent() is not None:
            item = item.parent()
        return self.tree.indexOfTopLevelItem(item)

    def add_panel(self, kind: str = "time") -> int:
        self.template.panels.append(Panel(kind=kind))
        self.rebuild(select=(len(self.template.panels) - 1, None))
        self.changed()
        return len(self.template.panels) - 1

    def add_line(self, line: Line, panel: int | None = None) -> None:
        """A line onto a panel; onto a new panel when there is none, or the panel does
        not take it (a free surface line onto a time panel, anything onto a 3D view)."""
        if (
            panel is None
            or not 0 <= panel < len(self.template.panels)
            or not fits(line, self.template.panels[panel].kind)
        ):
            panel = self.add_panel("profile" if line.data == "profile" else "time")
        self.template.panels[panel].lines.append(line)
        self.rebuild(select=(panel, len(self.template.panels[panel].lines) - 1))
        self.changed()

    def remove_selected(self) -> None:
        panel, line = self._selected()
        if panel is None:
            return
        if line is None:
            del self.template.panels[panel]
            select = (
                (min(panel, len(self.template.panels) - 1), None) if self.template.panels else None
            )
        else:
            del self.template.panels[panel].lines[line]
            select = (panel, None)
        self.rebuild(select=select)
        self.changed()

    def move_selected(self, step: int) -> None:
        panel, line = self._selected()
        if panel is None:
            return
        if line is None:
            items, index = self.template.panels, panel
        else:
            items, index = self.template.panels[panel].lines, line
        other = index + step
        if not 0 <= other < len(items):
            return
        items[index], items[other] = items[other], items[index]
        self.rebuild(select=(other, None) if line is None else (panel, other))
        self.changed()

    def _selected(self) -> tuple[int | None, int | None]:
        item = self.tree.currentItem()
        if item is None:
            return None, None
        if item.parent() is None:
            return self.tree.indexOfTopLevelItem(item), None
        return self.tree.indexOfTopLevelItem(item.parent()), item.parent().indexOfChild(item)

    def rebuild(self, select: tuple[int, int | None] | None = None) -> None:
        """The figure tree from the template."""
        self._updating = True
        self.tree.clear()
        for number, panel in enumerate(self.template.panels, start=1):
            top = QTreeWidgetItem([self._panel_label(number, panel)])
            for line in panel.lines:
                child = QTreeWidgetItem([line.title()])
                if line.color:
                    child.setForeground(0, QColor(line.color))
                top.addChild(child)
            self.tree.addTopLevelItem(top)
            top.setExpanded(True)
        self._updating = False
        if select is not None:
            panel, line = select
            top = self.tree.topLevelItem(panel)
            if top is not None:
                self.tree.setCurrentItem(top.child(line) if line is not None else top)
        self._show_selected()

    def _panel_label(self, number: int, panel: Panel) -> str:
        label = f"Panel {number}: {PANEL_LABELS.get(panel.kind, panel.kind)}"
        if panel.title:
            label += f" – {panel.title}"
        elif panel.kind == "view" and panel.time is not None:
            label += f" (t = {panel.time:g} s)"
        return label

    def _show_selected(self) -> None:
        """The settings of what is selected (nothing: the figure's)."""
        panel, line = self._selected()
        self._updating = True
        template = self.template
        if panel is None:
            self.settings_stack.setCurrentWidget(self.figure_form)
            self.style_box.setCurrentText(template.style)
            index = self.width_box.findData(template.width_mm)
            if index < 0:
                self.width_box.addItem(f"{template.width_mm:g} mm", template.width_mm)
                index = self.width_box.count() - 1
            self.width_box.setCurrentIndex(index)
            self.height_box.setValue(template.height_mm or 0)
            self.font_box.setValue(template.font_size or 0)
            self.title_box.setText(template.title)
            self.columns_box.setValue(max(int(template.columns or 1), 1))
            self.letters_box.setChecked(template.letters)
        elif line is None:
            self.settings_stack.setCurrentWidget(self.panel_form)
            p = template.panels[panel]
            self.kind_box.setCurrentIndex(max(self.kind_box.findData(p.kind), 0))
            self.ylabel_box.setText(p.ylabel)
            self.ymin_box.setText("" if p.ymin is None else f"{p.ymin:g}")
            self.ymax_box.setText("" if p.ymax is None else f"{p.ymax:g}")
            self.logy_box.setChecked(p.logy)
            self.start_box.setValue(p.start or 0)
            self.end_box.setValue(p.end or 0)
            self.compare_box.setChecked(p.compare)
            self.panel_title_box.setText(p.title)
            self.span_box.setValue(max(int(p.span or 1), 1))
            self.moment_box.setValue(-1 if p.time is None else p.time)
            self.view_box.setCurrentIndex(max(self.view_box.findData(p.view), 0))
            self.bars_box.setChecked(p.colour_bars)
            if self.notes_box.toPlainText() != notes_text(p.notes):
                self.notes_box.setPlainText(notes_text(p.notes))
            self._panel_rows(p.kind)
        else:
            self.settings_stack.setCurrentWidget(self.line_form)
            item = template.panels[panel].lines[line]
            self.label_box.setText(item.label)
            self.color_button.setText(item.color or "Style's colour")
            self.linestyle_box.setCurrentIndex(max(self.linestyle_box.findData(item.linestyle), 0))
            self.marker_box.setCurrentIndex(max(self.marker_box.findData(item.marker), 0))
            self.width_line_box.setValue(item.width)
        self._updating = False

    def _figure_changed(self) -> None:
        if self._updating:
            return
        t = self.template
        t.style = self.style_box.currentText()
        t.width_mm = float(self.width_box.currentData() or t.width_mm)
        t.height_mm = self.height_box.value() or None
        t.font_size = self.font_box.value() or None
        t.title = self.title_box.text().strip()
        t.columns = self.columns_box.value()
        t.letters = self.letters_box.isChecked()
        self.changed()

    def _panel_changed(self) -> None:
        panel, _ = self._selected()
        if self._updating or panel is None:
            return
        p = self.template.panels[panel]
        p.kind = self.kind_box.currentData()
        p.ylabel = self.ylabel_box.text().strip()
        p.ymin, p.ymax = _float(self.ymin_box.text()), _float(self.ymax_box.text())
        p.logy = self.logy_box.isChecked()
        p.start = self.start_box.value() or None
        p.end = self.end_box.value() or None
        p.compare = self.compare_box.isChecked() and p.kind == "time"
        p.title = self.panel_title_box.text().strip()
        p.span = self.span_box.value()
        moment = self.moment_box.value()
        p.time = None if moment < 0 else moment
        p.view = self.view_box.currentData()
        p.colour_bars = self.bars_box.isChecked()
        p.notes = parse_notes(self.notes_box.toPlainText())
        self._panel_rows(p.kind)
        self.tree.topLevelItem(panel).setText(0, self._panel_label(panel + 1, p))
        self.changed()

    def take_view(self) -> None:
        """The selected 3D view panel shows the view as it is now: the camera, the time,
        the look, and 2D from the top when the window shows that."""
        panel, _ = self._selected()
        if panel is None or self.template.panels[panel].kind != "view":
            return
        p = self.template.panels[panel]
        scene = self.session.scene
        view = getattr(getattr(self.main_window, "viewport", None), "view", None)
        camera = view.camera_state() if view is not None else scene.camera
        p.camera = dict(camera) if camera else None
        p.time = float(scene.time)
        p.look = getattr(view, "style_name", "") or ""
        p.view = "top" if scene.plan_view and scene.plan_side == "top" else "camera"
        self._show_selected()
        self.tree.topLevelItem(panel).setText(0, self._panel_label(panel + 1, p))
        self.changed()

    def _line_changed(self) -> None:
        panel, line = self._selected()
        if self._updating or line is None:
            return
        item = self.template.panels[panel].lines[line]
        item.label = self.label_box.text().strip()
        item.linestyle = self.linestyle_box.currentData()
        item.marker = self.marker_box.currentData()
        item.width = self.width_line_box.value()
        self.tree.currentItem().setText(0, item.title())
        self.changed()

    def _choose_colour(self) -> None:
        panel, line = self._selected()
        if line is None:
            return
        item = self.template.panels[panel].lines[line]
        chosen = QColorDialog.getColor(QColor(item.color or "#2a5aa0"), self, "Line colour")
        if not chosen.isValid():
            return
        item.color = chosen.name()
        self.color_button.setText(item.color)
        self.tree.currentItem().setForeground(0, chosen)
        self.changed()

    def changed(self) -> None:
        """Draw the preview again soon (edits come in bursts)."""
        self._redraw.start()

    # ------------------------------------------------------------- drawing
    def figure(self, dpi: int = 300):
        from lagoonviz.render.figure import template_figure, template_spec

        spec = template_spec(self.template, dpi=dpi)
        return template_figure(self.session, self.template, spec), spec

    def draw_preview(self) -> None:
        if not any(p.lines or p.kind == "view" for p in self.template.panels):
            self.preview.setPixmap(QPixmap())
            self.preview.setText("Drag a time series onto the figure")
            self.problems.clear()
            return
        import io

        import matplotlib

        try:
            figure, spec = self.figure(dpi=110)
            buffer = io.BytesIO()
            with matplotlib.rc_context(spec.rc()):
                figure.savefig(buffer, format="png", dpi=110)
        except Exception as error:  # noqa: BLE001 (shown in the window)
            log.info("figure preview failed:\n%s", traceback.format_exc())
            self.problems.setText(f"Cannot draw: {error}")
            return
        image = QImage()
        image.loadFromData(buffer.getvalue(), "PNG")
        pixmap = QPixmap.fromImage(image)
        room = self.preview.size()
        if room.width() > 50 and room.height() > 50:  # not laid out yet: as drawn
            pixmap = pixmap.scaled(room, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.preview.setPixmap(pixmap)
        self.problems.setText("Left out: " + "; ".join(figure.missing) if figure.missing else "")

    # ------------------------------------------------------------- files
    def open_template(self, path: str | None = None) -> None:
        if not path:
            path, _ = QFileDialog.getOpenFileName(
                self, "Open figure template", "", f"Figure template (*{SUFFIX})"
            )
        if not path:
            return
        try:
            self.template = FigureTemplate.load(path)
        except (OSError, ValueError) as error:
            QMessageBox.warning(self, "Open figure template", str(error))
            return
        for panel in self.template.panels:
            for line in panel.lines:
                if line.file and line.file not in self._measured:
                    self._measured.append(line.file)
        self.fill_sources()
        self.rebuild()
        self.changed()

    def save_template(self, path: str | None = None) -> str | None:
        if not path:
            path, _ = QFileDialog.getSaveFileName(
                self, "Save figure template", f"figure{SUFFIX}", f"Figure template (*{SUFFIX})"
            )
        if not path:
            return None
        written = self.template.save(path)
        self.main_window.statusBar().showMessage(f"Figure template saved to {written}", 10000)
        return str(written)

    def export(self, path: str | None = None, dpi: int = 300) -> str | None:
        from lagoonviz.render.figure import save_figure

        if not path:
            path, _ = QFileDialog.getSaveFileName(
                self,
                "Export figure",
                "figure.pdf",
                "PDF (*.pdf);;SVG (*.svg);;EPS (*.eps);;PNG (*.png);;TIFF (*.tif)",
            )
        if not path:
            return None
        try:
            figure, spec = self.figure(dpi=dpi)
            written = save_figure(figure, path, spec)
        except Exception as error:  # noqa: BLE001 (reported)
            QMessageBox.warning(self, "Export figure", f"The figure could not be made:\n{error}")
            return None
        from lagoonviz.render.script import template_code

        self.session.record_output(
            template_code(self.template.to_dict(), written, dpi, self.session.scene.time)
        )
        self.main_window.statusBar().showMessage(f"Figure saved to {written}", 15000)
        return str(written)


def _number(low: float, high: float, suffix: str, special: str) -> QDoubleSpinBox:
    box = QDoubleSpinBox()
    box.setRange(low, high)
    box.setDecimals(1)
    box.setSuffix(suffix)
    box.setSpecialValueText(special)
    box.setKeyboardTracking(False)
    return box


def _float(text: str) -> float | None:
    try:
        return float(text.replace(",", "."))
    except ValueError:
        return None


def open_figure_editor(window) -> FigureEditor | None:
    """File → Figure editor (one per window, raised when open again)."""
    try:
        import matplotlib  # noqa: F401
    except ImportError:
        QMessageBox.warning(
            window, "Figure editor", "Figures need matplotlib; run setup_lagoon.command again."
        )
        return None
    editor = getattr(window, "_figure_editor", None)
    if editor is None:
        editor = window._figure_editor = FigureEditor(window)
    else:
        editor.fill_sources()
    editor.show()
    editor.raise_()
    editor.activateWindow()
    return editor
