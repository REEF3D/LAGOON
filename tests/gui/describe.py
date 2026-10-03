"""Plain-text descriptions of the window, for the agreed snapshots (snapshot_tools).

They record what a user sees and can do: menus with shortcuts, toolbar, panels,
the rows of the properties panel with their editors, values, choices and whether
they are enabled. Not pixels or fonts, so they hold on every platform (except
shortcuts, which follow each platform's conventions: those go per platform).
"""

from __future__ import annotations

from PySide6.QtGui import QKeySequence
from PySide6.QtWidgets import (
    QAbstractButton,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QLabel,
    QLayout,
    QLineEdit,
    QMenu,
    QSlider,
    QSpinBox,
    QWidget,
)


def _num(value: float) -> str:
    return f"{value:.6g}"


def _text(text: str) -> str:
    return text.replace("&", "").replace("\n", " / ")


def _flags(widget: QWidget) -> str:
    flags = []
    if widget.isHidden():
        flags.append("hidden")
    if not widget.isEnabled():
        flags.append("disabled")
    return f" ({', '.join(flags)})" if flags else ""


def widget_line(widget: QWidget) -> str | None:
    """One line for a leaf widget; None for containers (described by their parts)."""
    if isinstance(widget, QCheckBox):
        line = f"checkbox {_text(widget.text())!r} {'on' if widget.isChecked() else 'off'}"
    elif isinstance(widget, QAbstractButton):
        state = ""
        if widget.isCheckable():
            state = " [x]" if widget.isChecked() else " [ ]"
        tip = f" tip={_text(widget.toolTip())!r}" if widget.toolTip() else ""
        line = f"button {_text(widget.text())!r}{state}{tip}"
    elif isinstance(widget, QComboBox):
        items = [widget.itemText(i) for i in range(widget.count())]
        line = f"choice {widget.currentText()!r} of {items}"
    elif isinstance(widget, QDoubleSpinBox):
        line = (
            f"number {_num(widget.value())} in [{_num(widget.minimum())}, "
            f"{_num(widget.maximum())}] decimals {widget.decimals()}"
        )
    elif isinstance(widget, QSpinBox):
        line = f"integer {widget.value()} in [{widget.minimum()}, {widget.maximum()}]"
    elif isinstance(widget, QSlider):
        line = f"slider in [{widget.minimum()}, {widget.maximum()}]"
    elif isinstance(widget, QLineEdit):
        line = f"text {widget.text()!r}"
    elif isinstance(widget, QLabel):
        line = f"label {_text(widget.text())!r}"
    elif widget.layout() is not None:
        return None
    else:
        line = type(widget).__name__
    return line + _flags(widget)


def describe_widget(widget: QWidget, indent: str = "") -> list[str]:
    """A widget and, for containers, the widgets of its layout, in layout order."""
    line = widget_line(widget)
    if line is not None:
        return [indent + line]
    lines = [f"{indent}{type(widget).__name__}{_flags(widget)}"]
    lines += describe_layout(widget.layout(), indent + "  ")
    return lines


def describe_layout(layout: QLayout, indent: str = "") -> list[str]:
    if isinstance(layout, QFormLayout):
        return describe_form(layout, indent)
    lines = []
    for index in range(layout.count()):
        item = layout.itemAt(index)
        if item.widget() is not None:
            lines += describe_widget(item.widget(), indent)
        elif item.layout() is not None:
            lines += describe_layout(item.layout(), indent)
    return lines


def describe_form(form: QFormLayout, indent: str = "") -> list[str]:
    lines = []
    for row in range(form.rowCount()):
        label = form.itemAt(row, QFormLayout.LabelRole)
        field = form.itemAt(row, QFormLayout.FieldRole)
        span = form.itemAt(row, QFormLayout.SpanningRole)
        name = label.widget().text() if label is not None and label.widget() else ""
        seen = set()
        for item in (field, span):
            if item is None or item.widget() is None or id(item.widget()) in seen:
                continue  # a spanning row answers for both roles
            seen.add(id(item.widget()))
            parts = describe_widget(item.widget(), indent + "  ")
            head = parts[0].strip()
            lines.append(f"{indent}{_text(name) + ': ' if name else ''}{head}")
            lines += parts[1:]
    return lines


def describe_menu(menu: QMenu, indent: str = "") -> list[str]:
    lines = []
    for action in menu.actions():
        if action.isSeparator():
            lines.append(f"{indent}----")
            continue
        if action.menu() is not None:
            lines.append(f"{indent}{_text(action.text())} >")
            lines += describe_menu(action.menu(), indent + "  ")
            continue
        shortcuts = [s.toString(QKeySequence.PortableText) for s in action.shortcuts()]
        keys = f"  [{', '.join(shortcuts)}]" if shortcuts else ""
        check = ""
        if action.isCheckable():
            check = "[x] " if action.isChecked() else "[ ] "
        state = "" if action.isEnabled() else " (disabled)"
        lines.append(f"{indent}{check}{_text(action.text())}{keys}{state}")
    return lines


def describe_menus(window) -> str:
    lines = []
    for action in window.menuBar().actions():
        lines.append(_text(action.text()))
        lines += describe_menu(action.menu(), "  ")
    return "\n".join(lines)


def describe_toolbar(window) -> str:
    lines = []
    for action in window.toolbar.actions():
        if action.isSeparator():
            lines.append("----")
        else:
            check = ("[x] " if action.isChecked() else "[ ] ") if action.isCheckable() else ""
            lines.append(f"{check}{_text(action.text())}")
    return "\n".join(lines)


def describe_docks(window) -> str:
    lines = []
    for name, dock in window.docks.items():
        area = window.dockWidgetArea(dock)
        area_name = getattr(area, "name", str(area))
        tabbed = sorted(other.objectName() for other in window.tabifiedDockWidgets(dock))
        lines.append(
            f"{name}: {dock.windowTitle()!r} in {area_name}"
            f"{' hidden' if dock.isHidden() else ''}"
            f"{' tabbed with ' + ', '.join(tabbed) if tabbed else ''}"
        )
    return "\n".join(lines)


def describe_properties(panel, replace: dict[str, str] | None = None) -> str:
    """The properties panel as it is: one line per row."""
    text = "\n".join(describe_widget(panel._scroll.widget()))
    for old, new in (replace or {}).items():
        text = text.replace(old, new)
    return text
