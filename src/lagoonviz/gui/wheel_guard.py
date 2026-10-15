"""The mouse wheel scrolls the panels, not the fields in them.

Over a number box, a choice list or a slider in a scrolled panel (the Properties
and Set-up panels on the right), the wheel scrolls the panel unless that field has
the focus (clicked into first): scrolling down a panel no longer turns a value or
an option by accident. Fields outside scrolled panels (the time slider) keep the
wheel.
"""

from __future__ import annotations

from PySide6.QtCore import QEvent, QObject, Qt
from PySide6.QtWidgets import (
    QAbstractScrollArea,
    QAbstractSlider,
    QAbstractSpinBox,
    QApplication,
    QComboBox,
    QScrollBar,
    QWidget,
)

#: the fields the wheel leaves alone without focus
GUARDED = (QAbstractSpinBox, QComboBox, QAbstractSlider)


def guarded(widget: QObject) -> bool:
    """Whether the wheel over this widget goes to the panel around it: a number box,
    choice list or slider (not a scroll bar) inside a scrolled panel, not focused."""
    if not isinstance(widget, GUARDED) or isinstance(widget, QScrollBar):
        return False
    if widget.hasFocus():
        return False
    parent = widget.parentWidget()
    while parent is not None:
        if isinstance(parent, QAbstractScrollArea):
            return True
        parent = parent.parentWidget()
    return False


class WheelGuard(QObject):
    """An application event filter: the wheel over a guarded field is ignored there,
    so Qt hands it on to the panel around it, which scrolls."""

    def eventFilter(self, watched: QObject, event: QEvent) -> bool:  # noqa: N802 (Qt)
        if event.type() == QEvent.Wheel and guarded(watched):
            if isinstance(watched, QWidget) and watched.focusPolicy() == Qt.WheelFocus:
                watched.setFocusPolicy(Qt.StrongFocus)  # the wheel gives no focus either
            event.ignore()
            return True  # not for the field; ignored, it goes on to its parents
        return False


def install(app: QApplication | None = None) -> WheelGuard | None:
    """Install the guard once on the application (MainWindow does)."""
    app = app or QApplication.instance()
    if app is None:
        return None
    guard = getattr(app, "_lagoon_wheel_guard", None)
    if guard is None:
        guard = WheelGuard(app)
        app.installEventFilter(guard)
        app._lagoon_wheel_guard = guard
    return guard
