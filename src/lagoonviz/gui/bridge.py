# SPDX-License-Identifier: GPL-3.0-or-later
"""Bridge between the Qt-free core and Qt: core callbacks become Qt signals."""

from __future__ import annotations

from PySide6.QtCore import QObject, Signal

from lagoonviz.core import Session
from lagoonviz.core.commands import Command


class SessionBridge(QObject):
    """Wraps a :class:`Session` and re-emits its events as Qt signals."""

    #: (event, item_id) with event "added", "removed", "changed", "reset"; item_id "" if none
    sceneChanged = Signal(str, str)
    #: current time in seconds
    timeChanged = Signal(float)
    #: undo/redo availability or labels changed
    historyChanged = Signal()
    #: one line for the console log
    logLine = Signal(str)

    def __init__(self, session: Session, parent: QObject | None = None):
        super().__init__(parent)
        self.session = session
        session.scene.subscribe(self._on_scene)
        session.stack.subscribe(self._on_stack)
        session.subscribe_log(self.logLine.emit)

    def _on_scene(self, event: str, item_id: str | None) -> None:
        if event == "time":
            self.timeChanged.emit(self.session.scene.time)
        else:
            self.sceneChanged.emit(event, item_id or "")

    def _on_stack(self, kind: str, command: Command | None) -> None:
        self.historyChanged.emit()
