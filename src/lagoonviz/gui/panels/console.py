# SPDX-License-Identifier: GPL-3.0-or-later
"""Python console: shows every command the GUI runs, and runs Python with ``session`` in scope."""

from __future__ import annotations

import contextlib
import io
import traceback

from PySide6.QtCore import Qt
from PySide6.QtGui import QFontDatabase, QKeyEvent
from PySide6.QtWidgets import QLineEdit, QPlainTextEdit, QVBoxLayout, QWidget

from lagoonviz.gui.bridge import SessionBridge


class _InputLine(QLineEdit):
    """Line edit with up/down history."""

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self.history: list[str] = []
        self._index = 0

    def remember(self, text: str) -> None:
        if text and (not self.history or self.history[-1] != text):
            self.history.append(text)
        self._index = len(self.history)

    def keyPressEvent(self, event: QKeyEvent) -> None:  # noqa: N802 (Qt naming)
        if event.key() == Qt.Key_Up and self.history:
            self._index = max(0, self._index - 1)
            self.setText(self.history[self._index])
        elif event.key() == Qt.Key_Down and self.history:
            self._index = min(len(self.history), self._index + 1)
            self.setText(self.history[self._index] if self._index < len(self.history) else "")
        else:
            super().keyPressEvent(event)


class ConsolePanel(QWidget):
    def __init__(self, bridge: SessionBridge, parent: QWidget | None = None):
        super().__init__(parent)
        self.setObjectName("console")
        self.session = bridge.session
        self.namespace: dict[str, object] = {"session": self.session, "scene": self.session.scene}

        mono = QFontDatabase.systemFont(QFontDatabase.FixedFont)
        self.log = QPlainTextEdit(self)
        self.log.setReadOnly(True)
        self.log.setFont(mono)
        self.log.setMaximumBlockCount(5000)
        self.input = _InputLine(self)
        self.input.setFont(mono)
        self.input.setPlaceholderText(">>> Python, e.g. session.add_item('gauge', x=2.0)")
        self.input.returnPressed.connect(self._run_input)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.log, 1)
        layout.addWidget(self.input)

        bridge.logLine.connect(self.append)
        self.append("# LAGOON console. Commands from the GUI appear here; `session` is in scope.")

    def append(self, text: str) -> None:
        self.log.appendPlainText(text)

    def run(self, source: str) -> None:
        """Run one line of Python (expression results are printed)."""
        self.append(f">>> {source}")
        buffer = io.StringIO()
        try:
            with contextlib.redirect_stdout(buffer):
                try:
                    code = compile(source, "<console>", "eval")
                except SyntaxError:
                    exec(compile(source, "<console>", "exec"), self.namespace)
                else:
                    result = eval(code, self.namespace)
                    if result is not None:
                        print(repr(result))
        except Exception:
            buffer.write(traceback.format_exc(limit=1))
        output = buffer.getvalue().rstrip()
        if output:
            self.append(output)

    def _run_input(self) -> None:
        source = self.input.text().strip()
        self.input.remember(source)
        self.input.clear()
        if source:
            self.run(source)
