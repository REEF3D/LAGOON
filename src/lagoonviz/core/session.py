# SPDX-License-Identifier: GPL-3.0-or-later
"""Session: the public scripting interface of LAGOON.

The GUI calls exactly these methods, so everything done in the GUI can be done in
a script, and the GUI's history can be saved as a script::

    from lagoonviz.core import Session

    session = Session()
    session.add_item("free_surface", colormap="balance")
    session.set_property("free_surface-1", "opacity", 0.8)
    session.save("flume.lagoon-session")
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

from lagoonviz.core.commands import (
    AddItem,
    Command,
    CommandStack,
    RemoveItem,
    Rename,
    SetProperty,
    SetVisible,
)
from lagoonviz.core.scene import Scene

#: Suggested file extension for saved sessions.
SESSION_EXTENSION = ".lagoon-session"

#: Log listener: ``callback(line)`` receives each recorded or undone command as text.
LogListener = Callable[[str], None]


class Session:
    def __init__(self) -> None:
        self.scene = Scene()
        self.stack = CommandStack(self.scene)
        self.path: Path | None = None
        self._origin: Path | None = None  # file the recorded script starts from
        self._log_listeners: list[LogListener] = []
        self.stack.subscribe(self._on_stack)

    # ------------------------------------------------------------- log
    def subscribe_log(self, callback: LogListener) -> None:
        self._log_listeners.append(callback)

    def _log(self, line: str) -> None:
        for callback in list(self._log_listeners):
            callback(line)

    def _on_stack(self, kind: str, command: Command | None) -> None:
        if command is None:
            return
        if kind == "do":
            self._log(command.to_script())
        elif kind == "merged":
            self._log(f"{command.to_script()}  # final value")
        elif kind == "merge":
            return
        else:
            self._log(f"# {kind}: {command.label}")

    # ------------------------------------------------------------- editing
    def add_item(self, kind: str, name: str | None = None, **properties: Any) -> str:
        """Add a scene item and return its id."""
        command = AddItem(kind, name, properties)
        self.stack.execute(command)
        assert command.item_id is not None
        return command.item_id

    def remove_item(self, item_id: str) -> None:
        self.scene.get(item_id)
        self.stack.execute(RemoveItem(item_id))

    def set_property(self, item_id: str, key: str, value: Any, *, merge: bool = False) -> None:
        """Set one property.

        ``merge=True`` folds repeated changes (slider drags) into one undo step.
        """
        self.stack.execute(SetProperty(self.scene, item_id, key, value), merge=merge)

    def set_visible(self, item_id: str, visible: bool) -> None:
        self.stack.execute(SetVisible(self.scene, item_id, visible))

    def rename(self, item_id: str, name: str) -> None:
        self.stack.execute(Rename(self.scene, item_id, name))

    def set_time(self, time: float) -> None:
        """Move the current time. View state: not an undo step."""
        self.scene.set_time(time)

    def undo(self) -> None:
        self.stack.undo()

    def redo(self) -> None:
        self.stack.redo()

    # ------------------------------------------------------------- files
    def save(self, path: str | Path) -> None:
        path = Path(path)
        self.scene.save(path)
        self.path = path

    @classmethod
    def open(cls, path: str | Path) -> Session:
        session = cls()
        session.load(path)
        return session

    def load(self, path: str | Path) -> None:
        path = Path(path)
        self.scene.load(path)
        self.stack.clear()
        self.path = path
        self._origin = path

    def reset(self) -> None:
        """Start an empty session."""
        self.scene.load_dict(Scene().to_dict())
        self.stack.clear()
        self.path = None
        self._origin = None

    # ------------------------------------------------------------- scripts
    def to_script(self) -> str:
        """The session's history as a runnable Python script."""
        lines = [
            "# LAGOON session script, recorded from the command history",
            "from lagoonviz.core import Session",
            "",
        ]
        if self._origin is not None:
            lines.append(f"session = Session.open({str(self._origin)!r})")
        else:
            lines.append("session = Session()")
        lines += [command.to_script() for command in self.stack.done]
        return "\n".join(lines) + "\n"

    @staticmethod
    def from_script(source: str) -> Session:
        """Run a recorded script and return the session it builds."""
        namespace: dict[str, Any] = {}
        exec(compile(source, "<lagoon script>", "exec"), namespace)
        return namespace["session"]
