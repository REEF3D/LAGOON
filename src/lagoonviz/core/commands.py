# SPDX-License-Identifier: GPL-3.0-or-later
"""Commands: every change to a scene is a command object.

Commands make changes undoable, and each command can write itself as one line of
Python. The command stack is therefore also the script recorder: replaying the
script in a fresh session rebuilds the same scene, with or without a GUI.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any

from lagoonviz.core.kinds import default_properties, kind_spec, validate
from lagoonviz.core.scene import Scene, SceneItem


def _py(value: Any) -> str:
    """Python literal for a value in a recorded script."""
    return repr(value)


class Command(ABC):
    """One undoable change to a scene."""

    #: Short human-readable label, e.g. for "Undo Add slice".
    label: str = "Command"

    @abstractmethod
    def do(self, scene: Scene) -> None: ...

    @abstractmethod
    def undo(self, scene: Scene) -> None: ...

    @abstractmethod
    def to_script(self) -> str:
        """One line of Python that repeats this command on ``session``."""

    def merge(self, other: Command) -> bool:
        """Absorb a directly following command (e.g. slider drags). Default: no."""
        return False


class AddItem(Command):
    def __init__(
        self,
        kind: str,
        name: str | None = None,
        properties: dict[str, Any] | None = None,
        visible: bool = True,
    ):
        spec = kind_spec(kind)
        self.visible = bool(visible)
        self.kind = kind
        self.name = name if name is not None else spec.label
        props = default_properties(kind)
        for key, value in (properties or {}).items():
            props[key] = validate(kind, key, value)
        self.properties = props
        self.overrides = dict(properties or {})
        self.item_id: str | None = None
        self.label = f"Add {self.name}"

    def do(self, scene: Scene) -> None:
        if self.item_id is None:  # first run assigns the id; redo reuses it
            self.item_id = scene.new_id(self.kind)
        else:
            scene.claim_id(self.item_id)
        scene.insert(
            SceneItem(self.item_id, self.kind, self.name, self.visible, dict(self.properties))
        )

    def undo(self, scene: Scene) -> None:
        assert self.item_id is not None
        scene.pop(self.item_id)
        # ids stay deterministic: a replayed script never sees undone adds
        scene.release_id(self.item_id)

    def to_script(self) -> str:
        args = [_py(self.kind)]
        if self.name != kind_spec(self.kind).label:
            args.append(f"name={_py(self.name)}")
        args += [f"{key}={_py(value)}" for key, value in self.overrides.items()]
        return f"session.add_item({', '.join(args)})"


class RemoveItem(Command):
    """Remove an item; a case goes with every item that shows it (its surfaces,
    volumes, slices, time series, ...), as one undo step."""

    def __init__(self, item_id: str):
        self.item_id = item_id
        #: what was taken out, in order, with where it stood
        self._removed: list[tuple[SceneItem, int]] = []
        self.label = f"Remove {item_id}"

    def do(self, scene: Scene) -> None:
        target = scene.get(self.item_id)
        ids = [self.item_id]
        if target.kind == "case":
            ids += _items_of_case(scene, self.item_id)
        self._removed = [scene.pop(item_id) for item_id in ids]
        self.label = f"Remove {self._removed[0][0].name}"

    def undo(self, scene: Scene) -> None:
        for item, index in reversed(self._removed):
            scene.insert(item, index)

    def to_script(self) -> str:
        return f"session.remove_item({_py(self.item_id)})"


def _items_of_case(scene: Scene, case_id: str) -> list[str]:
    """The items that show a case: those naming it, and, while it is the only case,
    those naming none (they show the first case)."""
    only = sum(item.kind == "case" for item in scene) == 1
    found = []
    for item in scene:
        if item.id == case_id or "case" not in item.properties:
            continue
        named = item.properties.get("case") or ""
        if named == case_id or (only and not named):
            found.append(item.id)
    return found


class SetProperty(Command):
    def __init__(self, scene: Scene, item_id: str, key: str, value: Any):
        item = scene.get(item_id)
        self.item_id = item_id
        self.key = key
        self.value = validate(item.kind, key, value)
        self.old = item.properties[key]
        self.label = f"Set {key}"

    def do(self, scene: Scene) -> None:
        scene.update(self.item_id, properties={self.key: self.value})

    def undo(self, scene: Scene) -> None:
        scene.update(self.item_id, properties={self.key: self.old})

    def merge(self, other: Command) -> bool:
        if isinstance(other, SetProperty) and (other.item_id, other.key) == (
            self.item_id,
            self.key,
        ):
            self.value = other.value
            return True
        return False

    def to_script(self) -> str:
        return f"session.set_property({_py(self.item_id)}, {_py(self.key)}, {_py(self.value)})"


class SetProperties(Command):
    """Set several properties of one item in one undo step."""

    def __init__(self, scene: Scene, item_id: str, values: dict[str, Any]):
        item = scene.get(item_id)
        self.item_id = item_id
        self.values = {key: validate(item.kind, key, value) for key, value in values.items()}
        self.old = {key: item.properties[key] for key in self.values}
        self.label = f"Set {', '.join(self.values)}"

    def do(self, scene: Scene) -> None:
        scene.update(self.item_id, properties=dict(self.values))

    def undo(self, scene: Scene) -> None:
        scene.update(self.item_id, properties=dict(self.old))

    def to_script(self) -> str:
        args = ", ".join(f"{key}={_py(value)}" for key, value in self.values.items())
        return f"session.set_properties({_py(self.item_id)}, {args})"


class SetMany(Command):
    """Set properties of several items in one undo step (``{item_id: {key: value}}``),
    recorded as one higher-level session call."""

    def __init__(self, scene: Scene, changes: dict[str, dict[str, Any]], label: str, script: str):
        self.values = {
            item_id: {
                key: validate(scene.get(item_id).kind, key, value) for key, value in values.items()
            }
            for item_id, values in changes.items()
        }
        self.old = {
            item_id: {key: scene.get(item_id).properties[key] for key in values}
            for item_id, values in self.values.items()
        }
        self.label = label
        self.script = script

    def do(self, scene: Scene) -> None:
        for item_id, values in self.values.items():
            scene.update(item_id, properties=dict(values))

    def undo(self, scene: Scene) -> None:
        for item_id, values in reversed(self.old.items()):
            scene.update(item_id, properties=dict(values))

    def to_script(self) -> str:
        return self.script


class SetVisible(Command):
    def __init__(self, scene: Scene, item_id: str, visible: bool):
        self.item_id = item_id
        self.visible = bool(visible)
        self.old = scene.get(item_id).visible
        self.label = "Show" if visible else "Hide"

    def do(self, scene: Scene) -> None:
        scene.update(self.item_id, visible=self.visible)

    def undo(self, scene: Scene) -> None:
        scene.update(self.item_id, visible=self.old)

    def to_script(self) -> str:
        return f"session.set_visible({_py(self.item_id)}, {_py(self.visible)})"


class Rename(Command):
    def __init__(self, scene: Scene, item_id: str, name: str):
        self.item_id = item_id
        self.name = str(name)
        self.old = scene.get(item_id).name
        self.label = "Rename"

    def do(self, scene: Scene) -> None:
        scene.update(self.item_id, name=self.name)

    def undo(self, scene: Scene) -> None:
        scene.update(self.item_id, name=self.old)

    def to_script(self) -> str:
        return f"session.rename({_py(self.item_id)}, {_py(self.name)})"


class OpenCase(Command):
    """Add a case and the items that show its outputs, as one undo step."""

    def __init__(self, path: str, name: str | None, children: list[AddItem]):
        self.path = path
        self.name = name
        self.case = AddItem("case", name, {"path": path})
        self.children = children
        self.label = f"Open {self.case.name}"

    @property
    def item_id(self) -> str | None:
        return self.case.item_id

    def do(self, scene: Scene) -> None:
        self.case.do(scene)
        for child in self.children:
            child.properties["case"] = self.case.item_id
            child.do(scene)

    def undo(self, scene: Scene) -> None:
        for child in reversed(self.children):
            child.undo(scene)
        self.case.undo(scene)

    def to_script(self) -> str:
        args = [_py(self.path)]
        if self.name is not None:
            args.append(f"name={_py(self.name)}")
        return f"session.open_case({', '.join(args)})"


#: Listener signature: ``callback(kind, command)`` with kind ``"do"``, ``"merge"`` (folded
#: into the previous command), ``"merged"`` (merge sequence closed), ``"undo"``, ``"redo"``
#: or ``"clear"`` (command is ``None`` for ``"clear"``).
StackListener = Callable[[str, Command | None], None]


class CommandStack:
    """Undo/redo stack. Its done-list is the recorded script of the session."""

    def __init__(self, scene: Scene, limit: int = 1000):
        self.scene = scene
        self.limit = limit
        self._done: list[Command] = []
        self._undone: list[Command] = []
        self._listeners: list[StackListener] = []
        self._merge_open = False
        self._merged_pending = False

    def subscribe(self, callback: StackListener) -> None:
        self._listeners.append(callback)

    def _notify(self, kind: str, command: Command | None) -> None:
        for callback in list(self._listeners):
            callback(kind, command)

    def execute(self, command: Command, merge: bool = False) -> Command:
        """Run a command. With ``merge=True`` it may fold into the previous one."""
        command.do(self.scene)
        if merge and self._merge_open and self._done and self._done[-1].merge(command):
            merged = self._done[-1]
            self._merged_pending = True
            self._undone.clear()
            self._notify("merge", merged)
            return merged
        else:
            self._done.append(command)
            if len(self._done) > self.limit:
                self._done.pop(0)
            merged = command
        self._merge_open = merge
        self._merged_pending = False
        self._undone.clear()
        self._notify("do", command)
        return merged

    def end_merge(self) -> None:
        """Close a merge sequence, e.g. when a slider is released."""
        if self._merge_open and self._merged_pending and self._done:
            self._notify("merged", self._done[-1])
        self._merge_open = False
        self._merged_pending = False

    def undo(self) -> Command | None:
        if not self._done:
            return None
        command = self._done.pop()
        command.undo(self.scene)
        self._undone.append(command)
        self._merge_open = False
        self._notify("undo", command)
        return command

    def redo(self) -> Command | None:
        if not self._undone:
            return None
        command = self._undone.pop()
        command.do(self.scene)
        self._done.append(command)
        self._merge_open = False
        self._notify("redo", command)
        return command

    def clear(self) -> None:
        self._done.clear()
        self._undone.clear()
        self._merge_open = False
        self._notify("clear", None)

    @property
    def can_undo(self) -> bool:
        return bool(self._done)

    @property
    def can_redo(self) -> bool:
        return bool(self._undone)

    @property
    def undo_label(self) -> str | None:
        return self._done[-1].label if self._done else None

    @property
    def redo_label(self) -> str | None:
        return self._undone[-1].label if self._undone else None

    @property
    def done(self) -> list[Command]:
        return list(self._done)
