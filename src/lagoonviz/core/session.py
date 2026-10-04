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

import os
from collections.abc import Callable
from pathlib import Path
from typing import Any

from lagoonviz.core.commands import (
    AddItem,
    Command,
    CommandStack,
    OpenCase,
    RemoveItem,
    Rename,
    SetMany,
    SetProperties,
    SetProperty,
    SetVisible,
)
from lagoonviz.core.data.manager import DataManager
from lagoonviz.core.data.reef3d import ROLES
from lagoonviz.core.kinds import KINDS
from lagoonviz.core.scene import Scene

#: Suggested file extension for saved sessions.
SESSION_EXTENSION = ".lagoon-session"

#: Log listener: ``callback(line)`` receives each recorded or undone command as text.
LogListener = Callable[[str], None]


class Session:
    def __init__(self) -> None:
        self.scene = Scene()
        self.stack = CommandStack(self.scene)
        #: opens the cases of the scene and loads frames (runtime state, not saved)
        self.data = DataManager(self.scene)
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

    def open_case(self, path: str | Path, name: str | None = None) -> str:
        """Open a REEF3D case and show its free surface and bed. Returns the case id.

        ``path`` is a case folder, one of its ``REEF3D_*`` output folders, or a file.
        """
        path = os.fspath(path)
        case = self.data.open(path)
        children = []
        for role, dataset in case.main_datasets().items():
            kind = ROLES[role]
            props: dict[str, Any] = {}
            if sum(d.role == role for d in case.datasets.values()) > 1:
                props["dataset"] = dataset.key
            variable = dataset.default_variable()
            if variable != KINDS[kind].properties["variable"].default:
                props["variable"] = variable
            children.append(AddItem(kind, None, props))
        for series in case.main_series():
            children.append(AddItem("series", series.name, {"source": series.key}))
        for profiles in case.main_profiles():
            children.append(AddItem("profile", profiles.name, {"source": profiles.key}))
        command = OpenCase(path, name, children)
        if name is None:
            command.case.name = Path(path).expanduser().name or str(path)
            command.label = f"Open {command.case.name}"
        self.stack.execute(command)
        assert command.item_id is not None
        # a run restarted in the same folder may start later: begin where it has data
        first = self.data.first_time(command.item_id)
        if first is not None and self.scene.time < first:
            self.set_time(first)
        return command.item_id

    def add_slice(self, normal: str = "y", position: float | None = None, **properties: Any) -> str:
        """Add a slice through the volume; without a position it cuts the middle.

        The position is computed here and recorded, so the script and the saved
        session cut at exactly the same place.
        """
        if position is None:
            center = self.data.domain_center("slice", properties)
            position = round(center["xyz".index(normal)], 6) if center else 0.0
        return self.add_item("slice", normal=normal, position=position, **properties)

    def add_iso_surface(
        self, iso_by: str | None = None, iso_value: float | None = None, **properties: Any
    ) -> str:
        """Add an iso-surface of the volume where ``iso_by`` equals ``iso_value``;
        without a variable, a sensible first one (see :meth:`DataManager.iso_default`),
        recorded so the script repeats it. Other properties (e.g. ``variable`` to colour
        by) as for :meth:`add_item`."""
        if iso_by is None:
            chosen = self.data.iso_default(properties)
            iso_by, default = chosen if chosen else ("none", 0.0)
            iso_value = default if iso_value is None else iso_value
        return self.add_item(
            "iso_surface", iso_by=iso_by, iso_value=float(iso_value or 0.0), **properties
        )

    def add_streamlines(
        self, seed_normal: str = "y", seed_position: float | None = None, **properties: Any
    ) -> str:
        """Add streamlines of the volume's velocity, seeded on a plane through the
        middle of the volume (the position is recorded, like a slice's)."""
        if seed_position is None:
            center = self.data.domain_center("streamlines", properties)
            seed_position = round(center["xyz".index(seed_normal)], 6) if center else 0.0
        return self.add_item(
            "streamlines", seed_normal=seed_normal, seed_position=seed_position, **properties
        )

    def remove_item(self, item_id: str) -> None:
        self.scene.get(item_id)
        self.stack.execute(RemoveItem(item_id))

    def set_property(self, item_id: str, key: str, value: Any, *, merge: bool = False) -> None:
        """Set one property.

        ``merge=True`` folds repeated changes (slider drags) into one undo step.
        """
        if key == "auto_range":
            if value:
                self.data.forget_range(item_id)  # found again, for what is shown now
            else:
                # switching it off keeps the colours as they are: the fixed range
                # starts at the automatic one
                shown = self.data.automatic_range(item_id)
                if shown is not None:
                    low, high = (float(f"{v:.6g}") for v in shown)
                    self.stack.execute(
                        SetProperties(
                            self.scene, item_id, {"auto_range": False, "vmin": low, "vmax": high}
                        )
                    )
                    return
        self.stack.execute(SetProperty(self.scene, item_id, key, value), merge=merge)

    def set_exaggeration(self, item_id: str, value: float, *, merge: bool = False) -> None:
        """Vertical exaggeration as an item's slider sets it: while its case has sync on
        (the default) the case's value, so every item of the case follows; else the
        item's own."""
        case = self.data.case_item(item_id)
        if case is not None and case.properties.get("sync_exaggeration", True):
            item_id = case.id
        self.set_property(item_id, "exaggeration", value, merge=merge)

    def sync_exaggeration(self, item_id: str, on: bool) -> None:
        """Switch the case-wide vertical exaggeration of an item's case on or off.

        Switching it off gives every item of the case the case's value as its own, so
        nothing jumps; one undo step.
        """
        case = self.data.case_item(item_id)
        if case is None:
            return
        changes: dict[str, dict[str, Any]] = {case.id: {"sync_exaggeration": bool(on)}}
        if not on:
            value = case.properties.get("exaggeration", 1.0)
            for other in self.scene:
                if (
                    other.id != case.id
                    and "exaggeration" in other.properties
                    and self.data.case_item(other.id) is case
                ):
                    changes[other.id] = {"exaggeration": value}
        self.stack.execute(
            SetMany(
                self.scene,
                changes,
                "Sync exaggeration" if on else "Unsync exaggeration",
                f"session.sync_exaggeration({item_id!r}, {bool(on)!r})",
            )
        )

    def set_properties(self, item_id: str, **values: Any) -> None:
        """Set several properties of one item as one undo step."""
        self.stack.execute(SetProperties(self.scene, item_id, values))

    def fit_range(self, item_id: str) -> tuple[float, float]:
        """Fix the colour range of an item to the values of the current frame."""
        from lagoonviz.core.data.manager import auto_range

        mesh = self.data.load(item_id)
        if mesh is None:
            raise ValueError(f"{item_id}: no data loaded")
        values = mesh.scalar(str(self.scene.get(item_id).properties.get("variable", "none")))
        low, high = auto_range(values)
        self.set_properties(item_id, auto_range=False, vmin=round(low, 6), vmax=round(high, 6))
        return low, high

    def set_visible(self, item_id: str, visible: bool) -> None:
        self.stack.execute(SetVisible(self.scene, item_id, visible))

    def rename(self, item_id: str, name: str) -> None:
        self.stack.execute(Rename(self.scene, item_id, name))

    def reload(self, case_id: str | None = None) -> list[str]:
        """Read the case folders again, keeping the scene, the time and the camera
        (new results of a running simulation). Not an undo step."""
        paths = self.data.reload(case_id)
        self.scene.data_changed()
        return paths

    def set_time(self, time: float) -> None:
        """Move the current time. View state: not an undo step."""
        self.scene.set_time(time)

    def set_plan_view(self, on: bool) -> None:
        """2D: every item drawn flat and seen from straight above (the free surface as
        a plane coloured by its values over the bed). View state: not an undo step."""
        self.scene.set_plan_view(on)

    def set_camera(self, state: dict[str, Any] | None) -> None:
        """Remember the 3D view (saved with the session). View state: not an undo step."""
        self.scene.camera = dict(state) if state else None

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
        # view state last, so the script shows the same frame and view
        if self.scene.time:
            lines.append(f"session.set_time({self.scene.time!r})")
        if self.scene.plan_view:
            lines.append("session.set_plan_view(True)")
        if self.scene.camera:
            lines.append(f"session.set_camera({self.scene.camera!r})")
        return "\n".join(lines) + "\n"

    @staticmethod
    def from_script(source: str) -> Session:
        """Run a recorded script and return the session it builds."""
        namespace: dict[str, Any] = {}
        exec(compile(source, "<lagoon script>", "exec"), namespace)
        return namespace["session"]
