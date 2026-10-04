# SPDX-License-Identifier: GPL-3.0-or-later
"""The scene: everything a LAGOON session shows, as plain serialisable data.

The scene knows nothing about Qt, GPUs or plotting. The GUI and the renderers
observe it; commands change it.
"""

from __future__ import annotations

import copy
import json
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from lagoonviz.core.kinds import KINDS, default_properties

#: Version of the session file layout written by :meth:`Scene.save`.
SESSION_FORMAT = 1

#: Observer signature: ``callback(event, item_id)``.
#: Events: ``"added"``, ``"removed"``, ``"changed"``, ``"time"``, ``"reset"``.
Observer = Callable[[str, str | None], None]


#: the sides a 2D view can look from
PLAN_SIDES = ("top", "front", "back", "right", "left")


@dataclass
class SceneItem:
    """One entry in the scene tree: a case, a surface, a slice, a gauge, ..."""

    id: str
    kind: str
    name: str
    visible: bool = True
    properties: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "kind": self.kind,
            "name": self.name,
            "visible": self.visible,
            "properties": copy.deepcopy(self.properties),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SceneItem:
        return cls(
            id=data["id"],
            kind=data["kind"],
            name=data["name"],
            visible=bool(data.get("visible", True)),
            properties=copy.deepcopy(data.get("properties", {})),
        )


class Scene:
    """Ordered collection of scene items plus the current time.

    Item ids are deterministic (``<kind>-<n>``), so a recorded script replayed in a
    fresh session produces the same ids and the same scene.
    """

    def __init__(self) -> None:
        self._items: dict[str, SceneItem] = {}
        self._counters: dict[str, int] = {}
        self._observers: list[Observer] = []
        self.time: float = 0.0
        #: saved 3D view (camera state from the renderer); view state, not undoable
        self.camera: dict[str, Any] | None = None
        #: 2D: orthographic, no rotation (view state, not undoable)
        self.plan_view: bool = False
        #: the side the 2D view looks from: "top" (everything drawn flat), or "front",
        #: "back", "right", "left" (an elevation of the shapes as they are)
        self.plan_side: str = "top"

    # ------------------------------------------------------------- observers
    def subscribe(self, callback: Observer) -> None:
        self._observers.append(callback)

    def unsubscribe(self, callback: Observer) -> None:
        self._observers.remove(callback)

    def _notify(self, event: str, item_id: str | None = None) -> None:
        for callback in list(self._observers):
            callback(event, item_id)

    # ------------------------------------------------------------- access
    def __iter__(self) -> Iterator[SceneItem]:
        return iter(self._items.values())

    def __len__(self) -> int:
        return len(self._items)

    def __contains__(self, item_id: object) -> bool:
        return item_id in self._items

    def get(self, item_id: str) -> SceneItem:
        try:
            return self._items[item_id]
        except KeyError:
            raise KeyError(f"no scene item with id {item_id!r}") from None

    def ids(self) -> list[str]:
        return list(self._items)

    def index_of(self, item_id: str) -> int:
        return self.ids().index(item_id)

    # ------------------------------------------------------------- changes
    # These are called by commands. User code should go through Session, so that
    # every change is undoable and recorded.
    def new_id(self, kind: str) -> str:
        n = self._counters.get(kind, 0) + 1
        self._counters[kind] = n
        return f"{kind}-{n}"

    def release_id(self, item_id: str) -> None:
        """Give back the most recently issued id of a kind (undo of an add)."""
        kind, _, number = item_id.rpartition("-")
        if number.isdigit() and self._counters.get(kind) == int(number):
            self._counters[kind] = int(number) - 1

    def claim_id(self, item_id: str) -> None:
        """Mark an id as issued again (redo of an add)."""
        kind, _, number = item_id.rpartition("-")
        if number.isdigit():
            self._counters[kind] = max(self._counters.get(kind, 0), int(number))

    def insert(self, item: SceneItem, index: int | None = None) -> None:
        if item.id in self._items:
            raise ValueError(f"duplicate scene item id {item.id!r}")
        if item.kind not in KINDS:
            raise ValueError(f"unknown item kind {item.kind!r}")
        merged = default_properties(item.kind)
        merged.update(item.properties)
        item.properties = merged
        entries = list(self._items.items())
        if index is None or index >= len(entries):
            entries.append((item.id, item))
        else:
            entries.insert(index, (item.id, item))
        self._items = dict(entries)
        self._notify("added", item.id)

    def pop(self, item_id: str) -> tuple[SceneItem, int]:
        index = self.index_of(item_id)
        item = self._items.pop(item_id)
        self._notify("removed", item_id)
        return item, index

    def update(self, item_id: str, **changes: Any) -> None:
        """Change ``name``, ``visible`` or entries of ``properties`` of one item."""
        item = self.get(item_id)
        for key, value in changes.items():
            if key == "name":
                item.name = str(value)
            elif key == "visible":
                item.visible = bool(value)
            elif key == "properties":
                item.properties.update(value)
            else:
                raise KeyError(f"cannot update field {key!r}")
        self._notify("changed", item_id)

    def set_time(self, time: float) -> None:
        self.time = float(time)
        self._notify("time", None)

    @property
    def flat(self) -> bool:
        """True when items are drawn flat: the 2D view from the top."""
        return self.plan_view and self.plan_side == "top"

    def set_plan_view(self, on: bool, side: str | None = None) -> None:
        """2D or 3D, and the side 2D looks from (None keeps it); observers draw
        everything again ("view")."""
        side = self.plan_side if side is None else str(side)
        if side not in PLAN_SIDES:
            raise ValueError(f"unknown 2D side {side!r}")
        if bool(on) != self.plan_view or side != self.plan_side:
            self.plan_view = bool(on)
            self.plan_side = side
            self._notify("view", None)

    def data_changed(self) -> None:
        """Tell observers to show everything again: the data behind the items changed
        (a reloaded case). Sent as "reset", like loading a session."""
        self._notify("reset", None)

    # ------------------------------------------------------------- files
    def to_dict(self) -> dict[str, Any]:
        return {
            "lagoon_session": SESSION_FORMAT,
            "time": self.time,
            "camera": copy.deepcopy(self.camera),
            "plan_view": self.plan_view,
            "plan_side": self.plan_side,
            "counters": dict(self._counters),
            "items": [item.to_dict() for item in self._items.values()],
        }

    def load_dict(self, data: dict[str, Any]) -> None:
        version = data.get("lagoon_session")
        if version != SESSION_FORMAT:
            raise ValueError(f"unsupported session format {version!r}")
        items = [SceneItem.from_dict(entry) for entry in data.get("items", [])]
        for item in items:
            if item.kind not in KINDS:
                raise ValueError(f"unknown item kind {item.kind!r}")
            merged = default_properties(item.kind)
            merged.update(item.properties)
            item.properties = merged
        self._items = {item.id: item for item in items}
        self._counters = {str(k): int(v) for k, v in data.get("counters", {}).items()}
        self.time = float(data.get("time", 0.0))
        camera = data.get("camera")
        self.camera = copy.deepcopy(camera) if isinstance(camera, dict) else None
        self.plan_view = bool(data.get("plan_view", False))
        side = str(data.get("plan_side", "top"))
        self.plan_side = side if side in PLAN_SIDES else "top"
        self._notify("reset", None)

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    def load(self, path: str | Path) -> None:
        self.load_dict(json.loads(Path(path).read_text(encoding="utf-8")))
