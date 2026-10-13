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

import math
import os
from collections.abc import Callable
from pathlib import Path
from typing import Any

from lagoonviz.core.commands import (
    AddItem,
    Command,
    CommandStack,
    EditInput,
    OpenCase,
    RemoveItem,
    Rename,
    SetGeoref,
    SetMany,
    SetMode,
    SetProperties,
    SetProperty,
    SetVisible,
)
from lagoonviz.core.data.manager import DataManager
from lagoonviz.core.data.reef3d import ROLES
from lagoonviz.core.data.setup import has_inputs
from lagoonviz.core.kinds import KINDS
from lagoonviz.core.scene import MODES, Scene

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
        #: figures, films and screenshots made: (command they followed, script code)
        self._outputs: list[tuple[Command | None, str]] = []
        self._log_listeners: list[LogListener] = []
        self.stack.subscribe(self._on_stack)
        #: where the last geo.dat came from (make_geo_dat): file name or web sources
        self.geo_sources: list[str] = []

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
        """Open a REEF3D case and show its free surface and bed (its volume is there,
        switched off). Returns the case id.

        ``path`` is a case folder, one of its ``REEF3D_*`` output folders, or a file.
        A case folder without output yet shows its set-up from the input files
        (control.txt, ctrl.txt, geo.dat): DIVEMesh's solids and a "Case set-up" item
        with the domain, the kinds of its sides, relaxation zones, gauges and probes. A
        case with results and input files gets that item too, switched off.
        """
        path = os.fspath(path)
        case = self.data.open(path)
        children = []
        for role, dataset in case.main_datasets().items():
            kind = ROLES[role]
            props: dict[str, Any] = {}
            if sum(d.role == role for d in case.datasets.values()) > 1 or role in ("solid", "topo"):
                props["dataset"] = dataset.key  # (solids: a structure; DIVEMesh's bed: a bed)
            variable = dataset.default_variable()
            if variable != KINDS[kind].properties["variable"].default:
                props["variable"] = variable
            children.append(AddItem(kind, None, props))
        volume = case.main_volume()
        if volume is not None:  # in the scene, switched off: a tick shows it
            props = {}
            if sum(d.role == "volume" for d in case.datasets.values()) > 1:
                props["dataset"] = volume.key
            children.append(AddItem("volume", None, props, visible=False))
        if case.setup_only:  # not run yet: domain, sides, zones, gauges of the input
            children.append(AddItem("setup", None, {}))
            if (Path(case.path) / "control.txt").is_file():
                children.append(AddItem("grid", None, {}))  # DIVEMesh's, or control.txt's
        elif has_inputs(case.path):  # results: the set-up is there, switched off
            children.append(AddItem("setup", None, {}, visible=False))
        for series in case.main_series():
            children.append(AddItem("series", series.name, {"source": series.key}))
        for profiles in case.main_profiles():
            children.append(AddItem("profile", profiles.name, {"source": profiles.key}))
        command = OpenCase(path, name, children)
        if name is None:
            given = Path(path).expanduser()
            if given.name == "REEF3D_Case":  # REEF3D's run log folder opens its case
                given = Path(case.path)
            command.case.name = given.name or str(path)
            command.label = f"Open {command.case.name}"
        self.stack.execute(command)
        assert command.item_id is not None
        if case.setup_only and self.scene.mode != "setup":
            self.set_mode("setup")  # nothing has run: work on its set-up
        # a run restarted in the same folder may start later: begin where it has data
        first = self.data.first_time(command.item_id)
        if first is not None and self.scene.time < first:
            self.set_time(first)
        return command.item_id

    def new_case(self, path: str | Path, name: str | None = None) -> str:
        """Start a case in a folder (made if needed): its control.txt and ctrl.txt are
        started with a comment line if they are not there, and the case opens in the
        set-up, ready for the commands. Returns the case id."""
        folder = Path(path).expanduser()
        folder.mkdir(parents=True, exist_ok=True)
        for file, program in (("control.txt", "DIVEMesh"), ("ctrl.txt", "REEF3D")):
            target = folder / file
            if not target.exists():
                target.write_text(f"// {program} input of {folder.name}, started in LAGOON\n")
        return self.open_case(folder, name)

    # ------------------------------------------------------------- a case from the map
    def new_case_from_map(
        self,
        place=None,
        length: float = 1000.0,
        width: float | None = None,
        name: str | None = None,
    ) -> str:
        """Start a case on the map: a draft case (in LAGOON's drafts folder, see
        :func:`drafts_folder`) with a domain ``length`` x ``width`` metres (B 10, cells
        of a hundredth of the length, B 1) centred on ``place`` (a name or
        "lat, lon" to search, a :class:`~lagoonviz.core.geo.places.Place`, or (lat,
        lon)), a background map round it, in the set-up's 2D view. Draw the domain,
        add the bathymetry, then :meth:`create_case` puts it in its folder. Returns
        the case id."""
        width = length / 2 if width is None else float(width)
        folder = _new_draft()
        cell = float(f"{length / 100:.2g}")
        (folder / "control.txt").write_text(
            "// DIVEMesh input, started on the map in LAGOON\n"
            f"B 1 {cell:g}\n"
            f"B 10 0.0 {float(length):g} 0.0 {float(width):g} 0.0 1.0\n"
        )
        (folder / "ctrl.txt").write_text("// REEF3D input, started on the map in LAGOON\n")
        case_id = self.open_case(folder, name)
        for item in [i for i in self.scene if i.kind == "grid"]:
            if self.data.case_item(item.id) is self.scene.get(case_id) and item.visible:
                self.set_visible(item.id, False)  # its lines would hide the map
        if place is not None:
            self.place_case(place, case=case_id)
        if self.scene.mode != "setup":
            self.set_mode("setup")
        self.set_plan_view(True, "top")
        return case_id

    def place_case(self, place, case: str | None = None, crs: str | None = None):
        """Put the centre of a case's domain at a place on the Earth (a name or "lat,
        lon" searched for, the first place found; a Place; or (lat, lon)): its
        georeference, in its CRS (the UTM zone of the place when it has none yet),
        the rotation kept; ``crs``: another CRS in metres to use. Gives the case a
        background map if it has none. Returns the place."""
        from lagoonviz.core.geo.crs import is_geographic, transformer, utm_for
        from lagoonviz.core.geo.places import Place, search

        if isinstance(place, str):
            found = search(place)
            if not found:
                raise ValueError(f"no place found for {place!r}")
            place = found[0]
        elif not isinstance(place, Place):
            lat, lon = (float(v) for v in place)
            place = Place(f"{lat:.5f}, {lon:.5f}", lat, lon, "coordinates")
        case_id = self._case_id(case)
        current = self.georef(case_id)
        if crs is None or is_geographic(crs):  # given: a CRS in metres, else the case's
            crs = current.crs if current is not None else None
        if crs is None or is_geographic(crs):
            crs = utm_for(place.lon, place.lat)
        rotation = current.rotation_deg if current is not None else 0.0
        big_x, big_y = transformer("EPSG:4326", crs)(place.lon, place.lat)
        domain = self.data.backgrounds.domain(case_id)
        cx, cy = ((domain[0] + domain[1]) / 2, (domain[2] + domain[3]) / 2) if domain else (0, 0)
        r = math.radians(rotation)
        origin = (
            round(float(big_x) - (cx * math.cos(r) - cy * math.sin(r)), 3),
            round(float(big_y) - (cx * math.sin(r) + cy * math.cos(r)), 3),
        )
        self.set_georef(
            crs,
            origin=origin,
            rotation_deg=rotation,
            vertical_datum=current.vertical_datum if current is not None else "",
            z_offset=current.z_offset if current is not None else 0.0,
            case=case,
        )
        owner = self.data.case_item(case_id)
        has_map = any(
            i.kind == "basemap" and (c := self.data.case_item(i.id)) is not None and c is owner
            for i in self.scene
        )
        if not has_map:
            from lagoonviz.core.data.background import SOURCE_CHOICES
            from lagoonviz.core.geo.tiles import source_for

            key = source_for(place.lon, place.lat).key
            source = next(label for label, k in SOURCE_CHOICES.items() if k == key)
            self.add_background_map(source, case=case_id)
        return place

    def is_draft(self, case: str | None = None) -> bool:
        """Whether a case is a draft started on the map, not yet in a folder of its
        own (:meth:`create_case`)."""
        folder = self.data.case_folder(self._case_id(case))
        if folder is None:
            return False
        try:
            folder.resolve().relative_to(drafts_folder().resolve())
        except ValueError:
            return False
        return True

    def create_case(self, folder: str | Path, case: str | None = None) -> Path:
        """Put a case (a draft started on the map) in its folder: its input files
        (edits saved first), georef.json, geo.dat and what DIVEMesh made are copied
        there, and the case is that folder from now on (named after it). The folder
        is made if needed and must be empty. Returns the folder."""
        import shutil

        case_id = self._case_id(case)
        source = self.data.case_folder(case_id)
        if source is None:
            raise ValueError(f"{case_id}: no case folder")
        target = Path(folder).expanduser()
        if target.exists() and any(target.iterdir()):
            raise ValueError(f"{target} is not empty: choose a new or empty folder")
        if target.resolve() == source.resolve():
            return target
        self.save_inputs(case_id)
        shutil.copytree(source, target, dirs_exist_ok=True)
        self.set_property(case_id, "path", str(target))
        self.rename(case_id, target.name)
        self.scene.inputs_changed(case_id)
        return target

    def set_mode(self, mode: str) -> None:
        """Work on the results ("results") or on the set-up ("setup") of the cases.

        In the set-up, results items are not drawn and the timeline rests; each case
        with input files gets a "Case set-up" item if it has none. Set-up items keep
        a check box of their own in each mode."""
        if mode not in MODES:
            raise ValueError(f"mode must be one of {', '.join(MODES)}, got {mode!r}")
        added = []
        if mode == "setup":
            for case_item in [i for i in self.scene if i.kind == "case"]:
                added += self._missing_setup_items(case_item.id)
        self.stack.execute(SetMode(self.scene, mode, added))

    def _missing_setup_items(self, case_id: str) -> list[AddItem]:
        """A "Case set-up" item for a case with input files, and a "DIVEMesh grid" item
        for one with control.txt, if it has none yet."""
        cases = [i for i in self.scene if i.kind == "case"]
        found = []
        for kind in ("setup", "grid"):
            named = {str(i.properties.get("case", "")) for i in self.scene if i.kind == kind}
            if case_id in named or (len(cases) == 1 and "" in named):
                continue
            setup = self.data.case_setup(case_id)
            if setup is None or (kind == "grid" and not setup.control.exists):
                continue
            found.append(AddItem(kind, None, {"case": case_id}))
        return found

    # ------------------------------------------------------------- input files
    def input_file(self, file: str, case: str | None = None):
        """A case's ``ctrl.txt`` or ``control.txt`` as it is being edited (a
        :class:`~lagoonviz.core.data.controlfile.ControlFile`; empty if the file does
        not exist yet). ``case``: the case item (default: the first)."""
        case_id = self._case_id(case)
        setup = self.data.case_setup(case_id, create=True)  # none yet: written on save
        if setup is None:
            raise ValueError(f"{case_id}: no case folder")
        if file == "ctrl.txt":
            return setup.ctrl
        if file == "control.txt":
            return setup.control
        raise ValueError(f"file must be 'ctrl.txt' or 'control.txt', got {file!r}")

    def _case_id(self, case: str | None) -> str:
        if case:
            if case not in self.scene or self.scene.get(case).kind != "case":
                raise ValueError(f"{case!r} is not a case item")
            return case
        found = next((i.id for i in self.scene if i.kind == "case"), None)
        if found is None:
            raise ValueError("open a case first")
        return found

    def _edit_input(self, file: str, case: str | None, change, script: str, label: str) -> None:
        case_id = self._case_id(case)
        current = self.input_file(file, case_id)
        changed = current.copy()
        change(changed)
        if changed.text() == current.text():
            return
        call = script + (f", case={case!r})" if case else ")")
        self.stack.execute(
            EditInput(lambda: self.input_file(file, case_id), case_id, changed.text(), call, label)
        )

    def set_input(
        self, file: str, code: str, values, occurrence: int = 0, case: str | None = None
    ) -> None:
        """Give a command of an input file new values: its ``occurrence``-th line (a
        new line if there are fewer), e.g. ``set_input("ctrl.txt", "P 51", [1.0, 0.5],
        occurrence=1)`` moves the second wave gauge. Edits are kept until
        :meth:`save_inputs`."""
        values = [float(v) if not isinstance(v, str) else v for v in values]
        script = f"session.set_input({file!r}, {code!r}, {values!r}"
        if occurrence:
            script += f", occurrence={occurrence}"
        self._edit_input(
            file,
            case,
            lambda f: f.set(code, values, occurrence),
            script,
            f"Set {code}",
        )

    def add_input(self, file: str, code: str, values, case: str | None = None) -> int:
        """One more line of a command (another gauge, solid, ...); returns its
        occurrence."""
        values = [float(v) if not isinstance(v, str) else v for v in values]
        self._edit_input(
            file,
            case,
            lambda f: f.add(code, values),
            f"session.add_input({file!r}, {code!r}, {values!r}",
            f"Add {code}",
        )
        return len(self.input_file(file, case).entries(code)) - 1

    def remove_input(
        self, file: str, code: str, occurrence: int | None = None, case: str | None = None
    ) -> None:
        """Remove a command's line (all its lines with ``occurrence`` None)."""
        script = f"session.remove_input({file!r}, {code!r}"
        if occurrence is not None:
            script += f", occurrence={occurrence}"
        self._edit_input(file, case, lambda f: f.remove(code, occurrence), script, f"Remove {code}")

    def set_input_text(self, file: str, text: str, case: str | None = None) -> None:
        """Replace the whole text of an input file (the text editor)."""
        self._edit_input(
            file,
            case,
            lambda f: f.set_text(text),
            f"session.set_input_text({file!r}, {text!r}",
            f"Edit {file}",
        )

    def save_inputs(self, case: str | None = None) -> list[Path]:
        """Write the edited input files of a case (the first by default); returns
        the files written. Not an undo step."""
        case_id = self._case_id(case)
        written = []
        for name in ("control.txt", "ctrl.txt"):
            file = self.input_file(name, case_id)
            if file.modified:
                written.append(file.write())
        if written:
            self.scene.inputs_changed(case_id)
        return written

    def inputs_modified(self, case: str | None = None) -> list[str]:
        """The input files of a case with edits not saved yet."""
        case_id = self._case_id(case)
        return [n for n in ("control.txt", "ctrl.txt") if self.input_file(n, case_id).modified]

    def run_divemesh(
        self,
        case_id: str | None = None,
        executable: str | Path | None = None,
        on_line=None,
    ) -> int:
        """Run DIVEMesh on a case (the first by default) and show what it made: its
        grid, solids and bed. Edits of control.txt not saved yet are written first
        (DIVEMesh reads the file). Returns DIVEMesh's exit code."""
        from lagoonviz.core.divemesh_run import run_divemesh

        if case_id is None:
            case_id = next((i.id for i in self.scene if i.kind == "case"), None)
        if case_id is None:
            raise ValueError("open a case first")
        folder = self.data.case_folder(case_id)
        if folder is None:
            raise ValueError(f"{case_id}: no case folder")
        setup = self.data.case_setup(case_id)
        if setup is not None and setup.control.modified:
            setup.control.write()
        code = run_divemesh(folder, executable, on_line)
        self.divemesh_done(case_id)
        return code

    def divemesh_done(self, case_id: str) -> None:
        """After DIVEMesh has run on a case: read its outputs again and give the case a
        grid item if it has none (the GUI runs DIVEMesh itself and calls this)."""
        self.data.reload(case_id)
        missing = [a for a in self._missing_setup_items(case_id) if a.kind == "grid"]
        for add in missing:
            self.add_item("grid", **add.overrides)
        case = self.data.case(case_id)
        topo = case.datasets.get("divemesh_topo") if case is not None else None
        shown = {str(i.properties.get("dataset", "")) for i in self.scene if i.kind == "bathymetry"}
        if topo is not None and topo.key not in shown and case.setup_only:
            self.add_item("bathymetry", case=case_id, dataset=topo.key)  # DIVEMesh's bed
        self.scene.data_changed()

    # ------------------------------------------------------------- the Earth
    def georef(self, case: str | None = None):
        """Where a case's model coordinates lie on the Earth (a
        :class:`lagoon_format.geo.Georef`: CRS, origin, rotation, vertical datum), or
        None. From its ``georef.json``, else its store or run log."""
        return self.data.georef(self._case_id(case))

    def set_georef(
        self,
        crs: str,
        origin=(0.0, 0.0),
        rotation_deg: float = 0.0,
        vertical_datum: str = "",
        z_offset: float = 0.0,
        case: str | None = None,
    ) -> None:
        """Georeference a case: its model's (0, 0) lies at ``origin`` in ``crs``
        ("EPSG:25833"), its x-axis turned ``rotation_deg`` counter-clockwise from east,
        its z = 0 at ``z_offset`` in the vertical datum. Written to ``georef.json`` in
        the case folder at once (REEF3D and DIVEMesh read it from there)."""
        from lagoon_format.geo import Georef

        from lagoonviz.core.geo.crs import is_geographic

        if is_geographic(str(crs)):
            raise ValueError(
                f"{crs} is in degrees: the model needs a CRS in metres (UTM, NTM, ...)"
            )
        case_id = self._case_id(case)
        current = self._georef_file(case_id)
        georef = Georef(
            str(crs),
            (float(origin[0]), float(origin[1])),
            float(rotation_deg),
            str(vertical_datum or ""),
            float(z_offset),
            extra=dict(current.extra) if current is not None else {},
        )
        script = f"session.set_georef({str(crs)!r}, origin={[georef.origin[0], georef.origin[1]]!r}"
        if georef.rotation_deg:
            script += f", rotation_deg={georef.rotation_deg!r}"
        if georef.vertical_datum:
            script += f", vertical_datum={georef.vertical_datum!r}"
        if georef.z_offset:
            script += f", z_offset={georef.z_offset!r}"
        script += f", case={case!r})" if case else ")"
        self._set_georef(case_id, georef, script, "Set georeference")

    def clear_georef(self, case: str | None = None) -> None:
        """Remove a case's georef.json."""
        case_id = self._case_id(case)
        if self._georef_file(case_id) is None:
            return
        script = "session.clear_georef(" + (f"case={case!r})" if case else ")")
        self._set_georef(case_id, None, script, "Remove georeference")

    def turn_model(self, rotation_deg: float, about=None, case: str | None = None) -> None:
        """Turn a case's model on the map to a new rotation (degrees counter-clockwise
        from east), about a model point (default: the centre of its domain), which
        stays where it is on the map. The domain keeps its model coordinates."""
        case_id = self._case_id(case)
        current = self.georef(case_id)
        if current is None:
            raise ValueError(f"{case_id}: set the georeference first (set_georef)")
        if about is None:
            domain = self.data.backgrounds.domain(case_id)
            about = (
                ((domain[0] + domain[1]) / 2, (domain[2] + domain[3]) / 2)
                if domain is not None
                else (0.0, 0.0)
            )
        about = (float(about[0]), float(about[1]))
        turned = current.rotated(float(rotation_deg), about)
        if turned == current:
            return
        script = f"session.turn_model({float(rotation_deg)!r}, about={list(about)!r}"
        script += f", case={case!r})" if case else ")"
        self._set_georef(case_id, turned, script, "Turn the model")

    def _georef_file(self, case_id: str):
        from lagoon_format.geo import GeorefError
        from lagoon_format.geo import read as read_georef

        folder = self.data.case_folder(case_id)
        if folder is None:
            raise ValueError(f"{case_id}: no case folder")
        try:
            return read_georef(folder)
        except GeorefError:
            return None  # replaced by the new one

    def _set_georef(self, case_id: str, georef, script: str, label: str) -> None:
        folder = self.data.case_folder(case_id)
        if folder is None:
            raise ValueError(f"{case_id}: no case folder")
        folder.mkdir(parents=True, exist_ok=True)
        self.stack.execute(SetGeoref(folder, case_id, georef, script, label))

    def set_domain(
        self, x0: float, x1: float, y0: float, y1: float, case: str | None = None
    ) -> None:
        """The domain's horizontal extent (B 10 of control.txt, model coordinates);
        its heights stay as they are (0 to 1 m if there are none)."""
        if not (x1 > x0 and y1 > y0):
            raise ValueError(f"the domain needs x0 < x1 and y0 < y1, got {x0, x1, y0, y1}")
        case_id = self._case_id(case)
        current = self.input_file("control.txt", case_id).values("B 10") or []
        z0, z1 = (current[4], current[5]) if len(current) >= 6 else (0.0, 1.0)
        values = [float(x0), float(x1), float(y0), float(y1), float(z0), float(z1)]
        self.set_input("control.txt", "B 10", [round(v, 6) for v in values], case=case)

    def make_geo_dat(
        self,
        source: str | Path,
        crs: str | None = None,
        method: str = "tin",
        spacing: float | None = None,
        depths: bool = False,
        case: str | None = None,
    ) -> int:
        """Bathymetry for DIVEMesh: a GeoTIFF, ASCII grid or x y z file (in ``crs``,
        or its own; ``"model"``: already in model coordinates) interpolated (``"tin"``,
        ``"idw"``, ``"nearest"``) onto a grid over the domain and written as geo.dat
        in model coordinates, every ``spacing`` metres (default: DIVEMesh's cell size,
        B 1). ``depths``: the file gives depths (positive down). Turns G 10 on.

        No file: ``source="best"`` downloads the bathymetry for the domain
        (Kartverket's land and sea model in Norway, EMODnet for the rest of
        European seas), ``"kartverket"`` or ``"emodnet"`` one of them.
        Returns the number of points written."""
        from lagoon_format.geo import Georef

        from lagoonviz.core.geo.bathy import (
            WEB,
            geo_grid,
            grid_heights,
            load,
            web_heights,
            write_geo_dat,
        )

        case_id = self._case_id(case)
        folder = self.data.case_folder(case_id)
        if folder is None:
            raise ValueError(f"{case_id}: no case folder")
        if crs == "model":
            georef = Georef("model")
        else:
            georef = self.georef(case_id)
            if georef is None:
                raise ValueError(
                    "the case has no georeference: set it first (set_georef), or give "
                    "crs='model' for a file in model coordinates"
                )
        domain = self.data.backgrounds.domain(case_id)
        if domain is None:
            raise ValueError("the domain is not known: set B 10 first (set_domain)")
        if spacing is None:
            spacing = self.input_file("control.txt", case_id).value("B 1") or 0.0
        xs, ys = geo_grid(domain[:4], float(spacing or 0.0))
        web = isinstance(source, str) and source in WEB and not Path(source).exists()
        if web:
            if crs == "model":
                raise ValueError("bathymetry from the web needs the case's georeference")
            heights, used = web_heights(source, georef, xs, ys, float(xs[1] - xs[0]))
            self.geo_sources = used
        else:
            bathy = load(Path(source).expanduser(), "model" if crs == "model" else crs, depths)
            if crs == "model":
                bathy.crs = "model"
                if bathy.raster is not None:
                    bathy.raster.crs = "model"
            heights = grid_heights(bathy, georef, xs, ys, method)
            self.geo_sources = [Path(source).name]
        count = write_geo_dat(folder / "geo.dat", xs, ys, heights)
        setup = self.data.case_setup(case_id, create=True)
        if setup is not None:
            setup.geo_changed()
        code = f"session.make_geo_dat({str(source)!r}"
        for key, value, default in (
            ("crs", crs, None),
            ("method", method, "tin"),
            ("spacing", spacing, None),
            ("depths", depths, False),
            ("case", case, None),
        ):
            if value != default:
                code += f", {key}={value!r}"
        self.record_output(code + ")")
        self.set_input("control.txt", "G 10", [1], case=case)
        self.scene.inputs_changed(case_id)
        return count

    def add_background_map(self, source: str = "Kartverket topo", **properties: Any) -> str:
        """A background map round a georeferenced case: map tiles (``source``, see
        ``kinds.BASEMAP_SOURCES``), or ``source="GeoTIFF file", file=...``."""
        return self.add_item("basemap", source=source, **properties)

    @property
    def mode(self) -> str:
        """ "results" or "setup" (see :meth:`set_mode`)."""
        return self.scene.mode

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

    def set_plan_view(self, on: bool, side: str | None = None) -> None:
        """2D: orthographic, no rotation. From the top (``side="top"``) every item is
        drawn flat (the free surface as a plane coloured by its values over the bed);
        from the "front", "back", "right" or "left" it is an elevation of the shapes.
        ``side=None`` keeps the side. View state: not an undo step."""
        self.scene.set_plan_view(on, side)

    # ------------------------------------------------------------- camera path
    def add_camera_key(self, time: float | None = None, state: dict[str, Any] | None = None) -> int:
        """A camera keyframe: the camera ``state`` (default: the saved view) at ``time``
        (default: now); a key at the same time is replaced. Turns the camera path on.
        Returns the number of keys. View state: not an undo step."""
        state = state if state is not None else self.scene.camera
        if not state:
            raise ValueError("no camera to key: show the 3D view first")
        when = self.scene.time if time is None else float(time)
        keys = [k for k in self.scene.camera_keys if abs(k["time"] - when) > 1e-9]
        keys.append({"time": when, "state": dict(state)})
        self.scene.set_camera_keys(keys, path=True)
        return len(self.scene.camera_keys)

    def remove_camera_key(self, time: float | None = None) -> bool:
        """Remove the keyframe at ``time`` (default: now); False if there is none."""
        when = self.scene.time if time is None else float(time)
        keys = [k for k in self.scene.camera_keys if abs(k["time"] - when) > 1e-6]
        if len(keys) == len(self.scene.camera_keys):
            return False
        self.scene.set_camera_keys(keys, path=self.scene.camera_path and bool(keys))
        return True

    def orbit_camera(
        self,
        degrees: float = 360.0,
        start: float | None = None,
        end: float | None = None,
        centre=None,
    ) -> int:
        """Camera keyframes that take the saved view around the scene: ``degrees``
        about the vertical axis through ``centre`` (default: the middle of the data
        shown now) from ``start`` to ``end`` (default: the timeline's first and last
        frame). Replaces the keyframes there were; turns the camera path on. Returns
        the number of keys."""
        from lagoonviz.core.camera_path import orbit_keys

        state = self.scene.camera
        if not state:
            raise ValueError("no camera to orbit with: show the 3D view first")
        if start is None or end is None:
            times = self.data.timeline()[0]
            if not len(times):
                raise ValueError("no timeline: open a case, or give start and end")
            start = float(times[0]) if start is None else start
            end = float(times[-1]) if end is None else end
        if centre is None:
            centre = self.data_centre()
            if centre is None:
                raise ValueError("nothing shown to orbit around: give a centre")
        keys = orbit_keys(state, centre, float(start), float(end), float(degrees))
        self.scene.set_camera_keys([k.to_dict() for k in keys], path=True)
        return len(keys)

    def data_centre(self):
        """The middle of the box of the visible data at the current time, or None."""
        import numpy as np

        lows, highs = [], []
        for item_id in self.data.data_items():
            if not self.scene.get(item_id).visible:
                continue
            mesh = self.data.load(item_id)
            if mesh is None:
                continue
            points = np.asarray(self.data.display(item_id, mesh).positions, float).reshape(-1, 3)
            points = points[np.isfinite(points).all(axis=1)]
            if len(points):
                lows.append(points.min(axis=0))
                highs.append(points.max(axis=0))
        if not lows:
            return None
        return ((np.min(lows, axis=0) + np.max(highs, axis=0)) / 2).tolist()

    def clear_camera_keys(self) -> None:
        self.scene.set_camera_keys([], path=False)

    def set_camera_path(self, on: bool) -> None:
        """Follow the camera path through the keyframes (views and films), or not."""
        self.scene.set_camera_keys(self.scene.camera_keys, path=bool(on))

    def camera_at(self, time: float | None = None) -> dict[str, Any] | None:
        """The camera on the path at a time (default: now); None when the camera path
        is off or there are no keys."""
        if not self.scene.camera_path or not self.scene.camera_keys:
            return None
        from lagoonviz.core.camera_path import CameraKey, camera_at

        keys = [CameraKey.from_dict(k) for k in self.scene.camera_keys]
        return camera_at(keys, self.scene.time if time is None else float(time))

    def set_sun(self, azimuth: float | None = None, elevation: float | None = None) -> None:
        """Put the sun at ``azimuth`` degrees (from +x towards +y: 0 shines from +x,
        90 from +y) and ``elevation`` degrees above the horizon; both None: where the
        look puts it. It lights the 3D view, films and path tracing alike. View state:
        not an undo step."""
        if azimuth is None and elevation is None:
            self.scene.set_sun(None)
            return
        current = self.sun_angles()
        self.scene.set_sun(
            (
                current[0] if azimuth is None else float(azimuth),
                current[1] if elevation is None else float(elevation),
            )
        )

    def set_play_range(self, start: float | None = None, end: float | None = None) -> None:
        """Play only the part of the timeline from ``start`` to ``end`` (times); both
        None: all of it. Playback, films and the export dialog keep to it. View state:
        not an undo step."""
        if start is None and end is None:
            self.scene.set_play_range(None)
            return
        times = self.data.timeline()[0]
        first = float(times[0]) if len(times) else 0.0
        last = float(times[-1]) if len(times) else first
        self.scene.set_play_range(
            (first if start is None else float(start), last if end is None else float(end))
        )

    def sun_angles(self, look: str | None = None) -> tuple[float, float]:
        """(azimuth, elevation) of the sun in degrees: the session's, else the look's."""
        import math

        if self.scene.sun is not None:
            return self.scene.sun
        x, y, z = sun_vector(look)
        return math.degrees(math.atan2(y, x)) % 360.0, math.degrees(math.asin(z))

    def sun_direction(self) -> tuple[float, float, float] | None:
        """The unit vector towards the session's sun, or None (the look's own)."""
        if self.scene.sun is None:
            return None
        return direction_of(*self.scene.sun)

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
        self._outputs.clear()
        self.path = path
        self._origin = path

    def reset(self) -> None:
        """Start an empty session."""
        self.scene.load_dict(Scene().to_dict())
        self.stack.clear()
        self._outputs.clear()
        self.path = None
        self._origin = None

    # ------------------------------------------------------------- scripts
    def record_output(self, code: str) -> None:
        """Note a figure, film or screenshot made now, as the Python that makes it
        again (``lagoonviz.render.script`` writes it). It goes into the recorded
        script after the change it followed; not an undo step."""
        done = self.stack.done
        self._outputs.append((done[-1] if done else None, code.rstrip("\n")))

    @property
    def outputs(self) -> list[str]:
        """The code of the outputs made in this session, in order."""
        return [code for _, code in self._outputs]

    def to_script(self) -> str:
        """The session's history as a runnable Python script. Figures, films and
        screenshots made are in it too, each where it was made; they are made again
        when the script is run (``python script.py``), not by ``from_script``."""
        lines = [
            "# LAGOON session script, recorded from the command history",
            "from lagoonviz.core import Session",
            "",
        ]
        if self._outputs:
            lines += [
                "# figures, films and screenshots: made again only when run as a script",
                'MAKE_OUTPUTS = __name__ == "__main__"',
                "",
            ]
        if self._origin is not None:
            lines.append(f"session = Session.open({str(self._origin)!r})")
        else:
            lines.append("session = Session()")
        done = self.stack.done
        after: dict[int, list[str]] = {}
        later: list[str] = []  # made after changes since undone
        for anchor, code in self._outputs:
            if anchor is None:
                after.setdefault(-1, []).append(code)
                continue
            index = next((i for i, command in enumerate(done) if command is anchor), None)
            if index is None:
                later.append(code)
            else:
                after.setdefault(index, []).append(code)
        lines += after.get(-1, [])
        for index, command in enumerate(done):
            lines.append(command.to_script())
            lines += after.get(index, [])
        if later:
            lines.append("# made with changes that were undone afterwards")
            lines += later
        # view state last, so the script shows the same frame and view
        if self.scene.time:
            lines.append(f"session.set_time({self.scene.time!r})")
        if self.scene.plan_view and self.scene.plan_side == "top":
            lines.append("session.set_plan_view(True)")
        elif self.scene.plan_view:
            lines.append(f"session.set_plan_view(True, side={self.scene.plan_side!r})")
        if self.scene.camera:
            lines.append(f"session.set_camera({self.scene.camera!r})")
        for key in self.scene.camera_keys:
            lines.append(f"session.add_camera_key({key['time']!r}, {key['state']!r})")
        if self.scene.camera_keys and not self.scene.camera_path:
            lines.append("session.set_camera_path(False)")
        if self.scene.sun is not None:
            azimuth, elevation = self.scene.sun
            lines.append(f"session.set_sun({azimuth!r}, {elevation!r})")
        if self.scene.play_range is not None:
            start, end = self.scene.play_range
            lines.append(f"session.set_play_range({start!r}, {end!r})")
        return "\n".join(lines) + "\n"

    @staticmethod
    def from_script(source: str) -> Session:
        """Run a recorded script and return the session it builds."""
        namespace: dict[str, Any] = {}
        exec(compile(source, "<lagoon script>", "exec"), namespace)
        return namespace["session"]


def drafts_folder() -> Path:
    """Where cases started on the map wait for their folder: ``LAGOON_DRAFTS``, else
    the user's application data (``~/Library/Application Support/LAGOON/Drafts`` on
    a Mac)."""
    import sys

    given = os.environ.get("LAGOON_DRAFTS")
    if given:
        return Path(given).expanduser()
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Application Support" / "LAGOON" / "Drafts"
    if sys.platform.startswith("win"):
        return Path(os.environ.get("APPDATA", Path.home())) / "LAGOON" / "Drafts"
    base = Path(os.environ.get("XDG_DATA_HOME", Path.home() / ".local" / "share"))
    return base / "lagoon" / "drafts"


def _new_draft() -> Path:
    """A new, empty draft folder: untitled-1, untitled-2, ..."""
    root = drafts_folder()
    root.mkdir(parents=True, exist_ok=True)
    n = 1
    while (root / f"untitled-{n}").exists():
        n += 1
    folder = root / f"untitled-{n}"
    folder.mkdir()
    return folder


def direction_of(azimuth: float, elevation: float) -> tuple[float, float, float]:
    """The unit vector towards a sun at ``azimuth`` (degrees from +x towards +y) and
    ``elevation`` (degrees above the horizon)."""
    import math

    a, e = math.radians(azimuth), math.radians(elevation)
    return (math.cos(e) * math.cos(a), math.cos(e) * math.sin(a), math.sin(e))


def sun_vector(look: str | None = None) -> tuple[float, float, float]:
    """The unit vector towards the sun of a look (its own, without the session's)."""
    import math

    from lagoonviz.core.looks import DEFAULT_SUN, LOOK_SUNS

    x, y, z = LOOK_SUNS.get(look or "", DEFAULT_SUN)
    n = math.sqrt(x * x + y * y + z * z)
    return x / n, y / n, z / n
