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
    Batch,
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
    WriteCaseFile,
)
from lagoonviz.core.data.manager import DataManager
from lagoonviz.core.data.reef3d import ROLES
from lagoonviz.core.data.setup import has_inputs, reads_geo_dat
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
        #: background map item -> (its case folder, its case item), to keep it there
        self._map_folders: dict = {}
        self._maps_removed: list = []
        self.scene.subscribe(self._remember_maps)
        #: where the last geo.dat came from (make_geo_dat): file name or web sources
        self.geo_sources: list[str] = []
        #: what the last sea_conditions could not read (NorKyst before 2017, ...)
        self.sea_notes: list[str] = []

    # ------------------------------------------------------------- log
    def subscribe_log(self, callback: LogListener) -> None:
        self._log_listeners.append(callback)

    def _log(self, line: str) -> None:
        for callback in list(self._log_listeners):
            callback(line)

    def _on_stack(self, kind: str, command: Command | None) -> None:
        self._maps_taken_away()
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
        """Add a scene item and return its id (a "River corridor": the case's
        corridor, laid out first if it has none, :meth:`add_corridor`)."""
        if kind == "corridor" and not self._replaying_corridor(properties):
            return self.add_corridor(case=properties.get("case") or None)
        if kind == "multiblock" and not self._replaying_multiblock(properties):
            return self.add_multiblock(case=properties.get("case") or None)
        command = AddItem(kind, name, properties)
        self.stack.execute(command)
        assert command.item_id is not None
        return command.item_id

    def _replaying_corridor(self, properties: dict) -> bool:
        """An item of a corridor there is already (added by itself, e.g. by a script
        of an older session): just the item."""
        try:
            case_id = self._case_id(properties.get("case") or None)
        except ValueError:
            return True
        return self.corridor(case_id) is not None and not self._corridor_items(case_id)

    def _replaying_multiblock(self, properties: dict) -> bool:
        """An item of blocks there are already (added by itself): just the item."""
        try:
            case_id = self._case_id(properties.get("case") or None)
        except ValueError:
            return True
        return self.multiblock(case_id) is not None and not self._multiblock_items(case_id)

    def open_case(self, path: str | Path, name: str | None = None) -> str:
        """Open a REEF3D case and show its free surface and bed (its volume is there,
        switched off; a CFD run's free surface is the level set phi at 0, an
        iso-surface of its volume), and its bodies. Returns the case id.

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
            if sum(d.role == role for d in case.datasets.values()) > 1 or role in (
                "solid",
                "topo",
                "forcing",
            ):
                props["dataset"] = dataset.key  # (solids: a structure; DIVEMesh's bed: a bed)
            variable = dataset.default_variable()
            if variable != KINDS[kind].properties["variable"].default:
                props["variable"] = variable
            if dataset.solver == "SEASTATE":  # Hs on Turbo, arrows where the waves go
                from lagoonviz.core.data import seastate

                if kind != "seastate" and variable == seastate.DEFAULT_VARIABLE:
                    props["colormap"] = seastate.DEFAULT_COLORMAP
                if kind == "seastate" and "wave_direction" in dataset.components:
                    props["vectors_of"] = "wave_direction"
            if kind == "amr_blocks":  # there, switched off: a tick shows them
                children.append(AddItem(kind, None, props, visible=False))
                continue
            if kind == "volume" and "phi" in dataset.components:
                # CFD: its free surface, the level set phi at 0; the volume switched off
                children.append(AddItem(kind, None, props, visible=False))
                free = {"iso_by": "phi", "iso_value": 0.0}
                if "dataset" in props:
                    free["dataset"] = props["dataset"]
                if "velocity" in dataset.components:
                    free["variable"] = "velocity"
                children.append(AddItem("iso_surface", "Free surface", free))
                continue
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
        if (Path(case.path) / "corridor.json").is_file():  # its river corridor
            children.append(AddItem("corridor", None, {}, visible=case.setup_only))
        if (Path(case.path) / "blocks.json").is_file():  # its multi-block grid
            children.append(AddItem("multiblock", None, {}, visible=case.setup_only))
        if case.by_role("geodat") is not None and reads_geo_dat(case.path):
            # geo.dat's points, in the set-up (not when DIVEMesh leaves geo.dat out:
            # then it is another case's, left in the folder)
            children.append(AddItem("geodat", None, {}, visible=case.setup_only))
        for series in case.main_series():
            children.append(AddItem("series", series.name, {"source": series.key}))
        for profiles in case.main_profiles():
            children.append(AddItem("profile", profiles.name, {"source": profiles.key}))
        children += self._background_for(case)
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
        elif not case.setup_only and self.scene.mode == "setup":
            # results opened while working on a set-up (e.g. the case set up in LAGOON,
            # then run): the set-up leaves results out, so show them
            self.set_mode("results")
        # a run restarted in the same folder may start later: begin where it has data
        first = self.data.first_time(command.item_id)
        if first is not None and self.scene.time < first:
            self.set_time(first)
        return command.item_id

    def _background_for(self, case) -> list[AddItem]:
        """The background map a georeferenced case opens with: as its folder keeps
        it (LAGOON_map.json, from the last time it was set), else the place's best
        map; none for a case without a georeference, or whose map was taken away."""
        from lagoon_format.geo import FILE

        from lagoonviz.core.data.background import Backgrounds, map_for

        folder = Path(case.path)
        folder = folder.parent if folder.is_file() or folder.suffix == ".lagoon" else folder
        try:
            georef, _ = Backgrounds._find_georef(folder, case, (folder / FILE).is_file())
        except Exception:  # noqa: BLE001 (a georeference that does not read: no map)
            return []
        found = map_for(folder, georef, own=reads_geo_dat(folder))
        if found is None:
            return []
        visible = bool(found.pop("visible", True))
        return [AddItem("basemap", None, found, visible=visible)]

    def _remember_maps(self, event: str, item_id: str | None) -> None:
        """A background map set, shown or hidden, or taken away: kept in its case's
        folder for the next time the case opens (LAGOON_map.json)."""
        from lagoonviz.core.data.background import remember_map

        folders = self._map_folders
        if event in ("added", "changed") and item_id in self.scene:
            item = self.scene.get(item_id)
            if item.kind != "basemap":
                return
            folder = self.data.case_folder(item_id)
            if folder is None or not folder.is_dir():
                return
            folders[item_id] = (folder, self.data.case_item(item_id))
            remember_map(folder, item.properties, item.visible)
        elif event == "removed" and item_id in folders:
            # decided once the command is done: the map taken away, or its case closed
            self._maps_removed.append(folders.pop(item_id))

    def _maps_taken_away(self) -> None:
        """After a command: a background map removed while its case stays open was
        taken away (kept so, the case opens without it); with its case closed (or
        the case's opening undone) nothing changes."""
        from lagoonviz.core.data.background import remember_map

        removed, self._maps_removed = self._maps_removed, []
        for folder, case_item in removed:
            if case_item is None or case_item.id not in self.scene:
                continue
            others = [
                i
                for i in self.scene
                if i.kind == "basemap" and self.data.case_folder(i.id) == folder
            ]
            if not others:
                remember_map(folder, None)

    def new_case(self, path: str | Path, name: str | None = None) -> str:
        """Start a case in a folder (made if needed): its control.txt and ctrl.txt are
        started with a comment line if they are not there, and the case opens in the
        set-up, ready for the commands. Returns the case id."""
        folder = Path(path).expanduser()
        if folder.suffix == ".lagoon" or (folder / "zarr.json").is_file():
            raise ValueError(
                f"{folder.name} is a LAGOON store (a run's output), not a case folder: "
                f"open it with File → Open case, or start the case in {folder.parent}"
            )
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
        georeference, in its CRS (the UTM zone of the place when it has none yet, or
        when the place lies outside the area its CRS is meant for: another UTM zone,
        ETRS89 outside Europe), the rotation kept; ``crs``: another CRS in metres to
        use. Gives the case a background map if it has none. Returns the place."""
        from lagoonviz.core.geo.crs import fits, is_geographic, transformer, utm_for
        from lagoonviz.core.geo.places import Place, search

        if isinstance(place, str):
            found = search(place, near=self.location(case))
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
            if crs is not None and not fits(crs, place.lon, place.lat):
                crs = None  # e.g. Norway's UTM zone for Mumbai: the place's own
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

    def location(self, case: str | None = None) -> tuple[float, float] | None:
        """Latitude and longitude of the centre of a case's domain (the open case's;
        None without a georeference or a case). A place search puts the places close
        to it first."""
        from lagoonviz.core.geo.crs import transformer

        try:
            case_id = self._case_id(case)
        except (KeyError, ValueError):
            return None
        georef = self.georef(case_id)
        if georef is None:
            return None
        domain = self.data.backgrounds.domain(case_id)
        cx, cy = ((domain[0] + domain[1]) / 2, (domain[2] + domain[3]) / 2) if domain else (0, 0)
        try:
            lon, lat = transformer(georef.crs, "EPSG:4326")(*georef.to_world(cx, cy))
        except (ValueError, OSError):
            return None
        return float(lat), float(lon)

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

    def create_case(
        self, folder: str | Path, case: str | None = None, overwrite: bool = False
    ) -> Path:
        """Put a case (a draft started on the map) in its folder: its input files
        (edits saved first), georef.json, geo.dat and what DIVEMesh made are copied
        there, and the case is that folder from now on (named after it). The folder
        is made if needed; files of the same names already in it are kept unless
        ``overwrite`` (then an error says which). Returns the folder."""
        import shutil

        case_id = self._case_id(case)
        source = self.data.case_folder(case_id)
        if source is None:
            raise ValueError(f"{case_id}: no case folder")
        target = Path(folder).expanduser()
        if target.resolve() == source.resolve():
            return target
        clashes = self.case_file_clashes(target, case_id)
        if clashes and not overwrite:
            raise FileExistsError(
                f"{target} already has {', '.join(clashes)}: choose another folder, or "
                "overwrite them"
            )
        self.save_inputs(case_id)
        shutil.copytree(source, target, dirs_exist_ok=True)
        self.set_property(case_id, "path", str(target))
        self.rename(case_id, target.name)
        self.scene.inputs_changed(case_id)
        return target

    def case_file_clashes(self, folder: str | Path, case: str | None = None) -> list[str]:
        """The files and folders of a case that a folder already has (what
        :meth:`create_case` would overwrite there)."""
        source = self.data.case_folder(self._case_id(case))
        target = Path(folder).expanduser()
        if source is None or not target.is_dir():
            return []
        return sorted(p.name for p in source.iterdir() if (target / p.name).exists())

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
        """A "Case set-up" item for a case with input files, a "DIVEMesh grid" item
        for one with control.txt and a "geo.dat points" item for one with geo.dat, if
        it has none yet."""
        cases = [i for i in self.scene if i.kind == "case"]
        found = []
        for kind in ("setup", "grid", "geodat", "corridor", "multiblock"):
            named = {str(i.properties.get("case", "")) for i in self.scene if i.kind == kind}
            if case_id in named or (len(cases) == 1 and "" in named):
                continue
            if kind == "geodat":
                case = self.data.case(case_id)
                if (
                    case is not None
                    and case.by_role("geodat") is not None
                    and reads_geo_dat(case.path, self.input_file("control.txt", case_id))
                ):
                    found.append(AddItem(kind, None, {"case": case_id}))
                continue
            setup = self.data.case_setup(case_id)
            if setup is None or (kind == "grid" and not setup.control.exists):
                continue
            if kind == "corridor" and setup.corridor is None:
                continue
            if kind == "multiblock" and setup.multiblock is None:
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
        command = self._input_edit(file, case, change, script, label)
        if command is not None:
            self.stack.execute(command)

    def _input_edit(self, file: str, case: str | None, change, script: str, label: str):
        """The command that changes an input file (None: no change)."""
        case_id = self._case_id(case)
        current = self.input_file(file, case_id)
        changed = current.copy()
        change(changed)
        if changed.text() == current.text():
            return None
        call = script + (f", case={case!r})" if case else ")")
        return EditInput(
            lambda: self.input_file(file, case_id), case_id, changed.text(), call, label
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

    def add_structure(self, code: str, values=None, case: str | None = None) -> int:
        """A porous structure or a solid, one undo step: ``code`` one of
        :data:`lagoonviz.core.structures.KINDS` - porous (ctrl.txt): B 270 box, B 274
        cylinder, B 281/B 282 wedges, B 291 plate for CFD, B 210 - B 218 for NHFLOW;
        solids: DIVEMesh's S 10 box, S 32/S 33 cylinders, S 51 sphere, S 61/S 62
        wedges (control.txt), NHFLOW's direct forcing A 581 - A 588 (ctrl.txt).
        ``values``: its line (default: :meth:`structure_default`). NHFLOW's porous
        material B 201 is added too when the case has none. Returns the structure's
        occurrence."""
        from lagoonviz.core import structures

        kind = structures.BY_CODE.get(code)
        if kind is None:
            raise ValueError(f"{code!r} is not a structure ({', '.join(structures.BY_CODE)})")
        case_id = self._case_id(case)
        if values is None:
            values = self.structure_default(code, case_id)
        values = [float(v) for v in values]
        if len(values) != kind.count:
            raise ValueError(f"{code} takes {kind.count} values, got {len(values)}")

        def change(f):
            material = structures.NHFLOW_MATERIAL
            if kind.material == material and material not in f:
                f.add(material, list(structures.DEFAULT_MATERIAL))
            f.add(code, values)

        self._edit_input(
            kind.file,
            case,
            change,
            f"session.add_structure({code!r}, {values!r}",
            f"Add {kind.phrase()}",
        )
        return len(self.input_file(kind.file, case_id).entries(code)) - 1

    def structure_default(self, code: str, case: str | None = None, area=None) -> list[float]:
        """The line of a structure ``code`` over a plan area (x0, x1, y0, y1; default
        the middle of the domain), from the bed (the domain's bottom) up: a porous one
        to the still water level (F 60), of rubble; a solid to the domain's top (for
        NHFLOW's σ-grid, half the water depth above the still water)."""
        from lagoonviz.core import structures
        from lagoonviz.core.data.setup import SIGMA_SOLVERS

        kind = structures.BY_CODE[code]
        setup = self.data.case_setup(self._case_id(case), create=True)
        domain = setup.domain if setup is not None else None
        if domain is None:
            domain = (0.0, 10.0, 0.0, 1.0, 0.0, 1.0)
        bottom, top = domain[4], domain[5]
        water = setup.water_level if setup is not None else None
        if water is not None and water > bottom:
            if kind.group == "porous":
                top = water
            elif setup.solver in SIGMA_SOLVERS:
                top = round(water + 0.5 * (water - bottom), 6)
        two_d = bool(setup is not None and setup.two_d)
        area = area if area is not None else structures.default_area(domain, two_d)
        return structures.from_area(kind, area, (bottom, top))

    # ------------------------------------------------------------- river corridor
    def corridor(self, case: str | None = None):
        """A case's river corridor (:class:`lagoonviz.core.corridor.Corridor`, from its
        corridor.json) or None."""
        setup = self.data.case_setup(self._case_id(case))
        return setup.corridor if setup is not None else None

    def _case_file(self, case_id: str, name: str, text: str | None, script: str, label: str):
        folder = self.data.case_folder(case_id)
        if folder is None:
            raise ValueError(f"{case_id}: no case folder")
        setup = self.data.case_setup(case_id, create=True)

        def changed():
            if setup is not None:
                setup.files_changed()

        return WriteCaseFile(folder / name, case_id, text, script, label, changed)

    def set_corridor(self, corridor, case: str | None = None) -> None:
        """Give a case a river corridor (a Corridor or its dict as in corridor.json;
        None removes it). Written to corridor.json at once; one undo step."""
        from lagoonviz.core import corridor as corridors

        if isinstance(corridor, dict):
            corridor = corridors.Corridor.from_dict(corridor)
        data = corridor.to_dict() if corridor is not None else None
        self._put_corridor(case, corridor, f"session.set_corridor({data!r}", "Set river corridor")

    def _put_corridor(self, case: str | None, corridor, script: str, label: str) -> None:
        from lagoonviz.core import corridor as corridors

        case_id = self._case_id(case)
        text = corridor.text() if corridor is not None else None
        current = self.corridor(case_id)
        if (current.text() if current is not None else None) == text:
            return
        if case:
            script += ("" if script.endswith("(") else ", ") + f"case={case!r}"
        script += ")"
        commands = [self._case_file(case_id, corridors.FILE, text, script, label)]
        items = self._corridor_items(case_id)
        if corridor is not None and not items:  # the corridor is an item of the scene
            commands.append(AddItem("corridor", None, {"case": case_id}))
        elif corridor is None:
            commands += [RemoveItem(i) for i in items]
        self.stack.execute(commands[0] if len(commands) == 1 else Batch(commands, script, label))

    def _corridor_items(self, case_id: str) -> list[str]:
        """The "River corridor" items of a case."""
        found = []
        for item in self.scene:
            if item.kind != "corridor":
                continue
            owner = self.data.case_item(item.id)
            if owner is not None and owner.id == case_id:
                found.append(item.id)
        return found

    def add_corridor(self, case: str | None = None) -> str:
        """A "River corridor" item: the case's corridor, laid out first if it has
        none (along the thalweg with geo.dat, else straight along the domain).
        Returns the item's id."""
        case_id = self._case_id(case)
        if self.corridor(case_id) is None:
            setup = self.data.case_setup(case_id, create=True)
            if setup is not None and (setup.path / "geo.dat").is_file():
                self.auto_corridor(margin=5.0, case=case)
            else:
                self.start_corridor(case=case)
        items = self._corridor_items(case_id)
        if items:
            return items[0]
        return self.add_item("corridor", case=case_id)

    def reset_corridor(self, case: str | None = None) -> None:
        """Lay the corridor out again the way it was made (along the domain, over the
        area drawn, or along the thalweg with its margin): the dragged points are
        dropped, the grid's settings (cells, smoothing, curve, bed) kept."""
        from dataclasses import replace

        case_id = self._case_id(case)
        current = self._corridor_of(case_id)
        made = dict(current.extra.get("made") or {"how": "domain", "points": len(current.left)})
        how = made.get("how", "domain")
        if how == "thalweg":
            fresh = self._thalweg_corridor(
                case_id, **{k: v for k, v in made.items() if k != "how"}
            )[0]
        elif how == "line":
            fresh = self._traced_corridor(case_id, made["line"], made["width"], made["points"])
        elif how == "fitted":
            fresh = self._refitted_corridor(case_id, current, made)
        else:
            area = made.get("area") if how == "area" else None
            fresh = self._straight_corridor(case_id, area, int(made.get("points", 12)))
        keep = ("curve", "cells_along", "cells_across", "smoothing", "bed", "z_start", "slope")
        fresh = replace(fresh, **{k: getattr(current, k) for k in keep})
        fresh.extra = dict(current.extra)
        script = "session.reset_corridor("
        self._put_corridor(case, fresh, script, "Reset river corridor")

    def _refitted_corridor(self, case_id: str, current, made: dict):
        """A fitted corridor made again: the corridor it was fitted from (drawn, over
        an area, along the domain) laid out again and fitted to the bed."""
        from dataclasses import replace

        from lagoonviz.core import corridor as corridors

        source = dict(made.get("from") or {})
        how = source.get("how", "domain")
        if how == "line":
            start = self._traced_corridor(
                case_id, source["line"], source["width"], source["points"]
            )
        elif how == "thalweg":
            start = self._thalweg_corridor(
                case_id, **{k: v for k, v in source.items() if k != "how"}
            )[0]
        else:
            area = source.get("area") if how == "area" else None
            start = self._straight_corridor(case_id, area, int(source.get("points", 12)))
        keep = ("curve", "cells_along", "cells_across", "smoothing")
        start = replace(start, **{k: getattr(current, k) for k in keep})
        setup = self.data.case_setup(case_id, create=True)
        if setup is None or not (setup.path / "geo.dat").is_file():
            raise ValueError("fitting the river needs the bed: geo.dat (Map tab: Make geo.dat)")
        level = made.get("level", setup.water_level)
        fresh, _ = corridors.fit_to_bed(
            start, setup.geo_points(), level, float(made.get("margin", 0.0))
        )
        return fresh

    def _straight_corridor(self, case_id: str, area, points: int):
        from lagoonviz.core import corridor as corridors

        setup = self.data.case_setup(case_id, create=True)
        domain = setup.domain if setup is not None else None
        how = "area" if area is not None else "domain"
        if area is None:
            if domain is None:
                raise ValueError("no domain yet: give the area of the corridor")
            xs, xe, ys, ye = domain[:4]
            if xe - xs >= ye - ys:
                area = (xs, xe, ys + 0.1 * (ye - ys), ye - 0.1 * (ye - ys))
            else:
                area = (xs + 0.1 * (xe - xs), xe - 0.1 * (xe - xs), ys, ye)
        area = [round(float(v), 6) for v in area]
        options = {"z_start": float(domain[4])} if domain is not None else {}
        made = corridors.default(area, int(points), **options)
        made.extra["made"] = {"how": how, "points": int(points)} | (
            {"area": area} if how == "area" else {}
        )
        return made

    def _corridor_of(self, case_id: str):
        current = self.corridor(case_id)
        if current is None:
            raise ValueError(f"{case_id}: no river corridor yet (start_corridor)")
        return current

    def start_corridor(self, area=None, points: int = 12, case: str | None = None) -> None:
        """A straight river corridor over a plan area (x0, x1, y0, y1; default the
        domain without a tenth on each side across), flowing along its longer side,
        ``points`` points on each bank."""
        case_id = self._case_id(case)
        start = self._straight_corridor(case_id, area, int(points))
        given = start.extra["made"].get("area")
        script = "session.start_corridor(" + (f"{given!r}, " if given else "")
        script += f"points={int(points)}"
        self._put_corridor(case, start, script, "Start river corridor")

    def auto_corridor(
        self,
        margin: float = 0.0,
        level: float | None = None,
        points: int = 12,
        width: float | None = None,
        sides: tuple[int, int] | None = None,
        case: str | None = None,
    ):
        """A corridor that follows the thalweg of the bed (geo.dat): from the deepest
        point of the upstream side to the deepest of the downstream side (``sides``,
        default: the inflow and outflow sides, C 1x = 1 and 2, else the ends of the
        domain's longer axis), its banks where the water below ``level`` (default: F
        60, else halfway up the banks) reaches, plus ``margin`` m (or ``width`` m
        wide), ``points`` points on each bank. Returns the thalweg found."""
        case_id = self._case_id(case)
        made, thalweg = self._thalweg_corridor(case_id, margin, level, points, width, sides)
        words = [f"margin={float(margin)!r}"]
        if level is not None:
            words.append(f"level={float(level)!r}")
        words.append(f"points={int(points)}")
        if width is not None:
            words.append(f"width={float(width)!r}")
        if sides is not None:
            words.append(f"sides={tuple(int(v) for v in sides)!r}")
        script = "session.auto_corridor(" + ", ".join(words)
        self._put_corridor(case, made, script, "Corridor along the thalweg")
        return thalweg

    def _thalweg_corridor(
        self, case_id: str, margin=0.0, level=None, points=12, width=None, sides=None
    ):
        from lagoonviz.core import corridor as corridors

        setup = self.data.case_setup(case_id, create=True)
        if setup is None or not (setup.path / "geo.dat").is_file():
            raise ValueError("following the thalweg needs the bed: geo.dat (Map tab: Make geo.dat)")
        domain = setup.domain
        if domain is None:
            raise ValueError("no domain yet (B 10)")
        recipe = {"how": "thalweg", "margin": float(margin), "points": int(points)}
        if level is not None:
            recipe["level"] = float(level)
        if width is not None:
            recipe["width"] = float(width)
        if sides is not None:
            recipe["sides"] = [int(v) for v in sides]
        if sides is None:
            inflow = [k for k, v in setup.sides.items() if v == 1 and k <= 4]
            outflow = [k for k, v in setup.sides.items() if v == 2 and k <= 4]
            if inflow and outflow:
                sides = (inflow[0], outflow[0])
            else:
                sides = (1, 4) if domain[1] - domain[0] >= domain[3] - domain[2] else (3, 2)
        if level is None:
            level = setup.water_level
        thalweg = corridors.find_thalweg(
            setup.geo_points(), domain, level, int(sides[0]), int(sides[1])
        )
        made = corridors.from_thalweg(thalweg, float(margin), int(points), width)
        made.extra["made"] = recipe
        return made, thalweg

    # ------------------------------------------------------------- a river drawn first
    def river_line(self, case: str | None = None) -> list[tuple[float, float]] | None:
        """The river drawn on the map (river.json): its points, upstream first; else
        the line the case's corridor was made along; None if neither."""
        case_id = self._case_id(case)
        setup = self.data.case_setup(case_id, create=True)
        line = setup.river_line if setup is not None else None
        if line is not None:
            return [(float(x), float(y)) for x, y in line]
        current = self.corridor(case_id)
        made = dict(current.extra.get("made") or {}) if current is not None else {}
        if made.get("how") == "fitted":
            made = dict(made.get("from") or {})
        if made.get("how") != "line":
            return None
        return [(float(x), float(y)) for x, y in made["line"]]

    def draw_river(self, line, case: str | None = None) -> None:
        """The river drawn on the map: a line ((x, y) points, upstream first; None
        removes it), kept in river.json, one undo step. Then :meth:`corridor_domain`
        sets the domain around it (for geo.dat) and :meth:`create_river` makes the
        corridor and its grid along it."""
        import numpy as np

        from lagoonviz.core import corridor as corridors

        case_id = self._case_id(case)
        text = None
        if line is not None:
            points = np.asarray(line, float).reshape(-1, 2)
            keep = np.concatenate([[True], np.hypot(*np.diff(points, axis=0).T) > 1e-9])
            points = points[keep]
            if len(points) < 2:
                raise ValueError("a river needs two points at least")
            text = corridors.river_text(points)
            line = [[round(float(x), 4), round(float(y), 4)] for x, y in points]
        script = f"session.draw_river({line!r}"
        if case:
            script += f", case={case!r}"
        script += ")"
        label = "Draw the river" if line is not None else "Remove the river drawn"
        self.stack.execute(self._case_file(case_id, corridors.RIVER_FILE, text, script, label))

    def create_river(
        self,
        width: float,
        points: int = 12,
        margin: float = 0.0,
        level: float | None = None,
        smoothing: int = 20,
        case: str | None = None,
    ):
        """The river corridor and its grid along the river drawn (:meth:`draw_river`):
        ``width`` m wide, ``points`` points on each bank; with geo.dat fitted to the
        river in the bed (its thalweg from the upstream end to the downstream end,
        its banks where the water below ``level`` reaches plus ``margin`` m, at most
        half as far again as ``width``). ``smoothing``: Winslow sweeps of the grid.
        One undo step. Returns the thalweg found (None without geo.dat)."""
        from lagoonviz.core import corridor as corridors

        case_id = self._case_id(case)
        line = self.river_line(case_id)
        if line is None:
            raise ValueError("draw the river on the map first (Draw the river)")
        made = self._traced_corridor(case_id, line, width, points, smoothing=int(smoothing))
        setup = self.data.case_setup(case_id, create=True)
        thalweg = None
        if setup is not None and (setup.path / "geo.dat").is_file():
            used = setup.water_level if level is None else level
            drawn = made.extra["made"]
            made, thalweg = corridors.fit_to_bed(made, setup.geo_points(), used, float(margin))
            recipe = {"how": "fitted", "margin": float(margin), "points": int(points)}
            if level is not None:
                recipe["level"] = float(level)
            recipe["from"] = drawn
            made.extra["made"] = recipe
        words = [f"{float(width)!r}", f"points={int(points)}"]
        if float(margin):
            words.append(f"margin={float(margin)!r}")
        if level is not None:
            words.append(f"level={float(level)!r}")
        if int(smoothing) != 20:
            words.append(f"smoothing={int(smoothing)}")
        self._put_corridor(case, made, "session.create_river(" + ", ".join(words), "Create river")
        return thalweg

    def _traced_corridor(self, case_id: str, line, width, points, **options):
        import numpy as np

        from lagoonviz.core import corridor as corridors

        if not float(width) > 0:
            raise ValueError(f"the river's width must be above 0, got {width}")
        line = [[round(float(x), 4), round(float(y), 4)] for x, y in np.asarray(line, float)]
        setup = self.data.case_setup(case_id, create=True)
        domain = setup.domain if setup is not None else None
        if domain is not None:
            options.setdefault("z_start", float(domain[4]))
        made = corridors.from_line(line, float(width), int(points), **options)
        made.extra["made"] = {
            "how": "line",
            "line": line,
            "width": float(width),
            "points": int(points),
        }
        return made

    def corridor_domain(
        self, margin: float | None = None, width: float | None = None, case: str | None = None
    ):
        """The domain (B 10's x and y) around the river drawn (``width`` m wide;
        default: the corridor's width) or else around the corridor, ``margin`` m
        beyond it on every side (default: half the river's width), its edges on
        DIVEMesh's cells (B 1): where geo.dat is made for the river. Returns (x0, x1,
        y0, y1)."""
        import numpy as np

        case_id = self._case_id(case)
        setup = self.data.case_setup(case_id, create=True)
        if setup is not None and setup.ctrl.int("G 50", 0) in (1, 2):
            raise ValueError(
                "B 10 holds the logical box of the case's curvilinear grid (G 50 in "
                "ctrl.txt): set the domain on the map before making the grid"
            )
        current = self.corridor(case_id)
        drawn = setup.river_line if setup is not None else None
        if width is None and current is not None:
            width = float(np.mean(np.hypot(*(current.left - current.right).T)))
        if drawn is not None:
            if width is None:
                width = 0.1 * float(np.hypot(*np.diff(drawn, axis=0).T).sum())
            from lagoonviz.core import corridor as corridors

            dense = corridors.resample(drawn, max(20 * len(drawn), 100))
            low = dense.min(axis=0) - 0.5 * float(width)
            high = dense.max(axis=0) + 0.5 * float(width)
        elif current is not None:
            banks = np.vstack([current.bank_line("left"), current.bank_line("right")])
            low, high = banks.min(axis=0), banks.max(axis=0)
        else:
            raise ValueError("draw the river on the map first (Draw the river)")
        if margin is None:
            margin = 0.5 * float(width)
        step = float(self.input_file("control.txt", case_id).value("B 1") or 0.0)
        low = low - float(margin)
        high = high + float(margin)
        if step > 0:
            low = np.floor(low / step) * step
            high = np.ceil(high / step) * step
        x0, y0 = (float(v) for v in low)
        x1, y1 = (float(v) for v in high)
        self.set_domain(x0, x1, y0, y1, case=case)
        return x0, x1, y0, y1

    def fit_corridor(
        self,
        margin: float = 0.0,
        level: float | None = None,
        case: str | None = None,
    ):
        """The corridor (drawn by eye, or dragged) made to follow the river in the
        bed (geo.dat): along its thalweg from the upstream end of the corridor to the
        downstream end, its banks where the water below ``level`` (default: F 60,
        else halfway up the banks) reaches, plus ``margin`` m; its points, curve and
        grid kept. Returns the thalweg found."""
        from lagoonviz.core import corridor as corridors

        case_id = self._case_id(case)
        current = self._corridor_of(case_id)
        setup = self.data.case_setup(case_id, create=True)
        if setup is None or not (setup.path / "geo.dat").is_file():
            raise ValueError("fitting the river needs the bed: geo.dat (Map tab: Make geo.dat)")
        used = setup.water_level if level is None else level
        made, thalweg = corridors.fit_to_bed(current, setup.geo_points(), used, float(margin))
        before = dict(current.extra.get("made") or {})
        recipe = {"how": "fitted", "margin": float(margin), "points": len(made.left)}
        if level is not None:
            recipe["level"] = float(level)
        recipe["from"] = before.get("from", before) if before.get("how") == "fitted" else before
        made.extra = dict(current.extra)
        made.extra["made"] = recipe
        words = [f"margin={float(margin)!r}"]
        if level is not None:
            words.append(f"level={float(level)!r}")
        self._put_corridor(
            case, made, "session.fit_corridor(" + ", ".join(words), "Fit the river to the bed"
        )
        return thalweg

    def move_corridor_point(
        self, side: str, index: int, x: float, y: float, case: str | None = None
    ) -> None:
        """Move a point of a bank ("left" or "right" looking downstream; index 0
        upstream) to (x, y): the bank follows it."""
        moved = self._corridor_of(self._case_id(case)).moved(side, int(index), x, y)
        script = f"session.move_corridor_point({side!r}, {int(index)}, {float(x)!r}, {float(y)!r}"
        self._put_corridor(case, moved, script, "Move corridor point")

    def set_corridor_options(
        self, points: int | None = None, case: str | None = None, **options
    ) -> None:
        """Change the corridor: ``points`` per bank (the banks resampled), ``curve``
        ("through", "bspline"), ``cells_along``, ``cells_across``, ``smoothing``
        (Winslow iterations), ``bed`` ("slope", "geo"), ``z_start``, ``slope``."""
        from dataclasses import replace

        current = self._corridor_of(self._case_id(case))
        changed = replace(current, **options) if options else current
        if points is not None and int(points) != len(current.left):
            changed = changed.with_points(int(points))
        words = ([f"points={int(points)}"] if points is not None else []) + [
            f"{k}={v!r}" for k, v in options.items()
        ]
        script = "session.set_corridor_options(" + ", ".join(words)
        self._put_corridor(case, changed, script, "Change river corridor")

    def reverse_corridor(self, case: str | None = None) -> None:
        """The river flowing the other way (upstream and downstream swapped)."""
        reversed_ = self._corridor_of(self._case_id(case)).reversed()
        self._put_corridor(case, reversed_, "session.reverse_corridor(", "Reverse river corridor")

    def remove_corridor(self, case: str | None = None) -> None:
        """Remove a case's corridor.json (curvgrid.dat and the input files stay)."""
        self._put_corridor(case, None, "session.remove_corridor(", "Remove river corridor")

    def corridor_grid(self, case: str | None = None):
        """The grid the corridor makes: (x, y, zb, quality)."""
        from lagoonviz.core import corridor as corridors

        case_id = self._case_id(case)
        current = self._corridor_of(case_id)
        x, y = current.grid()
        setup = self.data.case_setup(case_id)
        points = None
        if current.bed == "geo" and setup is not None and (setup.path / "geo.dat").is_file():
            points = setup.geo_points(limit=10**9)
        z = current.bed_heights(x, y, points)
        return x, y, z, corridors.quality(x, y)

    def make_corridor_grid(self, case: str | None = None):
        """Write the corridor's grid as curvgrid.dat and set the case up for it, one
        undo step: G 50 2 in ctrl.txt, and DIVEMesh's grid the logical box of ni x nj
        cells (B 10's x and y, B 1; its heights kept). Refused if cells fold. Returns
        the grid's :class:`~lagoonviz.core.corridor.Quality`."""
        from lagoonviz.core import corridor as corridors

        case_id = self._case_id(case)
        x, y, z, quality = self.corridor_grid(case_id)
        if quality.folded:
            raise ValueError(
                f"{quality.folded} cells of the corridor's grid fold: move the points, "
                "use fewer cells along or more smoothing"
            )
        ni, nj = quality.cells
        mean = 0.5 * (sum(quality.along) / 2 + sum(quality.across) / 2)
        step = corridors.nice_step(mean)
        x0, x1, y0, y1 = corridors.logical_box(ni, nj, step)
        control = self.input_file("control.txt", case_id)
        old = control.values("B 10") or [0.0, 1.0, 0.0, 1.0, 0.0, 1.0]
        box = [x0, x1, y0, y1, *(old + [0.0, 1.0])[4:6]]

        def logical(f):
            f.set("B 10", box)
            f.set("B 1", [step])
            if "B 2" in f:
                f.remove("B 2")

        commands = [
            self._case_file(
                case_id, corridors.GRID_FILE, corridors.grid_text(x, y, z), "", "curvgrid.dat"
            )
        ]
        for file, change in (
            ("ctrl.txt", lambda f: f.set("G 50", [2], kinds="int")),
            ("control.txt", logical),
        ):
            command = self._input_edit(file, case_id, change, "", file)
            if command is not None:
                commands.append(command)
        script = "session.make_corridor_grid(" + (f"case={case!r})" if case else ")")
        self.stack.execute(Batch(commands, script, "Make corridor grid"))
        return quality

    # ------------------------------------------------------------- multi-block grid
    def multiblock(self, case: str | None = None):
        """A case's multi-block grid (:class:`lagoonviz.core.multiblock.Topology`, from
        its blocks.json) or None."""
        setup = self.data.case_setup(self._case_id(case))
        return setup.multiblock if setup is not None else None

    def _multiblock_of(self, case_id: str):
        current = self.multiblock(case_id)
        if current is None:
            raise ValueError(f"{case_id}: no multi-block grid yet (start_multiblock)")
        return current

    def _multiblock_items(self, case_id: str) -> list[str]:
        """The "Multi-block grid" items of a case."""
        found = []
        for item in self.scene:
            if item.kind != "multiblock":
                continue
            owner = self.data.case_item(item.id)
            if owner is not None and owner.id == case_id:
                found.append(item.id)
        return found

    def _put_multiblock(self, case: str | None, topology, script: str, label: str) -> None:
        from lagoonviz.core import multiblock as mb

        case_id = self._case_id(case)
        text = topology.text() if topology is not None else None
        current = self.multiblock(case_id)
        if (current.text() if current is not None else None) == text:
            return
        if case:
            script += ("" if script.endswith("(") else ", ") + f"case={case!r}"
        script += ")"
        commands = [self._case_file(case_id, mb.FILE, text, script, label)]
        items = self._multiblock_items(case_id)
        if topology is not None and not items:  # the grid is an item of the scene
            commands.append(AddItem("multiblock", None, {"case": case_id}))
        elif topology is None:
            commands += [RemoveItem(i) for i in items]
        self.stack.execute(commands[0] if len(commands) == 1 else Batch(commands, script, label))

    def _change_multiblock(self, case, change, script: str, label: str) -> None:
        current = self._multiblock_of(self._case_id(case))
        self._put_multiblock(case, change(current), script, label)

    def set_multiblock(self, topology, case: str | None = None) -> None:
        """Give a case a multi-block grid (a Topology or its dict as in blocks.json;
        None removes it). Written to blocks.json at once; one undo step."""
        from lagoonviz.core import multiblock as mb

        if isinstance(topology, dict):
            topology = mb.Topology.from_dict(topology)
        data = topology.to_dict() if topology is not None else None
        self._put_multiblock(case, topology, f"session.set_multiblock({data!r}", "Set blocks")

    def add_multiblock(self, case: str | None = None) -> str:
        """A "Multi-block grid" item: the case's blocks, started first if it has none
        (from its river corridor, else one block over the domain). Returns its id."""
        case_id = self._case_id(case)
        if self.multiblock(case_id) is None:
            if self.corridor(case_id) is not None:
                self.multiblock_from_corridor(case=case)
            else:
                self.start_multiblock(case=case)
        items = self._multiblock_items(case_id)
        if items:
            return items[0]
        return self.add_item("multiblock", case=case_id)

    def start_multiblock(self, area=None, cells=None, case: str | None = None) -> None:
        """One block over a plan area (x0, x1, y0, y1; default the domain), i along x,
        ``cells`` (ni, nj) (default: about 40 along the longer side, square cells)."""
        from lagoonviz.core import multiblock as mb

        case_id = self._case_id(case)
        given = area
        if area is None:
            setup = self.data.case_setup(case_id, create=True)
            domain = setup.domain if setup is not None else None
            if domain is None:
                raise ValueError("no domain yet: give the area of the block")
            area = domain[:4]
        area = [round(float(v), 6) for v in area]
        if cells is None:
            lx, ly = abs(area[1] - area[0]), abs(area[3] - area[2])
            step = max(lx, ly) / 40.0 or 1.0
            cells = (max(4, round(lx / step)), max(4, round(ly / step)))
        cells = (int(cells[0]), int(cells[1]))
        script = "session.start_multiblock(" + (f"{area!r}, " if given is not None else "")
        script += f"cells={cells!r}"
        self._put_multiblock(case, mb.from_box(area, cells), script, "Start blocks")

    def multiblock_from_corridor(self, case: str | None = None) -> None:
        """The case's river corridor as block 0 of a multi-block grid (its banks the
        j faces, its ends the i faces; cells and smoothing of its grid)."""
        from lagoonviz.core import multiblock as mb

        case_id = self._case_id(case)
        made = mb.from_corridor(self._corridor_of(case_id))
        script = "session.multiblock_from_corridor("
        self._put_multiblock(case, made, script, "Blocks from corridor")

    def move_block_vertex(self, name: str, x: float, y: float, case: str | None = None) -> None:
        """Move a corner (vertex) of the blocks: every edge from it follows."""
        script = f"session.move_block_vertex({name!r}, {float(x)!r}, {float(y)!r}"
        self._change_multiblock(
            case, lambda t: t.moved_vertex(name, x, y), script, "Move block corner"
        )

    def move_block_point(
        self, edge: str, index: int, x: float, y: float, case: str | None = None
    ) -> None:
        """Move a point that shapes an edge (index 0: the first after its start)."""
        script = f"session.move_block_point({edge!r}, {int(index)}, {float(x)!r}, {float(y)!r}"
        self._change_multiblock(
            case, lambda t: t.moved_point(edge, int(index), x, y), script, "Move edge point"
        )

    def set_block_edge(self, edge: str, case: str | None = None, **options) -> None:
        """Change an edge: ``cells`` (carried on to the opposite faces of the blocks),
        ``curve`` ("straight", "through", "bspline"), ``ratio`` (last cell / first)."""
        words = ", ".join(f"{k}={v!r}" for k, v in options.items())
        script = f"session.set_block_edge({edge!r}" + (f", {words}" if words else "")
        self._change_multiblock(
            case, lambda t: t.with_edge(edge, **options), script, "Change block edge"
        )

    def set_block_face(
        self, block: int, face: str, kind: str, value: float | None = None, case: str | None = None
    ) -> None:
        """The boundary of a face without an interface: "wall", "inflow" (a discharge
        in m3/s) or "outflow" (a water level in m)."""
        script = f"session.set_block_face({int(block)}, {face!r}, {kind!r}"
        if value is not None:
            script += f", {float(value)!r}"
        self._change_multiblock(
            case,
            lambda t: t.with_bc(int(block), face, kind, value),
            script,
            "Set block boundary",
        )

    def set_block_options(self, block: int, case: str | None = None, **options) -> None:
        """Change a block: ``name``, ``smoothing`` (Winslow sweeps)."""
        words = ", ".join(f"{k}={v!r}" for k, v in options.items())
        script = f"session.set_block_options({int(block)}" + (f", {words}" if words else "")
        self._change_multiblock(
            case, lambda t: t.with_block(int(block), **options), script, "Change block"
        )

    def extrude_block(
        self, edge: str, length: float, cells: int = 10, case: str | None = None
    ) -> None:
        """One more block outside a boundary edge, ``length`` m deep, ``cells`` cells."""
        script = f"session.extrude_block({edge!r}, {float(length)!r}, cells={int(cells)}"
        self._change_multiblock(
            case, lambda t: t.extruded(edge, length, int(cells)), script, "Add block"
        )

    def split_block_edge(self, edge: str, x: float, y: float, case: str | None = None) -> None:
        """Cut an edge in two at its node nearest to (x, y) (a block then meets only
        one part, e.g. a river entering a harbour)."""
        script = f"session.split_block_edge({edge!r}, {float(x)!r}, {float(y)!r}"
        self._change_multiblock(case, lambda t: t.split_edge(edge, x, y), script, "Cut block edge")

    def add_block(self, area, cells=(10, 10), case: str | None = None) -> None:
        """One more block over a rectangle (x0, x1, y0, y1), with its own edges."""
        area = [round(float(v), 6) for v in area]
        cells = (int(cells[0]), int(cells[1]))
        script = f"session.add_block({area!r}, cells={cells!r}"
        self._change_multiblock(case, lambda t: t.added_box(area, cells), script, "Add block")

    def remove_block(self, block: int, case: str | None = None) -> None:
        """Take a block out (with the edges and corners only it used)."""
        script = f"session.remove_block({int(block)}"
        self._change_multiblock(case, lambda t: t.removed_block(int(block)), script, "Remove block")

    def make_native_block(self, block: int, case: str | None = None) -> None:
        """Make a block block 0, the grid of the run (DIVEMesh's box)."""
        script = f"session.make_native_block({int(block)}"
        self._change_multiblock(case, lambda t: t.with_native(int(block)), script, "Block 0")

    def scale_multiblock(self, factor: float, case: str | None = None) -> None:
        """Every edge with ``factor`` times its cells (2: twice as fine)."""
        script = f"session.scale_multiblock({float(factor)!r}"
        self._change_multiblock(case, lambda t: t.scaled(float(factor)), script, "Scale blocks")

    def remove_multiblock(self, case: str | None = None) -> None:
        """Remove a case's blocks.json (blocks.dat and the input files stay)."""
        self._put_multiblock(case, None, "session.remove_multiblock(", "Remove blocks")

    def multiblock_check(self, case: str | None = None):
        """What DIVEMesh and REEF3D would say about the blocks: (issues, quality)
        (:func:`lagoonviz.core.multiblock.check`, :func:`~lagoonviz.core.multiblock.quality`)."""
        from lagoonviz.core import multiblock as mb

        topology = self._multiblock_of(self._case_id(case))
        issues = mb.check(topology)
        quality = None if any(i.level == "error" for i in issues) else mb.quality(topology)
        return issues, quality

    def make_multiblock_grid(self, case: str | None = None):
        """Write the blocks as blocks.dat and set the case up for them, one undo step:
        R 20 1 and the sides of block 0 (C 11 - C 14) in control.txt, its box (B 10's x
        and y; its heights and B 1 kept); G 50 1, the native in- and outflow (B 60 1,
        W 10, F 62) and the open faces of the other blocks (G 54) in ctrl.txt. Refused
        while a rule is broken. Returns the grid's quality."""
        from lagoonviz.core import multiblock as mb

        case_id = self._case_id(case)
        topology = self._multiblock_of(case_id)
        issues = [i for i in mb.check(topology) if i.level == "error"]
        if issues:
            raise ValueError(
                "the blocks break a rule: "
                + issues[0].text
                + (f" (and {len(issues) - 1} more)" if len(issues) > 1 else "")
            )
        setup = self.data.case_setup(case_id)
        points = None
        if topology.bed == "geo" and setup is not None and (setup.path / "geo.dat").is_file():
            points = setup.geo_points(limit=10**9)
        nodes = []
        for x, y in topology.grids():
            nodes.append((x, y, mb.bed_heights(topology, x, y, points)))
        found = mb.settings(topology)
        control = self.input_file("control.txt", case_id)
        old = control.values("B 10") or [0.0, 1.0, 0.0, 1.0, 0.0, 1.0]
        box = [*found.box, *(old + [0.0, 1.0])[4:6]]

        def divemesh(f):
            f.set("R 20", [1], kinds="int")
            f.set("B 10", box)
            if "B 1" not in f and "B 2" not in f:
                f.set("B 1", [mb.curvi.nice_step(found.spacing)])
            for code, kind in found.sides.items():
                f.set(code, [kind], kinds="int")

        def reef3d(f):
            f.set("G 50", [1], kinds="int")
            f.remove("G 54")
            for b, kind, value in found.g54:
                f.add("G 54", [b, kind, value], kinds=["int", "int", "double"])
            if found.inflow is not None:
                f.set("B 60", [1], kinds="int")
                f.set("W 10", [found.inflow])
            if found.level is not None:
                f.set("F 62", [found.level])

        commands = [self._case_file(case_id, mb.GRID_FILE, mb.blocks_text(nodes), "", "blocks.dat")]
        for file, change in (("ctrl.txt", reef3d), ("control.txt", divemesh)):
            command = self._input_edit(file, case_id, change, "", file)
            if command is not None:
                commands.append(command)
        script = "session.make_multiblock_grid(" + (f"case={case!r})" if case else ")")
        self.stack.execute(Batch(commands, script, "Make multi-block grid"))
        return mb.quality(topology)

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

    def check_input(self, case_id: str | None = None):
        """Check a case's input before a run (the first case by default): ctrl.txt
        and control.txt together, DIVEMesh's grid for them, and what the run will
        cost; a :class:`lagoonviz.core.input_check.InputReport` (``print(report.text())``).
        Edits not saved yet are checked as they are on disk."""
        from lagoonviz.core.input_check import check_case

        if case_id is None:
            case_id = next((i.id for i in self.scene if i.kind == "case"), None)
        if case_id is None:
            raise ValueError("open a case first")
        folder = self.data.case_folder(case_id)
        if folder is None:
            raise ValueError(f"{case_id}: no case folder")
        return check_case(folder)

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

    def try_rotation(self, rotation_deg: float | None, about=None, case: str | None = None) -> None:
        """Show a case on the map as if its model were turned to ``rotation_deg``
        (degrees counter-clockwise from east), about a model point (default: its
        origin, as the georeference's rotation turns it) - the maps turn and move,
        north follows - without writing anything: what a rotation slider shows while
        it is dragged. None: as it is. Not an undo step; :meth:`set_georef` or
        :meth:`turn_model` then keeps it."""
        case_id = self._case_id(case)
        current = self.georef(case_id)
        tried = None
        if rotation_deg is not None and current is not None:
            point = (0.0, 0.0) if about is None else (float(about[0]), float(about[1]))
            tried = current.rotated(float(rotation_deg), point)
        self.data.backgrounds.set_preview(case_id, tried)
        self.scene.georef_tried(case_id)

    def set_rotation(self, rotation_deg: float, about=None, case: str | None = None) -> None:
        """Turn a case's georeference to ``rotation_deg`` about a model point (x, y):
        that point stays where it is on the map (the Map tab's slider: the middle of
        the view). Default: the model's origin."""
        case_id = self._case_id(case)
        current = self.georef(case_id)
        if current is None:
            raise ValueError(f"{case_id}: set the georeference first (set_georef)")
        if about is None:
            turned = current.replace(rotation_deg=float(rotation_deg))
        else:
            about = [round(float(about[0]), 6), round(float(about[1]), 6)]
            turned = current.rotated(float(rotation_deg), (about[0], about[1]))
        if turned == current:
            return
        script = f"session.set_rotation({float(rotation_deg)!r}"
        if about is not None:
            script += f", about={about!r}"
        script += f", case={case!r})" if case else ")"
        self._set_georef(case_id, turned, script, "Turn the georeference")

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
        spacing: float | str | None = None,
        depths: bool = False,
        case: str | None = None,
        progress=None,
        lat_below_msl: float = 0.0,
        most_points: int | None = None,
        nearshore: str | None = None,
        dean_a: float | None = None,
        trust_depth: float | None = None,
        overlays=None,
        icesat2: bool = False,
    ) -> int:
        """Bathymetry for DIVEMesh: a GeoTIFF, ASCII grid or x y z file (in ``crs``,
        or its own; ``"model"``: already in model coordinates) interpolated (``"tin"``,
        ``"idw"``, ``"nearest"``) onto a grid over the domain and written as geo.dat
        in model coordinates, every ``spacing`` metres (default: DIVEMesh's cell size,
        B 1; ``"finest"``: the finest the data has, Kartverket's 1 m in Norway,
        Copernicus' 30 m on land elsewhere, a file's own cells), coarser where it would
        be more than ``most_points`` points (default 4 million; 0: no limit).
        ``depths``: the file gives depths (positive down). Turns G 10 on.

        No file: ``source="best"`` downloads the bathymetry for the domain
        (Kartverket's land and sea model in Norway, EMODnet for the rest of
        European seas), ``"kartverket"`` or ``"emodnet"`` one of them. With "best"
        the coast lies where Copernicus' 30 m land has it, the coarser sea sources
        held below zero there; ``lat_below_msl``: how far EMODnet's datum (the lowest
        astronomical tide) lies below mean sea level here [m], so its depths share
        the land's zero.

        Next to the coast (downloads): ``nearshore`` "dean" (the default: a beach
        profile, depth ``dean_a`` x^(2/3) at x metres from the coast, bent to meet
        the sea's data), "smooth" (the smoothest bed from the coast to the sea's
        data: creeks, mudflats) or "flat" (at most 1 m deep, as LAGOON did before),
        where the coarse sea source cannot be trusted: close to the coast, or
        shallower than ``trust_depth`` [m] (ETOPO's flat 1 and 2 m over tidal flats).
        ``overlays``: files of one's own laid over the download (paths, or dicts
        with ``path``, ``crs``, ``depths``, ``below_msl``: how far the file's zero,
        a chart's datum, lies below mean sea level), the later over the earlier;
        ICESat-2 ATL24 files (``.h5``, or a folder of them) guide the nearshore
        profile. ``icesat2``: download ATL24 for the domain (a NASA Earthdata
        login: ``EARTHDATA_TOKEN`` or ``~/.netrc``).
        ``progress(fraction, text)`` hears how far it is (and may raise
        ``bathy.Cancelled``). Returns the number of points written."""
        job = self.geo_dat_job(
            source,
            crs,
            method,
            spacing,
            depths,
            case,
            lat_below_msl,
            most_points,
            nearshore=nearshore,
            dean_a=dean_a,
            trust_depth=trust_depth,
            overlays=overlays,
            icesat2=icesat2,
        )
        job.run(progress)
        return job.finish()

    def geo_dat_job(
        self,
        source: str | Path,
        crs: str | None = None,
        method: str = "tin",
        spacing: float | str | None = None,
        depths: bool = False,
        case: str | None = None,
        lat_below_msl: float = 0.0,
        most_points: int | None = None,
        nearshore: str | None = None,
        dean_a: float | None = None,
        trust_depth: float | None = None,
        overlays=None,
        icesat2: bool = False,
    ) -> GeoDatJob:
        """:meth:`make_geo_dat` in two parts, for a GUI: ``job.run(progress)`` does
        the work (reading, downloading, writing geo.dat) and may run in a worker
        thread; ``job.finish()`` then, in the thread of the session, turns G 10 on
        and records it. Checks what it can at once (raises ValueError)."""
        from lagoon_format.geo import Georef

        from lagoonviz.core.geo import nearshore as near
        from lagoonviz.core.geo.bathy import (
            FINEST,
            WEB,
            finest_file,
            finest_web,
            geo_grid,
            load,
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
        given_spacing = spacing
        web = isinstance(source, str) and source in WEB and not Path(source).exists()
        if web and crs == "model":
            raise ValueError("bathymetry from the web needs the case's georeference")
        if nearshore is not None and nearshore not in near.NEARSHORE:
            raise ValueError(
                f"nearshore must be one of {', '.join(near.NEARSHORE)}, got {nearshore!r}"
            )
        if dean_a is not None and not float(dean_a) > 0:
            raise ValueError(f"Dean's A must be above 0, got {dean_a}")
        specs = [near.spec_of(entry) for entry in overlays or []]
        if specs and not web:
            raise ValueError("files of one's own go over a download, not over another file")
        for spec in specs:
            if not Path(spec["path"]).exists():
                raise ValueError(f"{spec['path']}: no such file or folder")
        if icesat2 and not web:
            raise ValueError("ICESat-2 soundings go with a download")
        bathy = None
        if spacing == FINEST:
            if web:
                spacing, _ = finest_web(str(source), georef, domain[:4])
            else:
                bathy = load(Path(source).expanduser(), None if crs == "model" else crs, depths)
                if crs == "model":
                    bathy.crs = "model"
                spacing = finest_file(bathy)
        elif spacing is None:
            spacing = self.input_file("control.txt", case_id).value("B 1") or 0.0
        xs, ys = geo_grid(domain[:4], float(spacing or 0.0), most=most_points)
        code = f"session.make_geo_dat({str(source)!r}"
        for key, value, default in (
            ("crs", crs, None),
            ("method", method, "tin"),
            ("spacing", given_spacing if given_spacing is not None else spacing, None),
            ("depths", depths, False),
            ("case", case, None),
            ("lat_below_msl", float(lat_below_msl or 0.0), 0.0),
            ("most_points", most_points, None),
            ("nearshore", nearshore, None),
            ("dean_a", dean_a, None),
            ("trust_depth", trust_depth, None),
            ("overlays", specs, []),
            ("icesat2", bool(icesat2), False),
        ):
            if value != default:
                code += f", {key}={value!r}"
        job = GeoDatJob(
            self,
            case_id,
            case,
            folder,
            georef,
            xs,
            ys,
            source,
            web,
            crs,
            method,
            depths,
            code + ")",
            recipe={
                "source": str(source),
                "web": bool(web),
                "file_crs": crs,
                "method": method,
                "depths": bool(depths),
                "lat_below_msl": float(lat_below_msl or 0.0),
                "most_points": most_points,
                "nearshore": nearshore or near.DEFAULT,
                "dean_a": float(near.DEAN_A if dean_a is None else dean_a),
                "trust_depth": None if trust_depth is None else float(trust_depth),
                "overlays": specs,
                "icesat2": bool(icesat2),
                "spacing": float(xs[1] - xs[0]) if len(xs) > 1 else float(spacing or 0.0),
                "domain": [float(v) for v in domain[:4]],
                "crs": georef.crs if crs != "model" else "",
                "origin": list(georef.origin) if crs != "model" else [0.0, 0.0],
                "rotation_deg": georef.rotation_deg if crs != "model" else 0.0,
            },
        )
        job.bathy = bathy  # read already (for its finest spacing): not read again
        job.finest = float(spacing or 0.0) if given_spacing == FINEST else None
        return job

    def geo_dat_state(self, case: str | None = None) -> str:
        """Why the case's geo.dat no longer fits it, or "" (fits, or not known): made
        before the model was turned or moved on the map, in another CRS, or for a
        smaller domain. DIVEMesh would then mesh the bed of the old place;
        :meth:`remake_geo_dat` makes it again from the same source."""
        from lagoonviz.core.geo import recipe

        case_id = self._case_id(case)
        made = recipe.read(self.data.case_folder(case_id))
        return recipe.stale(made, self.georef(case_id), self.data.backgrounds.domain(case_id))

    def geo_dat_recipe(self, case: str | None = None) -> dict | None:
        """How the case's geo.dat was made (its source, spacing, the georeference and
        domain it was made for), or None when LAGOON did not make it."""
        from lagoonviz.core.geo import recipe

        return recipe.read(self.data.case_folder(self._case_id(case)))

    def remake_geo_dat_job(self, case: str | None = None) -> GeoDatJob:
        """:meth:`remake_geo_dat` in two parts (``job.run``, ``job.finish``)."""
        made = self.geo_dat_recipe(case)
        if made is None:
            raise ValueError("geo.dat was not made by LAGOON: make it in the Map tab")
        source = str(made.get("source", ""))
        if not made.get("web") and not Path(source).expanduser().is_file():
            raise ValueError(f"geo.dat's source {source} is not there any more: choose it again")
        for spec in made.get("overlays") or []:
            if not Path(str(spec.get("path", ""))).expanduser().exists():
                raise ValueError(f"geo.dat's file {spec.get('path')} is not there any more")
        trust_depth = made.get("trust_depth")
        return self.geo_dat_job(
            source,
            crs=made.get("file_crs"),
            method=str(made.get("method", "tin")),
            spacing=float(made.get("spacing") or 0.0) or None,
            depths=bool(made.get("depths", False)),
            case=case,
            lat_below_msl=float(made.get("lat_below_msl", 0.0) or 0.0),
            most_points=made.get("most_points"),
            # a geo.dat made before the nearshore profile: as it was made
            nearshore=str(made.get("nearshore", "flat")),
            dean_a=made.get("dean_a"),
            trust_depth=None if trust_depth is None else float(trust_depth),
            overlays=made.get("overlays") or None,
            icesat2=bool(made.get("icesat2", False)),
        )

    def remake_geo_dat(self, case: str | None = None, progress=None) -> int:
        """geo.dat again from the source it was made from, for the case as it is now
        (turned, moved, its domain): the same spacing, method and file. Returns the
        number of points."""
        job = self.remake_geo_dat_job(case)
        job.run(progress)
        return job.finish()

    # ------------------------------------------------------------- the sea
    def find_storms(
        self,
        first_year: int | None = None,
        last_year: int | None = None,
        count: int = 10,
        case: str | None = None,
        progress=None,
        source: str | None = None,
    ) -> list:
        """The largest wave events at a case (default the last ten years):
        :class:`~lagoonviz.core.geo.metocean.Storm` s, the highest first. ``source``:
        "MET Norway" (NORA3) or "Open-Meteo" (ERA5); default MET Norway in its waters,
        Open-Meteo elsewhere. Read at the middle of the wave generation side (else the
        domain's centre); the case needs a georeference. Not an undo step."""
        from lagoonviz.core import sea

        first, last = sea.default_years()
        found, _, _ = sea.find_storms(
            self,
            self._case_id(case),
            first_year or first,
            last_year or last,
            count,
            progress,
            source=source,
        )
        return found

    def sea_conditions(
        self, start, end, case: str | None = None, progress=None, source: str | None = None
    ) -> list[str]:
        """The sea at a case from ``start`` to ``end`` (UTC: ISO text, a datetime or
        seconds since 1970): the waves and wind and the water level and current (NORA3
        and NorKyst, or ERA5 and Météo-France's sea level with the tide from
        Open-Meteo; ``source`` as for :meth:`find_storms`), as tables in the case's
        ``LAGOON_Sea`` folder, shown as time series items (time 0 at ``start``).
        Returns the new items."""
        from lagoonviz.core import sea

        case_id = self._case_id(case)
        written = sea.conditions(self, case_id, start, end, progress, source=source)
        self.sea_notes = written.notes
        return self._show_sea_tables(case_id, written.files)

    def write_waves(self, when, case: str | None = None, apply: bool = True) -> dict:
        """The waves at a case at the hour ``when`` (UTC) from ERA5 (Open-Meteo): Hs, Tp
        and the mean direction for a JONSWAP spectrum; with ``apply`` the wave
        generation of ctrl.txt follows them (B 85 2, B 92 31, B 93, B 105; one undo
        step). For NORA3's 2D spectrum: :meth:`write_spectrum`."""
        from lagoonviz.core import sea

        return sea.waves_at(self, self._case_id(case), when, apply)

    def river_discharge(
        self,
        start,
        end,
        target: str | int = "B 60",
        case: str | None = None,
        apply: bool = True,
        at: tuple[float, float] | None = None,
    ):
        """GloFAS's discharge of the river at a case (Open-Meteo) from ``start`` to
        ``end`` (UTC), hourly, time 0 at ``start``: ``hydrograph.dat`` for ioFlow
        (``target`` "B 60", B 60 2) or ``hydrograph_Q_<n>.dat`` for side n as a patch
        (B 440, B 421); a table shown as a time series. Read at the river corridor's
        upstream end, else the inflow side's middle, else the domain's centre; ``at``
        = (x, y) to choose. With ``apply`` ctrl.txt follows (one undo step)."""
        from lagoonviz.core import sea

        case_id = self._case_id(case)
        found = sea.river(self, case_id, start, end, target, apply, at)
        self._show_sea_tables(case_id, [f for f in found.files if f.parent.name == sea.FOLDER])
        return found

    def find_gauges(
        self,
        case: str | None = None,
        sources=("Kartverket", "NOAA", "IOC"),
        count: int = 8,
        radius_km: float = 300.0,
    ) -> list:
        """The tide gauges nearest to a case (Kartverket, NOAA CO-OPS, IOC):
        :class:`~lagoonviz.core.geo.gauges.Station` s, the nearest first."""
        from lagoonviz.core import sea

        found, _ = sea.find_gauges(self, self._case_id(case), sources, count, radius_km)
        return found

    def gauge_levels(
        self, station, start, end, case: str | None = None, progress=None
    ) -> list[str]:
        """The water levels measured at a gauge (:meth:`find_gauges`; or its source and
        code as "NOAA:8518750") from ``start`` to ``end`` (UTC), with the tide
        predicted where its source has it, as a table in ``LAGOON_Sea`` shown as a time
        series (time 0 at ``start``): to compare a run with. Returns the new items."""
        from lagoonviz.core import sea

        case_id = self._case_id(case)
        station = self._station(station, case_id)
        _, table = sea.gauge_levels(self, case_id, station, start, end, progress)
        return self._show_sea_tables(case_id, [table])

    def gauge_boundaries(
        self, station, start, end, sides, case: str | None = None, apply: bool = True
    ) -> list:
        """A gauge's observed water level as the water level at sides of a case's
        domain (``hydrograph_FSF_<n>.dat``; with ``apply`` B 440, B 422: one undo step).
        Returns the files written."""
        from lagoonviz.core import sea

        case_id = self._case_id(case)
        station = self._station(station, case_id)
        return sea.gauge_boundaries(self, case_id, station, start, end, list(sides), apply)

    def _station(self, station, case_id: str):
        """A gauge given as a Station, or as "SOURCE:CODE" (found near the case)."""
        from lagoonviz.core.geo.gauges import Station

        if isinstance(station, Station):
            return station
        source, _, code = str(station).partition(":")
        for found in self.find_gauges(case_id, sources=(source,), count=10**6, radius_km=1e9):
            if found.code == code:
                return found
        raise ValueError(f"no gauge {station!r}")

    def _show_sea_tables(self, case_id: str, files: list[Path]) -> list[str]:
        self.data.reload(case_id)
        case = self.data.case(case_id)
        added = []
        for path in files:
            found = None
            if case is not None:
                found = next(
                    (f for f in case.series.values() if f.path.resolve() == path.resolve()), None
                )
            if found is None:
                continue
            shown = [
                i
                for i in self.scene
                if i.kind == "series"
                and i.properties.get("source") == found.key
                and self.data.case_item(i.id).id == case_id
            ]
            if shown:
                added.append(shown[0].id)
                continue
            added.append(self.add_item("series", name=found.name, source=found.key, case=case_id))
        return added

    def write_spectrum(
        self, when, case: str | None = None, apply: bool = True, half_width: float = 90.0
    ) -> dict:
        """NORA3's 2D wave spectrum at a case at ``when`` (UTC) as REEF3D's
        ``spectrum-file-2d.dat`` in the case folder; with ``apply`` the wave
        generation of ctrl.txt follows it (B 85 11, B 92 31, B 93, B 105, B 130 1; one
        undo step, saved with the input files). Returns Hs, Tp, the direction in the
        model and the file."""
        from lagoonviz.core import sea

        return sea.spectrum(self, self._case_id(case), when, apply, half_width)

    def write_boundaries(
        self,
        start,
        end,
        sides=None,
        case: str | None = None,
        apply: bool = True,
        progress=None,
        source: str | None = None,
    ) -> list:
        """The water level and discharge along the sides of a case's domain from
        ``start`` to ``end`` (UTC; NorKyst, or Open-Meteo's sea level and current with
        the depth from the case's bed; ``source`` as for :meth:`find_storms`): REEF3D's
        ``hydrograph_FSF_<n>.dat`` and ``hydrograph_Q_<n>.dat`` per side in the water
        (n: 1 x min, 2 y max, 3 y min, 4 x max; ``sides`` to choose), a table of them
        shown as a time series; with ``apply`` ctrl.txt gets the patches (B 440, B 421,
        B 422; one undo step). Returns the sides written."""
        from lagoonviz.core import sea

        case_id = self._case_id(case)
        chosen = sea.boundaries(self, case_id, start, end, sides, apply, progress, source=source)
        folder = self.data.case_folder(case_id)
        if chosen and folder is not None:
            from lagoonviz.core.geo.metocean import _utc

            table = folder / sea.FOLDER / f"Boundaries-{_utc(start):%Y%m%d%H}.dat"
            self._show_sea_tables(case_id, [table])
        return chosen

    def add_background_map(self, source: str = "Kartverket topo", **properties: Any) -> str:
        """A background map round a georeferenced case: map tiles (``source``, see
        ``kinds.BASEMAP_SOURCES``), or ``source="GeoTIFF file", file=...``."""
        return self.add_item("basemap", source=source, **properties)

    @property
    def mode(self) -> str:
        """ "results" or "setup" (see :meth:`set_mode`)."""
        return self.scene.mode

    def _volume_case(self, properties: dict[str, Any]) -> dict[str, Any]:
        """Properties of a new item that cuts a volume, with the case it cuts: the one
        given, else (several cases open) the first case that has a volume, named so
        the script and the saved session cut the same one. One case: left unnamed."""
        if properties.get("case") or sum(i.kind == "case" for i in self.scene) < 2:
            return properties
        case_id = self.data.volume_case()
        return {**properties, "case": case_id} if case_id else properties

    def add_of(self, kind: str, dataset: str, case: str | None = None) -> str:
        """Add an item that shows a chosen output (see :meth:`DataManager.addable`): an
        iso-surface or iso-volume of a volume, a free surface or a bed; a free surface
        of SFLOW beside NHFLOW's; a slice of an earlier run's volume. ``case``: the
        case item (several cases open). Returns the item id."""
        properties: dict[str, Any] = {"dataset": dataset}
        if case:
            properties["case"] = case
        elif sum(i.kind == "case" for i in self.scene) > 1:
            raise ValueError("several cases are open: say which one (case=)")
        from lagoonviz.core.data.manager import shows

        found = self._case_datasets(case).get(dataset)
        if found is None:
            raise ValueError(f"no output {dataset!r} in the case")
        if not shows(kind, found):
            raise ValueError(f"{found.label}: not something a {KINDS[kind].label} shows")
        if kind == "statistics":
            properties["source"] = properties.pop("dataset")
            return self.add_statistics(**properties)
        if kind == "iso_surface":
            return self.add_iso_surface(**properties)
        if kind == "iso_volume":
            return self.add_iso_volume(**properties)
        if kind == "slice":
            return self.add_slice("y", **properties)
        if kind == "streamlines":
            return self.add_streamlines(**properties)
        variable = found.default_variable()
        spec = KINDS[kind].properties.get("variable")
        if spec is not None and variable != spec.default and variable in found.variables():
            properties["variable"] = variable
        return self.add_item(kind, **properties)

    def _case_datasets(self, case: str | None) -> dict:
        case_item = self.data._case_item_for(case or "")
        if case_item is None:
            return {}
        try:
            return self.data.open(case_item.properties["path"]).datasets
        except (KeyError, OSError, ValueError):
            return {}

    def add_statistics(
        self,
        source: str = "",
        start: float = 0.0,
        end: float = 0.0,
        variables=None,
        wet_depth: float = 0.01,
        case: str | None = None,
        progress=None,
        compute: bool = True,
        **properties: Any,
    ) -> str:
        """A statistics map of a surface output over a time window (``source``: its key,
        default the free surface; ``end`` 0: to the last output): per point the max,
        min, mean, time of the maximum and change of ``variables`` (default eta and
        velocity; a bed's elevation), the water depth's maximum, the
        significant wave height Hs from eta, when the water first arrives, how long it
        stays and the maximum inundation. Worked out now (``compute``) and kept in the
        case's ``LAGOON_Statistics`` folder. Returns the item."""
        if isinstance(variables, (list, tuple)):
            variables = ", ".join(variables)
        props: dict[str, Any] = {
            "source": source,
            "start": float(start),
            "end": float(end),
            "of_variables": variables or "",
            "wet_depth": float(wet_depth),
            **properties,
        }
        if case:
            props["case"] = case
        elif sum(i.kind == "case" for i in self.scene) > 1:
            props["case"] = next(i.id for i in self.scene if i.kind == "case")
        item = self.add_item("statistics", **props)
        if compute:
            try:
                self.compute_statistics(item, progress)
            except Exception:
                self.stack.undo()  # nothing to show: no item either
                raise
        return item

    def compute_statistics(self, item_id: str, progress=None, again: bool = False) -> Path:
        """Work a statistics item's map out (reading every output of its window) and
        keep it in the case's ``LAGOON_Statistics`` folder; a map kept for the same
        recipe is used as it is unless ``again``. Not an undo step. Returns the file."""
        job = self.statistics_job(item_id, again)
        job.run(progress)
        return job.finish()

    def statistics_job(self, item_id: str, again: bool = False):
        """:meth:`compute_statistics` in two parts, for a GUI: ``job.run(progress)``
        reads the outputs and writes the map (it may run in a worker thread);
        ``job.finish()`` then, in the session's thread, shows it."""
        item = self.scene.get(item_id)
        if item.kind != "statistics":
            raise ValueError(f"{item.name} is no statistics item")
        case = self.data.case(item_id)
        if case is None:
            raise ValueError("the case is not open")
        recipe = self.data.statistics_recipe(item.properties, case)
        if recipe is None:
            raise ValueError("the case has no free surface or bed to make statistics of")
        source = case.datasets.get(recipe.source)
        if source is None:
            raise ValueError(f"no output {recipe.source!r} in the case")
        return _StatisticsJob(
            self, item_id, self.data.statistics_file(item_id), source, recipe, again
        )

    def _statistics_ready(self, item_id: str) -> None:
        item = self.scene.get(item_id)
        dataset = self.data.dataset(item_id)
        if dataset is not None and item.properties.get("variable") not in dataset.variables():
            item.properties["variable"] = dataset.default_variable()  # (not an undo step)
        self.scene.data_changed()

    def add_slice(self, normal: str = "y", position: float | None = None, **properties: Any) -> str:
        """Add a slice through the volume; without a position it cuts the middle.
        With several cases open, ``case=`` names the one to cut (by default the
        first that has a volume).

        The position is computed here and recorded, so the script and the saved
        session cut at exactly the same place.
        """
        properties = self._volume_case(properties)
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
        by, ``case``) as for :meth:`add_item`."""
        properties = self._volume_case(properties)
        if iso_by is None:
            chosen = self.data.iso_default(properties)
            iso_by, default = chosen if chosen else ("none", 0.0)
            iso_value = default if iso_value is None else iso_value
        return self.add_item(
            "iso_surface", iso_by=iso_by, iso_value=float(iso_value or 0.0), **properties
        )

    def add_iso_volume(
        self,
        iso_by: str | None = None,
        iso_min: float | None = None,
        iso_max: float | None = None,
        **properties: Any,
    ) -> str:
        """Add an iso-volume: the part of the volume where ``iso_by`` lies between
        ``iso_min`` and ``iso_max``; without a variable a first one (phi >= 0, the water
        of a CFD run; see :meth:`DataManager.iso_volume_default`), recorded so the
        script repeats it."""
        properties = self._volume_case(properties)
        if iso_by is None:
            chosen = self.data.iso_volume_default(properties)
            iso_by, low, high = chosen if chosen else ("none", 0.0, 1.0)
            iso_min = low if iso_min is None else iso_min
            iso_max = high if iso_max is None else iso_max
        return self.add_item(
            "iso_volume",
            iso_by=iso_by,
            iso_min=float(iso_min if iso_min is not None else 0.0),
            iso_max=float(iso_max if iso_max is not None else 1.0),
            **properties,
        )

    def add_streamlines(
        self, seed_normal: str = "y", seed_position: float | None = None, **properties: Any
    ) -> str:
        """Add streamlines of the volume's velocity, seeded on a plane through the
        middle of the volume (the position is recorded, like a slice's)."""
        properties = self._volume_case(properties)
        if seed_position is None:
            center = self.data.domain_center("streamlines", properties)
            seed_position = round(center["xyz".index(seed_normal)], 6) if center else 0.0
        return self.add_item(
            "streamlines", seed_normal=seed_normal, seed_position=seed_position, **properties
        )

    def remove_item(self, item_id: str) -> None:
        item = self.scene.get(item_id)
        if item.kind == "corridor":  # the corridor itself goes: its corridor.json too
            owner = self.data.case_item(item_id)
            if owner is not None and self.corridor(owner.id) is not None:
                from lagoonviz.core import corridor as corridors

                script = f"session.remove_item({item_id!r})"
                commands = [RemoveItem(i) for i in self._corridor_items(owner.id)] + [
                    self._case_file(owner.id, corridors.FILE, None, "", "corridor.json")
                ]
                self.stack.execute(Batch(commands, script, "Remove river corridor"))
                return
        if item.kind == "multiblock":  # the blocks go: their blocks.json too
            owner = self.data.case_item(item_id)
            if owner is not None and self.multiblock(owner.id) is not None:
                from lagoonviz.core import multiblock as mb

                script = f"session.remove_item({item_id!r})"
                commands = [RemoveItem(i) for i in self._multiblock_items(owner.id)] + [
                    self._case_file(owner.id, mb.FILE, None, "", "blocks.json")
                ]
                self.stack.execute(Batch(commands, script, "Remove blocks"))
                return
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
        if key == "case" and self.scene.get(item_id).kind in ("slice", "streamlines"):
            moved = self._cut_moved_to_case(item_id, str(value))
            if moved:
                self.stack.execute(SetProperties(self.scene, item_id, moved))
                return
        self.stack.execute(SetProperty(self.scene, item_id, key, value), merge=merge)

    def _cut_moved_to_case(self, item_id: str, case_id: str) -> dict[str, Any] | None:
        """A slice (or streamlines' seed plane) put on another case: its new properties,
        through the middle of that case's volume when its plane would miss it; None
        when it cuts it where it is."""
        item = self.scene.get(item_id)
        props = {**item.properties, "case": case_id, "dataset": ""}
        dataset = self.data.dataset_for(item.kind, props)
        if dataset is None:
            return None
        axis_key, position_key = (
            ("normal", "position") if item.kind == "slice" else ("seed_normal", "seed_position")
        )
        axis = "xyz".index(str(item.properties.get(axis_key, "y")))
        low, high = dataset.bounds(0)
        position = float(item.properties.get(position_key, 0.0))
        if low[axis] <= position <= high[axis]:
            return {"case": case_id, "dataset": ""}
        middle = round(float(low[axis] + high[axis]) / 2, 6)
        return {"case": case_id, "dataset": "", position_key: middle}

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
        """Fix the colour range of an item to the values of the current frame (their
        min and max times the item's "range_percent")."""
        from lagoonviz.core.data.manager import RANGE_FROM, auto_range, scaled_range

        mesh = self.data.load(item_id)
        if mesh is None:
            raise ValueError(f"{item_id}: no data loaded")
        props = self.scene.get(item_id).properties
        values = mesh.scalar(str(props.get("variable", "none")))
        low, high = scaled_range(
            *auto_range(values, str(props.get("range_from", RANGE_FROM[0]))), props
        )
        # rounded to significant digits, not decimals: k (1e-6) or epsilon (1e-10) of a
        # free surface must not become 0 to 0
        self.set_properties(
            item_id, auto_range=False, vmin=float(f"{low:.6g}"), vmax=float(f"{high:.6g}")
        )
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

    def set_legend(self, **settings) -> dict:
        """Where films and saved images put the colour bars and the time, and how they
        look; the 3D view's bars on or off (lagoonviz.core.legend.LAYOUT): ``bars``
        (a corner: "lower left", "lower right", "upper left", "upper right"),
        ``time`` (a corner or "off"), ``length`` (of the image's width), ``thickness``
        and ``text`` (1: as by default), ``box`` (a light box behind them), ``view``
        ("below the view" or "off"). ``reset=True`` first goes back to the defaults.
        Each bar's title, numbers and ticks are the item's (``legend_*``). View
        state: not an undo step. Returns the whole layout."""
        from lagoonviz.core.legend import layout

        reset = bool(settings.pop("reset", False))
        current = {} if reset else dict(self.scene.legend)
        current.update(settings)
        self.scene.set_legend(current)
        return layout(self.scene.legend)

    def legend_layout(self) -> dict:
        """The whole legend layout (:meth:`set_legend`)."""
        from lagoonviz.core.legend import layout

        return layout(self.scene.legend)

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
        if self.scene.legend:
            settings = ", ".join(f"{k}={v!r}" for k, v in self.scene.legend.items())
            lines.append(f"session.set_legend({settings})")
        return "\n".join(lines) + "\n"

    @staticmethod
    def from_script(source: str) -> Session:
        """Run a recorded script and return the session it builds."""
        namespace: dict[str, Any] = {}
        exec(compile(source, "<lagoon script>", "exec"), namespace)
        return namespace["session"]


class GeoDatJob:
    """Making geo.dat (:meth:`Session.geo_dat_job`): ``run`` the work, then
    ``finish`` in the session's thread."""

    def __init__(
        self,
        session,
        case_id,
        case,
        folder,
        georef,
        xs,
        ys,
        source,
        web,
        crs,
        method,
        depths,
        code,
        recipe=None,
    ):
        self.session = session
        self.recipe = recipe
        self.case_id, self.case, self.folder = case_id, case, folder
        self.georef, self.xs, self.ys = georef, xs, ys
        self.source, self.web, self.crs = source, web, crs
        self.method, self.depths, self.code = method, depths, code
        self.count = 0
        self.sources: list[str] = []
        self.bathy = None
        #: with spacing "finest": the finest the data has [m] (the grid's may be
        #: coarser, at most MOST_POINTS points)
        self.finest: float | None = None

    def run(self, progress=None) -> int:
        """Read or download the bathymetry, carry it onto the grid, write geo.dat
        (any thread). Returns the number of points."""
        from lagoonviz.core.geo.bathy import grid_heights, load, web_heights, write_geo_dat

        def tell(fraction: float, text: str) -> None:
            if progress is not None:
                progress(max(0.0, min(1.0, fraction)), text)

        if self.web:
            made = self.recipe or {}
            overlays = self._overlays(made, tell)
            heights, self.sources = web_heights(
                self.source,
                self.georef,
                self.xs,
                self.ys,
                float(self.xs[1] - self.xs[0]),
                progress=lambda f, t: tell(0.2 + 0.6 * f, t),
                lat_below_msl=float(made.get("lat_below_msl", 0.0) or 0.0),
                nearshore=str(made.get("nearshore") or "dean"),
                dean_a=made.get("dean_a"),
                overlays=overlays,
                method=self.method,
                trust_depth=made.get("trust_depth"),
            )
        else:
            tell(0.0, f"Reading {Path(self.source).name}…")
            crs = "model" if self.crs == "model" else self.crs
            bathy = self.bathy or load(Path(self.source).expanduser(), crs, self.depths)
            if self.crs == "model":
                bathy.crs = "model"
                if bathy.raster is not None:
                    bathy.raster.crs = "model"
            tell(0.3, "Onto the grid…")
            heights = grid_heights(bathy, self.georef, self.xs, self.ys, self.method)
            self.sources = [Path(self.source).name]
        self.count = write_geo_dat(
            self.folder / "geo.dat",
            self.xs,
            self.ys,
            heights,
            progress=lambda f, t: tell(0.8 + 0.2 * f, t),
        )
        if self.recipe is not None:
            from lagoonviz.core.geo import recipe

            recipe.write(self.folder, dict(self.recipe, sources=list(self.sources)))
        return self.count

    def _overlays(self, made: dict, tell) -> list:
        """The files of one's own of the recipe, read (ICESat-2's cut to the domain),
        and ICESat-2 downloaded when asked."""
        from lagoonviz.core.geo import icesat2
        from lagoonviz.core.geo.download import _world_box
        from lagoonviz.core.geo.nearshore import Overlay, load_overlay

        box = (float(self.xs[0]), float(self.xs[-1]), float(self.ys[0]), float(self.ys[-1]))
        bounds = _world_box(self.georef, box, "EPSG:4326", pad=0.0)
        overlays = []
        for spec in made.get("overlays") or []:
            tell(0.0, f"Reading {Path(spec['path']).name}…")
            overlays.append(load_overlay(spec, bounds))
        if made.get("icesat2"):
            files = icesat2.download(bounds, progress=lambda f, t: tell(0.2 * f, t))
            overlays.insert(0, Overlay(icesat2.soundings(files, bounds), 0.0, True))
        return overlays

    def finish(self) -> int:
        """After run, in the session's thread: G 10 1, the set-up reads geo.dat
        again, the script line. Returns the number of points."""
        session = self.session
        session.geo_sources = list(self.sources)
        session.data.reload(self.case_id)  # geo.dat's points: a dataset of the case
        setup = session.data.case_setup(self.case_id, create=True)
        if setup is not None:
            setup.geo_changed()
        session.record_output(self.code)
        session.set_input("control.txt", "G 10", [1], case=self.case)
        for add in [a for a in session._missing_setup_items(self.case_id) if a.kind == "geodat"]:
            session.add_item("geodat", **add.overrides)
        session.scene.inputs_changed(self.case_id)
        session.scene.data_changed()
        return self.count


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


class _StatisticsJob:
    """A statistics map to work out (:meth:`Session.statistics_job`)."""

    def __init__(self, session, item_id, path, source, recipe, again) -> None:
        self.session, self.item_id, self.path = session, item_id, path
        self.source, self.recipe, self.again = source, recipe, again
        self.times = session.data.dataset_times(source)

    def run(self, progress=None) -> None:
        from lagoonviz.core import statistics

        if self.path.is_file() and not self.again:
            return
        mesh, facts = statistics.compute(self.source, self.times, self.recipe, progress)
        statistics.save(self.path, mesh, facts, self.recipe)

    def finish(self) -> Path:
        case = self.session.data.case(self.item_id)
        if case is not None:
            case.datasets.pop("statistics_" + self.path.stem, None)  # read the new one
        self.session._statistics_ready(self.item_id)
        return self.path
