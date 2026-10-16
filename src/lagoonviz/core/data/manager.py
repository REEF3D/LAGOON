# SPDX-License-Identifier: GPL-3.0-or-later
"""Data manager: from scene items to meshes, with caching and background loading.

The scene only stores descriptions (which case, which dataset, which variable).
:class:`DataManager` opens the cases those descriptions point to, maps the scene
time to frames, and keeps recently used frames in memory. :class:`BackgroundLoader`
reads frames in worker threads so a GUI never waits on the disk; it has no Qt
dependency, the GUI only forwards its callbacks to the main thread.
"""

from __future__ import annotations

import threading
from collections import OrderedDict
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

from lagoonviz.core.colormaps import SERIES_PALETTES, plain_color
from lagoonviz.core.contours import contour_levels, contour_segments
from lagoonviz.core.data.background import Backgrounds
from lagoonviz.core.data.divemesh_grid import (
    PREVIEW,
    Grid,
    grid_lines,
    read_partition,
    read_preview,
    sigma_grid,
    uniform_grid,
)
from lagoonviz.core.data.mesh import SurfaceMesh, exaggerate_normals
from lagoonviz.core.data.profiles import Profile, ProfileLine, ProfileSet
from lagoonviz.core.data.reef3d import CASE_ROLES, ROLES, Case, Dataset, open_case
from lagoonviz.core.data.series import SeriesFile, TimeSeries, parse_columns
from lagoonviz.core.data.setup import (
    ITEM_GROUPS,
    SIGMA_SOLVERS,
    CaseSetup,
    has_inputs,
    read_setup,
)
from lagoonviz.core.kinds import kind_spec
from lagoonviz.core.scene import MODE_KINDS, SETUP_KINDS, Scene, SceneItem
from lagoonviz.core.vectors import arrow_segments, grid_spacing, pick_points

#: item kinds that show a dataset, and the dataset role they use by default
DATA_KINDS = {kind: role for role, kind in ROLES.items() if role not in ("solid", "topo")} | {
    "slice": "volume",
    "iso_surface": "volume",
    "iso_volume": "volume",
    "streamlines": "volume",
}
#: 2D: heights of the flat layers (the free surface over objects over the bed)
PLAN_LAYERS = {
    "bathymetry": 0.0,
    "volume": 0.5,
    "slice": 0.5,
    "iso_surface": 0.5,
    "iso_volume": 0.5,
    "streamlines": 2.5,
    "structure": 1.0,
    "body": 1.5,
    "dem": 1.5,
    "ice": 1.5,
    "rods": 1.5,
    "particles": 2.2,
    "geodat": 3.5,  # over the water, under the set-up's marks
    "seastate": 3.0,  # where the free surface would be
    "mooring": 2.0,
    "free_surface": 3.0,
    "amr": 3.0,
}
#: kinds drawn as points, one screen-space dot each
POINT_KINDS = ("particles", "geodat")
#: kinds drawn as the free surface: the still water level, dry land, the probe
WATER_KINDS = ("free_surface", "amr")
#: 2D: gauges and free surface lines on top of everything
PLAN_MARKS = 4.0
AXES = {"x": 0, "y": 1, "z": 2}


@dataclass
class DisplaySurface:
    """Everything a renderer needs to draw one item at one time, as plain arrays."""

    item_id: str
    positions: np.ndarray  # (n, 3) float32, exaggeration applied
    triangles: np.ndarray  # (m, 3) uint32
    normals: np.ndarray  # (n, 3) float32
    values: np.ndarray | None  # (n,) float32 to colour by, None for a plain colour
    vmin: float
    vmax: float
    colormap: str
    opacity: float
    visible: bool
    variable: str
    frame: int
    #: colour without a colour map (material or kind)
    color: str = "#cccccc"
    #: (k, 2) uint32 line segments for line items (mooring lines)
    segments: np.ndarray | None = None
    line_width: float = 3.0
    #: point items (particles): drawn as points of this size [px], not triangles
    point_size: float | None = None
    #: the item's material (bodies, structures, DEM elements), for the cinematic looks
    material: str = ""


@dataclass
class ContourLines:
    """Contour lines of one item: segments in 3D and the levels they show."""

    item_id: str
    segments: np.ndarray  # (k, 2, 3) float32
    levels: list[float]
    visible: bool


@dataclass
class Trails:
    """Paths of particles (or element centres) over the outputs before: segments in 3D."""

    item_id: str
    segments: np.ndarray  # (k, 2, 3) float32
    visible: bool
    color: str = "#7f7f7f"


@dataclass
class VectorArrows:
    """Arrows of one item: shafts and head strokes as segments in 3D."""

    item_id: str
    segments: np.ndarray  # (k, 2, 3) float32
    visible: bool


@dataclass
class Trace:
    """One line of a time-series plot."""

    item_id: str
    name: str
    unit: str
    time: np.ndarray
    values: np.ndarray


@dataclass
class GaugeMarks:
    """Wave gauges of a series item in 3D: a staff per gauge and a dot at the water.

    Each gauge has the colour of its line in the plots, so 3D and plots read together.
    """

    item_id: str
    names: list[str]
    staffs: np.ndarray  # (k, 2, 3) float32: bottom and top of each staff
    dots: np.ndarray  # (k, 3) float32; z is NaN when the time is outside the record
    colors: list[str]
    visible: bool


@dataclass
class SetupMarks:
    """The signs of a "Case set-up" item in 3D, grouped by colour: lines, dots, filled
    areas and labels (text, position, colour), where the view draws them."""

    item_id: str
    lines: list[tuple[str, np.ndarray]]  # (colour, (n, 2, 3) float32)
    points: list[tuple[str, np.ndarray]]  # (colour, (m, 3) float32)
    faces: list[tuple[str, np.ndarray]]  # (colour, (k, 3, 3) float32)
    labels: list[tuple[str, tuple[float, float, float], str]]
    visible: bool
    #: the signs of the command selected in the set-up editor: (n, 2, 3) lines and
    #: (m, 3) dots, drawn over the others
    highlight: tuple[np.ndarray, np.ndarray] | None = None


@dataclass
class Probe:
    """Values at a point of an item's surface."""

    item_id: str
    name: str
    point: tuple[float, float, float]
    values: dict[str, float]
    variable: str

    def summary(self) -> str:
        """One line: the coloured variable first, then the position."""
        parts = [self.name]
        if self.variable in self.values:
            parts.append(f"{self.variable} = {self.values[self.variable]:.4g}")
        x, y, z = self.point
        parts.append(f"x {x:.3f}  y {y:.3f}  z {z:.3f}")
        return " · ".join(parts)


@dataclass(frozen=True)
class FrameKey:
    case: str
    dataset: str
    index: int
    arrays: tuple[str, ...]
    #: what is read from the frame: () the surface, ("slice", axis, position) a section
    extra: tuple = ()
    #: every how many grid lines a surface is read (1: full detail; more while
    #: the time moves in the streaming memory mode)
    detail: int = 1


def _kind_of(key) -> tuple:
    """What frames read alike share: case, dataset, what is read, detail."""
    if isinstance(key, FrameKey):
        return (key.case, key.dataset, key.extra, key.detail)
    return (key,)


class FrameCache:
    """Least-recently-used cache of meshes with a memory budget: ``max_bytes``, or
    the memory use mode's (:mod:`lagoonviz.core.memory`) when not given."""

    def __init__(self, max_bytes: int | None = None):
        self._max_bytes = max_bytes
        self._items: OrderedDict[FrameKey, SurfaceMesh] = OrderedDict()
        self._bytes = 0
        self._lock = threading.Lock()
        self.hits = 0
        self.misses = 0
        #: the size of the last frame of each dataset (and detail): what reading
        #: ahead would add (see BackgroundLoader.prefetch)
        self._sizes: dict[tuple, int] = {}

    @staticmethod
    def size_of(mesh: SurfaceMesh) -> int:
        # triangles are shared between frames of a dataset, so they are not counted
        normals = mesh.normals.nbytes if mesh.normals is not None else 0
        return mesh.points.nbytes + normals + sum(a.nbytes for a in mesh.point_data.values())

    def get(self, key: FrameKey) -> SurfaceMesh | None:
        with self._lock:
            mesh = self._items.get(key)
            if mesh is None:
                self.misses += 1
                return None
            self._items.move_to_end(key)
            self.hits += 1
            return mesh

    @property
    def max_bytes(self) -> int:
        if self._max_bytes is not None:
            return self._max_bytes
        from lagoonviz.core import memory

        return memory.current().cache_bytes

    @max_bytes.setter
    def max_bytes(self, value: int | None) -> None:
        self._max_bytes = value
        self.trim()

    def put(self, key: FrameKey, mesh: SurfaceMesh) -> None:
        with self._lock:
            if key in self._items:
                self._bytes -= self.size_of(self._items.pop(key))
            self._items[key] = mesh
            size = self.size_of(mesh)
            self._bytes += size
            self._sizes[_kind_of(key)] = size
            self._evict(1)

    def size_hint(self, key: FrameKey) -> int | None:
        """The size of the last frame read like ``key`` (same dataset and kind), if
        any was."""
        with self._lock:
            return self._sizes.get(_kind_of(key))

    def trim(self) -> None:
        """Down to the budget (after the memory use mode changed)."""
        with self._lock:
            self._evict(0)

    def _evict(self, keep: int) -> None:
        budget = self.max_bytes
        while self._bytes > budget and len(self._items) > keep:
            _, old = self._items.popitem(last=False)
            self._bytes -= self.size_of(old)

    def __contains__(self, key: object) -> bool:
        with self._lock:
            return key in self._items

    def __len__(self) -> int:
        return len(self._items)

    @property
    def nbytes(self) -> int:
        return self._bytes

    def clear(self) -> None:
        with self._lock:
            self._items.clear()
            self._bytes = 0
            self._sizes.clear()

    def drop_case(self, case: str) -> int:
        """Forget the frames of one case (its files changed). Returns how many."""
        with self._lock:
            keys = [key for key in self._items if key.case == case]
            for key in keys:
                self._bytes -= self.size_of(self._items.pop(key))
            return len(keys)


class DataManager:
    """Opens the cases of a scene and loads the frames its items show."""

    def __init__(self, scene: Scene, cache: FrameCache | None = None):
        self.scene = scene
        self.cache = cache or FrameCache()
        self._cases: dict[str, Case] = {}
        self._errors: dict[str, str] = {}
        self._resolved: dict[str, str] = {}
        self._auto_ranges: dict[tuple, tuple[float, float]] = {}
        #: (dataset key, variable) -> a variable's values sampled over the run, made
        #: in a loader thread when the first frame is read (see _sampled_values)
        self._samples: dict[tuple, list[np.ndarray]] = {}
        self._samples_lock = threading.Lock()
        #: particle trails: ids and their centres of each output read for them
        self._trail_centres: dict[FrameKey, tuple[np.ndarray, np.ndarray]] = {}
        self._still_water: dict[tuple, float] = {}
        self._wet: dict[str, tuple] = {}  # item -> (base, kept triangles[, depth])
        self._logged: tuple[tuple, np.ndarray | None] | None = None
        #: set-ups of case folders (their input files), by folder
        self._setups: dict[str, CaseSetup] = {}
        #: the command selected in the set-up editor: (case item, file, code, occurrence)
        self.highlight: tuple[str, str, str, int] | None = None
        self._lock = threading.Lock()
        #: background maps and the georeference of cases
        self.backgrounds = Backgrounds(self)
        #: the time is moving (playing, the timeline dragged): in the streaming memory
        #: mode, large surfaces are read coarser meanwhile (FrameKey.detail)
        self.moving = False
        #: the settings of each open case's derived quantities (case key -> Settings)
        self._derived: dict[str, object] = {}
        scene.subscribe(self._derived_changed)

    # ------------------------------------------------------------- cases
    def reload(self, case_item_id: str | None = None) -> list[str]:
        """Read a case's folders again (all cases by default): new outputs of a running
        simulation, a restart, new gauges or profiles. The frames read so far are
        forgotten, since a restart may have rewritten them; colour ranges are kept, so
        colours stay comparable. Returns the reloaded case paths."""
        paths = []
        for item in self.scene:
            if item.kind != "case" or (case_item_id and item.id != case_item_id):
                continue
            raw = str(item.properties.get("path", ""))
            if not raw:
                continue
            key = self._resolved.get(raw) or str(Path(raw).expanduser().resolve())
            with self._lock:
                self._cases.pop(key, None)
                self._errors.pop(key, None)
            self.cache.drop_case(key)
            paths.append(key)
        self._still_water.clear()
        self._wet.clear()
        self._logged = None
        with self._samples_lock:
            self._samples.clear()  # the run may have grown: sampled again
        for key in [k for k in self._setups if not self._setups[k].control.modified]:
            if not self._setups[key].ctrl.modified:
                del self._setups[key]  # read again (edits not saved yet are kept)
        return paths

    def signature(self, case_item_id: str) -> tuple:
        """A cheap fingerprint of a case's output folders (their modification times
        and the run logs' sizes): it changes when the simulation writes something."""
        case_item = self.scene.get(case_item_id)
        root = Path(str(case_item.properties.get("path", ""))).expanduser()
        if root.is_file():
            root = root.parent
        if root.name.startswith("REEF3D_"):
            root = root.parent
        stamps = []
        try:
            for folder in sorted(root.iterdir()):
                if folder.is_dir() and folder.name.startswith("REEF3D_"):
                    stamps.append((folder.name, folder.stat().st_mtime))
                    if folder.name == "REEF3D_Case":
                        stamps += [(f.name, f.stat().st_size) for f in folder.glob("*.jsonl")]
                    if folder.suffix == ".lagoon":  # a store grows its outputs' time arrays
                        stamps += [
                            (f.parent.parent.name, f.stat().st_mtime)
                            for f in sorted(folder.glob("*/time/zarr.json"))
                            + sorted(folder.glob("*/*/time/zarr.json"))  # bodies, profiles
                        ]
        except OSError:
            return ()
        return tuple(stamps)

    def open(self, path: str | Path) -> Case:
        """Open (or return the already open) case at ``path``."""
        raw = str(path)
        key = self._resolved.get(raw)
        if key is None:
            key = self._resolved[raw] = str(Path(path).expanduser().resolve())
        with self._lock:
            case = self._cases.get(key)
            if case is None:
                try:
                    case = open_case(key)
                except (OSError, ValueError) as error:
                    self._errors[key] = str(error)
                    raise
                self._cases[key] = case
                self._errors.pop(key, None)
                self._attach_derived(key, case)
            return case

    # ------------------------------------------------------------- derived quantities
    def derived_settings(self, key: str):
        """The settings of a case's derived quantities (Manning's n, its formulas),
        from its case item."""
        from lagoonviz.core.data import derived

        for item in self.scene:
            if item.kind != "case":
                continue
            raw = str(item.properties.get("path", ""))
            if not raw or (self._resolved.get(raw) or str(Path(raw).expanduser().resolve())) != key:
                continue
            try:
                formulas = derived.parse_formulas(str(item.properties.get("formulas", "")))
            except ValueError:
                formulas = []
            return derived.Settings(float(item.properties.get("manning_n", 0.025)), formulas)
        return derived.Settings()

    def _attach_derived(self, key: str, case) -> None:
        from lagoonviz.core.data import derived

        settings = self.derived_settings(key)
        self._derived[key] = settings
        for dataset in case.datasets.values():
            try:
                derived.attach(dataset, settings)
            except Exception:  # noqa: BLE001 (a dataset without derived quantities)
                continue

    def _derived_changed(self, event: str, item_id: str | None) -> None:
        """A case's formulas or Manning's n changed: its datasets derive anew, and the
        frames and colour ranges read with the old ones are forgotten."""
        if event not in ("changed", "added") or not item_id or item_id not in self.scene:
            return
        item = self.scene.get(item_id)
        if item.kind != "case":
            return
        raw = str(item.properties.get("path", ""))
        key = self._resolved.get(raw) or (str(Path(raw).expanduser().resolve()) if raw else "")
        case = self._cases.get(key)
        if case is None or self.derived_settings(key) == self._derived.get(key):
            return
        self._attach_derived(key, case)
        self.cache.drop_case(key)
        with self._samples_lock:
            self._samples.clear()
        self._auto_ranges.clear()

    def derived_problems(self, item_id: str) -> list[str]:
        """Why some of a case's formulas give nothing (in the datasets they fit none):
        a formula that does not read, or uses variables no output has."""
        from lagoonviz.core.data import derived

        case_item = self.case_item(item_id)
        if case_item is None:
            return []
        try:
            formulas = derived.parse_formulas(str(case_item.properties.get("formulas", "")))
        except ValueError as error:
            return [str(error)]
        case = self.case(case_item.id)
        if case is None:
            return []
        problems = []
        for name, _ in formulas:
            plans = [getattr(d, "_derived", None) for d in case.datasets.values()]
            plans = [p for p in plans if p is not None]
            if any(name in p.components for p in plans):
                continue
            why = next((p.problems[name] for p in plans if name in p.problems), None)
            problems.append(why or f"{name}: no output has its variables")
        return problems

    def error(self, item_id: str) -> str | None:
        """Why the case of an item could not be opened, if it could not."""
        case_item = self.case_item(item_id)
        if case_item is None:
            return "no case loaded"
        path = case_item.properties.get("path", "")
        if not path:
            return "no path"
        return self._errors.get(self._resolved.get(path, str(Path(path).expanduser().resolve())))

    def case_item(self, item_id: str) -> SceneItem | None:
        """The case item an item belongs to (itself for a case)."""
        if item_id not in self.scene:
            return None
        item = self.scene.get(item_id)
        if item.kind == "case":
            return item
        return self._case_item_for(str(item.properties.get("case", "")))

    def _case_item_for(self, ref: str) -> SceneItem | None:
        if ref and ref in self.scene and self.scene.get(ref).kind == "case":
            return self.scene.get(ref)
        for other in self.scene:  # unassigned items use the first case
            if other.kind == "case":
                return other
        return None

    def case(self, item_id: str) -> Case | None:
        case_item = self.case_item(item_id)
        if case_item is None or not case_item.properties.get("path"):
            return None
        try:
            return self.open(case_item.properties["path"])
        except (OSError, ValueError):
            return None

    def dataset(self, item_id: str) -> Dataset | None:
        """The dataset an item shows: its ``dataset`` property, else the one for its role."""
        item = self.scene.get(item_id)
        return self.dataset_for(item.kind, item.properties)

    def dataset_for(self, kind: str, properties: dict) -> Dataset | None:
        """The dataset an item of this kind with these properties would show."""
        case_item = self._case_item_for(str(properties.get("case", "")))
        if case_item is None or not case_item.properties.get("path"):
            return None
        try:
            case = self.open(case_item.properties["path"])
        except (OSError, ValueError):
            return None
        key = properties.get("dataset", "")
        if key and key in case.datasets:
            return case.datasets[key]
        role = DATA_KINDS.get(kind)
        if kind == "streamlines" and properties.get("lines_in") == "free surface":
            role = "free_surface"  # streamlines on the free surface use its output
        return case.by_role(role) if role else None

    def volume_case(self, prefer: str | None = None) -> str | None:
        """The case new slices, iso-surfaces and streamlines cut: ``prefer`` (a case
        item, or any item of it) if its case has a volume, else the first case
        that has one; None if none has."""
        cases = [item.id for item in self.scene if item.kind == "case"]
        if prefer:
            owner = self.case_item(prefer)
            if owner is not None:
                cases.insert(0, owner.id)
        for case_id in dict.fromkeys(cases):
            if self.dataset_for("slice", {"case": case_id}) is not None:
                return case_id
        return None

    # ------------------------------------------------------------- choices
    def dataset_choices(self, item_id: str) -> list[str]:
        """The case's datasets an item can show: slices and streamlines a volume,
        iso-surfaces and iso-volumes a volume or a surface (free surface, bed, ...)."""
        case = self.case(item_id)
        if case is None:
            return []
        kind = self.scene.get(item_id).kind if item_id in self.scene else ""
        if kind in ("slice", "streamlines", "volume", "iso_surface", "iso_volume"):
            return [k for k, d in case.datasets.items() if shows(kind, d)]
        return list(case.datasets)

    def addable(self, kind: str) -> list[tuple[str, str, str]]:
        """The data a new item of this kind can show, in every open case: (case item,
        dataset key, label), e.g. an iso-surface of NHFLOW's volume, of SFLOW's free
        surface or of its bed. The label names the case when several are open."""
        cases = [i for i in self.scene if i.kind == "case"]
        found = []
        for case_item in cases:
            try:
                case = self.open(case_item.properties["path"])
            except (KeyError, OSError, ValueError):
                continue
            for key, dataset in case.datasets.items():
                if not shows(kind, dataset):
                    continue
                label = dataset.label + (f" [{key}]" if key != dataset.key else "")
                if getattr(dataset, "earlier_run", False):
                    label += " (earlier run)"
                if len(cases) > 1:
                    label = f"{case_item.name}: {label}"
                found.append((case_item.id, key, label))
        return found

    def variable_choices(self, item_id: str) -> list[str]:
        dataset = self.dataset(item_id)
        return ["none", *dataset.variables()] if dataset else ["none"]

    def choices(self, item_id: str, provider: str) -> list[str]:
        if provider == "variables":
            return self.variable_choices(item_id)
        if provider == "datasets":
            return ["", *self.dataset_choices(item_id)]
        if provider == "cases":
            return ["", *[i.id for i in self.scene if i.kind == "case"]]
        if provider == "series":
            case = self.case(item_id)
            return ["", *case.series] if case else [""]
        if provider == "profiles":
            case = self.case(item_id)
            return ["", *case.profiles] if case else [""]
        if provider == "vectors":
            dataset = self.dataset(item_id)
            names = [n for n, c in dataset.components.items() if c == 3] if dataset else []
            return ["none", *names]
        raise ValueError(f"unknown choice provider {provider!r}")

    def labelled_choices(self, item_id: str, provider: str) -> list[tuple[str, str]]:
        """(label, value) pairs for a choice list, e.g. ("NHFLOW free surface", "nhflow_...")."""
        values = self.choices(item_id, provider)
        if provider == "datasets":
            case = self.case(item_id)
            dataset = self.dataset(item_id) if item_id in self.scene else None
            auto = f"automatic ({dataset.label})" if dataset is not None else "automatic"
            return [
                (
                    auto
                    if v == ""
                    else f"{case.datasets[v].label} [{v}]"
                    if case and v in case.datasets
                    else v,
                    v,
                )
                for v in values
            ]
        if provider == "cases":
            return [
                ("first case" if v == "" else f"{self.scene.get(v).name} [{v}]", v) for v in values
            ]
        if provider == "series":
            case = self.case(item_id)
            return [
                ("first file" if v == "" else case.series[v].label if case else v, v)
                for v in values
            ]
        if provider == "profiles":
            case = self.case(item_id)
            return [
                ("first" if v == "" else case.profiles[v].label if case else v, v) for v in values
            ]
        return [(v, v) for v in values]

    # ------------------------------------------------------------- time
    def print_interval(self) -> float | None:
        """Seconds between frames of the first case that knows it."""
        for item in self.scene:
            if item.kind == "case":
                dt = float(item.properties.get("print_interval", 0.0) or 0.0)
                if dt > 0:
                    return dt
                case = self.case(item.id)
                if case is not None and case.print_interval:
                    return case.print_interval
        return None

    def n_frames(self) -> int:
        counts = [
            d.last_step + 1
            for item in self.scene
            if item.kind in DATA_KINDS and (d := self.dataset(item.id)) is not None
        ]
        if not counts:
            counts = [
                c.n_frames
                for item in self.scene
                if item.kind == "case" and (c := self.case(item.id))
            ]
        return max(counts, default=0)

    def logged_times(self) -> np.ndarray | None:
        """Simulation times of the shown outputs from REEF3D's run log: all the times
        at which a shown surface or volume has a frame. None unless every one of them
        is in a run log (then the timeline goes by output numbers, as before)."""
        datasets = []
        for item_id in self.data_items():
            dataset = self.dataset(item_id)
            # beds and DIVEMesh's solids and bed are shown at any time: not part of it
            if dataset is not None and dataset.role not in ("bed", *CASE_ROLES):
                datasets.append(dataset)
        key = tuple((d.folder, d.key, len(d)) for d in datasets)
        if self._logged is not None and self._logged[0] == key:
            return self._logged[1]
        times = None
        if datasets and all(d.times is not None for d in datasets):
            times = np.unique(np.round(np.concatenate([d.times for d in datasets]), 9))
        self._logged = (key, times)
        return times

    def timeline(self) -> tuple[np.ndarray, np.ndarray, str]:
        """(times, frame numbers, unit) of the timeline's frames.

        With a run log: the logged simulation times, numbered from 0. Without: REEF3D's
        output numbers from the first with data, times = number x output interval (or
        the numbers themselves when the interval is not known)."""
        times = self.logged_times()
        if times is not None and times.size:
            return times, np.arange(times.size), "s"
        n = self.n_frames()
        if n == 0:
            return np.zeros(0), np.zeros(0, int), "s" if self.series_items() else "frame"
        steps = np.arange(min(self.first_output(), n - 1), n)
        dt = self.print_interval()
        return (steps * dt, steps, "s") if dt else (steps.astype(float), steps, "frame")

    def time_range(self) -> tuple[float, float, str]:
        """(start, end, unit) of the timeline: seconds if known, otherwise frames."""
        times = self.logged_times()
        if times is not None and times.size:
            return 0.0, float(times[-1]), "s"
        n = self.n_frames()
        dt = self.print_interval()
        if dt:
            return 0.0, max(n - 1, 0) * dt, "s"
        if n == 0 and self.series_items():
            return 0.0, self.series_end(), "s"  # only time series: plain seconds
        return 0.0, float(max(n - 1, 0)), "frame"

    def time_in_seconds(self) -> bool:
        """True if the scene time is in seconds, so plots can follow the timeline."""
        return self.time_range()[2] == "s"

    def frame_index(self, item_id: str, time: float | None = None) -> int | None:
        dataset = self.dataset(item_id)
        if dataset is None:
            return None
        time = self.scene.time if time is None else time
        if dataset.times is not None and self.logged_times() is not None:
            index = dataset.index_at_time(time)
            if index is None and dataset.role == "bed":
                return 0  # a bed is shown at any time: it rarely changes between runs
            return index
        dt = self.print_interval()
        if dataset.times is not None and dt:
            # its own times (DEM, written at its own interval) on a timeline in seconds
            index = dataset.index_at_time(time)
            return 0 if index is None and dataset.role == "bed" else index
        output = time / dt if dt else time
        # before this run's first output (e.g. outputs left from a run that was
        # restarted): nothing to show. One output of slack: mooring output starts at 1.
        if dataset.role != "bed" and round(output) < dataset.frames[0].step - 1:
            return None  # (a bed is shown at any time: it rarely changes between runs)
        return dataset.frame_index(output)

    def first_output(self, case_item_id: str | None = None) -> int:
        """First REEF3D output with data among the shown surfaces and volumes (beds
        aside), of one case or of all. 0 without data. The timeline starts here, so a
        run whose first outputs were overwritten by a restart never shows nothing."""
        firsts = []
        for item_id in self.data_items():
            if case_item_id is not None and self.case_item(item_id) is not self.scene.get(
                case_item_id
            ):
                continue
            dataset = self.dataset(item_id)
            if dataset is not None and dataset.role != "bed":
                firsts.append(dataset.frames[0].step)
        return min(firsts) if firsts else 0

    def first_time(self, case_item_id: str) -> float | None:
        """Timeline position of :meth:`first_output` for a case (None if it is 0)."""
        times = self.logged_times()
        if times is not None:
            return float(times[0]) if times.size and times[0] > 0 else None
        first = self.first_output(case_item_id)
        if not first:
            return None
        dt = self.print_interval()
        return first * dt if dt else float(first)

    # ------------------------------------------------------------- loading
    @staticmethod
    def arrays_for(item: SceneItem) -> tuple[str, ...]:
        arrays = set()
        # colours, contour lines, iso-surfaces, arrows
        for key in ("variable", "contour_by", "iso_by", "vectors_of", "streamlines_of"):
            variable = str(item.properties.get(key, "none"))
            if variable and variable != "none":
                arrays.add(variable.partition(".")[0])
        if item.kind in WATER_KINDS:
            # dry land is hidden: FNPF writes the depth, NHFLOW and SFLOW wetdry
            arrays.update(("eta", "depth", "wetdry"))
        if item.kind == "bathymetry":
            arrays.add("depth")  # still water depth: the level for vertical exaggeration
        if int(item.properties.get("trail_outputs", 0) or 0) > 0:
            arrays.add("id")  # trails follow each particle by its id
        return tuple(sorted(arrays))

    def frame_key(self, item_id: str, time: float | None = None) -> FrameKey | None:
        dataset = self.dataset(item_id)
        index = self.frame_index(item_id, time)
        case = self.case(item_id)
        if dataset is None or index is None or case is None:
            return None
        arrays = tuple(
            a for a in self.arrays_for(self.scene.get(item_id)) if a in dataset.components
        )
        item = self.scene.get(item_id)
        extra: tuple = ()
        if item.kind == "slice":
            axis = AXES.get(str(item.properties.get("normal", "y")), 1)
            extra = ("slice", axis, float(item.properties.get("position", 0.0)))
        elif item.kind == "iso_surface":
            variable = str(item.properties.get("iso_by", "none"))
            if variable == "none":
                return None  # nothing chosen yet
            extra = ("iso", variable, float(item.properties.get("iso_value", 0.0)))
        elif item.kind == "iso_volume":
            variable = str(item.properties.get("iso_by", "none"))
            if variable == "none":
                return None
            props = item.properties
            extra = (
                "isovol",
                variable,
                float(props.get("iso_min", 0.0)),
                float(props.get("iso_max", 1.0)),
            )
        elif item.kind == "streamlines":
            props = item.properties
            vector = str(props.get("streamlines_of", "none"))
            if vector == "none":
                return None
            extra = (
                "stream",
                vector,
                AXES.get(str(props.get("seed_normal", "y")), 1),
                float(props.get("seed_position", 0.0)),
                int(props.get("seeds", 60)),
                str(props.get("lines_in", "volume")),
            )
        elif item.kind == "amr" and not item.properties.get("patches", True):
            extra = ("levels", 0)  # the base grid alone, without the refined patches
        detail = self._detail(dataset) if not extra else 1
        return FrameKey(self._case_key(item_id), dataset.key, index, arrays, extra, detail)

    def _detail(self, dataset) -> int:
        """Every how many grid lines a surface is read: coarser only while the time
        moves in the streaming memory mode, and only surfaces that change with time
        (a bed stays as it is)."""
        if not self.moving or len(dataset) < 2:
            return 1
        from lagoonviz.core import memory

        points = memory.current().detail_points
        return dataset.detail_stride(points) if points else 1

    def _case_key(self, item_id: str) -> str:
        """The key of an item's case among the open cases: the path it was opened
        with, resolved (a LAGOON store opened on its own is that store, not the
        folder around it, which may hold other runs)."""
        case_item = self.case_item(item_id)
        assert case_item is not None
        raw = str(case_item.properties.get("path", ""))
        return self._resolved.get(raw) or str(Path(raw).expanduser().resolve())

    def cached(self, item_id: str, time: float | None = None) -> SurfaceMesh | None:
        key = self.frame_key(item_id, time)
        return self.cache.get(key) if key else None

    def load_key(self, key: FrameKey) -> SurfaceMesh:
        mesh = self.cache.get(key)
        if mesh is not None:
            return mesh
        case = self._cases.get(key.case) or self.open(key.case)
        dataset = case.datasets[key.dataset]
        if key.extra and key.extra[0] == "slice":
            mesh = dataset.read_slice(key.index, key.extra[1], key.extra[2], list(key.arrays))
        elif key.extra and key.extra[0] == "iso":
            mesh = dataset.read_iso(key.index, key.extra[1], key.extra[2], list(key.arrays))
        elif key.extra and key.extra[0] == "isovol":
            _, variable, low, high = key.extra
            mesh = dataset.read_iso_volume(key.index, variable, low, high, list(key.arrays))
        elif key.extra and key.extra[0] == "stream":
            _, vector, axis, position, count, lines_in = key.extra
            mesh = dataset.read_streamlines(
                key.index, vector, axis, position, count, list(key.arrays), lines_in
            )
        elif key.extra and key.extra[0] == "levels":
            mesh = dataset.read(key.index, list(key.arrays), max_level=key.extra[1])
        elif key.detail > 1:
            mesh = dataset.read(key.index, list(key.arrays), detail=key.detail)
        else:
            mesh = dataset.read(key.index, list(key.arrays))
            if dataset.surface_like and len(dataset) > 2:
                # the colour range's samples over the run, here in the loader thread
                for name in key.arrays:
                    if dataset.components.get(name) in (1, 3):
                        self._sampled_values(dataset, name)
        self.cache.put(key, mesh)
        return mesh

    def load(self, item_id: str, time: float | None = None) -> SurfaceMesh | None:
        """Read (or fetch from the cache) the mesh an item shows at a time. Blocking."""
        key = self.frame_key(item_id, time)
        return self.load_key(key) if key else None

    # ------------------------------------------------------------- display
    def value_range(self, item_id: str, mesh: SurfaceMesh) -> tuple[float, float]:
        """Colour range: fixed from the item, or automatic.

        The automatic range is found once and then kept, so colours stay comparable
        while playing. For surfaces it comes from a spread of frames over the run (the
        first frame alone is often calm water); otherwise from the frame shown.
        :meth:`Session.fit_range` fits it to the current frame instead.
        """
        item = self.scene.get(item_id)
        props = item.properties
        if not props.get("auto_range", True):
            return float(props.get("vmin", 0.0)), float(props.get("vmax", 1.0))
        low, high = self._range_of(item_id, str(props.get("variable", "none")), mesh)
        return scaled_range(low, high, props)

    def _range_mode(self, item_id: str) -> str:
        """How an item's automatic range is found (RANGE_FROM)."""
        if item_id not in self.scene:
            return RANGE_FROM[0]
        return str(self.scene.get(item_id).properties.get("range_from", RANGE_FROM[0]))

    def _range_of(self, item_id: str, variable: str, mesh: SurfaceMesh) -> tuple[float, float]:
        """The automatic range of a variable of an item, found once and kept.

        Values all alike (a flat bed at the start of a scour run) set no range: it is
        found again in the next frame. Items cut from a volume (slices, iso-surfaces:
        too costly to sample over the run) widen their range to the values of each
        frame shown, so a bed that scours later does not run out of colours."""
        dataset = self.dataset(item_id)
        mode = self._range_mode(item_id)
        key = (item_id, dataset.key if dataset else "", variable, mode)
        kept = self._auto_ranges.get(key)
        if kept is None:
            values = [mesh.scalar(variable)]
            if dataset is not None and dataset.surface_like and len(dataset) > 2:
                values += self._sampled_values(dataset, variable)
            joined = np.concatenate([v.ravel() for v in values])
            found = auto_range(joined, mode)
            if _spread(joined):
                self._auto_ranges[key] = found
            return found
        if dataset is not None and not dataset.surface_like:
            values = mesh.scalar(variable).ravel()
            if _spread(values):
                low, high = auto_range(values, mode)
                low, high = min(kept[0], low), max(kept[1], high)
                if mode == RANGE_FROM[0] and low < 0.0 < high:  # symmetric, as auto_range
                    low, high = -max(-low, high), max(-low, high)
                self._auto_ranges[key] = (low, high)
        return self._auto_ranges[key]

    def vector_arrows(
        self, item_id: str, mesh: SurfaceMesh, shown: DisplaySurface
    ) -> VectorArrows | None:
        """Arrows of an item's "vectors_of" variable at about "vector_count" points
        spread evenly over its surface as shown. The longest arrows (the variable's
        automatic range, kept from frame to frame) reach about the next arrow, times
        "vector_scale". Vertical parts are exaggerated like the surface; in 2D the
        arrows lie flat on top and show the horizontal part."""
        item = self.scene.get(item_id)
        props = item.properties
        name = str(props.get("vectors_of", "none"))
        vectors = mesh.point_data.get(name) if name != "none" else None
        if vectors is None or vectors.ndim != 2 or vectors.shape[1] != 3:
            return None
        if vectors.shape[0] != shown.positions.shape[0] or shown.triangles.size == 0:
            return None
        triangles = shown.triangles
        shown_triangles = triangles[
            ~((triangles[:, 0] == triangles[:, 1]) & (triangles[:, 1] == triangles[:, 2]))
        ]
        used = np.unique(shown_triangles)  # not on hidden dry land
        count = int(props.get("vector_count", 400))
        picked = pick_points(mesh.points, used, count)
        if not picked.size:
            return None
        spacing = grid_spacing(mesh.points[used], count)
        _, high = self._range_of(item_id, name, mesh)  # the speed of the longest arrows
        length = 0.9 * spacing * float(props.get("vector_scale", 1.0)) / max(high, 1e-12)
        direction = np.array(vectors[picked], dtype=np.float64)
        base = np.array(shown.positions[picked], dtype=np.float64)
        if self.scene.flat:
            direction[:, 2] = 0.0
            base[:, 2] = PLAN_MARKS - 0.5
        else:
            direction[:, 2] *= self.exaggeration(item_id)
            base[:, 2] += 2e-3 * float(max(np.ptp(shown.positions, axis=0).max(), 1e-9))
        segments = arrow_segments(base, direction, length)
        return VectorArrows(item_id, segments, shown.visible)

    def patch_outlines(
        self, item_id: str, mesh: SurfaceMesh, shown: DisplaySurface
    ) -> Trails | None:
        """The borders of an AMR surface's refined patches, or a slice's grid lines (the
        edges of the cells it cuts), on the surface as shown, lifted like contour lines
        (a slice's: off its plane to both sides); None when the item draws none."""
        item = self.scene.get(item_id)
        if item.kind == "amr":
            colour = str(item.properties.get("outlines", "dark"))
        elif "grid_lines" in item.properties:  # slices, surfaces, volumes
            colour = str(item.properties.get("grid_lines", "off"))
        else:
            return None
        if colour == "off" or shown.positions.shape[0] != mesh.n_points:
            return None
        outlines = mesh.outlines
        if outlines is None and item.kind not in ("amr", "slice"):
            outlines = self._grid_lines(item_id, mesh)  # made when first asked for
        return self._lines_on(item_id, outlines, shown, colour)

    def amr_grid_lines(
        self, item_id: str, mesh: SurfaceMesh, shown: DisplaySurface
    ) -> Trails | None:
        """An AMR surface's grid lines: the edges of the cells shown, of every level
        (drawn beside its patch outlines, so both can be shown in their own colours);
        None for other items or when off."""
        item = self.scene.get(item_id)
        if item.kind != "amr":
            return None
        colour = str(item.properties.get("grid_lines", "off"))
        if colour == "off" or shown.positions.shape[0] != mesh.n_points:
            return None
        return self._lines_on(item_id, self._grid_lines(item_id, mesh), shown, colour)

    def _lines_on(
        self, item_id: str, edges: np.ndarray | None, shown: DisplaySurface, colour: str
    ) -> Trails | None:
        """Edges (node pairs) drawn on a surface as shown, lifted off it."""
        if edges is None or not len(edges):
            return None
        item = self.scene.get(item_id)
        edges = edges.astype(np.int64)
        if item.kind != "slice" and shown.triangles.size:  # not over hidden dry land
            used = np.zeros(shown.positions.shape[0], bool)
            used[shown.triangles.ravel()] = True
            edges = edges[used[edges].all(axis=1)]
            if not len(edges):
                return None
        segments = np.array(shown.positions[edges], np.float32)
        span = np.ptp(shown.positions, axis=0)
        # just off the surface (the lines are its cells' edges): far enough not to
        # flicker into it, near enough not to float over it in a shallow case
        share = 2.5e-3 if item.kind == "slice" else 3e-4
        lift = 0.06 if self.scene.flat else share * float(max(span.max(), 1e-9))
        axis = AXES.get(str(item.properties.get("normal", "z")), 2) if item.kind == "slice" else 2
        if self.scene.flat or (item.kind in ("slice", "amr") and axis == 2):
            segments[..., 2] += lift
        elif item.kind == "slice":  # a vertical slice: on both of its sides
            other = segments.copy()
            segments[..., axis] += lift
            other[..., axis] -= lift
            segments = np.concatenate([segments, other])
        elif shown.normals is not None and len(shown.normals) == len(shown.positions):
            # off the surface along its normals (a volume's sides face sideways)
            segments += lift * np.asarray(shown.normals, np.float32)[edges]
        else:
            segments[..., 2] += lift
        return Trails(
            item_id, segments, shown.visible, "#ffffff" if colour == "white" else "#1e2833"
        )

    def _grid_lines(self, item_id: str, mesh: SurfaceMesh) -> np.ndarray | None:
        """The grid lines of a surface or a volume's outside: the edges of its
        triangles without the cells' diagonals (mesh.grid_edges), made once for the
        triangles of a dataset (they are shared by its frames) and only when shown."""
        from lagoonviz.core.data.mesh import grid_edges

        triangles = mesh.triangles
        if not triangles.size:
            return None
        dataset = self.dataset(item_id)
        across = dataset is None or dataset.surface_like
        key = (id(triangles), triangles.shape[0], mesh.n_points, across)
        cache = self.__dict__.setdefault("_grid_line_cache", {})
        edges = cache.get(key)
        if edges is None:
            edges = grid_edges(triangles, mesh.points, across=across)
            if len(cache) > 8:
                cache.clear()
            cache[key] = edges
        return edges

    def contour_lines(
        self, item_id: str, mesh: SurfaceMesh, shown: DisplaySurface
    ) -> ContourLines | None:
        """Contour lines of an item's "contour_by" variable on its surface as shown
        (exaggerated, or flat in 2D), lifted a little so the surface does not hide
        them. Levels are round numbers over the variable's automatic range, kept from
        frame to frame. None when the item draws none."""
        item = self.scene.get(item_id)
        variable = str(item.properties.get("contour_by", "none"))
        if variable == "none" or shown.triangles.size == 0:
            return None
        try:
            values = mesh.scalar(variable)
        except KeyError:
            return None
        if values is None or values.shape[0] != shown.positions.shape[0]:
            return None
        low, high = self._range_of(item_id, variable, mesh)
        levels = contour_levels(low, high, int(item.properties.get("contour_levels", 10)))
        segments = contour_segments(shown.positions, shown.triangles, values, levels)
        if segments.size:
            span = np.ptp(shown.positions, axis=0)
            lift = 0.05 if self.scene.flat else 2e-3 * float(max(span.max(), 1e-9))
            segments[..., 2] += lift
        return ContourLines(item_id, segments, levels, shown.visible)

    def _sampled_values(self, dataset, variable: str, samples: int = 5) -> list[np.ndarray]:
        """A variable's values in a few frames spread over a run (surfaces: cheap),
        kept: each frame's values thinned to at most 200 000, with its lowest and
        highest. Usually made in a loader thread already (load_key), not while the
        window waits."""
        key = (dataset.folder, dataset.key, variable)
        with self._samples_lock:
            found = self._samples.get(key)
        if found is not None:
            return found
        found = []
        array = variable.partition(".")[0]
        if array not in dataset.components:
            return found
        for index in np.unique(np.linspace(0, len(dataset) - 1, samples).round().astype(int)):
            try:
                values = dataset.sample_values(int(index), variable)
            except (OSError, ValueError, KeyError):
                continue
            if values is None:
                continue
            values = np.asarray(values).ravel()
            finite = values[np.isfinite(values)]
            if not finite.size:
                continue
            step = finite.size // 200_000 + 1
            found.append(np.concatenate([finite[::step], [finite.min(), finite.max()]]))
        with self._samples_lock:
            if len(self._samples) > 64:
                self._samples.clear()
            self._samples[key] = found
        return found

    def automatic_range(self, item_id: str) -> tuple[float, float] | None:
        """The automatic colour range of an item as shown (found on first display, or
        now from its current frame). None without data."""
        item = self.scene.get(item_id)
        variable = str(item.properties.get("variable", "none"))
        if variable == "none":
            return None
        dataset = self.dataset(item_id)
        key = (item_id, dataset.key if dataset else "", variable, self._range_mode(item_id))
        if key not in self._auto_ranges:
            mesh = self.load(item_id)
            if mesh is None:
                return None
            auto = item.properties.get("auto_range", True)
            item.properties["auto_range"] = True  # the automatic range, whatever is set
            try:
                found = self.value_range(item_id, mesh)
            except KeyError:
                return None
            finally:
                item.properties["auto_range"] = auto
            kept = self._auto_ranges.get(key)  # (values all alike: not kept)
            return found if kept is None else scaled_range(*kept, item.properties)
        return scaled_range(*self._auto_ranges[key], item.properties)

    def forget_range(self, item_id: str) -> None:
        for key in [k for k in self._auto_ranges if k[0] == item_id]:
            del self._auto_ranges[key]

    def still_water_level(self, item_id: str, mesh: SurfaceMesh) -> float:
        """Reference height for vertical exaggeration: the still water level.

        A free surface gives it as ``z - eta``, a bed as ``z + depth`` (REEF3D writes
        the still water depth with the bed); otherwise the case's free surface is asked.
        """
        dataset = self.dataset(item_id)
        key = (dataset.folder, dataset.key) if dataset else (item_id,)
        if key in self._still_water:
            return self._still_water[key]
        z = mesh.points[:, 2].astype(np.float64)
        eta = mesh.point_data.get("eta")
        depth = mesh.point_data.get("depth")
        if eta is not None and eta.ndim == 1:
            level = float(np.median(z - eta))
        elif depth is not None and depth.ndim == 1:
            level = float(np.median(z + depth))
        else:
            item = self.scene.get(item_id)
            if item.kind not in WATER_KINDS and self._has_free_surface(item_id):
                return self._water_reference(item_id)[0]  # not cached: may not be known yet
            level = float(np.median(z))
        self._still_water[key] = level
        return level

    def _has_free_surface(self, item_id: str) -> bool:
        case_item = self.case_item(item_id)
        return any(
            other.kind in WATER_KINDS and self.case_item(other.id) is case_item
            for other in self.scene
        )

    def placed(self, item_id: str, points: np.ndarray) -> np.ndarray:
        """Points of an object item where the view draws them: exaggerated about the
        case's water level, or flat on the item's layer in 2D."""
        item = self.scene.get(item_id)
        points = np.array(points, dtype=np.float32, copy=True)
        exaggeration = self.exaggeration(item_id)
        if exaggeration != 1.0 and len(points):
            level = self.case_water_level(item_id)
            points[:, 2] = level + exaggeration * (points[:, 2] - level)
        if self.scene.flat:
            points[:, 2] = PLAN_LAYERS.get(item.kind, 1.0)
        return points

    def particle_trails(self, item_id: str, mesh: SurfaceMesh | None = None) -> Trails | None:
        """Paths of an item's particles (DEM elements: their centres) over its
        "trail_outputs" outputs before the one shown, joined by their ``id``; None
        when the item draws none or the output has no ids."""
        item = self.scene.get(item_id)
        count = int(item.properties.get("trail_outputs", 0) or 0)
        key = self.frame_key(item_id)
        if count <= 0 or key is None:
            return None
        mesh = mesh if mesh is not None else self.load_key(key)
        if mesh is None or "id" not in mesh.point_data:
            return None
        parts = []
        newer = self._centres(key, mesh)
        for index in range(key.index - 1, max(key.index - count, 0) - 1, -1):
            older_key = FrameKey(key.case, key.dataset, index, key.arrays, key.extra)
            older = self._centres(older_key)
            if older is None:
                break
            _, a, b = np.intersect1d(older[0], newer[0], assume_unique=True, return_indices=True)
            if a.size:
                parts.append(np.stack([older[1][a], newer[1][b]], axis=1))
            newer = older
        case_item = self.case_item(item_id)
        visible = item.visible and (case_item is None or case_item.visible)
        if not parts:
            return Trails(item_id, np.zeros((0, 2, 3), np.float32), visible)
        segments = np.concatenate(parts).reshape(-1, 3)
        segments = self.placed(item_id, segments).reshape(-1, 2, 3)
        return Trails(item_id, segments, visible, self._plain_color(item))

    def _centres(self, key: FrameKey, mesh: SurfaceMesh | None = None):
        """(ids, centre of each id's points) of an output, kept for the next frames."""
        found = self._trail_centres.get(key)
        if found is not None:
            return found
        if mesh is None:
            try:
                mesh = self.load_key(key)
            except (OSError, ValueError, KeyError, IndexError):
                return None
        ids = mesh.point_data.get("id")
        if ids is None or not len(ids):
            found = (np.zeros(0, np.int64), np.zeros((0, 3), np.float64))
        else:
            unique, inverse = np.unique(np.asarray(ids).astype(np.int64), return_inverse=True)
            counts = np.bincount(inverse, minlength=len(unique)).astype(np.float64)
            points = np.asarray(mesh.points, np.float64)
            centres = (
                np.stack(
                    [
                        np.bincount(inverse, weights=points[:, c], minlength=len(unique))
                        for c in range(3)
                    ],
                    axis=1,
                )
                / counts[:, None]
            )
            found = (unique, centres)
        if len(self._trail_centres) > 2048:
            self._trail_centres.clear()
        self._trail_centres[key] = found
        return found

    def display(self, item_id: str, mesh: SurfaceMesh) -> DisplaySurface:
        """Turn a loaded mesh into arrays for a renderer, following the item's settings."""
        item = self.scene.get(item_id)
        props = item.properties
        positions = mesh.points
        normals = mesh.with_normals().normals
        exaggeration = self.exaggeration(item_id)
        if item.kind in WATER_KINDS:
            self.still_water_level(item_id, mesh)  # once per dataset; gauges use it
        if exaggeration != 1.0:
            if item.kind in (*WATER_KINDS, "bathymetry"):
                level = self.still_water_level(item_id, mesh)
            else:  # bodies, mooring lines, volumes, slices: about the case's water level
                level = self.case_water_level(item_id)
            positions = positions.copy()
            positions[:, 2] = level + exaggeration * (positions[:, 2] - level)
            normals = exaggerate_normals(normals, exaggeration)
        if self.scene.flat:
            # 2D: flat, stacked by kind so the free surface lies over the bed; the
            # normals stay those of the shapes, so the light falls as in the 3D view
            # (waves and the bed's relief shaded, the water's glint)
            positions = np.array(positions, dtype=np.float32, copy=True)
            positions[:, 2] = self._plan_layer(item_id)
        variable = str(props.get("variable", "none"))
        try:
            values = mesh.scalar(variable)
        except KeyError:
            values = None
        vmin, vmax = self.value_range(item_id, mesh) if values is not None else (0.0, 1.0)
        index = self.frame_index(item_id)
        case_item = self.case_item(item_id)
        triangles = mesh.triangles
        if item.kind in WATER_KINDS and props.get("hide_dry", True):
            triangles = self._wet_triangles(item_id, mesh)
        elif item.kind == "seastate" and props.get("hide_dry", True) and "wet" in mesh.point_data:
            from lagoonviz.core.data.seastate import wet_triangles

            triangles = wet_triangles(triangles, mesh.point_data["wet"])
        elif item.kind == "bathymetry" and props.get("hide_wet", False):
            level = self.still_water_level(item_id, mesh)
            triangles = self._kept(item_id, mesh, _dry_triangles(mesh, level))
        return DisplaySurface(
            item_id=item_id,
            positions=np.ascontiguousarray(positions, dtype=np.float32),
            triangles=triangles,
            normals=normals,
            values=values,
            vmin=vmin,
            vmax=vmax,
            colormap=str(props.get("colormap", "viridis")),
            opacity=float(props.get("opacity", 1.0)),
            visible=item.visible and (case_item is None or case_item.visible),
            variable=variable if values is not None else "none",
            frame=index if index is not None else 0,
            color=self._plain_color(item),
            segments=mesh.segments,
            line_width=float(props.get("line_width", 3.0)),
            point_size=float(props.get("point_size", 4.0)) if item.kind in POINT_KINDS else None,
            material=str(props.get("material", "") or ""),
        )

    def _plan_layer(self, item_id: str) -> float:
        """The height of an item's flat layer in 2D: its kind's, and for an iso-line
        or iso-patch on a surface just over that surface (the free surface's over the
        water, the bed's over the bed)."""
        item = self.scene.get(item_id)
        if item.kind in ("iso_surface", "iso_volume"):
            dataset = self.dataset(item_id)
            if dataset is not None and dataset.kind != "UnstructuredGrid":
                kind = ROLES.get(dataset.role, "")
                return PLAN_LAYERS.get(kind, 1.0) + 0.1
        return PLAN_LAYERS.get(item.kind, 1.0)

    # ------------------------------------------------------------- time series
    def series_items(self) -> list[str]:
        return [item.id for item in self.scene if item.kind == "series"]

    def series_file(self, item_id: str) -> SeriesFile | None:
        case = self.case(item_id)
        if case is None or not case.series:
            return None
        source = str(self.scene.get(item_id).properties.get("source", ""))
        if source in case.series:
            return case.series[source]
        return None if source else next(iter(case.series.values()))

    def series(self, item_id: str) -> TimeSeries | None:
        found = self.series_file(item_id)
        if found is None:
            return None
        try:
            return found.read()
        except (OSError, ValueError):
            return None

    def traces(self, item_id: str) -> list[Trace]:
        """The lines an item plots (its columns of its file)."""
        table = self.series(item_id)
        if table is None:
            return []
        names = parse_columns(str(self.scene.get(item_id).properties.get("columns", "")), table)
        return [
            Trace(item_id, name, table.units.get(name, ""), table.time, table.columns[name])
            for name in names
        ]

    def gauge_marks(self, item_id: str, palette: list[str] | None = None) -> GaugeMarks | None:
        """Wave gauges of a series item for the 3D view (flat on top in 2D)."""
        return self._planned(self._gauge_marks(item_id, palette))

    def profile_marks(self, item_id: str, palette: list[str] | None = None) -> GaugeMarks | None:
        """Free surface line lines for the 3D view (flat on top in 2D)."""
        return self._planned(self._profile_marks(item_id, palette))

    def _planned(self, marks: GaugeMarks | None) -> GaugeMarks | None:
        if marks is None or not self.scene.flat:
            return marks
        staffs = np.array(marks.staffs, dtype=np.float32, copy=True)
        dots = np.array(marks.dots, dtype=np.float32, copy=True)
        staffs[..., 2] = PLAN_MARKS
        if dots.size:
            dots[:, 2] = np.where(np.isfinite(dots[:, 2]), PLAN_MARKS, np.nan)
        return GaugeMarks(marks.item_id, marks.names, staffs, dots, marks.colors, marks.visible)

    def _gauge_marks(self, item_id: str, palette: list[str] | None = None) -> GaugeMarks | None:
        """Positions of the gauges a series item plots, at the scene time (None if the
        file has no gauge positions)."""
        if item_id not in self.scene or self.scene.get(item_id).kind != "series":
            return None
        item = self.scene.get(item_id)
        table = self.series(item_id)
        if table is None or not table.locations:
            return None
        palette = list(palette or SERIES_PALETTES["light"])
        level, exaggeration = self._water_reference(item_id)
        time = self.scene.time if self.time_in_seconds() else None
        names, staffs, dots, colors = [], [], [], []
        for index, trace in enumerate(self.traces(item_id)):
            where = table.locations.get(trace.name)
            if where is None:
                continue
            x, y = where[0], where[1]
            finite = trace.values[np.isfinite(trace.values)]
            amplitude = float(np.abs(finite).max()) if finite.size else 0.0
            # gauges give the elevation above still water; a few solvers write heights
            offset = level
            if finite.size and level and abs(float(np.median(finite)) - level) < 0.5 * abs(level):
                offset = 0.0
                amplitude = float(np.abs(finite - level).max())
            half = 2.0 * (amplitude if amplitude > 0 else 1.0) * exaggeration
            staffs.append(((x, y, level - half), (x, y, level + half)))
            eta = table.value_at(trace.name, time) if time is not None else float("nan")
            dots.append((x, y, level + exaggeration * (eta + offset - level)))
            names.append(trace.name)
            colors.append(palette[index % len(palette)])
        if not names:
            return None
        case_item = self.case_item(item_id)
        visible = bool(
            item.visible
            and item.properties.get("gauges_3d", True)
            and (case_item is None or case_item.visible)
        )
        return GaugeMarks(
            item_id,
            names,
            np.array(staffs, np.float32).reshape(-1, 2, 3),
            np.array(dots, np.float32).reshape(-1, 3),
            colors,
            visible,
        )

    @staticmethod
    def _plain_color(item: SceneItem) -> str:
        props = item.properties
        default = kind_spec(item.kind).properties.get("colormap")
        return plain_color(
            item.kind,
            props.get("material"),
            props.get("colormap"),
            default.default if default is not None else None,
        )

    def _wet_triangles(self, item_id: str, mesh: SurfaceMesh) -> np.ndarray:
        """:func:`_wet_triangles`, reusing the last array while the shoreline stays put
        (the renderer then skips re-uploading the triangle list): at once when the
        depth is the very array of before (a store's still water depth, read once)."""
        depth = mesh.point_data.get("depth")
        if depth is None:
            depth = mesh.point_data.get("wetdry")
        previous = self._wet.get(item_id)
        if (
            depth is not None
            and previous is not None
            and previous[0] is mesh.triangles
            and len(previous) > 2
            and previous[2] is depth
        ):
            return previous[1]
        triangles = self._kept(item_id, mesh, _wet_triangles(mesh))
        if depth is not None and triangles is not mesh.triangles:
            self._wet[item_id] = (mesh.triangles, triangles, depth)
        return triangles

    def _kept(self, item_id: str, mesh: SurfaceMesh, triangles: np.ndarray) -> np.ndarray:
        """A filtered triangle list, or the last one if it has not changed."""
        if triangles is mesh.triangles:
            return triangles
        previous = self._wet.get(item_id)
        if (
            previous is not None
            and previous[0] is mesh.triangles
            and np.array_equal(previous[1], triangles)
        ):
            return previous[1]
        self._wet[item_id] = (mesh.triangles, triangles)
        return triangles

    def exaggeration(self, item_id: str) -> float:
        """The vertical exaggeration an item is drawn with: its case's, when the case
        has "Same for all items" on (the default), otherwise the item's own."""
        item = self.scene.get(item_id)
        case_item = self.case_item(item_id)
        if case_item is not None and case_item.properties.get("sync_exaggeration", True):
            return float(case_item.properties.get("exaggeration", 1.0))
        return float(item.properties.get("exaggeration", 1.0))

    def case_water_level(self, item_id: str) -> float:
        """The still water level of an item's case, from its free surface (z - eta) or
        its bed (z + depth); read from the first frame once if not known yet."""
        case = self.case(item_id)
        if case is None:
            return 0.0
        for role in ("free_surface", "amr", "bed"):
            dataset = case.by_role(role)
            if dataset is None:
                continue
            key = (dataset.folder, dataset.key)
            if key in self._still_water:
                return self._still_water[key]
            arrays = [a for a in ("eta", "depth") if a in dataset.components]
            if not arrays:
                continue
            try:
                mesh = dataset.read(0, arrays)
            except (OSError, ValueError):
                continue
            z = mesh.points[:, 2].astype(np.float64)
            if "eta" in mesh.point_data:
                level = float(np.median(z - mesh.point_data["eta"]))
            else:
                level = float(np.median(z + mesh.point_data["depth"]))
            self._still_water[key] = level
            return level
        return 0.0

    def _water_reference(self, item_id: str) -> tuple[float, float]:
        """Still water level and vertical exaggeration for gauges and profiles."""
        return self.case_water_level(item_id), self.exaggeration(item_id)

    # ------------------------------------------------------------- free surface lines
    def profile_items(self) -> list[str]:
        return [item.id for item in self.scene if item.kind == "profile"]

    def profile_set(self, item_id: str) -> ProfileSet | None:
        case = self.case(item_id)
        if case is None or not case.profiles:
            return None
        source = str(self.scene.get(item_id).properties.get("source", ""))
        if source in case.profiles:
            return case.profiles[source]
        return None if source else next(iter(case.profiles.values()))

    def profile(self, item_id: str, time: float | None = None) -> Profile | None:
        """The profile file shown at a time (the scene time in seconds by default)."""
        found = self.profile_set(item_id)
        if found is None or not found.paths:
            return None
        time = self.scene.time if time is None else time
        try:
            return found.at(float(time))
        except (OSError, ValueError):
            return None

    def profile_lines(self, item_id: str, time: float | None = None) -> list[ProfileLine]:
        """The lines an item shows, in the order of its ``lines`` property."""
        profile = self.profile(item_id, time)
        if profile is None:
            return []
        text = str(self.scene.get(item_id).properties.get("lines", ""))
        wanted = [int(t) for t in text.replace(";", ",").split(",") if t.strip().isdigit()]
        numbers = [n for n in wanted if n in profile.lines] or sorted(profile.lines)
        return [profile.lines[n] for n in numbers]

    def _profile_marks(self, item_id: str, palette: list[str] | None = None) -> GaugeMarks | None:
        """A shown profile as lines in 3D, at the still water level of the case's free
        surface (exaggerated like it), in the colours of its plot lines. The lines
        travel as segments in ``staffs``; ``dots`` stays empty."""
        if item_id not in self.scene:
            return None
        item = self.scene.get(item_id)
        lines = self.profile_lines(item_id)
        if not lines:
            return None
        palette = list(palette or SERIES_PALETTES["light"])
        level, exaggeration = self._water_reference(item_id)
        segments, colors, names = [], [], []
        for index, line in enumerate(lines):
            ok = np.isfinite(line.x) & np.isfinite(line.eta)
            x, eta = line.x[ok], line.eta[ok]
            if x.size < 2:
                continue
            # elevation above still water, or (some solvers) the surface height
            absolute = level and abs(float(np.median(eta)) - level) < 0.5 * abs(level)
            z = level + exaggeration * ((eta - level) if absolute else eta)
            across = np.full_like(x, line.y)
            points = np.stack([x, across, z] if line.axis == "x" else [across, x, z], axis=1)
            segments.append(np.stack([points[:-1], points[1:]], axis=1))
            colors += [palette[index % len(palette)]] * (x.size - 1)
            names.append(f"line {line.number}")
        if not segments:
            return None
        case_item = self.case_item(item_id)
        visible = bool(
            item.visible
            and item.properties.get("profile_3d", True)
            and (case_item is None or case_item.visible)
        )
        return GaugeMarks(
            item_id,
            names,
            np.concatenate(segments).astype(np.float32),
            np.zeros((0, 3), np.float32),
            colors,
            visible,
        )

    def series_end(self) -> float:
        """Last time of the shown time series (0 without any)."""
        ends = [
            float(table.time[-1])
            for item_id in self.series_items()
            if (table := self.series(item_id)) is not None and len(table)
        ]
        return max(ends, default=0.0)

    def iso_default(self, properties: dict) -> tuple[str, float] | None:
        """A first iso-surface for a volume: REEF3D's level set ``phi`` at 0 (the free
        surface of CFD runs), else the middle of the velocity (or the first variable).
        On a surface (free surface, bed) an iso-line: of the variable it is coloured
        by, at 0 when that changes sign (eta: the still water line), else its middle."""
        dataset = self.dataset_for("iso_surface", properties)
        if dataset is None:
            return None
        if dataset.kind != "UnstructuredGrid":
            found = self._surface_range(dataset)
            if found is None:
                return None
            variable, low, high = found
            middle = 0.0 if low < 0.0 < high else 0.5 * (low + high)
            return variable, float(f"{middle:.3g}")
        names = dataset.variables()
        if "phi" in names:
            return "phi", 0.0
        variable = "velocity" if "velocity" in names else (names[0] if names else None)
        if variable is None:
            return None
        mesh = dataset.read(0, [variable])  # the volume's outer surface: a fair sample
        low, high = auto_range(mesh.scalar(variable))
        middle = 0.5 * (low + high) if not (low < 0.0 < high) else 0.5 * high
        return variable, float(f"{middle:.3g}")

    def iso_volume_default(self, properties: dict) -> tuple[str, float, float] | None:
        """A first iso-volume: where REEF3D's level set phi >= 0 (the water of CFD
        runs), else the upper half of the velocity (or the first variable)."""
        dataset = self.dataset_for("iso_volume", properties)
        if dataset is None:
            return None
        if dataset.kind != "UnstructuredGrid":  # a surface: its upper part
            found = self._surface_range(dataset)
            if found is None:
                return None
            variable, low, high = found
            start = 0.0 if low < 0.0 < high else 0.5 * (low + high)
            return variable, float(f"{start:.3g}"), float(f"{high:.3g}")
        names = dataset.variables()
        variable = ("phi" if "phi" in names else "velocity" if "velocity" in names else None) or (
            names[0] if names else None
        )
        if variable is None:
            return None
        mesh = dataset.read(0, [variable])  # the volume's outer surface: a fair sample
        low, high = auto_range(mesh.scalar(variable))
        top = float(f"{high:.3g}") if high > 0 else high
        if variable == "phi":
            return variable, 0.0, max(top, 1.0)
        return variable, float(f"{0.5 * (low + high):.3g}"), top

    def _surface_range(self, dataset) -> tuple[str, float, float] | None:
        """A surface's first variable for iso-lines and its range in the first frame
        (its own default, eta on a free surface)."""
        names = dataset.variables()
        variable = dataset.default_variable()
        if variable not in names:
            variable = names[0] if names else None
        if variable is None or variable == "none":
            return None
        mesh = dataset.read(0, [variable])
        low, high = auto_range(mesh.scalar(variable))
        return variable, float(low), float(high)

    def domain_center(self, kind: str, properties: dict) -> tuple[float, float, float] | None:
        """Centre of the first frame of the dataset an item would show (to place slices)."""
        dataset = self.dataset_for(kind, properties)
        if dataset is None:
            return None
        low, high = dataset.bounds(0)
        return tuple(float(v) for v in (low + high) / 2)  # type: ignore[return-value]

    def probe_trace(self, probe: Probe) -> Trace | None:
        """The probe's time series: eta under the orange dot on a free surface (the
        coloured variable on other surfaces) over all frames, at the nearest grid
        point. Times as on the timeline. None when the item has nothing to plot."""
        item = self.scene.get(probe.item_id)
        dataset = self.dataset(probe.item_id)
        if dataset is None or item.kind in ("slice", "iso_surface", "iso_volume", "streamlines"):
            return None
        variable = (
            "eta" if item.kind in WATER_KINDS and "eta" in dataset.components else probe.variable
        )
        if not variable or variable == "none":
            return None
        where, values = dataset.point_series(variable, probe.point[0], probe.point[1])
        if dataset.times is not None:
            times = np.asarray(dataset.times, dtype=np.float64)
        else:
            dt = self.print_interval() or 1.0
            times = np.array([frame.step * dt for frame in dataset.frames], dtype=np.float64)
        unit = "m" if variable in ("eta", "elevation", "depth", "WL") else ""
        name = f"{item.name} {variable} at x {where[0]:.1f}, y {where[1]:.1f}"
        return Trace("probe", name, unit, times, np.asarray(values, dtype=np.float64))

    def probe(
        self, item_id: str, triangle: int, weights: tuple[float, float, float]
    ) -> Probe | None:
        """Values where the mouse hit an item's surface (current frame, from the cache)."""
        key = self._shown_key(item_id)
        mesh = self.cache.get(key) if key else None
        if mesh is None or not 0 <= triangle < mesh.n_triangles:
            return None
        point, values = mesh.sample(triangle, weights)
        item = self.scene.get(item_id)
        variable = str(item.properties.get("variable", "none"))
        return Probe(item_id, item.name, tuple(float(v) for v in point), values, variable)

    def _shown_key(self, item_id: str) -> FrameKey | None:
        return self.frame_key(item_id)

    def data_items(self) -> list[str]:
        """Ids of the scene items that show data."""
        return [item.id for item in self.scene if item.kind in DATA_KINDS]

    def shown(self, item_id: str) -> bool:
        """Whether an item is to be drawn: ticked, its case ticked, and drawn in the
        current mode. Frames of other items are not read (a large volume switched
        off costs nothing while playing)."""
        item = self.scene.get(item_id)
        if not item.visible or not self.drawn(item_id):
            return False
        case_item = self.case_item(item_id)
        return case_item is None or case_item.visible

    def shown_items(self) -> list[str]:
        """The data items to be drawn (:meth:`shown`)."""
        return [i for i in self.data_items() if self.shown(i)]

    # ------------------------------------------------------------- modes
    def item_mode(self, item_id: str) -> str:
        """Where an item belongs: "setup" (Case set-up, the DIVEMesh grid), "both"
        (cases, DIVEMesh's solids and bed: the geometry of the set-up and of the
        results) or "results" (what a run wrote)."""
        item = self.scene.get(item_id)
        if item.kind in MODE_KINDS:  # the set-up items and geo.dat's points
            return "setup"
        if item.kind in ("case", "basemap"):
            return "both"
        if item.kind in DATA_KINDS:
            dataset = self.dataset(item_id)
            if dataset is not None and dataset.solver == "DIVEMesh":
                return "both"
        return "results"

    def drawn(self, item_id: str) -> bool:
        """Whether the current mode draws an item at all (its check box still decides):
        the set-up leaves the results out; the results draw set-up items when ticked."""
        if self.scene.mode != "setup":
            return True
        return self.item_mode(item_id) != "results"

    # ------------------------------------------------------------- the Earth
    def georef(self, item_id: str):
        """The georeference of an item's case (a ``lagoon_format.geo.Georef``): its
        georef.json, else its store's or run log's; None without one."""
        return self.backgrounds.georef(item_id)

    # ------------------------------------------------------------- the set-up
    def case_folder(self, item_id: str) -> Path | None:
        """The case folder of an item's case (also of a case with no outputs)."""
        case = self.case(item_id)
        if case is not None:
            return Path(case.path)
        case_item = self.case_item(item_id)
        if case_item is None or not case_item.properties.get("path"):
            return None
        folder = Path(str(case_item.properties["path"])).expanduser()
        return folder.parent if folder.is_file() else folder

    def case_setup(self, item_id: str, create: bool = False) -> CaseSetup | None:
        """The set-up of the case an item belongs to: what its input files say (None
        if it has none, unless ``create``: then empty files, to be written)."""
        folder = self.case_folder(item_id)
        if folder is None:
            return None
        key = str(folder.resolve())
        setup = self._setups.get(key)
        if setup is None:
            if not has_inputs(folder) and not create:
                return None
            setup = read_setup(folder)
            case = self.case(item_id)
            if setup.domain is None and case is not None:
                extent = _results_extent(case)
                if extent is not None:
                    setup.use_domain(*extent, "the results")
            self._setups[key] = setup
        setup.refresh()  # after edits: the facts of the files as they are now
        return setup

    def setup_items(self) -> list[str]:
        """Ids of the set-up items: "Case set-up" and "DIVEMesh grid"."""
        return [item.id for item in self.scene if item.kind in SETUP_KINDS]

    def grid(self, item_id: str) -> Grid | None:
        """DIVEMesh's grid of an item's case: the one DIVEMesh wrote, else the uniform
        grid control.txt sets up (None: neither)."""
        folder = self.case_folder(item_id)
        if folder is None:
            return None
        setup = self.case_setup(item_id)
        if setup is not None and setup.curvgrid is not None:
            return None  # DIVEMesh's is the logical box: the set-up draws the curvilinear grid
        grid = read_preview(folder)
        if grid is None:
            if setup is None or setup.stretched:
                return None
            setup.refresh()
            grid = uniform_grid(setup.domain, setup.cells)
        if grid is not None and setup is not None:
            # FNPF and NHFLOW: REEF3D scales the heights to σ and stretches them from the
            # bed to the still water: drawn there, as ParaView shows the results
            setup.refresh()
            if setup.solver in SIGMA_SOLVERS and setup.domain is not None:
                grid = sigma_grid(grid, setup.domain[4], setup.domain[5])
        return grid

    def _grid_marks(self, item_id: str) -> SetupMarks | None:
        item = self.scene.get(item_id)
        grid = self.grid(item_id)
        if grid is None:
            return None
        props = item.properties
        lines = [("#6f7d8f", grid_lines(grid, str(props.get("faces")), int(props.get("every", 0))))]
        folder = self.case_folder(item_id)
        if props.get("partition", True) and folder is not None and grid.source == "DIVEMesh":
            planes = read_partition(folder)
            if len(planes) and grid.heights is not None:  # a σ-grid: its heights as drawn
                z0, z1, bottom, top = grid.heights
                planes = np.array(planes, np.float64, copy=True)
                planes[..., 2] = bottom + (planes[..., 2] - z0) * (top - bottom) / (z1 - z0)
            if len(planes):
                lines.append(("#d9822b", planes))
        setup = self.case_setup(item_id)
        level = self._setup_water_level(item_id, setup) if setup is not None else 0.0
        placed = [(c, self._placed_marks(item_id, s, level)) for c, s in lines if len(s)]
        return SetupMarks(item_id, placed, [], [], [], item.visible)

    def _placed_marks(self, item_id: str, array: np.ndarray, level: float) -> np.ndarray:
        """Set-up signs where the view draws them: exaggerated about the water level,
        flat in 2D."""
        out = np.array(array, np.float32, copy=True)
        if not out.size:
            return out
        flat_view = out.reshape(-1, 3)
        exaggeration = self.exaggeration(item_id)
        if exaggeration != 1.0:
            flat_view[:, 2] = level + exaggeration * (flat_view[:, 2] - level)
        if self.scene.flat:
            flat_view[:, 2] = PLAN_MARKS
        return out

    def setup_signature(self, item_id: str) -> tuple | None:
        """What a "Case set-up" item's drawing depends on: when it changes, draw again."""
        item = self.scene.get(item_id)
        setup = self.case_setup(item_id)
        grid_file = None
        if item.kind == "grid":
            folder = self.case_folder(item_id)
            preview = folder / PREVIEW if folder is not None else None
            if preview is not None and preview.is_file():
                grid_file = (preview.stat().st_mtime_ns, preview.stat().st_size)
            if setup is None and grid_file is None:
                return None
        elif setup is None:
            return None
        if setup is not None:
            setup.refresh()
        return (
            tuple(sorted(item.properties.items())),
            item.visible,
            grid_file,
            id(setup),
            setup.control.revision if setup else None,
            setup.ctrl.revision if setup else None,
            setup.files_revision if setup else None,
            setup.domain if setup else None,
            self.exaggeration(item_id),
            self._setup_water_level(item_id, setup) if setup else 0.0,
            self.scene.flat,
            self.highlight,
        )

    def setup_marks(self, item_id: str) -> SetupMarks | None:
        """The signs a "Case set-up" item draws: the groups it has switched on, placed
        as the case's other items are (exaggerated, flat in 2D)."""
        item = self.scene.get(item_id)
        if item.kind == "grid":
            return self._grid_marks(item_id)
        setup = self.case_setup(item_id)
        if setup is None:
            return None
        props = item.properties
        if item.kind == "corridor":  # the river corridor: its banks, points and grid
            shown = []
            for mark in setup.marks():
                if mark.group == "corridor":
                    if not props.get("show_points", True):
                        mark = replace(mark, points=np.zeros((0, 3)))
                    shown.append(mark)
                elif mark.group == "corridor_grid" and props.get("show_grid", True):
                    shown.append(mark)
        elif item.kind == "multiblock":  # the blocks: outlines, points, cells, quality, rules
            shown = []
            for mark in setup.marks():
                if mark.group == "multiblock":
                    if not props.get("show_points", True):
                        mark = replace(mark, points=np.zeros((0, 3)))
                    shown.append(mark)
                elif mark.group == "multiblock_grid" and props.get("show_grid", True):
                    shown.append(mark)
                elif mark.group == "multiblock_quality" and props.get("show_quality", True):
                    shown.append(mark)
                elif mark.group == "multiblock_issues" and props.get("show_issues", True):
                    shown.append(mark)
        else:
            shown = [
                m
                for m in setup.marks()
                if m.group not in ITEM_GROUPS and props.get(f"show_{m.group}", True)
            ]
        lines: dict[str, list[np.ndarray]] = {}
        points: dict[str, list[np.ndarray]] = {}
        faces: dict[str, list[np.ndarray]] = {}
        labels = []
        level = self._setup_water_level(item_id, setup)
        flat = self.scene.flat

        def place(array: np.ndarray) -> np.ndarray:
            return self._placed_marks(item_id, array, level)

        chosen = self._highlighted(item_id, setup)
        for mark in shown:
            if chosen is not None and mark is chosen:
                continue  # drawn over the others, below
            if len(mark.segments):
                lines.setdefault(mark.color, []).append(place(mark.segments))
            if len(mark.points):
                points.setdefault(mark.color, []).append(place(mark.points))
            if mark.faces is not None and len(mark.faces) and not flat:
                faces.setdefault(mark.color, []).append(place(mark.faces))
            if props.get("labels", True):
                for text, where in mark.labels:
                    if text:
                        p = place(np.asarray([where], float))[0]
                        labels.append((text, (float(p[0]), float(p[1]), float(p[2])), mark.color))
        highlight = None
        if chosen is not None:
            highlight = (place(chosen.segments), place(chosen.points))
        return SetupMarks(
            item_id,
            [(c, np.concatenate(v)) for c, v in lines.items()],
            [(c, np.concatenate(v)) for c, v in points.items()],
            [(c, np.concatenate(v)) for c, v in faces.items()],
            labels,
            item.visible,
            highlight,
        )

    def _highlighted(self, item_id: str, setup: CaseSetup):
        """The mark of the command selected in the set-up editor, if this item shows it."""
        if self.highlight is None:
            return None
        case_id, file, code, occurrence = self.highlight
        case_item = self.case_item(item_id)
        if case_item is None or case_item.id != case_id:
            return None
        for mark in setup.marks():
            if (mark.file, mark.code, mark.occurrence) == (file, code, occurrence):
                return mark
        return None

    def set_highlight(self, case_id: str, file: str, code: str, occurrence: int = 0) -> None:
        """Show which signs a command of an input file makes (the set-up editor's
        selection); ``code`` "" for none. View state, not an undo step."""
        chosen = (case_id, file, code, occurrence) if code else None
        if chosen != self.highlight:
            self.highlight = chosen
            self.scene.highlight_changed(case_id)

    def _setup_water_level(self, item_id: str, setup: CaseSetup) -> float:
        """The level the set-up is exaggerated about: the results' (as the other items),
        else the still water level of ctrl.txt (F 60)."""
        case = self.case(item_id)
        if case is not None and not case.setup_only:
            return self.case_water_level(item_id)
        return setup.water_level if setup.water_level is not None else 0.0


def _results_extent(case: Case) -> tuple[np.ndarray, np.ndarray] | None:
    """The box the shown results fill (free surface, bed, volume: first output)."""
    lows, highs = [], []
    for role, dataset in case.main_datasets().items():
        if role not in ("free_surface", "bed", "volume", "amr"):
            continue
        try:
            low, high = dataset.bounds(0)
        except (OSError, ValueError, KeyError, IndexError):
            continue
        lows.append(np.asarray(low, float))
        highs.append(np.asarray(high, float))
    if not lows:
        return None
    low, high = np.min(lows, axis=0), np.max(highs, axis=0)
    if not (high[:2] > low[:2]).all():
        return None
    if high[2] <= low[2]:
        high = high.copy()
        high[2] = low[2] + 1e-3 * max(high[0] - low[0], high[1] - low[1])
    return low, high


def shows(kind: str, dataset) -> bool:
    """Whether a new item of this kind can show this dataset: slices, streamlines and
    volumes a volume; iso-surfaces and iso-volumes a volume or a surface (free
    surface, bed, AMR: iso-lines, iso-patches); a free surface, bed or AMR surface
    the outputs of its role (a bed: DIVEMesh's too)."""
    volume = dataset.kind == "UnstructuredGrid"
    if kind in ("slice", "streamlines", "volume"):
        return volume
    if kind in ("iso_surface", "iso_volume"):
        return volume or getattr(dataset, "surface_like", False) or dataset.role == "amr"
    roles = {"free_surface": ("free_surface",), "bathymetry": ("bed", "topo"), "amr": ("amr",)}
    return dataset.role in roles.get(kind, ())


def _wet_triangles(mesh: SurfaceMesh) -> np.ndarray:
    """The free surface without dry land, so the bed shows there.

    FNPF writes the water ``depth``: a triangle goes when all its corners have none.
    NHFLOW and SFLOW write ``wetdry``, at a point the share of its four cells that are
    wet (0, 0.25, ... 1): a triangle goes when its corners are wet by half or less on
    average, so the surface ends at the wet cells' edge (a wet cell keeps both its
    triangles, a dry one next to it loses them). A triangle that goes stays in the
    index list as a degenerate triangle, so the list keeps its length as the
    shoreline moves and the renderer only updates its buffer.
    """
    depth = mesh.point_data.get("depth")
    wetdry = mesh.point_data.get("wetdry")
    if depth is not None and depth.ndim == 1:
        dry = depth <= 0.0
        if not dry.any():
            return mesh.triangles
        gone = dry[mesh.triangles].all(axis=1)
    elif wetdry is not None and wetdry.ndim == 1:
        wet = np.asarray(wetdry, np.float32)
        if bool((wet >= 1.0).all()):
            return mesh.triangles
        corners = wet[mesh.triangles]
        gone = ~(corners.sum(axis=1) > 1.5)  # mean > 0.5 (NaN: gone)
    else:
        return mesh.triangles
    if not gone.any():
        return mesh.triangles
    triangles = mesh.triangles.copy()
    triangles[gone] = 0
    return triangles


def _dry_triangles(mesh: SurfaceMesh, level: float) -> np.ndarray:
    """The bed without what lies under still water, so only dry land shows.

    Under water means a still water ``depth`` above zero, as REEF3D writes it with the
    bed; without it, below the still water level. A triangle goes when all its corners
    are under water, so the shoreline stays closed. Gone triangles stay degenerate in
    the list (see :func:`_wet_triangles`).
    """
    depth = mesh.point_data.get("depth")
    if depth is not None and depth.ndim == 1:
        wet = depth > 0.0
    else:
        wet = mesh.points[:, 2] < level
    gone = wet[mesh.triangles].all(axis=1)
    if not gone.any():
        return mesh.triangles
    triangles = mesh.triangles.copy()
    triangles[gone] = 0
    return triangles


def _spread(values: np.ndarray | None) -> bool:
    """Whether the values differ (an automatic range can be found from them)."""
    if values is None or values.size == 0:
        return False
    finite = values[np.isfinite(values)]
    return bool(finite.size) and float(finite.max() - finite.min()) >= 1e-12


def scaled_range(low: float, high: float, props: dict) -> tuple[float, float]:
    """A range's min and max times an item's "range_percent" (100 %: as found;
    95 %: 0.95 x min to 0.95 x max)."""
    factor = float(props.get("range_percent", 100.0)) / 100.0
    if factor == 1.0:
        return low, high
    return low * factor, high * factor


#: how the automatic colour range is found: the 1st to the 99th percentile (symmetric
#: around zero when the values change sign: still water white in the diverging maps,
#: the colours strong), or the lowest to the highest value (as ParaView)
RANGE_FROM = ("1st - 99th percentile", "lowest to highest")


def auto_range(values: np.ndarray | None, mode: str = RANGE_FROM[1]) -> tuple[float, float]:
    """The automatic range of values (see RANGE_FROM): by default the lowest to the
    highest finite value, as ParaView scales its colours."""
    if values is None or values.size == 0:
        return 0.0, 1.0
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return 0.0, 1.0
    if mode == RANGE_FROM[0]:
        sample = finite if finite.size <= 200_000 else finite[:: finite.size // 200_000 + 1]
        low, high = (float(v) for v in np.percentile(sample, [1.0, 99.0]))
        if low < 0.0 < high:
            bound = max(-low, high)
            low, high = -bound, bound
    else:
        low, high = float(finite.min()), float(finite.max())
    if high - low < 1e-12:
        low, high = low - 0.5, high + 0.5
    return low, high


# ----------------------------------------------------------------- background
#: ``callback(key, mesh, error)``; called in a worker thread
LoadCallback = Callable[[FrameKey, SurfaceMesh | None, BaseException | None], None]


class BackgroundLoader:
    """Read frames in worker threads. Requests for the same frame are shared.

    A request can name its ``owner`` (a view's item): when the owner asks for another
    frame, the frame it asked for before is dropped if no other owner needs it and
    reading has not started. Dragging the timeline over hundreds of frames thus reads
    only the frame where the slider is, not every frame it passed (they used to queue
    up, and playing or the view waited for all of them).
    """

    def __init__(self, manager: DataManager, workers: int = 2):
        self.manager = manager
        self.workers = workers
        self._pool = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="lagoon-io")
        self._pending: dict[FrameKey, Future] = {}
        self._wants: dict[object, FrameKey] = {}  # owner -> the frame it waits for
        self._holders: dict[FrameKey, set] = {}  # frame -> owners waiting for it
        self._lock = threading.Lock()

    def set_workers(self, workers: int) -> None:
        """Read with this many threads from now on (frames being read finish)."""
        if workers == self.workers:
            return
        old = self._pool
        with self._lock:
            self._pool = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="lagoon-io")
            self.workers = workers
        old.shutdown(wait=False)

    def want(self, owner: object, key: FrameKey | None) -> None:
        """``owner`` now needs ``key`` (None: nothing). Its earlier frame is dropped
        if nobody else needs it and it has not started loading."""
        drop = None
        with self._lock:
            old = self._wants.pop(owner, None)
            if key is not None:
                self._wants[owner] = key
                self._holders.setdefault(key, set()).add(owner)
            if old is not None and old != key:
                holders = self._holders.get(old)
                if holders is not None:
                    holders.discard(owner)
                    if not holders:
                        del self._holders[old]
                        drop = self._pending.get(old)
        if drop is not None:
            drop.cancel()  # outside the lock: cancelling runs _done, which takes it

    def request(
        self, key: FrameKey, callback: LoadCallback | None = None, owner: object = None
    ) -> Future:
        if owner is not None:
            self.want(owner, key)
        with self._lock:
            future = self._pending.get(key)
            new = future is None
            if new:
                future = self._pool.submit(self.manager.load_key, key)
                self._pending[key] = future
        if new:
            # outside the lock: a future that is already done runs the callback at
            # once, in this thread, and _done takes the lock (this deadlocked)
            future.add_done_callback(lambda f, k=key: self._done(k, f))
        if callback is not None:

            def deliver(f: Future, k: FrameKey = key) -> None:
                if f.cancelled():
                    return  # nobody wants it any more
                error = f.exception()
                callback(k, None if error else f.result(), error)

            future.add_done_callback(deliver)
        return future

    def _done(self, key: FrameKey, future: Future) -> None:
        with self._lock:
            if self._pending.get(key) is future:
                del self._pending[key]
            if not self._holders.get(key, True):
                self._holders.pop(key, None)

    def prefetch(
        self,
        item_ids: list[str],
        time: float,
        frames_ahead: int = 4,
        times: np.ndarray | None = None,
    ) -> int:
        """Queue the next frames of the given items (used while playing): at the next
        ``times`` of the timeline if given, otherwise every output interval. Never
        more than the frame cache holds besides the frame shown (frames read further
        ahead would push out each other before being shown). Returns how many
        time steps were asked for."""
        manager = self.manager
        if times is not None and len(times):
            start = int(np.searchsorted(times, time + 1e-9, side="right"))
            upcoming = [float(t) for t in times[start : start + frames_ahead]]
        else:
            dt = manager.print_interval() or 1.0
            upcoming = [time + ahead * dt for ahead in range(1, frames_ahead + 1)]
        if not upcoming:
            return 0
        keys = [manager.frame_key(item_id, upcoming[0]) for item_id in item_ids]
        step = sum(manager.cache.size_hint(k) or 0 for k in keys if k is not None)
        if step:
            fits = int(manager.cache.max_bytes // step) - 2  # the shown frame, one spare
            upcoming = upcoming[: max(1, fits)]
        for when in upcoming:
            for item_id in item_ids:
                key = manager.frame_key(item_id, when)
                if key is not None and key not in manager.cache:
                    self.request(key)
        return len(upcoming)

    @property
    def busy(self) -> int:
        with self._lock:
            return len(self._pending)

    def shutdown(self) -> None:
        self._pool.shutdown(wait=False, cancel_futures=True)
