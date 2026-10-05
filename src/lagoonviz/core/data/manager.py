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
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from lagoonviz.core.colormaps import SERIES_PALETTES, plain_color
from lagoonviz.core.contours import contour_levels, contour_segments
from lagoonviz.core.data.mesh import SurfaceMesh, exaggerate_normals
from lagoonviz.core.data.profiles import Profile, ProfileLine, ProfileSet
from lagoonviz.core.data.reef3d import ROLES, Case, Dataset, open_case
from lagoonviz.core.data.series import SeriesFile, TimeSeries, parse_columns
from lagoonviz.core.kinds import kind_spec
from lagoonviz.core.scene import Scene, SceneItem
from lagoonviz.core.vectors import arrow_segments, grid_spacing, pick_points

#: item kinds that show a dataset, and the dataset role they use by default
DATA_KINDS = {kind: role for role, kind in ROLES.items()} | {
    "slice": "volume",
    "iso_surface": "volume",
    "streamlines": "volume",
}
#: 2D: heights of the flat layers (the free surface over objects over the bed)
PLAN_LAYERS = {
    "bathymetry": 0.0,
    "volume": 0.5,
    "slice": 0.5,
    "iso_surface": 0.5,
    "streamlines": 2.5,
    "structure": 1.0,
    "body": 1.5,
    "dem": 1.5,
    "ice": 1.5,
    "rods": 1.5,
    "particles": 2.2,
    "mooring": 2.0,
    "free_surface": 3.0,
    "amr": 3.0,
}
#: kinds drawn as the free surface: the still water level, dry land, the probe
WATER_KINDS = ("free_surface", "amr")
#: 2D: gauges and wave profiles on top of everything
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


class FrameCache:
    """Least-recently-used cache of meshes with a memory budget."""

    def __init__(self, max_bytes: int = 1 << 30):
        self.max_bytes = max_bytes
        self._items: OrderedDict[FrameKey, SurfaceMesh] = OrderedDict()
        self._bytes = 0
        self._lock = threading.Lock()
        self.hits = 0
        self.misses = 0

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

    def put(self, key: FrameKey, mesh: SurfaceMesh) -> None:
        with self._lock:
            if key in self._items:
                self._bytes -= self.size_of(self._items.pop(key))
            self._items[key] = mesh
            self._bytes += self.size_of(mesh)
            while self._bytes > self.max_bytes and len(self._items) > 1:
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
        #: particle trails: ids and their centres of each output read for them
        self._trail_centres: dict[FrameKey, tuple[np.ndarray, np.ndarray]] = {}
        self._still_water: dict[tuple, float] = {}
        self._wet: dict[str, tuple[np.ndarray, np.ndarray]] = {}  # item -> (base, wet)
        self._logged: tuple[tuple, np.ndarray | None] | None = None
        self._lock = threading.Lock()

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
            return case

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

    # ------------------------------------------------------------- choices
    def dataset_choices(self, item_id: str) -> list[str]:
        case = self.case(item_id)
        return list(case.datasets) if case else []

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
            if dataset is not None and dataset.role != "bed":
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
            arrays.update(("eta", "depth"))  # depth: dry land is hidden (FNPF writes it)
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
        return FrameKey(str(case.path), dataset.key, index, arrays, extra)

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
        elif key.extra and key.extra[0] == "stream":
            _, vector, axis, position, count, lines_in = key.extra
            mesh = dataset.read_streamlines(
                key.index, vector, axis, position, count, list(key.arrays), lines_in
            )
        else:
            mesh = dataset.read(key.index, list(key.arrays))
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
        return self._range_of(item_id, str(props.get("variable", "none")), mesh)

    def _range_of(self, item_id: str, variable: str, mesh: SurfaceMesh) -> tuple[float, float]:
        """The automatic range of a variable of an item, found once and kept."""
        dataset = self.dataset(item_id)
        key = (item_id, dataset.key if dataset else "", variable)
        if key not in self._auto_ranges:
            values = [mesh.scalar(variable)]
            if dataset is not None and dataset.surface_like and len(dataset) > 2:
                values += self._sampled_values(dataset, variable)
            self._auto_ranges[key] = auto_range(np.concatenate([v.ravel() for v in values]))
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
        """The borders of an AMR surface's refined patches on the surface as shown,
        lifted like contour lines; None when the item draws none."""
        item = self.scene.get(item_id)
        colour = str(item.properties.get("outlines", "dark"))
        if item.kind != "amr" or colour == "off" or mesh.outlines is None:
            return None
        if not len(mesh.outlines) or shown.positions.shape[0] != mesh.n_points:
            return None
        segments = np.array(shown.positions[mesh.outlines.astype(np.int64)], np.float32)
        span = np.ptp(shown.positions, axis=0)
        segments[..., 2] += 0.06 if self.scene.flat else 2.5e-3 * float(max(span.max(), 1e-9))
        return Trails(
            item_id, segments, shown.visible, "#ffffff" if colour == "white" else "#1e2833"
        )

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

    @staticmethod
    def _sampled_values(dataset, variable: str, samples: int = 5) -> list[np.ndarray]:
        """A variable's values in a few frames spread over a run (surfaces: cheap)."""
        found = []
        array = variable.partition(".")[0]
        if array not in dataset.components:
            return found
        for index in np.unique(np.linspace(0, len(dataset) - 1, samples).round().astype(int)):
            try:
                found.append(dataset.read(int(index), [array]).scalar(variable))
            except (OSError, ValueError, KeyError):
                continue
        return found

    def automatic_range(self, item_id: str) -> tuple[float, float] | None:
        """The automatic colour range of an item as shown (found on first display, or
        now from its current frame). None without data."""
        item = self.scene.get(item_id)
        variable = str(item.properties.get("variable", "none"))
        if variable == "none":
            return None
        dataset = self.dataset(item_id)
        key = (item_id, dataset.key if dataset else "", variable)
        if key not in self._auto_ranges:
            mesh = self.load(item_id)
            if mesh is None:
                return None
            auto = item.properties.get("auto_range", True)
            item.properties["auto_range"] = True  # the automatic range, whatever is set
            try:
                self.value_range(item_id, mesh)
            except KeyError:
                return None
            finally:
                item.properties["auto_range"] = auto
        return self._auto_ranges.get(key)

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
            # 2D: flat, stacked by kind so the free surface lies over the bed
            positions = np.array(positions, dtype=np.float32, copy=True)
            positions[:, 2] = PLAN_LAYERS.get(item.kind, 1.0)
            normals = np.tile(np.array([0.0, 0.0, 1.0], np.float32), (len(positions), 1))
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
            point_size=float(props.get("point_size", 4.0)) if item.kind == "particles" else None,
        )

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
        """Wave profile lines for the 3D view (flat on top in 2D)."""
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
        (the renderer then skips re-uploading the triangle list)."""
        return self._kept(item_id, mesh, _wet_triangles(mesh))

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

    # ------------------------------------------------------------- wave profiles
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
            points = np.stack([x, np.full_like(x, line.y), z], axis=1)
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
        surface of CFD runs), else the middle of the velocity (or the first variable)."""
        dataset = self.dataset_for("iso_surface", properties)
        if dataset is None or dataset.kind != "UnstructuredGrid":
            return None
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
        if dataset is None or item.kind in ("slice", "iso_surface", "streamlines"):
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


def _wet_triangles(mesh: SurfaceMesh) -> np.ndarray:
    """The free surface without dry land, so the bed shows there.

    A triangle goes when all its corners have no water depth. It stays in the index
    list as a degenerate triangle, so the list keeps its length as the shoreline moves
    and the renderer only updates its buffer.
    """
    depth = mesh.point_data.get("depth")
    if depth is None or depth.ndim != 1:
        return mesh.triangles
    dry = depth <= 0.0
    if not dry.any():
        return mesh.triangles
    gone = dry[mesh.triangles].all(axis=1)
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


def auto_range(values: np.ndarray | None) -> tuple[float, float]:
    """1st to 99th percentile; symmetric around zero when the values change sign."""
    if values is None or values.size == 0:
        return 0.0, 1.0
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return 0.0, 1.0
    sample = finite if finite.size <= 200_000 else finite[:: finite.size // 200_000 + 1]
    low, high = (float(v) for v in np.percentile(sample, [1.0, 99.0]))
    if low < 0.0 < high:
        bound = max(-low, high)
        low, high = -bound, bound
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
        self._pool = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="lagoon-io")
        self._pending: dict[FrameKey, Future] = {}
        self._wants: dict[object, FrameKey] = {}  # owner -> the frame it waits for
        self._holders: dict[FrameKey, set] = {}  # frame -> owners waiting for it
        self._lock = threading.Lock()

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
    ) -> None:
        """Queue the next frames of the given items (used while playing): at the next
        ``times`` of the timeline if given, otherwise every output interval."""
        if times is not None and len(times):
            start = int(np.searchsorted(times, time + 1e-9, side="right"))
            upcoming = [float(t) for t in times[start : start + frames_ahead]]
        else:
            dt = self.manager.print_interval() or 1.0
            upcoming = [time + ahead * dt for ahead in range(1, frames_ahead + 1)]
        for when in upcoming:
            for item_id in item_ids:
                key = self.manager.frame_key(item_id, when)
                if key is not None and key not in self.manager.cache:
                    self.request(key)

    @property
    def busy(self) -> int:
        with self._lock:
            return len(self._pending)

    def shutdown(self) -> None:
        self._pool.shutdown(wait=False, cancel_futures=True)
