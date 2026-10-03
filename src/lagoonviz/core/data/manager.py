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

from lagoonviz.core.data.mesh import SurfaceMesh, exaggerate_normals
from lagoonviz.core.data.reef3d import ROLES, Case, Dataset, open_case
from lagoonviz.core.data.series import SeriesFile, TimeSeries, parse_columns
from lagoonviz.core.scene import Scene, SceneItem

#: item kinds that show a dataset, and the dataset role they use by default
DATA_KINDS = {kind: role for role, kind in ROLES.items()} | {"slice": "volume"}
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


@dataclass
class Trace:
    """One line of a time-series plot."""

    item_id: str
    name: str
    unit: str
    time: np.ndarray
    values: np.ndarray


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


class DataManager:
    """Opens the cases of a scene and loads the frames its items show."""

    def __init__(self, scene: Scene, cache: FrameCache | None = None):
        self.scene = scene
        self.cache = cache or FrameCache()
        self._cases: dict[str, Case] = {}
        self._errors: dict[str, str] = {}
        self._resolved: dict[str, str] = {}
        self._auto_ranges: dict[tuple, tuple[float, float]] = {}
        self._still_water: dict[tuple, float] = {}
        self._lock = threading.Lock()

    # ------------------------------------------------------------- cases
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

    def time_range(self) -> tuple[float, float, str]:
        """(start, end, unit) of the timeline: seconds if known, otherwise frames."""
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
        dt = self.print_interval()
        return dataset.frame_index(time / dt if dt else time)

    # ------------------------------------------------------------- loading
    @staticmethod
    def arrays_for(item: SceneItem) -> tuple[str, ...]:
        variable = str(item.properties.get("variable", "none"))
        arrays = {variable.partition(".")[0]} if variable and variable != "none" else set()
        if item.kind == "free_surface":
            arrays.add("eta")
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

        The automatic range is taken from the first frame shown and then kept, so
        colours stay comparable while playing. :meth:`Session.fit_range` refits it.
        """
        item = self.scene.get(item_id)
        props = item.properties
        if not props.get("auto_range", True):
            return float(props.get("vmin", 0.0)), float(props.get("vmax", 1.0))
        variable = str(props.get("variable", "none"))
        dataset = self.dataset(item_id)
        key = (item_id, dataset.key if dataset else "", variable)
        if key not in self._auto_ranges:
            self._auto_ranges[key] = auto_range(mesh.scalar(variable))
        return self._auto_ranges[key]

    def forget_range(self, item_id: str) -> None:
        for key in [k for k in self._auto_ranges if k[0] == item_id]:
            del self._auto_ranges[key]

    def still_water_level(self, item_id: str, mesh: SurfaceMesh) -> float:
        """Reference height for vertical exaggeration: the surface height where eta = 0."""
        dataset = self.dataset(item_id)
        key = (dataset.folder, dataset.key) if dataset else (item_id,)
        if key not in self._still_water:
            z = mesh.points[:, 2].astype(np.float64)
            eta = mesh.point_data.get("eta")
            level = np.median(z - eta) if eta is not None and eta.ndim == 1 else np.median(z)
            self._still_water[key] = float(level)
        return self._still_water[key]

    def display(self, item_id: str, mesh: SurfaceMesh) -> DisplaySurface:
        """Turn a loaded mesh into arrays for a renderer, following the item's settings."""
        item = self.scene.get(item_id)
        props = item.properties
        positions = mesh.points
        normals = mesh.with_normals().normals
        exaggeration = float(props.get("exaggeration", 1.0))
        if exaggeration != 1.0:
            level = self.still_water_level(item_id, mesh)
            positions = positions.copy()
            positions[:, 2] = level + exaggeration * (positions[:, 2] - level)
            normals = exaggerate_normals(normals, exaggeration)
        variable = str(props.get("variable", "none"))
        try:
            values = mesh.scalar(variable)
        except KeyError:
            values = None
        vmin, vmax = self.value_range(item_id, mesh) if values is not None else (0.0, 1.0)
        index = self.frame_index(item_id)
        case_item = self.case_item(item_id)
        return DisplaySurface(
            item_id=item_id,
            positions=np.ascontiguousarray(positions, dtype=np.float32),
            triangles=mesh.triangles,
            normals=normals,
            values=values,
            vmin=vmin,
            vmax=vmax,
            colormap=str(props.get("colormap", "viridis")),
            opacity=float(props.get("opacity", 1.0)),
            visible=item.visible and (case_item is None or case_item.visible),
            variable=variable if values is not None else "none",
            frame=index if index is not None else 0,
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

    def series_end(self) -> float:
        """Last time of the shown time series (0 without any)."""
        ends = [
            float(table.time[-1])
            for item_id in self.series_items()
            if (table := self.series(item_id)) is not None and len(table)
        ]
        return max(ends, default=0.0)

    def domain_center(self, kind: str, properties: dict) -> tuple[float, float, float] | None:
        """Centre of the first frame of the dataset an item would show (to place slices)."""
        dataset = self.dataset_for(kind, properties)
        if dataset is None:
            return None
        low, high = dataset.bounds(0)
        return tuple(float(v) for v in (low + high) / 2)  # type: ignore[return-value]

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
    """Read frames in worker threads. Requests for the same frame are shared."""

    def __init__(self, manager: DataManager, workers: int = 2):
        self.manager = manager
        self._pool = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="lagoon-io")
        self._pending: dict[FrameKey, Future] = {}
        self._lock = threading.Lock()

    def request(self, key: FrameKey, callback: LoadCallback | None = None) -> Future:
        with self._lock:
            future = self._pending.get(key)
            if future is None:
                future = self._pool.submit(self.manager.load_key, key)
                self._pending[key] = future
                future.add_done_callback(lambda _f, k=key: self._done(k))
        if callback is not None:

            def deliver(f: Future, k: FrameKey = key) -> None:
                error = f.exception()
                callback(k, None if error else f.result(), error)

            future.add_done_callback(deliver)
        return future

    def _done(self, key: FrameKey) -> None:
        with self._lock:
            self._pending.pop(key, None)

    def prefetch(self, item_ids: list[str], time: float, frames_ahead: int = 4) -> None:
        """Queue the next frames of the given items (used while playing)."""
        dt = self.manager.print_interval() or 1.0
        for ahead in range(1, frames_ahead + 1):
            for item_id in item_ids:
                key = self.manager.frame_key(item_id, time + ahead * dt)
                if key is not None and key not in self.manager.cache:
                    self.request(key)

    @property
    def busy(self) -> int:
        with self._lock:
            return len(self._pending)

    def shutdown(self) -> None:
        self._pool.shutdown(wait=False, cancel_futures=True)
