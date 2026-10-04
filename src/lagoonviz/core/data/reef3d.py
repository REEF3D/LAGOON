# SPDX-License-Identifier: GPL-3.0-or-later
"""REEF3D cases: find the output of a run and read it frame by frame.

A REEF3D run writes one folder per kind of output next to ``ctrl.txt``::

    REEF3D_NHFLOW_VTU/       REEF3D-NHFLOW-00000012.pvtu  + one .vtu per MPI rank
    REEF3D_NHFLOW_VTP_FSF/   REEF3D-NHFLOW-FSF-00000012.pvtp  (free surface)
    REEF3D_NHFLOW_VTP_BED/   REEF3D-NHFLOW-BED-00000012.pvtp  (bed)
    REEF3D_FNPF_VTU/ ...     REEF3D_CFD_VTU/ ...

:func:`open_case` accepts the case folder, one of these output folders, or a single
file. Each output folder becomes a :class:`Dataset`; its frames are the numbered
files, merged across MPI ranks into one surface.
"""

from __future__ import annotations

import logging
import re
import threading
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from lagoon_format.run import Output, Run, RunLog, read_case

from lagoonviz.core.data.mesh import (
    CutPlan,
    PieceMerger,
    SurfaceMesh,
    VolumePlan,
    orient_by_field,
    plan_cut,
    plan_level,
    scalar_values,
    triangulate,
    tubes,
    variable_names,
)
from lagoonviz.core.data.profiles import ProfileSet, find_profiles
from lagoonviz.core.data.series import SeriesFile, find_series
from lagoonviz.core.data.vtkxml import (
    legacy_point_arrays,
    read_header,
    read_legacy,
    read_parallel_header,
    read_piece,
    read_time_value,
    split_name,
)

log = logging.getLogger(__name__)

SOLVERS = ("CFD", "NHFLOW", "FNPF", "SFLOW", "POTENTIAL")

#: dataset roles and the scene item kind that shows them
ROLES = {
    "free_surface": "free_surface",
    "bed": "bathymetry",
    "volume": "volume",
    "body": "body",
    "mooring": "mooring",
    "structure": "structure",
    "dem": "dem",
    "ice": "ice",
    "rods": "rods",
    "particles": "particles",
}

_FOLDER = re.compile(r"^REEF3D_(?P<solver>[A-Z0-9]+)_(?P<what>VTU|VTP(?:_(?P<tag>[A-Z0-9_]+))?)$")
_SUFFIXES = (".pvtu", ".pvtp", ".vtu", ".vtp", ".vtk")
#: folders of objects in the flow: floating bodies, mooring lines, porous or forced
#: structures, DEM elements (REEF3D_DEM_VTP, written by the DEM for any solver)
_OBJECT_FOLDER = re.compile(
    r"^REEF3D_(?:(?P<solver>[A-Z0-9]+)_)?"
    r"(?P<what>6DOF_VTP|6DOF_Mooring|VRANS_VTP|FORCING_VTP|DEM_VTP|ICE|RodTree"
    r"|CPM_Particle|Particles)$"
)
_OBJECT_ROLES = {
    "6DOF_VTP": "body",
    "6DOF_Mooring": "mooring",
    "VRANS_VTP": "structure",
    "FORCING_VTP": "structure",
    "DEM_VTP": "dem",
    "ICE": "ice",  # FNPF ice floes
    "RodTree": "rods",  # rod trees (corals, vegetation): lines with a radius
    "CPM_Particle": "particles",  # CFD's CPM sediment particles (one file per MPI rank)
    "Particles": "particles",  # NHFLOW's Lagrangian particles
}
#: object folders written like the main output: per output a .pvtp and a file per
#: MPI rank, or one serial file per output (not one file per object)
_PARALLEL_OBJECTS = ("CPM_Particle", "Particles")
#: roles whose files are drawn as they are (no merging of MPI pieces)
PLAIN_ROLES = ("body", "mooring", "structure", "dem", "ice", "rods", "particles")
#: roles whose outputs have their own interval: their times come from the files
OWN_TIME_ROLES = ("dem", "ice", "rods", "particles")
#: roles whose cell data (one value per floe or rod) is shown on their points
CELL_DATA_ROLES = ("ice", "rods")
#: folders whose files are numbered one per output (not one per object)
_NUMBERED_FOLDERS = ("DEM_VTP", "_ICE", "RodTree")
_PIECE_STEP = re.compile(r"^(?P<stem>.+?)-(?P<piece>\d+)-(?P<step>\d+)$")
_SERIAL_STEP = re.compile(r"^(?P<stem>.+?)-(?P<step>\d+)$")


def _role(what: str, tag: str | None, kind: str) -> str:
    if tag in ("FSF", "FS"):
        return "free_surface"
    if tag == "BED":
        return "bed"
    if what == "VTU" or kind == "UnstructuredGrid":
        return "volume"
    return (tag or "surface").lower()


ROLE_LABELS = {
    "free_surface": "free surface",
    "bed": "bed",
    "volume": "volume",
    "body": "bodies",
    "mooring": "mooring lines",
    "structure": "structures",
    "dem": "DEM elements",
    "ice": "ice floes",
    "rods": "rod trees",
    "particles": "particles",
}


@dataclass
class Frame:
    """One output time of one dataset: the parallel file or the single serial file."""

    step: int
    path: Path  # .pvtu/.pvtp or the serial file
    pieces: list[Path]


class Dataset:
    """One kind of output of a run (free surface, bed, volume) over all its frames."""

    def __init__(self, key: str, solver: str, role: str, frames: list[Frame], folder: Path):
        if not frames:
            raise ValueError(f"{folder}: no frames")
        self.key = key
        self.solver = solver
        self.role = role
        self.frames = frames
        self.folder = folder
        self.kind, self.components = self._describe(frames[0])
        self._merger = PieceMerger("xy" if self.surface_like else "xyz")
        self._triangles = np.zeros((0, 3), np.uint32)
        self._volume_plan: VolumePlan | None = None
        self._step_array: np.ndarray | None = None
        self._cells: dict[int, tuple] = {}  # piece number -> cells, for slices
        self._column_plans: dict[tuple, object] = {}  # piece sizes -> ColumnPlan (streamlines)
        self._cuts: dict[tuple, CutPlan] = {}  # vertical slices: cut plans per piece
        #: set for output left over from an earlier run in the same folder
        self.earlier_run = False
        #: from the run log (REEF3D_Case), when there is one: the run that wrote these
        #: frames, when it started, and the simulation time of every frame
        self.run_id: str | None = None
        self.run_started = ""
        self.times: np.ndarray | None = None
        self._lock = threading.Lock()  # one merge plan per dataset
        self._span: tuple[float, float] | None = None

    # ------------------------------------------------------------- description
    def _describe(self, first: Frame) -> tuple[str, dict[str, int]]:
        """The kind of grid and the point arrays (name: components), from the first file."""
        if first.pieces[0].suffix == ".vtk":
            return "Lines", legacy_point_arrays(first.pieces[0])
        header = read_header(first.pieces[0])
        kind = "Lines" if header.lines else header.kind
        components = {name: info.components for name, info in header.point_data.items()}
        if self.role in CELL_DATA_ROLES:
            for name, info in header.cell_data.items():
                components.setdefault(name, info.components)
        return kind, components

    @property
    def label(self) -> str:
        label = f"{self.solver} {ROLE_LABELS.get(self.role, self.role)}"
        if self.earlier_run:
            when = f" {self.run_started[:16].replace('T', ' ')}" if self.run_started else ""
            label += f" (earlier run{when}, outputs {self.steps[0]}–{self.last_step})"
        return label

    @property
    def surface_like(self) -> bool:
        """2.5D surfaces whose points move only vertically."""
        return self.role in ("free_surface", "bed")

    @property
    def steps(self) -> list[int]:
        return [frame.step for frame in self.frames]

    def __len__(self) -> int:
        return len(self.frames)

    def variables(self) -> list[str]:
        """Colour-by choices, e.g. ``eta``, ``velocity``, ``velocity.x``."""
        return variable_names(self.components)

    def default_variable(self) -> str:
        preferred = {
            "free_surface": ("eta", "elevation", "WL"),
            "bed": (),  # plain sand; elevation and sediment fields stay one click away
            "volume": ("velocity", "pressure", "elevation"),
            "particles": ("velocity", "age"),
        }.get(self.role, ())
        if self.role in ("bed", "body", "structure", "dem", "ice", "rods"):
            return "none"  # drawn in their material colour, like sand or steel
        for name in preferred:
            if name in self.components:
                return name
        return next(iter(self.components), "none")

    def written_span(self) -> tuple[float, float]:
        """Modification times of the first and last frame (when the run wrote them)."""
        if self._span is None:
            try:
                first = self.frames[0].pieces[0].stat().st_mtime
                last = self.frames[-1].pieces[0].stat().st_mtime
            except OSError:
                first = last = 0.0
            self._span = (min(first, last), max(first, last))
        return self._span

    def same_run_as(self, other: Dataset, margin: float = 600.0) -> bool:
        """True if both were written during one run: by the run log if both are in it,
        otherwise by overlapping write times."""
        if self.run_id and other.run_id:
            return self.run_id == other.run_id
        a0, a1 = self.written_span()
        b0, b1 = other.written_span()
        # DEM output belongs to whichever solver it is coupled to
        solvers = self.solver == other.solver or bool(
            {self.solver, other.solver} & {"DEM", "REEF3D"}
        )
        return solvers and a0 <= b1 + margin and b0 <= a1 + margin

    @property
    def last_step(self) -> int:
        return self.frames[-1].step

    def index_at_time(self, time: float) -> int | None:
        """Index of the frame shown at a simulation time (run log times): the last one
        written at or before it; None before the first."""
        if self.times is None:
            return None
        index = int(np.searchsorted(self.times, time + 1e-9, side="right")) - 1
        return index if index >= 0 else None

    def frame_index(self, frame: float) -> int:
        """Index of the frame shown at a (fractional) output number on the timeline.

        The timeline counts REEF3D's output numbers, so a dataset with missing or
        leftover outputs still lines up in time: the latest frame at or before the
        position is shown (the first frame before it starts).
        """
        if self._step_array is None:
            self._step_array = np.array(self.steps)
        index = int(np.searchsorted(self._step_array, round(frame), side="right")) - 1
        return min(max(index, 0), len(self.frames) - 1)

    # ------------------------------------------------------------- reading
    def read(self, index: int, variables: list[str] | None = None) -> SurfaceMesh:
        """Read one frame as a surface with the arrays needed for ``variables``.

        ``variables=None`` reads every array. Surfaces are merged across MPI ranks;
        volumes are reduced to their outer boundary.
        """
        frame = self.frames[index]
        arrays = None
        if variables is not None:
            arrays = sorted({v.partition(".")[0] for v in variables if v and v != "none"})
            arrays = [name for name in arrays if name in self.components]
            if self.role == "free_surface" and "eta" in self.components and "eta" not in arrays:
                arrays.append("eta")  # needed for vertical exaggeration
        if self.role in PLAIN_ROLES:
            if self.role == "rods" and arrays is not None and "radius" in self.components:
                arrays = sorted({*arrays, "radius"})
            mesh = self._read_plain(frame, arrays, self.role in CELL_DATA_ROLES)
            if self.role == "rods" and mesh.segments is not None and "radius" in mesh.point_data:
                return tubes(mesh.points, mesh.segments, mesh.point_data["radius"], mesh.point_data)
            return mesh
        # reading runs in parallel; only making a new plan is serialised
        light = [self._piece(path, point_arrays=arrays, with_cells=False) for path in frame.pieces]
        if self.kind == "UnstructuredGrid":
            with self._lock:
                plan = self._volume_plan
                if plan is None or not plan.accepts(light):
                    full = [self._piece(path, point_arrays=[]) for path in frame.pieces]
                    plan = self._volume_plan = VolumePlan.build(full)
                    del full
            points, data = plan.apply(light)
            triangles = plan.triangles
        else:
            with self._lock:
                if not self._merger.accepts(light):
                    full = [self._piece(path, point_arrays=[]) for path in frame.pieces]
                    merged = self._merger.make_plan(full).apply(full)
                    self._triangles = np.ascontiguousarray(
                        triangulate(merged.connectivity, merged.offsets), dtype=np.uint32
                    )
                plan = self._merger.plan
                triangles = self._triangles
            merged = plan.apply(light)
            points, data = merged.points, merged.point_data
        mesh = SurfaceMesh(
            points=np.ascontiguousarray(points, dtype=np.float32),
            triangles=triangles,
            point_data=data,
        )
        return mesh.with_normals()  # here, so it runs in the loader thread

    def _plain_piece(self, path: Path, arrays: list[str] | None, cell_data: bool):
        """One file of an object output (a store hands out its points instead)."""
        if path.suffix == ".vtk":
            return read_legacy(path, arrays)
        return read_piece(path, point_arrays=arrays, cell_to_points=cell_data)

    def _read_plain(
        self, frame: Frame, arrays: list[str] | None, cell_data: bool = False
    ) -> SurfaceMesh:
        """Bodies, mooring lines, structures, particles: the pieces side by side, not
        merged."""
        points, triangles, segments, data_parts = [], [], [], []
        offset = 0
        for path in frame.pieces:
            piece = self._plain_piece(path, arrays, cell_data)
            points.append(piece.points)
            if piece.kind == "Lines":
                segments.append(piece.connectivity.reshape(-1, 2) + offset)
            elif piece.offsets.size:
                triangles.append(triangulate(piece.connectivity, piece.offsets) + offset)
            data_parts.append(piece.point_data)
            offset += piece.points.shape[0]
        names = set.intersection(*(set(d) for d in data_parts)) if data_parts else set()
        mesh = SurfaceMesh(
            points=np.ascontiguousarray(np.concatenate(points), dtype=np.float32),
            triangles=np.ascontiguousarray(
                np.concatenate(triangles) if triangles else np.zeros((0, 3)), dtype=np.uint32
            ),
            point_data={name: np.concatenate([d[name] for d in data_parts]) for name in names},
            segments=np.concatenate(segments).astype(np.uint32) if segments else None,
        )
        return mesh.with_normals()

    def read_slice(
        self, index: int, axis: int, position: float, variables: list[str] | None = None
    ) -> SurfaceMesh:
        """The section of a volume frame with the plane ``x[axis] = position``.

        Each MPI piece is cut on its own; the cells of each piece are read once and
        kept, so later frames read only points and the arrays to colour by.
        """
        if self.kind != "UnstructuredGrid":
            raise ValueError(f"{self.label}: only volumes can be sliced")
        frame = self.frames[index]
        arrays = self._arrays(variables)
        parts = []
        for number, path in enumerate(frame.pieces):
            piece = self._piece(path, point_arrays=arrays, with_cells=False)
            _, connectivity, offsets, types = self._piece_cells(number, path, piece)
            # σ-grids move only vertically: an x or y plane cuts the same edges in every frame
            key = (number, axis, float(position), piece.points.shape[0])
            plan = self._cuts.get(key) if axis in (0, 1) else None
            if plan is None:
                plan = plan_cut(piece.points, connectivity, offsets, types, axis, position)
                if axis in (0, 1):
                    if len(self._cuts) > 64:  # positions dragged through: keep it small
                        self._cuts.clear()
                    self._cuts[key] = plan
            parts.append(plan.apply(piece.points, piece.point_data))
        return _joined(parts)

    def point_series(self, variable: str, x: float, y: float) -> tuple[tuple, np.ndarray]:
        """A variable over all frames at the grid point nearest to (x, y), e.g. eta
        under the probe: ((x, y, z) of that point, values per frame).

        REEF3D writes the same layout into every frame of an MPI piece, so the place
        of the value in the file is found once and every frame reads only its few
        bytes (thousands of frames in a second); a frame with another layout is read
        normally.
        """
        name, _, component = variable.partition(".")
        if name not in self.components:
            raise KeyError(f"no variable {name!r} in {self.label}")
        best = None
        for number, path in enumerate(self.frames[0].pieces):
            piece = read_piece(path, point_arrays=[], with_cells=False)
            if not len(piece.points):
                continue
            distance = (piece.points[:, 0] - x) ** 2 + (piece.points[:, 1] - y) ** 2
            index = int(np.argmin(distance))
            if best is None or distance[index] < best[0]:
                best = (
                    float(distance[index]),
                    number,
                    index,
                    tuple(map(float, piece.points[index])),
                )
        if best is None:
            raise ValueError(f"{self.label}: no points")
        _, number, index, where = best
        header = read_header(self.frames[0].pieces[number])
        info = header.point_data[name]
        width = info.components * info.dtype.itemsize
        fast = info.format == "appended" and not header.compressed and header.data_start is not None
        if fast:
            start = header.data_start + info.offset + header.header_dtype.itemsize + index * width
            size = self.frames[0].pieces[number].stat().st_size
        values = []
        for frame in self.frames:
            path = frame.pieces[number] if number < len(frame.pieces) else None
            raw = None
            if fast and path is not None and path.stat().st_size == size:
                with open(path, "rb") as handle:
                    handle.seek(start)
                    raw = np.frombuffer(handle.read(width), info.dtype)
            if raw is None or raw.size != info.components:
                if path is None:
                    values.append(np.full(info.components, np.nan))
                    continue
                raw = read_piece(path, point_arrays=[name], with_cells=False).point_data[name]
                raw = np.asarray(raw).reshape(-1, info.components)[index]
            values.append(np.asarray(raw, dtype=np.float64).ravel())
        values = np.array(values)
        if info.components == 1:
            return where, values[:, 0]
        if component:
            return where, values[:, "xyz".index(component)]
        return where, np.linalg.norm(values, axis=1)

    def read_iso(
        self, index: int, variable: str, value: float, variables: list[str] | None = None
    ) -> SurfaceMesh:
        """The iso-surface of a volume frame where ``variable`` equals ``value``.

        Planned anew for every frame (the field moves), with the cells of each piece
        read once and kept, like slices; neighbouring cut points are merged so the
        surface shades smoothly.
        """
        if self.kind != "UnstructuredGrid":
            raise ValueError(f"{self.label}: iso-surfaces need a volume")
        frame = self.frames[index]
        arrays = self._arrays([*(variables or []), variable]) or []
        parts = []
        for number, path in enumerate(frame.pieces):
            piece = self._piece(path, point_arrays=arrays, with_cells=False)
            _, connectivity, offsets, types = self._piece_cells(number, path, piece)
            field = scalar_values(piece.point_data, variable)
            if field is None:
                raise ValueError("an iso-surface needs a variable")
            plan = plan_level(field.astype(np.float64) - value, connectivity, offsets, types, True)
            plan = orient_by_field(plan, piece.points, field)
            parts.append(plan.apply(piece.points, piece.point_data))
        return _joined(parts)

    def read_streamlines(
        self,
        index: int,
        vector: str,
        axis: int,
        position: float,
        count: int,
        variables: list[str] | None = None,
        lines_in: str = "volume",
    ) -> SurfaceMesh:
        """Streamlines of a vector field, as line segments carrying the sampled
        variables. ``lines_in``:

        * ``"volume"``: through the volume, seeded on the plane ``x[axis] = position``;
        * ``"seed plane"``: in that plane only (the part of the field across the plane
          left out), like on a slice;
        * ``"free surface"``: on this free surface output, along the horizontal part
          of the field, seeded over the wet surface.
        """
        from lagoonviz.core.streamlines import ColumnPlan, SheetGrid, trace, trace_on_sheet

        arrays = self._arrays([*(variables or []), vector]) or []
        if lines_in == "free surface":
            extra = ["depth"] if "depth" in self.components else []
            mesh = self.read(index, [*arrays, *extra])
            if vector not in mesh.point_data or mesh.point_data[vector].ndim != 2:
                raise ValueError(f"{vector!r} is not a vector field of {self.label}")
            depth = mesh.point_data.get("depth")
            sheet = SheetGrid.from_points(
                mesh.points, mesh.point_data, wet=None if depth is None else depth > 0
            )
            lines = trace_on_sheet(sheet, vector, sheet.seeds(count))
            return _lines_mesh(lines, sheet)
        if self.kind != "UnstructuredGrid":
            raise ValueError(f"{self.label}: streamlines need a volume")
        pieces = [
            self._piece(p, point_arrays=arrays, with_cells=False) for p in self.frames[index].pieces
        ]
        names = set.intersection(*(set(p.point_data) for p in pieces)) if pieces else set()
        sizes = tuple(len(p.points) for p in pieces)
        with self._lock:
            plan = self._column_plans.get(sizes)
        if plan is None:  # once per grid: later frames only pick their values
            plan = ColumnPlan.build([p.points for p in pieces])
            with self._lock:
                self._column_plans[sizes] = plan
        grid = plan.grid(
            np.concatenate([p.points for p in pieces]),
            {n: np.concatenate([p.point_data[n] for p in pieces]) for n in names},
        )
        if vector not in grid.fields or grid.fields[vector].ndim != 4:
            raise ValueError(f"{vector!r} is not a vector field of {self.label}")
        follow = vector
        if lines_in == "seed plane":
            in_plane = grid.fields[vector].copy()
            in_plane[..., axis] = 0.0  # what crosses the plane is left out
            grid.fields["_in_plane"] = in_plane
            follow = "_in_plane"
        lines = trace(grid, follow, grid.seeds(axis, position, count))
        grid.fields.pop("_in_plane", None)
        return _lines_mesh(lines, grid)

    def _piece_cells(self, number: int, path: Path, piece) -> tuple:
        """The cells of one MPI piece, read once (the grid does not change)."""
        with self._lock:
            cells = self._cells.get(number)
            if cells is None or cells[0] != piece.points.shape[0]:
                full = self._piece(path, point_arrays=[])
                cells = (
                    full.points.shape[0],
                    full.connectivity.astype(np.int32),
                    full.offsets.astype(np.int64),
                    full.types,
                )
                self._cells[number] = cells
        return cells

    def _piece(self, path: Path, point_arrays: list[str] | None = None, with_cells: bool = True):
        """One piece of a frame (a VTK file here; a block of a LAGOON store in
        :class:`~lagoonviz.core.data.lagoon.StoreDataset`)."""
        return read_piece(path, point_arrays=point_arrays, with_cells=with_cells)

    def bounds(self, index: int = 0) -> tuple[np.ndarray, np.ndarray]:
        """Lower and upper corner of a frame's points."""
        pieces = [
            self._piece(p, point_arrays=[], with_cells=False) for p in self.frames[index].pieces
        ]
        points = np.concatenate([p.points for p in pieces])
        return points.min(axis=0), points.max(axis=0)

    def _arrays(self, variables: list[str] | None) -> list[str] | None:
        if variables is None:
            return None
        names = sorted({v.partition(".")[0] for v in variables if v and v != "none"})
        return [name for name in names if name in self.components]


@dataclass
class Case:
    """A REEF3D run: its datasets and, if known, the time between outputs."""

    path: Path
    datasets: dict[str, Dataset] = field(default_factory=dict)
    print_interval: float | None = None  # seconds between frames (ctrl.txt P 30)
    #: gauge, 6DOF and force tables (read on first use)
    series: dict[str, SeriesFile] = field(default_factory=dict)
    #: wave profiles along lines (WSFLINE), by solver
    profiles: dict[str, ProfileSet] = field(default_factory=dict)

    @property
    def solver(self) -> str:
        for dataset in self.datasets.values():
            return dataset.solver
        return ""

    def by_role(self, role: str) -> Dataset | None:
        """The dataset shown for a role when the item does not name one (see :func:`preferred`)."""
        matches = [d for d in self.datasets.values() if d.role == role]
        return preferred(matches) if matches else None

    @property
    def n_frames(self) -> int:
        """Length of the timeline: the highest output number plus one."""
        return max((d.last_step + 1 for d in self.datasets.values()), default=0)

    def main_profiles(self) -> list[ProfileSet]:
        """Wave profiles written during the shown run (as for :meth:`main_series`)."""
        shown = list(self.main_datasets().values())
        if not shown:
            return list(self.profiles.values())
        start = min(d.written_span()[0] for d in shown) - 600.0
        end = max(d.written_span()[1] for d in shown) + 600.0
        return [p for p in self.profiles.values() if start <= p.written_span()[1] <= end]

    def main_series(self, limit: int = 4) -> list[SeriesFile]:
        """Time series to plot when the case is opened: those of the shown run.

        A series belongs to the run if its file was last written while the run's
        surfaces were written. Wave gauges come first, then forces and motions.
        """
        shown = list(self.main_datasets().values())
        if shown:
            start = min(d.written_span()[0] for d in shown) - 600.0
            end = max(d.written_span()[1] for d in shown) + 600.0
        else:
            start, end = -np.inf, np.inf
        order = {"wave gauges": 0, "6DOF": 1, "forces": 2}
        picked = []
        for item in sorted(self.series.values(), key=lambda f: (order.get(f.group, 9), f.key)):
            try:
                written = item.path.stat().st_mtime
            except OSError:
                continue
            if not start <= written <= end:
                continue
            if item.group == "6DOF" and not any(
                word in item.path.stem for word in ("forces", "position")
            ):
                continue  # velocities and mooring lines stay available, not shown
            picked.append(item)
        return picked[:limit]

    def main_datasets(self) -> dict[str, Dataset]:
        """What to show when the case is opened: the most recent run's surfaces.

        A folder can hold output of several runs (REEF3D overwrites only the files it
        writes again), e.g. an FNPF run next to an older NHFLOW run, or a bed from an
        earlier run. The free surface (or bed) written last picks the run; outputs
        written during that run join it; a bed of the same solver joins in any case.
        Without surfaces, the volume is shown.
        """
        surfaces = [d for d in self.datasets.values() if d.role in ("free_surface", "bed")]
        if not surfaces:
            volumes = [d for d in self.datasets.values() if d.role == "volume"]
            if not volumes:
                return {}
            return {"volume": max(volumes, key=lambda d: d.written_span()[1])}
        free = [d for d in surfaces if d.role == "free_surface"]
        lead = preferred(free or surfaces)
        chosen = {lead.role: lead}
        for role in ("bed", "free_surface"):
            if role in chosen:
                continue
            options = [d for d in surfaces if d.role == role and d.same_run_as(lead)]
            if not options and role == "bed":
                # the bed rarely changes between runs: take the solver's latest one
                options = [d for d in surfaces if d.role == "bed" and d.solver == lead.solver]
                options = [preferred(options)] if options else []
            if options:
                chosen[role] = max(options, key=len)
        # objects of the same run: bodies, mooring lines, structures
        for role in ("body", "mooring", "structure", "dem", "ice", "rods", "particles"):
            options = [d for d in self.datasets.values() if d.role == role and d.same_run_as(lead)]
            if options:
                chosen[role] = max(options, key=len)
        # beds are drawn first, so a transparent free surface shows them
        order = (
            "bed",
            "structure",
            "body",
            "dem",
            "ice",
            "rods",
            "particles",
            "mooring",
            "free_surface",
        )
        return {role: chosen[role] for role in order if role in chosen}

    def describe(self) -> str:
        lines = [f"REEF3D case {self.path}"]
        for dataset in self.datasets.values():
            lines.append(
                f"  {dataset.key}: {dataset.label}, {len(dataset)} frames, "
                f"{len(dataset.frames[0].pieces)} pieces, "
                f"variables: {', '.join(dataset.components)}"
            )
        return "\n".join(lines)


# ----------------------------------------------------------------- discovery
def _frames_by_piece(folder: Path) -> dict[str, list[Frame]]:
    """Series named ``<stem>-<piece>-<step>`` (bodies, mooring lines) or unnumbered.

    Each output step becomes one frame whose pieces are the bodies or lines; a file
    without numbers (a fixed structure) is one frame at step 0.
    """
    steps: dict[str, dict[int, dict[int, Path]]] = {}
    # DEM output numbers its files; structure files may number structures instead
    numbered = folder.name.endswith(_NUMBERED_FOLDERS)
    for path in folder.iterdir():
        if path.suffix not in _SUFFIXES or path.name.startswith("."):
            continue
        match = _PIECE_STEP.match(path.stem)
        serial = _SERIAL_STEP.match(path.stem) if numbered else None
        if match:
            stem, piece, step = (
                match.group("stem"),
                int(match.group("piece")),
                int(match.group("step")),
            )
        elif serial:  # one file per output: REEF3D-DEM-000012
            stem, piece, step = serial.group("stem"), 0, int(serial.group("step"))
        else:
            stem, piece, step = path.stem, 0, 0
        steps.setdefault(stem, {}).setdefault(step, {})[piece] = path
    series = {}
    for stem, by_step in steps.items():
        series[stem] = [
            Frame(step, pieces[min(pieces)], [pieces[k] for k in sorted(pieces)])
            for step, pieces in sorted(by_step.items())
        ]
    return series


def _frames_in(folder: Path) -> dict[str, list[Frame]]:
    """Group the files of a folder into series of frames, keyed by file stem."""
    objects = _OBJECT_FOLDER.match(folder.name)
    if objects and objects.group("what") not in _PARALLEL_OBJECTS:
        return _frames_by_piece(folder)
    parallel: dict[str, dict[int, Path]] = {}
    serial: dict[str, dict[int, Path]] = {}
    for path in folder.iterdir():
        if path.suffix not in _SUFFIXES or path.name.startswith("."):
            continue
        stem, step, rank = split_name(path)
        if step is None:
            continue
        if path.suffix in (".pvtu", ".pvtp"):
            parallel.setdefault(stem, {})[step] = path
        elif rank is None:
            serial.setdefault(stem, {})[step] = path
    series: dict[str, list[Frame]] = {}
    for stem, steps in parallel.items():
        frames = []
        for step in sorted(steps):
            path = steps[step]
            header = read_parallel_header(path) if not frames else None
            if header is not None:
                pieces = header.pieces
                names = [p.name for p in pieces]
            else:
                # pieces follow the same naming in every frame: avoid parsing each file
                old = f"{split_name(frames[0].path)[1]:0{_width(frames[0].path)}d}"
                new = f"{step:0{_width(path)}d}"
                pieces = [path.parent / n.replace(f"-{old}-", f"-{new}-", 1) for n in names]
            frames.append(Frame(step, path, pieces))
        series[stem] = frames
    for stem, steps in serial.items():
        if stem not in series:
            series[stem] = [Frame(s, steps[s], [steps[s]]) for s in sorted(steps)]
    return series


def _width(path: Path) -> int:
    match = re.search(r"-(\d+)(?:-\d+)?$", path.stem)
    return len(match.group(1)) if match else 8


_OWNERS: dict[Path, tuple[tuple, dict]] = {}


def _owners(case_dir: Path) -> dict[Path, tuple[Run, Output]]:
    """Which run wrote each file, and at which time, from the case's run log
    (REEF3D_Case/REEF3D_*_run.jsonl); empty without one. Read once per change."""
    case_dir = case_dir.resolve()
    logs = sorted((case_dir / "REEF3D_Case").glob("REEF3D_*_run.jsonl")) + sorted(
        case_dir.glob("REEF3D_*_run.jsonl")
    )
    stamp = tuple((p.name, p.stat().st_mtime, p.stat().st_size) for p in logs)
    cached = _OWNERS.get(case_dir)
    if cached is not None and cached[0] == stamp:
        return cached[1]
    found: dict[Path, tuple[Run, Output]] = {}
    log = read_case(case_dir) if logs else None
    if log is not None:
        for path, (run, output, _stream) in log.owners().items():
            found[(case_dir / path).resolve()] = (run, output)
    _OWNERS[case_dir] = (stamp, found)
    return found


def run_log(case_dir: Path) -> RunLog | None:
    """The run log of a case folder, if REEF3D wrote one."""
    return read_case(case_dir.resolve())


def _split(frames: list[Frame], owners: dict) -> list[tuple[list[Frame], Run | None, list]]:
    """Frames grouped by the run that wrote them, the latest run first, each with the
    simulation times of its frames. Uses the run log; frames it does not know are
    split by modification times, as before."""
    if owners:
        by_run: dict[str, tuple[Run, list[Frame], list[float]]] = {}
        unknown: list[Frame] = []
        for frame in frames:
            owner = owners.get(frame.path.resolve())
            if owner is None:
                unknown.append(frame)
                continue
            run, output = owner
            entry = by_run.setdefault(run.id, (run, [], []))
            entry[1].append(frame)
            entry[2].append(output.time)
        if by_run:
            groups = sorted(by_run.values(), key=lambda e: e[0].started, reverse=True)
            result = [(f, run, times) for run, f, times in groups]
            result += [(run, None, []) for run in split_runs(unknown)] if unknown else []
            return result
    return [(run, None, []) for run in split_runs(frames)]


def _lines_mesh(lines: list[np.ndarray], grid) -> SurfaceMesh:
    """Polylines as one mesh of segments, with every field of ``grid`` sampled."""
    if not lines:
        empty = np.zeros((0, 3), np.float32)
        return SurfaceMesh(
            empty, np.zeros((0, 3), np.uint32), {}, segments=np.zeros((0, 2), np.uint32)
        )
    points = np.concatenate(lines)
    starts = np.cumsum([0] + [len(line) for line in lines[:-1]])
    segments = np.concatenate(
        [
            np.stack([s + np.arange(len(line) - 1), s + np.arange(1, len(line))], axis=1)
            for s, line in zip(starts, lines, strict=True)
        ]
    )
    data = {name: grid.sample(name, points)[0].astype(np.float32) for name in grid.fields}
    return SurfaceMesh(
        points.astype(np.float32),
        np.zeros((0, 3), np.uint32),
        data,
        segments=segments.astype(np.uint32),
    )


def _joined(parts: list[SurfaceMesh]) -> SurfaceMesh:
    """The surfaces cut from each MPI piece as one mesh."""
    points = np.concatenate([p.points for p in parts])
    shift = np.cumsum([0] + [p.points.shape[0] for p in parts[:-1]])
    triangles = np.concatenate(
        [p.triangles.astype(np.int64) + s for p, s in zip(parts, shift, strict=True)]
    )
    names = set.intersection(*(set(p.point_data) for p in parts)) if parts else set()
    data = {name: np.concatenate([p.point_data[name] for p in parts]) for name in names}
    mesh = SurfaceMesh(points, np.ascontiguousarray(triangles, dtype=np.uint32), data)
    return mesh.with_normals()


def _dataset_from_folder(folder: Path) -> list[Dataset]:
    match = _FOLDER.match(folder.name)
    objects = _OBJECT_FOLDER.match(folder.name)
    owners = _owners(folder.parent) if folder.parent.exists() else {}
    datasets = []
    for stem, frames in _frames_in(folder).items():
        if objects:
            # DEM and rod tree output belong to whichever solver they are coupled to
            solver = objects.group("solver") or (
                "DEM" if objects.group("what") == "DEM_VTP" else "REEF3D"
            )
            role = _OBJECT_ROLES[objects.group("what")]
            key = f"{solver.lower()}_{role}"
            datasets += _datasets_by_run(key, solver, role, frames, folder, owners)
            continue
        # REEF3D-NHFLOW-FSF -> solver NHFLOW, tag FSF
        parts = stem.split("-")
        solver = parts[1] if len(parts) > 1 and parts[0] == "REEF3D" else "REEF3D"
        tag = "_".join(parts[2:]) or None
        what = "VTU" if frames[0].path.suffix in (".pvtu", ".vtu") else "VTP"
        if match:
            solver = match.group("solver")
            tag = match.group("tag") or tag
            what = "VTU" if match.group("what") == "VTU" else "VTP"
        kind = "UnstructuredGrid" if what == "VTU" else "PolyData"
        role = _role(what, tag, kind)
        key = f"{solver.lower()}_{role}"
        datasets += _datasets_by_run(key, solver, role, frames, folder, owners)
    return datasets


def _datasets_by_run(key, solver, role, frames, folder, owners) -> list[Dataset]:
    datasets = []
    for number, (run_frames, run, times) in enumerate(_split(frames, owners)):
        dataset = Dataset(
            key if number == 0 else f"{key}_earlier{number}", solver, role, run_frames, folder
        )
        dataset.earlier_run = number > 0
        if run is not None:
            dataset.run_id = run.id
            dataset.run_started = run.started
            order = np.argsort(times, kind="stable")
            if np.any(np.diff(np.asarray(times)) < 0):  # keep frames in time order
                dataset.frames = [run_frames[i] for i in order]
            dataset.times = np.asarray(times, dtype=np.float64)[order]
        elif role in OWN_TIME_ROLES:
            _times_from_files(dataset)
        datasets.append(dataset)
    return datasets


def _times_from_files(dataset: Dataset) -> None:
    """Simulation times from the files' TimeValue field data, for outputs written at
    their own interval (DEM: E 15), which output numbers cannot place on the timeline.
    Left unset unless every frame has one."""
    times = [
        read_time_value(frame.pieces[0] if frame.pieces else frame.path) for frame in dataset.frames
    ]
    if not times or any(t is None for t in times):
        return
    order = np.argsort(times, kind="stable")
    dataset.frames = [dataset.frames[i] for i in order]
    dataset.times = np.asarray(times, dtype=np.float64)[order]


def preferred(datasets: list[Dataset], short: float = 0.1) -> Dataset:
    """The run to show among alternatives: the latest, unless it is only a short start.

    A run stopped after a few outputs (a test, a crash) is usually not what one
    wants to look at; if the latest run has less than ``short`` times the outputs of
    the longest one, the longest is shown. The others stay selectable.
    """
    if all(d.run_started for d in datasets):  # the run log knows when each run began
        latest = max(datasets, key=lambda d: d.run_started)
    else:
        latest = max(datasets, key=lambda d: d.written_span()[1])
    longest = max(datasets, key=len)
    return latest if len(latest) >= short * len(longest) else longest


def split_runs(frames: list[Frame], tolerance: float = 1.0) -> list[list[Frame]]:
    """Split a series into the runs that wrote it, the latest run first.

    A run writes its outputs in order, so their modification times increase with
    the output number. REEF3D started again in the same folder overwrites outputs
    from the beginning; the outputs it did not reach are older than the ones before
    them. Each such drop in time starts the leftovers of an earlier run.
    """
    if len(frames) < 2:
        return [frames]
    times = []
    for frame in frames:
        try:
            times.append(frame.path.stat().st_mtime)
        except OSError:
            times.append(0.0)
    runs: list[list[Frame]] = [[frames[0]]]
    run_times = [[times[0]]]
    for frame, written, before in zip(frames[1:], times[1:], times[:-1], strict=True):
        if written < before - tolerance:
            runs.append([])
            run_times.append([])
        runs[-1].append(frame)
        run_times[-1].append(written)
    order = sorted(range(len(runs)), key=lambda i: max(run_times[i]), reverse=True)
    return [runs[i] for i in order]


def read_print_interval(case_dir: Path) -> float | None:
    """Time between outputs from ``ctrl.txt`` (``P 30 dt``), if the run used it."""
    ctrl = case_dir / "ctrl.txt"
    if not ctrl.is_file():
        return None
    try:
        for line in ctrl.read_text(errors="replace").splitlines():
            fields = line.split()
            if len(fields) >= 3 and fields[0] == "P" and fields[1] == "30":
                value = float(fields[2])
                return value if value > 0 else None
    except (OSError, ValueError):
        return None
    return None


def _prefer_stores(datasets: list[Dataset], stored: list[Dataset]) -> list[Dataset]:
    """A LAGOON store's output replaces the VTK output it was converted from, unless
    the VTK output has outputs the store does not have yet (a run still going on,
    not converted again): then the VTK output is shown and the store is left out."""
    if not stored:
        return datasets
    kept = list(datasets)
    for store in stored:
        recorded = getattr(store, "recorded", None)
        if recorded:  # the store says which output it was converted from
            same = [d for d in datasets if d.key == recorded]
        else:
            same = [
                d
                for d in datasets
                if d.role == store.role
                and d.solver == store.solver
                and not d.earlier_run
                and (not d.run_id or not store.run_id or d.run_id == store.run_id)
            ]
        if any(d.last_step > store.last_step for d in same):
            where = getattr(getattr(store, "store", None), "path", store.folder.parent)
            log.info("%s: the VTK output is newer than %s", store.key, Path(where).name)
            continue
        kept = [d for d in kept if d not in same]
        kept.append(store)
    return kept


def open_case(path: str | Path, stores: bool = True) -> Case:
    """Open a REEF3D case folder, one output folder, one output file or a LAGOON
    store. ``stores=False`` leaves LAGOON stores in the case folder out (the
    converter reads the VTK output)."""
    from lagoonviz.core.data.lagoon import find_stores, store_datasets, stored_series

    path = Path(path).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"{path} does not exist")
    datasets: list[Dataset] = []
    stored: list[Dataset] = []
    store_paths: list[Path] = []
    if path.is_dir() and path in find_stores(path.parent):  # a LAGOON store
        case_dir = path.parent
        stored = store_datasets(path)
        store_paths = [path]
    elif path.is_file() and path.suffix == ".dat":
        # a gauge or force table: its case is the folder above REEF3D_*
        case_dir = path.parent.parent if path.parent.name.startswith("REEF3D_") else path.parent
    elif path.is_file():
        if path.suffix not in _SUFFIXES:
            raise ValueError(f"{path.name}: not a VTU/VTP file")
        folder = path.parent
        stem = split_name(path)[0]
        found = [d for d in _dataset_from_folder(folder) if d.frames[0].path.name.startswith(stem)]
        datasets = found
        case_dir = folder.parent if _FOLDER.match(folder.name) else folder
    elif _FOLDER.match(path.name) or _OBJECT_FOLDER.match(path.name):
        datasets = _dataset_from_folder(path)
        case_dir = path.parent
    elif path.name.startswith("REEF3D_"):  # e.g. REEF3D_NHFLOW_6DOF
        case_dir = path.parent
    else:
        case_dir = path
        for folder in sorted(path.iterdir()):
            if folder.is_dir() and (
                _FOLDER.match(folder.name) or _OBJECT_FOLDER.match(folder.name)
            ):
                datasets += _dataset_from_folder(folder)
        store_paths = find_stores(path) if stores else []
        for store in store_paths:
            stored += store_datasets(store)
        if not datasets and not stored:
            # a plain folder with VTU/VTP files
            datasets = _dataset_from_folder(path)
    datasets = _prefer_stores(datasets, stored)
    series = find_series(case_dir)
    profiles = find_profiles(case_dir)
    if store_paths:  # tables and profiles of LAGOON stores replace the text files
        series, profiles = stored_series(store_paths, series, profiles)
    if not datasets and not series and not profiles:
        raise ValueError(f"no REEF3D VTU/VTP output or time series found in {path}")
    case = Case(
        case_dir,
        print_interval=read_print_interval(case_dir),
        series=series,
        profiles=profiles,
    )
    for dataset in datasets:
        key = dataset.key
        n = 2
        while key in case.datasets:
            key = f"{dataset.key}_{n}"
            n += 1
        dataset.key = key
        case.datasets[key] = dataset
    return case
