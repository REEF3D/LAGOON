# SPDX-License-Identifier: GPL-3.0-or-later
# Architect: Hans Bihs
"""Statistics over time as maps: every point of a surface output (free surface, bed)
over a time window, as MIKE's and SMS's result statistics or Crayfish's temporal
aggregation:

- per variable (a vector: its magnitude): ``<v>_max``, ``<v>_min``, ``<v>_mean``,
  ``<v>_time_of_max`` (the simulation time) and ``<v>_change`` (last minus first:
  the bed's erosion and deposition);
- with the free surface's ``eta``: ``Hs`` = 4 times its standard deviation over the
  window (the significant wave height Hm0: wave penetration, harbour agitation),
  ``eta_range`` (highest crest to lowest trough) and ``setup`` (its mean);
- with a water depth (NHFLOW's WL, SFLOW's waterlevel, FNPF's depth + eta; or
  wetdry): ``first_wet`` (when the water first arrives, NaN where it never does),
  ``wet_duration`` [s], ``ever_wet`` (1/0: the maximum inundation), ``max_depth`` and
  ``max_level`` (the highest water level: run-up).

Waves and velocities count only while a point is wet, where the output says. The
map lies at the mean of the surface's heights over the window. The result is kept
in the case's ``LAGOON_Statistics`` folder (an .npz file per source and window), so a
session opens it again without reading the run.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from lagoonviz.core.data.mesh import SurfaceMesh, scalar_values

FOLDER = "LAGOON_Statistics"
#: a water depth above this [m] is wet (the default of an item)
WET_DEPTH = 0.01
#: variables of a free surface or bed taken without being asked for (those it has)
AUTO = {
    "free_surface": ("eta", "velocity", "water_depth", "hazard", "froude", "bed_shear"),
    "bed": ("elevation", "velocity", "bedload", "suspended", "shields"),
    "seastate": ("Hs", "Tp", "Tm01"),
}

Progress = Callable[[float, str], None]


@dataclass
class Recipe:
    """What a statistics map is made of: the source output, the window [s] (``end`` 0:
    to the last output), the variables (empty: :data:`AUTO`) and the wet depth."""

    source: str
    start: float = 0.0
    end: float = 0.0
    variables: list[str] = field(default_factory=list)
    wet_depth: float = WET_DEPTH

    def name(self) -> str:
        """The file name of its result."""
        text = json.dumps(asdict(self), sort_keys=True)
        digest = hashlib.sha1(text.encode()).hexdigest()[:10]
        safe = "".join(c if c.isalnum() or c in "_-" else "_" for c in self.source)
        return f"{safe}-{digest}.npz"


def default_variables(dataset) -> list[str]:
    """The variables :data:`AUTO` names that a dataset has (on file or derived)."""
    wanted = AUTO.get(dataset.role, ())
    found = [v for v in wanted if v in dataset.components]
    if not found:  # an output of its own kind: its scalars and vectors
        found = [n for n, c in dataset.components.items() if c in (1, 3)][:6]
    return found


def _wetness(dataset) -> str | None:
    """How a dataset says where it is wet: "water_depth" (derived), "wetdry", None."""
    if "water_depth" in dataset.components:
        return "water_depth"
    if "wetdry" in dataset.components:
        return "wetdry"
    return None


def compute(
    dataset, times: np.ndarray, recipe: Recipe, progress: Progress | None = None
) -> tuple[SurfaceMesh, dict]:
    """The statistics map of a dataset over the recipe's window: (the map, facts).
    ``times``: the simulation time of every frame of the dataset."""
    times = np.asarray(times, np.float64)
    end = recipe.end if recipe.end > 0 else float(times[-1])
    chosen = [k for k, t in enumerate(times) if recipe.start - 1e-9 <= t <= end + 1e-9]
    if len(chosen) < 2:
        raise ValueError(
            f"the window {recipe.start:g}–{end:g} s holds {len(chosen)} output(s): "
            "statistics need two at least"
        )
    variables = [v for v in (recipe.variables or default_variables(dataset))]
    unknown = [v for v in variables if v.partition(".")[0] not in dataset.components]
    if unknown:
        raise ValueError(f"{dataset.label} has no {', '.join(unknown)}")
    wet_by = _wetness(dataset)
    eta = "eta" if "eta" in dataset.components and dataset.role == "free_surface" else None
    arrays = sorted({*variables, *([wet_by] if wet_by else []), *([eta] if eta else [])})
    # how long each output stands for: half-way to its neighbours
    t = times[chosen]
    edges = np.concatenate([[t[0]], 0.5 * (t[1:] + t[:-1]), [t[-1]]])
    weight = np.diff(edges)
    first = None
    acc: dict[str, dict[str, np.ndarray]] = {}
    z_sum = None
    for number, (k, w) in enumerate(zip(chosen, weight, strict=True)):
        if progress is not None:
            progress(number / len(chosen), f"Output {number + 1} of {len(chosen)}")
        mesh = dataset.read(k, arrays)
        n = mesh.n_points
        if first is None:
            first = mesh
            z_sum = np.zeros(n)
            wet_time = np.zeros(n)
            wet_count = np.zeros(n)
            first_wet = np.full(n, np.nan)
            max_depth = np.full(n, -np.inf)
            max_level = np.full(n, -np.inf)
        elif n != first.n_points:
            raise ValueError(
                f"{dataset.label}: its points change from output to output (AMR, a "
                "restart with another grid): statistics need the same points throughout"
            )
        z = mesh.points[:, 2].astype(np.float64)
        z_sum += z
        if wet_by == "water_depth":
            depth = np.asarray(mesh.point_data["water_depth"], np.float64)
            wet = depth > recipe.wet_depth
            max_depth = np.maximum(max_depth, np.where(np.isfinite(depth), depth, -np.inf))
        elif wet_by == "wetdry":
            wet = np.asarray(mesh.point_data["wetdry"], np.float64) > 0.5
        else:
            wet = np.ones(n, bool)
        if wet_by:
            first_wet = np.where(np.isnan(first_wet) & wet, times[k], first_wet)
            wet_time += np.where(wet, w, 0.0)
            max_level = np.maximum(max_level, np.where(wet, z, -np.inf))
        wet_count += wet
        series = {v: scalar_values(mesh.point_data, v) for v in variables}
        if eta and eta not in series:
            series[eta] = scalar_values(mesh.point_data, eta)
        for name, values in series.items():
            values = np.asarray(values, np.float64)
            a = acc.get(name)
            if a is None:
                a = acc[name] = {
                    "max": np.full(n, -np.inf),
                    "min": np.full(n, np.inf),
                    "sum": np.zeros(n),
                    "sq": np.zeros(n),
                    "weight": np.zeros(n),
                    "tmax": np.full(n, np.nan),
                    "first": values.copy(),
                    "last": values.copy(),
                }
            # waves and velocities only while wet (the depth itself always)
            use = (wet if name not in ("water_depth", "elevation") else np.ones(n, bool)) & (
                np.isfinite(values)
            )
            higher = use & (values > a["max"])
            a["tmax"] = np.where(higher, times[k], a["tmax"])
            a["max"] = np.where(higher, values, a["max"])
            a["min"] = np.where(use & (values < a["min"]), values, a["min"])
            a["sum"] += np.where(use, values * w, 0.0)
            a["sq"] += np.where(use, values * values * w, 0.0)
            a["weight"] += np.where(use, w, 0.0)
            a["last"] = np.where(np.isfinite(values), values, a["last"])
    if progress is not None:
        progress(1.0, "Statistics")
    data: dict[str, np.ndarray] = {}
    for name, a in acc.items():
        with np.errstate(invalid="ignore", divide="ignore"):
            mean = np.where(a["weight"] > 0, a["sum"] / a["weight"], np.nan)
            var = np.where(a["weight"] > 0, a["sq"] / a["weight"] - mean * mean, np.nan)
        never = a["weight"] <= 0
        base = name.replace(".", "_")
        if name in variables:
            data[f"{base}_max"] = np.where(never, np.nan, a["max"])
            data[f"{base}_min"] = np.where(never, np.nan, a["min"])
            data[f"{base}_mean"] = mean
            data[f"{base}_time_of_max"] = np.where(never, np.nan, a["tmax"])
            data[f"{base}_change"] = a["last"] - a["first"]
        if name == eta:
            data["Hs"] = 4.0 * np.sqrt(np.clip(var, 0.0, None))
            data["eta_range"] = np.where(never, np.nan, a["max"] - a["min"])
            data["setup"] = mean
    if wet_by:
        data["first_wet"] = first_wet
        data["wet_duration"] = wet_time
        data["ever_wet"] = (wet_count > 0).astype(np.float64)
        data["max_level"] = np.where(np.isfinite(max_level), max_level, np.nan)
        if wet_by == "water_depth":
            data["max_depth"] = np.where(np.isfinite(max_depth), max_depth, np.nan)
    points = first.points.copy()
    points[:, 2] = (z_sum / len(chosen)).astype(points.dtype)
    out = SurfaceMesh(
        points,
        first.triangles.copy(),
        {k: np.asarray(v, np.float32) for k, v in data.items()},
    )
    facts = {
        "source": recipe.source,
        "label": dataset.label,
        "start": float(t[0]),
        "end": float(t[-1]),
        "outputs": len(chosen),
        "variables": variables,
        "wetness": wet_by or "",
    }
    return out, facts


def save(path: Path, mesh: SurfaceMesh, facts: dict, recipe: Recipe) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays = {f"data_{k}": v for k, v in mesh.point_data.items()}
    np.savez_compressed(
        path,
        points=mesh.points,
        triangles=mesh.triangles,
        facts=np.array(json.dumps({**facts, "recipe": asdict(recipe)})),
        **arrays,
    )
    return path


def load(path: Path) -> tuple[SurfaceMesh, dict]:
    with np.load(path, allow_pickle=False) as found:
        data = {k[5:]: np.asarray(found[k]) for k in found.files if k.startswith("data_")}
        mesh = SurfaceMesh(np.asarray(found["points"]), np.asarray(found["triangles"]), data)
        facts = json.loads(str(found["facts"]))
    return mesh, facts


# ======================================================================== as a dataset
from lagoonviz.core.data.reef3d import Dataset, Frame  # noqa: E402

#: roles a statistics map can be made of
SOURCE_ROLES = ("free_surface", "bed", "seastate")


class StatisticsDataset(Dataset):
    """A statistics map kept in its .npz file: one frame, shown at any time."""

    def __init__(self, key: str, path: Path) -> None:
        self._mesh, self.facts = load(path)
        super().__init__(key, "LAGOON", "statistics", [Frame(0, path, [path])], path.parent)

    def _describe(self, first: Frame) -> tuple[str, dict[str, int]]:
        return "PolyData", {n: 1 for n in self._mesh.point_data}

    @property
    def surface_like(self) -> bool:
        return True

    @property
    def label(self) -> str:
        f = self.facts
        return f"Statistics of {f.get('label', '')} {f.get('start', 0):g}–{f.get('end', 0):g} s"

    def default_variable(self) -> str:
        for name in ("Hs", "max_depth", "elevation_change", "ever_wet"):
            if name in self.components:
                return name
        return next((n for n in self.components if n.endswith("_max")), "none")

    def read(self, index: int = 0, variables: list[str] | None = None, **_) -> SurfaceMesh:
        data = self._mesh.point_data
        if variables is not None:
            names = {v.partition(".")[0] for v in variables}
            data = {k: v for k, v in data.items() if k in names or k == "ever_wet"}
        return SurfaceMesh(self._mesh.points, self._mesh.triangles, dict(data)).with_normals()

    def point_series(self, variable: str, x: float, y: float):
        mesh = self._mesh
        k = int(np.argmin((mesh.points[:, 0] - x) ** 2 + (mesh.points[:, 1] - y) ** 2))
        value = mesh.scalar(variable)
        return tuple(float(v) for v in mesh.points[k]), np.array([float(value[k])])

    def written_span(self) -> tuple[float, float]:
        stamp = self.frames[0].path.stat().st_mtime
        return stamp, stamp
