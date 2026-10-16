# SPDX-License-Identifier: GPL-3.0-or-later
# Architect: Hans Bihs
"""Derived quantities: variables worked out from those a run wrote, as ParaView's
calculator does, that the scene colours by like any other (contour lines,
iso-surfaces, slices, colour ranges).

Built in, where the inputs are there:

- surfaces of shallow flow (free surface, AMR surface) with a velocity and a water
  depth (NHFLOW's ``WL``, SFLOW's ``waterlevel``, FNPF's ``depth`` + ``eta``):
  ``water_depth`` h, ``froude`` |u|/sqrt(g h), ``unit_discharge`` h·u (a vector),
  ``hazard`` h·|u| (depth times velocity), ``bed_shear`` rho g n² |u|² / h^(1/3)
  (Manning's n of the case);
- surfaces with a velocity (also the bed): ``vorticity`` dv/dx - du/dy;
- volumes with a velocity (CFD, NHFLOW): ``vorticity`` (the curl, a vector) and
  ``q_criterion`` Q = (|Omega|² - |S|²)/2 (vortices where Q > 0).

And the case's own formulas (its *Formulas*: ``name = expression; ...``), numpy
expressions of the variables (``velocity.x``, ``eta``, a built-in above, an earlier
formula), with ``sqrt``, ``abs``, ``mag`` (a vector's length), ``where``,
``maximum``, ``minimum``, ``log``, ``exp``, ``sin``, ``cos``, ``pi``, ``g``, ``rho``.

Gradients come from the cells: each cell's least-squares fit, averaged to its
points (surfaces in plan, volumes in 3D). A volume's derived quantities are worked
out piece by piece before it is cut (slices, iso-surfaces), so they hold inside it.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np

G = 9.81
RHO = 1000.0
#: a water depth below this [m] is dry for the quantities that divide by it
DRY_DEPTH = 1e-3
#: the default Manning's n [s/m^(1/3)] for the bed shear stress
MANNING = 0.025
#: roles of surfaces of shallow flow
FLOW_ROLES = ("free_surface", "amr")
#: roles of surfaces that may carry a velocity
SURFACE_ROLES = ("free_surface", "amr", "bed")

LABELS = {
    "water_depth": ("Water depth", "m"),
    "froude": ("Froude number", ""),
    "unit_discharge": ("Unit discharge h·u", "m²/s"),
    "hazard": ("Flood hazard h·|u|", "m²/s"),
    "bed_shear": ("Bed shear stress", "Pa"),
    "vorticity": ("Vorticity", "1/s"),
    "q_criterion": ("Q-criterion", "1/s²"),
}


@dataclass
class Settings:
    """A case's settings for its derived quantities."""

    manning: float = MANNING
    formulas: list[tuple[str, str]] = field(default_factory=list)


def parse_formulas(text: str) -> list[tuple[str, str]]:
    """``name = expression`` lines (or separated by ``;``) as (name, expression);
    raises ValueError on one that is not."""
    found = []
    for part in re.split(r"[;\n]", text or ""):
        part = part.strip()
        if not part:
            continue
        name, sep, expression = part.partition("=")
        name, expression = name.strip(), expression.strip()
        if not sep or not re.fullmatch(r"[A-Za-z_]\w*", name) or not expression:
            raise ValueError(f"not a formula 'name = expression': {part!r}")
        if "__" in expression:
            raise ValueError(f"{name}: '__' is not allowed in a formula")
        found.append((name, expression))
    return found


# ======================================================================== recipes
def _depth_inputs(components: dict[str, int]) -> list[str] | None:
    if components.get("WL") == 1:
        return ["WL"]
    if components.get("waterlevel") == 1:
        return ["waterlevel"]
    if components.get("depth") == 1 and components.get("eta") == 1:
        return ["depth", "eta"]
    return None


def _depth(data: dict[str, np.ndarray]) -> np.ndarray:
    if "WL" in data:
        h = data["WL"]
    elif "waterlevel" in data:
        h = data["waterlevel"]
    else:
        h = data["depth"] + data["eta"]
    return np.clip(np.asarray(h, np.float64), 0.0, None)


def _horizontal(data) -> np.ndarray:
    u = np.asarray(data["velocity"], np.float64).reshape(len(data["velocity"]), -1)
    return u[:, :2]


@dataclass(frozen=True)
class Recipe:
    """A built-in derived quantity: what it needs and how many components it has."""

    name: str
    components: int
    inputs: tuple[str, ...]
    #: needs the cells (a gradient)
    gradient: bool = False


def recipes(role: str, volume: bool, components: dict[str, int]) -> list[Recipe]:
    """The built-in quantities a dataset has the inputs for."""
    found: list[Recipe] = []
    velocity = components.get("velocity") == 3
    if not velocity:
        return found
    if volume:
        found.append(Recipe("vorticity", 3, ("velocity",), True))
        found.append(Recipe("q_criterion", 1, ("velocity",), True))
        return found
    if role in FLOW_ROLES:
        depth = _depth_inputs(components)
        if depth is not None:
            needs = ("velocity", *depth)
            found += [
                Recipe("water_depth", 1, tuple(depth)),
                Recipe("froude", 1, needs),
                Recipe("unit_discharge", 3, needs),
                Recipe("hazard", 1, needs),
                Recipe("bed_shear", 1, needs),
            ]
    if role in SURFACE_ROLES:
        found.append(Recipe("vorticity", 1, ("velocity",), True))
    return found


def compute_builtin(name: str, data: dict, settings: Settings, gradient=None) -> np.ndarray:
    """A built-in quantity from the point data (``gradient(values, dims)``: the
    gradient at the points, for vorticity and Q)."""
    if name == "water_depth":
        return _depth(data).astype(np.float32)
    if name in ("froude", "unit_discharge", "hazard", "bed_shear"):
        h = _depth(data)
        uv = _horizontal(data)
        speed = np.hypot(uv[:, 0], uv[:, 1])
        wet = h > DRY_DEPTH
        if name == "froude":
            out = np.where(wet, speed / np.sqrt(G * np.where(wet, h, 1.0)), 0.0)
        elif name == "unit_discharge":
            out = np.zeros((len(h), 3))
            out[:, :2] = uv * h[:, None]
        elif name == "hazard":
            out = h * speed
        else:
            n = float(settings.manning)
            out = np.where(wet, RHO * G * n * n * speed**2 / np.cbrt(np.where(wet, h, 1.0)), 0.0)
        return np.asarray(out, np.float32)
    if name in ("vorticity", "q_criterion"):
        u = np.asarray(data["velocity"], np.float64).reshape(len(data["velocity"]), 3)
        volume = gradient.dims == 3
        if not volume:  # a surface: dv/dx - du/dy
            j = gradient(u[:, :2])  # (n, 2, 2): d u_i / d x_j
            return (j[:, 1, 0] - j[:, 0, 1]).astype(np.float32)
        j = gradient(u)  # (n, 3, 3)
        if name == "vorticity":
            curl = np.stack(
                [j[:, 2, 1] - j[:, 1, 2], j[:, 0, 2] - j[:, 2, 0], j[:, 1, 0] - j[:, 0, 1]],
                axis=1,
            )
            return curl.astype(np.float32)
        s = 0.5 * (j + np.transpose(j, (0, 2, 1)))
        w = 0.5 * (j - np.transpose(j, (0, 2, 1)))
        q = 0.5 * ((w * w).sum(axis=(1, 2)) - (s * s).sum(axis=(1, 2)))
        return q.astype(np.float32)
    raise KeyError(name)


# ======================================================================== formulas
_FUNCTIONS = {
    "sqrt": np.sqrt,
    "abs": np.abs,
    "where": np.where,
    "maximum": np.maximum,
    "minimum": np.minimum,
    "log": np.log,
    "log10": np.log10,
    "exp": np.exp,
    "sin": np.sin,
    "cos": np.cos,
    "tan": np.tan,
    "arctan2": np.arctan2,
    "clip": np.clip,
    "mag": lambda v: np.linalg.norm(np.asarray(v, np.float64), axis=-1, keepdims=True),
    "pi": np.pi,
    "g": G,
    "rho": RHO,
}
_COMPONENT = re.compile(r"\b([A-Za-z_]\w*)\.([xyz])\b")
_NAME = re.compile(r"\b([A-Za-z_]\w*)\b")


def _prepared(expression: str) -> str:
    """Components as columns (n, 1), so they mix with scalars (n, 1) and vectors (n, 3)."""
    return _COMPONENT.sub(
        lambda m: f"{m.group(1)}[..., {'xyz'.index(m.group(2))}:{'xyz'.index(m.group(2)) + 1}]",
        expression,
    )


def formula_inputs(expression: str) -> set[str]:
    """The variable names a formula uses (not its functions and constants)."""
    names = set(_NAME.findall(_COMPONENT.sub(lambda m: m.group(1), expression)))
    return {n for n in names if n not in _FUNCTIONS}


def evaluate(expression: str, data: dict[str, np.ndarray], count: int) -> np.ndarray:
    """A formula over point data of ``count`` points: (count,) or (count, 3)."""
    namespace = dict(_FUNCTIONS)
    used = formula_inputs(expression)
    for name, values in data.items():
        if name in used:
            values = np.asarray(values, np.float64)
            namespace[name] = values.reshape(len(values), -1)  # scalars as (n, 1)
    try:
        with np.errstate(all="ignore"):
            value = eval(  # noqa: S307 (the user's own formula, without builtins)
                compile(_prepared(expression), "<formula>", "eval"),
                {"__builtins__": {}},
                namespace,
            )
    except Exception as error:  # noqa: BLE001 (any mistake in a formula)
        raise ValueError(f"{expression!r}: {error}") from None
    value = np.asarray(value, np.float64)
    if value.ndim == 0:
        value = np.full(count, float(value))
    if value.ndim == 2 and value.shape[1] == 1:
        value = value[:, 0]
    if value.shape[0] != count or value.ndim > 2 or (value.ndim == 2 and value.shape[1] != 3):
        raise ValueError(f"{expression!r}: gives shape {value.shape}, not a value per point")
    return value.astype(np.float32)


# ======================================================================== the plan
@dataclass
class Plan:
    """What a dataset can derive: name -> components, the built-ins' recipes, the
    formulas (in order), and the formulas that do not work (name -> why)."""

    components: dict[str, int] = field(default_factory=dict)
    builtin: dict[str, Recipe] = field(default_factory=dict)
    formulas: list[tuple[str, str]] = field(default_factory=list)
    problems: dict[str, str] = field(default_factory=dict)
    settings: Settings = field(default_factory=Settings)

    def inputs(self, names) -> set[str]:
        """The arrays on file the derived ``names`` need (formulas through the
        built-ins and earlier formulas they use)."""
        need: set[str] = set()
        todo = [n for n in names if n in self.components]
        seen = set()
        while todo:
            name = todo.pop()
            if name in seen:
                continue
            seen.add(name)
            if name in self.builtin:
                need.update(self.builtin[name].inputs)
                continue
            expression = dict(self.formulas).get(name)
            if expression is None:
                need.add(name)
                continue
            for used in formula_inputs(expression):
                if used in self.components and used != name:
                    todo.append(used)
                else:
                    need.add(used)
        return {n for n in need if n not in self.components}

    def needs_cells(self, names) -> bool:
        """Whether a derived name (or one it uses) is a gradient."""
        todo, seen = [n for n in names if n in self.components], set()
        while todo:
            name = todo.pop()
            if name in seen:
                continue
            seen.add(name)
            if name in self.builtin:
                if self.builtin[name].gradient:
                    return True
                continue
            expression = dict(self.formulas).get(name, "")
            todo += [u for u in formula_inputs(expression) if u in self.components and u != name]
        return False

    def compute(self, data: dict, names, gradient=None) -> None:
        """Add the derived ``names`` (and what they use) to point data, in place."""
        wanted = self._closure(names)
        count = _count(data)
        if count is None:
            return
        for name in self.builtin:
            if name in wanted and name not in data:
                if self.builtin[name].gradient and gradient is None:
                    continue
                data[name] = compute_builtin(name, data, self.settings, gradient)
        for name, expression in self.formulas:
            if name in wanted and name not in data:
                try:
                    data[name] = evaluate(expression, data, count)
                except ValueError:
                    continue

    def _closure(self, names) -> set[str]:
        out, todo = set(), [n for n in names if n in self.components]
        while todo:
            name = todo.pop()
            if name in out:
                continue
            out.add(name)
            expression = dict(self.formulas).get(name, "")
            todo += [u for u in formula_inputs(expression) if u in self.components]
        return out


def _count(data: dict) -> int | None:
    for values in data.values():
        return len(values)
    return None


def plan_for(role: str, volume: bool, raw: dict[str, int], settings: Settings) -> Plan:
    """What a dataset with these arrays on file can derive."""
    plan = Plan(settings=settings)
    for recipe in recipes(role, volume, raw):
        if recipe.name in raw:
            continue  # written by the run itself
        plan.builtin[recipe.name] = recipe
        plan.components[recipe.name] = recipe.components
    known = {**raw, **plan.components}
    for name, expression in settings.formulas:
        if name in raw:
            plan.problems[name] = f"{name}: the run writes a variable of this name"
            continue
        used = formula_inputs(expression)
        missing = sorted(u for u in used if u not in known)
        if missing:
            plan.problems[name] = f"{name}: no {', '.join(missing)} here"
            continue
        probe = {u: np.ones((2, 3)) if known[u] == 3 else np.ones(2) for u in used}
        try:
            value = evaluate(expression, probe, 2)
        except ValueError as error:
            plan.problems[name] = f"{name}: {error}"
            continue
        plan.formulas.append((name, expression))
        plan.components[name] = 3 if value.ndim == 2 else 1
        known[name] = plan.components[name]
    return plan


# ======================================================================== gradients
def cell_groups(connectivity, offsets) -> list[np.ndarray]:
    """VTK cells (connectivity, offsets: their ends) grouped by their number of
    points: arrays (cells, points)."""
    connectivity = np.asarray(connectivity, np.int64)
    ends = np.asarray(offsets, np.int64)
    starts = np.concatenate([[0], ends[:-1]])
    sizes = ends - starts
    groups = []
    for size in np.unique(sizes):
        if size < 3:
            continue
        chosen = starts[sizes == size]
        groups.append(connectivity[chosen[:, None] + np.arange(size)[None, :]])
    return groups


class Gradient:
    """The gradient of point values over cells: each cell's least-squares plane
    (or linear field in 3D), averaged to its points. ``dims`` 2: in plan (x, y), for
    surfaces; 3 for volumes. Cells are taken some :data:`BATCH` at a time, so a large
    volume does not need its whole fit in memory at once."""

    BATCH = 200_000

    def __init__(self, points: np.ndarray, groups: list[np.ndarray], dims: int) -> None:
        self.points = np.asarray(points, np.float64)[:, :dims]
        self.groups = [np.asarray(g, np.int64) for g in groups if g.shape[1] > dims]
        self.dims = dims

    def __call__(self, values: np.ndarray) -> np.ndarray:
        """(n, c) values -> (n, c, dims): d value_c / d x_j at the points."""
        values = np.asarray(values, np.float64)
        n, c = values.shape
        total = np.zeros((n, c, self.dims))
        count = np.zeros(n)
        for group in self.groups:
            for start in range(0, len(group), self.BATCH):
                cells = group[start : start + self.BATCH]
                x = self.points[cells]  # (m, k, dims)
                x = x - x.mean(axis=1, keepdims=True)
                f = values[cells]  # (m, k, c)
                f = f - f.mean(axis=1, keepdims=True)
                # the normal equations (X^T X) g = X^T f, cell by cell
                a = np.einsum("mkd,mke->mde", x, x)
                b = np.einsum("mkd,mkc->mdc", x, f)
                ok = np.abs(np.linalg.det(a)) > 1e-12 * np.einsum("mdd->m", a) ** self.dims
                g = np.zeros((len(cells), self.dims, c))
                if ok.any():
                    g[ok] = np.linalg.solve(a[ok], b[ok])
                g = np.transpose(g, (0, 2, 1))  # (m, c, dims)
                ok &= np.isfinite(g).all(axis=(1, 2))
                cells, g = cells[ok], g[ok]
                k = cells.shape[1]
                where = cells.ravel()
                spread = np.repeat(g.reshape(len(g), -1), k, axis=0)  # (m k, c dims)
                for column in range(c * self.dims):
                    total.reshape(n, -1)[:, column] += np.bincount(
                        where, weights=spread[:, column], minlength=n
                    )
                count += np.bincount(where, minlength=n)
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.where(count[:, None, None] > 0, total / count[:, None, None], 0.0)


# ======================================================================== datasets
#: roles of outputs without derived quantities (objects, the set-up's)
PLAIN = ("body", "mooring", "structure", "dem", "ice", "rods", "particles", "geodat", "solid",
         "topo", "seastate")  # fmt: skip


def attach(dataset, settings: Settings | None = None) -> Plan:
    """Give a dataset its derived quantities (again, for new settings): they join its
    variables, and its reading works them out when asked for."""
    settings = settings or Settings()
    if getattr(dataset, "role", "") in PLAIN:
        return Plan(settings=settings)
    raw = getattr(dataset, "_raw_components", None)
    if raw is None:
        raw = dict(dataset.components)
        dataset._raw_components = raw
        _wrap(dataset)
    plan = plan_for(dataset.role, _is_volume(dataset), raw, settings)
    dataset._derived = plan
    dataset.components = {**raw, **plan.components}
    return plan


def _is_volume(dataset) -> bool:
    return getattr(dataset, "kind", "") == "UnstructuredGrid"


def _names(variables) -> list[str]:
    return [v.partition(".")[0] for v in variables or [] if v and v != "none"]


def _wrap(dataset) -> None:
    """Reading that works the derived quantities out (instance methods in front of
    the dataset's own)."""
    from lagoonviz.core.data.reef3d import Dataset

    volume = _is_volume(dataset)
    own_read = dataset.read
    own_slice = getattr(dataset, "read_slice", None)
    own_values = getattr(dataset, "read_values", None)
    own_sample = getattr(dataset, "sample_values", None)
    own_piece = getattr(dataset, "_piece", None)
    #: a store's own fast paths (faces, slices) read arrays straight from it
    store = type(dataset).read is not Dataset.read

    def asked(variables) -> list[str]:
        plan = dataset._derived
        return [n for n in _names(variables) if n in plan.components]

    def raw_names(variables, plan) -> list[str]:
        names = [v for v in variables if v.partition(".")[0] not in plan.components]
        return names + sorted(plan.inputs(asked(variables)))

    def read(index=0, variables=None, **kw):
        plan = dataset._derived
        wanted = asked(variables)
        if not wanted:
            return own_read(index, variables, **kw)
        if volume:
            if store:  # through the pieces, where the derived quantities are made
                return Dataset.read(dataset, index, variables)
            return own_read(index, variables, **kw)
        mesh = own_read(index, raw_names(variables, plan), **kw)
        gradient = None
        if plan.needs_cells(wanted) and mesh.triangles.size:
            gradient = Gradient(mesh.points, [np.asarray(mesh.triangles, np.int64)], 2)
        plan.compute(mesh.point_data, wanted, gradient)
        return mesh

    dataset.read = read

    if own_slice is not None and volume and store:

        def read_slice(index, axis, position, variables=None):
            if asked(variables):
                return Dataset.read_slice(dataset, index, axis, position, variables)
            return own_slice(index, axis, position, variables)

        dataset.read_slice = read_slice

    if own_values is not None:

        def read_values(index, variable):
            if asked([variable]):
                return dataset.read(index, [variable]).scalar(variable)
            return own_values(index, variable)

        dataset.read_values = read_values

    if own_sample is not None:

        def sample_values(index, variable):
            if asked([variable]):
                return dataset.read(index, [variable]).scalar(variable)
            return own_sample(index, variable)

        dataset.sample_values = sample_values

    own_series = getattr(dataset, "point_series", None)
    if own_series is not None:

        def point_series(variable, x, y):
            """A derived quantity over the run at the point nearest to (x, y): every
            frame read for it (slower than a variable on file)."""
            if not asked([variable]):
                return own_series(variable, x, y)
            first = dataset.read(0, [variable])
            d = (first.points[:, 0] - x) ** 2 + (first.points[:, 1] - y) ** 2
            k = int(np.argmin(d))
            where = tuple(float(v) for v in first.points[k])
            values = np.full(len(dataset.frames), np.nan)
            for index in range(len(dataset.frames)):
                mesh = first if index == 0 else dataset.read(index, [variable])
                if k < mesh.n_points:
                    values[index] = float(mesh.scalar(variable)[k])
            return where, values

        dataset.point_series = point_series

    if own_piece is not None and volume:
        numbers: dict = {}

        def number_of(path) -> int:
            if path not in numbers:
                for frame in dataset.frames:
                    for k, p in enumerate(frame.pieces):
                        numbers.setdefault(p, k)
            return numbers.get(path, 0)

        def piece(path, point_arrays=None, with_cells=True):
            plan = dataset._derived
            wanted = asked(point_arrays) if point_arrays is not None else []
            if not wanted:
                return own_piece(path, point_arrays=point_arrays, with_cells=with_cells)
            names = [a for a in point_arrays if a not in plan.components]
            names = sorted(set(names) | plan.inputs(wanted))
            found = own_piece(path, point_arrays=names, with_cells=with_cells)
            gradient = None
            if plan.needs_cells(wanted) and len(found.points):
                if with_cells and found.connectivity is not None and len(found.connectivity):
                    connectivity, offsets = found.connectivity, found.offsets
                else:
                    _, connectivity, offsets, _ = dataset._piece_cells(number_of(path), path, found)
                gradient = Gradient(found.points, cell_groups(connectivity, offsets), 3)
            plan.compute(found.point_data, wanted, gradient)
            return found

        dataset._piece = piece
