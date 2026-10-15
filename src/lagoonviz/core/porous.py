# SPDX-License-Identifier: GPL-3.0-or-later
# Architect: Hans Bihs
"""Porous structures of a REEF3D case (VRANS): breakwaters, rubble mounds, porous
cylinders, as the lines of ctrl.txt that make them.

REEF3D::CFD gives each structure its own material on its line::

    B 270 xs xe ys ye zs ze  n d50 α β        box
    B 274 xc yc zs ze r      n d50 α β        vertical cylinder
    B 281 xs xe ys ye zs ze  n d50 α β        wedge along x
    B 282 xs xe ys ye zs ze  n d50 α β        wedge along y
    B 291 xs xe ys ye zs ze d  n d50 α β      plate along x, d thick

REEF3D::NHFLOW takes the shapes only (B 210 - B 219); one material for all of them
is ``B 201 n d50 α β``. A wedge's top runs from height zs at its start to ze at its
end, its base at the lower of the two; a plate is the band of thickness d above
that line.

n is the porosity, d50 the mean stone diameter in m, α and β the linear and
nonlinear resistance coefficients of the Forchheimer terms (REEF3D's vrans_ini.cpp):

    A = α (1-n)² / (n³ d50²),   B = β (1 + 7.5/KC) (1-n) / (n³ d50)
"""

from __future__ import annotations

from dataclasses import dataclass

#: the material: porosity, mean stone diameter, linear and nonlinear coefficients
MATERIAL = ("porosity n", "d50 [m]", "α", "β")
#: a rubble mound of quarry stone (α and β as Jensen et al. 2014 used them in REEF3D)
DEFAULT_MATERIAL = (0.4, 0.05, 500.0, 2.0)
#: NHFLOW's one material for all its porous structures
NHFLOW_MATERIAL = "B 201"

_BOX = ("x start [m]", "x end [m]", "y start [m]", "y end [m]", "z start [m]", "z end [m]")
_WEDGE_X = (*_BOX[:4], "z at x start [m]", "z at x end [m]")
_WEDGE_Y = (*_BOX[:4], "z at y start [m]", "z at y end [m]")


@dataclass(frozen=True)
class PorousKind:
    """One kind of porous structure: its ctrl.txt command and the values of a line."""

    code: str
    name: str
    #: how it is drawn and made from an area: "box", "zcyl", "ycyl", "sphere",
    #: "wedge_x", "wedge_y", "plate_x"
    shape: str
    #: the names of the shape's values, in the order of the line
    geometry: tuple[str, ...]
    #: the solver that reads it: "CFD" (material on the line) or "NHFLOW" (B 201)
    solver: str

    @property
    def material_on_line(self) -> bool:
        return self.solver == "CFD"

    @property
    def count(self) -> int:
        """The number of values of a line."""
        return len(self.geometry) + (len(MATERIAL) if self.material_on_line else 0)

    def label(self) -> str:
        return f"{self.name} ({self.code})"

    def split(self, values) -> tuple[list[float], list[float]]:
        """A line's values as (shape, material); the material is empty for NHFLOW."""
        values = [float(v) for v in values]
        n = len(self.geometry)
        return values[:n], values[n : n + len(MATERIAL)] if self.material_on_line else []

    def join(self, geometry, material=None) -> list[float]:
        """The values of a line from its shape and (CFD) its material."""
        values = [float(v) for v in geometry]
        if len(values) != len(self.geometry):
            raise ValueError(f"{self.code} takes {len(self.geometry)} shape values")
        if self.material_on_line:
            material = DEFAULT_MATERIAL if material is None else material
            values += [float(v) for v in material]
        return values

    def describe(self, values) -> str:
        """A short text of where it is: "x 0.30 – 0.59 m, z 0 – 0.58 m"."""
        g, m = self.split(values)
        if len(g) < len(self.geometry):
            return "incomplete line"
        if self.shape in ("box", "wedge_x", "wedge_y", "plate_x"):
            text = f"x {g[0]:g} – {g[1]:g}, y {g[2]:g} – {g[3]:g}, z {g[4]:g} – {g[5]:g} m"
        elif self.shape == "zcyl":
            text = f"at ({g[0]:g}, {g[1]:g}), r {g[4]:g}, z {g[2]:g} – {g[3]:g} m"
        elif self.shape == "ycyl":
            text = f"at x {g[0]:g}, z {g[1]:g}, r {g[4]:g}, y {g[2]:g} – {g[3]:g} m"
        else:  # sphere
            text = f"at ({g[0]:g}, {g[1]:g}, {g[2]:g}), r {g[3]:g} m"
        if m:
            text += f"; n {m[0]:g}, d50 {m[1]:g} m"
        return text


KINDS = (
    PorousKind("B 270", "Box", "box", _BOX, "CFD"),
    PorousKind(
        "B 274",
        "Vertical cylinder",
        "zcyl",
        ("x centre [m]", "y centre [m]", "z start [m]", "z end [m]", "radius [m]"),
        "CFD",
    ),
    PorousKind("B 281", "Wedge along x", "wedge_x", _WEDGE_X, "CFD"),
    PorousKind("B 282", "Wedge along y", "wedge_y", _WEDGE_Y, "CFD"),
    PorousKind("B 291", "Plate along x", "plate_x", (*_WEDGE_X, "thickness [m]"), "CFD"),
    PorousKind("B 210", "Box", "box", _BOX, "NHFLOW"),
    PorousKind(
        "B 212",
        "Cylinder along y",
        "ycyl",
        ("x centre [m]", "z centre [m]", "y start [m]", "y end [m]", "radius [m]"),
        "NHFLOW",
    ),
    PorousKind(
        "B 213",
        "Vertical cylinder",
        "zcyl",
        ("x centre [m]", "y centre [m]", "z start [m]", "z end [m]", "radius [m]"),
        "NHFLOW",
    ),
    PorousKind(
        "B 215",
        "Sphere",
        "sphere",
        ("x centre [m]", "y centre [m]", "z centre [m]", "radius [m]"),
        "NHFLOW",
    ),
    PorousKind("B 217", "Wedge along x", "wedge_x", _WEDGE_X, "NHFLOW"),
    PorousKind("B 218", "Wedge along y", "wedge_y", _WEDGE_Y, "NHFLOW"),
)
BY_CODE = {k.code: k for k in KINDS}


def kinds_for(solver: str) -> list[PorousKind]:
    """The porous structures the REEF3D module (A 10: "CFD", "NHFLOW", ...) reads;
    none for SFLOW and FNPF."""
    return [k for k in KINDS if k.solver == solver.upper()]


@dataclass(frozen=True)
class Structure:
    """A porous structure of a case: its kind and which line of its command."""

    kind: PorousKind
    occurrence: int
    values: tuple[float, ...]

    def label(self) -> str:
        return f"{self.kind.name} ({self.kind.code} #{self.occurrence + 1})"


def structures(ctrl, solver: str = "") -> list[Structure]:
    """The porous structures in a ctrl.txt (a ControlFile), in the order of KINDS;
    only those of ``solver`` when given."""
    found = []
    for kind in KINDS if not solver else kinds_for(solver):
        for occurrence, line in enumerate(ctrl.entries(kind.code)):
            found.append(Structure(kind, occurrence, tuple(line.numbers() or ())))
    return found


def snap(value: float, step: float) -> float:
    """A coordinate rounded to the cell size (centimetres without one)."""
    step = step if step and step > 0 else 0.01
    return round(round(value / step) * step, 6)


def from_area(kind: PorousKind, area, heights, material=None) -> list[float]:
    """A line of ``kind`` filling a plan area (x0, x1, y0, y1) between two heights
    (bottom, top): a box or wedge over the area, the largest cylinder or sphere in
    it. A wedge rises from the bottom at its start to the top at its end."""
    x0, x1, y0, y1 = (float(v) for v in area)
    x0, x1 = sorted((x0, x1))
    y0, y1 = sorted((y0, y1))
    zb, zt = sorted(float(v) for v in heights)
    xc, yc, zc = 0.5 * (x0 + x1), 0.5 * (y0 + y1), 0.5 * (zb + zt)
    if kind.shape in ("box", "wedge_x", "wedge_y"):
        geometry = [x0, x1, y0, y1, zb, zt]
    elif kind.shape == "plate_x":  # a slab a tenth of the height thick
        geometry = [x0, x1, y0, y1, zb, zt, round(0.1 * (zt - zb), 6) or 0.01]
    elif kind.shape == "zcyl":
        geometry = [xc, yc, zb, zt, 0.5 * min(x1 - x0, y1 - y0)]
    elif kind.shape == "ycyl":
        geometry = [xc, zc, y0, y1, 0.5 * min(x1 - x0, zt - zb)]
    elif kind.shape == "sphere":
        geometry = [xc, yc, zc, 0.5 * min(x1 - x0, y1 - y0, zt - zb)]
    else:
        raise ValueError(kind.shape)
    return kind.join([round(v, 6) for v in geometry], material)


def default_area(domain, two_d: bool = False):
    """Where a new structure goes without one drawn: the middle fifth of the domain
    in x, across all of it in a 2D case, the middle fifth in y otherwise."""
    xs, xe, ys, ye = (float(v) for v in domain[:4])
    lx, ly = xe - xs, ye - ys
    x0, x1 = xs + 0.4 * lx, xs + 0.6 * lx
    y0, y1 = (ys, ye) if two_d else (ys + 0.4 * ly, ys + 0.6 * ly)
    return x0, x1, y0, y1
