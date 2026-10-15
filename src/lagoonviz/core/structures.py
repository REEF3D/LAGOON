# SPDX-License-Identifier: GPL-3.0-or-later
# Architect: Hans Bihs
"""Structures of a case set up by shape: porous structures (VRANS) and solids, as
the input-file lines that make them.

**Porous** (ctrl.txt). REEF3D::CFD gives each structure its own material on its
line::

    B 270 xs xe ys ye zs ze  n d50 α β        box
    B 274 xc yc zs ze r      n d50 α β        vertical cylinder
    B 281 xs xe ys ye zs ze  n d50 α β        wedge along x
    B 282 xs xe ys ye zs ze  n d50 α β        wedge along y
    B 291 xs xe ys ye zs ze d  n d50 α β      plate along x, d thick

REEF3D::NHFLOW takes the shapes only (B 210 - B 218); one material for all of them
is ``B 201 n d50 α β``. n is the porosity, d50 the mean stone diameter in m, α and
β the linear and nonlinear resistance coefficients of the Forchheimer terms
(REEF3D's vrans_ini.cpp): A = α (1-n)² / (n³ d50²), B = β (1 + 7.5/KC) (1-n) / (n³
d50).

**Solids**. DIVEMesh's solids (control.txt, every module; NHFLOW takes them with
A 580): box S 10, cylinder along y across the domain S 32 (xc zc r), vertical
cylinder over the whole height S 33 (xc yc r), sphere S 51, wedges S 61 and S 62.
REEF3D::NHFLOW's direct forcing solids (ctrl.txt): box A 581, cylinders A 583 (y)
and A 584 (z), sphere A 586, wedges A 587 and A 588.

A wedge's top runs from height zs at its start to ze at its end, its base at the
lower of the two (DIVEMesh's and REEF3D's geo_primitive); a plate is the band of
thickness d above that line.
"""

from __future__ import annotations

from dataclasses import dataclass

CTRL, CONTROL = "ctrl.txt", "control.txt"
#: the material of a porous structure: porosity, stone diameter, coefficients
MATERIAL = ("porosity n", "d50 [m]", "α", "β")
#: a rubble mound of quarry stone (α and β as Jensen et al. 2014 used them in REEF3D)
DEFAULT_MATERIAL = (0.4, 0.05, 500.0, 2.0)
#: NHFLOW's one material for all its porous structures
NHFLOW_MATERIAL = "B 201"

_BOX = ("x start [m]", "x end [m]", "y start [m]", "y end [m]", "z start [m]", "z end [m]")
_WEDGE_X = (*_BOX[:4], "z at x start [m]", "z at x end [m]")
_WEDGE_Y = (*_BOX[:4], "z at y start [m]", "z at y end [m]")
_ZCYL = ("x centre [m]", "y centre [m]", "z start [m]", "z end [m]", "radius [m]")
_YCYL = ("x centre [m]", "z centre [m]", "y start [m]", "y end [m]", "radius [m]")
_SPHERE = ("x centre [m]", "y centre [m]", "z centre [m]", "radius [m]")


@dataclass(frozen=True)
class Kind:
    """One kind of structure: its command and the values of a line."""

    code: str
    name: str
    #: how it is drawn and made from an area: "box", "zcyl", "ycyl", "sphere",
    #: "wedge_x", "wedge_y", "plate_x", "zcyl_full" (xc yc r), "ycyl_full" (xc zc r)
    shape: str
    #: the names of the shape's values, in the order of the line
    geometry: tuple[str, ...]
    #: "porous" or "solid"
    group: str
    #: the file of the command
    file: str = CTRL
    #: the REEF3D modules that read it (empty: every one, DIVEMesh's)
    modules: tuple[str, ...] = ()
    #: the material: "line" (on the line, CFD), "B 201" (NHFLOW) or "" (a solid)
    material: str = ""

    @property
    def material_on_line(self) -> bool:
        return self.material == "line"

    @property
    def count(self) -> int:
        """The number of values of a line."""
        return len(self.geometry) + (len(MATERIAL) if self.material_on_line else 0)

    def label(self) -> str:
        return f"{self.name} ({self.code})"

    def phrase(self) -> str:
        """Its name within a sentence: "vertical cylinder (DIVEMesh) (S 33)"."""
        return f"{self.name[:1].lower()}{self.name[1:]} ({self.code})"

    def split(self, values) -> tuple[list[float], list[float]]:
        """A line's values as (shape, material); no material but on CFD's lines."""
        values = [float(v) for v in values]
        n = len(self.geometry)
        return values[:n], values[n : n + len(MATERIAL)] if self.material_on_line else []

    def join(self, geometry, material=None) -> list[float]:
        """The values of a line from its shape and (CFD porous) its material."""
        values = [float(v) for v in geometry]
        if len(values) != len(self.geometry):
            raise ValueError(f"{self.code} takes {len(self.geometry)} shape values")
        if self.material_on_line:
            material = DEFAULT_MATERIAL if material is None else material
            values += [float(v) for v in material]
        return values

    def describe(self, values) -> str:
        """A short text of where it is: "x 0.30 – 0.59, y 0 – 1, z 0 – 0.58 m"."""
        g, m = self.split(values)
        if len(g) < len(self.geometry):
            return "incomplete line"
        if self.shape in ("box", "wedge_x", "wedge_y", "plate_x"):
            text = f"x {g[0]:g} – {g[1]:g}, y {g[2]:g} – {g[3]:g}, z {g[4]:g} – {g[5]:g} m"
        elif self.shape == "zcyl":
            text = f"at ({g[0]:g}, {g[1]:g}), r {g[4]:g}, z {g[2]:g} – {g[3]:g} m"
        elif self.shape == "ycyl":
            text = f"at x {g[0]:g}, z {g[1]:g}, r {g[4]:g}, y {g[2]:g} – {g[3]:g} m"
        elif self.shape == "zcyl_full":
            text = f"at ({g[0]:g}, {g[1]:g}), r {g[2]:g} m, over the height"
        elif self.shape == "ycyl_full":
            text = f"at x {g[0]:g}, z {g[1]:g}, r {g[2]:g} m, across the domain"
        else:  # sphere
            text = f"at ({g[0]:g}, {g[1]:g}, {g[2]:g}), r {g[3]:g} m"
        if m:
            text += f"; n {m[0]:g}, d50 {m[1]:g} m"
        return text


def _porous(code, name, shape, geometry, module):
    material = "line" if module == "CFD" else NHFLOW_MATERIAL
    return Kind(code, name, shape, geometry, "porous", CTRL, (module,), material)


KINDS = (
    # porous: CFD, each with its material
    _porous("B 270", "Box", "box", _BOX, "CFD"),
    _porous("B 274", "Vertical cylinder", "zcyl", _ZCYL, "CFD"),
    _porous("B 281", "Wedge along x", "wedge_x", _WEDGE_X, "CFD"),
    _porous("B 282", "Wedge along y", "wedge_y", _WEDGE_Y, "CFD"),
    _porous("B 291", "Plate along x", "plate_x", (*_WEDGE_X, "thickness [m]"), "CFD"),
    # porous: NHFLOW, B 201 for all
    _porous("B 210", "Box", "box", _BOX, "NHFLOW"),
    _porous("B 212", "Cylinder along y", "ycyl", _YCYL, "NHFLOW"),
    _porous("B 213", "Vertical cylinder", "zcyl", _ZCYL, "NHFLOW"),
    _porous("B 215", "Sphere", "sphere", _SPHERE, "NHFLOW"),
    _porous("B 217", "Wedge along x", "wedge_x", _WEDGE_X, "NHFLOW"),
    _porous("B 218", "Wedge along y", "wedge_y", _WEDGE_Y, "NHFLOW"),
    # solids: NHFLOW's direct forcing
    Kind("A 581", "Box, direct forcing", "box", _BOX, "solid", CTRL, ("NHFLOW",)),
    Kind("A 583", "Cylinder along y, direct forcing", "ycyl", _YCYL, "solid", CTRL, ("NHFLOW",)),
    Kind("A 584", "Vertical cylinder, direct forcing", "zcyl", _ZCYL, "solid", CTRL, ("NHFLOW",)),
    Kind("A 586", "Sphere, direct forcing", "sphere", _SPHERE, "solid", CTRL, ("NHFLOW",)),
    Kind("A 587", "Wedge along x, direct forcing", "wedge_x", _WEDGE_X, "solid", CTRL, ("NHFLOW",)),
    Kind("A 588", "Wedge along y, direct forcing", "wedge_y", _WEDGE_Y, "solid", CTRL, ("NHFLOW",)),
    # solids: DIVEMesh's, for every module
    Kind("S 10", "Box (DIVEMesh)", "box", _BOX, "solid", CONTROL),
    Kind(
        "S 33",
        "Vertical cylinder (DIVEMesh)",
        "zcyl_full",
        ("x centre [m]", "y centre [m]", "radius [m]"),
        "solid",
        CONTROL,
    ),
    Kind(
        "S 32",
        "Cylinder along y (DIVEMesh)",
        "ycyl_full",
        ("x centre [m]", "z centre [m]", "radius [m]"),
        "solid",
        CONTROL,
    ),
    Kind("S 51", "Sphere (DIVEMesh)", "sphere", _SPHERE, "solid", CONTROL),
    Kind("S 61", "Wedge along x (DIVEMesh)", "wedge_x", _WEDGE_X, "solid", CONTROL),
    Kind("S 62", "Wedge along y (DIVEMesh)", "wedge_y", _WEDGE_Y, "solid", CONTROL),
)
BY_CODE = {k.code: k for k in KINDS}


def kinds_for(group: str, solver: str) -> list[Kind]:
    """The structures of a group ("porous", "solid") the REEF3D module (A 10:
    "CFD", "NHFLOW", ...; "" without ctrl.txt) reads. Porous: CFD and NHFLOW only.
    Solids: DIVEMesh's for every module, NHFLOW's direct forcing first."""
    solver = solver.upper()
    return [k for k in KINDS if k.group == group and (not k.modules or solver in k.modules)]


@dataclass(frozen=True)
class Structure:
    """A structure of a case: its kind and which line of its command."""

    kind: Kind
    occurrence: int
    values: tuple[float, ...]

    def label(self) -> str:
        return f"{self.kind.name} ({self.kind.code} #{self.occurrence + 1})"


def structures(files, kinds) -> list[Structure]:
    """The structures of ``kinds`` in a case's input files ({file name:
    ControlFile}), in the order of the kinds."""
    found = []
    for kind in kinds:
        control = files.get(kind.file)
        if control is None:
            continue
        for occurrence, line in enumerate(control.entries(kind.code)):
            found.append(Structure(kind, occurrence, tuple(line.numbers() or ())))
    return found


def snap(value: float, step: float) -> float:
    """A coordinate rounded to the cell size (centimetres without one)."""
    step = step if step and step > 0 else 0.01
    return round(round(value / step) * step, 6)


def from_area(kind: Kind, area, heights, material=None) -> list[float]:
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
    elif kind.shape == "zcyl_full":
        geometry = [xc, yc, 0.5 * min(x1 - x0, y1 - y0)]
    elif kind.shape == "ycyl":
        geometry = [xc, zc, y0, y1, 0.5 * min(x1 - x0, zt - zb)]
    elif kind.shape == "ycyl_full":
        geometry = [xc, zc, 0.5 * min(x1 - x0, zt - zb)]
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
