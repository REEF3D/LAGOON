# SPDX-License-Identifier: GPL-3.0-or-later
# Architect: Hans Bihs
"""The solids of a case as DIVEMesh made them: structures (a monopile, a breakwater)
and the blocks that shape the bed (a sloping beach).

DIVEMesh writes every solid of ``control.txt`` (the S commands and an STL geometry)
as triangles into ``DIVEMesh_Paraview/REEF3D_Solid.vtp``; that file is read when it
is there. Without it (DIVEMesh run elsewhere), the common solids are built from
``control.txt`` itself, as DIVEMesh builds them: boxes (S 10), cylinders (S 32,
S 33), cones (S 41 - S 43), spheres (S 51), wedges (S 61 - S 63) and polyhedra from
their corners (S 81 - S 84). Cylinders and cones without their own extent span the
domain (B 10).

Only a case's own solids are shown (:func:`find_solids`): in a folder where several
cases run, the solid file there can be another case's.

The faces of solids that make the bed lie in the bed REEF3D writes; they are left
out (:func:`bed_faces`), so the two never fight over the same place on screen.
"""

from __future__ import annotations

import logging
import math
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)

#: where DIVEMesh writes its solids, relative to the case folder
SOLID_VTP = Path("DIVEMesh_Paraview") / "REEF3D_Solid.vtp"
#: what DIVEMesh writes at the end of every run (its partition and grid preview)
RUN_MARKS = (
    Path("DIVEMesh_Paraview") / "DIVEMesh_Partition.vtp",
    Path("DIVEMesh_Paraview") / "DIVEMesh_grid-preview.vts",
)
#: seconds a solid or bed file may be older than a run's last files and still be its
STALE_AFTER = 3600.0
#: segments around cylinders, cones and spheres built from control.txt
SEGMENTS = 48


def find_solids(case_dir: str | Path) -> Path | None:
    """DIVEMesh's solid file of a case, else its control.txt if that has solids.

    Only the case's own: DIVEMesh leaves ``REEF3D_Solid.vtp`` alone when a case has
    no solids, so in a folder where several cases run (the home folder) the file may
    be another case's. It is passed over when DIVEMesh has run since without writing
    it (:func:`left_over`), and when control.txt has no solid commands. With solids
    in control.txt, the file is taken unless it is older than control.txt (DIVEMesh
    not run since): then the solids are built from control.txt, if it has ones that
    can be built. Without control.txt (results copied without their input), a file
    that is not left over is taken.
    """
    case_dir = Path(case_dir)
    vtp = case_dir / SOLID_VTP
    if vtp.is_file() and left_over(vtp, case_dir):
        log.info("%s: not shown, DIVEMesh has run since without writing it", vtp)
        vtp_found = False
    else:
        vtp_found = vtp.is_file()
    control = case_dir / "control.txt"
    if not control.is_file():
        return vtp if vtp_found else None
    solids = read_commands(control)[1]
    if not solids:
        if vtp_found:
            log.info("%s: not shown, control.txt has no solids", vtp)
        return None
    buildable = any(code in BUILDERS for code, _ in solids)
    if vtp_found and (not buildable or not _older(vtp, control)):
        return vtp
    return control if buildable else None


def left_over(path: Path, case_dir: str | Path, after: float = STALE_AFTER) -> bool:
    """True if a file DIVEMesh writes only for some cases (``REEF3D_Solid.vtp``,
    ``REEF3D_Topo.vtp``) is left over from an earlier DIVEMesh run: older than the
    files it writes at the end of every run (:data:`RUN_MARKS`) by more than
    ``after`` seconds (a run takes minutes; what is left over is usually days old)."""
    times = []
    for mark in RUN_MARKS:
        try:
            times.append((Path(case_dir) / mark).stat().st_mtime)
        except OSError:
            continue
    if not times:
        return False
    try:
        return path.stat().st_mtime < max(times) - after
    except OSError:
        return False


def _older(path: Path, than: Path, tolerance: float = 2.0) -> bool:
    """True if ``path`` was written before ``than`` (beyond ``tolerance`` seconds)."""
    try:
        return path.stat().st_mtime < than.stat().st_mtime - tolerance
    except OSError:
        return False


def read_commands(control: Path) -> tuple[list[float] | None, list[tuple[int, list[float]]]]:
    """The domain (B 10: xs, xe, ys, ye, zs, ze) and the solid commands (code, values)
    of a control.txt, in order."""
    domain = None
    solids = []
    try:
        text = control.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None, []
    for line in text.splitlines():
        words = line.split("//")[0].split()
        if len(words) < 2 or words[0] not in ("B", "S"):
            continue
        try:
            code = int(words[1])
            values = [float(w) for w in words[2:]]
        except ValueError:
            continue
        if words[0] == "B" and code == 10 and len(values) >= 6:
            domain = values[:6]
        elif words[0] == "S" and code in BUILDERS:
            solids.append((code, values))
        elif words[0] == "S" and code in OTHER_SOLIDS:
            solids.append((code, values))
    return domain, solids


def solids_from_control(control: Path) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Triangles of the solids of a control.txt: points (3 per triangle, so faces stay
    flat), triangles, and the commands that could not be built (as "S 12")."""
    domain, solids = read_commands(control)
    points, skipped = [], []
    for code, values in solids:
        builder = BUILDERS.get(code)
        try:
            triangles = builder(values, domain) if builder else None
        except (IndexError, ValueError, ZeroDivisionError):
            triangles = None
        if triangles is None or not len(triangles):
            skipped.append(f"S {code}")
            continue
        points.append(np.asarray(triangles, np.float64).reshape(-1, 3))
    if skipped:
        log.info("%s: not built from control.txt: %s", control, ", ".join(sorted(set(skipped))))
    if not points:
        return np.zeros((0, 3), np.float32), np.zeros((0, 3), np.uint32), skipped
    p = np.concatenate(points).astype(np.float32)
    return p, np.arange(len(p), dtype=np.uint32).reshape(-1, 3), skipped


# ======================================================================== shapes
def _convex(points) -> list:
    """Triangles of a convex solid from its corners, facing out (a fan over the faces
    of its hull, found by brute force: solids have few corners)."""
    p = np.asarray(points, float)
    triangles = []
    n = len(p)
    seen = set()
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                normal = np.cross(p[j] - p[i], p[k] - p[i])
                size = np.linalg.norm(normal)
                if size < 1e-12:
                    continue
                normal /= size
                side = (p - p[i]) @ normal
                tol = 1e-9 * max(np.ptp(p, axis=0).max(), 1.0)
                if (side > tol).any() and (side < -tol).any():
                    continue  # not a face of the hull
                if (side > tol).any():  # the others above: turn it out
                    normal = -normal
                face = tuple(sorted(np.flatnonzero(np.abs(side) <= tol)))
                if face in seen:
                    continue
                seen.add(face)
                triangles += _fan(p[list(face)], normal)
    return triangles


def _fan(corners: np.ndarray, normal: np.ndarray) -> list:
    """A flat convex polygon (its corners in any order) as triangles facing ``normal``."""
    centre = corners.mean(axis=0)
    u = corners[0] - centre
    if np.linalg.norm(u) < 1e-12:
        u = corners[1] - centre
    u /= np.linalg.norm(u)
    v = np.cross(normal, u)
    angle = np.arctan2((corners - centre) @ v, (corners - centre) @ u)
    ring = corners[np.argsort(angle)]
    return [(ring[0], ring[i], ring[i + 1]) for i in range(1, len(ring) - 1)]


def _box(v, _domain):
    xs, xe, ys, ye, zs, ze = v[:6]
    return _convex([(x, y, z) for x in (xs, xe) for y in (ys, ye) for z in (zs, ze)])


def _prism(section, axis: int, start: float, end: float):
    """A prism: a polygon in the plane across ``axis`` (pairs of the other two
    coordinates, in order) from ``start`` to ``end`` along it."""
    corners = []
    for a, b in section:
        for t in (start, end):
            point = [0.0, 0.0, 0.0]
            others = [i for i in range(3) if i != axis]
            point[others[0]], point[others[1]], point[axis] = a, b, t
            corners.append(point)
    return _convex(corners)


def _wedge_x(v, _domain):
    """S 61: across x, the top rises (or falls) from zs at xs to ze at xe."""
    xs, xe, ys, ye, zs, ze = v[:6]
    if zs < ze:
        section = [(xs, zs), (xe, zs), (xe, ze)]
    else:
        section = [(xs, ze), (xe, ze), (xs, zs)]
    return _prism(section, 1, ys, ye)


def _wedge_y(v, _domain):
    """S 62: as S 61, along y."""
    xs, xe, ys, ye, zs, ze = v[:6]
    if zs < ze:
        section = [(ys, zs), (ye, zs), (ye, ze)]
    else:
        section = [(ys, ze), (ye, ze), (ys, zs)]
    return _prism(section, 0, xs, xe)


def _wedge_z(v, _domain):
    """S 63: across x, the solid widens from xs at zs to xe at ze."""
    xs, xe, ys, ye, zs, ze = v[:6]
    if zs < ze:
        section = [(xs, zs), (xe, ze), (xs, ze)]
    else:
        section = [(xs, zs), (xe, ze), (xe, zs)]
    return _prism(section, 1, ys, ye)


def _round(axis: int, centre: tuple[float, float], start: float, end: float, r1, r2):
    """A cylinder or cone along ``axis`` (radius r1 at start, r2 at end), closed."""
    others = [i for i in range(3) if i != axis]
    phi = np.linspace(0.0, 2 * math.pi, SEGMENTS + 1)
    triangles = []

    def point(t, r, angle):
        p = [0.0, 0.0, 0.0]
        p[others[0]] = centre[0] + r * math.cos(angle)
        p[others[1]] = centre[1] + r * math.sin(angle)
        p[axis] = t
        return np.array(p)

    def middle(t):
        p = [0.0, 0.0, 0.0]
        p[others[0]], p[others[1]], p[axis] = centre[0], centre[1], t
        return np.array(p)

    for a, b in zip(phi[:-1], phi[1:], strict=True):
        s0, s1 = point(start, r1, a), point(start, r1, b)
        e0, e1 = point(end, r2, a), point(end, r2, b)
        triangles += [(s0, s1, e1), (s0, e1, e0)]
        if r1 > 0:
            triangles.append((middle(start), s1, s0))
        if r2 > 0:
            triangles.append((middle(end), e0, e1))
    # facing out: the sides away from the axis, the caps away from the middle
    out = []
    mid = 0.5 * (middle(start) + middle(end))
    for t in triangles:
        t = np.asarray(t)
        normal = np.cross(t[1] - t[0], t[2] - t[0])
        out.append(t if normal @ (t.mean(axis=0) - mid) >= 0 else t[::-1])
    return out


def _cylinder_y(v, domain):
    """S 32: xc, zc, r; across the domain in y."""
    if domain is None:
        return None
    return _round(1, (v[0], v[1]), domain[2], domain[3], v[2], v[2])


def _cylinder_z(v, domain):
    """S 33: xc, yc, r; from the domain's bottom to its top."""
    if domain is None:
        return None
    return _round(2, (v[0], v[1]), domain[4], domain[5], v[2], v[2])


def _cone_x(v, _domain):
    return _round(0, (v[0], v[1]), v[2], v[3], v[4], v[5])  # yc, zc, xs, xe, rs, re


def _cone_y(v, _domain):
    return _round(1, (v[0], v[1]), v[2], v[3], v[4], v[5])  # xc, zc, ys, ye, rs, re


def _cone_z(v, _domain):
    return _round(2, (v[0], v[1]), v[2], v[3], v[4], v[5])  # xc, yc, zs, ze, rs, re


def _sphere(v, _domain):
    xc, yc, zc, r = v[:4]
    rows = SEGMENTS // 2
    theta = np.linspace(0.0, math.pi, rows + 1)
    phi = np.linspace(0.0, 2 * math.pi, SEGMENTS + 1)

    def at(t, p):
        return np.array(
            [
                xc + r * math.sin(t) * math.cos(p),
                yc + r * math.sin(t) * math.sin(p),
                zc + r * math.cos(t),
            ]
        )

    triangles = []
    for t0, t1 in zip(theta[:-1], theta[1:], strict=True):
        for p0, p1 in zip(phi[:-1], phi[1:], strict=True):
            a, b, c, d = at(t0, p0), at(t1, p0), at(t1, p1), at(t0, p1)
            if t0 > 0:
                triangles.append((a, b, d))
            if t1 < math.pi:
                triangles.append((b, c, d))
    return triangles


def _corners(n: int):
    """S 81 - S 84: convex solids from their n corners (x, y, z each)."""

    def build(v, _domain):
        if len(v) < 3 * n:
            return None
        return _convex(np.asarray(v[: 3 * n], float).reshape(n, 3))

    return build


BUILDERS = {
    10: _box,
    32: _cylinder_y,
    33: _cylinder_z,
    41: _cone_x,
    42: _cone_y,
    43: _cone_z,
    51: _sphere,
    61: _wedge_x,
    62: _wedge_y,
    63: _wedge_z,
    81: _corners(4),
    82: _corners(5),
    83: _corners(6),
    84: _corners(8),
}
#: solid commands DIVEMesh knows that are only shown from its own solid file
OTHER_SOLIDS = (1, 11, 12, 37, 52, 53, 54, 121, 131, 141)


# ======================================================================== the bed
def bed_faces(
    points: np.ndarray, triangles: np.ndarray, bed: np.ndarray, tolerance: float | None = None
) -> np.ndarray:
    """Which triangles of the solids lie in the bed (the top of a block that makes the
    bed: its corners on the bed's surface, facing up). ``bed``: the bed's points."""
    t = np.asarray(triangles, np.int64).reshape(-1, 3)
    out = np.zeros(len(t), bool)
    bed = np.asarray(bed, float).reshape(-1, 3)
    bed = bed[np.isfinite(bed).all(axis=1)]
    if not len(t) or len(bed) < 3:
        return out
    p = np.asarray(points, float)
    low = bed[:, :2].min(axis=0)
    span = np.maximum(np.ptp(bed[:, :2], axis=0), 1e-9)
    cells = int(np.clip(np.sqrt(len(bed)), 4, 1024))
    size = span / cells
    ij = np.clip(((bed[:, :2] - low) / size).astype(int), 0, cells - 1)
    # the highest and lowest bed point in each cell (empty cells from their neighbours)
    top = np.full((cells, cells), np.nan)
    bottom = np.full((cells, cells), np.nan)
    np.fmax.at(top, (ij[:, 0], ij[:, 1]), bed[:, 2])
    np.fmin.at(bottom, (ij[:, 0], ij[:, 1]), bed[:, 2])
    top, bottom = _filled(top, np.fmax), _filled(bottom, np.fmin)
    if tolerance is None:
        tolerance = 0.01 * max(float(np.ptp(bed[:, 2])), 1e-3)
    corners = p[t]  # (m, 3, 3)
    inside = ((corners[..., :2] >= low - size) & (corners[..., :2] <= low + span + size)).all(
        axis=(1, 2)
    )
    cij = np.clip(((corners[..., :2] - low) / size).astype(int), 0, cells - 1)
    hi = top[cij[..., 0], cij[..., 1]]
    lo = bottom[cij[..., 0], cij[..., 1]]
    z = corners[..., 2]
    on_bed = ((z <= hi + tolerance) & (z >= lo - tolerance)).all(axis=1)
    normal = np.cross(corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0])
    up = normal[:, 2] > 0.3 * np.linalg.norm(normal, axis=1)
    return inside & on_bed & up


def _filled(grid: np.ndarray, pick) -> np.ndarray:
    """Empty (NaN) cells of a grid filled from their neighbours (``pick``: np.fmax or
    np.fmin, which pass NaN over)."""
    rows, cols = grid.shape
    for _ in range(max(rows, cols)):
        empty = np.isnan(grid)
        if not empty.any() or empty.all():
            break
        padded = np.pad(grid, 1, constant_values=np.nan)
        around = pick.reduce(
            [padded[a : a + rows, b : b + cols] for a in range(3) for b in range(3)]
        )
        grid = np.where(empty, around, grid)
    return grid
