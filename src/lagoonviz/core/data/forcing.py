# SPDX-License-Identifier: GPL-3.0-or-later
# Architect: Hans Bihs
"""REEF3D::NHFLOW's direct forcing solids (ctrl.txt A 581 - A 589) as surfaces.

NHFLOW makes its solids of simple shapes (nhflow_geometry_objects_forcing.cpp), each
command one solid:

    A 581 xs xe ys ye zs ze          box
    A 583 xc zc ys ye r              cylinder along y
    A 584 xc yc zs ze r              vertical cylinder
    A 585 x1 y1 z1 r1 x2 y2 z2 r2    jacket member: a cylinder (cone) from one end to the other
    A 586 xc yc zc r                 sphere
    A 587 xs xe ys ye zs ze          wedge along x: the top from zs at xs to ze at xe
    A 588 xs xe ys ye zs ze          wedge along y
    A 589 xs xe ys ye zs ze          wedge across x, widening from xs at zs to xe at ze

and writes their triangles, with those of the grid file's solids (A 580) and an STL
(A 590), to ``REEF3D_NHFLOW_FORCING_VTP/REEF3D-NHFLOW-FORCING.vtp`` when it starts.
The case shows that file; before a run, or when ctrl.txt was changed since, the
solids are built from ctrl.txt (an STL only from the file).
"""

from __future__ import annotations

import logging
import math
from pathlib import Path

import numpy as np

from lagoonviz.core.data.divemesh import (
    SEGMENTS,
    _box,
    _older,
    _round,
    _sphere,
    _wedge_x,
    _wedge_y,
    _wedge_z,
)

log = logging.getLogger(__name__)

#: the file NHFLOW writes its forcing solids to
FORCING_FOLDER = "REEF3D_NHFLOW_FORCING_VTP"
FORCING_VTP = f"{FORCING_FOLDER}/REEF3D-NHFLOW-FORCING.vtp"


def _ycyl(v):
    xc, zc, ys, ye, r = v[:5]
    return _round(1, (xc, zc), ys, ye, r, r)


def _zcyl(v):
    xc, yc, zs, ze, r = v[:5]
    return _round(2, (xc, yc), zs, ze, r, r)


def _member(v):
    """A 585: a cylinder (or cone) between two points, radius r1 at the first."""
    a = np.asarray(v[0:3], float)
    b = np.asarray(v[4:7], float)
    r1, r2 = float(v[3]), float(v[7])
    axis = b - a
    length = float(np.linalg.norm(axis))
    if length <= 0:
        return None
    axis /= length
    helper = np.array([1.0, 0.0, 0.0]) if abs(axis[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    u = np.cross(axis, helper)
    u /= np.linalg.norm(u)
    w = np.cross(axis, u)
    phi = np.linspace(0.0, 2 * math.pi, SEGMENTS + 1)
    ring_a = [a + r1 * (math.cos(t) * u + math.sin(t) * w) for t in phi]
    ring_b = [b + r2 * (math.cos(t) * u + math.sin(t) * w) for t in phi]
    triangles = []
    for k in range(SEGMENTS):
        triangles += [
            (ring_a[k], ring_b[k], ring_b[k + 1]),
            (ring_a[k], ring_b[k + 1], ring_a[k + 1]),
        ]
        if r1 > 0:
            triangles.append((a, ring_a[k + 1], ring_a[k]))
        if r2 > 0:
            triangles.append((b, ring_b[k], ring_b[k + 1]))
    mid = 0.5 * (a + b)
    out = []
    for t in triangles:  # facing out
        t = np.asarray(t)
        normal = np.cross(t[1] - t[0], t[2] - t[0])
        out.append(t if normal @ (t.mean(axis=0) - mid) >= 0 else t[::-1])
    return out


#: command -> (values per line, builder)
SHAPES = {
    581: (6, lambda v: _box(v, None)),
    583: (5, _ycyl),
    584: (5, _zcyl),
    585: (8, _member),
    586: (4, lambda v: _sphere(v, None)),
    587: (6, lambda v: _wedge_x(v, None)),
    588: (6, lambda v: _wedge_y(v, None)),
    589: (6, lambda v: _wedge_z(v, None)),
}


def read_forcing(ctrl: Path) -> list[tuple[int, list[float]]]:
    """The direct forcing commands of a ctrl.txt (code, values), in order."""
    try:
        text = ctrl.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return []
    found = []
    for line in text.splitlines():
        words = line.split("//")[0].split()
        if len(words) < 2 or words[0] != "A":
            continue
        try:
            code = int(words[1])
            values = [float(w) for w in words[2:]]
        except ValueError:
            continue
        if code in SHAPES and len(values) >= SHAPES[code][0]:
            found.append((code, values))
    return found


def forcing_from_ctrl(ctrl: Path) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Triangles of the direct forcing solids of a ctrl.txt: points (3 per triangle),
    triangles, and the commands that could not be built (as "A 585")."""
    points, skipped = [], []
    for code, values in read_forcing(ctrl):
        try:
            triangles = SHAPES[code][1](values)
        except (IndexError, ValueError, ZeroDivisionError):
            triangles = None
        if triangles is None or not len(triangles):
            skipped.append(f"A {code}")
            continue
        points.append(np.asarray(triangles, np.float64).reshape(-1, 3))
    if skipped:
        log.info("%s: not built: %s", ctrl, ", ".join(sorted(set(skipped))))
    if not points:
        return np.zeros((0, 3), np.float32), np.zeros((0, 3), np.uint32), skipped
    p = np.concatenate(points).astype(np.float32)
    return p, np.arange(len(p), dtype=np.uint32).reshape(-1, 3), skipped


def find_forcing(case_dir: str | Path) -> Path | None:
    """Where a case's direct forcing solids come from: NHFLOW's file of them, unless
    ctrl.txt was changed since (its solids then built from ctrl.txt, if it has any);
    ctrl.txt before a run; None without any. Only NHFLOW (A 10 5) has them."""
    case_dir = Path(case_dir)
    vtp = case_dir / FORCING_VTP
    ctrl = case_dir / "ctrl.txt"
    if not ctrl.is_file():
        return vtp if vtp.is_file() else None
    if not _nhflow(ctrl):
        return None
    commands = read_forcing(ctrl)
    if vtp.is_file() and (not commands or not _older(vtp, ctrl)):
        return vtp
    return ctrl if commands else None


def _nhflow(ctrl: Path) -> bool:
    """Whether a ctrl.txt sets up an NHFLOW run (A 10 5)."""
    try:
        text = ctrl.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return False
    for line in text.splitlines():
        words = line.split("//")[0].split()
        if words[:2] == ["A", "10"] and len(words) > 2:
            try:
                return int(float(words[2])) == 5
            except ValueError:
                return False
    return False
