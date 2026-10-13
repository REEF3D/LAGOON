# SPDX-License-Identifier: GPL-3.0-or-later
# Architect: Hans Bihs
"""REEF3D::SEASTATE's maps of the sea state (``REEF3D_SEASTATE_VTP``).

SEASTATE, REEF3D's spectral wave model, writes per output a flat map at the still
water level with the integrated parameters of the wave spectrum at every grid
node: the significant wave height ``Hs``, the mean periods ``Tm01`` and ``Tm-10``,
the peak period ``Tp``, the mean direction ``dir`` (degrees, counter-clockwise from
x, where the waves go) and its directional ``spread``, the water ``depth`` and the
``wet`` fraction; with wind, ``U10x`` and ``U10y``; coupled to a flow solver, the
wave forces ``Fx``, ``Fy`` and the Stokes transport ``MStx``, ``MSty``; with
rollers, ``roller``, ``Dr`` and ``Dw``. Its AMR output (``REEF3D_SEASTATE_AMR``)
is read as any AMR surface.

LAGOON adds vectors made from them (:data:`DERIVED`), so arrows can show where the
waves go, the wind blows, the forces push.
"""

from __future__ import annotations

import numpy as np

#: vectors made from SEASTATE's scalars: name -> the scalars they come from
DERIVED: dict[str, tuple[str, ...]] = {
    "wave_direction": ("dir",),  # unit vectors of the mean wave direction
    "wave_vector": ("dir", "Hs"),  # the same, as long as the waves are high
    "wind": ("U10x", "U10y"),
    "wave_force": ("Fx", "Fy"),
    "stokes_transport": ("MStx", "MSty"),
}
#: what to colour a sea state by first, and how
DEFAULT_VARIABLE = "Hs"
DEFAULT_COLORMAP = "rainbow"  # Turbo, as SEASTATE's own maps


def derived_components(components: dict[str, int]) -> dict[str, int]:
    """The vectors that can be made from a file's arrays (name: 3)."""
    return {name: 3 for name, needs in DERIVED.items() if all(n in components for n in needs)}


def sources(names: list[str] | None, components: dict[str, int]) -> list[str] | None:
    """The arrays to read for ``names``: the derived vectors replaced by their
    scalars (None: all)."""
    if names is None:
        return None
    found: list[str] = []
    for name in names:
        for source in DERIVED.get(name, (name,)):
            if source in components and source not in found:
                found.append(source)
    return found


def add_derived(point_data: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Point data with the vectors that can be made from it."""
    out = dict(point_data)
    if "dir" in point_data:
        angle = np.radians(np.asarray(point_data["dir"], np.float64))
        unit = np.stack([np.cos(angle), np.sin(angle), np.zeros_like(angle)], axis=1)
        out["wave_direction"] = unit.astype(np.float32)
        if "Hs" in point_data:
            height = np.asarray(point_data["Hs"], np.float64)[:, None]
            out["wave_vector"] = (unit * height).astype(np.float32)
    for name, (a, b) in (
        ("wind", ("U10x", "U10y")),
        ("wave_force", ("Fx", "Fy")),
        ("stokes_transport", ("MStx", "MSty")),
    ):
        if a in point_data and b in point_data:
            x = np.asarray(point_data[a], np.float32)
            out[name] = np.stack([x, np.asarray(point_data[b], np.float32), 0 * x], axis=1)
    return out


def wet_triangles(triangles: np.ndarray, wet: np.ndarray) -> np.ndarray:
    """The triangles of a sea state map on wet nodes only (dry land left out)."""
    triangles = np.asarray(triangles)
    if not triangles.size:
        return triangles
    wet_node = np.asarray(wet) > 0.0
    return np.ascontiguousarray(triangles[wet_node[triangles].all(axis=1)])
