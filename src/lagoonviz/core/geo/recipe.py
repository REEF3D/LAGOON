# SPDX-License-Identifier: GPL-3.0-or-later
# Architect: Hans Bihs
"""What geo.dat was made from, and whether it still fits the case.

LAGOON writes geo.dat in model coordinates: the bathymetry turned and moved into
the model's frame through the georeference. Turning or moving the model on the map
afterwards (or giving it another CRS, or a larger domain) leaves geo.dat as it was:
DIVEMesh would then mesh the bed of the old place. So beside geo.dat LAGOON keeps
how it was made (``LAGOON_geo.dat.json``): the georeference and domain it was made
for and its source, to say when it no longer fits and to make it again.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

FILE = "LAGOON_geo.dat.json"
GEO = "geo.dat"


def _stamp(path: Path) -> list[float]:
    stat = path.stat()
    return [float(stat.st_size), round(float(stat.st_mtime), 3)]


def write(folder: Path, recipe: dict) -> Path:
    """Keep how geo.dat (just written in ``folder``) was made."""
    folder = Path(folder)
    data = dict(recipe, file=_stamp(folder / GEO))
    path = folder / FILE
    path.write_text(json.dumps(data, indent=1) + "\n", encoding="utf-8")
    return path


def read(folder: Path | None) -> dict | None:
    """How the geo.dat in ``folder`` was made, or None: not by LAGOON, or changed
    since (another program wrote it)."""
    if folder is None:
        return None
    folder = Path(folder)
    try:
        data = json.loads((folder / FILE).read_text(encoding="utf-8"))
        if not isinstance(data, dict) or data.get("file") != _stamp(folder / GEO):
            return None
    except (OSError, ValueError):
        return None
    return data


def stale(recipe: dict | None, georef, domain) -> str:
    """Why geo.dat no longer fits the case (its georeference, its domain (x0, x1, y0,
    y1, ...)), or "" when it does or nothing is known."""
    if not recipe:
        return ""
    if recipe.get("file_crs") != "model" and georef is not None and recipe.get("crs"):
        if str(recipe["crs"]) != georef.crs:
            return f"geo.dat was made in {recipe['crs']}; the case is in {georef.crs} now"
        before = float(recipe.get("rotation_deg", 0.0))
        if abs(math.remainder(before - georef.rotation_deg, 360.0)) > 1e-6:
            return (
                f"geo.dat was made with the model turned {before:g}°; it is turned "
                f"{georef.rotation_deg:g}° now"
            )
        origin = recipe.get("origin") or [0.0, 0.0]
        if math.hypot(origin[0] - georef.origin[0], origin[1] - georef.origin[1]) > 0.01:
            return "the model was moved on the map since geo.dat was made"
    made = recipe.get("domain")
    if domain is not None and made and len(made) >= 4:
        slack = 0.5 * float(recipe.get("spacing") or 0.0) + 1e-6
        x0, x1, y0, y1 = (float(v) for v in domain[:4])
        if (
            x0 < made[0] - slack
            or x1 > made[1] + slack
            or y0 < made[2] - slack
            or y1 > made[3] + slack
        ):
            return "the domain reaches beyond geo.dat"
    return ""


def describe(recipe: dict | None) -> str:
    """Where geo.dat came from, in a few words."""
    if not recipe:
        return ""
    source = str(recipe.get("source", ""))
    if recipe.get("web"):
        what = "downloaded (" + (", ".join(recipe.get("sources") or []) or source) + ")"
    else:
        what = Path(source).name or "a file"
    return f"{what}, every {float(recipe.get('spacing') or 0):g} m"
