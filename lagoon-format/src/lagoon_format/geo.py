# SPDX-License-Identifier: MIT
# Architect: Hans Bihs
"""Georeference of a REEF3D case: where its model coordinates lie on the Earth.

REEF3D and DIVEMesh compute in model coordinates (metres, x and y in the plane, z
up). The georeference block ties them to a coordinate reference system (CRS)::

    X = X0 + x cos(r) - y sin(r)
    Y = Y0 + x sin(r) + y cos(r)
    Z = z + z_offset

with ``origin = (X0, Y0)`` in the CRS, ``r = rotation_deg`` counter-clockwise from
the CRS's x-axis (east) to the model's x-axis, and ``z_offset`` the height of the
model's z = 0 in the vertical datum. A case keeps it in ``georef.json`` in its
folder; REEF3D copies it into the run log's run record and the store's root
attributes (SPEC.md 1.2, RUN.md).

Standard library and NumPy only: CRS transforms beyond the affine model <-> CRS
step are the application's (pyproj or LAGOON's own). MIT-licensed.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

#: the file a case keeps its georeference in (in the case folder)
FILE = "georef.json"
#: root attribute of a store, and key of the run log's run record
KEY = "georef"
#: version of the georeference block
VERSION = 1


class GeorefError(ValueError):
    pass


@dataclass(frozen=True)
class Georef:
    """Where a case's model coordinates lie: CRS, origin, rotation, vertical datum."""

    #: the horizontal CRS: "EPSG:<code>" (e.g. "EPSG:25833") or a WKT2 string
    crs: str
    #: the model's (0, 0) in the CRS, in its units (metres for a projected CRS)
    origin: tuple[float, float] = (0.0, 0.0)
    #: counter-clockwise angle from the CRS's x-axis (east) to the model's x-axis [deg]
    rotation_deg: float = 0.0
    #: name of the vertical datum, free text or "EPSG:<code>" (e.g. "EPSG:5941", NN2000)
    vertical_datum: str = ""
    #: the height of the model's z = 0 in the vertical datum [m]
    z_offset: float = 0.0
    #: keys of the block this version does not know, kept as they were
    extra: dict[str, Any] = field(default_factory=dict, compare=False)

    def __post_init__(self) -> None:
        if not isinstance(self.crs, str) or not self.crs.strip():
            raise GeorefError("georef: crs must be a non-empty string")
        try:
            x0, y0 = (float(v) for v in self.origin)
        except (TypeError, ValueError):
            raise GeorefError(f"georef: origin must be two numbers, got {self.origin!r}") from None
        values = (x0, y0, float(self.rotation_deg), float(self.z_offset))
        if not all(math.isfinite(v) for v in values):
            raise GeorefError("georef: origin, rotation_deg and z_offset must be finite")
        object.__setattr__(self, "crs", self.crs.strip())
        object.__setattr__(self, "origin", (x0, y0))
        object.__setattr__(self, "rotation_deg", float(self.rotation_deg))
        object.__setattr__(self, "z_offset", float(self.z_offset))
        object.__setattr__(self, "vertical_datum", str(self.vertical_datum or ""))

    # ------------------------------------------------------------- the CRS
    @property
    def epsg(self) -> int | None:
        """The EPSG code of the CRS, or None for a WKT2 string."""
        return epsg_code(self.crs)

    @property
    def bearing_deg(self) -> float:
        """The model's x-axis as a compass bearing (clockwise from north) [deg]."""
        return (90.0 - self.rotation_deg) % 360.0

    # ------------------------------------------------------------- transforms
    def _cos_sin(self) -> tuple[float, float]:
        r = math.radians(self.rotation_deg)
        return math.cos(r), math.sin(r)

    def to_world(self, x, y, z=None):
        """Model coordinates to the CRS (and heights to the vertical datum).

        Arrays or numbers; returns ``(X, Y)`` or ``(X, Y, Z)`` with ``z`` given."""
        c, s = self._cos_sin()
        x = np.asarray(x, float)
        y = np.asarray(y, float)
        big_x = self.origin[0] + c * x - s * y
        big_y = self.origin[1] + s * x + c * y
        if z is None:
            return big_x, big_y
        return big_x, big_y, np.asarray(z, float) + self.z_offset

    def to_model(self, big_x, big_y, big_z=None):
        """The CRS (and the vertical datum) to model coordinates."""
        c, s = self._cos_sin()
        dx = np.asarray(big_x, float) - self.origin[0]
        dy = np.asarray(big_y, float) - self.origin[1]
        x = c * dx + s * dy
        y = -s * dx + c * dy
        if big_z is None:
            return x, y
        return x, y, np.asarray(big_z, float) - self.z_offset

    def points_to_world(self, points: np.ndarray) -> np.ndarray:
        """(n, 2) or (n, 3) model points to the CRS."""
        points = np.asarray(points, float)
        out = points.copy()
        out[:, 0], out[:, 1] = self.to_world(points[:, 0], points[:, 1])
        if points.shape[1] > 2:
            out[:, 2] = points[:, 2] + self.z_offset
        return out

    def points_to_model(self, points: np.ndarray) -> np.ndarray:
        """(n, 2) or (n, 3) CRS points to model coordinates."""
        points = np.asarray(points, float)
        out = points.copy()
        out[:, 0], out[:, 1] = self.to_model(points[:, 0], points[:, 1])
        if points.shape[1] > 2:
            out[:, 2] = points[:, 2] - self.z_offset
        return out

    def rotated(self, rotation_deg: float, about: tuple[float, float] = (0.0, 0.0)) -> Georef:
        """The same place, the model turned to a new rotation about a model point
        (which stays where it is on the map), e.g. the centre of the domain."""
        world = self.to_world(about[0], about[1])
        turned = Georef(self.crs, (0.0, 0.0), rotation_deg, self.vertical_datum, self.z_offset)
        offset = turned.to_world(about[0], about[1])
        origin = (float(world[0] - offset[0]), float(world[1] - offset[1]))
        return self.replace(origin=origin, rotation_deg=float(rotation_deg))

    def replace(self, **changes: Any) -> Georef:
        values = {
            "crs": self.crs,
            "origin": self.origin,
            "rotation_deg": self.rotation_deg,
            "vertical_datum": self.vertical_datum,
            "z_offset": self.z_offset,
            "extra": dict(self.extra),
        }
        values.update(changes)
        return Georef(**values)

    # ------------------------------------------------------------- JSON
    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = dict(self.extra)
        out.update(
            {
                "version": VERSION,
                "crs": self.crs,
                "origin": [self.origin[0], self.origin[1]],
                "rotation_deg": self.rotation_deg,
                "vertical_datum": self.vertical_datum,
                "z_offset": self.z_offset,
            }
        )
        return out

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Georef:
        if not isinstance(data, dict):
            raise GeorefError("georef: expected a JSON object")
        version = data.get("version", VERSION)
        if not isinstance(version, int) or version > VERSION:
            raise GeorefError(f"georef: version {version!r} is newer than {VERSION}")
        if "crs" not in data:
            raise GeorefError("georef: crs is missing")
        known = ("version", "crs", "origin", "rotation_deg", "vertical_datum", "z_offset")
        origin = data.get("origin", [0.0, 0.0])
        if not isinstance(origin, list | tuple) or len(origin) != 2:
            raise GeorefError(f"georef: origin must be [X0, Y0], got {origin!r}")
        return cls(
            crs=str(data["crs"]),
            origin=(origin[0], origin[1]),
            rotation_deg=data.get("rotation_deg", 0.0),
            vertical_datum=str(data.get("vertical_datum", "") or ""),
            z_offset=data.get("z_offset", 0.0),
            extra={k: v for k, v in data.items() if k not in known},
        )

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2) + "\n"

    @classmethod
    def from_json(cls, text: str) -> Georef:
        try:
            data = json.loads(text)
        except json.JSONDecodeError as error:
            raise GeorefError(f"georef: not JSON ({error})") from None
        return cls.from_dict(data)


def epsg_code(crs: str) -> int | None:
    """The EPSG code of "EPSG:25833" (also "epsg:25833", "urn:ogc:def:crs:EPSG::25833",
    "25833"), else None."""
    text = str(crs).strip()
    upper = text.upper()
    for prefix in ("URN:OGC:DEF:CRS:EPSG::", "URN:OGC:DEF:CRS:EPSG:", "EPSG:"):
        if upper.startswith(prefix):
            text = text[len(prefix) :]
            break
    return int(text) if text.isdigit() else None


# ------------------------------------------------------------------ files
def read(folder: str | Path) -> Georef | None:
    """The georeference of a case folder (``georef.json``), or None without one.
    A file that cannot be read raises :class:`GeorefError`."""
    path = Path(folder) / FILE
    if not path.is_file():
        return None
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as error:
        raise GeorefError(f"{path}: {error}") from None
    try:
        return Georef.from_json(text)
    except GeorefError as error:
        raise GeorefError(f"{path}: {error}") from None


def write(folder: str | Path, georef: Georef | None) -> Path:
    """Write (or with None remove) ``georef.json`` in a case folder."""
    path = Path(folder) / FILE
    if georef is None:
        path.unlink(missing_ok=True)
    else:
        path.write_text(georef.to_json(), encoding="utf-8")
    return path


def from_attributes(attributes: dict[str, Any]) -> Georef | None:
    """The georeference in a store's root attributes or a run record (key ``georef``),
    or None; a block that does not read is left out (None)."""
    block = attributes.get(KEY) if isinstance(attributes, dict) else None
    if block is None:
        return None
    try:
        return Georef.from_dict(block)
    except GeorefError:
        return None


__all__ = [
    "FILE",
    "KEY",
    "VERSION",
    "Georef",
    "GeorefError",
    "epsg_code",
    "from_attributes",
    "read",
    "write",
]
