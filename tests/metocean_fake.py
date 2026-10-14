# SPDX-License-Identifier: GPL-3.0-or-later
# Architect: Hans Bihs
"""A stand-in for MET Norway's THREDDS (OPeNDAP .dds, .das and .ascii answers in
its text format, as it answered in October 2026) over small made-up datasets: NORA3's
wave subset and spectra, NorKyst v2 and v3."""

from __future__ import annotations

import datetime as dt
import re
import urllib.parse

import numpy as np

BASE = "https://thredds.met.no/thredds/dodsC/"


class Var:
    def __init__(self, dims, values, attrs=None, grid=True, kind="Float32"):
        self.dims = list(dims)  # [(name, size)]
        self.values = np.asarray(values)
        self.attrs = attrs or {}
        self.grid = grid
        self.kind = kind


class FakeThredds:
    def __init__(self):
        self.datasets: dict[str, dict[str, Var]] = {}
        self.asked: list[str] = []

    def __call__(self, url: str) -> bytes:
        self.asked.append(url)
        if not url.startswith(BASE):
            raise OSError(f"HTTP Error 404: {url}")
        rest = url[len(BASE) :]
        path, _, query = rest.partition("?")
        path, _, suffix = path.rpartition(".")
        data = self.datasets.get(path)
        if data is None:
            raise OSError(f"HTTP Error 404: Not Found {url}")
        if suffix == "dds":
            return self._dds(path, data).encode()
        if suffix == "das":
            return self._das(data).encode()
        if suffix == "ascii":
            return self._ascii(path, data, urllib.parse.unquote(query)).encode()
        raise OSError(f"HTTP Error 400: {url}")

    @staticmethod
    def _shape(dims) -> str:
        return "".join(f"[{d} = {n}]" for d, n in dims)

    def _dds(self, path, data) -> str:
        lines = ["Dataset {"]
        for name, var in data.items():
            if var.grid and len(var.dims) > 0:
                lines += [
                    "    Grid {",
                    "     ARRAY:",
                    f"        {var.kind} {name}{self._shape(var.dims)};",
                    "     MAPS:",
                    *[f"        Float64 {d}[{d} = {n}];" for d, n in var.dims],
                    f"    }} {name};",
                ]
            else:
                lines.append(f"    {var.kind} {name}{self._shape(var.dims)};")
        lines.append(f"}} {path};")
        return "\n".join(lines) + "\n"

    def _das(self, data) -> str:
        lines = ["Attributes {"]
        for name, var in data.items():
            lines.append(f"    {name} {{")
            for key, value in var.attrs.items():
                lines.append(f"        Float32 {key} {value};")
            lines.append("    }")
        lines.append("}")
        return "\n".join(lines) + "\n"

    def _ascii(self, path, data, query) -> str:
        out = ["Dataset {", f"}} {path};", "-" * 45]
        for part in re.findall(r"(\w+)((?:\[[\d:]+\])*)", query):
            name, ranges = part
            var = data[name]
            index = []
            for r, _dim in zip(re.findall(r"\[([\d:]+)\]", ranges), var.dims, strict=False):
                bits = [int(b) for b in r.split(":")]
                if len(bits) == 1:
                    index.append(slice(bits[0], bits[0] + 1))
                elif len(bits) == 2:
                    index.append(slice(bits[0], bits[1] + 1))
                else:
                    index.append(slice(bits[0], bits[2] + 1, bits[1]))
            while len(index) < len(var.dims):
                index.append(slice(None))
            values = np.asarray(var.values)[tuple(index)]
            header = f"{name}.{name}" if var.grid else name
            shape = "".join(f"[{n}]" for n in values.shape)
            out.append(f"{header}{shape}")
            if values.ndim <= 1:
                out.append(", ".join(_fmt(v) for v in np.ravel(values)))
            else:
                rows = values.reshape(-1, values.shape[-1])
                for k, row in zip(np.ndindex(*values.shape[:-1]), rows, strict=True):
                    out.append(
                        "".join(f"[{i}]" for i in k) + ", " + ", ".join(_fmt(v) for v in row)
                    )
            out.append("")
            if var.grid:
                for (d, _), s in zip(var.dims, index, strict=True):
                    out.append(f"{name}.{d}[{len(range(*s.indices(1 << 30)))}]")
                    out.append("0.0")
                    out.append("")
            out.append("")
        return "\n".join(out) + "\n"


def _fmt(v) -> str:
    v = float(v)
    return repr(v) if not v.is_integer() or abs(v) > 1e15 else str(int(v)) + ".0"


# ----------------------------------------------------------------- datasets
def _stamp(year, month, day=1, hour=0) -> float:
    return dt.datetime(year, month, day, hour, tzinfo=dt.UTC).timestamp()


def nora3_month(fake, year, month, lat0=61.0, lon0=4.0, ni=40, nj=30, storm_hour=None):
    """A NORA3 wave subset month: a 40 x 30 grid round (lat0, lon0) (3 km), land in
    the east, a storm at ``storm_hour`` (Hs 9 m from the west)."""
    from lagoonviz.core.geo.metocean import hours_in_month, nora3_path

    n = hours_in_month(year, month)
    ii, jj = np.meshgrid(np.arange(ni), np.arange(nj), indexing="ij")
    lat = lat0 + (jj - nj / 2) * 0.027
    lon = lon0 + (ii - ni / 2) * 0.055
    depth = np.where(ii < ni - 8, 50.0 + ii, -9999999.0)  # the east is land
    t = _stamp(year, month) + 3600.0 * np.arange(n)
    hours = np.arange(n)
    hs = 1.0 + 0.5 * np.sin(hours / 30.0)
    if storm_hour is not None:
        hs = hs + 8.0 * np.exp(-(((hours - storm_hour) / 12.0) ** 2))
    field = lambda v: np.broadcast_to(v, (ni, nj, n)).copy()  # noqa: E731
    fill = {"_FillValue": -9999999.0}
    hs_f = field(hs)
    hs_f[ii >= ni - 8] = -9999999.0
    dims = [("rlon", ni), ("rlat", nj), ("time", n)]
    fake.datasets[nora3_path(year, month)] = {
        "hs": Var(dims, hs_f, fill),
        "tp": Var(dims, field(8.0 + hs), fill),
        "thq": Var(dims, field(np.full(n, 90.0)), fill),  # going east: from the west
        "ff": Var(dims, field(5.0 + 2.0 * hs), fill),
        "dd": Var(dims, field(np.full(n, 70.0)), fill),
        "latitude": Var(dims[:2], lat),
        "longitude": Var(dims[:2], lon),
        "model_depth": Var(dims[:2], depth, fill),
        "time": Var([("time", n)], t, grid=False, kind="Float64"),
    }


def spectra_day(fake, day: dt.date, lat0=61.0, lon0=4.0):
    """NORA3 spectra of a day: a few points, a JONSWAP-like spectrum going east."""
    from lagoonviz.core.geo.metocean import spectra_path

    freq = 0.0345 * 1.1 ** np.arange(30)
    direction = 7.5 + 15.0 * np.arange(24)
    lats = lat0 + np.array([0.0, 0.2, -0.3, 1.0])
    lons = lon0 + np.array([0.0, 0.3, -0.2, 1.0])
    fp = 0.1
    s_f = (freq / fp) ** -5 * np.exp(-1.25 * (freq / fp) ** -4)
    spread = np.cos(np.radians(direction - 90.0)) ** 2 * (np.cos(np.radians(direction - 90.0)) > 0)
    e = s_f[:, None] * spread[None, :]
    e = e / (np.sum(e * np.gradient(freq)[:, None]) * np.radians(15.0)) * (4.0 / 4.0) ** 2
    spec = np.broadcast_to(e, (24, 1, len(lats), 30, 24)).copy()
    fake.datasets[spectra_path(day)] = {
        "SPEC": Var(
            [("time", 24), ("y", 1), ("x", len(lats)), ("freq", 30), ("direction", 24)],
            spec,
            {"_FillValue": -99999.0},
        ),
        "latitude": Var([("y", 1), ("x", len(lats))], lats[None]),
        "longitude": Var([("y", 1), ("x", len(lats))], lons[None]),
        "freq": Var([("freq", 30)], freq, grid=False),
        "direction": Var([("direction", 24)], direction, grid=False),
        "time": Var([("time", 24)], 3600.0 * np.arange(24), grid=False, kind="Int32"),
    }
    return e  # Hs 4 m


def norkyst(fake, path, times, lat0=61.0, lon0=4.0, ny=30, nx=40, packed=True):
    """NorKyst over a 30 x 40 grid (800 m): depth 20 m, a tide of 0.5 m with a 12.42 h
    period, a current of 0.3 m/s east at the surface, 0.1 m/s at 15 m."""
    jj, ii = np.meshgrid(np.arange(ny), np.arange(nx), indexing="ij")
    lat = lat0 + (jj - ny / 2) * 0.0072
    lon = lon0 + (ii - nx / 2) * 0.0149
    depths = np.array([0.0, 3.0, 10.0, 15.0, 25.0, 50.0])
    nt = len(times)
    zeta = 0.5 * np.sin(2 * np.pi * (np.asarray(times) / 3600.0) / 12.42)
    zeta = np.broadcast_to(zeta[:, None, None], (nt, ny, nx)).copy()
    u = np.zeros((nt, len(depths), ny, nx))
    for k, d in enumerate(depths):
        u[:, k] = 0.3 - 0.2 * min(d, 15.0) / 15.0
    u[:, depths > 20.0] = np.nan
    v = np.zeros_like(u)
    v[:, depths > 20.0] = np.nan
    kind, attrs = "Float32", {"_FillValue": 1.0e37}
    if packed:  # v3: Int16, scaled
        kind, attrs = "Int16", {"_FillValue": -32767.0, "scale_factor": 0.001}
        zeta = np.round(zeta / 0.001)
        u = np.where(np.isnan(u), -32767, np.round(u / 0.001))
        v = np.where(np.isnan(v), -32767, np.round(v / 0.001))
    else:
        u = np.where(np.isnan(u), 1.0e37, u)
        v = np.where(np.isnan(v), 1.0e37, v)
    dims3 = [("time", nt), ("Y", ny), ("X", nx)]
    dims4 = [("time", nt), ("depth", len(depths)), ("Y", ny), ("X", nx)]
    fake.datasets[path] = {
        "zeta": Var(dims3, zeta, attrs, kind=kind),
        "u_eastward": Var(dims4, u, attrs, kind=kind),
        "v_northward": Var(dims4, v, attrs, kind=kind),
        "h": Var([("Y", ny), ("X", nx)], np.full((ny, nx), 20.0)),
        "lat": Var([("Y", ny), ("X", nx)], lat),
        "lon": Var([("Y", ny), ("X", nx)], lon),
        "depth": Var([("depth", len(depths))], depths, grid=False, kind="Float64"),
        "time": Var([("time", nt)], np.asarray(times), grid=False, kind="Float64"),
    }
