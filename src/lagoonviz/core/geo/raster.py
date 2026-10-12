# SPDX-License-Identifier: GPL-3.0-or-later
# Architect: Hans Bihs
"""Rasters and point clouds of the Earth: bathymetry, terrain, images.

:class:`GeoImage` is a grid of values (heights) or colours with its place in a CRS
(the GDAL geotransform). It is read from

- GeoTIFF (``.tif``, ``.tiff``): with rasterio when installed (any GeoTIFF, only
  the window that is needed), else with tifffile or Pillow (the GeoTIFF tags of
  plain north-up rasters);
- Esri ASCII grids (``.asc``), e.g. EMODnet and Kartverket downloads;

and :func:`read_points` reads scattered soundings (``.xyz``, ``.txt``, ``.csv``:
x y z per line). A CRS that a file does not name is given by the user.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from lagoon_format.geo import epsg_code

from lagoonviz.core.geo.crs import transformer

RASTER_SUFFIXES = (".tif", ".tiff", ".asc")
POINT_SUFFIXES = (".xyz", ".txt", ".csv", ".dat", ".pts")


class RasterError(ValueError):
    pass


@dataclass
class GeoImage:
    """A raster in a CRS: ``data`` (rows, cols) of values or (rows, cols, 3|4) uint8
    colours, row 0 first in the file (north for a north-up raster), and the GDAL
    geotransform: X = t0 + col t1 + row t2, Y = t3 + col t4 + row t5 at the pixels'
    corners."""

    data: np.ndarray
    transform: tuple[float, float, float, float, float, float]
    crs: str | None = None
    nodata: float | None = None
    name: str = ""

    @property
    def shape(self) -> tuple[int, int]:
        return int(self.data.shape[0]), int(self.data.shape[1])

    @property
    def is_image(self) -> bool:
        return self.data.ndim == 3

    def bounds(self) -> tuple[float, float, float, float]:
        """(x0, y0, x1, y1) in the CRS: the corners' extent."""
        rows, cols = self.shape
        corners = np.array([[0, 0], [cols, 0], [0, rows], [cols, rows]], float)
        x, y = self.world(corners[:, 0], corners[:, 1])
        return float(x.min()), float(y.min()), float(x.max()), float(y.max())

    def world(self, col, row):
        t = self.transform
        col = np.asarray(col, float)
        row = np.asarray(row, float)
        return t[0] + col * t[1] + row * t[2], t[3] + col * t[4] + row * t[5]

    def pixel(self, x, y):
        """Fractional (col, row) of CRS points, pixel corners at whole numbers."""
        t = self.transform
        det = t[1] * t[5] - t[2] * t[4]
        if det == 0:
            raise RasterError("the raster's geotransform is singular")
        dx = np.asarray(x, float) - t[0]
        dy = np.asarray(y, float) - t[3]
        col = (t[5] * dx - t[2] * dy) / det
        row = (-t[4] * dx + t[1] * dy) / det
        return col, row

    def cell_size(self) -> float:
        t = self.transform
        return float(np.sqrt(abs(t[1] * t[5] - t[2] * t[4])))

    def values(self) -> np.ndarray:
        """The heights as float, NaN where there is no data."""
        if self.is_image:
            raise RasterError(f"{self.name or 'the raster'} is an image, not heights")
        values = np.asarray(self.data, float)
        if self.nodata is not None and np.isfinite(self.nodata):
            values = np.where(values == self.nodata, np.nan, values)
        # bathymetry products mark missing data with huge numbers
        return np.where(np.abs(values) > 1e20, np.nan, values)

    def sample(self, x, y, crs: str | None = None) -> np.ndarray:
        """Bilinear values at points (in ``crs``, default the raster's): heights as
        float with NaN outside, colours as float RGBA (0 alpha outside)."""
        x = np.asarray(x, float)
        y = np.asarray(y, float)
        if crs is not None and self.crs is not None and crs != self.crs:
            x, y = transformer(crs, self.crs)(x, y)
        col, row = self.pixel(x, y)
        data = self.data.astype(float) if self.is_image else self.values()
        return bilinear(data, col - 0.5, row - 0.5)

    def points(self, step: int = 1) -> np.ndarray:
        """The pixel centres with data as (n, 3) CRS points (every ``step``-th)."""
        values = self.values()[::step, ::step]
        rows, cols = np.nonzero(np.isfinite(values))
        x, y = self.world(cols * step + 0.5, rows * step + 0.5)
        return np.column_stack([x, y, values[rows, cols]])


def bilinear(data: np.ndarray, col: np.ndarray, row: np.ndarray) -> np.ndarray:
    """Bilinear interpolation of a (rows, cols[, c]) array at fractional indices
    (index i at the centre of cell i); NaN (colours: 0) outside."""
    rows, cols = data.shape[:2]
    inside = (col >= -0.5) & (col <= cols - 0.5) & (row >= -0.5) & (row <= rows - 0.5)
    c = np.clip(col, 0, cols - 1)
    r = np.clip(row, 0, rows - 1)
    c0 = np.clip(np.floor(c).astype(np.int64), 0, max(cols - 2, 0))
    r0 = np.clip(np.floor(r).astype(np.int64), 0, max(rows - 2, 0))
    c1 = np.minimum(c0 + 1, cols - 1)
    r1 = np.minimum(r0 + 1, rows - 1)
    fc = np.clip(c - c0, 0, 1)
    fr = np.clip(r - r0, 0, 1)
    if data.ndim == 3:
        fc = fc[..., None]
        fr = fr[..., None]
    top = data[r0, c0] * (1 - fc) + data[r0, c1] * fc
    bottom = data[r1, c0] * (1 - fc) + data[r1, c1] * fc
    out = top * (1 - fr) + bottom * fr
    if data.ndim == 3:
        return np.where(inside[..., None], out, 0.0)
    if data.dtype.kind == "f":
        # a cell with no data next to the point: the nearest cell's value, if it has one
        near = data[
            np.clip(np.rint(r).astype(np.int64), 0, rows - 1),
            np.clip(np.rint(c).astype(np.int64), 0, cols - 1),
        ]
        out = np.where(np.isfinite(out), out, near)
    return np.where(inside, out, np.nan)


# ------------------------------------------------------------------ reading
def read_raster(
    path: str | Path,
    crs: str | None = None,
    window: tuple[float, float, float, float] | None = None,
) -> GeoImage:
    """A GeoTIFF or Esri ASCII grid. ``crs``: the CRS when the file names none (or
    overrides it); ``window``: (x0, y0, x1, y1) in the raster's CRS, read only that
    part (rasterio) or cut to it."""
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".asc":
        image = _read_asc(path)
    elif suffix in (".tif", ".tiff"):
        image = _read_tiff(path, window)
    else:
        raise RasterError(f"{path.name}: not a raster LAGOON reads (GeoTIFF or .asc)")
    if crs:
        image.crs = crs
    if image.crs is None:
        image.crs = _prj_crs(path)
    image.name = path.name
    if window is not None:
        image = cut(image, window)
    return image


def cut(image: GeoImage, window: tuple[float, float, float, float], pad: int = 2) -> GeoImage:
    """The part of a raster over a window (x0, y0, x1, y1) in its CRS (a few pixels
    more round it)."""
    x0, y0, x1, y1 = window
    col, row = image.pixel([x0, x1, x0, x1], [y0, y0, y1, y1])
    rows, cols = image.shape
    c0 = int(max(np.floor(col.min()) - pad, 0))
    c1 = int(min(np.ceil(col.max()) + pad, cols))
    r0 = int(max(np.floor(row.min()) - pad, 0))
    r1 = int(min(np.ceil(row.max()) + pad, rows))
    if c1 <= c0 or r1 <= r0:
        raise RasterError(f"{image.name or 'the raster'} does not reach the area")
    t = image.transform
    x, y = image.world(c0, r0)
    transform = (float(x), t[1], t[2], float(y), t[4], t[5])
    return GeoImage(image.data[r0:r1, c0:c1], transform, image.crs, image.nodata, image.name)


def _read_asc(path: Path) -> GeoImage:
    header: dict[str, float] = {}
    with path.open("r", encoding="utf-8", errors="replace") as file:
        lines = 0
        while True:
            position = file.tell()
            line = file.readline()
            parts = line.split()
            if len(parts) == 2 and re.match(r"^[A-Za-z_]+$", parts[0]):
                header[parts[0].lower()] = float(parts[1])
                lines += 1
                continue
            file.seek(position)
            break
        values = np.array(file.read().split(), float)
    try:
        cols, rows = int(header["ncols"]), int(header["nrows"])
        size = header["cellsize"]
    except KeyError as missing:
        raise RasterError(f"{path.name}: no {missing} in the header of the ASCII grid") from None
    if values.size < rows * cols:
        raise RasterError(f"{path.name}: {values.size} values for {rows} x {cols} cells")
    data = values[: rows * cols].reshape(rows, cols)
    if "xllcenter" in header:
        x0, y0 = header["xllcenter"] - size / 2, header["yllcenter"] - size / 2
    else:
        x0, y0 = header.get("xllcorner", 0.0), header.get("yllcorner", 0.0)
    transform = (x0, size, 0.0, y0 + rows * size, 0.0, -size)
    return GeoImage(data, transform, None, header.get("nodata_value"))


def _prj_crs(path: Path) -> str | None:
    """The CRS of a .prj file next to a raster (with pyproj), else None."""
    prj = path.with_suffix(".prj")
    if not prj.is_file():
        return None
    text = prj.read_text(encoding="utf-8", errors="replace").strip()
    found = re.findall(r'AUTHORITY\["EPSG",\s*"?(\d+)"?\]', text)
    if found:
        return f"EPSG:{found[-1]}"
    try:
        import pyproj

        code = pyproj.CRS.from_wkt(text).to_epsg()
        return f"EPSG:{code}" if code else text
    except Exception:  # noqa: BLE001 (no pyproj, or WKT it does not read)
        return None


def _read_tiff(path: Path, window) -> GeoImage:
    try:
        import rasterio  # noqa: F401
    except ImportError:
        return _read_tiff_tags(path)
    return _read_rasterio(path, window)


def _read_rasterio(path: Path, window) -> GeoImage:
    import rasterio
    from rasterio.windows import from_bounds

    with rasterio.open(path) as source:
        area = None
        if window is not None:
            area = from_bounds(*window, transform=source.transform).round_offsets().round_lengths()
            area = area.intersection(rasterio.windows.Window(0, 0, source.width, source.height))
        if source.count >= 3 and source.dtypes[0] == "uint8":
            data = np.moveaxis(
                source.read(indexes=list(range(1, min(source.count, 4) + 1)), window=area), 0, -1
            )
        else:
            data = source.read(1, window=area)
        transform = source.window_transform(area) if area is not None else source.transform
        crs = source.crs.to_string() if source.crs is not None else None
        return GeoImage(
            data,
            (transform.c, transform.a, transform.b, transform.f, transform.d, transform.e),
            crs,
            source.nodata,
        )


#: GeoTIFF tags
PIXEL_SCALE, TIEPOINT, TRANSFORMATION, GEOKEYS, GDAL_NODATA = 33550, 33922, 34264, 34735, 42113


def _read_tiff_tags(path: Path) -> GeoImage:
    tags: dict[int, object] = {}
    data = None
    try:
        import tifffile

        with tifffile.TiffFile(path) as tif:
            page = tif.pages[0]
            for tag in page.tags.values():
                tags[tag.code] = tag.value
            data = page.asarray()
    except ImportError:
        from PIL import Image

        Image.MAX_IMAGE_PIXELS = None  # a large terrain raster is not a decompression bomb
        try:
            with Image.open(path) as image:
                tags = {int(k): v for k, v in image.tag_v2.items()}
                if image.mode in ("RGB", "RGBA", "P", "L") and image.mode != "L":
                    data = np.asarray(image.convert("RGBA"), np.uint8)
                else:
                    data = np.asarray(image)
        except OSError as error:
            raise RasterError(
                f"{path.name}: {error}; install rasterio (pip install 'lagoon-viz[geo]') "
                "to read every GeoTIFF"
            ) from None
    if data is None:
        raise RasterError(f"{path.name}: no image")
    if data.ndim == 3 and data.shape[0] in (3, 4) and data.shape[-1] not in (3, 4):
        data = np.moveaxis(data, 0, -1)  # planar bands
    if data.ndim == 3 and data.dtype != np.uint8:
        data = data[..., 0]  # a multi-band height raster: its first band
    transform = _geotransform(tags, path)
    crs, point = _geokeys(tags)
    if point:  # PixelIsPoint: the tie point is at a pixel's centre
        t = transform
        transform = (t[0] - 0.5 * (t[1] + t[2]), t[1], t[2], t[3] - 0.5 * (t[4] + t[5]), t[4], t[5])
    nodata = tags.get(GDAL_NODATA)
    try:
        nodata = float(str(nodata).strip("\x00 ")) if nodata is not None else None
    except ValueError:
        nodata = None
    return GeoImage(data, transform, crs, nodata)


def _floats(value) -> list[float]:
    if isinstance(value, bytes | str):
        return []
    try:
        return [float(v) for v in np.asarray(value, float).ravel()]
    except (TypeError, ValueError):
        return []


def _geotransform(tags: dict, path: Path) -> tuple[float, ...]:
    matrix = _floats(tags.get(TRANSFORMATION, []))
    if len(matrix) >= 16:
        return (matrix[3], matrix[0], matrix[1], matrix[7], matrix[4], matrix[5])
    scale = _floats(tags.get(PIXEL_SCALE, []))
    tie = _floats(tags.get(TIEPOINT, []))
    if len(scale) >= 2 and len(tie) >= 6:
        i, j, x, y = tie[0], tie[1], tie[3], tie[4]
        return (x - i * scale[0], scale[0], 0.0, y + j * scale[1], 0.0, -scale[1])
    world = _world_file(path)
    if world is not None:
        return world
    raise RasterError(f"{path.name}: not georeferenced (no GeoTIFF tags or world file)")


def _world_file(path: Path):
    for suffix in (".tfw", ".tifw", ".wld"):
        candidate = path.with_suffix(suffix)
        if candidate.is_file():
            v = [float(t) for t in candidate.read_text().split()[:6]]
            if len(v) == 6:  # A D B E C F: centre of the upper-left pixel
                a, d, b, e, c, f = v
                return (c - 0.5 * (a + b), a, b, f - 0.5 * (d + e), d, e)
    return None


def _geokeys(tags: dict) -> tuple[str | None, bool]:
    keys = (
        [int(v) for v in np.asarray(tags.get(GEOKEYS, []), int).ravel()] if GEOKEYS in tags else []
    )
    found: dict[int, int] = {}
    for n in range(4, len(keys) - 3, 4):
        key, location, _count, value = keys[n : n + 4]
        if location == 0:
            found[key] = value
    point = found.get(1025) == 2  # GTRasterTypeGeoKey: RasterPixelIsPoint
    for key in (3072, 2048):  # ProjectedCSTypeGeoKey, GeographicTypeGeoKey
        code = found.get(key)
        if code and code != 32767:  # 32767: user-defined
            return f"EPSG:{code}", point
    return None, point


def read_points(path: str | Path, limit: int | None = None) -> np.ndarray:
    """Scattered points ``x y z`` per line (separated by spaces, tabs, commas or
    semicolons; lines that do not start with a number, e.g. a header, are skipped),
    as (n, 3) float. A letter before the numbers (geo.dat with G 19) is skipped."""
    path = Path(path)
    fast = _read_points_fast(path, limit)
    if fast is not None:
        return fast
    rows = []
    number = re.compile(r"^[\s]*[A-Za-z]?\s*[-+.\d]")
    with path.open("r", encoding="utf-8", errors="replace") as file:
        for line in file:
            if not number.match(line):
                continue
            parts = re.split(r"[\s,;]+", line.strip())
            if parts and parts[0] and parts[0][0].isalpha():
                parts = parts[1:]
            if len(parts) < 3:
                continue
            try:
                rows.append((float(parts[0]), float(parts[1]), float(parts[2])))
            except ValueError:
                continue
            if limit is not None and len(rows) >= limit:
                break
    if not rows:
        raise RasterError(f"{path.name}: no x y z points")
    return np.asarray(rows, float)


def _read_points_fast(path: Path, limit: int | None) -> np.ndarray | None:
    """Plain numeric columns (after a header of a few lines) with NumPy's reader;
    None for anything else (the careful reader then takes it)."""
    try:
        with path.open("r", encoding="utf-8", errors="replace") as file:
            head = [file.readline() for _ in range(20)]
    except OSError as error:
        raise RasterError(f"{path.name}: {error}") from None
    skip = 0
    for line in head:
        if re.match(r"^\s*[-+.\d]", line):
            break
        skip += 1
    else:
        return None
    first = head[skip]
    delimiter = "," if "," in first else (";" if ";" in first else None)
    try:
        values = np.loadtxt(
            path, delimiter=delimiter, skiprows=skip, usecols=(0, 1, 2), max_rows=limit, ndmin=2
        )
    except (ValueError, IndexError):
        return None
    return values if len(values) else None


def crs_of(crs: str | None) -> str | None:
    """'EPSG:<code>' for a code given any way ("25833", "epsg:25833"), else as given."""
    if not crs:
        return None
    code = epsg_code(crs)
    return f"EPSG:{code}" if code is not None else str(crs)
