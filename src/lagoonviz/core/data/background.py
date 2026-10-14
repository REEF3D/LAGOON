# SPDX-License-Identifier: GPL-3.0-or-later
# Architect: Hans Bihs
"""Background maps of the "Background map" items, and the georeference of cases.

A case's georeference is its ``georef.json`` (the set-up's), else the one its
LAGOON store or run log was written with. A background map item shows a map or
satellite image round its case's domain, from a tile server or a GeoTIFF, warped
into the model's frame (``lagoonviz.core.geo``). Making one may fetch tiles: the
GUI calls :meth:`Backgrounds.image` in a worker thread; the result is kept until
what it depends on (:meth:`Backgrounds.signature`) changes.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from lagoon_format.geo import Georef, GeorefError
from lagoon_format.geo import read as read_georef

from lagoonviz.core.kinds import BASEMAP_PLACEMENTS

if TYPE_CHECKING:
    from lagoonviz.core.data.manager import DataManager

#: the "source" choices of the item -> tile source keys (lagoonviz.core.geo.tiles)
SOURCE_CHOICES = {
    "Kartverket topo": "kartverket_topo",
    "Kartverket grey": "kartverket_grey",
    "Kartverket nautical chart": "kartverket_chart",
    "Sentinel-2 2017 (satellite)": "sentinel2_2017",
    "OpenStreetMap": "openstreetmap",
    "GeoTIFF file": "file",
    "Tile server URL": "url",
}
PLACEMENTS = BASEMAP_PLACEMENTS
#: 2D: the map lies under the bed (PLAN_LAYERS["bathymetry"] = 0)
PLAN_LEVEL = -0.5


@dataclass
class BackgroundMap:
    """What the renderer draws for a background map item: an RGBA image on a
    horizontal rectangle (optionally with the domain left out)."""

    item_id: str
    image: np.ndarray  # (rows, cols, 4) uint8, row 0 at the rectangle's y0
    extent: tuple[float, float, float, float]  # x0, x1, y0, y1 (model)
    z: float
    #: the domain to leave out (x0, x1, y0, y1), or None
    hole: tuple[float, float, float, float] | None
    opacity: float
    visible: bool
    attribution: str = ""

    def mesh(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(positions (n, 3), triangles (m, 3), texture coordinates (n, 2)): a plain
        rectangle, or a frame round the hole."""
        x0, x1, y0, y1 = self.extent
        xs = [x0, x1]
        ys = [y0, y1]
        if self.hole is not None:
            hx0, hx1, hy0, hy1 = self.hole
            xs = [x0, max(hx0, x0), min(hx1, x1), x1]
            ys = [y0, max(hy0, y0), min(hy1, y1), y1]
        gx, gy = np.meshgrid(np.asarray(xs, float), np.asarray(ys, float))
        positions = np.column_stack([gx.ravel(), gy.ravel(), np.full(gx.size, self.z)])
        u = (gx.ravel() - x0) / max(x1 - x0, 1e-12)
        v = (gy.ravel() - y0) / max(y1 - y0, 1e-12)
        nx = len(xs)
        triangles = []
        for j in range(len(ys) - 1):
            for i in range(nx - 1):
                if self.hole is not None and i == 1 and j == 1:
                    continue  # the domain
                a = j * nx + i
                triangles += [(a, a + 1, a + nx + 1), (a, a + nx + 1, a + nx)]
        return (
            positions.astype(np.float32),
            np.asarray(triangles, np.uint32),
            np.column_stack([u, v]).astype(np.float32),
        )


class Backgrounds:
    """The background maps of a data manager's scene, made once per signature."""

    def __init__(self, data: DataManager):
        self.data = data
        self._made: dict[str, tuple[tuple, object]] = {}  # item -> (signature, image)
        self._errors: dict[str, str] = {}
        #: item -> (what it was found from, (georef, error))
        self._georefs: dict[str, tuple[tuple, tuple[Georef | None, str]]] = {}
        self._lock = threading.Lock()
        #: item -> the signature its map could not be made with; items being made
        self._failed: dict[str, tuple] = {}
        self._making: set[str] = set()
        #: the 2D top view's area and pixels (set_view), or None
        self._view: tuple[tuple[float, float, float, float], int] | None = None
        self.cache = None  # the tile cache, made when first needed

    # ------------------------------------------------------------- georeference
    def georef(self, item_id: str) -> Georef | None:
        """The georeference of an item's case: georef.json in its folder, else its
        store's, its run log's or DIVEMesh's grid's; None without one (or one that
        does not read)."""
        found, _ = self._georef(item_id)
        return found

    def north(self, item_id: str) -> tuple[float, float] | None:
        """True north as a unit (x, y) vector in the model coordinates of an item's
        case, at the middle of its domain, or None without a georeference: the
        georeference turned back, and the CRS's own turn there (its grid north is
        true north only on its central meridian: a UTM zone's edge, or a place far
        from the zone, turns it)."""
        import math

        from lagoonviz.core.geo.crs import transformer

        georef = self.georef(item_id)
        if georef is None:
            return None
        r = math.radians(georef.rotation_deg)
        grid = (math.sin(r), math.cos(r))  # the CRS's y in the model
        domain = self.domain(item_id)
        cx, cy = ((domain[0] + domain[1]) / 2, (domain[2] + domain[3]) / 2) if domain else (0, 0)
        try:
            x, y = (float(v) for v in georef.to_world(cx, cy))
            lon, lat = (float(v) for v in transformer(georef.crs, "EPSG:4326")(x, y))
            step = 0.001 if lat < 89.0 else -0.001  # about 110 m
            x2, y2 = (float(v) for v in transformer("EPSG:4326", georef.crs)(lon, lat + step))
        except Exception:  # noqa: BLE001 (a CRS that does not transform: its grid north)
            return grid
        dx, dy = (x2 - x) * math.copysign(1.0, step), (y2 - y) * math.copysign(1.0, step)
        length = math.hypot(dx, dy)
        if not math.isfinite(length) or length == 0:
            return grid
        # world to model: turned back by the rotation
        mx = (dx * math.cos(r) + dy * math.sin(r)) / length
        my = (-dx * math.sin(r) + dy * math.cos(r)) / length
        return (mx, my)

    def georef_error(self, item_id: str) -> str:
        return self._georef(item_id)[1]

    def _georef(self, item_id: str) -> tuple[Georef | None, str]:
        folder = self.data.case_folder(item_id)
        case = self.data.case(item_id)
        stamp = None
        if folder is not None:
            from lagoon_format.geo import FILE

            path = folder / FILE
            try:
                info = path.stat()
                stamp = (info.st_mtime_ns, info.st_size)
            except OSError:
                stamp = None
        key = (str(folder), stamp, id(case))
        known = self._georefs.get(item_id)
        if known is not None and known[0] == key:
            return known[1]
        found = self._find_georef(folder, case, stamp is not None)
        self._georefs[item_id] = (key, found)
        return found

    @staticmethod
    def _find_georef(folder: Path | None, case, has_file: bool) -> tuple[Georef | None, str]:
        if folder is not None and has_file:
            try:
                found = read_georef(folder)
            except GeorefError as error:
                return None, str(error)
            if found is not None:
                return found, ""
        if case is not None:
            for dataset in case.datasets.values():
                store = getattr(dataset, "store", None)
                found = getattr(store, "georef", None)
                if found is not None:
                    return found, ""
            from lagoonviz.core.data.reef3d import run_log

            try:
                log = run_log(Path(case.path))
            except (OSError, ValueError):
                log = None
            for run in reversed(log.runs if log is not None else []):
                if run.georef is not None:
                    return run.georef, ""
        if folder is not None:  # the one DIVEMesh kept in its grid
            from lagoonviz.core.data.divemesh_grid import read_grid_georef

            found = read_grid_georef(folder)
            if found is not None:
                return found, ""
        return None, ""

    def georef_file(self, item_id: str) -> Path | None:
        from lagoon_format.geo import FILE

        folder = self.data.case_folder(item_id)
        return folder / FILE if folder is not None else None

    # ------------------------------------------------------------- the model's box
    def domain(self, item_id: str) -> tuple[float, float, float, float, float, float] | None:
        """The case's domain (x0, x1, y0, y1, z0, z1): the set-up's (B 10), else the
        box of its results."""
        setup = self.data.case_setup(item_id)
        if setup is not None and setup.domain is not None:
            return tuple(float(v) for v in setup.domain)  # type: ignore[return-value]
        case = self.data.case(item_id)
        if case is None:
            return None
        from lagoonviz.core.data.manager import _results_extent

        extent = _results_extent(case)
        if extent is None:
            return None
        low, high = extent
        return (low[0], high[0], low[1], high[1], low[2], high[2])

    # ------------------------------------------------------------- items
    def items(self) -> list[str]:
        return [i.id for i in self.data.scene if i.kind == "basemap"]

    def error(self, item_id: str) -> str:
        return self._errors.get(item_id, "")

    def making(self, item_id: str) -> bool:
        """Whether the item's map is being made now (in a worker thread)."""
        with self._lock:
            return item_id in self._making

    def signature(self, item_id: str) -> tuple | None:
        """What an item's map depends on (None: nothing to draw)."""
        item = self.data.scene.get(item_id)
        georef = self.georef(item_id)
        domain = self.domain(item_id)
        if georef is None or domain is None:
            return None
        props = item.properties
        file = str(props.get("file", "") or "")
        stamp = None
        if props.get("source") == "GeoTIFF file" and file:
            path = Path(file).expanduser()
            stamp = (path.stat().st_mtime_ns, path.stat().st_size) if path.is_file() else None
        return (
            georef.to_json(),
            tuple(round(v, 6) for v in domain[:4]),
            str(props.get("source")),
            file,
            stamp,
            float(props.get("margin", 0.5)),
            int(props.get("pixels", 2048)),
        )

    def image(self, item_id: str):
        """The item's map in the model frame (a ModelImage), made now if what it
        depends on changed (this may fetch tiles). None when it cannot be made (see
        :meth:`error`). The GUI makes it in a worker thread with :meth:`job`."""
        job = self.job(item_id)
        if job is None:
            return self.cached(item_id)
        return job()

    def job(self, item_id: str):
        """The work of making an item's map, as a function to call in any thread
        (it reads only what it was given and stores what it made); None when there
        is nothing to make (made already, or it cannot be: see :meth:`error`)."""
        from lagoonviz.core.geo.basemap import background
        from lagoonviz.core.geo.raster import read_raster
        from lagoonviz.core.geo.tiles import SOURCES, TileCache, custom_source

        signature = self.signature(item_id)
        if signature is None:
            georef_error = self.georef_error(item_id)
            self._errors[item_id] = georef_error or (
                "the case has no georeference: set its CRS and origin in the set-up (Map)"
                if self.georef(item_id) is None
                else "the case's domain is not known yet"
            )
            return None
        if self.cached(item_id) is not None:
            return None
        with self._lock:
            if self._failed.get(item_id) == signature or item_id in self._making:
                return None  # failed as it is now (see error), or being made
        props = dict(self.data.scene.get(item_id).properties)
        georef = self.georef(item_id)
        domain = self.domain(item_id)
        assert georef is not None and domain is not None
        box, pixels = self._wanted(item_id, domain, props)
        choice = SOURCE_CHOICES.get(str(props.get("source")), "kartverket_topo")
        file = str(props.get("file", "") or "").strip()
        if self.cache is None:
            self.cache = TileCache()
        cache = self.cache
        with self._lock:
            self._making.add(item_id)

        def make(progress=None):
            try:
                if choice == "file":
                    if not file:
                        raise ValueError("choose the GeoTIFF file (File or URL)")
                    source = read_raster(Path(file).expanduser())
                    if source.crs is None:
                        source.crs = georef.crs  # an image without a CRS: the case's
                elif choice == "url":
                    source = custom_source(file)
                else:
                    source = SOURCES[choice]
                result = background(georef, box, source, pixels, cache, progress=progress)
            except Exception as error:  # noqa: BLE001 (a bad file or server: say why)
                self._errors[item_id] = str(error) or type(error).__name__
                with self._lock:
                    self._failed[item_id] = signature  # not again until it changes
                    self._making.discard(item_id)
                return None
            self._errors.pop(item_id, None)
            if result.missing:
                self._errors[item_id] = (
                    f"{result.missing} map tiles could not be fetched (offline?); "
                    "shown without them"
                )
            with self._lock:
                self._made[item_id] = (signature, result)
                self._failed.pop(item_id, None)
                self._making.discard(item_id)
            return result

        return make

    def cached(self, item_id: str):
        """The map made before for the item's current signature (and covering the 2D
        view finely enough, when it follows it), else None (no work)."""
        signature = self.signature(item_id)
        with self._lock:
            made = self._made.get(item_id)
        if made is None or made[0] != signature:
            return None
        domain = self.domain(item_id)
        props = self.data.scene.get(item_id).properties
        if domain is not None and not _covers(made[1], *self._wanted(item_id, domain, props)):
            return None  # made for another view
        return made[1]

    def latest(self, item_id: str):
        """The map made last for the item as it is now, even when it does not cover
        the 2D view (shown while the one for the view is being made), else None."""
        signature = self.signature(item_id)
        with self._lock:
            made = self._made.get(item_id)
        return made[1] if made is not None and made[0] == signature else None

    # ------------------------------------------------------------- the 2D view
    def set_view(self, box: tuple[float, float, float, float] | None, pixels: int = 0) -> None:
        """The area the 2D top view shows (x0, x1, y0, y1, model coordinates) and its
        width in pixels: the maps follow it, fetched again for a new area or zoom.
        None: the maps cover the domain and their margin (as in 3D)."""
        self._view = None if box is None else (tuple(float(v) for v in box), int(pixels))

    def _following(self) -> bool:
        return self._view is not None and self.data.scene.flat

    def _wanted(self, item_id: str, domain, props) -> tuple[tuple, int]:
        """The rectangle a map covers and its pixels along the longer side: the 2D
        view's and a quarter round it (when it follows the view), else the domain and
        its margin."""
        from lagoonviz.core.geo.basemap import expanded

        if self._following():
            box, pixels = self._view
            return expanded(box, 0.25), int(min(max(pixels * 1.5, 512), 4096))
        return expanded(domain[:4], float(props.get("margin", 0.5))), int(props.get("pixels", 2048))

    def shown(self, item_id: str, made) -> BackgroundMap:
        """Where and how the renderer draws a made map: below the model or at the
        still water level round the domain, flat under everything in 2D."""
        item = self.data.scene.get(item_id)
        props = item.properties
        domain = self.domain(item_id)
        placement = str(props.get("placement", PLACEMENTS[0]))
        hole = None
        if self.data.scene.flat:
            z = PLAN_LEVEL
        elif placement == PLACEMENTS[1]:
            z = self._water_level(item_id)
            hole = tuple(domain[:4]) if domain is not None else None
        else:
            low = domain[4] if domain is not None else 0.0
            level = self._water_level(item_id)
            size = max(domain[1] - domain[0], domain[3] - domain[2]) if domain else 1.0
            exaggeration = self.data.exaggeration(item_id)
            z = level + exaggeration * (low - level) - 0.01 * size
        return BackgroundMap(
            item_id,
            made.image,
            made.extent,
            float(z),
            hole,
            float(props.get("opacity", 1.0)),
            item.visible,
            made.attribution,
        )

    def _water_level(self, item_id: str) -> float:
        case = self.data.case(item_id)
        if case is not None and not case.setup_only:
            return self.data.case_water_level(item_id)
        setup = self.data.case_setup(item_id)
        if setup is not None and setup.water_level is not None:
            return float(setup.water_level)
        return 0.0

    def attributions(self) -> list[str]:
        """The credits of the maps drawn now (visible items, maps made), for films
        and figures."""
        found = []
        for item_id in self.items():
            if not self.data.scene.get(item_id).visible or not self.data.drawn(item_id):
                continue
            made = self.cached(item_id)
            if made is not None and made.attribution:
                found.append(made.attribution)
        return list(dict.fromkeys(found))

    def forget(self, item_id: str | None = None) -> None:
        with self._lock:
            if item_id is None:
                self._made.clear()
                self._failed.clear()
            else:
                self._made.pop(item_id, None)
                self._failed.pop(item_id, None)


def _covers(made, box, pixels: int) -> bool:
    """Whether a made map holds a rectangle (x0, x1, y0, y1) finely enough for
    ``pixels`` along its longer side (and not much more coarsely than wanted)."""
    x0, x1, y0, y1 = made.extent
    bx0, bx1, by0, by1 = box
    # the view's own area (the quarter round it may come from the map made before):
    # panning by up to a quarter of the view keeps the map
    gx, gy = (bx1 - bx0) / 6.0, (by1 - by0) / 6.0
    inside = x0 <= bx0 + gx and x1 >= bx1 - gx and y0 <= by0 + gy and y1 >= by1 - gy
    have = (x1 - x0) / max(made.image.shape[1], 1)  # metres per pixel
    want = max(bx1 - bx0, by1 - by0) / max(pixels, 1)
    return inside and want / 2.5 <= have <= 1.6 * want
