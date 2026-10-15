"""The Earth under a case: CRS transforms, map tiles, rasters, bathymetry to geo.dat,
background maps and the session's georeference."""

import io
import math
import sys

import numpy as np
import pytest
from lagoon_format.geo import FILE, Georef
from PIL import Image, TiffImagePlugin

from lagoonviz.core import Session
from lagoonviz.core.data.background import SOURCE_CHOICES
from lagoonviz.core.geo import crs as crs_module
from lagoonviz.core.geo.basemap import background, expanded, model_grid, warp
from lagoonviz.core.geo.bathy import (
    Bathymetry,
    geo_grid,
    grid_heights,
    interpolate,
    load,
    write_geo_dat,
)
from lagoonviz.core.geo.crs import CrsError, describe, transformer, utm_for
from lagoonviz.core.geo.raster import RasterError, read_points, read_raster
from lagoonviz.core.geo.tiles import (
    SOURCES,
    TILE,
    TileCache,
    TileSource,
    custom_source,
    mosaic,
    resolution,
    tile_bounds,
    tile_range,
    zoom_for,
)
from lagoonviz.core.kinds import BASEMAP_SOURCES

GEOREF = Georef("EPSG:25833", (270000.0, 7040000.0), 30.0)


# ------------------------------------------------------------------ CRS
def test_transverse_mercator_on_the_central_meridian_is_the_meridian_arc():
    from scipy.integrate import quad

    e2 = crs_module.F * (2 - crs_module.F)
    to_utm = transformer("EPSG:4326", "EPSG:32631")
    for lat in (10.0, 48.858222, 63.43, 70.0):
        arc = quad(
            lambda p: crs_module.A * (1 - e2) / (1 - e2 * np.sin(p) ** 2) ** 1.5,
            0,
            np.radians(lat),
            epsabs=1e-10,
        )[0]
        x, y = to_utm(3.0, lat)
        assert x == pytest.approx(500000.0) and y == pytest.approx(0.9996 * arc, abs=1e-4)


def test_utm_against_the_usgs_series_off_the_meridian():
    # Snyder's series (USGS Professional Paper 1395), good to a millimetre here
    x, y = transformer("EPSG:4326", "EPSG:32632")(10.3951, 63.4305)
    assert (x, y) == pytest.approx((569618.6286, 7034313.849), abs=2e-3)


def test_round_trips_through_lon_lat_and_web_mercator():
    rng = np.random.default_rng(0)
    lon, lat = rng.uniform(5, 30, 500), rng.uniform(56, 71, 500)
    for code in ("EPSG:25833", "EPSG:32632", "EPSG:5110", "EPSG:3857"):
        x, y = transformer("EPSG:4326", code)(lon, lat)
        back = transformer(code, "EPSG:4326")(x, y)
        assert np.allclose(back[0], lon, atol=1e-9) and np.allclose(back[1], lat, atol=1e-9)
    x, y = transformer("EPSG:25833", "EPSG:25832")(*transformer("EPSG:4326", "EPSG:25833")(10, 63))
    assert transformer("EPSG:25832", "EPSG:4326")(x, y) == pytest.approx((10, 63))


def test_crs_names_and_zones():
    assert describe("EPSG:25833") == "ETRS89 / UTM zone 33N"
    assert describe("EPSG:5110") == "ETRS89 / NTM zone 10"
    assert utm_for(10.39, 63.43) == "EPSG:25832" and utm_for(5.3, 60.4) == "EPSG:25832"
    assert utm_for(-70.0, -33.0) == "EPSG:32719"


@pytest.mark.skipif(crs_module.has_pyproj(), reason="pyproj transforms any CRS")
def test_other_crs_need_pyproj():
    with pytest.raises(CrsError, match="pyproj"):
        transformer("EPSG:2056", "EPSG:4326")


# ------------------------------------------------------------------ tiles
def _png(colour) -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (TILE, TILE), colour).save(buffer, "PNG")
    return buffer.getvalue()


class FakeServer:
    """Tiles coloured by their column and row; ``down``: tiles it does not have."""

    def __init__(self, down=()):
        self.asked = []
        self.down = set(down)

    def __call__(self, url: str) -> bytes:
        self.asked.append(url)
        z, x, y = (int(v.split(".")[0]) for v in url.split("/")[-3:])
        if (x, y) in self.down:
            raise OSError("404")
        return _png((x % 256, y % 256, z))


SOURCE = TileSource("test", "Test", "https://tiles.test/{z}/{x}/{y}", "© test", "CC0")


def test_tile_maths():
    assert resolution(0) == pytest.approx(156543.03392804097)
    assert zoom_for(resolution(12)) == 12 and zoom_for(resolution(12) * 0.9) == 13
    col0, col1, row0, row1 = tile_range(tile_bounds(10, 555, 260), 10)
    assert (col0, row0) == (555, 260) and col1 - col0 <= 1 and row1 - row0 <= 1


def test_a_mosaic_is_put_together_and_cached(tmp_path):
    server = FakeServer()
    cache = TileCache(tmp_path, server)
    bounds = tile_bounds(8, 140, 70)
    shrunk = (bounds[0] + 1, bounds[1] + 1, bounds[2] + 1000, bounds[3] - 1)
    made = mosaic(SOURCE, shrunk, 8, cache)
    assert made.image.shape == (TILE, 2 * TILE, 4) and made.missing == 0
    assert tuple(made.image[0, 0, :3]) == (140, 70, 8) and tuple(made.image[0, -1, :3]) == (
        141,
        70,
        8,
    )
    assert made.bounds[0] == pytest.approx(bounds[0])
    asked = len(server.asked)
    mosaic(SOURCE, shrunk, 8, cache)
    assert len(server.asked) == asked  # from the disk cache
    assert (tmp_path / "test" / "8" / "140" / "70.png").is_file()


def test_tiles_that_cannot_be_had_are_left_transparent(tmp_path):
    cache = TileCache(tmp_path, FakeServer(down={(141, 70)}))
    bounds = tile_bounds(8, 140, 70)
    made = mosaic(SOURCE, (bounds[0], bounds[1], bounds[2] + 1000, bounds[3] - 1), 8, cache)
    assert made.missing == 1 and made.image[0, -1, 3] == 0 and made.image[0, 0, 3] == 255


def test_sources_name_their_terms():
    for source in SOURCES.values():
        assert source.attribution and source.licence and "{z}" in source.url
    assert "Kartverket" in SOURCES["kartverket_topo"].attribution
    assert SOURCES["sentinel2_2017"].licence == SOURCES["sentinel2_2016"].licence == "CC BY 4.0"
    assert custom_source("https://x.test/{z}/{x}/{y}.png").attribution == "x.test"
    with pytest.raises(ValueError):
        custom_source("https://x.test/{z}/{x}.png")
    assert tuple(SOURCE_CHOICES) == BASEMAP_SOURCES


def test_sentinel_2_in_europe_and_elsewhere(tmp_path):
    """The 2017 mosaic has images of Europe only: elsewhere (Mumbai) the 2016 world
    mosaic; the old name of the choice still reads."""
    from lagoonviz.core.data.background import MAP_FILE, _here, remembered_map
    from lagoonviz.core.geo.tiles import here, source_for
    from lagoonviz.core.kinds import validate

    s2 = SOURCES["sentinel2_2017"]
    assert here(s2, 10.4, 63.4) is s2  # Trondheim
    assert here(s2, 72.9, 19.1).key == "sentinel2_2016"  # Mumbai
    assert source_for(72.9, 19.1, "sentinel2_2017").key == "sentinel2_2016"
    assert here(SOURCES["openstreetmap"], 72.9, 19.1).key == "openstreetmap"
    mumbai = Georef("EPSG:32643", (274766.7, 2107774.7), 0.0)
    assert _here(s2, mumbai, (0.0, 2000.0, 0.0, 1000.0)).key == "sentinel2_2016"
    old = "Sentinel-2 2017 (satellite)"
    assert validate("basemap", "source", old) == "Sentinel-2 (satellite)"
    (tmp_path / MAP_FILE).write_text(f'{{"source": "{old}"}}', "utf-8")
    assert remembered_map(tmp_path)["source"] == "Sentinel-2 (satellite)"


# ------------------------------------------------------------------ rasters
def write_geotiff(path, data, x0, y0, size, epsg=25833, point=False, size_y=None):
    """A north-up GeoTIFF (float32 heights or uint8 RGB) written with Pillow."""
    info = TiffImagePlugin.ImageFileDirectory_v2()
    info[33550] = (float(size), float(size if size_y is None else size_y), 0.0)
    info.tagtype[33550] = 12
    info[33922] = (0.0, 0.0, 0.0, float(x0), float(y0), 0.0)
    info.tagtype[33922] = 12
    keys = (1, 1, 0, 2, 1025, 0, 1, 2 if point else 1, 3072, 0, 1, epsg)
    info[34735] = keys
    info.tagtype[34735] = 3
    info[42113] = "-9999"
    info.tagtype[42113] = 2
    if data.ndim == 3:
        image = Image.fromarray(data.astype(np.uint8), "RGB")
    else:
        image = Image.fromarray(data.astype(np.float32), "F")
    image.save(path, tiffinfo=info)


def plane(x, y):
    """A sloping bed: -20 m at the origin's east, rising to the north-east."""
    return -20.0 + 0.01 * (x - 270000.0) + 0.02 * (y - 7040000.0)


@pytest.fixture
def terrain(tmp_path):
    x0, y0, size, n = 269000.0, 7042000.0, 10.0, 300
    cols, rows = np.meshgrid(np.arange(n) + 0.5, np.arange(n) + 0.5)
    data = plane(x0 + cols * size, y0 - rows * size)
    data[:3, :3] = -9999.0
    path = tmp_path / "terrain.tif"
    write_geotiff(path, data, x0, y0, size)
    return path


@pytest.mark.parametrize("reader", ["tifffile", "pillow"])
def test_geotiff_heights(terrain, monkeypatch, reader):
    if reader == "pillow":
        monkeypatch.setitem(sys.modules, "tifffile", None)
    elif pytest.importorskip("tifffile") is None:
        return
    raster = read_raster(terrain)
    assert raster.crs == "EPSG:25833" and raster.shape == (300, 300)
    assert raster.bounds() == pytest.approx((269000.0, 7039000.0, 272000.0, 7042000.0))
    values = raster.values()
    assert np.isnan(values[0, 0]) and values[10, 10] == pytest.approx(
        plane(269105, 7041895), abs=1e-3
    )
    sampled = raster.sample([270000.0, 270123.4], [7040000.0, 7040567.8])
    assert sampled == pytest.approx(
        plane(np.array([270000.0, 270123.4]), np.array([7040000.0, 7040567.8])), abs=1e-3
    )


def test_ascii_grid_and_points(tmp_path):
    asc = tmp_path / "bed.asc"
    asc.write_text(
        "ncols 3\nnrows 2\nxllcorner 100\nyllcorner 200\ncellsize 5\nNODATA_value -9999\n"
        "1 2 3\n4 5 -9999\n"
    )
    (tmp_path / "bed.prj").write_text('PROJCS["x",AUTHORITY["EPSG","25832"]]')
    raster = read_raster(asc)
    assert raster.crs == "EPSG:25832" and raster.bounds() == (100.0, 200.0, 115.0, 210.0)
    assert raster.values()[0, 0] == 1 and np.isnan(raster.values()[1, 2])
    xyz = tmp_path / "soundings.csv"
    xyz.write_text("x,y,z\n1.5,2.5,-3\n4,5,-6.25\n")
    assert read_points(xyz).tolist() == [[1.5, 2.5, -3.0], [4.0, 5.0, -6.25]]
    marked = tmp_path / "geo.dat"
    marked.write_text("a 1 2 3\nb 4 5 6\n")
    assert read_points(marked).tolist() == [[1, 2, 3], [4, 5, 6]]
    with pytest.raises(RasterError):
        read_raster(tmp_path / "soundings.csv")


# ------------------------------------------------------------------ warping and bathymetry
def test_a_raster_warped_into_a_turned_model(terrain):
    raster = read_raster(terrain)
    xs, ys = np.linspace(100, 900, 41), np.linspace(-300, 400, 36)
    heights = warp(raster, GEOREF.replace(z_offset=2.0), xs, ys)
    gx, gy = np.meshgrid(xs, ys)
    big_x, big_y = GEOREF.to_world(gx, gy)
    assert np.allclose(heights, plane(big_x, big_y) - 2.0, atol=1e-3)


def test_points_interpolated_onto_the_grid(terrain):
    raster = read_raster(terrain)
    world = raster.points(step=3)
    lon, lat = transformer("EPSG:25833", "EPSG:4326")(world[:, 0], world[:, 1])
    soundings = Bathymetry(
        "s", "EPSG:4326", points=np.column_stack([lon, lat, -world[:, 2]]), depths=True
    )
    xs, ys = np.linspace(100, 900, 21), np.linspace(-300, 400, 15)
    gx, gy = np.meshgrid(xs, ys)
    truth = plane(*GEOREF.to_world(gx, gy))
    assert np.allclose(grid_heights(soundings, GEOREF, xs, ys, "tin"), truth, atol=1e-3)
    assert np.allclose(grid_heights(soundings, GEOREF, xs, ys, "idw"), truth, atol=1.0)
    assert np.allclose(grid_heights(soundings, GEOREF, xs, ys, "nearest"), truth, atol=1.0)
    # outside the points: the nearest one's height, no holes
    far = interpolate(np.array([[0, 0, 1.0], [1, 0, 2.0], [0, 1, 3.0]]), [5.0], [5.0])
    assert far.shape == (1, 1) and np.isfinite(far).all()


def test_geo_dat_grid_and_file(tmp_path):
    xs, ys = geo_grid((0.0, 10.0, 0.0, 5.0), 1.0)
    assert xs[0] == -2.0 and xs[-1] == 12.0 and len(ys) == 10
    heights = np.zeros((len(ys), len(xs)))
    heights[0, 0] = np.nan
    count = write_geo_dat(tmp_path / "geo.dat", xs, ys, heights)
    lines = (tmp_path / "geo.dat").read_text().splitlines()
    assert count == len(lines) == len(xs) * len(ys) - 1 and lines[0] == "-1.0000 -2.0000 0.0000"


def test_a_background_map_round_the_domain(tmp_path):
    server = FakeServer()
    box = expanded((0.0, 1000.0, 0.0, 500.0), 0.25)
    made = background(GEOREF, box, SOURCE, 512, TileCache(tmp_path, server))
    assert made.image.shape[1] == 512 and made.image.dtype == np.uint8
    assert made.attribution == "© test" and made.zoom is not None and made.missing == 0
    assert (made.image[..., 3] == 255).all()
    xs, ys, size = model_grid(box, 512)
    assert size == pytest.approx(1500 / 512)
    # a model point's colour is the tile under it on the map
    lon, lat = transformer("EPSG:25833", "EPSG:4326")(*GEOREF.to_world(500.0, 250.0))
    merc = transformer("EPSG:4326", "EPSG:3857")(lon, lat)
    tile = tile_range((merc[0], merc[1], merc[0], merc[1]), made.zoom)
    i, j = np.searchsorted(ys, 250.0), np.searchsorted(xs, 500.0)
    assert tuple(made.image[i, j, :2]) == (tile[0] % 256, tile[2] % 256)


def test_a_geotiff_picture_as_background(tmp_path):
    picture = np.zeros((100, 100, 3), np.uint8)
    picture[:, 50:] = (200, 10, 10)  # the east half red
    path = tmp_path / "ortho.tif"
    write_geotiff(path, picture, 269500.0, 7040500.0, 10.0)
    image = read_raster(path)
    made = background(
        Georef("EPSG:25833", (270000.0, 7040000.0)), (-400, 400, -400, 400), image, 80
    )
    assert made.image.shape == (80, 80, 4)
    assert tuple(made.image[40, 70, :3]) == (200, 10, 10) and tuple(made.image[40, 10, :3]) == (
        0,
        0,
        0,
    )
    assert made.image[1, 1, 3] == 255


# ------------------------------------------------------------------ the session
CONTROL = """\
B 1 50.0
B 10 0.0 1000.0 0.0 500.0 -30.0 5.0
C 11 6
"""


@pytest.fixture
def fjord(tmp_path):
    case = tmp_path / "fjord"
    case.mkdir()
    (case / "control.txt").write_text(CONTROL)
    (case / "ctrl.txt").write_text("A 10 5\nF 60 0.0\n")
    return case


def test_the_session_sets_and_turns_the_georeference(fjord):
    session = Session()
    session.open_case(fjord)
    assert session.georef() is None
    session.set_georef("EPSG:25833", origin=(270000.0, 7040000.0), rotation_deg=30.0)
    assert Georef.from_json((fjord / FILE).read_text()) == GEOREF
    assert session.georef() == GEOREF
    centre = GEOREF.to_world(500.0, 250.0)
    session.turn_model(45.0)
    assert session.georef().rotation_deg == 45.0
    assert session.georef().to_world(500.0, 250.0) == pytest.approx(centre)
    session.undo()
    assert session.georef() == GEOREF
    session.undo()
    assert not (fjord / FILE).exists() and session.georef() is None
    session.redo()
    script = session.to_script()
    assert (
        "session.set_georef('EPSG:25833', origin=[270000.0, 7040000.0], rotation_deg=30.0)"
        in script
    )


def test_the_session_sets_the_domain_and_makes_geo_dat(fjord, terrain):
    session = Session()
    session.open_case(fjord)
    with pytest.raises(ValueError, match="georeference"):
        session.make_geo_dat(terrain)
    session.set_georef("EPSG:25833", origin=(270000.0, 7040000.0), rotation_deg=30.0)
    session.set_domain(100.0, 900.0, -300.0, 400.0)
    assert session.input_file("control.txt").values("B 10") == [
        100.0,
        900.0,
        -300.0,
        400.0,
        -30.0,
        5.0,
    ]
    count = session.make_geo_dat(terrain)
    points = np.loadtxt(fjord / "geo.dat")
    assert count == len(points) and points[:, 0].min() == pytest.approx(0.0)
    assert np.diff(np.unique(points[:, 0]))[0] == pytest.approx(50.0)
    assert np.allclose(points[:, 2], plane(*GEOREF.to_world(points[:, 0], points[:, 1])), atol=1e-3)
    assert session.input_file("control.txt").int("G 10") == 1
    setup = session.data.case_setup(session.scene.ids()[0])
    assert len(setup.geo_points()) == count
    geodat = next(i for i in session.scene if i.kind == "geodat")  # its points, an item
    assert session.data.load(geodat.id).n_points == count
    assert "session.make_geo_dat(" in session.to_script()


def test_geo_dat_follows_a_turned_model(fjord, terrain):
    """geo.dat is in model coordinates: turned or moved on the map, the model needs
    it again (DIVEMesh would mesh the bed of before); made again from its source."""
    session = Session()
    session.open_case(fjord)
    session.set_georef("EPSG:25833", origin=(270000.0, 7040000.0), rotation_deg=30.0)
    assert session.geo_dat_state() == "" and session.geo_dat_recipe() is None
    session.make_geo_dat(terrain, spacing=25.0)
    made = session.geo_dat_recipe()
    assert made["rotation_deg"] == 30.0 and made["spacing"] == 25.0
    assert made["source"] == str(terrain) and session.geo_dat_state() == ""
    session.turn_model(45.0)
    assert session.geo_dat_state().startswith("geo.dat was made with the model turned 30°")
    session.undo()  # turned back: it fits again
    assert session.geo_dat_state() == ""
    session.redo()
    count = session.remake_geo_dat()
    assert session.geo_dat_state() == "" and session.geo_dat_recipe()["rotation_deg"] == 45.0
    points = np.loadtxt(fjord / "geo.dat")
    turned = session.data.georef(session.scene.ids()[0])
    assert count == len(points)
    assert np.allclose(points[:, 2], plane(*turned.to_world(points[:, 0], points[:, 1])), atol=1e-3)
    assert np.diff(np.unique(points[:, 0]))[0] == pytest.approx(25.0)  # the same spacing
    # moved, another CRS, a larger domain
    current = session.data.georef(session.scene.ids()[0])
    session.set_georef("EPSG:25833", origin=(270100.0, 7040000.0), rotation_deg=45.0)
    assert "moved" in session.geo_dat_state()
    session.set_georef(current.crs, origin=current.origin, rotation_deg=45.0)
    assert session.geo_dat_state() == ""
    session.set_domain(0.0, 1500.0, 0.0, 500.0)
    assert session.geo_dat_state() == "the domain reaches beyond geo.dat"
    # geo.dat written by another program: nothing known
    (fjord / "geo.dat").write_text("0 0 -10\n10 0 -10\n0 10 -10\n")
    assert session.geo_dat_state() == "" and session.geo_dat_recipe() is None
    with pytest.raises(ValueError, match="not made by LAGOON"):
        session.remake_geo_dat()


def test_a_background_map_item(fjord, tmp_path):
    session = Session()
    session.open_case(fjord)
    item = session.add_background_map("Kartverket topo", margin=0.2, pixels=256)
    backgrounds = session.data.backgrounds
    assert backgrounds.image(item) is None and "georeference" in backgrounds.error(item)
    session.set_georef("EPSG:25833", origin=(270000.0, 7040000.0), rotation_deg=30.0)
    backgrounds.cache = TileCache(tmp_path / "tiles", FakeServer())
    made = backgrounds.image(item)
    assert made is not None and made.extent == pytest.approx((-200.0, 1200.0, -200.0, 700.0))
    assert backgrounds.cached(item) is made
    shown = backgrounds.shown(item, made)
    positions, triangles, uv = shown.mesh()
    assert shown.hole is None and len(triangles) == 2 and shown.z < -30.0
    session.set_property(item, "placement", "at the still water level")
    shown = backgrounds.shown(item, made)
    positions, triangles, uv = shown.mesh()
    assert shown.z == 0.0 and shown.hole == (0.0, 1000.0, 0.0, 500.0) and len(triangles) == 16
    assert uv.min() == 0.0 and uv.max() == 1.0
    session.turn_model(10.0)
    assert backgrounds.cached(item) is None  # turned: made again
    assert session.data.item_mode(item) == "both"


def test_a_rotation_tried_turns_the_map_and_north(fjord, tmp_path):
    """A rotation slider dragged: the map made before is shown turned to the
    rotation tried (as the georeference would put it), north follows; nothing is
    written until it is kept. The map made for the old rotation is shown turned to
    the new one while the new one is made."""
    session = Session()
    case = session.open_case(fjord)
    item = session.add_background_map("Kartverket topo", margin=0.2, pixels=256)
    session.set_georef("EPSG:25833", origin=(270000.0, 7040000.0), rotation_deg=30.0)
    backgrounds = session.data.backgrounds
    backgrounds.cache = TileCache(tmp_path / "tiles", FakeServer())
    made = backgrounds.image(item)
    north = backgrounds.north(case)
    assert backgrounds.shown(item, made).transform is None
    seen = []
    session.scene.subscribe(lambda event, _id: seen.append(event))
    # the georeference's rotation, about the model's origin
    session.try_rotation(40.0)
    assert "georef_tried" in seen and session.georef() == GEOREF  # nothing written
    shown = backgrounds.shown(item, made)
    positions = shown.mesh()[0]
    corner = GEOREF.to_world(made.extent[0], made.extent[2])  # where it is on the Earth
    tried = GEOREF.rotated(40.0)
    assert positions[0, :2] == pytest.approx(tried.to_model(*corner), abs=1e-3)
    turned = backgrounds.north(case)
    angle = math.degrees(math.atan2(turned[1], turned[0]) - math.atan2(north[1], north[0]))
    assert angle == pytest.approx(-10.0, abs=0.01)  # north turns back as the model turns
    session.try_rotation(None)
    assert backgrounds.shown(item, made).transform is None
    # the model turned about the domain's centre: the centre stays on the map
    session.try_rotation(50.0, about=(500.0, 250.0))
    centre = GEOREF.to_world(500.0, 250.0)
    assert backgrounds.shown_georef(case).to_world(500.0, 250.0) == pytest.approx(centre)
    session.try_rotation(None)
    # kept: the georeference written, the old map shown turned until the new is made
    session.set_rotation(40.0)
    assert session.georef().rotation_deg == 40.0 and session.georef().origin == GEOREF.origin
    assert backgrounds.cached(item) is None and backgrounds.latest(item) is made
    assert backgrounds.shown(item, made).transform is not None
    assert session.to_script().splitlines()[-1] == "session.set_rotation(40.0)"
    session.undo()
    assert session.georef() == GEOREF
    # about a point (the middle of the view): it stays where it is on the map
    there = GEOREF.to_world(700.0, 100.0)
    session.set_rotation(55.0, about=(700.0, 100.0))
    assert session.georef().to_world(700.0, 100.0) == pytest.approx(there)
    assert session.to_script().splitlines()[-1] == (
        "session.set_rotation(55.0, about=[700.0, 100.0])"
    )


def test_a_store_converted_keeps_the_case_georeference(tmp_path):
    from lagoon_format.geo import write
    from lagoon_format.store import Store
    from reef3d_fixtures import make_run

    from lagoonviz.core.convert import convert

    run = make_run(tmp_path / "run", frames=2)
    write(run, GEOREF)
    report = convert(run)
    assert Store(report.store).georef == GEOREF
    session = Session()
    case = session.open_case(report.store)
    (run / FILE).unlink()
    assert session.data.georef(case) == GEOREF  # the store's


def test_load_tells_rasters_from_points(tmp_path, terrain):
    assert load(terrain).raster is not None
    xyz = tmp_path / "s.xyz"
    xyz.write_text("1 2 3\n")
    found = load(xyz, "25833")
    assert found.points is not None and found.crs == "EPSG:25833"


def test_the_sketch_and_films_show_the_map_and_its_credit(fjord, tmp_path):
    from matplotlib.image import AxesImage

    from lagoonviz.render.movie import credit
    from lagoonviz.render.setup_sketch import setup_figure

    session = Session()
    session.open_case(fjord)
    session.set_georef("EPSG:25833", origin=(270000.0, 7040000.0), rotation_deg=30.0)
    item = session.add_background_map("Kartverket topo", pixels=256)
    session.data.backgrounds.cache = TileCache(tmp_path / "tiles", FakeServer())
    figure = setup_figure(session)
    plan = figure.axes[0]
    maps = [i for i in plan.get_images() if isinstance(i, AxesImage) and i.get_zorder() == -1]
    assert len(maps) == 1 and maps[0].get_extent() == pytest.approx((-500, 1500, -500, 1000))
    assert plan.get_xlim()[1] < 1200  # the plan keeps to the domain
    assert any("Kartverket" in t.get_text() for t in plan.texts)
    assert session.data.backgrounds.attributions() == ["© Kartverket"]
    frame = np.zeros((360, 640, 4), np.uint8)
    marked = credit(frame, ["© Kartverket"])
    assert marked.shape == frame.shape and marked[-3:, -40:].any() and not marked[:300].any()
    session.set_visible(item, False)
    assert session.data.backgrounds.attributions() == []


def test_divemesh_keeps_the_georeference_in_its_grid(fjord):
    import struct

    from lagoonviz.core.data.divemesh_grid import grid_sections, read_grid_georef

    folder = fjord / "DIVEMesh_Grid"
    folder.mkdir()
    text = GEOREF.to_json().replace("\n", " ").strip().encode()

    def section(tag, payload):
        return tag + struct.pack("<q", len(payload)) + payload

    data = b"DMGEOM02" + struct.pack("<ii", 2, 0x01020304)
    data += section(b"GHDR", struct.pack("<ii", 1, 0) + struct.pack("<i", 0))
    data += section(b"GREF", text) + section(b"END ", b"")
    (folder / "grid-geometry.dat").write_bytes(data)
    assert list(grid_sections(folder / "grid-geometry.dat")) == ["GHDR", "GREF"]
    assert read_grid_georef(fjord) == GEOREF
    session = Session()
    case = session.open_case(fjord)
    assert session.data.georef(case) == GEOREF  # no georef.json: DIVEMesh's
    assert read_grid_georef(fjord.parent) is None


def test_a_model_needs_a_crs_in_metres(fjord):
    session = Session()
    session.open_case(fjord)
    with pytest.raises(ValueError, match="metres"):
        session.set_georef("EPSG:4326", origin=(10.4, 63.4))


# ------------------------------------------------------------------ a case from the map
def test_coordinates_typed_in_many_ways():
    from lagoonviz.core.geo.places import parse_coordinates

    for text in (
        "63.44, 10.40",
        "63.44,10.40",
        "63.44N 10.4E",
        "10.4E 63.44N",
        "lat 63.44 lon 10.40",
        "lon=10.4, lat=63.44",
        "63,44 10,40",
    ):
        place = parse_coordinates(text)
        assert place is not None and (place.lat, place.lon) == pytest.approx((63.44, 10.4)), text
    assert parse_coordinates("33.9S 18.4E").lat == pytest.approx(-33.9)
    for text in ("Nidelva", "Trondheim 2", "63.44 10.40 5", "95 10"):
        assert parse_coordinates(text) is None, text


def test_place_names_from_kartverket_and_openstreetmap():
    import json

    from lagoonviz.core.geo.places import search

    def fetch(url):
        if "geonorge" in url:
            assert "sok=Nidelva" in url
            return json.dumps(
                {
                    "navn": [
                        {
                            "skrivemåte": "Nidelva",
                            "navneobjekttype": "Elv",
                            "kommuner": [{"kommunenavn": "Trondheim"}],
                            "representasjonspunkt": {"øst": 10.40, "nord": 63.43},
                        }
                    ]
                }
            ).encode()
        return json.dumps(
            [
                {
                    "name": "Nidelva",
                    "lat": "63.4301",
                    "lon": "10.4003",
                    "type": "river",
                    "display_name": "Nidelva, Trondheim, Trøndelag, Norge",
                },
                {
                    "name": "Nidelva",
                    "lat": "58.4",
                    "lon": "8.7",
                    "type": "river",
                    "display_name": "Nidelva, Arendal, Agder, Norge",
                },
            ]
        ).encode()

    found = search("Nidelva", fetch)
    assert [p.source for p in found] == ["Kartverket", "OpenStreetMap"]  # the same once
    assert found[0].where == "Trondheim" and "Elv" in found[0].label()
    assert search("63.44, 10.40", fetch)[0].kind == "coordinates"

    def down(url):
        raise OSError("no network")

    with pytest.raises(OSError, match="place search"):
        search("Nidelva", down)


def _kartverket_names(url):
    """A fake of Kartverket's place names: two Nidelvas (Arendal's first, as the
    service lists them), Munkholmen, Trondheim; the url's search and fuzzy flag."""
    import json
    import urllib.parse

    query = dict(urllib.parse.parse_qsl(urllib.parse.urlsplit(url).query))
    names = {
        "Nidelva": [
            ("Nidelva", "Elv", "Arendal", 8.70, 58.45),
            ("Nidelva", "Elv", "Trondheim", 10.40, 63.43),
        ],
        "Munkholmen": [("Munkholmen", "Øy", "Trondheim", 10.383, 63.451)],
        "Trondheim": [("Trondheim", "By", "Trondheim", 10.395, 63.430)],
    }
    text, fuzzy = query["sok"], query["fuzzy"] == "true"
    if text.endswith("*"):
        found = [e for n, es in names.items() if n.startswith(text[:-1]) for e in es]
    elif fuzzy:
        found = [
            e
            for n, es in names.items()
            if len(n) == len(text) + 1 and n[:3] == text[:3]
            for e in es
        ]
    else:
        found = names.get(text, [])
    return json.dumps(
        {
            "navn": [
                {
                    "skrivemåte": name,
                    "navneobjekttype": kind,
                    "kommuner": [{"kommunenavn": town}],
                    "representasjonspunkt": {"øst": lon, "nord": lat},
                }
                for name, kind, town, lon, lat in found
            ]
        }
    ).encode()


def _names_only_in_norway(url):
    return _kartverket_names(url) if "geonorge" in url else b"[]"


def test_a_place_name_with_its_municipality_or_near_the_case():
    from lagoonviz.core.geo.places import search

    found = search("Nidelva", _names_only_in_norway)
    assert {p.where for p in found} == {"Arendal", "Trondheim"}
    assert search("Nidelva, Trondheim", _names_only_in_norway)[0].where == "Trondheim"
    assert search("Nidelva, Arendal", _names_only_in_norway)[0].where == "Arendal"
    near_trondheim = search("Nidelva", _names_only_in_norway, near=(63.43, 10.39))
    assert near_trondheim[0].where == "Trondheim"
    near_arendal = search("Nidelva", _names_only_in_norway, near=(58.46, 8.77))
    assert near_arendal[0].where == "Arendal"


def test_a_place_name_begun_or_misspelt():
    from lagoonviz.core.geo.places import search

    assert search("Munkh", _names_only_in_norway)[0].name == "Munkholmen"  # beginning
    assert search("Trondhem", _names_only_in_norway)[0].name == "Trondheim"  # fuzzy
    assert search("Xyzzy", _names_only_in_norway) == []


def test_a_case_started_on_the_map(tmp_path):
    from lagoonviz.core.session import drafts_folder

    session = Session()
    case = session.new_case_from_map((63.44, 10.40), length=2000.0)
    assert session.is_draft(case) and session.mode == "setup" and session.scene.plan_view
    folder = session.data.case_folder(case)
    assert folder.parent == drafts_folder() and folder.name.startswith("untitled-")
    georef = session.georef(case)
    assert georef.crs == "EPSG:25832"
    lon, lat = transformer("EPSG:25832", "EPSG:4326")(*georef.to_world(1000.0, 500.0))
    assert (lat, lon) == pytest.approx((63.44, 10.40), abs=1e-7)  # the domain's centre
    maps = [i for i in session.scene if i.kind == "basemap"]
    assert len(maps) == 1 and maps[0].properties["source"] == "Kartverket topo"
    grid = next(i for i in session.scene if i.kind == "grid")
    assert not grid.visible  # its lines would hide the map
    assert session.input_file("control.txt", case).values("B 10") == [0, 2000, 0, 1000, 0, 1]
    # moved somewhere else: the same domain there, the map kept
    session.place_case((60.39, 5.32), case=case)
    assert session.georef(case).crs == "EPSG:25832"
    assert len([i for i in session.scene if i.kind == "basemap"]) == 1
    # into its folder
    session.set_domain(0.0, 1500.0, 0.0, 600.0, case=case)
    target = tmp_path / "bergen"
    (tmp_path / "full").mkdir()
    (tmp_path / "full" / "control.txt").write_text("B 1 1\n")
    assert session.case_file_clashes(tmp_path / "full", case) == ["control.txt"]
    with pytest.raises(FileExistsError, match="control.txt"):
        session.create_case(tmp_path / "full", case=case)
    session.create_case(target, case=case)
    assert not session.is_draft(case) and session.scene.get(case).name == "bergen"
    assert session.data.case_folder(case) == target
    assert "B 10 0 1500 0 600 0 1" in (target / "control.txt").read_text().replace(".0 ", " ")
    assert Georef.from_json((target / FILE).read_text()) == session.georef(case)
    session.undo()
    session.undo()
    assert session.is_draft(case)


def test_a_place_outside_norway_gets_openstreetmap():
    session = Session()
    case = session.new_case_from_map((-33.9, 18.4), length=500.0)
    assert session.georef(case).crs == "EPSG:32734"
    assert (
        next(i for i in session.scene if i.kind == "basemap").properties["source"]
        == "OpenStreetMap"
    )


def test_the_map_follows_the_2d_view(fjord, tmp_path):
    session = Session()
    session.open_case(fjord)
    session.set_georef("EPSG:25833", origin=(270000.0, 7040000.0))
    item = session.add_background_map("OpenStreetMap", margin=0.2, pixels=256)
    backgrounds = session.data.backgrounds
    backgrounds.cache = TileCache(tmp_path / "tiles", FakeServer())
    first = backgrounds.image(item)
    assert first.extent == pytest.approx((-200.0, 1200.0, -200.0, 700.0))
    # in 3D the view does not matter
    backgrounds.set_view((5000.0, 6000.0, 0.0, 500.0), 800)
    assert backgrounds.cached(item) is first
    session.set_plan_view(True, "top")
    assert backgrounds.cached(item) is None and backgrounds.latest(item) is first
    zoomed_out = backgrounds.image(item)
    assert zoomed_out.extent == pytest.approx((4750.0, 6250.0, -250.0, 750.0))
    # a small pan keeps it; zooming in a lot makes it again, finer
    backgrounds.set_view((5100.0, 6100.0, 0.0, 500.0), 800)
    assert backgrounds.cached(item) is zoomed_out
    backgrounds.set_view((5400.0, 5500.0, 200.0, 250.0), 800)
    assert backgrounds.cached(item) is None
    zoomed_in = backgrounds.image(item)
    width = zoomed_in.extent[1] - zoomed_in.extent[0]
    assert width == pytest.approx(150.0) and zoomed_in.image.shape[1] == 1200
    assert zoomed_in.missing == 0
    backgrounds.set_view(None)
    assert backgrounds.cached(item) is None  # back to the domain and its margin


# ------------------------------------------------------------------ bathymetry from the web
class FakeWcs:
    """Kartverket and EMODnet as GeoTIFF answers of a sloping bed (Kartverket only
    north of y = 7040500, as if the rest had no survey)."""

    def __init__(self, tmp_path):
        self.tmp = tmp_path
        self.asked = []

    def __call__(self, url):
        import re
        from urllib.parse import parse_qs, urlparse

        self.asked.append(url)
        if "copernicus" in url:  # open sea: no land tile
            raise OSError("HTTP Error 404: Not Found")
        if "geonorge" in url:
            q = {k: v[0] for k, v in parse_qs(urlparse(url).query).items()}
            assert q["coverage"] == "nhm_dtm_topobathy_25833" and q["crs"] == "EPSG:25833"
            x0, y0, x1, y1 = (float(v) for v in q["bbox"].split(","))
            w, h = int(q["width"]), int(q["height"])
            cols, rows = np.meshgrid(np.arange(w) + 0.5, np.arange(h) + 0.5)
            x, y = x0 + cols * (x1 - x0) / w, y1 - rows * (y1 - y0) / h
            data = np.where(y > 7040500, plane(x, y), -32767.0)
            path = self.tmp / "k.tif"
            write_geotiff(path, data, x0, y1, (x1 - x0) / w)
        else:
            lat0, lat1 = (float(v) for v in re.search(r"Lat\(([^,]+),([^)]+)\)", url).groups())
            lon0, lon1 = (float(v) for v in re.search(r"Long\(([^,]+),([^)]+)\)", url).groups())
            n = 200
            cols, rows = np.meshgrid(np.arange(n) + 0.5, np.arange(n) + 0.5)
            lon = lon0 + cols * (lon1 - lon0) / n
            lat = lat1 - rows * (lat1 - lat0) / n
            x, y = transformer("EPSG:4326", "EPSG:25833")(lon, lat)
            path = self.tmp / "e.tif"
            write_geotiff(
                path,
                plane(x, y) - 1.0,
                lon0,
                lat1,
                (lon1 - lon0) / n,
                4326,
                size_y=(lat1 - lat0) / n,
            )
        return path.read_bytes()


def test_geo_dat_from_the_web(fjord, tmp_path, monkeypatch):
    from lagoonviz.core.geo import download

    monkeypatch.setenv("LAGOON_CACHE", str(tmp_path / "cache"))
    server = FakeWcs(tmp_path)
    monkeypatch.setattr(download, "http_fetch", server)
    session = Session()
    session.open_case(fjord)
    session.set_georef("EPSG:25833", origin=(270000.0, 7040000.0), rotation_deg=30.0)
    count = session.make_geo_dat("best")
    # shallow sea: Copernicus asked for land where the coast could be (none: no tile)
    assert len(server.asked) == 3 and "copernicus" in server.asked[-1]
    assert session.geo_sources == ["Kartverket, land and sea (Norway)", "EMODnet (European seas)"]
    points = np.loadtxt(fjord / "geo.dat")
    assert count == len(points)
    world = GEOREF.to_world(points[:, 0], points[:, 1])
    truth = plane(*world)
    north = world[1] > 7040600
    assert np.allclose(points[north, 2], truth[north], atol=0.05)  # Kartverket's
    south = world[1] < 7040300
    assert np.allclose(points[south, 2], truth[south] - 1.0, atol=1.5)  # EMODnet's
    session.make_geo_dat("kartverket")
    assert len(server.asked) == 3  # the same area: from the cache
    assert "session.make_geo_dat('best'" in session.to_script()


def test_north_in_model_coordinates(fjord):
    session = Session()
    case = session.open_case(fjord)
    assert session.data.backgrounds.north(case) is None
    session.set_georef("EPSG:25833", origin=(270000.0, 7040000.0), rotation_deg=30.0)
    north = session.data.backgrounds.north(case)
    # the model point north of the domain's middle lies due north of it on the Earth:
    # not along the CRS's y, which is 4° off true north 4.6° west of UTM 33's meridian
    domain = session.data.backgrounds.domain(case)
    cx, cy = (domain[0] + domain[1]) / 2, (domain[2] + domain[3]) / 2
    to_lonlat = transformer("EPSG:25833", "EPSG:4326")
    lon0, lat0 = to_lonlat(*GEOREF.to_world(cx, cy))
    lon1, lat1 = to_lonlat(*GEOREF.to_world(cx + 1000 * north[0], cy + 1000 * north[1]))
    assert lon1 == pytest.approx(lon0, abs=1e-6) and lat1 > lat0
    grid = (math.sin(math.radians(30.0)), math.cos(math.radians(30.0)))
    turn = math.degrees(math.atan2(grid[0], grid[1]) - math.atan2(north[0], north[1]))
    assert turn == pytest.approx(-(15.0 - lon0) * math.sin(math.radians(lat0)), abs=0.1)


def test_a_place_far_from_the_cases_crs(fjord):
    """Mumbai for a case georeferenced in Norway: the place's UTM zone (Norway's
    would turn and stretch the map), and true north for the north-up view."""
    from lagoonviz.core.geo.crs import fits
    from lagoonviz.core.geo.places import Place

    assert fits("EPSG:25833", 10.4, 63.4) and not fits("EPSG:25833", 72.8, 19.0)
    assert not fits("EPSG:32633", 72.8, 19.0) and fits("EPSG:32643", 72.8, 19.0)
    assert not fits("EPSG:32643", 72.8, -19.0) and fits("EPSG:3857", 72.8, 19.0)
    session = Session()
    case = session.open_case(fjord)
    session.set_georef("EPSG:25833", origin=(270000.0, 7040000.0), rotation_deg=0.0)
    session.place_case(Place("Mumbai", 18.94, 72.84, "city"), case=case)
    georef = session.data.georef(case)
    assert georef.crs == "EPSG:32643" and georef.rotation_deg == 0.0
    north = session.data.backgrounds.north(case)
    # 2.2° west of UTM 43's meridian at 19° N: grid north 0.7° off true north
    assert math.degrees(math.atan2(north[0], north[1])) == pytest.approx(0.71, abs=0.05)
    # a CRS asked for is kept, fitting or not
    session.place_case(Place("Mumbai", 18.94, 72.84, "city"), case=case, crs="EPSG:3857")
    assert session.data.georef(case).crs == "EPSG:3857"


def test_a_case_opened_again_has_its_map(fjord):
    """The background map is kept with the case (LAGOON_map.json): its results opened
    anew, in another session, show it; a georeferenced case without one gets the
    place's best map; a map taken away stays away."""
    from lagoonviz.core.data.background import MAP_FILE

    def maps(session):
        return [i for i in session.scene if i.kind == "basemap"]

    session = Session()
    session.open_case(fjord)
    assert maps(session) == []  # no georeference: no map
    session.set_georef("EPSG:25833", origin=(270000.0, 7040000.0))
    item = session.add_background_map("OpenStreetMap", margin=1.0)
    session.set_property(item, "opacity", 0.6)
    assert (fjord / MAP_FILE).is_file()
    again = Session()
    case = again.open_case(fjord)
    found = maps(again)
    assert len(found) == 1 and found[0].properties["source"] == "OpenStreetMap"
    assert found[0].properties["margin"] == 1.0 and found[0].properties["opacity"] == 0.6
    assert again.data.case_item(found[0].id).id == case
    while case in again.scene:  # the opening undone: the map is not "taken away"
        again.undo()
    assert maps(again) == [] and len(maps(Session_open(fjord))) == 1
    third = Session()
    third.open_case(fjord)
    third.set_visible(maps(third)[0].id, False)  # hidden: opens hidden
    fourth = Session()
    fourth.open_case(fjord)
    assert not maps(fourth)[0].visible
    fourth.remove_item(maps(fourth)[0].id)  # taken away: opens without
    assert maps(Session_open(fjord)) == []
    fourth.undo()  # back: kept again
    assert len(maps(Session_open(fjord))) == 1
    # no map kept, and the case's bed not from geo.dat: the georeference may be
    # another case's, left in the folder, so no map
    (fjord / MAP_FILE).unlink()
    assert maps(Session_open(fjord)) == []
    # the case reads its geo.dat (G 10 1): the place's best, Kartverket's in Norway
    control = fjord / "control.txt"
    control.write_text(control.read_text() + "\nG 10 1\n")
    assert maps(Session_open(fjord))[0].properties["source"] == "Kartverket topo"


def test_another_cases_geo_dat_and_place_are_left_out(fjord):
    """geo.dat and georef.json left in the folder by another case: a case whose
    control.txt does not read geo.dat (no G 10 1) opens without geo.dat's points or
    a map; with G 10 1 it has both."""
    from lagoonviz.core.data.setup import reads_geo_dat

    session = Session()
    session.open_case(fjord)
    session.set_georef("EPSG:25833", origin=(270000.0, 7040000.0))
    (fjord / "geo.dat").write_text("0 0 -5\n10 0 -5\n0 10 -4\n10 10 -3\n")
    assert not reads_geo_dat(fjord)
    again = Session_open(fjord)
    kinds = {i.kind for i in again.scene}
    assert "geodat" not in kinds and "basemap" not in kinds
    case_id = next(i.id for i in again.scene if i.kind == "case")
    assert not [a for a in again._missing_setup_items(case_id) if a.kind == "geodat"]
    # turned on while editing: geo.dat's points are missing then
    again.set_input("control.txt", "G 10", [1], case=case_id)
    assert [a for a in again._missing_setup_items(case_id) if a.kind == "geodat"]
    control = fjord / "control.txt"
    control.write_text(control.read_text() + "\nG 10 1\n")
    kinds = {i.kind for i in Session_open(fjord).scene}
    assert "geodat" in kinds and "basemap" in kinds


def Session_open(folder):  # noqa: N802 (reads as a constructor)
    session = Session()
    session.open_case(folder)
    return session


def test_the_map_lies_under_the_results_too(fjord):
    session = Session()
    session.open_case(fjord)
    item = session.add_background_map("OpenStreetMap")
    assert session.mode == "setup" and session.scene.get(item).visible
    session.set_mode("results")
    assert session.scene.get(item).visible and session.data.drawn(item)


def test_making_geo_dat_tells_its_progress_and_can_be_cancelled(fjord, terrain):
    from lagoonviz.core.geo.bathy import Cancelled

    session = Session()
    session.open_case(fjord)
    session.set_georef("EPSG:25833", origin=(270000.0, 7040000.0), rotation_deg=30.0)
    session.set_domain(100.0, 900.0, -300.0, 400.0)
    heard = []
    session.make_geo_dat(terrain, spacing=2.0, progress=lambda f, t: heard.append((f, t)))
    fractions = [f for f, _ in heard]
    assert fractions == sorted(fractions) and fractions[0] == 0.0 and fractions[-1] == 1.0
    assert any("Writing geo.dat" in t for _, t in heard)
    before = (fjord / "geo.dat").read_text()

    def stop(fraction, text):
        if "Writing" in text and fraction > 0.85:
            raise Cancelled()

    job = session.geo_dat_job(terrain, spacing=1.0)
    with pytest.raises(Cancelled):
        job.run(stop)
    assert (fjord / "geo.dat").read_text() == before  # the one there was
    assert not (fjord / "geo.dat.part").exists()


class FakeWorld:
    """Copernicus tiles (land east of 18.40 E rising 1 m per 0.001°, sea 0 west of it)
    and ETOPO's CSV (the sea 20 m deep everywhere)."""

    def __init__(self, tmp_path):
        self.tmp = tmp_path
        self.asked = []

    def __call__(self, url):
        import re

        self.asked.append(url)
        if "copernicus" in url:
            name = url.rsplit("/", 1)[-1]
            lat = int(name[23:25]) * (-1 if name[22] == "S" else 1)
            lon = int(name[30:33]) * (-1 if name[29] == "W" else 1)
            if lon != 18:
                raise OSError("HTTP Error 404: Not Found")
            n = 360
            cols, rows = np.meshgrid(np.arange(n) + 0.5, np.arange(n) + 0.5)
            lons, lats = lon + cols / n, lat + 1 - rows / n
            data = np.where(lons > 18.40, (lons - 18.40) * 1000.0, 0.0)
            path = self.tmp / "cop.tif"
            write_geotiff(path, data, lon, lat + 1, 1.0 / n, 4326)
            return path.read_bytes()
        assert "erddap" in url
        from urllib.parse import unquote

        q = unquote(url)
        variable = "elevation" if "GEBCO" in q else "z"
        depth = -30.0 if "GEBCO" in q else -20.0
        lat0, stride, lat1 = re.search(variable + r"\[\(([^)]+)\):(\d+):\(([^)]+)\)\]", q).groups()
        lon0, _, lon1 = re.search(r"\]\[\(([^)]+)\):(\d+):\(([^)]+)\)\]", q).groups()
        step = 1 / 240 * int(stride)  # ERDDAP's answer includes both ends
        lats = np.arange(float(lat0), float(lat1) + step / 2, step)
        lons = np.arange(float(lon0), float(lon1) + step / 2, step)
        rows = [f"latitude,longitude,{variable}", "degrees_north,degrees_east,m"]
        rows += [f"{a},{o},{depth}" for a in lats for o in lons]
        return ("\n".join(rows) + "\n").encode()


def test_geo_dat_from_the_world_outside_europe(tmp_path, monkeypatch):
    from lagoonviz.core.geo import download

    monkeypatch.setenv("LAGOON_CACHE", str(tmp_path / "cache"))
    server = FakeWorld(tmp_path)
    monkeypatch.setattr(download, "http_fetch", server)
    session = Session()
    case = session.new_case_from_map((-33.9, 18.40), length=2000.0)
    count = session.make_geo_dat("best", spacing=50.0)
    assert session.geo_sources == [
        "Copernicus DEM GLO-30 (land, world)",
        "NOAA ETOPO 2022 (land and sea, world)",
    ]
    points = np.loadtxt(session.data.case_folder(case) / "geo.dat")
    assert count == len(points)
    georef = session.georef(case)
    lon, lat = transformer(georef.crs, "EPSG:4326")(*georef.to_world(points[:, 0], points[:, 1]))
    land = lon > 18.405
    sea = lon < 18.391  # beyond 1.5 of ETOPO's 450 m cells from the coast: trusted
    shore = (lon > 18.393) & (lon < 18.398)  # nearer: a beach profile up to the coast
    assert np.allclose(points[land, 2], (lon[land] - 18.40) * 1000.0, atol=0.5)
    assert sea.any() and np.allclose(points[sea, 2], -20.0)
    assert ((points[shore, 2] > -20.01) & (points[shore, 2] < -0.1)).all()
    assert np.ptp(points[shore, 2]) > 5.0  # no flat 1 m, nor ETOPO's 20 m to the coast
    assert sum("copernicus" in u for u in server.asked) == 1  # the one tile there


def test_the_map_tells_how_many_tiles_it_has(tmp_path):
    heard = []
    box = expanded((0.0, 1000.0, 0.0, 500.0), 0.25)
    made = background(
        GEOREF,
        box,
        SOURCE,
        512,
        TileCache(tmp_path, FakeServer()),
        progress=lambda d, t: heard.append((d, t)),
    )
    total = heard[-1][1]
    assert heard[0] == (0, total) and heard[-1] == (total, total) and made.missing == 0
    assert [d for d, _ in heard] == list(range(total + 1))


def test_geo_dat_from_gebco(tmp_path, monkeypatch):
    from lagoonviz.core.geo import download

    monkeypatch.setenv("LAGOON_CACHE", str(tmp_path / "cache"))
    server = FakeWorld(tmp_path)
    monkeypatch.setattr(download, "http_fetch", server)
    session = Session()
    case = session.new_case_from_map((-33.9, 18.30), length=1000.0)
    session.make_geo_dat("gebco", spacing=100.0)
    assert session.geo_sources == ["GEBCO 2020 (land and sea, world)"]
    assert "GEBCO_2020.csv" in server.asked[-1] and "elevation" in server.asked[-1]
    points = np.loadtxt(session.data.case_folder(case) / "geo.dat")
    assert np.allclose(points[:, 2], -30.0)


def test_the_location_of_a_case_and_a_search_near_it(monkeypatch):
    from lagoonviz.core.geo import places

    session = Session()
    case = session.new_case_from_map((58.46, 8.77), length=2000.0)
    assert session.location(case) == pytest.approx((58.46, 8.77), abs=1e-6)
    monkeypatch.setattr(places, "_http", _names_only_in_norway)
    monkeypatch.setattr(places.search, "__defaults__", (_names_only_in_norway, None))
    place = session.place_case("Nidelva", case=case)  # Arendal's: the case is there
    assert place.where == "Arendal"


def test_geo_dat_at_the_finest_the_data_has(fjord, terrain, tmp_path, monkeypatch):
    """spacing="finest": Kartverket's 1 m in Norway, Copernicus' 30 m elsewhere on
    land, a file's own cells or point spacing; never more than MOST_POINTS."""
    from lagoonviz.core.geo import bathy, download

    session = Session()
    case = session.open_case(fjord)
    session.set_georef("EPSG:25833", origin=(270000.0, 7040000.0))
    domain = session.data.backgrounds.domain(case)[:4]
    georef = session.georef(case)
    assert bathy.finest_web("best", georef, domain) == (1.0, "Kartverket, land and sea (Norway)")
    assert bathy.finest_web("emodnet", georef, domain)[0] == 115.0
    cape = Georef("EPSG:32734", (260000.0, 6245000.0))  # Cape Town: no Kartverket
    assert bathy.finest_web("best", cape, domain)[0] == 30.0
    # a GeoTIFF of 10 m cells
    session.make_geo_dat(terrain, spacing="finest")
    points = np.loadtxt(fjord / "geo.dat")
    assert np.diff(np.unique(points[:, 0]))[0] == pytest.approx(10.0)
    assert "spacing='finest'" in session.to_script()
    # soundings 7 m apart, in longitude and latitude
    xs, ys = np.meshgrid(np.arange(-80, 81) * 7.0, np.arange(-50, 51) * 7.0)
    world = GEOREF.to_world(500.0 + xs.ravel(), 250.0 + ys.ravel())
    lons, lats = transformer("EPSG:25833", "EPSG:4326")(*world)
    soundings = tmp_path / "soundings.xyz"
    np.savetxt(soundings, np.column_stack([lons, lats, plane(*world)]), fmt="%.8f")
    assert bathy.finest_file(bathy.load(soundings, "EPSG:4326")) == pytest.approx(7.0, rel=0.02)
    # the web at its finest: held to MOST_POINTS for a large domain
    monkeypatch.setattr(bathy, "MOST_POINTS", 10_000)
    monkeypatch.setenv("LAGOON_CACHE", str(tmp_path / "cache"))
    monkeypatch.setattr(download, "http_fetch", FakeWcs(tmp_path))
    job = session.geo_dat_job("best", spacing="finest")
    assert job.finest == 1.0 and len(job.xs) * len(job.ys) <= 10_000
    assert job.xs[1] - job.xs[0] > 1.0


def test_geo_dat_with_the_most_points_asked_for(fjord, terrain):
    session = Session()
    session.open_case(fjord)
    session.set_georef("EPSG:25833", origin=(270000.0, 7040000.0))
    few = session.geo_dat_job(terrain, spacing=1.0, most_points=5_000)
    assert len(few.xs) * len(few.ys) <= 5_000 and few.xs[1] - few.xs[0] > 1.0
    unlimited = session.geo_dat_job(terrain, spacing=1.0, most_points=0)
    assert unlimited.xs[1] - unlimited.xs[0] == pytest.approx(1.0)  # 0: no limit
    session.make_geo_dat(terrain, spacing=1.0, most_points=5_000)
    assert "most_points=5000" in session.to_script()
    assert session.geo_dat_recipe()["most_points"] == 5_000
