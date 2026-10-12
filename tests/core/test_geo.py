"""The Earth under a case: CRS transforms, map tiles, rasters, bathymetry to geo.dat,
background maps and the session's georeference."""

import io
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
        z, x, y = (int(v) for v in url.split("/")[-3:])
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
    assert SOURCES["sentinel2_2017"].licence == "CC BY 4.0"
    assert custom_source("https://x.test/{z}/{x}/{y}.png").attribution == "x.test"
    with pytest.raises(ValueError):
        custom_source("https://x.test/{z}/{x}.png")
    assert tuple(SOURCE_CHOICES) == BASEMAP_SOURCES


# ------------------------------------------------------------------ rasters
def write_geotiff(path, data, x0, y0, size, epsg=25833, point=False):
    """A north-up GeoTIFF (float32 heights or uint8 RGB) written with Pillow."""
    info = TiffImagePlugin.ImageFileDirectory_v2()
    info[33550] = (float(size), float(size), 0.0)
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
    assert "session.make_geo_dat(" in session.to_script()


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
