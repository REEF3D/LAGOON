"""The georeference block (SPEC 1.2): transforms, JSON, georef.json, store and run log."""

import json

import numpy as np
import pytest
from lagoon_format.geo import FILE, Georef, GeorefError, epsg_code, from_attributes, read, write
from lagoon_format.run import read_case
from lagoon_format.store import Store, StoreWriter

TRONDHEIM = Georef("EPSG:25833", (270000.0, 7040000.0), 30.0, "EPSG:5941", -1.5)


def test_model_to_world_and_back():
    g = TRONDHEIM
    big_x, big_y, big_z = g.to_world(100.0, 0.0, 0.0)
    assert big_x == pytest.approx(270000.0 + 100 * np.cos(np.radians(30)))
    assert big_y == pytest.approx(7040000.0 + 100 * np.sin(np.radians(30)))
    assert big_z == pytest.approx(-1.5)
    points = np.random.default_rng(1).uniform(-500, 500, (50, 3))
    back = g.points_to_model(g.points_to_world(points))
    assert np.allclose(back, points, atol=1e-8)
    # the model's y-axis is 90 degrees further round
    big_x, big_y = g.to_world(0.0, 10.0)
    assert (big_x - 270000.0, big_y - 7040000.0) == pytest.approx(
        (-5.0, 10 * np.cos(np.radians(30)))
    )


def test_bearing_and_turning_about_a_point():
    g = TRONDHEIM
    assert g.bearing_deg == pytest.approx(60.0)
    turned = g.rotated(75.0, about=(400.0, 150.0))
    assert turned.rotation_deg == 75.0
    assert turned.to_world(400.0, 150.0) == pytest.approx(g.to_world(400.0, 150.0))
    assert turned.crs == g.crs and turned.z_offset == g.z_offset


def test_json_round_trip_keeps_unknown_keys():
    data = TRONDHEIM.to_dict() | {"note": "Nidelva mouth"}
    back = Georef.from_json(json.dumps(data))
    assert back == TRONDHEIM and back.extra == {"note": "Nidelva mouth"}
    assert back.to_dict()["note"] == "Nidelva mouth"
    assert Georef.from_dict({"crs": "EPSG:32632"}).origin == (0.0, 0.0)


@pytest.mark.parametrize(
    "block",
    [
        {"origin": [0, 0]},
        {"crs": "", "origin": [0, 0]},
        {"crs": "EPSG:25833", "origin": [1.0]},
        {"crs": "EPSG:25833", "origin": [1.0, "x"]},
        {"crs": "EPSG:25833", "rotation_deg": float("nan")},
        {"crs": "EPSG:25833", "version": 2},
    ],
)
def test_bad_blocks_are_refused(block):
    with pytest.raises(GeorefError):
        Georef.from_dict(block)
    assert from_attributes({"georef": block}) is None


def test_epsg_codes():
    assert epsg_code("EPSG:25833") == 25833
    assert epsg_code("epsg:4326") == 4326
    assert epsg_code("urn:ogc:def:crs:EPSG::3857") == 3857
    assert epsg_code("5105") == 5105
    assert epsg_code('PROJCRS["x"]') is None


def test_georef_json_in_a_case_folder(tmp_path):
    assert read(tmp_path) is None
    path = write(tmp_path, TRONDHEIM)
    assert path == tmp_path / FILE and read(tmp_path) == TRONDHEIM
    (tmp_path / FILE).write_text("{not json")
    with pytest.raises(GeorefError):
        read(tmp_path)
    write(tmp_path, None)
    assert not (tmp_path / FILE).exists()


def test_a_store_keeps_the_georeference(tmp_path):
    path = tmp_path / "REEF3D_FNPF.lagoon"
    StoreWriter.create(path, "FNPF", georef=TRONDHEIM)
    store = Store(path)
    assert store.georef == TRONDHEIM and store.info["version"] == "0.4"
    writer = StoreWriter.open(path)
    writer.set_georef(TRONDHEIM.replace(rotation_deg=0.0))
    assert Store(path).georef.rotation_deg == 0.0
    writer.set_georef(None)
    assert Store(path).georef is None


def test_a_store_falls_back_to_its_run_record(tmp_path):
    path = tmp_path / "REEF3D_NHFLOW.lagoon"
    run = {"type": "run", "run": "20261012T100000Z-abcd", "georef": TRONDHEIM.to_dict()}
    StoreWriter.create(path, "NHFLOW", run=run)
    assert Store(path).georef == TRONDHEIM


def test_the_run_log_gives_the_georeference(tmp_path):
    folder = tmp_path / "REEF3D_Case"
    folder.mkdir()
    record = {
        "type": "run",
        "format": "reef3d-run",
        "version": 1,
        "run": "20261012T100000Z-abcd",
        "solver": "NHFLOW",
        "started": "2026-10-12T10:00:00Z",
        "streams": {},
        "georef": TRONDHEIM.to_dict(),
    }
    (folder / "REEF3D_NHFLOW_run.jsonl").write_text(json.dumps(record) + "\n")
    log = read_case(tmp_path)
    assert log.runs[0].georef == TRONDHEIM
