"""LAGOON stores: the Zarr v3 layer and the store reader/writer (SPEC.md)."""

import json

import numpy as np
import pytest
from lagoon_format.store import Store, StoreWriter, is_store, sigma_heights, tile_size
from lagoon_format.zarr3 import Array, Group, codecs_for, crc32c


# ----------------------------------------------------------------- zarr layer
def test_plain_array_round_trip(tmp_path):
    data = np.arange(7 * 5, dtype=np.float32).reshape(7, 5)
    a = Array.create(tmp_path / "a", data.shape, "float32", [3, 2], codecs=codecs_for("gzip"))
    a[:] = data
    np.testing.assert_array_equal(Array(tmp_path / "a")[:], data)
    np.testing.assert_array_equal(a[2:6, 1], data[2:6, 1])
    a[1, 1:4] = -1
    data[1, 1:4] = -1
    np.testing.assert_array_equal(Array(tmp_path / "a")[...], data)
    meta = json.loads((tmp_path / "a" / "zarr.json").read_text())
    assert meta["codecs"][1] == {"name": "numcodecs.shuffle", "configuration": {"elementsize": 4}}
    assert (tmp_path / "a" / "c" / "0" / "0").is_file()  # default key encoding


def test_sharded_array_reads_only_what_it_needs(tmp_path):
    rng = np.random.default_rng(1)
    data = rng.normal(size=(5, 3, 10, 13)).astype(np.float32)
    a = Array.create(
        tmp_path / "s",
        data.shape,
        "float32",
        [4, 3, 10, 14],
        inner=[1, 3, 5, 7],
        codecs=codecs_for("gzip"),
        dimension_names=["time", "level", "y", "x"],
    )
    a[:] = data
    b = Array(tmp_path / "s")
    np.testing.assert_array_equal(b[:], data)
    np.testing.assert_array_equal(b[:, 1, 7, 12], data[:, 1, 7, 12])
    np.testing.assert_array_equal(b[3], data[3])
    # one file per shard: two along time, one over the block
    files = sorted(
        p.relative_to(tmp_path / "s").as_posix() for p in (tmp_path / "s").rglob("c/*/*/*/*")
    )
    assert files == ["c/0/0/0/0", "c/1/0/0/0"]
    # an output written into a partial shard later keeps the earlier ones
    a.resize([6, 3, 10, 13])
    extra = np.full((3, 10, 13), 7.0, np.float32)
    a[5] = extra
    c = Array(tmp_path / "s")
    np.testing.assert_array_equal(c[:5], data)
    np.testing.assert_array_equal(c[5], extra)


def test_unwritten_chunks_read_as_fill(tmp_path):
    a = Array.create(tmp_path / "f", [4, 4], "float32", [4, 4], inner=[2, 2])
    a[0:2, 0:2] = 1.0
    values = Array(tmp_path / "f")[:]
    assert np.all(values[:2, :2] == 1) and np.isnan(values[2:]).all()
    # all-NaN tiles are left out of the shard: one inner chunk + the 4-entry index
    size = (tmp_path / "f" / "c" / "0" / "0").stat().st_size
    assert size == 16 + 4 * 16


def test_crc32c_known_value():
    assert crc32c(b"123456789") == 0xE3069283


def test_groups(tmp_path):
    g = Group.create(tmp_path / "g", {"answer": 42})
    assert Group(tmp_path / "g").attrs == {"answer": 42}
    Array.create(tmp_path / "g" / "x", [3], "float64")
    assert g.members() == ["x"] and "x" in g


# ----------------------------------------------------------------- stores
def _sigma_run(tmp_path, outputs=5, shard_time=2):
    """A σ-grid of 9 × 6 columns and 4 levels in two overlapping blocks."""
    x = np.linspace(0, 8, 9)
    y = np.linspace(0, 5, 6)
    sigma = np.array([0.0, 0.3, 0.7, 1.0])
    blocks = [((0, 0), (5, 6)), ((4, 0), (5, 6))]  # x index 4 in both
    writer = StoreWriter.create(tmp_path / "run.lagoon", solver="NHFLOW", run={"run": "R1"})
    out = writer.add_output(
        "volume",
        "sigma",
        x,
        y,
        {"velocity": 3, "pressure": 1},
        blocks,
        sigma=sigma,
        shard_time=shard_time,
        tile=4,
    )
    truth = []
    for t in range(outputs):
        bed = -2.0 - 0.1 * x[None, :] + 0 * y[:, None]
        top = 0.1 * np.sin(x[None, :] + t) + 0 * y[:, None]
        z = sigma_heights(bed, top, sigma)
        u = np.stack([z * 0 + t, z * 0 + x[None, None, :], z], axis=-1)
        p = -1000 * 9.81 * (z - top[None])
        truth.append((bed, top, u, p))
        parts = []
        for (i0, _j0), (nx_b, _ny_b) in blocks:
            window = slice(i0, i0 + nx_b)
            parts.append(
                {
                    "z_bed": bed[:, window],
                    "z_surface": top[:, window],
                    "velocity": u[:, :, window],
                    "pressure": p[:, :, window],
                }
            )
        out.append(0.5 * t, 10 * t, parts)
    writer.close()
    return tmp_path / "run.lagoon", truth


def test_sigma_store_round_trip(tmp_path):
    path, truth = _sigma_run(tmp_path)
    assert is_store(path)
    store = Store(path)
    assert store.solver == "NHFLOW" and store.run_id == "R1"
    volume = store["volume"]
    np.testing.assert_allclose(volume.times, [0, 0.5, 1, 1.5, 2])
    assert list(volume.steps) == [0, 10, 20, 30, 40]
    assert volume.shape == (4, 6, 9)
    bed, top, u, p = truth[3]
    np.testing.assert_allclose(volume.read("velocity", 3), u, atol=1e-6)
    np.testing.assert_allclose(volume.read("pressure", 3), p, rtol=1e-6)
    np.testing.assert_allclose(volume.heights(3), sigma_heights(bed, top, volume.sigma), atol=1e-6)
    eta = volume.series("z_surface", i=6, j=2)
    np.testing.assert_allclose(eta, [t[1][2, 6] for t in truth], atol=1e-6)
    u_top = volume.series("velocity", i=2, j=1)
    np.testing.assert_allclose(u_top[:, 0], np.arange(5))
    np.testing.assert_allclose(
        volume.column_heights(6, 2)[3], volume.heights(3)[:, 2, 6], atol=1e-5
    )


def test_appending_to_a_store(tmp_path):
    """Converting again appends: a partial shard is completed, not replaced."""
    path, _ = _sigma_run(tmp_path, outputs=3, shard_time=2)
    writer = StoreWriter.open(path)
    out = writer.output("volume")
    assert out.count == 3
    blocks = Store(path)["volume"].blocks
    parts = []
    for block in blocks:
        nx_b, ny_b = block.size
        parts.append(
            {
                "z_bed": np.full((ny_b, nx_b), -3.0),
                "z_surface": np.zeros((ny_b, nx_b)),
                "velocity": np.ones((4, ny_b, nx_b, 3)),
                "pressure": np.zeros((4, ny_b, nx_b)),
            }
        )
    out.append(9.0, 99, parts)
    writer.close()
    volume = Store(path)["volume"]
    assert list(volume.steps) == [0, 10, 20, 99]
    assert volume.read("velocity", 2)[0, 0, 0, 0] == 2  # output 2 kept
    assert np.all(volume.read("velocity", 3) == 1)


def test_surface_store(tmp_path):
    x, y = np.arange(4.0), np.arange(3.0)
    writer = StoreWriter.create(tmp_path / "s.lagoon", solver="FNPF")
    out = writer.add_output(
        "free_surface",
        "surface",
        x,
        y,
        {"eta": 1},
        [((0, 0), (4, 3))],
        source="REEF3D_FNPF_VTP_FSF",
    )
    eta = np.arange(12, dtype=np.float32).reshape(3, 4) / 10
    out.append(1.0, 1, [{"z": eta + 5, "eta": eta}])
    writer.close()
    surface = Store(tmp_path / "s.lagoon")["free_surface"]
    assert surface.grid == "surface" and surface.shape == (3, 4)
    np.testing.assert_allclose(surface.read("eta", 0), eta)
    np.testing.assert_allclose(surface.heights(0), eta + 5)
    assert surface.units["eta"] == "m"


def test_store_metadata_is_plain_zarr(tmp_path):
    path, _ = _sigma_run(tmp_path)
    root = json.loads((path / "zarr.json").read_text())
    assert root["attributes"]["lagoon"]["format"] == "lagoon"
    assert root["attributes"]["lagoon"]["outputs"] == ["volume"]
    meta = json.loads((path / "volume" / "blocks" / "b0000" / "velocity" / "zarr.json").read_text())
    assert meta["dimension_names"] == ["time", "level", "y", "x", "component"]
    assert meta["shape"] == [5, 4, 6, 5, 3]
    assert meta["codecs"][0]["name"] == "sharding_indexed"


def test_tile_size():
    from lagoon_format.store import chunk_tiles

    assert chunk_tiles(151, 176, 11) == (51, 59, 1) and chunk_tiles(2, 376, 31) == (2, 376, 3)
    assert tile_size(151) == 51 and tile_size(176) == 59 and tile_size(30) == 30


def test_not_a_store(tmp_path):
    Group.create(tmp_path / "g")
    assert not is_store(tmp_path / "g") and not is_store(tmp_path / "nothing")
    with pytest.raises(ValueError):
        Store(tmp_path / "g")


def test_stores_open_in_zarr_python(tmp_path):
    """The store is plain Zarr v3: zarr-python (with numcodecs' shuffle) reads it."""
    zarr = pytest.importorskip("zarr")
    path, truth = _sigma_run(tmp_path)
    root = zarr.open_group(str(path), mode="r")
    assert root.attrs["lagoon"]["format"] == "lagoon"
    velocity = root["volume/blocks/b0000/velocity"]
    assert velocity.shape == (5, 4, 6, 5, 3)
    np.testing.assert_allclose(velocity[3], truth[3][2][:, :, :5], atol=1e-6)
    np.testing.assert_allclose(root["volume/time"][:], [0, 0.5, 1, 1.5, 2])
    # and what zarr-python writes with these codecs reads here
    from numcodecs.zarr3 import Shuffle

    other = zarr.create_array(
        str(tmp_path / "z.zarr"),
        shape=(4, 6),
        chunks=(2, 3),
        shards=(4, 6),
        dtype="float32",
        filters=None,
        serializer="auto",
        compressors=[Shuffle(elementsize=4), zarr.codecs.GzipCodec(level=1)],
        fill_value=np.nan,
    )
    values = np.arange(24, dtype=np.float32).reshape(4, 6)
    other[:] = values
    np.testing.assert_array_equal(Array(tmp_path / "z.zarr")[:], values)


def test_shard_index_checksum(tmp_path):
    """A shard whose index does not match its CRC-32C (REEF3D is appending to it)
    reads as not written yet, instead of as garbage."""
    a = Array.create(tmp_path / "c", [2, 4], "float32", [2, 4], inner=[1, 4], index_crc=True)
    a[:] = np.arange(8, dtype=np.float32).reshape(2, 4)
    np.testing.assert_array_equal(Array(tmp_path / "c")[1], [4, 5, 6, 7])
    shard = tmp_path / "c" / "c" / "0" / "0"
    data = bytearray(shard.read_bytes())
    data[-1] ^= 0xFF
    shard.write_bytes(bytes(data))
    assert np.isnan(Array(tmp_path / "c")[:]).all()


def test_paraview_reader_grids(tmp_path):
    """What the ParaView plugin hands to VTK: structured grids, x fastest."""
    import importlib.util
    from pathlib import Path

    plugin = Path(__file__).parents[2] / "lagoon-format" / "paraview" / "lagoon_reader.py"
    spec = importlib.util.spec_from_file_location("lagoon_reader", plugin)
    reader = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(reader)
    path, truth = _sigma_run(tmp_path)
    store = Store(path)
    assert reader.all_times(store) == [0, 0.5, 1, 1.5, 2]
    volume = store["volume"]
    assert reader.frame_index(volume.times, 1.2) == 2
    dims, points, arrays = reader.structured_block(volume, volume.blocks[1], 3)
    assert dims == (5, 6, 4) and points.shape == (120, 3)
    np.testing.assert_allclose(points[:5, 0], volume.x[4:9])  # x fastest
    np.testing.assert_allclose(
        arrays["velocity"].reshape(4, 6, 5, 3), truth[3][2][:, :, 4:9], atol=1e-6
    )
