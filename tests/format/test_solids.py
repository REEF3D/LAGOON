# SPDX-License-Identifier: GPL-3.0-or-later
# Architect: Hans Bihs
"""The case's solids in a store (SPEC 12): written once, read back."""

import json

import numpy as np
import pytest
from lagoon_format.store import Store, StoreError, StoreWriter


def test_solids_round_trip(tmp_path):
    path = tmp_path / "run.lagoon"
    writer = StoreWriter.create(path, "CFD")
    assert Store(path).solids() is None  # none written
    points = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], np.float64)
    triangles = np.array([[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]])
    writer.set_solids(points, triangles, "DIVEMesh_Paraview/REEF3D_Solid.vtp")
    found, faces, info = Store(path).solids()
    np.testing.assert_array_equal(found, points)
    np.testing.assert_array_equal(faces, triangles)
    assert found.dtype == np.float64 and faces.dtype == np.int32
    assert info == {"role": "solid", "source": "DIVEMesh_Paraview/REEF3D_Solid.vtp"}
    meta = json.loads((path / "solids" / "points" / "zarr.json").read_text())
    assert meta["shape"] == [4, 3] and meta["data_type"] == "float64"
    # written again: replaced
    writer.set_solids(points[:3], triangles[:1])
    assert Store(path).solids()[1].shape == (1, 3)
    with pytest.raises(StoreError):
        writer.set_solids(points[:3], triangles)  # a corner that is not there


def test_solids_as_reef3d_writes_them(tmp_path):
    """REEF3D writes the arrays plain (codec bytes, no compression), one chunk each;
    float32 points are read too."""
    path = tmp_path / "run.lagoon"
    StoreWriter.create(path, "CFD")
    points = np.arange(18, dtype="<f4").reshape(6, 3)
    triangles = np.arange(6, dtype="<i4").reshape(2, 3)
    group = path / "solids"
    group.mkdir()
    (group / "zarr.json").write_text(
        json.dumps(
            {"zarr_format": 3, "node_type": "group", "attributes": {"lagoon": {"role": "solid"}}}
        )  # fmt: skip
    )
    for name, values, dtype in (("points", points, "float32"), ("triangles", triangles, "int32")):
        (group / name / "c" / "0").mkdir(parents=True)
        (group / name / "zarr.json").write_text(
            json.dumps(
                {
                    "zarr_format": 3,
                    "node_type": "array",
                    "shape": list(values.shape),
                    "data_type": dtype,
                    "chunk_grid": {
                        "name": "regular",
                        "configuration": {"chunk_shape": list(values.shape)},
                    },
                    "chunk_key_encoding": {"name": "default", "configuration": {"separator": "/"}},
                    "fill_value": 0,
                    "codecs": [{"name": "bytes", "configuration": {"endian": "little"}}],
                    "attributes": {},
                }
            )  # fmt: skip
        )
        (group / name / "c" / "0" / "0").write_bytes(values.tobytes())
    found, faces, _ = Store(path).solids()
    np.testing.assert_array_equal(found, points)
    np.testing.assert_array_equal(faces, triangles)
