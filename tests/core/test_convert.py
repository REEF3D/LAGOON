"""VTU/VTP to LAGOON store, and LAGOON stores opened as cases (phase 3)."""

import numpy as np
import pytest
from lagoon_format.store import Store
from reef3d_fixtures import DEPTH, eta, make_run

from lagoonviz.core import Session
from lagoonviz.core.convert import ConvertError, convert, level_offsets, main, nominal_sigma
from lagoonviz.core.data.lagoon import StoreDataset
from lagoonviz.core.data.reef3d import open_case


@pytest.fixture
def converted(tmp_path):
    run = make_run(tmp_path / "run", frames=4, print_interval=0.5)
    report = convert(run, shard_time=2)
    return run, report


def test_convert_a_case(converted):
    run, report = converted
    assert report.store == run / "REEF3D_NHFLOW.lagoon"
    assert report.outputs == {"volume": 4, "free_surface": 4, "bed": 4}
    store = Store(report.store)
    assert store.solver == "NHFLOW"
    volume = store["volume"]
    np.testing.assert_allclose(volume.times, [0, 0.5, 1.0, 1.5])  # from P 30
    assert len(volume.blocks) == 2 and volume.shape[0] == 3
    np.testing.assert_allclose(volume.sigma, [0, 0.5, 1])
    # the free surface's z is still water depth + eta
    surface = store["free_surface"]
    x, y = np.meshgrid(surface.x, surface.y)
    np.testing.assert_allclose(surface.heights(2), DEPTH + eta(x, y, 2), atol=1e-5)
    np.testing.assert_allclose(surface.read("eta", 2), eta(x, y, 2), atol=1e-6)
    # the volume's top level is the free surface
    np.testing.assert_allclose(volume.heights(2)[-1], DEPTH + eta(x, y, 2), atol=1e-5)
    assert not volume.has("z_offset")  # exact σ-levels: no corrections stored


def test_converting_again_appends(tmp_path):
    run = make_run(tmp_path / "run", frames=3, print_interval=0.5)
    first = convert(run, shard_time=2, limit=2)
    assert first.outputs["volume"] == 2
    again = convert(run, shard_time=2)
    assert again.outputs == {"volume": 1, "free_surface": 1, "bed": 1}
    assert list(Store(again.store)["volume"].steps) == [0, 1, 2]
    assert convert(run).outputs == {"volume": 0, "free_surface": 0, "bed": 0}


def test_store_of_another_run_is_not_appended_to(tmp_path, monkeypatch):
    from lagoonviz.core import convert as module

    run = make_run(tmp_path / "run", frames=2)
    monkeypatch.setattr(module, "_run_record", lambda case, d: {"type": "run", "run": "R1"})
    report = convert(run)
    assert Store(report.store).run_id == "R1"
    monkeypatch.setattr(module, "_run_record", lambda case, d: {"type": "run", "run": "R2"})
    with pytest.raises(ConvertError, match="holds run R1"):
        convert(run)


def test_a_store_opens_like_the_vtk_output(converted):
    run, report = converted
    vtk = open_case(run / "REEF3D_NHFLOW_VTU").datasets["nhflow_volume"]
    case = open_case(run)
    volume = case.datasets["nhflow_volume"]
    assert isinstance(volume, StoreDataset) and "(LAGOON)" in volume.label
    assert volume.steps == vtk.steps and volume.variables() == vtk.variables()
    np.testing.assert_allclose(volume.times, [0, 0.5, 1.0, 1.5])
    # the volume's faces: the same points (the store's faces keep their own edge points)
    a, b = vtk.read(2, ["velocity"]), volume.read(2, ["velocity"])
    same = {tuple(p) for p in np.round(a.points, 4)}
    assert same == {tuple(p) for p in np.round(b.points, 4)}
    assert set(b.point_data) == {"velocity"} and b.normals is not None
    bottom = b.normals[np.isclose(b.points[:, 2], 0.0)]
    assert bottom[:, 2].min() < -0.9 and bottom[:, 2].max() <= 1e-6  # outwards: down or aside
    # slices, iso-surfaces and streamlines go through the same code
    # a vertical slice reads only the two grid lines around it: the exact section
    section = volume.read_slice(1, 0, 2.05, ["pressure"])
    np.testing.assert_allclose(section.points[:, 0], 2.05)
    expected = 1000 * 9.81 * (DEPTH - section.points[:, 2])  # linear in z: interpolates exactly
    np.testing.assert_allclose(section.point_data["pressure"], expected, rtol=1e-4, atol=0.5)
    cut = vtk.read_slice(1, 0, 2.05, ["pressure"])
    np.testing.assert_allclose(section.bounds()[0], cut.bounds()[0], atol=1e-5)
    np.testing.assert_allclose(section.bounds()[1], cut.bounds()[1], atol=1e-5)
    lines = volume.read_streamlines(1, "velocity", 0, 1.0, 10)
    assert lines.segments is not None and len(lines.segments)
    # a probe reads one chunk per output, at the top of the column
    where, values = volume.point_series("velocity.z", 3.0, 1.0)
    x, y = where[0], where[1]
    expected = [DEPTH + eta(np.array(x), np.array(y), f) for f in range(4)]
    np.testing.assert_allclose(values, expected, atol=1e-5)  # velocity.z is z in the fixture
    assert x == pytest.approx(3.0, abs=0.06) and where[2] == pytest.approx(expected[0], abs=1e-5)


def test_a_store_alone_opens_as_a_case(converted):
    run, report = converted
    case = open_case(report.store)
    assert case.path == run
    assert {d.role for d in case.datasets.values()} == {"volume", "free_surface", "bed"}
    assert all(isinstance(d, StoreDataset) for d in case.datasets.values())


def test_newer_vtk_output_wins_over_the_store(tmp_path):
    run = make_run(tmp_path / "run", frames=3)
    convert(run, limit=2)
    case = open_case(run)
    assert not isinstance(case.datasets["nhflow_volume"], StoreDataset)


def test_session_shows_a_converted_case(converted):
    run, _ = converted
    s = Session()
    s.open_case(run)
    surface = s.data.load("free_surface-1")
    assert surface is not None and surface.points.shape[0] > 0
    probe = s.data.probe("free_surface-1", 3, (1.0, 0.0, 0.0))
    assert probe is not None


def test_sigma_and_level_offsets():
    z = np.array([[[0.0, -2.0]], [[1.0, -1.0]], [[2.0, 0.0]]])  # (3 levels, 1, 2 columns)
    sigma = nominal_sigma([z])
    np.testing.assert_allclose(sigma, [0, 0.5, 1])
    assert not level_offsets(z.astype(np.float32), sigma).any()
    bent = z.copy()
    bent[1, 0, 1] = -1.5  # a level off its σ
    offsets = level_offsets(bent.astype(np.float32), sigma)
    assert offsets[1, 0, 1] == pytest.approx(-0.5) and np.count_nonzero(offsets) == 1


def test_command_line(tmp_path, capsys):
    run = make_run(tmp_path / "run", frames=2)
    assert main([str(run), "--outputs", "free_surface"]) == 0
    assert "free_surface +2" in capsys.readouterr().out
    assert main([str(tmp_path / "missing")]) == 1


def test_not_convertible(tmp_path):
    (tmp_path / "empty").mkdir()
    with pytest.raises((ConvertError, ValueError)):
        convert(tmp_path / "empty")


def test_cfd_volume_is_a_cartesian_store(tmp_path):
    """CFD's levels are at fixed heights: the store keeps them once (z), no heights
    per output; faces, slices and probes read like the VTU output."""
    run = make_run(tmp_path / "cfd", solver="CFD", frames=3)
    report = convert(run)
    volume = Store(report.store)["volume"]
    assert volume.grid == "cartesian" and volume.nlevels == 3
    np.testing.assert_allclose(volume.z, [0.0, 1.5, 3.0])
    assert not volume.has("z_bed") and volume.heights(1).shape == (
        3,
        volume.shape[1],
        volume.shape[2],
    )
    vtk = open_case(run / "REEF3D_CFD_VTU").datasets["cfd_volume"]
    stored = open_case(run).datasets["cfd_volume"]
    assert isinstance(stored, StoreDataset)
    a, b = vtk.read(1, ["pressure"]), stored.read(1, ["pressure"])
    assert {tuple(p) for p in np.round(a.points, 4)} == {tuple(p) for p in np.round(b.points, 4)}
    section = stored.read_slice(1, 1, 2.0, ["pressure"])
    np.testing.assert_allclose(section.points[:, 1], 2.0)
    np.testing.assert_allclose(
        section.point_data["pressure"], 1000 * 9.81 * (DEPTH - section.points[:, 2]), rtol=1e-5
    )
    where, values = stored.point_series("pressure", 3.0, 1.0)
    assert where[2] == pytest.approx(3.0) and values.shape == (3,)


def _cfd_split_in_z(root, frames=2):
    """A CFD volume whose two MPI pieces split the levels (k 0-2 and 2-4)."""
    from reef3d_fixtures import write_parallel, write_piece

    folder = root / "REEF3D_CFD_VTU"
    folder.mkdir(parents=True)
    xs, ys, zs = np.arange(6.0), np.arange(4.0), np.linspace(0.0, 2.0, 5)
    for frame in range(frames):
        names = []
        for rank, (k0, k1) in enumerate(((0, 2), (2, 4))):
            z, y, x = np.meshgrid(zs[k0 : k1 + 1], ys, xs, indexing="ij")  # x fastest
            points = np.stack([x.ravel(), y.ravel(), z.ravel()], axis=1)
            nz, ny, nx = z.shape
            cells = []
            for k in range(nz - 1):
                for j in range(ny - 1):
                    for i in range(nx - 1):
                        a = (k * ny + j) * nx + i
                        up = ny * nx
                        cells.append(
                            [
                                a,
                                a + 1,
                                a + nx + 1,
                                a + nx,
                                a + up,
                                a + up + 1,
                                a + up + nx + 1,
                                a + up + nx,
                            ]
                        )
            cells = np.array(cells)
            data = {"pressure": (1000.0 * (frame + 1) * (2.0 - points[:, 2])).astype(np.float32)}
            name = f"REEF3D-CFD-{frame:08d}-{rank + 1:06d}.vtu"
            write_piece(
                folder / name,
                "UnstructuredGrid",
                points,
                data,
                cells.ravel(),
                np.arange(1, len(cells) + 1) * 8,
                np.full(len(cells), 12, np.uint8),
            )
            names.append(name)
        write_parallel(
            folder / f"REEF3D-CFD-{frame:08d}.pvtu", "UnstructuredGrid", names, {"pressure": 1}
        )
    return root


def test_cfd_pieces_split_in_z(tmp_path):
    """MPI splits CFD grids in z too: blocks with a level range (start i0, j0, k0)."""
    run = _cfd_split_in_z(tmp_path / "cfd")
    report = convert(run)
    volume = Store(report.store)["volume"]
    assert [(b.k0, b.nz) for b in volume.blocks] == [(0, 3), (2, 3)]
    np.testing.assert_allclose(volume.z, np.linspace(0, 2, 5))
    p = volume.read("pressure", 1)
    np.testing.assert_allclose(p, 2000.0 * (2.0 - volume.z)[:, None, None] + 0 * p)
    np.testing.assert_allclose(volume.series("pressure", 2, 1, 0), [2000.0, 4000.0])  # lower block
    np.testing.assert_allclose(volume.series("pressure", 2, 1), [0.0, 0.0])  # top level: upper
    stored = open_case(run).datasets["cfd_volume"]
    faces = stored.read(1, ["pressure"])
    np.testing.assert_allclose(faces.point_data["pressure"], 2000.0 * (2.0 - faces.points[:, 2]))
    section = stored.read_slice(1, 0, 2.5, ["pressure"])
    np.testing.assert_allclose(
        section.point_data["pressure"], 2000.0 * (2.0 - section.points[:, 2])
    )
    assert section.points[:, 2].max() == pytest.approx(2.0)
