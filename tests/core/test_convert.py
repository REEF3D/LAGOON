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


def test_time_series_and_wave_profiles_in_the_store(tmp_path):
    """Format 0.2: gauge, 6DOF tables and wave profiles go into the store; the case
    reads them from there while the text files are unchanged."""
    import os

    from lagoonviz.core.data.lagoon import StoreProfileSet, StoreSeriesFile
    from lagoonviz.core.data.series import read_series

    run = make_run(tmp_path / "run", frames=3, series=True, profiles=True, objects=True)
    report = convert(run)
    assert report.outputs["series"] == 4 and report.outputs["profiles"] == 6
    store = Store(report.store)
    assert store.info["version"] == "0.3"
    gauges = store.tables["nhflow_wsf_hg"]
    original = read_series(run / "REEF3D_NHFLOW_WSF" / "REEF3D-NHFLOW-WSF-HG.dat")
    np.testing.assert_array_equal(gauges.time, original.time)
    np.testing.assert_array_equal(gauges.column("P2"), original.columns["P2"])
    assert gauges.locations["P1"] == original.locations["P1"] and gauges.kind == "wave gauges"
    # the case reads the tables and profiles from the store
    case = open_case(run)
    series = case.series["nhflow_wsf_hg"]
    assert isinstance(series, StoreSeriesFile) and series.name == "Wave gauges"
    np.testing.assert_array_equal(series.read().columns["P3"], original.columns["P3"])
    profiles = case.profiles["nhflow_wsfline"]
    assert isinstance(profiles, StoreProfileSet) and len(profiles.times) == 6
    vtk = open_case(run, stores=False).profiles["nhflow_wsfline"]
    a, b = vtk.at(1.5), profiles.at(1.5)
    assert a.time == b.time and set(a.lines) == set(b.lines)
    np.testing.assert_allclose(b.lines[1].eta, a.lines[1].eta, atol=1e-6)
    np.testing.assert_allclose(b.lines[2].x, a.lines[2].x)
    # a file that grew (a running simulation) is read from the file again
    path = run / "REEF3D_NHFLOW_WSF" / "REEF3D-NHFLOW-WSF-HG.dat"
    path.write_text(path.read_text() + "\n")
    os.utime(path, (1e9, 1e9))
    assert not isinstance(open_case(run).series["nhflow_wsf_hg"], StoreSeriesFile)
    assert convert(run).outputs["series"] == 1  # converted again


def test_a_case_of_only_time_series_converts(tmp_path):
    from reef3d_fixtures import write_series

    root = tmp_path / "gauges"
    root.mkdir()
    write_series(root, "FNPF", 5.0)
    report = convert(root)
    assert set(report.outputs) == {"series"} and report.store.name == "REEF3D_FNPF.lagoon"
    assert "fnpf_wsf_hg" in Store(report.store).tables


def _tumbling_body(root, frames, number=1, rigid=True):
    """A second body that turns about z and x as it drifts (or, not rigid, swells)."""
    from reef3d_fixtures import BOX, BOX_FACES, write_piece

    folder = root / "REEF3D_NHFLOW_6DOF_VTP"
    tris = np.concatenate([BOX_FACES[:, [0, 1, 2]], BOX_FACES[:, [0, 2, 3]]])
    shape = (BOX - 0.5) * np.array([2.0, 1.0, 0.5], np.float32)
    for frame in range(frames):
        a, b = 0.3 * frame, 0.1 * np.sin(frame)
        rz = np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]])
        rx = np.array([[1, 0, 0], [0, np.cos(b), -np.sin(b)], [0, np.sin(b), np.cos(b)]])
        scale = 1.0 if rigid else 1.0 + 0.1 * frame
        points = (shape * scale) @ (rz @ rx).T + [6.0 + 0.1 * frame, 2.0, 1.4]
        flat = points.astype(np.float32)[tris.ravel()]
        write_piece(
            folder / f"REEF3D-6DOF-{number}-{frame:06d}.vtp",
            "PolyData",
            flat,
            {},
            np.arange(flat.shape[0]),
            np.arange(3, flat.shape[0] + 1, 3),
        )


def test_bodies_are_stored_as_mesh_and_motion(tmp_path):
    from lagoon_format.bodies import rigid_tolerance

    from lagoonviz.core.data.lagoon import StoreBodyDataset

    run = make_run(tmp_path / "run", frames=5, print_interval=0.5, objects=True)
    _tumbling_body(run, 5)
    vtk = open_case(run, stores=False).datasets["nhflow_body"]
    report = convert(run, shard_time=2, limit=3)
    assert report.outputs["bodies"] == 3
    assert convert(run).outputs["bodies"] == 2  # appended
    bodies = Store(report.store).bodies["nhflow_body"]
    assert sorted(bodies.bodies) == [0, 1] and all(b.rigid for b in bodies.bodies.values())
    np.testing.assert_allclose(bodies.times, [0, 0.5, 1.0, 1.5, 2.0])
    case = open_case(run)
    stored = case.datasets["nhflow_body"]
    assert isinstance(stored, StoreBodyDataset) and "(LAGOON)" in stored.label
    assert stored.steps == vtk.steps
    for index in range(5):
        a, b = vtk.read(index), stored.read(index)
        assert np.array_equal(a.triangles, b.triangles)
        assert np.abs(a.points - b.points).max() <= rigid_tolerance(a.points)
        np.testing.assert_allclose(a.normals, b.normals, atol=1e-4)


def test_a_body_that_is_not_rigid_keeps_its_vertices(tmp_path):
    run = make_run(tmp_path / "run", frames=4, print_interval=0.5, objects=True)
    _tumbling_body(run, 4, rigid=False)
    report = convert(run)
    bodies = Store(report.store).bodies["nhflow_body"]
    assert bodies.bodies[0].rigid and not bodies.bodies[1].rigid
    vtk = open_case(run, stores=False).datasets["nhflow_body"]
    stored = open_case(run).datasets["nhflow_body"]
    assert np.abs(vtk.read(3).points - stored.read(3).points).max() < 1e-5


def _same_meshes(vtk, stored, indices):
    assert stored.steps == vtk.steps and stored.role == vtk.role
    if vtk.times is not None:  # else the store has P 30 times the output number
        np.testing.assert_allclose(stored.times, vtk.times)
    for index in indices:
        a, b = vtk.read(index, None), stored.read(index, None)
        assert np.array_equal(a.points, b.points)
        assert np.array_equal(a.triangles, b.triangles)
        if a.segments is None:
            assert b.segments is None
        else:
            assert np.array_equal(a.segments, b.segments)
        assert set(a.point_data) == set(b.point_data)
        for name, values in a.point_data.items():
            assert np.array_equal(values, b.point_data[name]), name


def test_particles_and_objects_in_the_store(tmp_path):
    from reef3d_fixtures import write_cpm

    from lagoonviz.core.data.lagoon import StoreParticleDataset

    run = make_run(tmp_path / "cfd", solver="CFD", frames=4, print_interval=0.5)
    write_cpm(run, end=1.5, interval=0.25)
    vtk = open_case(run, stores=False).datasets["cfd_particles"]
    report = convert(run, limit=4)
    assert report.outputs["particles"] == 4
    assert convert(run).outputs["particles"] == 3  # appended
    stored = open_case(run).datasets["cfd_particles"]
    assert isinstance(stored, StoreParticleDataset) and stored.label.endswith("(LAGOON)")
    _same_meshes(vtk, stored, range(7))
    sets = Store(report.store).particles["cfd_particles"]
    assert sets.cells is None and "fluid velocity" in sets.point_fields


def test_dem_ice_rods_and_mooring_in_the_store(tmp_path):
    run = make_run(
        tmp_path / "objects", frames=4, print_interval=0.5, objects=True, dem=0.25, rods=0.5
    )
    vtk = open_case(run, stores=False)
    report = convert(run)
    for role in ("dem", "rods", "mooring"):
        assert report.outputs[role] > 0, role
    case = open_case(run)
    for key in ("dem_dem", "nhflow_mooring"):
        _same_meshes(vtk.datasets[key], case.datasets[key], range(len(vtk.datasets[key])))
    rods = next(k for k, d in vtk.datasets.items() if d.role == "rods")
    _same_meshes(vtk.datasets[rods], case.datasets[rods], range(len(vtk.datasets[rods])))
    store = Store(report.store)
    mooring = store.particles["nhflow_mooring"]
    assert mooring.cells == "lines"
    # the lines keep their cells: one cell set for all outputs
    assert mooring.group.array("cell_sets/cell_end").shape[0] == 1


def test_ice_floes_in_the_store(tmp_path):
    run = make_run(tmp_path / "ice", solver="FNPF", frames=3, print_interval=0.5, ice=0.25)
    vtk = open_case(run, stores=False)
    report = convert(run)
    assert report.outputs["ice"] > 0
    key = next(k for k, d in vtk.datasets.items() if d.role == "ice")
    _same_meshes(vtk.datasets[key], open_case(run).datasets[key], range(len(vtk.datasets[key])))


def test_an_iso_surface_is_left_out(tmp_path):
    """CFD's free surface is an iso-surface of the level set, not a grid: the store
    gets the rest."""
    from reef3d_fixtures import write_parallel, write_piece

    run = make_run(tmp_path / "cfd", solver="CFD", frames=2, print_interval=0.5)
    folder = run / "REEF3D_CFD_VTP_FSF"
    rng = np.random.default_rng(3)
    for frame in range(2):
        names = []
        for rank in range(2):
            points = rng.uniform([0, 0, 1.9], [8, 4, 2.1], (300, 3)).astype(np.float32)
            name = f"REEF3D-CFD-FSF-{frame:08d}-{rank + 1:06d}.vtp"
            tris = np.arange(300).reshape(-1, 3)
            write_piece(
                folder / name,
                "PolyData",
                points,
                {"elevation": points[:, 2].copy()},
                tris.ravel(),
                np.arange(3, 301, 3),
            )
            names.append(name)
        write_parallel(
            folder / f"REEF3D-CFD-FSF-{frame:08d}.pvtp", "PolyData", names, {"elevation": 1}
        )
    report = convert(run)
    assert "free_surface" in report.skipped and "free_surface" not in report.outputs
    assert report.outputs["volume"] == 2 and "left out: free_surface" in report.summary()


def test_a_store_opened_on_its_own_is_its_solvers_run(tmp_path):
    """Another solver's gauges in the same folder stay out; a store REEF3D has only
    just begun (no output yet) says so instead of opening something else."""
    run = make_run(tmp_path / "run", frames=3, print_interval=0.5)
    store = convert(run).store
    make_run(tmp_path / "run", solver="FNPF", frames=2, volume=False, series=True)
    s = Session()
    s.open_case(store)
    kinds = [i.kind for i in s.scene]
    assert "free_surface" in kinds and "series" not in kinds
    empty = run / "REEF3D_CFD.lagoon"
    empty.mkdir()
    (empty / "zarr.json").write_text(
        '{"zarr_format": 3, "node_type": "group", "attributes": {"lagoon": '
        '{"format": "lagoon", "version": "0.3", "solver": "CFD", "outputs": []}}}'
    )
    with pytest.raises(ValueError, match="no outputs yet"):
        open_case(empty)
