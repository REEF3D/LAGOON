# LAGOON — the free surface of a σ-grid run that writes only its volume
# Architect: Hans Bihs
"""FNPF or NHFLOW with ``P 10`` and no ``P 180`` write their volume (and bed) but no
free surface: LAGOON shows the top of the volume as the free surface, and in a flume
(one cell across) the volume itself, since from above a flume is a thin line."""

from __future__ import annotations

import numpy as np
from lagoon_format.store import StoreWriter

from lagoonviz.core.data.lagoon import VolumeTopDataset
from lagoonviz.core.data.reef3d import _flume, open_case

LEVEL = 600.0  # the still water level (F 60)


def _case(tmp_path, ny: int, length: float = 80.0, width: float = 40.0):
    """An FNPF case whose store holds a σ-grid volume over a flat bed at z = 0 and
    the bed (with its still water depth), no free surface."""
    (tmp_path / "ctrl.txt").write_text("A 10 3\nP 10 1\n")
    x = np.linspace(0.0, length, 9)
    y = np.linspace(0.0, width, ny)
    sigma = np.array([0.0, 0.5, 1.0])
    blocks = [((0, 0), (len(x), len(y)))]
    writer = StoreWriter.create(tmp_path / "REEF3D_FNPF.lagoon", solver="FNPF", run={"run": "R1"})
    volume = writer.add_output(
        "volume", "sigma", x, y, {"velocity": 3}, blocks, sigma=sigma, shard_time=2, tile=4
    )
    bed = writer.add_output("bed", "surface", x, y, {"depth": 1}, blocks, shard_time=2, tile=4)
    floor = np.zeros((len(y), len(x)))
    bed.append(0.0, 0, [{"z": floor, "depth": floor + LEVEL}])
    for t in range(3):
        top = LEVEL + np.sin(x[None, :] / 10 + t) + 0 * y[:, None]
        u = np.zeros((len(sigma), len(y), len(x), 3))
        u[..., 0] = t + 1.0
        volume.append(1.0 * t, t, [{"z_bed": floor, "z_surface": top, "velocity": u}])
    writer.close()
    return top


def test_the_top_of_the_volume_is_the_free_surface(tmp_path):
    top = _case(tmp_path, ny=6)
    case = open_case(tmp_path)
    tops = [d for d in case.datasets.values() if isinstance(d, VolumeTopDataset)]
    assert len(tops) == 1
    surface = tops[0]
    assert surface.role == "free_surface" and surface.derived and len(surface) == 3
    assert surface.still_level() == LEVEL  # the bed's height plus its still water depth
    mesh = surface.read(2)
    np.testing.assert_allclose(mesh.points[:, 2], top.ravel(), atol=1e-4)
    np.testing.assert_allclose(mesh.point_data["eta"], top.ravel() - LEVEL, atol=1e-4)
    np.testing.assert_allclose(mesh.point_data["depth"], LEVEL)
    np.testing.assert_allclose(mesh.point_data["velocity"][:, 0], 3.0)
    assert len(mesh.triangles) == 2 * 8 * 5
    # opened: the top and the bed, not the volume
    main = case.main_datasets()
    assert main["free_surface"] is surface
    assert main["bed"].role == "bed" and "volume" not in main


def test_a_flume_shows_its_volume(tmp_path):
    _case(tmp_path, ny=2, length=8640.0, width=1.5)
    case = open_case(tmp_path)
    main = case.main_datasets()
    assert main["volume"].role == "volume" and _flume(main["volume"])
    assert "free_surface" not in main
    assert main["bed"].role == "bed" and not getattr(main["bed"], "derived", False)
    # the top is there to be shown
    assert any(isinstance(d, VolumeTopDataset) for d in case.datasets.values())


def test_a_narrow_wide_grid_is_a_flume_too(tmp_path):
    _case(tmp_path, ny=4, length=1000.0, width=3.0)
    case = open_case(tmp_path)
    assert "volume" in case.main_datasets()


def test_no_top_when_the_run_writes_its_free_surface(tmp_path):
    _case(tmp_path, ny=6)
    from lagoon_format.store import StoreWriter as Writer

    writer = Writer.open(tmp_path / "REEF3D_FNPF.lagoon")
    x = np.linspace(0.0, 80.0, 9)
    y = np.linspace(0.0, 40.0, 6)
    fs = writer.add_output(
        "free_surface", "surface", x, y, {"eta": 1}, [((0, 0), (9, 6))], shard_time=2, tile=4
    )
    fs.append(0.0, 0, [{"z": np.full((6, 9), LEVEL), "eta": np.zeros((6, 9))}])
    writer.close()
    case = open_case(tmp_path)
    assert not any(isinstance(d, VolumeTopDataset) for d in case.datasets.values())


def test_the_session_shows_and_draws_them(tmp_path):
    from lagoonviz.core import Session

    _case(tmp_path, ny=6)
    s = Session()
    s.open_case(tmp_path)
    kinds = {i.kind: i for i in s.scene}
    free = kinds["free_surface"]
    assert free.visible and not kinds["volume"].visible
    mesh = s.data.load(free.id)
    assert mesh is not None and mesh.n_points == 6 * 9
    assert abs(float(np.median(mesh.points[:, 2])) - LEVEL) < 2
    flume = tmp_path / "flume"
    flume.mkdir()
    _case(flume, ny=2, length=8640.0, width=1.5)
    s = Session()
    s.open_case(flume)
    kinds = {i.kind: i for i in s.scene}
    assert kinds["volume"].visible and "free_surface" not in kinds
    mesh = s.data.load(kinds["volume"].id)
    assert mesh is not None and mesh.n_points
    assert float(mesh.points[:, 2].max()) > LEVEL - 2  # up to the free surface
