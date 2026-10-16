"""Statistics over time as maps: max, min, mean, time of the maximum and change per
point, the significant wave height from eta, the arrival of the water, how long it
stays and the maximum inundation; kept in the case folder, shown as a scene item."""

import numpy as np
import pytest
from reef3d_fixtures import eta, make_run

from lagoonviz.core import Session, statistics
from lagoonviz.core.data.mesh import SurfaceMesh


class Flood:
    """A stand-in output: a 1D beach of 5 points flooded from the left, one point
    more every output, and waves on it."""

    role = "free_surface"
    label = "a flood"
    components = {"eta": 1, "velocity": 3, "WL": 1}

    def __init__(self) -> None:
        self.points = np.array([[x, 0.0, 0.0] for x in range(5)], np.float32)

    def read(self, index, arrays):
        x = self.points[:, 0]
        depth = np.where(x <= index, 0.5, 0.0)
        wave = 0.2 * np.sin(1.3 * index) * (depth > 0)
        speed = np.where(depth > 0, 1.0 + index, 99.0)  # dry: nonsense, not counted
        points = self.points.copy()
        points[:, 2] = depth + wave
        data = {
            "eta": wave.astype(np.float32),
            "velocity": np.stack([speed, 0 * x, 0 * x], 1).astype(np.float32),
            "WL": depth.astype(np.float32),
        }
        return SurfaceMesh(points, np.array([[0, 1, 2], [2, 3, 4]], np.uint32), data)


def test_the_statistics_of_a_flood():
    times = np.arange(6) * 10.0  # outputs every 10 s
    mesh, facts = statistics.compute(Flood(), times, statistics.Recipe("flood"))
    d = mesh.point_data
    assert facts["outputs"] == 6 and facts["wetness"] == "WL"
    # point x is wet from output x on: arrival, how long, ever wet
    np.testing.assert_allclose(d["first_wet"], [0, 10, 20, 30, 40])
    # each output stands for the time half-way to its neighbours (5 s at the ends)
    np.testing.assert_allclose(d["wet_duration"], [50, 45, 35, 25, 15])
    np.testing.assert_allclose(d["ever_wet"], 1.0)
    np.testing.assert_allclose(d["max_depth"], 0.5)
    # the velocity only while wet: its maximum at the last output, never the dry 99
    np.testing.assert_allclose(d["velocity_max"], 6.0)
    np.testing.assert_allclose(d["velocity_time_of_max"], 50.0)
    np.testing.assert_allclose(d["velocity_min"], [1, 2, 3, 4, 5])
    # Hs: 4 sigma of eta while wet, at point 0 over all six outputs
    waves = 0.2 * np.sin(1.3 * np.arange(6))
    w = np.array([5, 10, 10, 10, 10, 5.0])
    mean = (waves * w).sum() / w.sum()
    sigma = np.sqrt(((waves - mean) ** 2 * w).sum() / w.sum())
    assert d["Hs"][0] == pytest.approx(4 * sigma, rel=1e-5)
    assert d["eta_range"][0] == pytest.approx(waves.max() - waves.min(), rel=1e-5)
    # a window: from 20 s
    mesh, facts = statistics.compute(Flood(), times, statistics.Recipe("flood", start=20.0))
    assert facts["outputs"] == 4 and mesh.point_data["first_wet"][0] == 20.0
    with pytest.raises(ValueError, match="two at least"):
        statistics.compute(Flood(), times, statistics.Recipe("flood", start=45.0))


def test_a_statistics_map_of_a_run(tmp_path):
    root = make_run(tmp_path / "run", frames=6, print_interval=0.5, volume=False)
    s = Session()
    s.open_case(root)
    item = s.add_statistics()
    shown = s.scene.get(item)
    data = s.data
    dataset = data.dataset(item)
    assert dataset is not None and dataset.role == "statistics"
    assert shown.properties["variable"] == "Hs"
    assert {"eta_max", "eta_min", "eta_mean", "velocity_max", "Hs"} <= set(dataset.components)
    mesh = dataset.read(0)  # all its fields
    # eta = 0.1 sin(x - frame/2) + 0.02 y over six outputs (equal weights inside)
    times = np.arange(6) * 0.5
    w = np.array([0.25, 0.5, 0.5, 0.5, 0.5, 0.25])
    x, y = mesh.points[:, 0], mesh.points[:, 1]
    series = np.array([eta(x, y, f) for f in range(6)], np.float64)
    mean = (series * w[:, None]).sum(0) / w.sum()
    hs = 4 * np.sqrt(((series - mean) ** 2 * w[:, None]).sum(0) / w.sum())
    np.testing.assert_allclose(mesh.point_data["Hs"], hs, rtol=1e-4, atol=1e-6)
    np.testing.assert_allclose(mesh.point_data["eta_max"], series.max(0), atol=1e-6)
    tmax = times[series.argmax(0)]
    np.testing.assert_allclose(mesh.point_data["eta_time_of_max"], tmax)
    # the same at any time; drawn
    s.set_time(2.0)
    assert data.frame_index(item) == 0
    assert data.display(item, mesh).values is not None
    # kept in the case folder: a new session finds it without reading the run
    path = data.statistics_file(item)
    assert path.parent.name == "LAGOON_Statistics" and path.is_file()
    again = Session()
    again.open_case(root)
    second = again.add_statistics()
    assert again.data.statistics_file(second) == path
    # another window: another map; a bed: its elevation's change
    later = s.add_statistics(start=1.0)
    assert data.statistics_file(later) != path
    bed = next(k for k, d in data.case(item).datasets.items() if d.role == "bed")
    bed_item = s.add_statistics(source=bed)
    assert "elevation_change" in data.dataset(bed_item).components
    # the script makes it again
    assert "add_item('statistics'" in s.to_script()


def test_statistics_need_the_same_points(tmp_path):
    from reef3d_fixtures import make_amr

    make_amr(tmp_path, "SFLOW", frames=3)
    s = Session()
    s.open_case(tmp_path)
    case = s.data.case(next(i.id for i in s.scene if i.kind == "case"))
    assert not [k for k, d in case.datasets.items() if d.role in statistics.SOURCE_ROLES]
    with pytest.raises(ValueError):
        s.add_statistics()
    assert not [i for i in s.scene if i.kind == "statistics"]  # taken back


def test_statistics_of_the_add_menu(tmp_path):
    root = make_run(tmp_path / "run", frames=3, print_interval=0.5, volume=False)
    s = Session()
    s.open_case(root)
    keys = [k for _c, k, _l in s.data.addable("statistics")]
    assert sorted(keys) == ["nhflow_bed", "nhflow_free_surface"]
    item = s.add_of("statistics", "nhflow_bed")
    assert s.scene.get(item).properties["source"] == "nhflow_bed"
