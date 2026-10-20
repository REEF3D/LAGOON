"""Bathymetry files of one's own for a coast: points files with named columns
(EMODnet's survey files), dense soundings averaged per cell, and geo.dat over the
blocks of a multi-block grid."""

import numpy as np
import pytest

from lagoonviz.core import Session
from lagoonviz.core.geo import bathy as B
from lagoonviz.core.geo import raster as R

EMO = """Longitude;Latitude;Minimum depth;Maximum depth;Average depth;Standard deviation
-9.30;38.65;10.2;11.0;10.6;0.2
-9.29;38.65;12.0;13.0;12.5;0.3
-9.28;38.66;8.0;9.0;8.4;0.1
"""


def test_points_files_with_named_columns(tmp_path):
    emo = tmp_path / "590_HR_RioTejo.emo"
    emo.write_text(EMO)
    assert R.header(emo)[4] == "Average depth"
    assert R.pick_columns(R.header(emo)) == (0, 1, 4)  # the average depth
    points = R.read_points(emo)
    assert points.shape == (3, 3) and points[1, 2] == 12.5
    assert R.read_points(emo, columns=["Longitude", "Latitude", "Maximum depth"])[0, 2] == 11.0
    assert R.read_points(emo, columns=[0, 1, 2])[2, 2] == 8.0
    with pytest.raises(R.RasterError, match="no column"):
        R.read_points(emo, columns=["lon", "lat", "nothing"])
    plain = tmp_path / "plain.xyz"
    plain.write_text("1 2 3\n4 5 6\n")
    assert R.read_points(plain).tolist() == [[1, 2, 3], [4, 5, 6]]
    found = B.load(emo, "EPSG:4326", depths=True)
    assert found.depths and len(found.points) == 3


def test_dense_soundings_are_averaged_per_cell():
    rng = np.random.default_rng(1)
    xs, ys = np.arange(0.0, 101.0, 10.0), np.arange(0.0, 51.0, 10.0)
    p = np.column_stack([rng.uniform(-5, 105, 20000), rng.uniform(-5, 55, 20000)])
    points = np.column_stack([p, 0.1 * p[:, 0] + rng.normal(0, 0.5, len(p))])  # noisy plane
    few = B.binned(points[:20], xs, ys)
    assert few is not None and len(few) == 20  # sparse: as they are
    dense = B.binned(points, xs, ys)
    assert len(dense) == len(xs) * len(ys)  # one a cell
    z = B.interpolate(points, xs, ys)
    gx, _ = np.meshgrid(xs, ys)
    assert np.abs(z - 0.1 * gx).max() < 0.6  # the noise averaged out


def test_geo_dat_reaches_round_the_blocks(tmp_path):
    folder = tmp_path / "coast"
    folder.mkdir()
    (folder / "control.txt").write_text("B 1 50.0\nB 10 0.0 100.0 0.0 100.0 0.0 5.0\n")
    (folder / "ctrl.txt").write_text("A 10 5\nF 60 10.0\n")
    gx, gy = np.meshgrid(np.arange(-500.0, 2501.0, 50.0), np.arange(-500.0, 1501.0, 50.0))
    np.savetxt(tmp_path / "bed.xyz", np.column_stack([gx.ravel(), gy.ravel(), (gx / 100).ravel()]))
    s = Session()
    s.open_case(folder)
    s.start_multiblock([0, 2000, 0, 1000], cells=(20, 10))  # far beyond B 10
    s.make_geo_dat(str(tmp_path / "bed.xyz"), crs="model", spacing=50.0)
    points = np.loadtxt(folder / "geo.dat")
    assert points[:, 0].min() <= -100 + 1e-6 and points[:, 0].max() >= 2100 - 1e-6
    assert points[:, 1].max() >= 1100 - 1e-6  # 5 % of the blocks' size beyond them
