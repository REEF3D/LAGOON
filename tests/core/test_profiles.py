"""Wave profiles along lines: REEF3D WSFLINE files and profile items."""

import numpy as np
import pytest
from reef3d_fixtures import PROFILE_LINES, PROFILE_X, make_run, profile_eta

from lagoonviz.core import Session
from lagoonviz.core.data.profiles import find_profiles, read_header, read_profile
from lagoonviz.core.data.reef3d import open_case


@pytest.fixture
def run(tmp_path):
    return make_run(tmp_path / "run", frames=6, volume=False, print_interval=0.5, profiles=True)


def test_read_profile_file(tmp_path):
    run = make_run(tmp_path / "fnpf", solver="FNPF", frames=3, volume=False, profiles=True)
    path = sorted((run / "REEF3D_FNPF_WSFLINE").glob("*.dat"))[1]
    time, ys = read_header(path)
    assert ys == PROFILE_LINES and time == pytest.approx(0.4)
    profile = read_profile(path)
    line = profile.lines[2]
    assert line.y == 3.0
    np.testing.assert_allclose(line.x, PROFILE_X)
    np.testing.assert_allclose(line.eta, profile_eta(PROFILE_X, 3.0, 0.4), atol=1e-8)
    np.testing.assert_allclose(line.theory, 0.9 * line.eta, atol=1e-8)


def test_the_profile_shown_follows_the_time(run):
    found = find_profiles(run)
    assert list(found) == ["nhflow_wsfline"]
    profiles = found["nhflow_wsfline"]
    assert len(profiles.paths) == 6 and profiles.times[-1] == pytest.approx(2.5)
    assert profiles.index_at(0.0) == 0 and profiles.index_at(1.2) == 2
    assert profiles.index_at(99.0) == 5 and profiles.index_at(-1.0) == 0
    assert profiles.at(1.0).lines[1].theory is None  # NHFLOW writes no wave theory
    low, high = profiles.value_range(1)
    assert low < -0.08 and high > 0.09


def test_profile_items_plot_and_draw_in_3d(run):
    case = open_case(run)
    assert "nhflow_wsfline" in case.profiles
    s = Session()
    s.open_case(run)
    item = next(i for i in s.scene if i.kind == "profile")
    assert item.name == "Wave profiles"
    s.set_time(1.0)
    lines = s.data.profile_lines(item.id)
    assert [line.number for line in lines] == [1, 2]
    np.testing.assert_allclose(lines[0].eta, profile_eta(PROFILE_X, 1.0, 1.0), atol=1e-8)
    s.set_property(item.id, "lines", "2")
    assert [line.number for line in s.data.profile_lines(item.id)] == [2]
    marks = s.data.profile_marks(item.id)
    assert marks.staffs.shape == (len(PROFILE_X) - 1, 2, 3)
    np.testing.assert_allclose(marks.staffs[:, 0, 1], 3.0)  # at the line's y
    assert marks.visible
    s.set_property(item.id, "profile_3d", False)
    assert not s.data.profile_marks(item.id).visible
    replayed = Session.from_script(s.to_script())
    assert replayed.scene.to_dict() == s.scene.to_dict()
