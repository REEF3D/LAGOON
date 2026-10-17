"""Cases with a REEF3D run log: runs and times come from the log, not from guessing."""

import numpy as np
import pytest
from reef3d_fixtures import make_run, write_run_log

from lagoonviz.core import Session
from lagoonviz.core.data.reef3d import open_case

TIMES = [0.0, 0.3, 0.7, 1.2]  # an uneven output interval, and no ctrl.txt


def _logged_run(tmp_path):
    root = make_run(tmp_path / "logged", frames=4, volume=False)
    write_run_log(
        root,
        "NHFLOW",
        [
            {
                "id": "20261003T154312Z-aaaa",
                "started": "2026-10-03T15:43:12Z",
                "outputs": {"fsf": list(enumerate(TIMES)), "bed": [(0, 0.0)]},
            }
        ],
    )
    return root


def test_times_come_from_the_log(tmp_path):
    case = open_case(_logged_run(tmp_path))
    fsf = case.datasets["nhflow_free_surface"]
    assert fsf.run_id == "20261003T154312Z-aaaa"
    np.testing.assert_allclose(fsf.times, TIMES)
    assert fsf.index_at_time(0.5) == 1 and fsf.index_at_time(-0.1) is None
    assert fsf.same_run_as(case.datasets["nhflow_bed"])


def test_the_timeline_runs_in_logged_seconds(tmp_path):
    s = Session()
    s.open_case(_logged_run(tmp_path))
    times, steps, unit = s.data.timeline()
    np.testing.assert_allclose(times, TIMES)
    assert unit == "s" and list(steps) == [0, 1, 2, 3] and s.data.time_in_seconds()
    s.set_time(1.0)
    assert s.data.frame_index("free_surface-1") == 2  # the output of t = 0.7
    assert s.data.frame_index("bathymetry-1") == 0


def test_a_restart_is_split_by_the_log_not_by_file_times(tmp_path):
    """The restart wrote output 0 again, then stopped; file times are all the same
    here, so only the log can tell the runs apart."""
    root = make_run(tmp_path / "restarted", frames=12, volume=False)
    write_run_log(
        root,
        "NHFLOW",
        [
            {
                "id": "20261001T080000Z-old0",
                "started": "2026-10-01T08:00:00Z",
                "outputs": {"fsf": [(n, 0.5 * n) for n in range(12)], "bed": [(0, 0.0)]},
            },
            {
                "id": "20261003T154312Z-new1",
                "started": "2026-10-03T15:43:12Z",
                "outputs": {"fsf": [(0, 0.0)]},
            },
        ],
    )
    case = open_case(root)
    latest = case.datasets["nhflow_free_surface"]
    earlier = case.datasets["nhflow_free_surface_earlier1"]
    assert latest.run_id.endswith("new1") and latest.steps == [0]
    assert earlier.run_id.endswith("old0") and earlier.steps == list(range(1, 12))
    assert "2026-10-01 08:00" in earlier.label
    # the long run is shown (the restart is a short start), with its own times
    assert case.by_role("free_surface") is earlier
    np.testing.assert_allclose(earlier.times, 0.5 * np.arange(1, 12))


def test_a_case_written_by_reef3d():
    """Free surface, bed, gauges and profiles of the NHFLOW Stokes5 tutorial, run by
    the patched REEF3D: 2.2 s, then a restart of 1.2 s in the same folder. The free
    surface is written every 0.25 s (P 182) while ctrl.txt says P 30 0.5 for the
    volume: only the log gives its frames the right times."""
    from pathlib import Path

    root = Path(__file__).parents[1] / "data" / "reef3d_stokes5_restart"
    case = open_case(root)
    restart = case.datasets["nhflow_free_surface"]
    earlier = case.datasets["nhflow_free_surface_earlier1"]
    assert restart.steps == [0, 1, 2, 3, 4] and earlier.steps == [5, 6, 7, 8]
    assert restart.times[1] == pytest.approx(0.365, abs=1e-3)
    assert earlier.times[-1] == pytest.approx(2.221, abs=1e-3)
    s = Session()
    s.open_case(root)
    times, _, unit = s.data.timeline()
    np.testing.assert_allclose(times, restart.times)
    assert unit == "s"
    # films take the timeline's frames and times (not frame x P 30)
    from lagoonviz.render.movie import frame_times

    np.testing.assert_allclose(frame_times(s), times)
    np.testing.assert_allclose(frame_times(s, 1, 3), times[1:4])
    s.set_time(0.9)
    mesh = s.data.load("free_surface-1")
    assert mesh is not None and s.data.frame_index("free_surface-1") == 3
    kinds = {i.kind for i in s.scene}
    assert {"free_surface", "bathymetry", "series", "profile"} <= kinds


def test_the_output_interval_of_the_control_file_the_run_started_with(tmp_path):
    """Bug: ctrl.txt was changed while the run went on (P 30 0.05 at the start, 0.025
    later): LAGOON's times were half of REEF3D's. P 30 comes from the copy of
    ctrl.txt the run log names for the shown run."""
    root = make_run(tmp_path / "changed", frames=4, volume=False, print_interval=0.025)
    copy = root / "REEF3D_Case" / "ctrl-20261017T154840Z-4f21.txt"
    copy.parent.mkdir(exist_ok=True)
    copy.write_text("N 40 3\nP 30 0.05\nP 10 1\n")
    run = {"id": "20261017T154840Z-4f21", "started": "2026-10-17T15:48:40Z"}
    run["outputs"] = {"fsf": [(k, 0.05 * k) for k in range(4)], "bed": [(0, 0.0)]}
    run["ctrl"] = str(copy.relative_to(root))
    write_run_log(root, "NHFLOW", [run])
    s = Session()
    s.open_case(root)
    assert s.data.case("case-1").print_interval == pytest.approx(0.05)
    assert s.data.time_range() == pytest.approx((0.0, 0.15, "s"))  # 3 outputs of 0.05 s
    s.data.dataset("free_surface-1").times = None  # without logged times: P 30 x output
    s.data._logged = None
    assert s.data.time_range() == pytest.approx((0.0, 0.15, "s"))
    # without the copy: ctrl.txt, as before
    copy.unlink()
    s = Session()
    s.open_case(root)
    assert s.data.case("case-1").print_interval == pytest.approx(0.025)
