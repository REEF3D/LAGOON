"""The play range: the part of the timeline that plays, saved with the session."""

import pytest
from reef3d_fixtures import make_run

from lagoonviz.core import Session
from lagoonviz.core.scene import Scene


def test_play_range_is_kept_and_saved(tmp_path):
    run = make_run(tmp_path / "run", frames=6, print_interval=0.5)
    session = Session()
    session.open_case(run)
    events = []
    session.scene.subscribe(lambda event, _item: events.append(event))
    session.set_play_range(2.0, 1.0)  # either order
    assert session.scene.play_range == (1.0, 2.0)
    assert events[-1] == "play_range"
    session.set_play_range(end=2.0)  # from the first time
    assert session.scene.play_range == (0.0, 2.0)
    assert "session.set_play_range(0.0, 2.0)" in session.to_script()
    path = tmp_path / "s.lagoon-session"
    session.scene.save(path)
    scene = Scene()
    scene.load(path)
    assert scene.play_range == (0.0, 2.0)
    session.set_play_range()
    assert session.scene.play_range is None
    assert "set_play_range" not in session.to_script()


def test_films_keep_to_the_play_range(tmp_path):
    from lagoonviz.render.movie import frame_times

    run = make_run(tmp_path / "run", frames=6, print_interval=0.5)
    session = Session()
    session.open_case(run)
    everything = frame_times(session)
    assert everything == pytest.approx([0.0, 0.5, 1.0, 1.5, 2.0, 2.5])
    session.set_play_range(0.5, 1.5)
    assert frame_times(session) == pytest.approx([0.5, 1.0, 1.5])
    assert frame_times(session, 0, 5) == pytest.approx(everything)  # asked for: as asked
