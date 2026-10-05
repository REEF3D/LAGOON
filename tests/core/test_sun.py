"""The sun: where the light comes from, in the session (view state, saved, scripted)."""

import math

import pytest

from lagoonviz.core import Session
from lagoonviz.core.looks import LOOK_SUNS
from lagoonviz.core.session import direction_of


def test_direction_of():
    assert direction_of(0, 0) == pytest.approx((1, 0, 0))
    assert direction_of(90, 0) == pytest.approx((0, 1, 0), abs=1e-12)
    assert direction_of(0, 90) == pytest.approx((0, 0, 1), abs=1e-12)
    x, y, z = direction_of(225, 30)
    assert math.hypot(x, y) == pytest.approx(math.cos(math.radians(30)))
    assert x < 0 and y < 0 and z == pytest.approx(0.5)


def test_session_sun(tmp_path):
    session = Session()
    events = []
    session.scene.subscribe(lambda event, _id: events.append(event))
    assert session.sun_direction() is None  # the look's own
    azimuth, elevation = session.sun_angles("open_ocean")
    x, y, z = LOOK_SUNS["open_ocean"]
    assert azimuth == pytest.approx(math.degrees(math.atan2(y, x)) % 360)
    assert elevation == pytest.approx(math.degrees(math.asin(z / math.hypot(x, y, z))))
    session.set_sun(400, 20)  # azimuth wraps
    assert session.scene.sun == (40.0, 20.0) and events[-1] == "sun"
    assert session.sun_direction() == pytest.approx(direction_of(40, 20))
    session.set_sun(elevation=35)  # one of the two
    assert session.scene.sun == (40.0, 35.0)
    assert not session.stack.can_undo  # view state, not an undo step
    session.save(tmp_path / "s.lagoon-session")
    assert Session.open(tmp_path / "s.lagoon-session").scene.sun == (40.0, 35.0)
    script = session.to_script()
    assert "session.set_sun(40.0, 35.0)" in script
    assert Session.from_script(script).scene.sun == (40.0, 35.0)
    session.set_sun()  # back to the look's
    assert session.scene.sun is None and "set_sun" not in session.to_script()
