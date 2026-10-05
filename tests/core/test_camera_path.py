"""Camera keyframes: a smooth camera path over the timeline."""

import numpy as np
import pytest

from lagoonviz.core import Session
from lagoonviz.core.camera_path import CameraKey, camera_at, sorted_keys


def _state(x, y, z, angle=0.0, fov=50.0):
    """A camera at (x, y, z) turned ``angle`` degrees about z (quaternion x, y, z, w)."""
    half = np.radians(angle) / 2
    return {
        "position": [x, y, z],
        "rotation": [0.0, 0.0, float(np.sin(half)), float(np.cos(half))],
        "fov": fov,
        "zoom": 1.0,
        "reference_up": [0.0, 0.0, 1.0],
        "maintain_aspect": True,
    }


KEYS = [
    CameraKey(0.0, _state(0, 0, 10, 0, 50)),
    CameraKey(10.0, _state(10, 0, 10, 90, 40)),
    CameraKey(30.0, _state(10, 20, 5, 180, 40)),
]


def _position(t):
    return np.array(camera_at(KEYS, t)["position"])


def test_through_the_keys_and_held_outside():
    for key in KEYS:
        assert np.allclose(_position(key.time), key.state["position"])
    assert np.allclose(_position(-5.0), KEYS[0].state["position"])
    assert np.allclose(_position(99.0), KEYS[-1].state["position"])
    assert camera_at([], 1.0) is None
    assert camera_at(KEYS[:1], 7.0) == KEYS[0].state
    # what does not move comes from the key before
    assert camera_at(KEYS, 5.0)["reference_up"] == [0.0, 0.0, 1.0]
    assert camera_at(KEYS, 5.0)["maintain_aspect"] is True


def test_smooth_and_easing():
    eps = 1e-4
    # no jump in speed at the inner key
    before = (_position(10.0) - _position(10.0 - eps)) / eps
    after = (_position(10.0 + eps) - _position(10.0)) / eps
    assert np.allclose(before, after, atol=1e-2)
    assert np.linalg.norm(before) > 0.3  # it does move through it
    # it eases in and out at the ends
    assert np.linalg.norm((_position(eps) - _position(0.0)) / eps) < 1e-2
    assert np.linalg.norm((_position(30.0) - _position(30.0 - eps)) / eps) < 1e-2


def test_rotation_and_field_of_view():
    two = KEYS[:2]
    state = camera_at(two, 5.0)
    q = np.array(state["rotation"])
    assert np.isclose(np.linalg.norm(q), 1.0)
    assert np.degrees(2 * np.arctan2(q[2], q[3])) == pytest.approx(45.0, abs=0.5)
    assert state["fov"] == pytest.approx(45.0)
    # q and -q are the same rotation: the path does not take the long way round
    flipped = [
        two[0],
        CameraKey(10.0, {**two[1].state, "rotation": [-v for v in two[1].state["rotation"]]}),
    ]
    q = np.array(camera_at(flipped, 5.0)["rotation"])
    assert abs(np.degrees(2 * np.arctan2(abs(q[2]), abs(q[3])))) == pytest.approx(45.0, abs=0.5)


def test_sorted_one_per_time():
    keys = sorted_keys([KEYS[2], KEYS[0], CameraKey(0.0, _state(1, 1, 1)), KEYS[1]])
    assert [k.time for k in keys] == [0.0, 10.0, 30.0]
    assert keys[0].state["position"] == [1, 1, 1]  # the later given wins


def test_session_keys(tmp_path):
    session = Session()
    with pytest.raises(ValueError, match="no camera"):
        session.add_camera_key()
    session.set_camera(_state(0, 0, 10))
    assert session.camera_at() is None  # no keys, no path
    assert session.add_camera_key() == 1
    session.set_time(10.0)
    session.add_camera_key(state=_state(10, 0, 10, 90))
    assert session.scene.camera_path
    assert session.camera_at(5.0)["position"] == pytest.approx([5.0, 0.0, 10.0])
    # the same time again: replaced
    assert session.add_camera_key(10.0, _state(20, 0, 10)) == 2
    session.set_camera_path(False)
    assert session.camera_at(5.0) is None
    session.set_camera_path(True)
    # saved and opened again; the recorded script makes them again
    path = tmp_path / "flight.lagoon-session"
    session.save(path)
    loaded = Session.open(path)
    assert loaded.scene.camera_keys == session.scene.camera_keys and loaded.scene.camera_path
    replayed = Session.from_script(session.to_script())
    assert replayed.scene.camera_keys == session.scene.camera_keys
    assert replayed.camera_at(5.0) == session.camera_at(5.0)
    session.set_camera_path(False)
    assert "set_camera_path(False)" in session.to_script()
    # removed one by one; a new session has none
    assert session.remove_camera_key(10.0) and not session.remove_camera_key(10.0)
    session.clear_camera_keys()
    assert session.scene.camera_keys == [] and not session.scene.camera_path
    session.add_camera_key(0.0, _state(0, 0, 1))
    session.reset()
    assert session.scene.camera_keys == []


def test_orbit_stays_on_the_circle_and_looks_at_the_centre():
    from lagoonviz.core.camera_path import orbit_keys

    def forward(state):
        x, y, z, w = state["rotation"]
        u = np.array([x, y, z])
        v = np.array([0.0, 0.0, -1.0])
        return v + 2 * np.cross(u, np.cross(u, v) + w * v)

    # a camera at (20, 0, 5) looking at the centre (0, 0, 5), along -x
    start = _state(20, 0, 5)
    from lagoonviz.render.pathtrace import look_rotation

    start["rotation"] = look_rotation([-1.0, 0.0, 0.0])
    np.testing.assert_allclose(forward(start), [-1, 0, 0], atol=1e-12)
    keys = orbit_keys(start, (0, 0, 5), 0.0, 12.0, 360.0)
    assert len(keys) == 13 and keys[0].time == 0.0 and keys[-1].time == 12.0
    for t in np.linspace(0.0, 12.0, 97):
        state = camera_at(keys, t)
        p = np.asarray(state["position"])
        assert np.hypot(p[0], p[1]) == pytest.approx(20.0, rel=3e-3)  # on the circle
        assert p[2] == pytest.approx(5.0)
        to_centre = (np.array([0, 0, 5]) - p) / np.linalg.norm(np.array([0, 0, 5]) - p)
        assert forward(state) @ to_centre > 0.999  # looking at the centre
    np.testing.assert_allclose(keys[3].state["position"], [0, 20, 5], atol=1e-9)  # 90 deg
    np.testing.assert_allclose(keys[-1].state["position"], [20, 0, 5], atol=1e-9)
    assert len(orbit_keys(start, (0, 0, 5), 0.0, 1.0, 45.0)) == 3


def test_session_orbit(tmp_path):
    from reef3d_fixtures import make_run

    session = Session()
    session.open_case(make_run(tmp_path / "run", volume=False))
    with pytest.raises(ValueError, match="camera"):
        session.orbit_camera()
    session.set_camera(_state(30, -20, 8))
    assert session.orbit_camera(degrees=90) == 4
    keys = session.scene.camera_keys
    assert [k["time"] for k in keys] == [0.0, 2 / 3, 4 / 3, 2.0]
    assert session.scene.camera_path
    centre = np.asarray(session.data_centre())
    first, last = (np.asarray(k["state"]["position"]) for k in (keys[0], keys[-1]))
    # a quarter turn about the vertical through the data's middle
    a, b = first[:2] - centre[:2], last[:2] - centre[:2]
    assert a[0] * b[1] - a[1] * b[0] > 0 and abs(a @ b) < 1e-6 * (a @ a)
    assert "session.add_camera_key(" in session.to_script()
