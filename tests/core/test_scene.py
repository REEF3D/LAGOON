import pytest

from lagoonviz.core import Scene, SceneItem


def test_insert_order_and_ids():
    scene = Scene()
    scene.insert(SceneItem(scene.new_id("slice"), "slice", "A"))
    scene.insert(SceneItem(scene.new_id("slice"), "slice", "B"))
    scene.insert(SceneItem(scene.new_id("gauge"), "gauge", "G"), index=0)
    assert scene.ids() == ["gauge-1", "slice-1", "slice-2"]


def test_unknown_kind_rejected():
    scene = Scene()
    with pytest.raises(ValueError):
        scene.insert(SceneItem("x-1", "nonsense", "X"))


def test_observer_events():
    scene = Scene()
    events = []
    scene.subscribe(lambda event, item_id: events.append((event, item_id)))
    scene.insert(SceneItem("gauge-1", "gauge", "G"))
    scene.update("gauge-1", name="G2")
    scene.set_time(1.5)
    scene.pop("gauge-1")
    assert events == [
        ("added", "gauge-1"),
        ("changed", "gauge-1"),
        ("time", None),
        ("removed", "gauge-1"),
    ]


def test_save_load_roundtrip(tmp_path):
    scene = Scene()
    scene.insert(SceneItem(scene.new_id("gauge"), "gauge", "G", properties={"x": 2.0}))
    scene.set_time(3.0)
    path = tmp_path / "s.lagoon-session"
    scene.save(path)
    other = Scene()
    other.load(path)
    assert other.to_dict() == scene.to_dict()
    # counters survive, so new ids do not collide
    assert other.new_id("gauge") == "gauge-2"


def test_load_fills_missing_defaults():
    scene = Scene()
    scene.load_dict(
        {
            "lagoon_session": 1,
            "items": [
                {"id": "slice-1", "kind": "slice", "name": "S", "properties": {"normal": "x"}}
            ],
        }
    )
    props = scene.get("slice-1").properties
    assert props["normal"] == "x" and props["colormap"] == "ocean"


def test_rejects_unknown_session_version():
    with pytest.raises(ValueError):
        Scene().load_dict({"lagoon_session": 99})
