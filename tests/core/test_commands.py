import pytest

from lagoonviz.core import Session


def test_add_undo_redo():
    s = Session()
    item = s.add_item("free_surface")
    assert item == "free_surface-1" and item in s.scene
    s.undo()
    assert item not in s.scene
    s.redo()
    assert item in s.scene and s.scene.get(item).name == "Free surface"


def test_redo_keeps_id_and_new_items_do_not_collide():
    s = Session()
    a = s.add_item("slice")
    s.undo()
    s.redo()
    b = s.add_item("slice")
    assert (a, b) == ("slice-1", "slice-2")


def test_remove_restores_position():
    s = Session()
    ids = [s.add_item("gauge") for _ in range(3)]
    s.remove_item(ids[1])
    assert s.scene.ids() == [ids[0], ids[2]]
    s.undo()
    assert s.scene.ids() == ids


def test_set_property_validates():
    s = Session()
    fs = s.add_item("free_surface")
    s.set_property(fs, "opacity", 0.5)
    assert s.scene.get(fs).properties["opacity"] == 0.5
    with pytest.raises(ValueError):
        s.set_property(fs, "opacity", 2.0)
    with pytest.raises(ValueError):
        s.set_property(fs, "colormap", "not-a-map")
    with pytest.raises(ValueError):
        s.set_property(fs, "nonsense", 1)
    s.undo()
    assert s.scene.get(fs).properties["opacity"] == 1.0
    s.set_property(fs, "colormap", "cividis")  # replaced by rainbow: old scripts still run
    assert s.scene.get(fs).properties["colormap"] == "rainbow"


def test_merge_makes_one_undo_step():
    s = Session()
    fs = s.add_item("free_surface")
    for value in (0.9, 0.8, 0.7):
        s.set_property(fs, "opacity", value, merge=True)
    s.stack.end_merge()
    assert len(s.stack.done) == 2
    s.undo()
    assert s.scene.get(fs).properties["opacity"] == 1.0


def test_undo_labels():
    s = Session()
    s.add_item("slice", name="Mid slice")
    assert s.stack.undo_label == "Add Mid slice"
    s.undo()
    assert s.stack.redo_label == "Add Mid slice" and not s.stack.can_undo


def test_new_command_clears_redo():
    s = Session()
    s.add_item("gauge")
    s.undo()
    s.add_item("slice")
    assert not s.stack.can_redo


def test_ids_after_undo_match_replay():
    """An undone add must not shift later ids, or recorded scripts would break."""
    s = Session()
    s.add_item("slice")
    s.add_item("slice")
    s.undo()
    third = s.add_item("slice")
    s.set_property(third, "normal", "x")
    replayed = Session.from_script(s.to_script())
    assert replayed.scene.to_dict() == s.scene.to_dict()


def test_removing_a_case_removes_what_shows_it(tmp_path):
    """A case goes with its surfaces, slices, time series...; another case's items
    stay; undo brings all of them back where they were."""
    from reef3d_fixtures import make_run

    first = make_run(tmp_path / "a", frames=2, series=True)
    second = make_run(tmp_path / "b", frames=2)
    s = Session()
    a = s.open_case(first)
    b = s.open_case(second)
    shows_a = [i.id for i in s.scene if i.properties.get("case") == a]
    shows_b = [i.id for i in s.scene if i.properties.get("case") == b]
    assert shows_a and shows_b and any(s.scene.get(i).kind == "series" for i in shows_a)
    before = s.scene.ids()
    s.remove_item(a)
    assert a not in s.scene and not any(i in s.scene for i in shows_a)
    assert b in s.scene and all(i in s.scene for i in shows_b)
    s.undo()
    assert s.scene.ids() == before
    s.redo()
    assert a not in s.scene and b in s.scene
    # the only case: items naming no case (they show it) go with it too
    s.remove_item(b)
    assert not [i for i in s.scene if "case" in i.properties]
    c = s.open_case(second)
    loose = s.add_item("slice")
    s.remove_item(c)
    assert loose not in s.scene
