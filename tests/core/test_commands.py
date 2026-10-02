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
        s.set_property(fs, "colormap", "rainbow")
    with pytest.raises(ValueError):
        s.set_property(fs, "nonsense", 1)
    s.undo()
    assert s.scene.get(fs).properties["opacity"] == 1.0


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
