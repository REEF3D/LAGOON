"""Set-up and results: what each mode draws, the set-up items' own check boxes, undo,
and saved sessions."""

from lagoonviz.core import Session


def _write_inputs(case):
    (case / "control.txt").write_text("B 10 0.0 8.0 0.0 4.0 -0.5 3.0\nB 1 0.5\n")
    (case / "ctrl.txt").write_text("A 10 5\nF 60 2.0\nP 51 4 2\n")


def test_a_case_that_has_not_run_opens_in_the_set_up(tmp_path):
    _write_inputs(tmp_path)
    session = Session()
    session.open_case(tmp_path)
    assert session.mode == "setup"
    setup = next(i for i in session.scene if i.kind == "setup")
    assert setup.visible and session.data.drawn(setup.id)


def test_each_mode_draws_its_own_items(reef3d_run):
    _write_inputs(reef3d_run)
    session = Session()
    session.open_case(reef3d_run)
    assert session.mode == "results"
    surface = next(i for i in session.scene if i.kind == "free_surface")
    setup = next(i for i in session.scene if i.kind == "setup")
    case = next(i for i in session.scene if i.kind == "case")
    assert not setup.visible  # over the results: off until ticked
    assert session.data.item_mode(surface.id) == "results"
    assert session.data.item_mode(setup.id) == "setup"
    assert session.data.item_mode(case.id) == "both"

    session.set_mode("setup")
    assert session.scene.get(setup.id).visible  # in the set-up: on
    assert not session.data.drawn(surface.id) and session.data.drawn(setup.id)
    assert session.scene.get(surface.id).visible  # its own check box is kept

    session.set_mode("results")
    assert not session.scene.get(setup.id).visible and session.data.drawn(surface.id)
    # ticked over the results, it stays ticked there
    session.set_visible(setup.id, True)
    session.set_mode("setup")
    session.set_visible(setup.id, False)  # off in the set-up
    session.set_mode("results")
    assert session.scene.get(setup.id).visible
    session.set_mode("setup")
    assert not session.scene.get(setup.id).visible


def test_switching_is_undone_and_recorded(reef3d_run):
    _write_inputs(reef3d_run)
    session = Session()
    session.open_case(reef3d_run)
    setup = next(i for i in session.scene if i.kind == "setup")
    session.set_mode("setup")
    assert "session.set_mode('setup')" in session.to_script()
    session.undo()
    assert session.mode == "results" and not session.scene.get(setup.id).visible
    session.redo()
    assert session.mode == "setup" and session.scene.get(setup.id).visible


def test_the_set_up_gets_an_item_for_a_case_without_one(reef3d_run):
    _write_inputs(reef3d_run)
    session = Session()
    case_id = session.open_case(reef3d_run)
    setup = next(i for i in session.scene if i.kind == "setup")
    session.remove_item(setup.id)
    session.set_mode("setup")
    added = [i for i in session.scene if i.kind == "setup"]
    assert len(added) == 1 and added[0].properties["case"] == case_id and added[0].visible
    session.undo()  # the switch and the item it added go together
    assert not any(i.kind == "setup" for i in session.scene) and session.mode == "results"


def test_a_saved_session_keeps_its_mode(reef3d_run, tmp_path):
    _write_inputs(reef3d_run)
    session = Session()
    session.open_case(reef3d_run)
    setup = next(i for i in session.scene if i.kind == "setup")
    session.set_mode("setup")
    path = tmp_path / "s.lagoon-session"
    session.save(path)
    again = Session()
    again.load(path)
    assert again.mode == "setup" and again.scene.get(setup.id).visible
    assert again.scene.other_visibility == {setup.id: False, "grid-1": False}
    again.set_mode("results")
    assert not again.scene.get(setup.id).visible


def test_results_opened_from_the_set_up_are_shown(reef3d_run, tmp_path):
    session = Session()
    session.new_case(tmp_path / "draft")  # working on a set-up
    assert session.mode == "setup"
    session.open_case(reef3d_run)  # then a case with results
    assert session.mode == "results"
    surface = next(i for i in session.scene if i.kind == "free_surface")
    assert session.data.drawn(surface.id)


def test_a_new_case_is_not_started_in_a_lagoon_store(tmp_path):
    import pytest

    store = tmp_path / "REEF3D_FNPF.lagoon"
    store.mkdir()
    (store / "zarr.json").write_text("{}")
    with pytest.raises(ValueError, match="LAGOON store"):
        Session().new_case(store)
    assert not (store / "control.txt").exists() and not (store / "ctrl.txt").exists()
