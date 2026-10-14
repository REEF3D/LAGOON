"""Several cases open at once: slices, iso-surfaces and streamlines cut the case
they are made for, not always the first."""

from reef3d_fixtures import make_run

from lagoonviz.core import Session


def _two(tmp_path, first_volume=True):
    s = Session()
    a = s.open_case(make_run(tmp_path / "a", frames=3, volume=first_volume))
    b = s.open_case(make_run(tmp_path / "b", frames=3))
    return s, a, b


def test_slices_cut_a_case_with_a_volume(tmp_path):
    s, a, b = _two(tmp_path, first_volume=False)
    assert s.data.volume_case() == b  # the first has no volume
    assert s.data.volume_case(a) == b
    slice_id = s.add_slice("y")
    assert s.scene.get(slice_id).properties["case"] == b
    assert s.data.load(slice_id).n_triangles
    iso = s.add_iso_surface()
    assert s.scene.get(iso).properties["case"] == b
    lines = s.add_streamlines()
    assert s.scene.get(lines).properties["case"] == b
    assert f"case={b!r}" in s.to_script()


def test_slices_cut_the_case_asked_for(tmp_path):
    s, a, b = _two(tmp_path)
    water_b = next(
        i.id for i in s.scene if i.kind == "free_surface" and s.data.case_item(i.id).id == b
    )
    assert s.data.volume_case(water_b) == b  # (the window: the selected item's case)
    first = s.add_slice("x", case=b)
    assert s.data.dataset(first) is not None
    assert s.data.case(first).path == s.data.case(water_b).path
    assert s.data.load(first).n_triangles


def test_a_slice_moved_to_another_case_still_cuts_it(tmp_path):
    s, a, b = _two(tmp_path)
    slice_id = s.add_slice("y", case=a)
    s.set_property(slice_id, "position", 100.0)  # beyond the volume
    s.set_property(slice_id, "case", b)
    props = s.scene.get(slice_id).properties
    assert props["case"] == b and 0.0 < props["position"] < 100.0
    assert s.data.load(slice_id).n_triangles
    s.undo()  # one step
    assert s.scene.get(slice_id).properties["case"] == a


def test_one_case_scripts_stay_as_they_were(reef3d_run):
    s = Session()
    s.open_case(reef3d_run)
    slice_id = s.add_slice("y")
    assert s.scene.get(slice_id).properties["case"] == ""
