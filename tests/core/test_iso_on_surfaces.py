"""Iso-surfaces and iso-volumes of the data asked for: a volume of the case, or a
surface (the free surface, the bed): there the iso-surface is the line where the
variable has the value, the iso-volume the part of the surface in the range."""

import numpy as np
import pytest
from reef3d_fixtures import make_run

from lagoonviz.core import Session


@pytest.fixture
def session(tmp_path):
    s = Session()
    s.open_case(make_run(tmp_path / "run", frames=3, print_interval=0.5))
    return s


def _key(session, role):
    case = session.data.case(next(i.id for i in session.scene if i.kind == "case"))
    return next(k for k, d in case.datasets.items() if d.role == role)


def test_an_iso_line_on_the_free_surface(session):
    surface = _key(session, "free_surface")
    item = session.add_iso_surface(dataset=surface)
    props = session.scene.get(item).properties
    assert props["iso_by"] == "eta" and props["iso_value"] == 0.0  # the still water line
    mesh = session.data.load(item)
    assert mesh.n_triangles == 0 and len(mesh.segments) > 0
    assert np.allclose(mesh.point_data["eta"], 0.0, atol=1e-5)  # on the line: eta = 0
    shown = session.data.display(item, mesh)
    assert shown.segments is not None and len(shown.positions) == 2 * len(mesh.segments)
    # another value: another line; one the surface never reaches: no line, no error
    session.set_property(item, "iso_value", 1e6)
    assert len(session.data.load(item).segments) == 0


def test_an_iso_volume_on_the_bed(session):
    bed = _key(session, "bed")
    item = session.add_iso_volume(dataset=bed)
    props = session.scene.get(item).properties
    assert props["iso_by"] != "none"
    mesh = session.data.load(item)
    low, high = sorted((props["iso_min"], props["iso_max"]))
    values = mesh.scalar(props["iso_by"])
    assert mesh.n_triangles > 0 and values.min() >= low - 1e-4 and values.max() <= high + 1e-4


def test_the_data_an_iso_item_can_show(session):
    volume, surface, bed = (_key(session, r) for r in ("volume", "free_surface", "bed"))
    iso = session.add_iso_surface(dataset=volume)
    assert set(session.data.dataset_choices(iso)) >= {volume, surface, bed}
    cut = session.add_slice()
    assert session.data.dataset_choices(cut) == [volume]  # slices cut volumes only
    # in 2D the line lies just over the surface it is on
    line = session.add_iso_surface(dataset=surface)
    session.set_plan_view(True)
    flat = session.data.display(line, session.data.load(line))
    water = next(i for i in session.scene if i.kind == "free_surface")
    under = session.data.display(water.id, session.data.load(water.id))
    assert flat.positions[0, 2] > under.positions[0, 2]
    # a script makes the same
    assert f"dataset={surface!r}" in session.to_script()


def test_items_added_for_the_output_chosen(tmp_path):
    """Add → iso-surface (iso-volume, free surface, bed, slice) offers every output
    of the open cases it can show; the item shows the one chosen: an SFLOW free
    surface beside an NHFLOW run, an iso-line on the bed, a slice of the volume."""
    from lagoonviz.core.data.manager import shows

    root = make_run(tmp_path / "run", frames=3, print_interval=0.5)
    make_run(root, solver="SFLOW", frames=2, volume=False)
    session = Session()
    session.open_case(root)
    data = session.data

    def keys(kind):
        return [key for _case, key, _label in data.addable(kind)]

    assert sorted(keys("free_surface")) == ["nhflow_free_surface", "sflow_free_surface"]
    assert sorted(keys("bathymetry")) == ["nhflow_bed", "sflow_bed"]
    assert set(keys("iso_surface")) >= {"nhflow_volume", "sflow_free_surface", "sflow_bed"}
    assert keys("slice") == ["nhflow_volume"]
    labels = dict((key, label) for _c, key, label in data.addable("free_surface"))
    assert "SFLOW" in labels["sflow_free_surface"]
    # one case open: the labels do not name it
    assert ":" not in labels["sflow_free_surface"]
    surface = session.add_of("free_surface", "sflow_free_surface")
    assert data.dataset(surface).key == "sflow_free_surface"
    iso = session.add_of("iso_surface", "sflow_bed")
    assert data.dataset(iso).key == "sflow_bed"
    assert session.scene.get(iso).properties["iso_by"] != "none"
    patch = session.add_of("iso_volume", "sflow_free_surface")
    assert data.dataset(patch).role == "free_surface"
    cut = session.add_of("slice", "nhflow_volume")
    assert session.scene.get(cut).kind == "slice"
    with pytest.raises(ValueError):
        session.add_of("slice", "sflow_bed")  # a slice cuts a volume
    with pytest.raises(ValueError):
        session.add_of("free_surface", "nope")
    assert not shows("bathymetry", data.dataset(surface))
    # the script repeats it
    assert "dataset='sflow_bed'" in session.to_script()


def test_items_added_for_the_output_chosen_of_several_cases(tmp_path):
    first = make_run(tmp_path / "a", frames=2)
    second = make_run(tmp_path / "b", solver="SFLOW", frames=2, volume=False)
    session = Session()
    one = session.open_case(first)
    two = session.open_case(second)
    choices = session.data.addable("bathymetry")
    assert {(c, k) for c, k, _ in choices} == {(one, "nhflow_bed"), (two, "sflow_bed")}
    assert all(label.startswith(("a: ", "b: ")) for _c, _k, label in choices)
    with pytest.raises(ValueError):  # which case?
        session.add_of("bathymetry", "sflow_bed")
    bed = session.add_of("bathymetry", "sflow_bed", case=two)
    assert session.data.case_item(bed).id == two
