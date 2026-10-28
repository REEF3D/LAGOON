"""The volume and what is cut from it (slices, iso-surfaces, iso-volumes) have the
free surface's options: a vertical scale of their own (with the case's *Sync*
off), and the volume's outside contour lines and arrows, also from a LAGOON
store (NHFLOW's)."""

import numpy as np
from reef3d_fixtures import make_run

from lagoonviz.core import Session
from lagoonviz.core.convert import convert
from lagoonviz.core.kinds import KINDS


def test_the_volume_has_the_options_of_the_free_surface():
    for kind in ("volume", "slice", "iso_surface", "iso_volume"):
        assert "exaggeration" in KINDS[kind].properties, kind
    for key in ("contour_by", "contour_levels", "vectors_of", "vector_count"):
        assert key in KINDS["volume"].properties


def test_the_vertical_scale_contours_and_arrows_of_a_stored_volume(tmp_path):
    root = make_run(tmp_path / "nhflow", solver="NHFLOW", frames=3, print_interval=0.5)
    convert(root)
    s = Session()
    s.open_case(root)
    volume = next(i for i in s.scene if i.kind == "volume")
    assert s.data.dataset(volume.id).source == "lagoon"  # from the store
    s.set_visible(volume.id, True)
    plain = s.data.display(volume.id, s.data.load(volume.id)).positions.copy()
    level = s.data.case_water_level(volume.id)
    # the case's scale (Sync on): the volume follows it, its slider sets it
    s.set_exaggeration(volume.id, 2.0)
    case = s.data.case_item(volume.id)
    assert case.properties["exaggeration"] == 2.0
    shown = s.data.display(volume.id, s.data.load(volume.id))
    np.testing.assert_allclose(shown.positions[:, 2], level + 2 * (plain[:, 2] - level), atol=1e-4)
    # Sync off: the volume's own scale
    s.sync_exaggeration(volume.id, False)
    s.set_exaggeration(volume.id, 4.0)
    assert volume.properties["exaggeration"] == 4.0 and case.properties["exaggeration"] == 2.0
    mesh = s.data.load(volume.id)
    shown = s.data.display(volume.id, mesh)
    np.testing.assert_allclose(shown.positions[:, 2], level + 4 * (plain[:, 2] - level), atol=1e-4)
    # contour lines and arrows on its outside
    s.set_property(volume.id, "contour_by", "pressure")
    s.set_property(volume.id, "vectors_of", "velocity")
    mesh = s.data.load(volume.id)
    shown = s.data.display(volume.id, mesh)
    lines = s.data.contour_lines(volume.id, mesh, shown)
    arrows = s.data.vector_arrows(volume.id, mesh, shown)
    assert lines is not None and len(lines.segments)
    assert arrows is not None and len(arrows.segments)
    # a slice of it with a scale of its own
    cut = s.add_slice("y", 2.0)
    s.set_exaggeration(cut, 3.0)
    assert s.scene.get(cut).properties["exaggeration"] == 3.0
