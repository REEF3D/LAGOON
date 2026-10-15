"""AMR output of REEF3D (FNPF, NHFLOW, SFLOW with A 270): the leaf surface."""

import numpy as np
import pytest
from reef3d_fixtures import AMR_DEPTH, AMR_FIELDS, amr_eta, make_amr, make_run

from lagoonviz.core import Session
from lagoonviz.core.data.amr import (
    index_files,
    leaf_mesh,
    read_frame,
    read_grid,
    value_at,
)
from lagoonviz.core.data.reef3d import open_case


def _area(mesh) -> float:
    p = mesh.points.astype(np.float64)
    t = mesh.triangles.astype(np.int64)
    cross = np.cross(p[t[:, 1]] - p[t[:, 0]], p[t[:, 2]] - p[t[:, 0]])
    return float(0.5 * np.abs(cross[:, 2]).sum())


def test_index_and_grids(tmp_path):
    folder = make_amr(tmp_path, "NHFLOW", frames=2, ranks=2)
    files = index_files(folder / "REEF3D-NHFLOW-AMR-00000001.vtm")
    assert [f.name for f in files] == [
        "REEF3D-NHFLOW-AMR-L0-00000001-0001.vtr",
        "REEF3D-NHFLOW-AMR-L0-00000001-0002.vtr",
        "REEF3D-NHFLOW-AMR-00000001-0001-0001.vtr",
        "REEF3D-NHFLOW-AMR-00000001-0001-0002.vtr",
        "REEF3D-NHFLOW-AMR-00000001-0002-0001.vtr",
    ]
    grids = read_frame(folder / "REEF3D-NHFLOW-AMR-00000001.vtm")
    assert [g.level for g in grids] == [0, 0, 1, 2, 1]
    assert all(g.time == 0.5 for g in grids)
    patch = grids[2]
    assert patch.box == (1.5, 3.5, 0.5, 2.5) and (patch.nx, patch.ny) == (8, 8)
    assert list(patch.cell_data) == list(AMR_FIELDS["NHFLOW"])
    # cell values at the cell centres, rows of x per y
    xc = 0.5 * (patch.x[:-1] + patch.x[1:])
    yc = 0.5 * (patch.y[:-1] + patch.y[1:])
    expected = amr_eta(*np.meshgrid(xc, yc), 0.5)
    assert np.allclose(patch.cell_data["eta"], expected)
    only = read_grid(files[0], ["eta"])
    assert list(only.cell_data) == ["eta"]


def test_leaf_surface_covers_the_domain_once(tmp_path):
    folder = make_amr(tmp_path, "NHFLOW", frames=3, ranks=2)
    for step in range(3):
        grids = read_frame(folder / f"REEF3D-NHFLOW-AMR-{step:08d}.vtm")
        mesh = leaf_mesh(grids)
        assert _area(mesh) == pytest.approx(8.0 * 3.0)  # no gaps, no overlaps
        # the finest grid is drawn where patches overlap
        level = mesh.point_data["level"]
        assert set(np.unique(level)) == {0.0, 1.0, 2.0}
        centre = mesh.points[mesh.triangles].mean(axis=1)
        inside = (np.abs(centre[:, 0] - (2.0 + 0.5 * step)) < 0.45) & (
            np.abs(centre[:, 1] - 1.5) < 0.45
        )
        assert (level[mesh.triangles[inside]] == 2.0).all()
        # heights: the elevation; eta for exaggeration; depth from WL for dry land
        assert np.allclose(mesh.points[:, 2], mesh.point_data["eta"] + AMR_DEPTH, atol=1e-6)
        assert "depth" in mesh.point_data
        # the borders of the patches (3 patches from the second output on)
        patches = sum(g.level > 0 for g in grids)
        borders = sum(2 * (g.nx + g.ny) for g in grids if g.level > 0)
        assert mesh.outlines.shape == (borders, 2) and patches == (2 if step == 0 else 3)


@pytest.mark.parametrize("solver", ["FNPF", "SFLOW"])
def test_other_solvers(tmp_path, solver):
    make_amr(tmp_path, solver, frames=2, ranks=1)
    case = open_case(tmp_path)
    dataset = case.datasets[f"{solver.lower()}_amr"]
    assert dataset.label == f"{solver} AMR surface"
    assert set(AMR_FIELDS[solver]) <= set(dataset.components)
    mesh = dataset.read(1, ["eta"])
    assert _area(mesh) == pytest.approx(24.0)
    assert ("depth" in mesh.point_data) == ("WL" in AMR_FIELDS[solver])
    if solver == "SFLOW":  # an Int32 field: read as such, drawn as float
        grid = read_grid(dataset.grid_files(1)[0], ["wetdry"])
        assert grid.cell_data["wetdry"].dtype == np.int32
        mesh = dataset.read(1, ["wetdry"])
        assert (mesh.point_data["wetdry"] == 1.0).all()
        from lagoon_format.store import Store

        from lagoonviz.core.convert import convert

        stored = Store(convert(tmp_path).store).amr["sflow_amr"]
        assert stored.fields["wetdry"] == (1, np.dtype(np.int32))
        assert stored.fields["eta"] == (1, np.dtype(np.float32))


def test_amr_grid_lines_and_patches_can_be_hidden(tmp_path):
    make_amr(tmp_path, "NHFLOW", frames=3)
    session = Session()
    session.open_case(tmp_path)
    item = next(i for i in session.scene if i.kind == "amr")
    session.set_time(1.0)
    mesh = session.data.load(item.id)
    shown = session.data.display(item.id, mesh)
    # grid lines: off by default, then every leaf cell's edges (no diagonals)
    assert session.data.amr_grid_lines(item.id, mesh, shown) is None
    session.set_property(item.id, "grid_lines", "white")
    lines = session.data.amr_grid_lines(item.id, mesh, shown)
    assert lines is not None and lines.color == "#ffffff"
    outlines = session.data.patch_outlines(item.id, mesh, shown)
    assert len(lines.segments) > len(outlines.segments)
    step = lines.segments[:, 1, :2] - lines.segments[:, 0, :2]
    assert (np.abs(step).min(axis=1) < 1e-6).all()  # along x or y only
    # the refined patches hidden: the base grid alone, whole, without outlines
    session.set_property(item.id, "patches", False)
    base = session.data.load(item.id)
    assert set(np.unique(base.point_data["level"])) == {0.0}
    assert base.outlines is None
    assert base.triangles.shape[0] < mesh.triangles.shape[0]
    shown = session.data.display(item.id, base)
    assert session.data.patch_outlines(item.id, base, shown) is None
    assert session.data.amr_grid_lines(item.id, base, shown) is not None
    session.set_property(item.id, "patches", True)
    assert session.data.load(item.id).outlines is not None


def test_case_with_amr(tmp_path):
    make_run(tmp_path, frames=3, print_interval=0.5)
    make_amr(tmp_path, "NHFLOW", frames=3)
    case = open_case(tmp_path)
    dataset = case.datasets["nhflow_amr"]
    assert dataset.times.tolist() == [0.0, 0.5, 1.0]
    assert dataset.default_variable() == "eta"
    assert "level" in dataset.variables()
    shown = case.main_datasets()
    assert "amr" in shown and "free_surface" not in shown and "bed" in shown
    # opened from the folder or from one of its index files
    assert list(open_case(tmp_path / "REEF3D_NHFLOW_AMR").datasets) == ["nhflow_amr"]
    one = tmp_path / "REEF3D_NHFLOW_AMR" / "REEF3D-NHFLOW-AMR-00000002.vtm"
    assert list(open_case(one).datasets) == ["nhflow_amr"]


def test_session_draws_the_amr_surface(tmp_path):
    make_amr(tmp_path, "NHFLOW", frames=3)
    session = Session()
    session.open_case(tmp_path)
    item = next(i for i in session.scene if i.kind == "amr")
    session.set_time(1.0)
    mesh = session.data.load(item.id)
    shown = session.data.display(item.id, mesh)
    assert shown.values is not None and shown.variable == "eta"
    outlines = session.data.patch_outlines(item.id, mesh, shown)
    assert outlines is not None and outlines.segments.shape[1:] == (2, 3)
    assert outlines.color == "#1e2833"
    # lines lie on the surface, a little above it
    assert (outlines.segments[..., 2] > AMR_DEPTH - 0.1).all()
    # colour by refinement level; white or no outlines
    session.set_property(item.id, "variable", "level")
    session.set_property(item.id, "outlines", "white")
    shown = session.data.display(item.id, mesh)
    assert set(np.unique(shown.values)) == {0.0, 1.0, 2.0}
    assert session.data.patch_outlines(item.id, mesh, shown).color == "#ffffff"
    session.set_property(item.id, "outlines", "off")
    assert session.data.patch_outlines(item.id, mesh, shown) is None
    # vertical scale about the still water level
    session.set_property(session.data.case_item(item.id).id, "exaggeration", 4.0)
    shown = session.data.display(item.id, mesh)
    level = session.data.still_water_level(item.id, mesh)
    assert level == pytest.approx(AMR_DEPTH, abs=1e-6)
    assert np.allclose(shown.positions[:, 2], level + 4.0 * mesh.point_data["eta"], atol=1e-5)
    # the recorded script rebuilds it
    replayed = Session.from_script(session.to_script())
    assert any(i.kind == "amr" for i in replayed.scene)


def test_probe_series_from_the_finest_grid(tmp_path):
    make_amr(tmp_path, "NHFLOW", frames=3)
    dataset = open_case(tmp_path).datasets["nhflow_amr"]
    where, values = dataset.point_series("eta", 2.05, 1.55)
    assert where[:2] == (2.05, 1.55)
    for step, value in enumerate(values):
        grids = read_frame(dataset.frames[step].path)
        expected, level = value_at(grids, "eta", 2.05, 1.55)
        assert value == pytest.approx(expected)
        assert level == (2 if step < 2 else 1)  # the level-2 patch moves away at t = 1


def test_convert_amr_to_the_store(tmp_path):
    from lagoon_format.store import Store

    from lagoonviz.core.convert import convert

    make_run(tmp_path, frames=3, print_interval=0.5)
    folder = make_amr(tmp_path, "NHFLOW", frames=3)
    report = convert(tmp_path)
    assert report.outputs["amr"] == 3 and "free_surface" in report.outputs
    stored = Store(report.store).amr["nhflow_amr"]
    assert stored.info["role"] == "amr" and stored.info["source"] == "REEF3D_NHFLOW_AMR"
    assert stored.times.tolist() == [0.0, 0.5, 1.0]
    grids = stored.grids(1)
    assert [(g.level, g.rank) for g in grids] == [(0, 0), (0, 1), (1, 0), (2, 0), (1, 1)]
    # the case now shows the store's AMR surface, drawn as from the VTK files
    case = open_case(tmp_path)
    dataset = case.datasets["nhflow_amr"]
    assert dataset.label == "NHFLOW AMR surface (LAGOON)"
    assert "amr" in case.main_datasets() and case.main_datasets()["amr"] is dataset
    vtk = open_case(tmp_path, stores=False).datasets["nhflow_amr"]
    for index in range(3):
        a, b = dataset.read(index, ["eta"]), vtk.read(index, ["eta"])
        assert np.allclose(a.points, b.points, atol=1e-5)
        assert np.array_equal(a.triangles, b.triangles)
        assert np.array_equal(a.outlines, b.outlines)
        assert np.allclose(a.point_data["eta"], b.point_data["eta"], atol=1e-6)
    _, stored_series = dataset.point_series("eta", 2.05, 1.55)
    _, vtk_series = vtk.point_series("eta", 2.05, 1.55)
    assert np.allclose(stored_series, vtk_series, atol=1e-6)
    # a run going on: converting again appends the outputs written since
    make_amr(tmp_path, "NHFLOW", frames=5)
    assert len(list(folder.glob("*.vtm"))) == 5
    report = convert(tmp_path)
    assert report.outputs["amr"] == 2
    assert len(Store(report.store).amr["nhflow_amr"]) == 5
