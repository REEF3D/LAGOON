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
    assert dataset.label == f"{solver} free surface AMR"
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
    assert dataset.label == "NHFLOW free surface AMR (LAGOON)"
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


def test_sflow_amr_velocity_from_u_v_w(tmp_path):
    """SFLOW writes u, v, w as scalars: LAGOON joins them into ``velocity`` (its
    magnitude, components and a point's time series)."""
    make_amr(tmp_path, "SFLOW", frames=3)
    case = open_case(tmp_path)
    dataset = case.datasets["sflow_amr"]
    assert dataset.components["velocity"] == 3
    assert {"velocity", "velocity.x", "velocity.y", "velocity.z"} <= set(dataset.variables())
    mesh = dataset.read(1, ["velocity"])
    u, v, w = (mesh.point_data[n] for n in ("u", "v", "w"))
    assert np.allclose(mesh.scalar("velocity"), np.sqrt(u * u + v * v + w * w), atol=1e-6)
    assert np.allclose(mesh.scalar("velocity.y"), v)
    _, speed = dataset.point_series("velocity", 2.0, 1.5)
    _, along = dataset.point_series("velocity.x", 2.0, 1.5)
    assert np.all(np.isfinite(speed)) and np.all(speed >= np.abs(along) - 1e-9)
    session = Session()
    session.open_case(tmp_path)
    item = next(i for i in session.scene if i.kind == "amr")
    session.set_property(item.id, "variable", "velocity")
    shown = session.data.display(item.id, session.data.load(item.id))
    assert shown.values is not None and shown.variable == "velocity"
    # NHFLOW writes no u, v: no velocity made up
    other = tmp_path / "nh"
    make_amr(other, "NHFLOW", frames=1)
    assert "velocity" not in open_case(other).datasets["nhflow_amr"].components


def test_sflow_amr_dry_land_by_wetdry():
    """SFLOW's AMR output says which cells are wet (wetdry); dry land is hidden by it,
    not by WL (no water depth on dry land)."""
    from lagoonviz.core.data.amr import AmrGrid
    from lagoonviz.core.data.manager import _wet_triangles

    x, y = np.linspace(0.0, 4.0, 5), np.linspace(0.0, 1.0, 2)
    wet = np.array([[1, 1, 0, 0]], np.int32)
    cells = {"eta": np.zeros((1, 4)), "WL": np.full((1, 4), 0.3), "wetdry": wet}
    mesh = leaf_mesh([AmrGrid(0, x, y, cell_data=cells)])
    assert "depth" not in mesh.point_data
    kept = _wet_triangles(mesh)
    left = kept[(kept != 0).any(axis=1)]
    assert len(left) == 4  # the two wet cells' triangles
    assert mesh.points[left.ravel(), 0].max() <= 2.0 + 1e-6


def test_sflow_amr_store_velocity_and_dry_land(tmp_path):
    """The same from a LAGOON store, as REEF3D writes SFLOW's AMR into it."""
    from lagoon_format.amr import AmrGrid as StoredGrid
    from lagoon_format.store import StoreWriter

    from lagoonviz.core.data.manager import _wet_triangles

    path = tmp_path / "REEF3D_SFLOW.lagoon"
    writer = StoreWriter.create(path, "SFLOW")
    names = ("eta", "elevation", "WL", "u", "v", "w", "wetdry")
    fields = {n: (1, "int32" if n == "wetdry" else "float32") for n in names}
    amr = writer.amr_writer(
        "sflow_amr",
        {
            "fields": fields,
            "source": "REEF3D_SFLOW_AMR",
            "attributes": {"dataset": "sflow_amr", "role": "amr", "refinement": 2},
        },
    )
    x, y = np.linspace(0.0, 4.0, 5), np.linspace(0.0, 1.0, 2)
    values = {
        "eta": np.zeros((1, 4), np.float32),
        "elevation": np.full((1, 4), 0.3, np.float32),
        "WL": np.full((1, 4), 0.3, np.float32),
        "u": np.full((1, 4), 0.3, np.float32),
        "v": np.full((1, 4), 0.4, np.float32),
        "w": np.zeros((1, 4), np.float32),
        "wetdry": np.array([[1, 1, 0, 0]], np.int32),
    }
    for t in range(2):
        amr.append(0.5 * t, t, [StoredGrid(0, x, y, values, 0)])
    writer.close()
    dataset = open_case(path).datasets["sflow_amr"]
    assert dataset.components["velocity"] == 3
    mesh = dataset.read(1, ["velocity", "eta", "depth", "wetdry"])
    assert np.allclose(mesh.scalar("velocity"), 0.5)
    _, speed = dataset.point_series("velocity", 1.0, 0.5)
    assert np.allclose(speed, 0.5)
    kept = _wet_triangles(mesh)
    assert int((kept != 0).any(axis=1).sum()) == 4


def test_a_store_written_over_by_a_new_run_shows_that_run_only(tmp_path):
    """REEF3D before 2026-10-15 started SFLOW's store on top of the earlier one: the
    new run wrote its free surface over the old one and left the earlier run's AMR
    surface. By the run log's streams LAGOON shows the latest run's outputs only."""
    import json

    from lagoon_format.amr import AmrGrid as StoredGrid
    from lagoon_format.store import StoreWriter

    path = tmp_path / "REEF3D_SFLOW.lagoon"
    writer = StoreWriter.create(path, "SFLOW", run={"type": "run", "run": "run-a"})
    x, y = np.arange(5.0), np.arange(3.0)
    surface = writer.add_output("free_surface", "surface", x, y, {"eta": 1}, [((0, 0), (5, 3))])
    for t in range(2):
        surface.append(0.5 * t, t, [{"z": np.zeros((3, 5)), "eta": np.zeros((3, 5))}])
    amr = writer.amr_writer(
        "sflow_amr",
        {
            "fields": {"eta": (1, "float32")},
            "source": "REEF3D_SFLOW_AMR",
            "attributes": {"dataset": "sflow_amr", "role": "amr", "refinement": 2},
        },
    )
    grid = StoredGrid(0, np.linspace(0, 4, 5), np.linspace(0, 2, 3), {"eta": np.zeros((2, 4))}, 0)
    amr.append(0.0, 0, [grid])
    writer.close()

    def stream(run, name, group):
        folder = "REEF3D_SFLOW.lagoon"
        return {
            "type": "stream",
            "run": run,
            "name": name,
            "stream": {"folder": folder, "group": group},
        }

    def log(*runs):
        (tmp_path / "REEF3D_Case").mkdir(exist_ok=True)
        lines = []
        for run, started, groups in runs:
            lines.append(
                {
                    "type": "run",
                    "format": "reef3d-run",
                    "version": 1,
                    "run": run,
                    "solver": "SFLOW",
                    "started": started,
                }
            )
            lines += [stream(run, f"lagoon_{g}", g) for g in groups]
        text = "".join(json.dumps(line) + "\n" for line in lines)
        (tmp_path / "REEF3D_Case" / "REEF3D_SFLOW_run.jsonl").write_text(text)

    # run a wrote both: both shown
    log(("run-a", "2026-10-15T17:31:19Z", ["free_surface", "amr/sflow_amr"]))
    assert {"sflow_free_surface", "sflow_amr"} <= set(open_case(tmp_path).datasets)
    # run b wrote its free surface into the store, no AMR (yet): the AMR is run a's
    log(
        ("run-a", "2026-10-15T17:31:19Z", ["free_surface", "amr/sflow_amr"]),
        ("run-b", "2026-10-15T21:13:17Z", ["free_surface"]),
    )
    case = open_case(tmp_path)
    assert "sflow_amr" not in case.datasets
    assert case.datasets["sflow_free_surface"].run_id == "run-b"
    assert "amr" not in case.main_datasets()
    # without the groups in the run log: as stored
    log(("run-b", "2026-10-15T21:13:17Z", []))
    assert "sflow_amr" in open_case(tmp_path).datasets


def test_no_crack_where_the_grids_meet(tmp_path):
    """Two ranks' level-0 grids side by side and a finer patch across their border:
    the surface is watertight (a node on a rank border gets the same height and
    normal from both grids; a finer grid's border lies on the coarse surface)."""
    from lagoonviz.core.data.amr import AmrGrid

    def surface(x, y):
        return 0.3 * np.sin(1.3 * x) * np.cos(0.9 * y) + 0.1 * x

    def grid(level, x0, x1, y0, y1, n):
        x = np.linspace(x0, x1, int(round((x1 - x0) * n)) + 1)
        y = np.linspace(y0, y1, int(round((y1 - y0) * n)) + 1)
        xc, yc = np.meshgrid(0.5 * (x[:-1] + x[1:]), 0.5 * (y[:-1] + y[1:]))
        return AmrGrid(level, x, y, cell_data={"eta": surface(xc, yc).astype(np.float32)})

    ranks = [grid(0, 0.0, 4.0, 0.0, 3.0, 4), grid(0, 4.0, 8.0, 0.0, 3.0, 4)]
    patch = grid(1, 3.0, 5.5, 1.0, 2.5, 8)  # over the ranks' border
    mesh = leaf_mesh([*ranks, patch])
    p = mesh.points.astype(np.float64)
    # the same (x, y) has one height and one normal
    used = np.unique(mesh.triangles)
    _, first, inverse = np.unique(np.round(p[used, :2], 6), axis=0, return_index=True,
                                  return_inverse=True)  # fmt: skip
    inverse = inverse.ravel()
    assert np.abs(p[used, 2] - p[used][first][inverse, 2]).max() < 1e-6
    border = np.isclose(p[used, 0], 4.0) & (mesh.point_data["level"][used] == 0)
    normals = mesh.normals[used]
    assert np.abs(normals - normals[first][inverse])[border].max() < 1e-6
    # the patch's border nodes on the coarse surface: linear along the coarse edges
    on_border = (mesh.point_data["level"][used] == 1) & (
        np.isclose(p[used, 1], 1.0) | np.isclose(p[used, 1], 2.5)
    )
    coarse = (mesh.point_data["level"][used] == 0) & np.isclose(p[used, 1], 1.0)
    cx = p[used][coarse]
    order = np.argsort(cx[:, 0])
    fine = p[used][on_border & np.isclose(p[used, 1], 1.0)]
    expected = np.interp(fine[:, 0], cx[order, 0], cx[order, 2])
    assert np.abs(fine[:, 2] - expected).max() < 1e-5
    # the surface covers the domain once still
    assert _area(mesh) == pytest.approx(8.0 * 3.0)
