"""NHFLOW's mesh refinement on a curvilinear grid (G 50), also the blocks of a
multi-block grid (G 50 1): the grids of each output with the x and y of every node
in the LAGOON store (REEF3D writes them with P 18), drawn as one surface: each
place once (the cells another grid shows, ``covered``, left out), the grids of two
ranks and two blocks meeting without a crack."""

import numpy as np
import pytest

from lagoonviz.core.data.amr import AmrGrid, leaf_mesh, value_at

N_AROUND, N_ACROSS = 20, 8


def ring(i0, i1, j0, j1, n=1):
    """Nodes of cells i0..i1 (of 20 round a quarter ring, r 10 to 14 m) and j0..j1
    (of 8 across it), n times finer."""
    a = np.linspace(i0, i1, (i1 - i0) * n + 1) * (0.5 * np.pi / N_AROUND)
    r = 10.0 + np.linspace(j0, j1, (j1 - j0) * n + 1) * (4.0 / N_ACROSS)
    return r[:, None] * np.cos(a)[None, :], r[:, None] * np.sin(a)[None, :]


def grid(level, i0, i1, j0, j1, n=1, covered=None):
    x, y = ring(i0, i1, j0, j1, n)
    ny, nx = x.shape[0] - 1, x.shape[1] - 1
    xc = 0.25 * (x[:-1, :-1] + x[1:, :-1] + x[:-1, 1:] + x[1:, 1:])
    cells = {
        "eta": 0.01 * xc,
        "elevation": 2.0 + 0.01 * xc,
        "covered": np.zeros((ny, nx)) if covered is None else covered,
    }
    return AmrGrid(level, x, y, cell_data=cells)


def output():
    """Block 0 on two ranks, its ring (the column i 18..20) the cells of block 1;
    block 1 a level-0 grid there; a patch of level 1 over cells i 4..8, j 2..6."""
    rank0 = np.zeros((N_ACROSS, 10))
    rank0[2:6, 4:8] = 1  # under the patch
    rank1 = np.zeros((N_ACROSS, 10))
    rank1[:, 8:] = 2  # the ring of block 0
    return [
        grid(0, 0, 10, 0, N_ACROSS, covered=rank0),
        grid(0, 10, 20, 0, N_ACROSS, covered=rank1),
        grid(0, 18, 20, 0, N_ACROSS),  # block 1
        grid(1, 4, 8, 2, 6, n=2),
    ]


def area(mesh):
    p = mesh.points[:, :2].astype(np.float64)
    a, b, c = (p[mesh.triangles[:, k]] for k in range(3))
    u, v = b - a, c - a
    return 0.5 * np.abs(u[:, 0] * v[:, 1] - u[:, 1] * v[:, 0]).sum()


def test_each_place_once_and_no_crack():
    mesh = leaf_mesh(output())
    level0 = sum(  # the quadrilaterals of level 0, each place once
        area(leaf_mesh([grid(0, i, i + 1, j, j + 1)]))
        for i in range(N_AROUND)
        for j in range(N_ACROSS)
    )
    # the patch's finer quadrilaterals lie a little outside the coarse ones (closer to the arc)
    assert area(mesh) == pytest.approx(level0, rel=2e-3)
    # the triangles: the leaf cells (two each)
    leaf = 20 * 8 - 16 + 8 * 8  # level 0 without the patch's cells, and the patch
    assert mesh.n_triangles == 2 * leaf
    # one point per place: the ranks and the blocks share the nodes on their borders
    nodes = (N_AROUND + 1) * (N_ACROSS + 1) + 9 * 9 - 5 * 5  # the patch's own nodes
    assert mesh.n_points == nodes
    # every edge between two triangles of the same level is shared: no crack between
    # the ranks or between block 0 and block 1 (open edges: the outside, the patch's
    # T-junctions along its border)
    edges = np.sort(np.concatenate([mesh.triangles[:, [0, 1]], mesh.triangles[:, [1, 2]],
                                    mesh.triangles[:, [2, 0]]]), axis=1)  # fmt: skip
    _, count = np.unique(edges, axis=0, return_counts=True)
    open_edges = (count == 1).sum()
    outside = 2 * (N_AROUND + N_ACROSS)
    patch_border = 4 * 4 + 8 * 4  # its 4 x 4 coarse edges, its 8 x 4 fine ones
    assert open_edges == outside + patch_border
    # heights from elevation, the values the means of the cells round each node
    assert mesh.points[:, 2] == pytest.approx(2.0 + mesh.point_data["eta"], abs=1e-5)
    assert set(np.unique(mesh.point_data["level"])) == {0.0, 1.0}
    assert mesh.outlines is not None and len(mesh.outlines) == 4 * 8  # the patch's border
    # the base grid alone: the cells under the patch shown, the ring never
    base = leaf_mesh(output(), max_level=0)
    assert base.n_triangles == 2 * N_AROUND * N_ACROSS
    assert area(base) == pytest.approx(level0, rel=1e-9)


def test_a_value_at_a_point():
    x, y = ring(6, 7, 3, 4)
    inside = (x.mean(), y.mean())  # in the patch
    value, level = value_at(output(), "eta", *inside)
    assert level == 1 and value == pytest.approx(0.01 * inside[0], abs=1e-3)
    x, y = ring(19, 20, 0, 1)  # the ring: block 1's
    value, level = value_at(output(), "eta", x.mean(), y.mean())
    assert level == 0 and value == pytest.approx(0.01 * x.mean(), abs=1e-3)


def test_from_the_store(tmp_path):
    """The store REEF3D writes (P 18): the set's grids with their nodes, opened with
    the case as its free surface AMR."""
    from lagoon_format.amr import AmrGrid as StoredGrid
    from lagoon_format.store import StoreWriter

    from lagoonviz.core import Session

    case = tmp_path / "river"
    case.mkdir()
    writer = StoreWriter.create(case / "REEF3D_NHFLOW.lagoon", "NHFLOW")
    fields = {"eta": (1, "float32"), "elevation": (1, "float32"), "covered": (1, "int32")}
    amr = writer.amr_writer(
        "nhflow_amr",
        {"fields": fields, "curvilinear": True, "attributes": {"dataset": "nhflow_amr"}},
    )
    for t in range(2):
        amr.append(
            float(t),
            t,
            [
                StoredGrid(g.level, g.x, g.y, {k: v for k, v in g.cell_data.items()})
                for g in output()
            ],
        )
    writer.close()
    s = Session()
    s.open_case(case)
    item = next(i for i in s.scene if i.kind == "amr")
    mesh = s.data.load(item.id)
    expected = leaf_mesh(output())
    assert mesh.n_triangles == expected.n_triangles and mesh.n_points == expected.n_points
    assert area(mesh) == pytest.approx(area(expected), rel=1e-6)
    shown = s.data.display(item.id, mesh)
    assert shown.triangles.size and len(shown.positions) == mesh.n_points


def write_vts(folder, step, grids, time):
    """The files REEF3D writes on a curvilinear grid (P 18 0): an index per output and
    a structured grid (.vts, ASCII) per grid, with the cell array ``covered``."""
    names = []
    for k, g in enumerate(grids):
        name = (
            f"REEF3D-NHFLOW-AMR-L0-{step:08d}-{k + 1:04d}.vts"
            if g.level == 0
            else f"REEF3D-NHFLOW-AMR-{step:08d}-0001-{k + 1:04d}.vts"
        )
        names.append(name)
        nx, ny = g.nx, g.ny

        def array(kind, label, values, components=1):
            text = " ".join(f"{v:.12g}" for v in np.ravel(values))
            return (f'<DataArray type="{kind}" Name="{label}" NumberOfComponents="{components}" '
                    f'format="ascii">{text}</DataArray>\n')  # fmt: skip

        cells = "".join(
            array(
                "Int32" if n == "covered" else "Float64",
                n,
                np.asarray(v).astype(int) if n == "covered" else v,
            )
            for n, v in g.cell_data.items()
        )
        points = np.stack([g.x, g.y, np.zeros_like(g.x)], axis=-1)
        (folder / name).write_text(
            '<?xml version="1.0"?>\n<VTKFile type="StructuredGrid" version="0.1" '
            'byte_order="LittleEndian">\n'
            f'<StructuredGrid WholeExtent="0 {nx} 0 {ny} 0 0">\n'
            f'<FieldData><DataArray type="Float64" Name="TimeValue" NumberOfTuples="1">{time}'
            f'</DataArray><DataArray type="Int32" Name="level" NumberOfTuples="1">{g.level}'
            "</DataArray></FieldData>\n"
            f'<Piece Extent="0 {nx} 0 {ny} 0 0">\n<CellData Scalars="eta">\n{cells}</CellData>\n'
            f"<Points>\n{array('Float64', 'Points', points, 3)}</Points>\n</Piece>\n"
            "</StructuredGrid>\n</VTKFile>\n"
        )
    (folder / f"REEF3D-NHFLOW-AMR-{step:08d}.vtm").write_text(
        '<?xml version="1.0"?>\n<VTKFile type="vtkMultiBlockDataSet" version="1.0">\n'
        "<vtkMultiBlockDataSet>\n"
        + "".join(f'<DataSet index="{k}" file="{n}"/>\n' for k, n in enumerate(names))
        + "</vtkMultiBlockDataSet>\n</VTKFile>\n"
    )


def test_from_the_vts_files_and_converted(tmp_path):
    """The structured grids (.vts) REEF3D writes without the store (P 18 0) read as
    the store's grids; lagoon-convert keeps them curvilinear."""
    from lagoon_format.store import Store

    from lagoonviz.core import Session
    from lagoonviz.core.convert import convert

    case = tmp_path / "river"
    folder = case / "REEF3D_NHFLOW_AMR"
    folder.mkdir(parents=True)
    for step in range(2):
        write_vts(folder, step, output(), 0.5 * step)
    s = Session()
    s.open_case(case)
    item = next(i for i in s.scene if i.kind == "amr")
    mesh = s.data.load(item.id)
    expected = leaf_mesh(output())
    assert mesh.n_triangles == expected.n_triangles and mesh.n_points == expected.n_points
    assert area(mesh) == pytest.approx(area(expected), rel=1e-6)
    convert(case)
    stored = Store(case / "REEF3D_NHFLOW.lagoon").amr["nhflow_amr"]
    assert stored.curvilinear and len(stored) == 2
    assert [g.x.shape for g in stored.grids(1)] == [g.x.shape for g in output()]


def test_the_velocity_of_the_top_layer():
    """NHFLOW's u_top, v_top (Cartesian, also on a curvilinear grid): the surface's
    velocity, its magnitude at the leaf nodes and at a point."""
    grids = output()
    for g in grids:
        cx = 0.25 * (g.x[:-1, :-1] + g.x[1:, :-1] + g.x[:-1, 1:] + g.x[1:, 1:])
        cy = 0.25 * (g.y[:-1, :-1] + g.y[1:, :-1] + g.y[:-1, 1:] + g.y[1:, 1:])
        g.cell_data["u_top"] = -0.1 * cy  # round the ring
        g.cell_data["v_top"] = 0.1 * cx
    mesh = leaf_mesh(grids)
    velocity = mesh.point_data["velocity"]
    assert velocity.shape == (mesh.n_points, 3) and np.all(velocity[:, 2] == 0)
    speed = mesh.scalar("velocity")
    r = np.hypot(mesh.points[:, 0], mesh.points[:, 1])
    assert np.allclose(speed, 0.1 * r, rtol=0.03)
    x, y = ring(6, 7, 3, 4)
    value, level = value_at(grids, "velocity", x.mean(), y.mean())
    # the nearest cell of the patch: a quarter of a coarse cell off
    assert level == 1 and value == pytest.approx(0.1 * np.hypot(x.mean(), y.mean()), rel=0.02)


def with_bed(grids, dry_from=13.0):
    """The grids with NHFLOW's ``bed``: 0.5 m under the surface, but a film of 1 mm
    (its wetting and drying depth) where the ring is wider than ``dry_from``."""
    for g in grids:
        cx = 0.25 * (g.x[:-1, :-1] + g.x[1:, :-1] + g.x[:-1, 1:] + g.x[1:, 1:])
        cy = 0.25 * (g.y[:-1, :-1] + g.y[1:, :-1] + g.y[:-1, 1:] + g.y[1:, 1:])
        dry = np.hypot(cx, cy) > dry_from
        g.cell_data["bed"] = g.cell_data["elevation"] - np.where(dry, 0.001, 0.5)
    return grids


def test_dry_land_from_the_elevation_above_the_bed():
    """NHFLOW on a curvilinear grid writes neither WL nor wet: dry land is where the
    elevation is above the bed by its film only, and is hidden there."""
    from lagoonviz.core.data.manager import _wet_triangles

    mesh = leaf_mesh(with_bed(output()))
    assert "wetdry" in mesh.point_data
    triangles = _wet_triangles(mesh)
    gone = np.all(triangles == 0, axis=1) & np.any(mesh.triangles != 0, axis=1)
    assert gone.any() and not gone.all()
    centre = mesh.points[mesh.triangles].mean(axis=1)
    r = np.hypot(centre[:, 0], centre[:, 1])
    assert r[gone].min() > 13.0 - 0.5  # the dry band (and the shore cells half wet)
    assert r[~gone].max() < 13.5 + 1e-6


def test_the_bed_of_all_blocks_and_patches(tmp_path):
    """On a curvilinear grid with blocks the bed output holds block 0 alone: the bed
    shown is the AMR grids' (all blocks, the patches' finer one), not converted."""
    from lagoon_format.store import Store

    from lagoonviz.core import Session
    from lagoonviz.core.convert import convert
    from lagoonviz.core.data.amr import AmrBedDataset

    case = tmp_path / "river"
    folder = case / "REEF3D_NHFLOW_AMR"
    folder.mkdir(parents=True)
    for step in range(2):
        write_vts(folder, step, with_bed(output()), 0.5 * step)
    s = Session()
    s.open_case(case)
    item = next(i for i in s.scene if i.kind == "bathymetry")
    assert isinstance(s.data.dataset(item.id), AmrBedDataset)
    assert len(s.data.dataset(item.id)) == 1  # a fixed bed: its first output
    bed = s.data.load(item.id)
    surface = leaf_mesh(with_bed(output()))
    assert bed.n_points == surface.n_points and area(bed) == pytest.approx(area(surface))
    np.testing.assert_allclose(bed.points[:, 2], surface.point_data["bed"], atol=1e-5)
    assert bed.normals is not None and s.data.display(item.id, bed).triangles.size
    amr = next(i for i in s.scene if i.kind == "amr")
    s.set_property(amr.id, "hide_dry", True)
    mesh = s.data.load(amr.id)
    shown = s.data.display(amr.id, mesh)
    assert (np.all(shown.triangles == 0, axis=1)).any()  # the dry band hidden
    convert(case)
    store = Store(case / "REEF3D_NHFLOW.lagoon")
    assert "bed" not in store.outputs and len(store.amr["nhflow_amr"]) == 2
