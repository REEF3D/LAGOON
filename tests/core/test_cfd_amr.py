"""CFD's adaptive mesh refinement: the refined blocks of each output (REEF3D_CFD_AMR,
3D rectilinear grids listed by an index per output) join the volume. The volume's
cells under a block are left out, a block's cells under a finer one too; slices,
iso-surfaces (the free surface, phi at 0) and the outer surface work on the whole,
and the blocks' outlines are an item of their own."""

import shutil
from pathlib import Path

import numpy as np
from reef3d_fixtures import DEPTH, eta, make_run

from lagoonviz.core import Session
from lagoonviz.core.data.amr3d import is_volume_amr, read_block

#: the blocks of every output: level 1 over four of the volume's cells (x 2-4, y 1-3,
#: z 1.5-3: around the free surface, about 2 m high), level 2 inside it
BLOCKS = [
    (1, (2.0, 4.0), (1.0, 3.0), (1.5, 3.0), (4, 4, 2)),
    (2, (2.5, 3.5), (1.5, 2.5), (1.5, 2.25), (4, 4, 2)),
]


def _ascii(values) -> str:
    return " ".join(f"{float(v):.6g}" for v in np.asarray(values).ravel())


def write_block(path: Path, level, xr, yr, zr, cells, frame, finer=None):
    nx, ny, nz = cells
    x, y, z = (np.linspace(a, b, n + 1) for (a, b), n in ((xr, nx), (yr, ny), (zr, nz)))
    zc, yc, xc = np.meshgrid(
        0.5 * (z[1:] + z[:-1]), 0.5 * (y[1:] + y[:-1]), 0.5 * (x[1:] + x[:-1]), indexing="ij"
    )
    phi = DEPTH + eta(xc, yc, frame) - zc
    velocity = np.stack([np.cos(xc), 0 * xc, zc], axis=-1)
    covered = np.zeros_like(xc)
    if finer is not None:
        (fx0, fx1), (fy0, fy1), (fz0, fz1) = finer
        covered = (xc > fx0) & (xc < fx1) & (yc > fy0) & (yc < fy1) & (zc > fz0) & (zc < fz1)
    arrays = [("phi", phi, 1), ("level", np.full_like(xc, level), 1),
              ("covered", covered.astype(float), 1), ("velocity", velocity, 3)]  # fmt: skip
    body = "".join(
        f'<DataArray type="Float32" Name="{n}" NumberOfComponents="{c}" format="ascii">'
        f"{_ascii(v)}</DataArray>\n"
        for n, v, c in arrays
    )
    path.write_text(
        '<?xml version="1.0"?>\n<VTKFile type="RectilinearGrid" version="0.1" '
        'byte_order="LittleEndian">\n'
        f'<RectilinearGrid WholeExtent="0 {nx} 0 {ny} 0 {nz}">\n'
        f'<Piece Extent="0 {nx} 0 {ny} 0 {nz}">\n<CellData Scalars="phi">\n{body}</CellData>\n'
        "<Coordinates>\n"
        + "".join(
            f'<DataArray type="Float32" Name="{a}" format="ascii">{_ascii(v)}</DataArray>\n'
            for a, v in zip("xyz", (x, y, z), strict=True)
        )
        + "</Coordinates>\n</Piece>\n</RectilinearGrid>\n</VTKFile>\n"
    )


def cfd_amr_case(tmp_path, frames=3, refined=(0, 1, 2)):
    root = make_run(tmp_path / "cfd", solver="CFD", frames=frames, print_interval=0.5)
    for name in ("REEF3D_CFD_VTP_FSF", "REEF3D_CFD_VTP_BED"):
        shutil.rmtree(root / name)
    amr = root / "REEF3D_CFD_AMR"
    amr.mkdir()
    for frame in refined:
        names = []
        for number, (level, xr, yr, zr, cells) in enumerate(BLOCKS):
            name = f"REEF3D-CFD-AMR-{frame:08d}-{number:05d}.vtr"
            finer = BLOCKS[1][1:4] if level == 1 else None
            write_block(amr / name, level, xr, yr, zr, cells, frame, finer)
            names.append(name)
        (amr / f"REEF3D-CFD-AMR-{frame:08d}.vtm").write_text(
            '<?xml version="1.0"?>\n<VTKFile type="vtkMultiBlockDataSet" version="1.0">\n'
            "<vtkMultiBlockDataSet>\n"
            + "".join(f'<DataSet index="{k}" file="{n}"/>\n' for k, n in enumerate(names))
            + "</vtkMultiBlockDataSet>\n</VTKFile>\n"
        )
    # a block left over from an earlier run (not in any index): not read
    shutil.copy(
        amr / "REEF3D-CFD-AMR-00000000-00001.vtr", amr / "REEF3D-CFD-AMR-00000000-00009.vtr"
    )
    return root


def test_a_block_is_read_with_its_cells(tmp_path):
    root = cfd_amr_case(tmp_path)
    assert is_volume_amr(root / "REEF3D_CFD_AMR")
    block = read_block(root / "REEF3D_CFD_AMR" / "REEF3D-CFD-AMR-00000000-00000.vtr")
    assert block.level == 1 and block.box == (2.0, 4.0, 1.0, 3.0, 1.5, 3.0)
    assert block.cells["phi"].shape == (2, 4, 4) and block.cells["velocity"].shape == (2, 4, 4, 3)
    assert block.cells["covered"].sum() == 2 * 2 * 1  # under the level 2 block


def test_the_volume_with_its_refined_blocks(tmp_path):
    root = cfd_amr_case(tmp_path)
    s = Session()
    case_id = s.open_case(root)
    case = s.data.case(case_id)
    volume = case.datasets["cfd_volume"]
    assert type(volume).__name__ == "RefinedVolume" and volume.components["level"] == 1
    assert len(volume.frames[0].pieces) == 2 + 2  # two ranks, two blocks (not the left-over)
    # the free surface: finer where the blocks are
    free = next(i for i in s.scene if i.kind == "iso_surface")
    mesh = s.data.load(free.id)
    surface = volume.read_iso(0, "phi", 0.0, ["level"])
    level = surface.point_data["level"]
    assert set(np.unique(level)) == {0.0, 1.0, 2.0}
    inner = surface.points[level == 2]
    assert np.all((inner[:, 0] >= 2.5 - 1e-6) & (inner[:, 0] <= 3.5 + 1e-6))
    expected = DEPTH + eta(inner[:, 0], inner[:, 1], 0)
    assert np.abs(inner[:, 2] - expected).max() < 0.05  # the surface where phi = 0
    assert mesh is not None and mesh.n_triangles
    # a slice through the blocks: no cell of the volume under a block
    cut = volume.read_slice(0, 1, 2.0, ["level"])
    coarse = cut.points[cut.point_data["level"] == 0]
    under = (coarse[:, 0] > 2.0 + 1e-6) & (coarse[:, 0] < 4.0 - 1e-6) & (coarse[:, 2] > 1.5 + 1e-6)
    assert not under.any()
    assert set(np.unique(cut.point_data["level"])) == {0.0, 1.0, 2.0}
    # the outer surface and its colour range by level
    outside = volume.read(1, ["velocity", "level"])
    assert outside.n_triangles and "level" in outside.point_data
    assert set(volume.sample_values(1, "level")) == {0.0, 1.0, 2.0}


def test_outputs_without_blocks_and_the_block_outlines(tmp_path):
    root = cfd_amr_case(tmp_path, frames=3, refined=(0, 1))
    s = Session()
    case_id = s.open_case(root)
    case = s.data.case(case_id)
    volume = case.datasets["cfd_volume"]
    assert len(volume.frames[2].pieces) == 2  # no index for output 2: the volume alone
    plain = volume.read_iso(2, "phi", 0.0, ["level"])
    assert set(np.unique(plain.point_data["level"])) == {0.0}
    blocks = next(i for i in s.scene if i.kind == "amr_blocks")
    assert not blocks.visible  # switched off: a tick shows them
    outlines = s.data.load(blocks.id)
    assert outlines.segments.shape == (2 * 12, 2)
    assert sorted(set(outlines.point_data["level"])) == [1.0, 2.0]
    assert case.datasets["cfd_amr_blocks"].steps == [0, 1]


def test_the_free_surface_shows_its_grid_lines(tmp_path):
    root = cfd_amr_case(tmp_path)
    s = Session()
    s.open_case(root)
    free = next(i for i in s.scene if i.kind == "iso_surface")
    assert free.properties["grid_lines"] == "off"
    s.set_property(free.id, "grid_lines", "dark")
    mesh = s.data.load(free.id)
    pairs = mesh.outlines.astype(np.int64)
    assert len(pairs) and pairs.max() < mesh.n_points  # on the surface's own points
    ends = mesh.points[pairs]
    # every line lies in a face of a cell: one coordinate the same at both ends
    same = np.isclose(ends[:, 0], ends[:, 1], atol=1e-5)
    assert same.any(axis=1).all()
    # finer inside the blocks: the level 2 block's x faces every 0.25 m
    xs = np.unique(np.round(ends[:, :, 0][same[:, 0]][:, 0], 4))
    inner = xs[(xs > 2.5) & (xs < 3.5)]
    assert {2.75, 3.0, 3.25} <= set(inner.tolist())
    shown = s.data.display(free.id, mesh)
    lines = s.data.patch_outlines(free.id, mesh, shown)
    # bug: only under the surface (it faces the rising phi: down); now on both sides
    assert lines is not None and lines.segments.shape == (2 * len(pairs), 2, 3)
    on_surface = shown.positions[pairs][:, :, 2]
    above, below = lines.segments[: len(pairs), :, 2], lines.segments[len(pairs) :, :, 2]
    assert ((above - on_surface) * (below - on_surface) <= 1e-12).all()  # either side
    assert (np.maximum(above, below) > on_surface).mean() > 0.9  # one of them above


def test_the_blocks_in_a_lagoon_store(tmp_path):
    """lagoon-convert puts the blocks into the store (an AMR set of 3D grids, as
    REEF3D does with P 18); the case opens from the store alone just the same."""
    from lagoon_format.store import Store

    from lagoonviz.core.convert import convert

    root = cfd_amr_case(tmp_path)
    s = Session()
    from_files = s.data.case(s.open_case(root)).datasets["cfd_volume"]
    expected = from_files.read_iso(1, "phi", 0.0, ["level"])
    convert(root)
    store = Store(root / "REEF3D_CFD.lagoon")
    blocks = store.amr["cfd_amr"]
    assert blocks.dims == 3 and len(blocks) == 3
    assert blocks.fields["velocity"][0] == 3 and len(blocks.grids(0)) == 2
    # the store alone: no VTU, no AMR folder
    shutil.rmtree(root / "REEF3D_CFD_VTU")
    shutil.rmtree(root / "REEF3D_CFD_AMR")
    s = Session()
    case = s.data.case(s.open_case(root))
    volume = case.datasets["cfd_volume"]
    assert type(volume).__name__ == "RefinedVolume"
    assert type(volume.blocks).__name__ == "StoreBlocks"
    got = volume.read_iso(1, "phi", 0.0, ["level"])
    assert set(np.unique(got.point_data["level"])) == {0.0, 1.0, 2.0}
    inner = got.points[got.point_data["level"] == 2]
    assert len(inner) == (expected.point_data["level"] == 2).sum()
    outlines = s.data.load(next(i.id for i in s.scene if i.kind == "amr_blocks"))
    assert outlines.segments.shape == (24, 2)
    # the blocks' own times from the store (REEF3D writes the volume after them: its
    # times lag one output, and the timeline must not fall back to P 30 x output)
    own = case.datasets["cfd_amr_blocks"]
    np.testing.assert_allclose(own.times, blocks.times)
    stored = zip(blocks.steps.tolist(), blocks.times.tolist(), strict=True)
    assert volume.blocks.times == dict(stored)
