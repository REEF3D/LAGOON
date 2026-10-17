"""A CFD case folder: its free surface is the level set phi at 0 (an iso-surface of
the volume), its floating bodies are shown, and CFD's 3D AMR blocks do not keep the
case from opening."""

import shutil
from pathlib import Path

import numpy as np
from reef3d_fixtures import DEPTH, make_run

from lagoonviz.core import Session

VTR = """<?xml version="1.0"?>
<VTKFile type="RectilinearGrid" version="0.1" byte_order="LittleEndian">
<RectilinearGrid WholeExtent="0 2 0 1 0 2">
<Piece Extent="0 2 0 1 0 2">
<CellData Scalars="phi">
<DataArray type="Float32" Name="phi" format="ascii">1 1 1 1</DataArray>
</CellData>
<Coordinates>
<DataArray type="Float32" Name="x" format="ascii">0 1 2</DataArray>
<DataArray type="Float32" Name="y" format="ascii">0 1</DataArray>
<DataArray type="Float32" Name="z" format="ascii">0 1 2</DataArray>
</Coordinates>
</Piece>
</RectilinearGrid>
</VTKFile>
"""


def cfd_case(tmp_path):
    root = make_run(tmp_path / "cfd", solver="CFD", frames=2, print_interval=0.5, objects=True)
    for name in ("REEF3D_CFD_VTP_FSF", "REEF3D_CFD_VTP_BED"):  # CFD writes a volume only
        shutil.rmtree(root / name)
    return root


def test_a_cfd_case_shows_its_free_surface_and_bodies(tmp_path):
    root = cfd_case(tmp_path)
    s = Session()
    s.open_case(root)
    kinds = {i.kind: i for i in s.scene}
    assert not kinds["volume"].visible  # there, switched off
    free = kinds["iso_surface"]
    assert free.name == "Free surface" and free.visible
    assert free.properties["iso_by"] == "phi" and free.properties["iso_value"] == 0.0
    assert free.properties["variable"] == "velocity"
    mesh = s.data.load(free.id)
    assert mesh is not None and mesh.n_points
    # the level set is DEPTH + eta - z: the surface lies about DEPTH high
    assert abs(float(np.median(mesh.points[:, 2])) - DEPTH) < 0.5
    assert "body" in kinds  # the floating body of the run


def test_cfd_amr_blocks_do_not_keep_the_case_from_opening(tmp_path):
    root = cfd_case(tmp_path)
    amr = root / "REEF3D_CFD_AMR"
    amr.mkdir()
    (amr / "REEF3D-CFD-AMR-00000000-00000.vtr").write_text(VTR)
    (amr / "REEF3D-CFD-AMR-00000000.vtm").write_text(
        '<?xml version="1.0"?>\n<VTKFile type="vtkMultiBlockDataSet" version="1.0">\n'
        '<vtkMultiBlockDataSet>\n<DataSet index="0" file="REEF3D-CFD-AMR-00000000-00000.vtr"/>\n'
        "</vtkMultiBlockDataSet>\n</VTKFile>\n"
    )
    s = Session()
    case_id = s.open_case(root)
    case = s.data.case(case_id)
    assert "cfd_volume" in case.datasets and not [d for d in case.datasets.values()
                                                  if d.role == "amr"]  # fmt: skip
    assert [i for i in s.scene if i.kind == "iso_surface"]


def test_the_solver_of_ctrl_txt_is_shown_next_to_older_runs(tmp_path):
    """Bug: a CFD run going on in a folder with an older NHFLOW (or FNPF) run showed
    the NHFLOW free surface, bed and volume: CFD writes no free surface, and any
    free surface won. ctrl.txt's solver (A 10) is shown when it wrote anything."""
    root = cfd_case(tmp_path)
    make_run(root, solver="NHFLOW", frames=3)  # the older run beside it
    ctrl = root / "ctrl.txt"
    ctrl.write_text("A 10 6\n" + ctrl.read_text())
    s = Session()
    s.open_case(root)
    assert s.data.case("case-1").ctrl_solver == "CFD"
    shown = {i.kind: s.data.dataset(i.id) for i in s.scene if s.data.dataset(i.id) is not None}
    assert "free_surface" not in shown and "bathymetry" not in shown
    assert shown["iso_surface"].solver == "CFD" and shown["volume"].solver == "CFD"
    assert "nhflow_volume" in s.data.choices("volume-1", "datasets")  # still there
    # ctrl.txt set up for NHFLOW: the NHFLOW run
    ctrl.write_text(ctrl.read_text().replace("A 10 6", "A 10 5"))
    s = Session()
    s.open_case(root)
    kinds = {i.kind: s.data.dataset(i.id) for i in s.scene if s.data.dataset(i.id) is not None}
    assert kinds["free_surface"].solver == "NHFLOW"
    # a solver that wrote nothing yet: as before, every run is looked at
    ctrl.write_text(ctrl.read_text().replace("A 10 5", "A 10 3"))
    s = Session()
    s.open_case(root)
    assert any(i.kind == "free_surface" for i in s.scene)


def test_the_store_of_the_run_going_on_beats_the_files_of_the_run_before():
    """Bug: REEF3D's store of a CFD run (P 18 1) names the output its bodies stand for
    (cfd_body); the 6DOF files of the run before had more outputs, so the store's
    bodies were left out and the run showed no floating body."""
    from types import SimpleNamespace

    from lagoonviz.core.data.reef3d import _prefer_stores

    def dataset(key, run, steps, recorded=None):
        return SimpleNamespace(
            key=key, role="body", solver="CFD", run_id=run, earlier_run=False,
            last_step=steps, recorded=recorded, folder=Path("store/bodies"),
        )  # fmt: skip

    files = dataset("cfd_body", "run-before", 13)
    store = dataset("cfd_body", "run-now", 4, recorded="cfd_body")
    kept = _prefer_stores([files], [store])
    assert store in kept and files in kept  # both runs, the store's is the run going on
    # the files of the same run, further on than the store: the files
    same_run = dataset("cfd_body", "run-now", 6)
    assert _prefer_stores([same_run], [store]) == [same_run]
    # converted from these files and complete: the store replaces them
    older = dataset("cfd_body", "run-now", 3)
    assert _prefer_stores([older], [store]) == [store]
