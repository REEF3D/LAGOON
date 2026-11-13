# Architect: Hans Bihs
"""The free surface of a VOF-PLIC run (F 80 4). REEF3D now prints phi as the distance to
the PLIC planes, so phi = 0 is the free surface as for the level set. Before, phi was
the solver's phase marker (+1 water, -1 air, (vof - 0.5) dz in the interface cells),
whose zero lies at the cell centres: such a run's free surface is drawn at VOF = 0.5.
The water-side colours of a PLIC run come from half a cell below the surface (no F 45
band)."""

import shutil
from pathlib import Path

import numpy as np
from reef3d_fixtures import DEPTH, make_run

from lagoonviz.core import Session
from lagoonviz.core.data.reef3d import _interface_cells, plic_case, plic_marker_phi


def _cfd(tmp_path: Path, plic: bool) -> Path:
    root = make_run(tmp_path / "cfd", solver="CFD", frames=2, print_interval=0.5, plic=plic)
    for name in ("REEF3D_CFD_VTP_FSF", "REEF3D_CFD_VTP_BED"):  # CFD writes a volume only
        shutil.rmtree(root / name)
    return root


def test_a_plic_run_with_the_phase_marker_shows_vof_at_one_half(tmp_path):
    root = _cfd(tmp_path, plic=True)
    s = Session()
    s.open_case(root)
    free = next(i for i in s.scene if i.kind == "iso_surface")
    assert free.name == "Free surface"
    assert free.properties["iso_by"] == "VOF" and free.properties["iso_value"] == 0.5
    mesh = s.data.load(free.id)
    assert mesh is not None and mesh.n_points
    # VOF = 0.5 on the points is the surface (DEPTH + eta, |eta| <= 0.18)
    assert abs(float(np.median(mesh.points[:, 2])) - DEPTH) < 0.2


def test_a_level_set_run_keeps_phi(tmp_path):
    root = _cfd(tmp_path, plic=False)
    s = Session()
    s.open_case(root)
    free = next(i for i in s.scene if i.kind == "iso_surface")
    assert free.properties["iso_by"] == "phi" and free.properties["iso_value"] == 0.0


class _Fake:
    def __init__(self, folder, phi, components=("phi", "VOF")):
        self.folder = folder
        self.components = {name: 1 for name in components}
        self._phi = np.asarray(phi, np.float32)

    def sample_values(self, index, variable):
        return self._phi


def test_the_marker_is_told_from_a_distance(tmp_path):
    case = tmp_path / "case"
    case.mkdir()
    (case / "ctrl.txt").write_text("F 80 4\nF 45 2.1\n")
    marker = np.r_[np.ones(80), -np.ones(80), np.linspace(-0.01, 0.01, 20)]
    distance = np.linspace(-0.8, 1.2, 180)  # the distance REEF3D prints now
    assert plic_case(case)
    assert plic_marker_phi(_Fake(case, marker))
    assert not plic_marker_phi(_Fake(case, distance))
    assert not plic_marker_phi(_Fake(case, marker, components=("phi",)))  # no VOF printed
    level_set = tmp_path / "ls"
    level_set.mkdir()
    (level_set / "ctrl.txt").write_text("F 30 3\n")
    assert not plic_case(level_set)
    assert not plic_marker_phi(_Fake(level_set, marker))


def test_the_water_side_of_a_plic_run_is_half_a_cell(tmp_path):
    case = tmp_path / "case"
    case.mkdir()
    (case / "ctrl.txt").write_text("F 45 1.6\nF 80 4\n")
    assert _interface_cells(case) == 0.5
    store_group = case / "REEF3D_CFD.lagoon" / "volume"  # found from a store's output too
    store_group.mkdir(parents=True)
    assert _interface_cells(store_group) == 0.5
