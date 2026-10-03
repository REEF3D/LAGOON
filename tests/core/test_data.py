"""Readers, merging of MPI pieces, case discovery, caching and background loading."""

import threading

import numpy as np
import pytest
from reef3d_fixtures import DEPTH, NX, NY, eta, make_run, write_piece

from lagoonviz.core import Session
from lagoonviz.core.colormaps import COLORMAPS, colors, lut
from lagoonviz.core.data.manager import BackgroundLoader, FrameCache, auto_range
from lagoonviz.core.data.mesh import PieceMerger, boundary_faces, scalar_values, triangulate
from lagoonviz.core.data.reef3d import open_case
from lagoonviz.core.data.vtkxml import (
    VTKFormatError,
    read_header,
    read_parallel_header,
    read_piece,
    split_name,
)

QUAD = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]], np.float32)


# ----------------------------------------------------------------- vtk xml
@pytest.mark.parametrize(
    "fmt, compressed", [("appended", False), ("binary", False), ("binary", True), ("ascii", False)]
)
def test_piece_roundtrip_all_formats(tmp_path, fmt, compressed):
    data = {
        "eta": np.array([0.1, 0.2, 0.3, 0.4], np.float32),
        "u": np.arange(12, dtype=np.float32).reshape(4, 3),
    }
    path = tmp_path / "quad.vtp"
    write_piece(
        path,
        "PolyData",
        QUAD,
        data,
        np.array([0, 1, 2, 3]),
        np.array([4]),
        fmt=fmt,
        compressed=compressed,
    )
    piece = read_piece(path)
    np.testing.assert_array_equal(piece.points, QUAD)
    np.testing.assert_array_equal(piece.point_data["eta"], data["eta"])
    np.testing.assert_array_equal(piece.point_data["u"], data["u"])
    np.testing.assert_array_equal(piece.connectivity, [0, 1, 2, 3])
    assert triangulate(piece.connectivity, piece.offsets).tolist() == [[0, 1, 2], [0, 2, 3]]


def test_reads_only_requested_arrays(tmp_path):
    path = tmp_path / "quad.vtp"
    write_piece(
        path,
        "PolyData",
        QUAD,
        {"a": np.ones(4, np.float32), "b": np.zeros(4, np.float32)},
        np.array([0, 1, 2]),
        np.array([3]),
    )
    piece = read_piece(path, point_arrays=["b"], with_cells=False)
    assert list(piece.point_data) == ["b"]
    assert piece.connectivity.size == 0
    with pytest.raises(KeyError):
        read_piece(path, point_arrays=["missing"])


def test_tolerates_oversized_block_header(tmp_path):
    """REEF3D::FNPF writes a vector-sized header for its scalar 'elevation' array."""
    path = tmp_path / "fnpf.vtp"
    elevation = np.array([1.0, 2.0, 3.0, 4.0], np.float32)
    write_piece(
        path,
        "PolyData",
        QUAD,
        {"elevation": elevation, "after": elevation * 2},
        np.array([0, 1, 2]),
        np.array([3]),
        bad_count="elevation",
    )
    piece = read_piece(path)
    np.testing.assert_array_equal(piece.point_data["elevation"], elevation)
    np.testing.assert_array_equal(piece.point_data["after"], elevation * 2)


def test_truncated_file_is_an_error(tmp_path):
    path = tmp_path / "quad.vtp"
    write_piece(
        path, "PolyData", QUAD, {"eta": np.ones(4, np.float32)}, np.array([0, 1, 2]), np.array([3])
    )
    raw = path.read_bytes()
    path.write_bytes(raw[: len(raw) // 2] + b"</AppendedData></VTKFile>")
    with pytest.raises(VTKFormatError):
        read_piece(path)


def test_not_a_vtk_file(tmp_path):
    path = tmp_path / "x.vtu"
    path.write_text("hello")
    with pytest.raises(VTKFormatError):
        read_header(path)


def test_split_name():
    assert split_name("REEF3D-NHFLOW-FSF-00000696-000001.vtp") == ("REEF3D-NHFLOW-FSF", 696, 1)
    assert split_name("REEF3D-NHFLOW-00000012.pvtu") == ("REEF3D-NHFLOW", 12, None)
    assert split_name("notes.vtu") == ("notes", None, None)


# ----------------------------------------------------------------- geometry
def test_boundary_of_hex_block():
    # 2 x 1 x 1 hexahedra: 10 outer quads -> 20 triangles, the shared face is inside
    nodes = np.array([[0, 1, 4, 3, 6, 7, 10, 9], [1, 2, 5, 4, 7, 8, 11, 10]])
    tris = boundary_faces(nodes.ravel(), np.array([8, 16]), np.array([12, 12]))
    assert tris.shape == (20, 3)
    shared = {1, 4, 7, 10}
    assert not any(set(t) <= shared for t in tris.tolist())


def test_scalar_values():
    data = {"u": np.array([[3, 4, 0], [0, 0, 1]], np.float32), "p": np.array([1, 2], np.float32)}
    np.testing.assert_allclose(scalar_values(data, "u"), [5, 1])
    np.testing.assert_allclose(scalar_values(data, "u.y"), [4, 0])
    assert scalar_values(data, "none") is None
    with pytest.raises(KeyError):
        scalar_values(data, "phi")


# ----------------------------------------------------------------- cases
def test_open_case_finds_datasets(reef3d_run):
    case = open_case(reef3d_run)
    assert set(case.datasets) == {"nhflow_free_surface", "nhflow_bed", "nhflow_volume"}
    fsf = case.by_role("free_surface")
    assert fsf.steps == [0, 1, 2]
    assert fsf.variables() == ["velocity", "velocity.x", "velocity.y", "velocity.z", "eta"]
    assert fsf.default_variable() == "eta"
    assert case.print_interval is None


def test_open_output_folder_or_file(reef3d_run):
    folder = reef3d_run / "REEF3D_NHFLOW_VTP_FSF"
    assert list(open_case(folder).datasets) == ["nhflow_free_surface"]
    one = open_case(folder / "REEF3D-NHFLOW-FSF-00000001.pvtp")
    assert list(one.datasets) == ["nhflow_free_surface"]
    assert one.path == reef3d_run
    with pytest.raises(FileNotFoundError):
        open_case(reef3d_run / "missing")
    empty = reef3d_run.parent / "empty"
    empty.mkdir()
    with pytest.raises(ValueError):
        open_case(empty)


def test_free_surface_frames_merge_ranks(reef3d_run):
    fsf = open_case(reef3d_run).by_role("free_surface")
    for index in range(3):
        mesh = fsf.read(index, ["eta"])
        assert mesh.n_points == NX * NY  # shared boundary points merged
        assert mesh.n_triangles == 2 * (NX - 1) * (NY - 1)
        x, y, z = mesh.points.T
        np.testing.assert_allclose(mesh.point_data["eta"], eta(x, y, index), atol=1e-6)
        np.testing.assert_allclose(z, DEPTH + mesh.point_data["eta"], atol=1e-5)
        assert list(mesh.point_data) == ["eta"]
        assert mesh.triangles.max() < mesh.n_points


def test_volume_outer_surface(reef3d_run):
    volume = open_case(reef3d_run).by_role("volume")
    mesh = volume.read(0, ["velocity"])
    nz = 3
    # only the points of the outer surface are kept
    assert mesh.n_points == NX * NY * nz - (NX - 2) * (NY - 2) * (nz - 2)
    assert mesh.triangles.max() == mesh.n_points - 1
    # outer quads of an (NX-1) x (NY-1) x (nz-1) block, two triangles each
    cx, cy, cz = NX - 1, NY - 1, nz - 1
    assert mesh.n_triangles == 2 * 2 * (cx * cy + cx * cz + cy * cz)


def test_merger_reuses_plan_without_cells(reef3d_run):
    fsf = open_case(reef3d_run).by_role("free_surface")
    merger = PieceMerger("xy")
    full = [read_piece(p) for p in fsf.frames[0].pieces]
    merger.merge(full)
    light = [read_piece(p, ["eta"], with_cells=False) for p in fsf.frames[1].pieces]
    assert merger.accepts(light)
    merged = merger.merge(light)
    assert merged.points.shape[0] == NX * NY


def test_parallel_header(reef3d_run):
    header = read_parallel_header(reef3d_run / "REEF3D_NHFLOW_VTU" / "REEF3D-NHFLOW-00000000.pvtu")
    assert header.kind == "UnstructuredGrid"
    assert len(header.pieces) == 2
    assert header.point_data == {"velocity": 3, "pressure": 1}


# ----------------------------------------------------------------- session and data manager
def test_session_open_case_items_and_replay(reef3d_run):
    s = Session()
    case_id = s.open_case(reef3d_run)
    kinds = [item.kind for item in s.scene]
    assert kinds == ["case", "bathymetry", "free_surface"]
    assert s.scene.get("free_surface-1").properties["case"] == case_id
    assert s.scene.get("bathymetry-1").properties["variable"] == "elevation"
    assert s.stack.undo_label == "Open run"
    s.undo()
    assert len(s.scene) == 0
    s.redo()
    assert len(s.scene) == 3
    replayed = Session.from_script(s.to_script())
    assert replayed.scene.to_dict() == s.scene.to_dict()


def test_time_and_frames(reef3d_run):
    s = Session()
    s.open_case(reef3d_run)
    assert s.data.time_range() == (0.0, 2.0, "frame")
    s.set_time(1.4)
    assert s.data.frame_index("free_surface-1") == 1
    s.set_time(99)
    assert s.data.frame_index("free_surface-1") == 2


def test_print_interval_from_ctrl(tmp_path):
    run = make_run(tmp_path / "timed", print_interval=0.5, volume=False)
    s = Session()
    s.open_case(run)
    assert s.data.time_range() == (0.0, 1.0, "s")
    s.set_time(0.5)
    assert s.data.frame_index("free_surface-1") == 1
    s.set_property("case-1", "print_interval", 0.25)
    assert s.data.time_range() == (0.0, 0.5, "s")


def test_display_exaggeration_and_range(reef3d_run):
    s = Session()
    s.open_case(reef3d_run)
    mesh = s.data.load("free_surface-1")
    shown = s.data.display("free_surface-1", mesh)
    assert shown.variable == "eta" and shown.colormap == "balance"
    assert shown.vmin == pytest.approx(-shown.vmax)  # eta changes sign: symmetric
    assert s.data.still_water_level("free_surface-1", mesh) == pytest.approx(DEPTH, abs=1e-5)
    s.set_property("free_surface-1", "exaggeration", 10.0)
    big = s.data.display("free_surface-1", mesh)
    np.testing.assert_allclose(
        big.positions[:, 2] - DEPTH, 10 * (mesh.points[:, 2] - DEPTH), atol=1e-4
    )
    # the automatic range is kept while playing
    s.set_time(2)
    later = s.data.display("free_surface-1", s.data.load("free_surface-1"))
    assert (later.vmin, later.vmax) == (shown.vmin, shown.vmax)
    low, high = s.fit_range("free_surface-1")
    props = s.scene.get("free_surface-1").properties
    assert props["auto_range"] is False and props["vmax"] == pytest.approx(high, abs=1e-5)
    assert s.stack.undo_label == "Set auto_range, vmin, vmax"


def test_variable_none_and_choices(reef3d_run):
    s = Session()
    s.open_case(reef3d_run)
    assert s.data.variable_choices("bathymetry-1") == ["none", "elevation", "depth"]
    s.set_property("bathymetry-1", "variable", "none")
    shown = s.data.display("bathymetry-1", s.data.load("bathymetry-1"))
    assert shown.values is None and shown.variable == "none"
    assert s.data.choices("free_surface-1", "datasets") == [
        "",
        "nhflow_bed",
        "nhflow_free_surface",
        "nhflow_volume",
    ]


def test_volume_item(reef3d_run):
    s = Session()
    s.open_case(reef3d_run)
    vol = s.add_item("volume")
    mesh = s.data.load(vol)
    assert list(mesh.point_data) == ["velocity"]


def test_cache_and_budget(reef3d_run):
    s = Session()
    s.open_case(reef3d_run)
    s.data.load("free_surface-1")
    s.data.load("free_surface-1")
    assert s.data.cache.hits == 1
    small = FrameCache(max_bytes=1)
    small.put("a", s.data.load("free_surface-1"))
    small.put("b", s.data.load("free_surface-1"))
    assert len(small) == 1 and "b" in small


def test_missing_case_reports_error(tmp_path):
    s = Session()
    s.add_item("case", path=str(tmp_path / "nowhere"))
    fs = s.add_item("free_surface")
    assert s.data.load(fs) is None
    assert "does not exist" in s.data.error(fs)


def test_background_loader(reef3d_run):
    s = Session()
    s.open_case(reef3d_run)
    loader = BackgroundLoader(s.data)
    got = []
    done = threading.Event()

    def callback(key, mesh, error):
        got.append((key.index, mesh.n_points, error))
        done.set()

    key = s.data.frame_key("free_surface-1", 2)
    loader.request(key, callback)
    assert done.wait(5)
    assert got == [(2, NX * NY, None)]
    loader.prefetch(["free_surface-1"], 0, frames_ahead=2)
    for future in list(loader._pending.values()):
        future.result(5)
    assert s.data.frame_key("free_surface-1", 1) in s.data.cache
    loader.shutdown()


# ----------------------------------------------------------------- colour
def test_colormaps():
    for name in COLORMAPS:
        table = lut(name)
        assert table.shape == (256, 4) and table.min() >= 0 and table.max() <= 1
    rgba = colors(np.array([-1.0, 0.0, 1.0]), "greys", -1, 1)
    np.testing.assert_allclose(rgba[0, :3], [17 / 255] * 3, atol=1e-6)
    np.testing.assert_allclose(rgba[2, :3], [1, 1, 1])


def test_auto_range():
    assert auto_range(None) == (0.0, 1.0)
    low, high = auto_range(np.linspace(-2, 1, 1001))
    assert low == -high
    assert auto_range(np.full(10, 3.0)) == (2.5, 3.5)


def test_exaggerated_normals_match_recomputed(reef3d_run):
    from lagoonviz.core.data.mesh import vertex_normals

    s = Session()
    s.open_case(reef3d_run)
    s.set_property("free_surface-1", "exaggeration", 20.0)
    shown = s.data.display("free_surface-1", s.data.load("free_surface-1"))
    direct = vertex_normals(shown.positions, shown.triangles)
    # interior points: exact up to the area weighting of neighbouring triangles
    assert np.mean(np.sum(shown.normals * direct, axis=1)) > 0.99


def test_parallel_reads_of_one_dataset(tmp_path):
    """Worker threads read frames of the same dataset at the same time."""
    from concurrent.futures import ThreadPoolExecutor

    run = make_run(tmp_path / "many", frames=12, ranks=3, volume=False)
    fsf = open_case(run).by_role("free_surface")
    with ThreadPoolExecutor(4) as pool:
        meshes = list(pool.map(lambda i: fsf.read(i, ["eta"]), range(12)))
    for index, mesh in enumerate(meshes):
        x, y, _ = mesh.points.T
        np.testing.assert_allclose(mesh.point_data["eta"], eta(x, y, index), atol=1e-6)
        assert mesh.n_points == NX * NY


def _age(folder, seconds_ago):
    import os
    import time

    stamp = time.time() - seconds_ago
    for path in folder.iterdir():
        os.utime(path, (stamp, stamp))


def test_folder_with_several_runs_shows_the_latest(tmp_path):
    """Like a home folder where NHFLOW ran last week and FNPF today."""
    root = tmp_path / "home"
    make_run(root, solver="NHFLOW", volume=False)
    make_run(root, solver="FNPF", volume=False)
    for folder in root.glob("REEF3D_NHFLOW_*"):
        _age(folder, 7 * 86400)
    _age(root / "REEF3D_FNPF_VTP_BED", 3 * 86400)  # a bed left from an older FNPF run
    s = Session()
    s.open_case(root)
    shown = {i.kind: i.properties.get("dataset") for i in s.scene if i.kind != "case"}
    assert shown == {"free_surface": "fnpf_free_surface"}  # the old bed is not mixed in
    assert "nhflow_free_surface" in s.data.dataset_choices("free_surface-1")


def test_hiding_the_case_hides_its_items(reef3d_run):
    s = Session()
    s.open_case(reef3d_run)
    mesh = s.data.load("free_surface-1")
    assert s.data.display("free_surface-1", mesh).visible
    s.set_visible("case-1", False)
    assert not s.data.display("free_surface-1", mesh).visible


def test_outputs_left_from_an_earlier_run_are_split_off(tmp_path):
    """REEF3D restarted in the same folder overwrote outputs 0 and 1 only."""
    import os
    import time

    run = make_run(tmp_path / "rerun", frames=5, volume=False)
    folder = run / "REEF3D_NHFLOW_VTP_FSF"
    now = time.time()
    for path in folder.iterdir():
        step = int(path.name.split("-")[3][:8])
        stamp = now - 3600 + step if step >= 2 else now + step  # 2..4 are an hour older
        os.utime(path, (stamp, stamp))
    case = open_case(folder)
    assert sorted(case.datasets) == ["nhflow_free_surface", "nhflow_free_surface_earlier1"]
    latest = case.datasets["nhflow_free_surface"]
    earlier = case.datasets["nhflow_free_surface_earlier1"]
    assert latest.steps == [0, 1] and earlier.steps == [2, 3, 4]
    assert earlier.label == "NHFLOW free surface (earlier run, outputs 2–4)"
    assert case.by_role("free_surface") is latest
    assert case.n_frames == 5
    # the timeline counts output numbers: output 3 is the second frame of the earlier run
    assert earlier.frame_index(3) == 1 and earlier.frame_index(0) == 0
    assert latest.frame_index(4) == 1


def test_a_short_restart_does_not_hide_the_long_run(tmp_path):
    import os
    import time

    run = make_run(tmp_path / "aborted", frames=30, volume=False)
    folder = run / "REEF3D_NHFLOW_VTP_FSF"
    now = time.time()
    for path in folder.iterdir():
        step = int(path.name.split("-")[3][:8])
        stamp = now + step if step < 2 else now - 3600 + step
        os.utime(path, (stamp, stamp))
    case = open_case(folder)
    assert case.by_role("free_surface").steps == list(range(2, 30))


def test_probe_values_on_a_surface(reef3d_run):
    s = Session()
    s.open_case(reef3d_run)
    mesh = s.data.load("free_surface-1")
    triangle = 7
    probe = s.data.probe("free_surface-1", triangle, (0.2, 0.3, 0.5))
    nodes = mesh.triangles[triangle].astype(int)
    expected = mesh.points[nodes].T @ np.array([0.2, 0.3, 0.5])
    np.testing.assert_allclose(probe.point, expected, rtol=1e-6)
    assert probe.values["eta"] == pytest.approx(
        float(mesh.point_data["eta"][nodes] @ np.array([0.2, 0.3, 0.5])), rel=1e-5
    )
    assert probe.summary().startswith("Free surface · eta = ")
    assert s.data.probe("free_surface-1", 10**9, (1, 0, 0)) is None


def test_probe_vector_components(reef3d_run):
    from lagoonviz.core.data.mesh import SurfaceMesh

    mesh = SurfaceMesh(
        np.eye(3, dtype=np.float32),
        np.array([[0, 1, 2]], np.uint32),
        {"u": np.array([[3, 4, 0], [3, 4, 0], [3, 4, 0]], np.float32)},
    )
    _, values = mesh.sample(0, (1, 1, 1))
    assert values == {"u": 5.0, "u.x": 3.0, "u.y": 4.0, "u.z": 0.0}
