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
    assert s.scene.get("bathymetry-1").properties["variable"] == "none"  # plain sand
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
    assert shown.variable == "eta" and shown.colormap == "reef3d"
    assert shown.vmin == pytest.approx(-shown.vmax)  # eta changes sign: symmetric
    assert s.data.still_water_level("free_surface-1", mesh) == pytest.approx(DEPTH, abs=1e-5)
    s.set_property("case-1", "exaggeration", 10.0)  # synced: applies to every item
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
    np.testing.assert_allclose(rgba[0, :3], [0x1C / 255] * 3, atol=1e-6)
    np.testing.assert_allclose(rgba[2, :3], [0xC8 / 255] * 3, atol=1e-6)  # never paper white


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
    # the bed rarely changes between runs: the solver's bed is shown with the new surface,
    # never the bed of another solver
    assert shown == {"bathymetry": "fnpf_bed", "free_surface": "fnpf_free_surface"}
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


def test_dry_land_is_left_out_of_the_free_surface(reef3d_run):
    from lagoonviz.core.data.manager import _wet_triangles
    from lagoonviz.core.data.mesh import SurfaceMesh

    points = np.zeros((4, 3), np.float32)
    triangles = np.array([[0, 1, 2], [1, 2, 3]], np.uint32)
    dry = SurfaceMesh(points, triangles, {"depth": np.array([0.0, -1.0, 0.0, 0.5])})
    kept = _wet_triangles(dry)
    assert kept.shape == triangles.shape  # same length: the GPU buffer is reused
    np.testing.assert_array_equal(kept, [[0, 0, 0], [1, 2, 3]])
    wet = SurfaceMesh(points, triangles, {"depth": np.ones(4)})
    assert _wet_triangles(wet) is triangles  # nothing dry: no copy

    s = Session()
    s.open_case(reef3d_run)
    mesh = s.data.load("free_surface-1")
    shown = s.data.display("free_surface-1", mesh)
    assert shown.triangles is mesh.triangles  # this run has no dry land


def test_a_run_that_starts_later_is_hidden_before_and_opened_at_its_start(tmp_path):
    """Outputs 0-1 were overwritten by a restart; the long run starts at output 2."""
    import os
    import time

    run = make_run(tmp_path / "restarted", frames=30, volume=False)
    folder = run / "REEF3D_NHFLOW_VTP_FSF"
    now = time.time()
    for path in folder.iterdir():
        step = int(path.name.split("-")[3][:8])
        stamp = now + step if step < 2 else now - 3600 + step
        os.utime(path, (stamp, stamp))
    s = Session()
    s.open_case(folder)
    assert s.scene.time == 2.0  # opened where the shown run has data
    assert s.data.frame_index("free_surface-1") == 0
    s.set_time(0.0)
    assert s.data.frame_index("free_surface-1") is None  # no frame from another run
    assert s.data.frame_key("free_surface-1") is None


def test_bed_exaggeration_about_still_water_level(reef3d_run):
    from reef3d_fixtures import DEPTH

    s = Session()
    s.open_case(reef3d_run)
    bed = next(i.id for i in s.scene if i.kind == "bathymetry")
    mesh = s.data.load(bed)
    assert "depth" in mesh.point_data
    s.set_property(bed, "exaggeration", 3.0)  # ignored while the case is synced
    shown = s.data.display(bed, mesh)
    np.testing.assert_allclose(shown.positions[:, 2], mesh.points[:, 2], rtol=1e-6)
    s.set_property("case-1", "sync_exaggeration", False)
    shown = s.data.display(bed, mesh)
    np.testing.assert_allclose(
        shown.positions[:, 2], DEPTH + 3.0 * (mesh.points[:, 2] - DEPTH), rtol=1e-5
    )


def test_case_exaggeration_applies_to_all_items(reef3d_run):
    from reef3d_fixtures import DEPTH

    s = Session()
    s.open_case(reef3d_run)
    s.set_property("case-1", "exaggeration", 5.0)
    bed = next(i.id for i in s.scene if i.kind == "bathymetry")
    for item in ("free_surface-1", bed):
        assert s.data.exaggeration(item) == 5.0
        mesh = s.data.load(item)
        shown = s.data.display(item, mesh)
        level = s.data.case_water_level(item)
        assert level == pytest.approx(DEPTH, abs=1e-4)
        np.testing.assert_allclose(
            shown.positions[:, 2] - DEPTH, 5.0 * (mesh.points[:, 2] - DEPTH), atol=1e-3
        )


def test_loader_does_not_deadlock_on_frames_that_finish_at_once():
    """A future done before its callback is added runs the callback in place."""
    import threading

    from lagoonviz.core.data.manager import BackgroundLoader

    class Instant:
        def load_key(self, key):
            return key

    loader = BackgroundLoader(Instant(), workers=2)
    finished = threading.Event()

    def hammer():
        futures = [loader.request(("frame", i % 3)) for i in range(3000)]
        for future in futures:
            future.result(timeout=5)
        finished.set()

    worker = threading.Thread(target=hammer, daemon=True)
    worker.start()
    assert finished.wait(timeout=20), "loader deadlocked"
    loader.shutdown()


def test_colour_map_and_colour_by_are_independent(reef3d_run):
    """Choosing a colour map never changes what is coloured, and the other way round."""
    s = Session()
    s.open_case(reef3d_run)
    bed = next(i.id for i in s.scene if i.kind == "bathymetry")
    assert s.scene.get(bed).properties["variable"] == "none"  # plain sand at first
    s.set_property(bed, "colormap", "rainbow")
    props = s.scene.get(bed).properties
    assert (props["colormap"], props["variable"]) == ("rainbow", "none")
    s.set_property(bed, "variable", "depth")
    shown = s.data.display(bed, s.data.load(bed))
    assert (shown.variable, shown.colormap) == ("depth", "rainbow")
    s.set_property(bed, "colormap", "greys")
    assert s.scene.get(bed).properties["variable"] == "depth"
    s.undo()
    props = s.scene.get(bed).properties
    assert (props["colormap"], props["variable"]) == ("rainbow", "depth")


def test_reload_picks_up_new_outputs(tmp_path):
    """A running simulation writes more outputs; reloading shows them."""
    import shutil

    from reef3d_fixtures import make_run

    full = make_run(tmp_path / "full", frames=5, volume=False)
    running = tmp_path / "running"
    shutil.copytree(full, running)
    folder = running / "REEF3D_NHFLOW_VTP_FSF"
    for written in (folder, running / "REEF3D_NHFLOW_VTP_BED"):
        for path in written.glob("*0000000[34]*"):
            path.unlink()  # outputs 3 and 4 not written yet
    s = Session()
    s.open_case(running)
    events = []
    s.scene.subscribe(lambda event, item: events.append(event))
    assert s.data.n_frames() == 3
    s.data.load("free_surface-1")
    before = s.data.signature("case-1")
    for path in (full / "REEF3D_NHFLOW_VTP_FSF").glob("*0000000[34]*"):
        shutil.copy(path, folder)
    import os

    os.utime(folder, (os.path.getmtime(folder) + 5,) * 2)
    assert s.data.signature("case-1") != before
    assert s.reload() == [str(running.resolve())]
    assert s.data.n_frames() == 5 and events[-1] == "reset"
    assert len(s.data.cache) == 0  # frames read before are read again
    s.set_time(4)
    assert s.data.load("free_surface-1") is not None
    assert [i.kind for i in s.scene].count("free_surface") == 1  # items kept


def test_switching_the_automatic_range_off_keeps_the_colours(reef3d_run):
    s = Session()
    s.open_case(reef3d_run)
    shown = s.data.automatic_range("free_surface-1")
    assert shown is not None and shown[0] < 0 < shown[1]
    s.set_property("free_surface-1", "auto_range", False)
    props = s.scene.get("free_surface-1").properties
    assert (props["vmin"], props["vmax"]) == pytest.approx(shown, rel=1e-5)
    s.undo()  # one step
    assert s.scene.get("free_surface-1").properties["auto_range"] is True
    s.set_property("free_surface-1", "auto_range", True)  # found again
    assert s.data.automatic_range("free_surface-1") == pytest.approx(shown)


def test_item_slider_sets_case_exaggeration_while_synced(reef3d_run):
    s = Session()
    s.open_case(reef3d_run)
    bed = next(i.id for i in s.scene if i.kind == "bathymetry")
    s.set_exaggeration(bed, 4.0)  # synced: the case's, for every item
    assert s.scene.get("case-1").properties["exaggeration"] == 4.0
    assert s.data.exaggeration("free_surface-1") == 4.0
    s.sync_exaggeration(bed, False)  # items keep the case's value as their own
    assert s.data.exaggeration(bed) == 4.0 and s.data.exaggeration("free_surface-1") == 4.0
    s.set_exaggeration(bed, 2.0)
    assert s.data.exaggeration(bed) == 2.0 and s.data.exaggeration("free_surface-1") == 4.0
    assert s.stack.undo_label == "Set exaggeration"
    s.undo()
    assert s.stack.undo_label == "Unsync exaggeration"
    s.undo()  # one step: sync on again, items' own values back
    assert s.scene.get("case-1").properties["sync_exaggeration"] is True
    assert s.scene.get(bed).properties["exaggeration"] == 1.0


def test_hide_wet_land_of_the_bed(reef3d_run):
    from lagoonviz.core.data.manager import _dry_triangles
    from lagoonviz.core.data.mesh import SurfaceMesh

    points = np.array([[0, 0, -1], [1, 0, -1], [0, 1, -1], [1, 1, 1]], np.float32)
    triangles = np.array([[0, 1, 2], [1, 3, 2]], np.int32)
    mesh = SurfaceMesh(points, triangles, {"depth": -points[:, 2]})
    kept = _dry_triangles(mesh, 0.0)
    assert kept.tolist() == [[0, 0, 0], [1, 3, 2]]  # all wet: gone; shoreline: kept
    no_depth = SurfaceMesh(points, triangles)
    assert _dry_triangles(no_depth, 0.0).tolist() == kept.tolist()  # below the level

    s = Session()
    s.open_case(reef3d_run)
    bed = next(i.id for i in s.scene if i.kind == "bathymetry")
    mesh = s.data.load(bed)
    assert s.data.display(bed, mesh).triangles is mesh.triangles  # off by default
    s.set_property(bed, "hide_wet", True)
    shown = s.data.display(bed, mesh)
    assert shown.triangles.shape == mesh.triangles.shape  # same length: buffer update only
    assert not shown.triangles.any()  # the test bed lies wholly under water


def test_plain_colour_follows_the_colour_map(reef3d_run):
    s = Session()
    s.open_case(reef3d_run)
    bed = next(i.id for i in s.scene if i.kind == "bathymetry")
    mesh = s.data.load(bed)
    assert s.data.display(bed, mesh).color == "#c4ae86"  # sand, as before
    for colormap, color in (("white", "#ffffff"), ("greys", "#9a9a9a")):
        s.set_property(bed, "colormap", colormap)
        shown = s.data.display(bed, mesh)
        assert shown.values is None and shown.color == color
    s.set_property(bed, "colormap", "rainbow")
    assert s.data.display(bed, mesh).color == "#00ff00"  # the middle of the map


def test_loader_drops_frames_nobody_waits_for():
    """Dragging the timeline asks for many frames in turn: only the last one is read."""
    import threading

    from lagoonviz.core.data.manager import BackgroundLoader

    gate = threading.Event()
    read = []

    class Slow:
        def load_key(self, key):
            gate.wait(5)
            read.append(key)
            return key

    loader = BackgroundLoader(Slow(), workers=1)
    delivered = []
    first = loader.request("busy", lambda k, m, e: delivered.append(k), owner="other")
    passed = []
    for i in range(20):
        passed.append(
            loader.request(f"frame {i}", lambda k, m, e: delivered.append(k), owner="view")
        )
        if i == 5:
            shared = loader.request("frame 5", owner="second view")  # needed elsewhere too
    gate.set()
    for future in (first, shared, passed[-1]):
        future.result(timeout=5)
    loader.shutdown()
    assert all(f.cancelled() for f in passed[:5] + passed[6:-1])
    assert not shared.cancelled() and shared is passed[5]
    assert set(read) == {"busy", "frame 5", "frame 19"}
    assert sorted(delivered) == ["busy", "frame 19", "frame 5"]  # dropped: no callback


def test_dem_elements(tmp_path):
    """DEM output (REEF3D_DEM_VTP, serial ASCII files at the DEM's own interval) is
    found, shown when the case opens, and placed in time by the files' TimeValue."""
    from reef3d_fixtures import dem_centres, make_run

    run = make_run(tmp_path / "dem", frames=5, print_interval=0.5, dem=0.25)
    s = Session()
    s.open_case(run)
    dem = next(i.id for i in s.scene if i.kind == "dem")
    dataset = s.data.dataset(dem)
    assert dataset.role == "dem" and len(dataset) == 9 and dataset.label
    np.testing.assert_allclose(dataset.times, np.arange(9) * 0.25)
    assert s.data.variable_choices(dem)[:2] == ["none", "velocity"]
    for time in (0.0, 0.75, 1.5):
        s.set_time(time)
        assert s.data.frame_index(dem) == round(time / 0.25)  # not output number x 0.5
        mesh = s.data.load(dem)
        shown = s.data.display(dem, mesh)
        assert shown.triangles.shape == (24, 3)  # two cubes
        assert shown.color == "#6f6a64"  # rock
        np.testing.assert_allclose(
            mesh.points.reshape(2, 8, 3).mean(axis=1), dem_centres(time), atol=1e-5
        )
    assert s.data.frame_index("free_surface-1") == 3  # the surfaces still by output number


def test_plan_view_draws_everything_flat(tmp_path):
    """2D: every item flat, the free surface over the bed, gauges on top; saved with
    the session and in the script, so the export films it the same way."""
    from reef3d_fixtures import make_run

    from lagoonviz.core.data.manager import PLAN_LAYERS, PLAN_MARKS

    run = make_run(tmp_path / "plan", frames=5, print_interval=0.5, series=True, dem=0.25)
    s = Session()
    s.open_case(run)
    s.set_property("case-1", "exaggeration", 5.0)  # makes no difference in 2D
    events = []
    s.scene.subscribe(lambda event, _id: events.append(event))
    s.set_plan_view(True)
    assert events == ["view"]
    for item_id in s.data.data_items():
        shown = s.data.display(item_id, s.data.load(item_id))
        kind = s.scene.get(item_id).kind
        np.testing.assert_allclose(shown.positions[:, 2], PLAN_LAYERS[kind])
        np.testing.assert_allclose(shown.normals[:, 2], 1.0)
        assert shown.values is not None or kind != "free_surface"  # still coloured by eta
    marks = s.data.gauge_marks("series-1")
    assert np.all(marks.staffs[..., 2] == PLAN_MARKS)
    path = tmp_path / "plan.lagoon-session"
    s.save(path)
    assert Session.open(path).scene.plan_view is True
    assert "session.set_plan_view(True)" in s.to_script()
    s.set_plan_view(False)
    bed = next(i.id for i in s.scene if i.kind == "bathymetry")
    assert np.ptp(s.data.display(bed, s.data.load(bed)).positions[:, 2]) > 0  # 3D again
