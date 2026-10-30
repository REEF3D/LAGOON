"""Particles: CFD's CPM sediment particles and NHFLOW's Lagrangian particles."""

import numpy as np
from reef3d_fixtures import make_run, particle_cloud, write_cpm, write_nhflow_particles

from lagoonviz.core import Session
from lagoonviz.core.data.reef3d import open_case


def test_cpm_particles_are_found_and_read(tmp_path):
    run = make_run(tmp_path / "cpm", solver="CFD", frames=4, print_interval=0.5)
    write_cpm(run, end=1.5, interval=0.25)
    particles = open_case(run).datasets["cfd_particles"]
    assert particles.role == "particles" and particles.label == "CFD particles"
    assert len(particles) == 7 and len(particles.frames[0].pieces) == 2
    np.testing.assert_allclose(particles.times, 0.25 * np.arange(7))  # TimeValue
    assert set(particles.variables()) >= {"velocity", "fluid velocity", "radius", "bedChange"}
    assert particles.default_variable() == "velocity"
    mesh = particles.read(3, ["velocity"])
    expected = np.concatenate([particle_cloud(0.75, 0), particle_cloud(0.75, 1)])
    assert np.array_equal(mesh.points, expected) and mesh.n_triangles == 0
    np.testing.assert_allclose(mesh.scalar("velocity"), np.hypot(0.3, 0.05), rtol=1e-6)


def test_session_shows_particles_as_points(tmp_path):
    run = make_run(tmp_path / "cpm", solver="CFD", frames=4, print_interval=0.5)
    write_cpm(run, end=1.5, interval=0.25)
    s = Session()
    s.open_case(run)
    kinds = [i.kind for i in s.scene]
    assert "particles" in kinds
    item = next(i for i in s.scene if i.kind == "particles")
    s.set_time(1.0)
    shown = s.data.display(item.id, s.data.load(item.id))
    assert shown.point_size == 4.0 and shown.variable == "velocity"
    assert len(shown.positions) == len(particle_cloud(1.0, 0)) + len(particle_cloud(1.0, 1))
    s.set_property(item.id, "point_size", 8.0)
    assert s.data.display(item.id, s.data.load(item.id)).point_size == 8.0


def test_nhflow_particles(tmp_path):
    run = make_run(tmp_path / "nh", frames=3, print_interval=0.5)
    write_nhflow_particles(run, end=1.0, interval=0.5)
    particles = open_case(run).datasets["nhflow_particles"]
    assert len(particles) == 3 and particles.steps == [0, 1, 2]
    mesh = particles.read(2, None)
    assert mesh.point_data["id"].dtype == np.int32
    np.testing.assert_allclose(mesh.point_data["age"], 1.0)


def test_particle_trails_follow_each_id(tmp_path):
    from reef3d_fixtures import write_nhflow_particles

    run = make_run(tmp_path / "nh", frames=3, print_interval=0.5)
    write_nhflow_particles(run, end=1.0, interval=0.25)  # 5 outputs
    s = Session()
    s.open_case(run)
    item = next(i for i in s.scene if i.kind == "particles")
    s.set_time(1.0)
    assert s.data.particle_trails(item.id) is None  # off by default
    s.set_property(item.id, "trail_outputs", 3)
    mesh = s.data.load(item.id)
    assert "id" in mesh.point_data  # read for the trails
    trails = s.data.particle_trails(item.id, mesh)
    # ids 0 ... n-1 in every output (the cloud grows): each id present in two
    # consecutive outputs gives one segment, over the 3 outputs before
    counts = [len(particle_cloud(t, 0, n0=20)) for t in (0.25, 0.5, 0.75, 1.0)]
    assert trails.segments.shape == (
        sum(min(a, b) for a, b in zip(counts, counts[1:], strict=False)),
        2,
        3,
    )
    # segments end where the particles are now, and start where they were
    newest = trails.segments[: counts[2]]
    np.testing.assert_allclose(newest[:, 1], particle_cloud(1.0, 0, n0=20)[: counts[2]], atol=1e-6)
    np.testing.assert_allclose(newest[:, 0], particle_cloud(0.75, 0, n0=20), atol=1e-6)
    # in 2D they lie on the particles' layer
    s.set_plan_view(True, "top")
    flat = s.data.particle_trails(item.id, mesh)
    assert np.all(flat.segments[..., 2] == flat.segments[0, 0, 2])


def test_dem_trails_follow_element_centres(tmp_path):
    from reef3d_fixtures import dem_centres

    run = make_run(tmp_path / "dem", frames=3, print_interval=0.5, volume=False, dem=0.25)
    s = Session()
    s.open_case(run)
    item = next(i for i in s.scene if i.kind == "dem")
    s.set_property(item.id, "trail_outputs", 1)
    s.set_time(0.5)
    trails = s.data.particle_trails(item.id)
    assert trails is not None and trails.segments.shape == (2, 2, 3)  # two elements
    ends = trails.segments[:, 1]
    np.testing.assert_allclose(np.sort(ends[:, 0]), np.sort(dem_centres(0.5)[:, 0]), atol=1e-4)


def test_the_cpm_bed(tmp_path):
    """The CPM bed files of the ranks (Q 58) as one surface: the bed level, the bed
    shear stress and the change since the first output, at their times."""
    from reef3d_fixtures import cpm_bed

    run = make_run(tmp_path / "cpm", solver="CFD", frames=4, print_interval=0.5)
    write_cpm(run, end=1.5, interval=0.25, bed=True)
    bed = open_case(run).datasets["cfd_cpm_bed"]
    assert bed.role == "cpm_bed" and bed.label == "CFD CPM bed" and len(bed) == 7
    np.testing.assert_allclose(bed.times, 0.25 * np.arange(7))
    assert bed.default_variable() == "bed_change"
    mesh = bed.read(4)
    assert mesh.points.shape == (20 * 4, 3)  # both ranks' columns, one grid
    assert mesh.n_triangles == 2 * 19 * 3
    x, y = mesh.points[:, 0].astype(float), mesh.points[:, 1].astype(float)
    np.testing.assert_allclose(mesh.points[:, 2], cpm_bed(x, y, 1.0), atol=1e-6)
    change = mesh.point_data["bed_change"]
    np.testing.assert_allclose(change, cpm_bed(x, y, 1.0) - cpm_bed(x, y, 0.0), atol=1e-6)
    np.testing.assert_allclose(mesh.point_data["tau"][:, 0], 0.5 + x, rtol=1e-5)
    assert np.abs(bed.read(0).point_data["bed_change"]).max() == 0.0
    # in the scene beside the particles, coloured by its change
    s = Session()
    s.open_case(run)
    item = next(i for i in s.scene if i.name == "CPM bed")
    assert item.kind == "bathymetry" and item.properties["variable"] == "bed_change"
    assert s.data.dataset(item.id).key == "cfd_cpm_bed"


def test_cpm_parcels_with_ids_have_paths(tmp_path):
    run = make_run(tmp_path / "cpm", solver="CFD", frames=4, print_interval=0.5)
    write_cpm(run, end=1.5, interval=0.25, ids=True)
    s = Session()
    s.open_case(run)
    item = next(i for i in s.scene if i.kind == "particles")
    s.set_time(1.0)
    s.set_property(item.id, "trail_outputs", 2)
    mesh = s.data.load(item.id)
    trails = s.data.particle_trails(item.id, mesh)
    assert trails is not None and len(trails.segments) > 0


def test_particles_sized_by_their_radius(tmp_path):
    run = make_run(tmp_path / "cpm", solver="CFD", frames=4, print_interval=0.5)
    write_cpm(run, end=1.5, interval=0.25)
    s = Session()
    s.open_case(run)
    item = next(i for i in s.scene if i.kind == "particles")
    s.set_time(1.0)
    assert s.data.display(item.id, s.data.load(item.id)).point_sizes is None  # on screen
    s.set_property(item.id, "size_by", "radius")
    s.set_property(item.id, "radius_scale", 20.0)
    shown = s.data.display(item.id, s.data.load(item.id))
    assert len(shown.point_sizes) == len(shown.positions)
    np.testing.assert_allclose(shown.point_sizes, 2 * 0.0005 * 20.0)  # the fixtures' radius


def test_the_cpm_bed_stays_in_its_files_when_converted(tmp_path):
    from lagoonviz.core.convert import convert

    run = make_run(tmp_path / "cpm", solver="CFD", frames=4, print_interval=0.5)
    write_cpm(run, end=1.5, interval=0.25, bed=True, ids=True)
    report = convert(run)
    assert "cpm_bed" not in report.outputs and report.outputs["particles"] == 7
    case = open_case(run)
    assert case.datasets["cfd_cpm_bed"].read(2).points.shape == (80, 3)
    stored = case.datasets["cfd_particles"]
    assert stored.source == "lagoon" and "id" in stored.read(1, None).point_data


def test_a_run_of_only_cpm_parcels_shows_them(tmp_path):
    """A CPM run without volume or surface output (P 10 0): its parcels and its bed
    are shown."""
    run = tmp_path / "cpm2d"
    run.mkdir()
    write_cpm(run, end=0.5, interval=0.25, bed=True, ids=True)
    s = Session()
    s.open_case(run)
    names = [(i.kind, i.name) for i in s.scene]
    assert names[0] == ("case", "cpm2d")
    assert ("particles", "Particles") in names and ("bathymetry", "CPM bed") in names


def test_the_cpm_bed_of_a_2d_run_is_a_strip(tmp_path):
    from lagoonviz.core.data.cpm import bed_datasets

    folder = tmp_path / "REEF3D_CFD_CPM_Particle"
    folder.mkdir()
    for count, t in enumerate((0.0, 0.5)):
        lines = [f"# time {t}", "# x y z_bed tau_x tau_y"]
        lines += [f"{0.05 + 0.1 * i:.2f} 0.005 {0.1 + 0.01 * i * t:.4f} 1 0" for i in range(5)]
        (folder / f"REEF3D-CFD-CPM-Bed-{count:08d}-000001.dat").write_text("\n".join(lines))
    (bed,) = bed_datasets(folder)
    mesh = bed.read(1)
    assert mesh.points.shape == (10, 3) and mesh.n_triangles == 8
    np.testing.assert_allclose(sorted(set(mesh.points[:, 1].round(4))), [-0.045, 0.055])
    np.testing.assert_allclose(mesh.point_data["bed_change"][:5], 0.005 * np.arange(5), atol=1e-6)


def test_particles_drawn_as_spheres(tmp_path):
    """Draw as spheres: a round, lit ball at each particle, as large as Sphere radius
    (or, with Size radius, each its own radius times Radius scale), coloured by the
    particle's value; the vertical scale moves the balls, it does not stretch them."""
    from lagoonviz.core.data.spheres import icosphere, sphere_mesh, subdivisions_for

    run = make_run(tmp_path / "cpm", solver="CFD", frames=4, print_interval=0.5)
    write_cpm(run, end=1.5, interval=0.25)
    s = Session()
    s.open_case(run)
    item = next(i for i in s.scene if i.kind == "particles")
    s.set_time(1.0)
    mesh = s.data.load(item.id)
    centres = mesh.points.astype(np.float64)
    n = len(centres)
    s.set_property(item.id, "shape", "spheres")
    s.set_property(item.id, "sphere_radius", 0.02)
    shown = s.data.display(item.id, s.data.load(item.id))
    corners = len(icosphere(subdivisions_for(n))[0])
    assert shown.point_size is None and shown.point_sizes is None  # a mesh, not points
    assert shown.positions.shape == (n * corners, 3) and shown.triangles.size
    assert shown.triangles.max() < len(shown.positions)
    balls = shown.positions.reshape(n, corners, 3).astype(np.float64)
    np.testing.assert_allclose(np.linalg.norm(balls - centres[:, None], axis=2), 0.02, rtol=1e-4)
    np.testing.assert_allclose(np.linalg.norm(shown.normals, axis=1), 1.0, rtol=1e-5)
    values = mesh.scalar("velocity")
    np.testing.assert_allclose(shown.values.reshape(n, corners), values[:, None].repeat(corners, 1))
    # each its own size
    s.set_property(item.id, "size_by", "radius")
    s.set_property(item.id, "radius_scale", 20.0)
    shown = s.data.display(item.id, s.data.load(item.id))
    balls = shown.positions.reshape(n, corners, 3).astype(np.float64)
    np.testing.assert_allclose(np.linalg.norm(balls - centres[:, None], axis=2), 0.01, rtol=1e-4)
    # the vertical scale moves the centres, the balls stay round
    s.set_exaggeration(item.id, 3.0)
    shown = s.data.display(item.id, s.data.load(item.id))
    balls = shown.positions.reshape(n, corners, 3).astype(np.float64)
    middle = balls.mean(axis=1)
    level = s.data.case_water_level(item.id)
    np.testing.assert_allclose(middle[:, 2], level + 3 * (centres[:, 2] - level), atol=1e-5)
    np.testing.assert_allclose(np.linalg.norm(balls - middle[:, None], axis=2), 0.01, rtol=1e-3)
    # back to points
    s.set_property(item.id, "shape", "points")
    assert s.data.display(item.id, s.data.load(item.id)).point_size == 4.0
    # many particles: coarser balls, all of them
    assert subdivisions_for(1_000) == 2 and subdivisions_for(50_000) == 1
    assert subdivisions_for(1_000_000) == 0
    positions, triangles, normals, _ = sphere_mesh(np.zeros((3, 3)), [1.0, 2.0, 3.0], None, 0)
    assert positions.shape == (36, 3) and triangles.shape == (60, 3) and normals.shape == (36, 3)
    # facing out: every triangle's normal away from its ball's centre
    a, b, c = (positions[triangles[:, k]] for k in range(3))
    assert (np.einsum("ij,ij->i", np.cross(b - a, c - a), a + b + c) > 0).all()
