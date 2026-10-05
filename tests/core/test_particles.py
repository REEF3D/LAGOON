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
