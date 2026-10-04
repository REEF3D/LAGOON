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
