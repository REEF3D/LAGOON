"""Derived quantities: the built-ins (water depth, Froude, unit discharge, hazard, bed
shear, vorticity, Q) where their inputs are, the case's own formulas, worked out
when an item shows them, on surfaces, AMR surfaces and through volumes."""

import numpy as np
import pytest
from reef3d_fixtures import make_amr, make_run

from lagoonviz.core import Session
from lagoonviz.core.data import derived
from lagoonviz.core.data.reef3d import open_case


# ----------------------------------------------------------------- pieces
def test_the_gradient_of_a_linear_field_is_exact():
    # a triangulated plan (not regular) and a hexahedral block
    rng = np.random.default_rng(1)
    x, y = np.meshgrid(np.linspace(0, 4, 5), np.linspace(0, 3, 4), indexing="ij")
    points = np.stack([x.ravel() + 0.1 * rng.random(x.size), y.ravel(), 0 * x.ravel()], 1)
    tris = []
    for i in range(4):
        for j in range(3):
            a = i * 4 + j
            tris += [(a, a + 4, a + 5), (a, a + 5, a + 1)]
    grad = derived.Gradient(points, [np.array(tris)], 2)
    u = np.stack([-points[:, 1], points[:, 0]], axis=1)  # solid-body rotation
    j = grad(u)
    np.testing.assert_allclose(j[:, 1, 0] - j[:, 0, 1], 2.0, atol=1e-9)
    # 3D: u = (2x + z, 3y, -x) on hexahedra
    xs, ys, zs = np.meshgrid(np.arange(3.0), np.arange(3.0), np.arange(3.0), indexing="ij")
    p3 = np.stack([xs.ravel(), ys.ravel(), zs.ravel() * 0.5], 1)

    def node(i, k, m):
        return (i * 3 + k) * 3 + m

    hexes = [
        [node(i, k, m), node(i + 1, k, m), node(i + 1, k + 1, m), node(i, k + 1, m),
         node(i, k, m + 1), node(i + 1, k, m + 1), node(i + 1, k + 1, m + 1), node(i, k + 1, m + 1)]
        for i in range(2) for k in range(2) for m in range(2)
    ]  # fmt: skip
    groups = derived.cell_groups(np.ravel(hexes), np.arange(1, len(hexes) + 1) * 8)
    grad3 = derived.Gradient(p3, groups, 3)
    v = np.stack([2 * p3[:, 0] + p3[:, 2], 3 * p3[:, 1], -p3[:, 0]], axis=1)
    j3 = grad3(v)
    np.testing.assert_allclose(j3[0], [[2, 0, 1], [0, 3, 0], [-1, 0, 0]], atol=1e-9)
    curl = derived.compute_builtin("vorticity", {"velocity": v}, derived.Settings(), grad3)
    np.testing.assert_allclose(curl, np.tile([0.0, 2.0, 0.0], (len(v), 1)), atol=1e-6)
    q = derived.compute_builtin("q_criterion", {"velocity": v}, derived.Settings(), grad3)
    # S = diag(2, 3, 0) (the z/x parts cancel), Omega's xz part 1: Q = (2 - 13) / 2
    np.testing.assert_allclose(q, -5.5, atol=1e-5)


def test_flow_quantities_from_depth_and_velocity():
    data = {
        "velocity": np.array([[3.0, 4.0, 0.5], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]], np.float32),
        "WL": np.array([2.0, 0.0, 8.0], np.float32),
    }
    s = derived.Settings(manning=0.03)
    h = derived.compute_builtin("water_depth", data, s)
    fr = derived.compute_builtin("froude", data, s)
    q = derived.compute_builtin("unit_discharge", data, s)
    hazard = derived.compute_builtin("hazard", data, s)
    tau = derived.compute_builtin("bed_shear", data, s)
    np.testing.assert_allclose(h, [2.0, 0.0, 8.0])
    np.testing.assert_allclose(
        fr, [5.0 / np.sqrt(9.81 * 2.0), 0.0, 2.0 / np.sqrt(9.81 * 8.0)], rtol=1e-6
    )
    np.testing.assert_allclose(q[0], [6.0, 8.0, 0.0])  # horizontal only
    np.testing.assert_allclose(hazard, [10.0, 0.0, 16.0])
    np.testing.assert_allclose(tau[0], 1000 * 9.81 * 0.03**2 * 25.0 / 2.0 ** (1 / 3), rtol=1e-5)
    assert tau[1] == 0.0  # dry
    # FNPF: the still water depth plus eta; SFLOW: waterlevel
    fnpf = {
        "velocity": data["velocity"],
        "depth": np.array([1.0, 1, 1]),
        "eta": np.array([0.5, -2, 0]),
    }
    np.testing.assert_allclose(derived.compute_builtin("water_depth", fnpf, s), [1.5, 0.0, 1.0])


def test_formulas():
    assert derived.parse_formulas("a = eta*2; b=sqrt(a)\n c = mag(velocity)") == [
        ("a", "eta*2"),
        ("b", "sqrt(a)"),
        ("c", "mag(velocity)"),
    ]
    for bad in ("eta*2", "2a = eta", "x = ", "y = eta.__class__"):
        with pytest.raises(ValueError):
            derived.parse_formulas(bad)
    raw = {"velocity": 3, "eta": 1}
    settings = derived.Settings(
        formulas=[
            ("speed2", "velocity.x**2 + velocity.y**2"),
            ("q", "velocity * eta"),
            ("bad", "nothere + 1"),
            ("eta", "eta + 1"),
            ("oops", "eta +"),
        ]  # fmt: skip
    )
    plan = derived.plan_for("free_surface", False, raw, settings)
    assert plan.components["speed2"] == 1 and plan.components["q"] == 3
    assert "no nothere" in plan.problems["bad"] and "writes" in plan.problems["eta"]
    assert "oops" in plan.problems
    assert plan.inputs(["speed2", "q"]) == {"velocity", "eta"}
    data = {"velocity": np.array([[1.0, 2.0, 0.0]]), "eta": np.array([2.0])}
    plan.compute(data, ["speed2", "q"])
    assert data["speed2"][0] == pytest.approx(5.0)
    np.testing.assert_allclose(data["q"][0], [2.0, 4.0, 0.0])


# ----------------------------------------------------------------- in the scene
def test_an_amr_surface_has_the_flow_quantities(tmp_path):
    make_amr(tmp_path, "SFLOW", frames=2)
    s = Session()
    s.open_case(tmp_path)
    item = next(i for i in s.scene if i.kind == "amr")
    choices = s.data.variable_choices(item.id)
    for name in ("water_depth", "froude", "unit_discharge", "hazard", "bed_shear", "vorticity"):
        assert name in choices
    assert "unit_discharge.x" in choices
    s.set_property(item.id, "variable", "froude")
    mesh = s.data.load(item.id)
    shown = s.data.display(item.id, mesh)
    assert shown.variable == "froude" and shown.values is not None
    v = mesh.point_data["velocity"]
    h = np.clip(mesh.point_data["WL"], 0, None)
    expected = np.where(
        h > 1e-3, np.hypot(v[:, 0], v[:, 1]) / np.sqrt(9.81 * np.maximum(h, 1e-9)), 0
    )
    np.testing.assert_allclose(mesh.point_data["froude"], expected, rtol=1e-4, atol=1e-6)
    # contour lines and iso-lines of a derived quantity
    s.set_property(item.id, "contour_by", "hazard")
    assert s.data.contour_lines(item.id, s.data.load(item.id), shown) is not None


def test_a_volume_has_vorticity_and_q_through_its_slices(tmp_path):
    make_run(tmp_path / "run", frames=2)
    s = Session()
    s.open_case(tmp_path / "run")
    case = s.data.case(next(i.id for i in s.scene if i.kind == "case"))
    volume = next(d for d in case.datasets.values() if d.role == "volume")
    assert volume.components["vorticity"] == 3 and volume.components["q_criterion"] == 1
    assert "water_depth" not in volume.components
    cut = s.add_slice("y", variable="q_criterion")
    mesh = s.data.load(cut)
    q = mesh.point_data["q_criterion"]
    assert np.isfinite(q).all() and (q < 0).all()  # strain, no rotation: Q < 0
    # inside the volume about -(sin² x + 1)/2 (the fixture's u = (cos x, 0, z))
    inner = (mesh.points[:, 0] > 1.5) & (mesh.points[:, 0] < 6.5)
    x = mesh.points[inner, 0]
    np.testing.assert_allclose(q[inner], -0.5 * (np.sin(x) ** 2 + 1.0), atol=0.25)
    # an iso-surface of Q works through the pieces too
    iso = s.add_iso_surface(iso_by="q_criterion", iso_value=-0.7)
    assert s.data.load(iso).n_points > 0


def test_the_cases_formulas(tmp_path):
    make_run(tmp_path / "run", frames=2, volume=False)
    s = Session()
    s.open_case(tmp_path / "run")
    case_id = next(i.id for i in s.scene if i.kind == "case")
    surface = next(i for i in s.scene if i.kind == "free_surface")
    s.set_property(case_id, "formulas", "u2 = velocity.x * 2; up = where(eta > 0, 1, 0)")
    assert "u2" in s.data.variable_choices(surface.id)
    s.set_property(surface.id, "variable", "u2")
    mesh = s.data.load(surface.id)
    np.testing.assert_allclose(
        mesh.point_data["u2"], 2 * mesh.point_data["velocity"][:, 0], rtol=1e-6
    )
    # a new formula of the same name: worked out anew, not the frame read before
    s.set_property(case_id, "formulas", "u2 = velocity.x * 3")
    mesh = s.data.load(surface.id)
    np.testing.assert_allclose(
        mesh.point_data["u2"], 3 * mesh.point_data["velocity"][:, 0], rtol=1e-6
    )
    # undo: back to the first
    s.undo()
    np.testing.assert_allclose(
        s.data.load(surface.id).point_data["u2"], 2 * mesh.point_data["velocity"][:, 0], rtol=1e-6
    )
    s.set_property(case_id, "formulas", "w = nothing * 2; bad = eta +")
    problems = s.data.derived_problems(case_id)
    assert any("nothing" in p for p in problems) and any("bad" in p for p in problems)
    assert "manning_n" in s.scene.get(case_id).properties


def test_a_probe_follows_a_derived_quantity(tmp_path):
    make_amr(tmp_path, "SFLOW", frames=3)
    case = open_case(tmp_path)
    from lagoonviz.core.data import derived as d

    dataset = case.datasets["sflow_amr"]
    d.attach(dataset)
    where, values = dataset.point_series("water_depth", 2.0, 1.0)
    assert len(values) == 3 and np.isfinite(values).all()


def test_q_through_a_store_volume(tmp_path):
    """A σ-grid volume of a LAGOON store: Q on its slices and faces through its
    blocks' cells (not the store's quick paths, which read arrays straight)."""
    from lagoon_format.store import StoreWriter

    nx, ny, sigma = 9, 5, np.linspace(0.0, 1.0, 4)
    x, y = np.arange(nx, dtype=float), np.arange(ny, dtype=float)
    blocks = [((0, 0), (5, ny)), ((4, 0), (5, ny))]
    path = tmp_path / "REEF3D_NHFLOW.lagoon"
    writer = StoreWriter.create(path, solver="NHFLOW")
    volume = writer.add_output("volume", "sigma", x, y, {"velocity": 3}, blocks, sigma=sigma)
    for t in range(2):
        bed = np.full((ny, nx), -2.0)
        top = np.zeros((ny, nx))
        z = bed[None] + sigma[:, None, None] * (top - bed)[None]  # (nz, ny, nx)
        xx = np.broadcast_to(x[None, None, :], z.shape)
        u = np.stack([np.cos(xx), 0 * xx, 0 * xx], axis=-1)
        columns = []
        for (i0, _), (nx_b, _) in blocks:
            w = slice(i0, i0 + nx_b)
            columns.append({"z_bed": bed[:, w], "z_surface": top[:, w], "velocity": u[:, :, w]})
        volume.append(0.5 * t, t, columns)
    writer.close()
    s = Session()
    s.open_case(path)
    cut = s.add_slice("y", position=2.0, variable="q_criterion")
    mesh = s.data.load(cut)
    q = mesh.point_data["q_criterion"]
    inner = (mesh.points[:, 0] > 1.5) & (mesh.points[:, 0] < 6.5)
    np.testing.assert_allclose(q[inner], -0.5 * np.sin(mesh.points[inner, 0]) ** 2, atol=0.25)
