# LAGOON — calculated fields: new variables of a case made of REEF3D's
# Architect: Hans Bihs
"""A case's *Calculated fields* (Kelley: the eddy viscosity over the fluid's,
nu_rel = eddyv / nu, and the fluid where it is large): variables of every output
that has what they are made of, to colour by and as iso-surfaces and iso-volumes."""

from __future__ import annotations

import numpy as np
import pytest
from test_iso_band import _cfd_store, _surface

from lagoonviz.core import Session
from lagoonviz.core.calculator import (
    CalculatorError,
    add_fields,
    constants_of,
    needed,
    parse,
    parse_quietly,
    usable,
)


def test_fields_from_text():
    fields = parse(
        "nu_rel = eddyv / nu  # relative viscosity\nspeed = mag(velocity); fast = speed > 1"
    )
    assert [f.name for f in fields] == ["nu_rel", "speed", "fast"]
    assert fields[0].inputs == ("eddyv",) and fields[2].inputs == ("speed",)
    data = {
        "eddyv": np.array([1e-6, 1e-4]),
        "velocity": np.array([[3.0, 4.0, 0.0], [0.1, 0.0, 0.0]]),
        "vorticity x": np.array([2.0, -2.0]),
    }
    out = add_fields(data, fields, {"nu": 1e-6})
    np.testing.assert_allclose(out["nu_rel"], [1.0, 100.0], rtol=1e-6)
    np.testing.assert_allclose(out["speed"], [5.0, 0.1], rtol=1e-6)
    np.testing.assert_array_equal(out["fast"], [1.0, 0.0])
    more = parse(
        "w2 = velocity.z * 2 + abs(vorticity_x); v2 = velocity * 2; s = where(fast, 1, -1)"
    )
    out = add_fields(out, more, {})
    np.testing.assert_allclose(out["w2"], [2.0, 2.0])
    assert out["v2"].shape == (2, 3) and out["s"].tolist() == [1.0, -1.0]
    # what an output can calculate, and what it reads for it
    components = {"eddyv": 1, "velocity": 3, "vorticity x": 1}
    assert usable([*fields, *more], components) == {
        "nu_rel": 1, "speed": 1, "fast": 1, "w2": 1, "v2": 3, "s": 1
    }  # fmt: skip
    assert usable(fields, {"velocity": 3}) == {"speed": 1, "fast": 1}  # no eddyv: no nu_rel
    arrays, made = needed(["fast", "pressure"], fields, {**components, "pressure": 1})
    assert sorted(arrays) == ["pressure", "velocity"] and [f.name for f in made] == [
        "speed",
        "fast",
    ]


@pytest.mark.parametrize(
    "line",
    [
        "x = __import__('os')",
        "x = velocity.real",
        "x = open('f')",
        "x = 'text'",
        "nu = 1",
        "= 2",
        "x",
    ],
)
def test_only_the_calculator_runs(line):
    with pytest.raises(CalculatorError):
        parse(line)
    assert parse_quietly(line) == []


def test_the_constants_of_the_case(tmp_path):
    (tmp_path / "ctrl.txt").write_text("W 1 1025\nW 2 1.3e-6 // sea water\nW 22 -9.80\n")
    found = constants_of(tmp_path)
    assert found["rho"] == 1025 and found["nu"] == 1.3e-6 and found["g"] == 9.80
    assert found["rho_air"] == 1.205


def test_calculated_fields_of_a_case(tmp_path):
    root = _cfd_store(tmp_path / "cfd", frames=1)
    s = Session()
    case_id = s.open_case(root)
    s.set_property(case_id, "calculated", "speed = mag(velocity); depth = pressure / (rho * g)")
    item = s.add_slice("y", position=1.5, variable="speed")
    assert {"speed", "depth"} <= set(s.data.variable_choices(item))
    mesh = s.data.load(item)
    np.testing.assert_allclose(
        mesh.point_data["speed"], np.linalg.norm(mesh.point_data["velocity"], axis=1), rtol=1e-5
    )
    shown = s.data.display(item, mesh)
    assert shown.values is not None and shown.variable == "speed"
    # the water deeper than 1 m: an iso-volume of the calculated depth
    deep = s.add_iso_volume("depth", 1.0, 100.0)
    body = s.data.load(deep)
    assert body.n_triangles
    iso = s.add_iso_surface("depth", 1.0)
    surface = s.data.load(iso)
    assert surface.n_triangles and np.allclose(surface.point_data["depth"], 1.0, atol=0.05)
    # 1 m (of the store's pressure, 9810 phi: 1 / 1.0018 of phi) under the free surface
    p = surface.points[::25].astype(float)
    water = np.array([_surface(np.array([x]), np.array([y]), 0)[0, 0] for x, y, _ in p])
    assert np.abs(water - p[:, 2] - 1.0 / 1.0018).max() < 0.05
