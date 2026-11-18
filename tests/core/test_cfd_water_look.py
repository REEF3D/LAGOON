# LAGOON — CFD's free surface is water in the looks
# Architect: Hans Bihs
"""CFD has no free surface file: its free surface is an iso-surface of the level set
phi at 0 (of VOF at 0.5 with VOF-PLIC). The 3D view's looks draw it as they draw a
free surface: water (glossy, the sky reflected, the look's water colour, on the
Open ocean's stand a little see-through with its sides down to the bed), not as a
plain matte surface."""

from __future__ import annotations

import numpy as np
import pytest
from test_iso_band import _cfd_store

from lagoonviz.core import Session
from lagoonviz.core.data.manager import is_water


def test_cfd_free_surface_is_water(tmp_path):
    from lagoonviz.render.cinematic import OPEN_OCEAN
    from lagoonviz.render.scene3d import DIORAMA_WATER_OPACITY, SceneRenderer

    root = _cfd_store(tmp_path / "cfd", frames=1)
    s = Session()
    s.open_case(root)
    free = next(i for i in s.scene if i.kind == "iso_surface")
    s.set_property(free.id, "variable", "none")
    assert is_water(free)
    other = s.add_iso_surface("velocity", 0.3)
    assert not is_water(s.scene.get(other))
    shown = s.data.display(free.id, s.data.load(free.id))
    assert shown.water
    view = SceneRenderer("open_ocean")
    view.update("iso_surface", shown)
    material = view._shown[free.id].mesh.material
    assert material.roughness == pytest.approx(OPEN_OCEAN.water_roughness)
    assert material.env_map is not None
    expected = tuple(int(OPEN_OCEAN.water_color[i : i + 2], 16) / 255 for i in (1, 3, 5))
    assert np.allclose(tuple(material.color)[:3], expected, atol=0.01)
    assert material.opacity == pytest.approx(DIORAMA_WATER_OPACITY)
    assert free.id in view._stand  # its sides down to the bed
    # in the engineering looks, glossy as a free surface
    view.set_style("studio")
    from lagoonviz.render.scene3d import STYLES

    assert view._shown[free.id].mesh.material.shininess == STYLES["studio"]["water_shininess"]
