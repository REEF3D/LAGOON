"""Headless rendering with pygfx: the design test for 3D views.

Skipped where pygfx or a GPU adapter is missing. CI installs Mesa's software
Vulkan driver on Linux so these run there too. With ``LAGOON_TEST_OUTPUT`` set,
the rendered images are written to that folder for a visual check.
"""

import os
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("pygfx")

from lagoonviz.core import Session  # noqa: E402
from lagoonviz.render.offscreen import main, render_session, write_png  # noqa: E402


def _has_adapter() -> bool:
    try:
        import wgpu

        request = getattr(wgpu.gpu, "request_adapter_sync", None) or wgpu.gpu.request_adapter
        return request(power_preference="high-performance") is not None
    except Exception:  # noqa: BLE001
        return False


pytestmark = pytest.mark.skipif(not _has_adapter(), reason="no GPU adapter")


def _keep(name: str, image: np.ndarray) -> None:
    folder = os.environ.get("LAGOON_TEST_OUTPUT")
    if folder:
        Path(folder).mkdir(parents=True, exist_ok=True)
        write_png(Path(folder) / name, image)


def _colours(image: np.ndarray) -> int:
    return len(np.unique(image[..., :3].reshape(-1, 3), axis=0))


def test_render_case(reef3d_run):
    session = Session()
    session.open_case(reef3d_run)
    image = render_session(session, (480, 270))
    _keep("render_case.png", image)
    assert image.shape == (270, 480, 4)
    # the surfaces add many colours to the two-colour background gradient
    assert _colours(image) > 200


def test_saved_session_renders_the_same(reef3d_run, tmp_path):
    session = Session()
    session.open_case(reef3d_run)
    session.set_property("free_surface-1", "exaggeration", 5.0)
    first = render_session(session, (320, 180), time=1)
    path = tmp_path / "view.lagoon-session"
    session.save(path)
    again = render_session(Session.open(path), (320, 180))
    _keep("render_session.png", again)
    assert np.array_equal(first, again) or np.mean(np.abs(first.astype(int) - again)) < 1.0


def test_hidden_items_are_not_drawn(reef3d_run):
    session = Session()
    session.open_case(reef3d_run)
    shown = render_session(session, (320, 180), view="top")
    for item in ("free_surface-1", "bathymetry-1"):
        session.set_visible(item, False)
    hidden = render_session(session, (320, 180), view="top")
    assert _colours(hidden) < _colours(shown)


def test_scene_renderer_updates_in_place(reef3d_run):
    from lagoonviz.render.scene3d import SceneRenderer

    session = Session()
    session.open_case(reef3d_run)
    view = SceneRenderer()
    first = session.data.display("free_surface-1", session.data.load("free_surface-1"))
    view.update("free_surface", first)
    mesh = view._shown["free_surface-1"].mesh
    session.set_time(2)
    later = session.data.display("free_surface-1", session.data.load("free_surface-1"))
    view.update("free_surface", later)
    assert view._shown["free_surface-1"].mesh is mesh  # buffers updated, not rebuilt
    np.testing.assert_allclose(mesh.geometry.positions.data, later.positions)


def test_command_line(reef3d_run, tmp_path, capsys):
    out = tmp_path / "frame.png"
    assert main([str(reef3d_run), str(out), "--size", "200x120", "--time", "2"]) == 0
    assert out.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
    assert "Free surface" in capsys.readouterr().out
