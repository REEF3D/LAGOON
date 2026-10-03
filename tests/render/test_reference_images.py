"""Agreed looks of the 3D view (part 3 of the regression suite).

Each test renders the small test case off screen and compares it with the agreed
image in ``tests/snapshots/images/<platform>``, within a small tolerance (see
``tests/snapshot_tools.py``). Colours, lighting, shadows, exaggeration and the
named views are covered; a change that alters them fails with the new image, the
agreed one and a difference image. Accept an intended change with
``pytest --update-snapshots`` and look at the new images before committing them.
"""

import pytest

pytest.importorskip("pygfx")

from lagoonviz.core import Session  # noqa: E402
from lagoonviz.render.offscreen import render_session  # noqa: E402

SIZE = (320, 180)


def _adapter() -> bool:
    try:
        import wgpu

        request = getattr(wgpu.gpu, "request_adapter_sync", None) or wgpu.gpu.request_adapter
        return request(power_preference="high-performance") is not None
    except Exception:  # noqa: BLE001
        return False


pytestmark = pytest.mark.skipif(not _adapter(), reason="no GPU adapter")


def _session(case, **properties) -> Session:
    session = Session()
    session.open_case(case)
    bed = next(i.id for i in session.scene if i.kind == "bathymetry")
    for key, value in properties.items():
        item_id, _, name = key.partition("__")
        item_id = {"bed": bed, "fs": "free_surface-1", "case": "case-1"}.get(item_id, item_id)
        session.set_property(item_id, name, value)
    return session


#: name: (properties set, view, look, items hidden); all at t = 1 s
LOOKS = {
    "studio-perspective": ({}, "perspective", "studio", ()),
    "ocean-perspective": ({}, "perspective", "ocean", ()),
    "studio-2d": ({}, "2d", "studio", ()),
    "studio-front": ({}, "front", "studio", ()),
    "rainbow-free-surface": ({"fs__colormap": "rainbow"}, "perspective", "studio", ()),
    "case-exaggeration-5": ({"case__exaggeration": 5.0}, "perspective", "studio", ()),
    "bed-white-by-depth": (
        {"bed__variable": "depth", "bed__colormap": "white"},
        "perspective",
        "studio",
        ("free_surface-1",),
    ),
    "bed-greys-plain": ({"bed__colormap": "greys"}, "perspective", "studio", ()),
    "bed-contours": (
        {"bed__contour_by": "elevation", "bed__contour_levels": 6},
        "perspective",
        "studio",
        ("free_surface-1",),
    ),
    "surface-contours-2d": ({"fs__contour_by": "eta"}, "2d", "studio", ()),
}


@pytest.mark.parametrize("name", list(LOOKS))
def test_agreed_look(reef3d_run, snapshots, name):
    properties, view, look, hidden = LOOKS[name]
    session = _session(reef3d_run, **properties)
    for item_id in hidden:
        session.set_visible(item_id, False)
    image = render_session(session, SIZE, time=1.0, view=view, style=look)
    snapshots.check_image(name, image)
