# SPDX-License-Identifier: GPL-3.0-or-later
"""Render a session to an image without a screen (clusters, batch jobs, tests).

    lagoon-render run/ view.png                     # a case folder: default view
    lagoon-render figure.lagoon-session view.png    # a saved session: its view and time
    lagoon-render run/ frame.png --time 120 --size 1920x1080

This is the design test of LAGOON's architecture: whatever the GUI shows can be
rebuilt from a saved session by code that never touches Qt.
"""

from __future__ import annotations

import argparse
import struct
import sys
import zlib
from pathlib import Path

import numpy as np

from lagoonviz.core import Session
from lagoonviz.core.kinds import KINDS


def _offscreen_canvas(size: tuple[int, int]):
    try:
        from rendercanvas.offscreen import RenderCanvas
    except ImportError:  # pygfx before rendercanvas
        from wgpu.gui.offscreen import WgpuCanvas as RenderCanvas
    try:
        return RenderCanvas(size=size, pixel_ratio=1)
    except TypeError:
        return RenderCanvas(size=size)


def render_session(
    session: Session,
    size: tuple[int, int] = (1600, 900),
    time: float | None = None,
    view: str | None = None,
    style: str = "studio",
) -> np.ndarray:
    """Draw the session's visible items and return an (h, w, 4) uint8 RGBA image."""
    import pygfx as gfx

    from lagoonviz.render.scene3d import SceneRenderer

    if time is not None:
        session.set_time(time)
    canvas = _offscreen_canvas(size)
    renderer = gfx.WgpuRenderer(canvas)
    scene = SceneRenderer(style)
    for item_id in session.data.data_items():
        item = session.scene.get(item_id)
        mesh = session.data.load(item_id)
        if mesh is not None:
            scene.update(item.kind, session.data.display(item_id, mesh))
    if view:
        scene.frame(view)
    elif session.scene.camera:
        scene.set_camera_state(session.scene.camera)
    else:
        scene.frame("perspective")
    canvas.request_draw(lambda: scene.render(renderer))
    image = np.asarray(canvas.draw())
    return image[..., :4].astype(np.uint8) if image.dtype != np.uint8 else image


def write_png(path: str | Path, image: np.ndarray) -> None:
    """Write an RGBA or RGB uint8 image as PNG (no imaging library needed)."""
    image = np.ascontiguousarray(image, dtype=np.uint8)
    height, width, channels = image.shape
    color_type = {3: 2, 4: 6}[channels]
    rows = np.concatenate([np.zeros((height, 1), np.uint8), image.reshape(height, -1)], axis=1)

    def chunk(tag: bytes, data: bytes) -> bytes:
        body = tag + data
        return struct.pack(">I", len(data)) + body + struct.pack(">I", zlib.crc32(body))

    header = struct.pack(">IIBBBBB", width, height, 8, color_type, 0, 0, 0)
    Path(path).write_bytes(
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", header)
        + chunk(b"IDAT", zlib.compress(rows.tobytes(), 6))
        + chunk(b"IEND", b"")
    )


def _session_for(path: Path) -> Session:
    if path.suffix == ".lagoon-session":
        return Session.open(path)
    if path.suffix == ".py":
        return Session.from_script(path.read_text(encoding="utf-8"))
    session = Session()
    session.open_case(path)
    return session


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="lagoon-render", description="Render a LAGOON session or REEF3D case to PNG"
    )
    parser.add_argument("source", help="case folder, output file, .lagoon-session or .py script")
    parser.add_argument("output", help="PNG file to write")
    parser.add_argument("--time", type=float, help="time (or frame) to show")
    parser.add_argument("--size", default="1600x900", help="image size, e.g. 1920x1080")
    parser.add_argument("--view", choices=["perspective", "top", "front", "right"])
    parser.add_argument("--style", choices=["studio", "ocean"], default="studio")
    args = parser.parse_args(argv)
    width, _, height = args.size.lower().partition("x")
    session = _session_for(Path(args.source).expanduser())
    image = render_session(session, (int(width), int(height)), args.time, args.view, args.style)
    write_png(args.output, image)
    shown = [i.name for i in session.scene if i.kind in KINDS and i.visible and i.kind != "case"]
    print(f"wrote {args.output} ({width}x{height}): {', '.join(shown) or 'nothing to show'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
