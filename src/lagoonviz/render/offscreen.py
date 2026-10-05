# SPDX-License-Identifier: GPL-3.0-or-later
"""Render a session to an image without a screen (clusters, batch jobs, tests).

    lagoon-render run/ view.png                     # a case folder: default view
    lagoon-render figure.lagoon-session view.png    # a saved session: its view and time
    lagoon-render run/ frame.png --time 120 --size 1920x1080
    lagoon-render run/ waves.mp4 --fps 25 --start 110 --end 400   # an animation
    lagoon-render run/ still.png --engine cycles --style open_ocean  # path traced (Blender)

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
from lagoonviz.render.movie import MOVIE_SUFFIXES, frame_times, write_movie


def render_session(
    session: Session,
    size: tuple[int, int] = (1600, 900),
    time: float | None = None,
    view: str | None = None,
    style: str = "studio",
) -> np.ndarray:
    """Draw the session's visible items and return an (h, w, 4) uint8 RGBA image."""
    from lagoonviz.render.movie import OffscreenView

    return OffscreenView(session, size, style, view).render(time)


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
    parser.add_argument(
        "--view", choices=["perspective", "top", "front", "right", "2d", "2d-front", "2d-right"]
    )
    from lagoonviz.render.cinematic import PRESETS

    parser.add_argument(
        "--style",
        choices=["studio", "ocean", *PRESETS],
        default="studio",
        help="look: studio, ocean, or a cinematic one (open_ocean, wave_flume, underwater)",
    )
    cycles = parser.add_argument_group("path tracing (Blender Cycles, a cinematic look)")
    cycles.add_argument(
        "--engine",
        choices=["pygfx", "cycles"],
        default="pygfx",
        help="pygfx: LAGOON's own renderer (fast); cycles: path traced by Blender",
    )
    cycles.add_argument("--samples", type=int, default=128, help="Cycles samples per pixel")
    cycles.add_argument("--blender", help="the Blender executable (default: found)")
    cycles.add_argument(
        "--fresh",
        action="store_true",
        help="render every frame again (default: keep frames rendered before from the "
        "same data, camera and settings, so a stopped film goes on)",
    )
    movie = parser.add_argument_group("animations (output .mp4 or .mov)")
    movie.add_argument("--fps", type=float, default=25.0, help="frames per second")
    movie.add_argument("--start", type=int, help="first output (default: first with data)")
    movie.add_argument("--end", type=int, help="last output (default: the last)")
    movie.add_argument("--every", type=int, default=1, help="film every n-th output")
    args = parser.parse_args(argv)
    width, _, height = args.size.lower().partition("x")
    size = (int(width), int(height))
    session = _session_for(Path(args.source).expanduser())
    if args.engine == "cycles":
        return _path_trace(session, args, size)
    if Path(args.output).suffix.lower() in MOVIE_SUFFIXES:
        times = frame_times(session, args.start, args.end, args.every)

        def report(done: int, total: int) -> bool:
            print(f"\rframe {done} / {total}", end="", flush=True)
            return True

        written = write_movie(
            session, args.output, times, size, args.fps, args.style, args.view, None, report
        )
        print(f"\nwrote {written} ({len(times)} frames, {width}x{height}, {args.fps:g} fps)")
        return 0
    image = render_session(session, size, args.time, args.view, args.style)
    write_png(args.output, image)
    shown = [i.name for i in session.scene if i.kind in KINDS and i.visible and i.kind != "case"]
    print(f"wrote {args.output} ({width}x{height}): {', '.join(shown) or 'nothing to show'}")
    return 0


def _path_trace(session: Session, args, size: tuple[int, int]) -> int:
    from lagoonviz.render.pathtrace import blender_progress, path_trace

    output = Path(args.output)
    look = args.style if args.style != "studio" else "open_ocean"
    if output.suffix.lower() == ".png":
        if args.time is not None:
            session.set_time(args.time)
        times = None
    else:
        times = frame_times(session, args.start, args.end, args.every)

    def report(done: int, total: int) -> bool:
        print(f"\rframe {done} / {total}", end="", flush=True)
        return True

    print(f"path tracing with Blender Cycles ({look}, {args.samples} samples)", flush=True)

    def blender_says(line: str) -> None:
        # Blender's own progress (kernels compiling, samples) and LAGOON's steps
        text = blender_progress(line)
        if text:
            print(f"\r{text[:100]:<100}", end="", flush=True)

    written = path_trace(
        session,
        output,
        times,
        size,
        look,
        args.samples,
        args.fps,
        args.blender,
        report,
        resume=not args.fresh,
        status=blender_says,
    )
    print(f"\nwrote {written} ({size[0]}x{size[1]}, Cycles)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
