# SPDX-License-Identifier: GPL-3.0-or-later
"""Animations: render a session frame by frame, off screen, into an MP4 (or PNGs).

    lagoon-render run/ wave.mp4 --size 1920x1080 --fps 25
    lagoon-render figure.lagoon-session wave.mp4 --start 110 --end 400 --every 2

Frames are drawn off screen at the requested size, so the film is smooth whatever
the window does. The next frame is read while the current one is drawn. MP4 needs
the ``imageio-ffmpeg`` package (it brings its own ffmpeg); without it, or for a
path without ``.mp4``/``.mov``, numbered PNGs are written instead.
"""

from __future__ import annotations

import logging
import time as time_module
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from lagoonviz.core import Session

#: ``overlay(image, time) -> image`` draws on a finished RGBA frame (time label, colour bars)
Overlay = Callable[[np.ndarray, float], np.ndarray]
#: ``progress(done, total) -> keep going``
Progress = Callable[[int, int], bool]

MOVIE_SUFFIXES = (".mp4", ".mov", ".m4v")
#: frames read ahead while one is drawn, and the threads reading them
READ_AHEAD = 6
READERS = 3

log = logging.getLogger("lagoon.movie")


def frame_times(
    session: Session, start: int | None = None, end: int | None = None, every: int = 1
) -> list[float]:
    """Timeline positions of the frames to film: outputs ``start`` to ``end`` (both
    included; default: all outputs with data), every ``every``-th one."""
    data = session.data
    n = data.n_frames()
    if n == 0:
        return []
    first = data.first_output()
    start = first if start is None else max(int(start), 0)
    end = n - 1 if end is None else min(int(end), n - 1)
    dt = data.print_interval() or 1.0
    return [step * dt for step in range(start, end + 1, max(int(every), 1))]


class OffscreenView:
    """A session drawn off screen; :meth:`render` gives the image at a time."""

    def __init__(
        self,
        session: Session,
        size: tuple[int, int] = (1600, 900),
        style: str = "studio",
        view: str | None = None,
    ) -> None:
        import pygfx as gfx

        from lagoonviz.render.scene3d import SceneRenderer

        self.session = session
        if view == "2d":
            session.set_plan_view(True)  # everything flat, as the 2D view shows it
        self.size = (int(size[0]), int(size[1]))
        self.canvas = _offscreen_canvas(self.size)
        self.renderer = gfx.WgpuRenderer(self.canvas)
        self.scene = SceneRenderer(style)
        self.view = view
        self._camera_set = False

    def show(self, time: float | None = None) -> None:
        """Bring the scene to a time: load the frames, update meshes and gauges."""
        session = self.session
        if time is not None:
            session.set_time(time)
        for item_id in session.data.data_items():
            item = session.scene.get(item_id)
            mesh = session.data.load(item_id)
            if mesh is None:
                self.scene.remove(item_id)  # no output of this run at this time
            else:
                self.scene.update(item.kind, session.data.display(item_id, mesh))
        profiles = set(session.data.profile_items())
        for item_id in session.data.series_items() + sorted(profiles):
            data = session.data
            marks = (
                data.profile_marks(item_id) if item_id in profiles else data.gauge_marks(item_id)
            )
            if marks is None:
                self.scene.remove_gauges(item_id)
            else:
                self.scene.update_gauges(marks)
        if not self._camera_set:
            if self.session.scene.plan_view:
                self.scene.set_plan(True)
            if self.view == "2d" or (
                self.session.scene.plan_view and not self.view and not session.scene.camera
            ):
                # from straight above, orthographic, as the 2D view
                self.scene.set_orthographic(True)
                self.scene.frame("top")
            elif self.view:
                self.scene.frame(self.view)
            elif session.scene.camera:
                self.scene.set_camera_state(session.scene.camera)
            else:
                self.scene.frame("perspective")
            self._camera_set = True

    def render(self, time: float | None = None) -> np.ndarray:
        """The (h, w, 4) uint8 RGBA image at a time (default: the session's time)."""
        self.show(time)
        self.canvas.request_draw(lambda: self.scene.render(self.renderer))
        image = np.asarray(self.canvas.draw())
        return image[..., :4].astype(np.uint8) if image.dtype != np.uint8 else image


def write_movie(
    session: Session,
    path: str | Path,
    times: list[float] | None = None,
    size: tuple[int, int] = (1920, 1080),
    fps: float = 25.0,
    style: str = "studio",
    view: str | None = None,
    overlay: Overlay | None = None,
    progress: Progress | None = None,
) -> Path:
    """Film the session. Returns what was written: the movie file, or the folder of
    PNGs. Stops early (keeping what was written) when ``progress`` returns False."""
    path = Path(path).expanduser()
    times = frame_times(session) if times is None else list(times)
    if not times:
        raise ValueError("nothing to film: no frames in the session")
    size = (int(size[0]) // 2 * 2, int(size[1]) // 2 * 2)  # H.264 needs even sizes
    view_ = OffscreenView(session, size, style, view)
    sink = _movie_sink(path, size, fps) if path.suffix.lower() in MOVIE_SUFFIXES else None
    folder = None
    if sink is None:
        folder = path.with_suffix("") if path.suffix else path
        folder.mkdir(parents=True, exist_ok=True)
    data = session.data
    # reading several frames ahead, and encoding beside the drawing: the three used to
    # take turns, and reading one frame at a time was most of an export
    readers = ThreadPoolExecutor(READERS, thread_name_prefix="lagoon-movie-read")
    writer = ThreadPoolExecutor(1, thread_name_prefix="lagoon-movie-write")
    spent = {"waiting for data": 0.0, "drawing": 0.0, "overlay": 0.0, "encoding": 0.0}
    started = time_module.perf_counter()

    def preload(when: float) -> None:
        for item_id in data.data_items():
            data.load(item_id, when)

    def write(number: int, image: np.ndarray) -> None:
        if sink is not None:
            sink.send(np.ascontiguousarray(image[..., :3]).tobytes())
        else:
            from lagoonviz.render.offscreen import write_png

            write_png(folder / f"frame_{number:05d}.png", image)

    ahead = {i: readers.submit(preload, times[i]) for i in range(min(READ_AHEAD, len(times)))}
    written = None
    done = 0
    try:
        for number, time in enumerate(times):
            clock = time_module.perf_counter()
            ahead.pop(number).result()  # this frame's data is read
            following = number + READ_AHEAD
            if following < len(times):
                ahead[following] = readers.submit(preload, times[following])
            spent["waiting for data"] += time_module.perf_counter() - clock
            clock = time_module.perf_counter()
            image = view_.render(time)
            spent["drawing"] += time_module.perf_counter() - clock
            if overlay is not None:
                clock = time_module.perf_counter()
                image = overlay(image, time)
                spent["overlay"] += time_module.perf_counter() - clock
            else:
                image = np.array(image, copy=True)  # the next drawing may reuse the buffer
            clock = time_module.perf_counter()
            if written is not None:
                written.result()  # one frame at a time in the encoder's queue
            spent["encoding"] += time_module.perf_counter() - clock
            written = writer.submit(write, number, image)
            done = number + 1
            if progress is not None and not progress(done, len(times)):
                break
        if written is not None:
            clock = time_module.perf_counter()
            written.result()
            spent["encoding"] += time_module.perf_counter() - clock
    finally:
        for future in ahead.values():
            future.cancel()
        readers.shutdown(wait=True, cancel_futures=True)
        writer.shutdown(wait=True)
        if sink is not None:
            clock = time_module.perf_counter()
            sink.close()  # ffmpeg finishes encoding
            spent["encoding"] += time_module.perf_counter() - clock
    total = time_module.perf_counter() - started
    log.info(
        "film: %d frames %dx%d in %.1f s (%.1f frames/s); %s",
        done,
        size[0],
        size[1],
        total,
        done / max(total, 1e-9),
        ", ".join(f"{name} {seconds:.1f} s" for name, seconds in spent.items()),
    )
    return path if sink is not None else folder


def _movie_sink(path: Path, size: tuple[int, int], fps: float):
    """An ffmpeg pipe taking RGB frames, or None when imageio-ffmpeg is missing."""
    try:
        import imageio_ffmpeg
    except ImportError:
        return None
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = imageio_ffmpeg.write_frames(
        str(path),
        size,
        fps=fps,
        codec="libx264",
        pix_fmt_in="rgb24",
        pix_fmt_out="yuv420p",
        quality=None,
        macro_block_size=2,
        ffmpeg_log_level="error",
        # visually lossless, plays everywhere (QuickTime, PowerPoint, browsers)
        # "fast" encodes about twice as fast as "medium" at the same quality setting
        # (files a little larger)
        output_params=["-crf", "17", "-preset", "fast", "-movflags", "+faststart"],
    )
    writer.send(None)  # start ffmpeg
    return writer


def movie_available() -> bool:
    """True if MP4 can be written (imageio-ffmpeg installed)."""
    try:
        import imageio_ffmpeg  # noqa: F401
    except ImportError:
        return False
    return True


def _offscreen_canvas(size: tuple[int, int]):
    try:
        from rendercanvas.offscreen import RenderCanvas
    except ImportError:  # pygfx before rendercanvas
        from wgpu.gui.offscreen import WgpuCanvas as RenderCanvas
    try:
        return RenderCanvas(size=size, pixel_ratio=1)
    except TypeError:
        return RenderCanvas(size=size)
