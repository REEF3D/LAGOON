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
import subprocess
import sys
import tempfile
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
#: frames read ahead while one is drawn, and the threads reading them (balanced
#: memory use; lagoonviz.core.memory sets them for each mode)
READ_AHEAD = 6
READERS = 3

log = logging.getLogger("lagoon.movie")

#: from this width up, films are encoded by the Mac's video hardware when it can
#: (4K with libx264 took longer than drawing the frames)
HARDWARE_WIDTH = 2560
#: bits per pixel and frame for the hardware encoder (about 31 Mbit/s for 4K at 25 fps)
HARDWARE_BITS = 0.15
#: how the last film was made: frames, size, seconds, encoder, time spent per stage
LAST_FILM: dict = {}
_HARDWARE: dict[str, bool] = {}


def frame_times(
    session: Session, start: int | None = None, end: int | None = None, every: int = 1
) -> list[float]:
    """Times of the frames to film: the timeline's frames ``start`` to ``end`` (both
    included, numbered as the timeline shows them; default: the session's play
    range, else all), every ``every``-th one. With a run log the times are the
    logged ones."""
    times, steps, _ = session.data.timeline()
    if not len(steps):
        return []
    played = session.scene.play_range
    if played is not None:
        nearest = [int(steps[int(np.abs(np.asarray(times) - t).argmin())]) for t in played]
        start = nearest[0] if start is None else start
        end = nearest[1] if end is None else end
    start = int(steps[0]) if start is None else int(start)
    end = int(steps[-1]) if end is None else int(end)
    chosen = np.nonzero((steps >= start) & (steps <= end))[0][:: max(int(every), 1)]
    return [float(times[i]) for i in chosen]


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
        if view and view.startswith("2d"):  # "2d" (the session's 2D side), "2d-front", ...
            side = view.partition("-")[2] or None
            session.set_plan_view(True, side)  # as the 2D view shows it
        self.size = (int(size[0]), int(size[1]))
        self.canvas = _offscreen_canvas(self.size)
        self.renderer = gfx.WgpuRenderer(self.canvas)
        self.scene = SceneRenderer(style)
        self.view = view
        self._camera_set = False
        #: what each "Case set-up" item was last drawn from (DataManager.setup_signature)
        self._setup_drawn: dict[str, tuple | None] = {}
        #: the background maps' credits in the corner of the image (figures put
        #: them in their axes instead)
        self.credit = True

    def show(self, time: float | None = None) -> None:
        """Bring the scene to a time: load the frames, update meshes and gauges."""
        session = self.session
        if time is not None:
            session.set_time(time)
        for item_id in session.data.data_items():
            item = session.scene.get(item_id)
            if not session.data.drawn(item_id):  # results items in the set-up
                self.scene.remove(item_id)
                continue
            mesh = session.data.load(item_id)
            if mesh is None:
                self.scene.remove(item_id)  # no output of this run at this time
            else:
                shown = session.data.display(item_id, mesh)
                self.scene.update(item.kind, shown)
                self.scene.update_contours(
                    item_id, session.data.contour_lines(item_id, mesh, shown)
                )
                self.scene.update_vectors(item_id, session.data.vector_arrows(item_id, mesh, shown))
                self.scene.update_trails(item_id, session.data.particle_trails(item_id, mesh))
                self.scene.update_outlines(
                    item_id, session.data.patch_outlines(item_id, mesh, shown)
                )
        profiles = set(session.data.profile_items())
        for item_id in session.data.series_items() + sorted(profiles):
            data = session.data
            marks = (
                data.profile_marks(item_id) if item_id in profiles else data.gauge_marks(item_id)
            )
            if marks is None or not data.drawn(item_id):
                self.scene.remove_gauges(item_id)
            else:
                self.scene.update_gauges(marks)
        for item_id in session.data.setup_items():  # the "Case set-up" items
            signature = session.data.setup_signature(item_id)
            if signature is not None and self._setup_drawn.get(item_id) == signature:
                continue  # unchanged since the last frame
            marks = session.data.setup_marks(item_id)
            if marks is None:
                self.scene.remove_setup(item_id)
            else:
                self.scene.update_setup(marks)
            self._setup_drawn[item_id] = signature
        backgrounds = session.data.backgrounds
        for item_id in self.scene.background_ids:
            if item_id not in backgrounds.items():
                self.scene.remove_background(item_id)
        for item_id in backgrounds.items():  # background maps (made now if needed)
            made = backgrounds.image(item_id) if session.data.drawn(item_id) else None
            if made is None:
                self.scene.remove_background(item_id)
            else:
                self.scene.update_background(backgrounds.shown(item_id, made))
        if not self._camera_set:
            cases = [i.id for i in session.scene if i.kind == "case"]
            norths = [session.data.backgrounds.north(c) for c in cases]
            self.scene.set_north(next((n for n in norths if n is not None), None))
            plan = session.scene
            if plan.flat:
                self.scene.set_plan(True)
            two_d = (self.view or "").startswith("2d")
            if two_d or (plan.plan_view and not self.view and not plan.camera):
                # orthographic from the 2D view's side, as the 2D view
                self.scene.set_orthographic(True)
                self.scene.frame(plan.plan_side)
            elif self.view:
                self.scene.frame(self.view)
            elif session.scene.camera:
                self.scene.set_camera_state(session.scene.camera)
            else:
                self.scene.frame("perspective")
            self._camera_set = True
        if session.scene.sun is not None or self.scene.sun_override is not None:
            self.scene.set_sun_direction(session.sun_direction())  # the session's sun
        # the camera path through the keyframes, when it is on: the camera of this time
        state = session.camera_at()
        if state is not None:
            self.scene.set_camera_state(state)

    def render(self, time: float | None = None) -> np.ndarray:
        """The (h, w, 4) uint8 RGBA image at a time (default: the session's time)."""
        self.show(time)
        self.canvas.request_draw(lambda: self.scene.render(self.renderer))
        image = np.asarray(self.canvas.draw())
        image = image[..., :4].astype(np.uint8) if image.dtype != np.uint8 else image
        if self.credit:
            image = credit(image, self.session.data.backgrounds.attributions())
        return image


def credit(image: np.ndarray, credits: list[str]) -> np.ndarray:
    """The credits of the background maps in the lower right corner of a frame
    (their sources' terms ask for them)."""
    if not credits:
        return image
    from PIL import Image, ImageDraw, ImageFont

    picture = Image.fromarray(np.ascontiguousarray(image[..., :4]), "RGBA")
    size = max(10, picture.height // 60)
    try:
        font = ImageFont.load_default(size=size)
    except TypeError:  # Pillow < 10.1
        font = ImageFont.load_default()
    draw = ImageDraw.Draw(picture, "RGBA")
    text = " · ".join(credits)
    left, top, right, bottom = draw.textbbox((0, 0), text, font=font)
    pad = size // 3
    x = picture.width - (right - left) - 2 * pad
    y = picture.height - (bottom - top) - 2 * pad
    draw.rectangle((x, y, picture.width, picture.height), fill=(255, 255, 255, 150))
    draw.text((x + pad - left, y + pad - top), text, fill=(28, 35, 48, 255), font=font)
    return np.asarray(picture, np.uint8)


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
    sink, encoder = (
        _movie_sink(path, size, fps) if path.suffix.lower() in MOVIE_SUFFIXES else (None, "PNG")
    )
    folder = None
    if sink is None:
        folder = path.with_suffix("") if path.suffix else path
        folder.mkdir(parents=True, exist_ok=True)
    data = session.data
    # reading several frames ahead, and encoding beside the drawing: the three used to
    # take turns, and reading one frame at a time was most of an export
    from lagoonviz.core import memory

    use = memory.current()
    read_ahead = use.film_read_ahead
    readers = ThreadPoolExecutor(use.film_readers, thread_name_prefix="lagoon-movie-read")
    writer = ThreadPoolExecutor(1, thread_name_prefix="lagoon-movie-write")
    spent = {"waiting for data": 0.0, "drawing": 0.0, "overlay": 0.0, "encoding": 0.0}
    started = time_module.perf_counter()

    def preload(when: float) -> None:
        for item_id in data.data_items():
            data.load(item_id, when)

    def write(number: int, image: np.ndarray) -> None:
        if sink is not None:
            # RGBA as drawn, without a copy (ffmpeg drops the alpha)
            sink.send(memoryview(np.ascontiguousarray(image).reshape(-1)))
        else:
            from lagoonviz.render.offscreen import write_png

            write_png(folder / f"frame_{number:05d}.png", image)

    ahead = {i: readers.submit(preload, times[i]) for i in range(min(read_ahead, len(times)))}
    written = None
    done = 0
    try:
        for number, time in enumerate(times):
            clock = time_module.perf_counter()
            ahead.pop(number).result()  # this frame's data is read
            following = number + read_ahead
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
    LAST_FILM.clear()
    LAST_FILM.update(frames=done, size=size, seconds=total, encoder=encoder, spent=dict(spent))
    log.info(
        "film: %d frames %dx%d in %.1f s (%.1f frames/s, %s); %s",
        done,
        size[0],
        size[1],
        total,
        done / max(total, 1e-9),
        encoder,
        ", ".join(f"{name} {seconds:.1f} s" for name, seconds in spent.items()),
    )
    return path if sink is not None else folder


def film_summary() -> str:
    """How the last film went, for the status bar: frames per second and where the
    time went."""
    if not LAST_FILM:
        return ""
    total = max(LAST_FILM["seconds"], 1e-9)
    width, height = LAST_FILM["size"]
    stages = ", ".join(
        f"{name} {100 * seconds / total:.0f} %"
        for name, seconds in LAST_FILM["spent"].items()
        if seconds / total >= 0.05
    )
    return (
        f"{LAST_FILM['frames']} frames {width}×{height} in {total:.0f} s "
        f"({LAST_FILM['frames'] / total:.1f} frames/s, {LAST_FILM['encoder']}; {stages})"
    )


def hardware_encoder() -> bool:
    """True if ffmpeg can encode H.264 with the Mac's video hardware (VideoToolbox):
    tried once with a tiny film."""
    if "h264_videotoolbox" in _HARDWARE:
        return _HARDWARE["h264_videotoolbox"]
    works = False
    if sys.platform == "darwin":
        try:
            import imageio_ffmpeg

            with tempfile.TemporaryDirectory() as folder:
                result = subprocess.run(
                    [
                        imageio_ffmpeg.get_ffmpeg_exe(),
                        "-hide_banner",
                        "-loglevel",
                        "error",
                        "-f",
                        "lavfi",
                        "-i",
                        "color=c=blue:s=256x144:d=0.2",
                        "-c:v",
                        "h264_videotoolbox",
                        "-b:v",
                        "1M",
                        str(Path(folder) / "probe.mp4"),
                    ],
                    capture_output=True,
                    timeout=20,
                )
                works = result.returncode == 0 and (Path(folder) / "probe.mp4").stat().st_size > 0
        except Exception:  # noqa: BLE001 (no ffmpeg, no encoder: the software one is used)
            works = False
    _HARDWARE["h264_videotoolbox"] = works
    return works


def encoder_settings(size: tuple[int, int], fps: float, hardware: bool) -> tuple[str, list[str]]:
    """The codec and its settings: the Mac's video hardware for large films when it
    is there (several times faster than libx264 at 4K), else libx264."""
    common = ["-movflags", "+faststart"]
    if hardware and size[0] >= HARDWARE_WIDTH:
        rate = int(HARDWARE_BITS * size[0] * size[1] * fps)
        return "h264_videotoolbox", [
            "-b:v",
            str(rate),
            "-maxrate",
            str(int(1.5 * rate)),
            "-bufsize",
            str(2 * rate),
            "-profile:v",
            "high",
            *common,
        ]
    # visually lossless, plays everywhere (QuickTime, PowerPoint, browsers); "fast"
    # encodes about twice as fast as "medium", "veryfast" (large films) again faster
    preset = "veryfast" if size[0] >= HARDWARE_WIDTH else "fast"
    return "libx264", ["-crf", "17", "-preset", preset, *common]


def _movie_sink(path: Path, size: tuple[int, int], fps: float):
    """An ffmpeg pipe taking RGBA frames and the encoder's name, or (None, "PNG")
    when imageio-ffmpeg is missing."""
    try:
        import imageio_ffmpeg
    except ImportError:
        return None, "PNG"
    path.parent.mkdir(parents=True, exist_ok=True)
    hardware = size[0] >= HARDWARE_WIDTH and hardware_encoder()
    codec, settings = encoder_settings(size, fps, hardware)
    writer = imageio_ffmpeg.write_frames(
        str(path),
        size,
        fps=fps,
        codec=codec,
        pix_fmt_in="rgba",
        pix_fmt_out="yuv420p",
        quality=None,
        macro_block_size=2,
        ffmpeg_log_level="error",
        output_params=settings,
    )
    writer.send(None)  # start ffmpeg
    return writer, {"h264_videotoolbox": "VideoToolbox"}.get(codec, codec)


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
