"""Film encoding: the codec for a size, frames handed to ffmpeg, the summary."""

import numpy as np
import pytest

from lagoonviz.render import movie


def test_large_films_use_the_video_hardware_when_there():
    codec, settings = movie.encoder_settings((3840, 2160), 25, hardware=True)
    assert codec == "h264_videotoolbox"
    assert settings[settings.index("-b:v") + 1] == str(int(0.15 * 3840 * 2160 * 25))
    codec, settings = movie.encoder_settings((3840, 2160), 25, hardware=False)
    assert codec == "libx264" and "veryfast" in settings
    codec, settings = movie.encoder_settings((1920, 1080), 25, hardware=True)
    assert codec == "libx264" and "fast" in settings  # HD: the software encoder is fast enough


def test_rgba_frames_go_to_ffmpeg_without_a_copy(tmp_path):
    pytest.importorskip("imageio_ffmpeg")
    import imageio_ffmpeg

    path = tmp_path / "film.mp4"
    sink, encoder = movie._movie_sink(path, (64, 32), 10)
    image = np.zeros((32, 64, 4), np.uint8)
    image[..., 0], image[..., 3] = 200, 255
    for _ in range(3):
        sink.send(memoryview(np.ascontiguousarray(image).reshape(-1)))
    sink.close()
    frames = imageio_ffmpeg.read_frames(str(path))
    meta = next(frames)
    first = np.frombuffer(next(frames), np.uint8).reshape(32, 64, 3)
    assert meta["size"] == (64, 32) and abs(int(first[16, 32, 0]) - 200) <= 3
    assert encoder in ("libx264", "VideoToolbox")


def test_film_summary():
    movie.LAST_FILM.clear()
    assert movie.film_summary() == ""
    movie.LAST_FILM.update(
        frames=100,
        size=(3840, 2160),
        seconds=50.0,
        encoder="VideoToolbox",
        spent={"waiting for data": 1.0, "drawing": 30.0, "overlay": 4.0, "encoding": 15.0},
    )
    text = movie.film_summary()
    assert "2.0 frames/s" in text and "drawing 60 %" in text and "encoding 30 %" in text
    assert "waiting" not in text  # under 5 %
