# SPDX-License-Identifier: GPL-3.0-or-later
# Architect: Hans Bihs
"""The legend drawn on film frames: the time and the colour bars of the coloured
items, where the session's legend layout puts them (lagoonviz.core.legend).

Drawn with Pillow, so films made by scripts have it too:

    write_movie(session, "wave.mp4", legend=True)
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np

from lagoonviz.core import Session
from lagoonviz.core.colormaps import lut
from lagoonviz.core.legend import colour_bars, place, plain

INK = (30, 40, 51, 255)
BOX = (255, 255, 255, 170)


@lru_cache(maxsize=16)
def _font(size: int):
    from PIL import ImageFont

    try:
        from matplotlib import font_manager

        return ImageFont.truetype(font_manager.findfont("DejaVu Sans"), size)
    except Exception:  # noqa: BLE001 (no matplotlib or font: Pillow's own)
        try:
            return ImageFont.load_default(size=size)
        except TypeError:  # Pillow < 10.1
            return ImageFont.load_default()


class LegendOverlay:
    """``overlay(image, time) -> image`` for :func:`lagoonviz.render.movie.write_movie`:
    the time and the colour bars on an RGBA frame."""

    def __init__(self, session: Session) -> None:
        self.session = session

    def __call__(self, image: np.ndarray, time: float) -> np.ndarray:
        from PIL import Image, ImageDraw

        session = self.session
        settings = session.legend_layout()
        bars = colour_bars(session)
        height, width = image.shape[:2]
        seconds = session.data.time_in_seconds()
        label = f"t = {time:.2f} s" if seconds else f"output {round(time)}"

        def measure(size: float) -> float:
            font = _font(max(int(round(size)), 6))
            return (
                float(font.getlength(label))
                if hasattr(font, "getlength")
                else 0.6 * size * len(label)
            )

        spots = place(width, height, len(bars), settings, time_width=measure)
        picture = Image.fromarray(np.ascontiguousarray(image[..., :4], dtype=np.uint8), "RGBA")
        layer = Image.new("RGBA", picture.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(layer)
        for frame in (spots.frame, spots.time_frame):
            if frame is not None:
                radius = int(4 * spots.unit)
                draw.rounded_rectangle(
                    (frame.x, frame.y, frame.right, frame.bottom), radius, fill=BOX
                )
        if spots.time is not None:
            font = _font(max(int(round(spots.time_font)), 6))
            box = spots.time
            right = "right" in settings["time"]
            anchor = "ra" if right else "la"
            draw.text(
                (box.right if right else box.x, box.y), label, font=font, fill=INK, anchor=anchor
            )
        font = _font(max(int(round(spots.font)), 6))
        for bar, title, strip, labels in zip(
            bars, spots.titles, spots.bars, spots.labels, strict=True
        ):
            draw.text((title.x, title.y), plain(bar.label), font=font, fill=INK)
            table = (np.asarray(lut(bar.colormap, 256))[:, :3] * 255).round().astype(np.uint8)
            columns = max(int(round(strip.w)), 1)
            rows = max(int(round(strip.h)), 1)
            ramp = table[np.linspace(0, len(table) - 1, columns).round().astype(int)]
            gradient = Image.fromarray(np.repeat(ramp[None], rows, axis=0), "RGB")
            layer.paste(gradient.convert("RGBA"), (int(round(strip.x)), int(round(strip.y))))
            draw.rectangle(
                (strip.x, strip.y, strip.x + columns, strip.y + rows),
                outline=(255, 255, 255, 120),
                width=max(int(spots.unit), 1),
            )
            span = bar.vmax - bar.vmin
            for k, (value, text) in enumerate(bar.ticks):
                where = 0.0 if span == 0 else (value - bar.vmin) / span
                x = strip.x + min(max(where, 0.0), 1.0) * columns
                anchor = "la" if k == 0 else "ra" if k == len(bar.ticks) - 1 else "ma"
                if 0 < k < len(bar.ticks) - 1:
                    draw.line(
                        (x, strip.bottom, x, strip.bottom + 3 * spots.unit),
                        fill=INK,
                        width=max(int(spots.unit), 1),
                    )
                draw.text((x, labels.y), text, font=font, fill=INK, anchor=anchor)
        out = Image.alpha_composite(picture, layer)
        return np.asarray(out, dtype=np.uint8).copy()
