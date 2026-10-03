"""Agreed snapshots: what the window looks like in structure, and how views render.

Once a behaviour or a look is agreed, its snapshot is kept in ``tests/snapshots``.
A later change that alters it fails the test with a diff (text) or the changed
images (render), so nothing changes by accident. An intended change is accepted
with::

    pytest --update-snapshots

which rewrites the snapshots; review them (git diff, the images) and commit them.

* text snapshots (``tests/snapshots/<name>.txt``): menus, toolbar, panels, the
  properties of every item kind. Plain text, so a change reads as a diff.
* reference images (``tests/snapshots/images/<platform>/<name>.png``): the 3D view
  rendered off screen. Kept per platform (GPUs differ a little) and compared with a
  small tolerance. A platform without references (e.g. a new CI runner) creates
  them and skips; set ``LAGOON_REQUIRE_SNAPSHOTS=1`` to fail instead.

A snapshot that does not exist yet is created and the test is skipped with a note,
so the first run on a machine records the agreed state.
"""

from __future__ import annotations

import difflib
import os
import struct
import sys
import zlib
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).parent / "snapshots"

#: render tolerance: mean colour difference (0..255) and the share of pixels allowed
#: to differ clearly (more than ``CLEAR`` in some channel)
MEAN_LIMIT = 2.0
CLEAR = 40
SHARE_LIMIT = 0.004


class Snapshots:
    def __init__(self, update: bool, output: Path):
        self.update = update
        self.output = output  # where failing images go

    def _missing(self, path: Path) -> None:
        if os.environ.get("LAGOON_REQUIRE_SNAPSHOTS"):
            pytest.fail(f"no snapshot {path.relative_to(ROOT)}; record it with --update-snapshots")
        pytest.skip(f"snapshot {path.relative_to(ROOT)} recorded now: review and commit it")

    # ------------------------------------------------------------- text
    def check_text(self, name: str, text: str) -> None:
        path = ROOT / f"{name}.txt"
        text = text.rstrip() + "\n"
        if self.update or not path.exists():
            existed = path.exists()
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text, encoding="utf-8")
            if not existed and not self.update:
                self._missing(path)
            return
        expected = path.read_text(encoding="utf-8")
        if expected == text:
            return
        diff = "".join(
            difflib.unified_diff(
                expected.splitlines(keepends=True),
                text.splitlines(keepends=True),
                f"agreed {path.relative_to(ROOT)}",
                "now",
            )
        )
        pytest.fail(
            f"{name} changed:\n{diff}\nIf this change is intended: pytest --update-snapshots",
            pytrace=False,
        )

    # ------------------------------------------------------------- images
    def check_image(self, name: str, image: np.ndarray) -> None:
        from lagoonviz.render.offscreen import write_png

        path = ROOT / "images" / sys.platform / f"{name}.png"
        if self.update or not path.exists():
            existed = path.exists()
            path.parent.mkdir(parents=True, exist_ok=True)
            write_png(path, image)
            if not existed and not self.update:
                self._missing(path)
            return
        expected = read_png(path)
        problem = compare_images(expected, image)
        if problem is None:
            return
        self.output.mkdir(parents=True, exist_ok=True)
        write_png(self.output / f"{name}-now.png", image)
        write_png(self.output / f"{name}-agreed.png", expected)
        if expected.shape == image.shape:
            write_png(self.output / f"{name}-diff.png", difference_image(expected, image))
        pytest.fail(
            f"{name} renders differently: {problem}\n"
            f"Images: {self.output}/{name}-now.png, -agreed.png, -diff.png\n"
            "If this change is intended: pytest --update-snapshots",
            pytrace=False,
        )


def compare_images(expected: np.ndarray, actual: np.ndarray) -> str | None:
    """None when the images match within the tolerance, else what differs."""
    if expected.shape[:2] != actual.shape[:2]:
        now, agreed = actual.shape, expected.shape
        return f"size {now[1]}x{now[0]}, agreed {agreed[1]}x{agreed[0]}"
    diff = np.abs(expected[..., :3].astype(int) - actual[..., :3].astype(int)).max(axis=-1)
    mean = float(diff.mean())
    share = float((diff > CLEAR).mean())
    if mean > MEAN_LIMIT or share > SHARE_LIMIT:
        return (
            f"mean difference {mean:.2f} (limit {MEAN_LIMIT}), "
            f"{share:.2%} of pixels clearly different (limit {SHARE_LIMIT:.2%})"
        )
    return None


def difference_image(expected: np.ndarray, actual: np.ndarray) -> np.ndarray:
    """Grey copy of the agreed image with the differing pixels in magenta."""
    grey = expected[..., :3].astype(float).mean(axis=-1, keepdims=True)
    out = np.repeat(0.35 * grey + 140, 3, axis=-1)
    diff = np.abs(expected[..., :3].astype(int) - actual[..., :3].astype(int)).max(axis=-1)
    out[diff > 8] = (230, 0, 160)
    return out.clip(0, 255).astype(np.uint8)


def read_png(path: str | Path) -> np.ndarray:
    """Read an 8-bit RGB or RGBA PNG (as written by ``write_png``; all row filters)."""
    data = Path(path).read_bytes()
    if data[:8] != b"\x89PNG\r\n\x1a\n":
        raise ValueError(f"{path}: not a PNG")
    pos, idat, header = 8, b"", None
    while pos < len(data):
        (length,) = struct.unpack(">I", data[pos : pos + 4])
        tag = data[pos + 4 : pos + 8]
        body = data[pos + 8 : pos + 8 + length]
        if tag == b"IHDR":
            header = struct.unpack(">IIBBBBB", body)
        elif tag == b"IDAT":
            idat += body
        pos += 12 + length
    width, height, depth, color_type, _, _, interlace = header
    if depth != 8 or color_type not in (2, 6) or interlace:
        raise ValueError(f"{path}: only 8-bit RGB/RGBA, not interlaced")
    channels = 4 if color_type == 6 else 3
    stride = width * channels
    raw = np.frombuffer(zlib.decompress(idat), np.uint8).reshape(height, stride + 1)
    out = np.zeros((height, stride), np.int32)
    previous = np.zeros(stride, np.int32)
    for y in range(height):
        kind, row = raw[y, 0], raw[y, 1:].astype(np.int32)
        if kind == 0:
            line = row
        elif kind == 2:
            line = (row + previous) & 255
        else:  # 1 sub, 3 average, 4 Paeth: depend on the pixel to the left
            line = np.zeros(stride, np.int32)
            for x in range(stride):
                left = line[x - channels] if x >= channels else 0
                up = previous[x]
                if kind == 1:
                    predictor = left
                elif kind == 3:
                    predictor = (left + up) // 2
                else:
                    corner = previous[x - channels] if x >= channels else 0
                    p = left + up - corner
                    pa, pb, pc = abs(p - left), abs(p - up), abs(p - corner)
                    predictor = left if pa <= pb and pa <= pc else up if pb <= pc else corner
                line[x] = (row[x] + predictor) & 255
        out[y] = line
        previous = line
    return out.astype(np.uint8).reshape(height, width, channels)
