# SPDX-License-Identifier: GPL-3.0-or-later
# Architect: Hans Bihs
"""The macOS app icon from LAGOON's square icon: the macOS shape (a rounded square
on the 1024 px grid, 824 px wide with a little margin) at every size an .iconset
needs. ``iconutil -c icns`` makes the .icns from the folder.

    python make_icon.py logo/icon_dark.png build/LAGOON.iconset
"""

from __future__ import annotations

import sys
from pathlib import Path

from PIL import Image, ImageDraw

SIZES = (16, 32, 128, 256, 512)  # each also at @2x
CANVAS, BODY, RADIUS = 1024, 824, 185  # Apple's grid for app icons


def rounded(source: Path) -> Image.Image:
    art = Image.open(source).convert("RGBA").resize((BODY, BODY), Image.LANCZOS)
    scale = 4  # draw the mask large and shrink it: smooth corners
    mask = Image.new("L", (BODY * scale, BODY * scale), 0)
    ImageDraw.Draw(mask).rounded_rectangle(
        (0, 0, BODY * scale - 1, BODY * scale - 1), RADIUS * scale, fill=255
    )
    mask = mask.resize((BODY, BODY), Image.LANCZOS)
    alpha = Image.composite(art.getchannel("A"), Image.new("L", art.size, 0), mask)
    art.putalpha(alpha)
    icon = Image.new("RGBA", (CANVAS, CANVAS), (0, 0, 0, 0))
    offset = (CANVAS - BODY) // 2
    icon.alpha_composite(art, (offset, offset))
    return icon


def main(source: str, folder: str) -> None:
    out = Path(folder)
    out.mkdir(parents=True, exist_ok=True)
    icon = rounded(Path(source))
    for size in SIZES:
        icon.resize((size, size), Image.LANCZOS).save(out / f"icon_{size}x{size}.png")
        icon.resize((2 * size, 2 * size), Image.LANCZOS).save(out / f"icon_{size}x{size}@2x.png")


if __name__ == "__main__":
    main(*sys.argv[1:3])
