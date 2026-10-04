# SPDX-License-Identifier: GPL-3.0-or-later
"""LAGOON colour maps as plain NumPy lookup tables.

The interactive viewport and the figure export (matplotlib, later) use the same
tables, so a colour on screen is the colour in the paper. Each map is a list of
evenly spaced control colours, interpolated linearly in sRGB.
"""

from __future__ import annotations

import numpy as np

from lagoonviz.core.kinds import COLORMAP_ALIASES, COLORMAPS

#: line colours of time-series plots; gauges in 3D take the colour of their plot line
SERIES_PALETTES: dict[str, list[str]] = {
    "light": ["#2b4c7e", "#e07b39", "#3a9a5b", "#4f8fcf", "#c9a227", "#8e5bb0", "#6b6b6b"],
    "dark": ["#5aa2e6", "#ef8a4b", "#56b870", "#e0c04a", "#c27ad1", "#7fc4cc", "#a7b1bc"],
}

_STOPS: dict[str, list[str]] = {
    # REEF3D's look: troughs in REEF3D blue, still water white, crests in warm red
    "reef3d": [
        "#2a5aa0",
        "#3f78bd",
        "#73a7dc",
        "#b8d5ef",
        "#f7f7f5",
        "#f9cdb8",
        "#f19479",
        "#de5a45",
        "#c4302b",
    ],
    # sequential for magnitudes (speed, depth): white to deep REEF3D blue
    "ocean": ["#f5f8fb", "#cfe0f1", "#8fbbe3", "#4f8fcf", "#3f78bd", "#2a5aa0"],
    # slack to taut in the REEF3D colours, without the white middle (lines on white)
    "tension": ["#2a5aa0", "#4f8fcf", "#9cc3e8", "#f19479", "#de5a45", "#c4302b"],
    # REEF3D blues: deep water to foam
    "lagoon_blue": ["#0b1d36", "#1b365d", "#2b4c7e", "#3a6ea8", "#4f8fcf", "#8fbfe6", "#d6ecfa"],
    # diverging, after cmocean "balance": troughs blue, crests red, zero light
    "balance": [
        "#181c43",
        "#1d3d8f",
        "#2a6fb8",
        "#6fa3cc",
        "#c3d7e3",
        "#f1eceb",
        "#e6bfae",
        "#d38a6c",
        "#bc4a37",
        "#8c1a26",
        "#3c0912",
    ],
    "viridis": [
        "#440154",
        "#48186a",
        "#472d7b",
        "#424086",
        "#3b528b",
        "#33638d",
        "#2c728e",
        "#26828e",
        "#21918c",
        "#1fa088",
        "#28ae80",
        "#3fbc73",
        "#5ec962",
        "#84d44b",
        "#addc30",
        "#d8e219",
        "#fde725",
    ],
    # rainbow as in Tecplot, with slightly muted ends so the extremes keep a gradient:
    # muted blue, blue, cyan, green, yellow, red, muted red
    "rainbow": ["#2b2bc4", "#0000ff", "#00ffff", "#00ff00", "#ffff00", "#ff0000", "#c42b2b"],
    # sediment: deep bed dark, shallow bed light sand
    "sand": ["#2f2418", "#5a4630", "#87704f", "#b39c72", "#d9c59a", "#f3e7c9"],
    # dark to mid-light grey: never paper white, so a grey bed stands out on white
    "greys": ["#1c1c1c", "#4a4a4a", "#7a7a7a", "#a6a6a6", "#c8c8c8"],
    # grey to white: low values a mid grey, high values white; plain (coloured by
    # nothing) it is pure white, shaded only by the light
    "white": ["#9ba0a6", "#d4d7db", "#ffffff"],
}

assert set(_STOPS) == set(COLORMAPS), "every colour map in kinds needs a table"


def _rgb(color: str) -> tuple[float, float, float]:
    color = color.lstrip("#")
    return tuple(int(color[i : i + 2], 16) / 255.0 for i in (0, 2, 4))  # type: ignore[return-value]


def lut(name: str, n: int = 256) -> np.ndarray:
    """An ``(n, 4)`` float32 RGBA table for a colour map."""
    name = COLORMAP_ALIASES.get(name, name)
    try:
        stops = np.array([_rgb(c) for c in _STOPS[name]], dtype=np.float64)
    except KeyError:
        raise ValueError(f"unknown colour map {name!r}; known: {', '.join(_STOPS)}") from None
    x = np.linspace(0.0, 1.0, len(stops))
    t = np.linspace(0.0, 1.0, n)
    rgb = np.stack([np.interp(t, x, stops[:, c]) for c in range(3)], axis=1)
    return np.concatenate([rgb, np.ones((n, 1))], axis=1).astype(np.float32)


def normalize(values: np.ndarray, vmin: float, vmax: float) -> np.ndarray:
    """Map values to [0, 1] for a colour range (NaN stays NaN)."""
    span = vmax - vmin if vmax != vmin else 1.0
    return np.clip((values - vmin) / span, 0.0, 1.0).astype(np.float32)


def colors(values: np.ndarray, name: str, vmin: float, vmax: float) -> np.ndarray:
    """RGBA colours for values (used by software paths and tests)."""
    table = lut(name)
    index = np.nan_to_num(normalize(values, vmin, vmax) * (len(table) - 1)).astype(int)
    return table[index]


#: colours of items shown without a variable
PLAIN_COLORS = {
    "free_surface": "#4f8fcf",
    "bathymetry": "#c4ae86",  # sand
    "volume": "#9aa7b4",
    "slice": "#9aa7b4",
    "mooring": "#e0a030",
    "body": "#8f9aa6",
    "structure": "#a7a39b",
    "dem": "#6f6a64",  # rock
    "ice": "#e3f1fa",
    "rods": "#9c7a4f",
    "iso_surface": "#8fb3d1",
    "streamlines": "#2b4c7e",
}
#: colours of the materials of bodies and structures
MATERIAL_COLORS = {
    "steel": "#8f9aa6",
    "concrete": "#b4afa6",
    "rock": "#6f6a64",
    "ice": "#e3f1fa",
    "wood": "#9c7a4f",
}


#: the one colour of a colour map for a surface coloured by nothing
MAP_PLAIN_COLORS = {"sand": "#c4ae86", "greys": "#9a9a9a", "white": "#ffffff"}


def map_plain_color(name: str) -> str:
    """One colour standing for a colour map (its middle, unless set above)."""
    name = COLORMAP_ALIASES.get(name, name)
    if name in MAP_PLAIN_COLORS:
        return MAP_PLAIN_COLORS[name]
    r, g, b, _ = lut(name, 3)[1]
    return f"#{round(r * 255):02x}{round(g * 255):02x}{round(b * 255):02x}"


def plain_color(
    kind: str,
    material: str | None = None,
    colormap: str | None = None,
    default_colormap: str | None = None,
) -> str:
    """The colour of an item drawn without a variable: its material's, else the
    kind's own while the colour map is the kind's default, else the colour map's
    (a bed with the "white" map is white)."""
    if material in MATERIAL_COLORS:
        return MATERIAL_COLORS[material]
    if colormap is not None and colormap != default_colormap:
        try:
            return map_plain_color(colormap)
        except ValueError:
            pass
    return PLAIN_COLORS.get(kind, "#cccccc")
