# SPDX-License-Identifier: GPL-3.0-or-later
"""LAGOON colour maps as plain NumPy lookup tables.

The interactive viewport and the figure export (matplotlib, later) use the same
tables, so a colour on screen is the colour in the paper. Each map is a list of
evenly spaced control colours, interpolated linearly in sRGB.
"""

from __future__ import annotations

import numpy as np

from lagoonviz.core.kinds import COLORMAPS

_STOPS: dict[str, list[str]] = {
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
    "cividis": [
        "#00224e",
        "#002e6a",
        "#1a386f",
        "#32436d",
        "#434e6c",
        "#535a6d",
        "#61656f",
        "#6f7073",
        "#7d7c78",
        "#8c8878",
        "#9b9476",
        "#aba072",
        "#bcae6c",
        "#cdbb63",
        "#dec958",
        "#f0d846",
        "#fee838",
    ],
    # sediment: deep bed dark, shallow bed light sand
    "sand": ["#2f2418", "#5a4630", "#87704f", "#b39c72", "#d9c59a", "#f3e7c9"],
    "greys": ["#111111", "#ffffff"],
}

assert set(_STOPS) == set(COLORMAPS), "every colour map in kinds needs a table"


def _rgb(color: str) -> tuple[float, float, float]:
    color = color.lstrip("#")
    return tuple(int(color[i : i + 2], 16) / 255.0 for i in (0, 2, 4))  # type: ignore[return-value]


def lut(name: str, n: int = 256) -> np.ndarray:
    """An ``(n, 4)`` float32 RGBA table for a colour map."""
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
