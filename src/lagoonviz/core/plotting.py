# SPDX-License-Identifier: GPL-3.0-or-later
"""Plot arithmetic shared by the interactive plots and (later) the figure export:
axis ranges, "nice" tick positions and tick labels. No drawing happens here.
"""

from __future__ import annotations

import math
from collections.abc import Iterable

import numpy as np


def nice_step(span: float, target: int = 5) -> float:
    """A step of 1, 2 or 5 times a power of ten giving about ``target`` intervals."""
    if not math.isfinite(span) or span <= 0:
        return 1.0
    raw = span / max(target, 1)
    power = 10.0 ** math.floor(math.log10(raw))
    for factor in (1.0, 2.0, 2.5, 5.0, 10.0):
        if raw <= factor * power * 1.0000001:
            return factor * power
    return 10.0 * power


def nice_ticks(low: float, high: float, target: int = 5) -> list[float]:
    """Tick positions inside [low, high] on a 1-2-5 grid."""
    if not (math.isfinite(low) and math.isfinite(high)) or high <= low:
        return [low] if math.isfinite(low) else []
    step = nice_step(high - low, target)
    first = math.ceil(low / step - 1e-9) * step
    ticks = []
    value = first
    while value <= high + step * 1e-9:
        ticks.append(0.0 if abs(value) < step * 1e-9 else value)
        value += step
    return ticks


def tick_label(value: float, step: float) -> str:
    """Short label with as many decimals as the tick step needs."""
    if value == 0:
        return "0"
    magnitude = max(abs(value), abs(step))
    if magnitude >= 1e5 or magnitude < 1e-3:
        return f"{value:.3g}".replace("e+0", "e").replace("e-0", "e-").replace("e+", "e")
    decimals = max(0, -math.floor(math.log10(step) + 1e-9)) if step < 1 else 0
    return f"{value:.{decimals}f}"


def value_range(arrays: Iterable[np.ndarray], pad: float = 0.06) -> tuple[float, float]:
    """Finite range of several arrays with a little padding; (−1, 1) if empty."""
    lows, highs = [], []
    for values in arrays:
        finite = values[np.isfinite(values)]
        if finite.size:
            lows.append(float(finite.min()))
            highs.append(float(finite.max()))
    if not lows:
        return -1.0, 1.0
    low, high = min(lows), max(highs)
    if high - low < 1e-12 * max(1.0, abs(high)):
        half = max(abs(high) * 0.05, 1e-9)
        return low - half, high + half
    margin = (high - low) * pad
    return low - margin, high + margin
