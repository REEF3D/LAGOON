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


def number_text(value: float, digits: int = 6) -> str:
    """A number as typed in a box: ``digits`` significant digits, the exponent only
    for very small or large ones (k = 1.15e-06, epsilon = 4.2e-10, 2.448, 0)."""
    if not math.isfinite(value):
        return str(value)
    if value == 0:
        return "0"
    text = f"{value:.{digits}g}"
    if "e" in text:
        mantissa, exponent = text.split("e")
        sign = "-" if exponent.startswith("-") else ""
        text = f"{mantissa}e{sign}{exponent.lstrip('+-').lstrip('0').rjust(2, '0')}"
    return text


def parse_number(text: str) -> float | None:
    """The number typed (a decimal point, an exponent allowed); None if it is none."""
    try:
        value = float(text.strip().replace(",", "."))
    except ValueError:
        return None
    return value if math.isfinite(value) else None


def partial_number(text: str) -> bool:
    """Whether text can still become a number by typing on ("", "-", "1.", "2e-")."""
    text = text.strip().replace(",", ".")
    if parse_number(text) is not None or text in ("", "-", "+", ".", "-.", "+."):
        return True
    return parse_number(text + "0") is not None or parse_number(text + "1") is not None


def range_step(low: float, high: float) -> float:
    """The step of the arrows of a colour range's boxes: a tenth of the range's order
    of magnitude (0.1 for 0 to 2.4, 1e-07 for k of 7e-7 to 1.2e-6)."""
    span = abs(float(high) - float(low))
    if not math.isfinite(span) or span <= 0:
        span = max(abs(float(low)), abs(float(high)))
    if not math.isfinite(span) or span <= 0:
        return 0.1
    return 10.0 ** (math.floor(math.log10(span)) - 1)


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
