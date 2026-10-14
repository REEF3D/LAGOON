# SPDX-License-Identifier: GPL-3.0-or-later
# Architect: Hans Bihs
"""Legends: the colour bars of the coloured items and where films put them.

- Each coloured item says whether it has a colour bar, its title, how its numbers
  are written and how many ticks it has (item properties ``legend``,
  ``legend_title``, ``legend_format``, ``legend_ticks``): the 3D view's bars, the
  films' and the figures' follow them.
- The session's legend layout (``Scene.legend``, :data:`LAYOUT`) says where films
  and saved images put the colour bars and the time, how long and thick the bars
  are, the size of the text, a box behind it, and whether the 3D view shows its
  bars under the view.
- :func:`place` lays them out on an image (pixels), for whatever paints them.

No Qt here.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field

from lagoonviz.core.plotting import nice_step, tick_label

#: how the numbers of a colour bar are written
NUMBER_FORMATS = ("auto", "integer", "1 decimal", "2 decimals", "3 decimals", "scientific")
#: corners of an image for the colour bars and the time of films and saved images
CORNERS = ("lower left", "lower right", "upper left", "upper right")
#: where the time goes on films: a corner, or nowhere
TIME_PLACES = (*CORNERS, "off")
#: where the 3D view shows its colour bars
VIEW_PLACES = ("below the view", "off")

#: the legend layout of films and saved images, and of the 3D view's bars
LAYOUT: dict = {
    "bars": "lower left",  # corner of the colour bars
    "time": "upper left",  # corner of the time, or "off"
    "length": 0.28,  # a bar's length, of the image's width
    "thickness": 1.0,  # the bars' thickness, 1: 14 px in a 1080-line image
    "text": 1.0,  # text size, 1: 20 px (the time 26 px) in a 1080-line image
    "box": False,  # a light box behind the bars and the time
    "view": "below the view",  # the 3D view's bars
}
#: limits of the layout's numbers
LIMITS = {"length": (0.05, 0.95), "thickness": (0.3, 4.0), "text": (0.3, 4.0)}


def layout(values: dict | None = None) -> dict:
    """A full, checked legend layout from some settings (the rest as :data:`LAYOUT`).
    Unknown keys and values out of their lists raise ``ValueError``."""
    out = dict(LAYOUT)
    for key, value in (values or {}).items():
        if key not in LAYOUT:
            raise ValueError(f"legend has no setting {key!r}; known: {', '.join(LAYOUT)}")
        if key == "bars":
            value = _choice(key, value, CORNERS)
        elif key == "time":
            value = _choice(key, value, TIME_PLACES)
        elif key == "view":
            value = _choice(key, value, VIEW_PLACES)
        elif key == "box":
            value = bool(value)
        else:
            low, high = LIMITS[key]
            value = min(max(float(value), low), high)
        out[key] = value
    return out


def _choice(key: str, value, choices: tuple[str, ...]) -> str:
    value = str(value)
    if value not in choices:
        raise ValueError(f"legend {key} must be one of {', '.join(choices)}, got {value!r}")
    return value


# ======================================================================== numbers
def format_number(value: float, how: str = "auto", step: float | None = None) -> str:
    """A number of a colour bar as ``how`` (:data:`NUMBER_FORMATS`) writes it; "auto"
    as many decimals as the steps between ticks need (else 4 significant digits)."""
    if not math.isfinite(value):
        return str(value)
    if how == "integer":
        return f"{value:.0f}".replace("-0", "0") if round(value) == 0 else f"{value:.0f}"
    if how in ("1 decimal", "2 decimals", "3 decimals"):
        text = f"{value:.{int(how[0])}f}"
        return text[1:] if text.startswith("-") and float(text) == 0 else text
    if how == "scientific":
        return f"{value:.2e}"
    if step:
        return tick_label(value, step)
    if value == 0:
        return "0"
    if abs(value) >= 1e4 or abs(value) < 1e-3:
        return f"{value:.2e}"
    return f"{value:.4g}"


def bar_ticks(vmin: float, vmax: float, count: int = 0, how: str = "auto"):
    """The ticks of a colour bar: (value, text) pairs. ``count`` 0: its two ends; 1:
    the ends and the middle; else ``count`` ticks evenly from end to end."""
    if count <= 0 or not (math.isfinite(vmin) and math.isfinite(vmax)) or vmin == vmax:
        values = [vmin, vmax]
    elif count == 1:
        values = [vmin, (vmin + vmax) / 2, vmax]
    else:
        values = [vmin + (vmax - vmin) * k / (count - 1) for k in range(count)]
    step = abs(vmax - vmin) / max(len(values) - 1, 1)
    step = nice_step(step * 10, 10) / 10 if step > 0 else None
    return [(v, format_number(v, how, step if how == "auto" else None)) for v in values]


# ======================================================================== bars
@dataclass
class ColourBar:
    """The colour scale of an item: its title, colour map, range and ticks."""

    label: str
    colormap: str
    vmin: float
    vmax: float
    ticks: list[tuple[float, str]] = field(default_factory=list)
    item_id: str = ""
    #: the item's tick count (0: the ends; figures then place their own)
    count: int = 0
    number_format: str = "auto"

    def __post_init__(self) -> None:
        if not self.ticks:
            self.ticks = bar_ticks(self.vmin, self.vmax, self.count, self.number_format)


def bar_title(item) -> str:
    """The title of an item's colour bar: its own, else "name · variable"."""
    props = item.properties
    title = str(props.get("legend_title", "") or "").strip()
    return title or f"{item.name} · {props.get('variable', 'none')}"


def colour_bars(session, loaded: dict | None = None) -> list[ColourBar]:
    """The colour bars of the visible coloured items that have one (``legend``), in
    the scene's order. ``loaded``: item id -> mesh already at hand (the 3D view's);
    items missing there are read (blocking)."""
    data = session.data
    bars = []
    for item_id in data.data_items():
        item = session.scene.get(item_id)
        props = item.properties
        variable = str(props.get("variable", "none"))
        if not item.visible or variable == "none" or not props.get("legend", True):
            continue
        case = data.case_item(item_id)
        if case is not None and not case.visible:
            continue
        mesh = (loaded or {}).get(item_id) if loaded is not None else data.load(item_id)
        if mesh is None:
            continue
        try:
            vmin, vmax = data.value_range(item_id, mesh)
        except KeyError:
            continue  # a variable the frame does not have (still loading)
        bars.append(
            ColourBar(
                bar_title(item),
                str(props.get("colormap", "viridis")),
                float(vmin),
                float(vmax),
                item_id=item_id,
                count=int(props.get("legend_ticks", 0) or 0),
                number_format=str(props.get("legend_format", "auto") or "auto"),
            )
        )
    return bars


# ======================================================================== layout
@dataclass
class Box:
    """A rectangle in pixels: left, top, width, height."""

    x: float
    y: float
    w: float
    h: float

    @property
    def right(self) -> float:
        return self.x + self.w

    @property
    def bottom(self) -> float:
        return self.y + self.h


@dataclass
class Placed:
    """Where the parts of a film's legend go on an image (pixels)."""

    unit: float  # pixels per pixel of a 1080-line image
    font: float  # bar text height [px]
    time_font: float  # the time's text height [px]
    time: Box | None  # the time's text
    titles: list[Box]  # each bar's title
    bars: list[Box]  # each bar's gradient
    labels: list[Box]  # the room of each bar's tick labels (under the bar)
    frame: Box | None  # the box behind the bars (with ``box``)
    time_frame: Box | None  # the box behind the time (with ``box``)


def place(
    width: int,
    height: int,
    count: int,
    settings: dict | None = None,
    time_shown: bool = True,
    time_width=None,
) -> Placed:
    """Lay out ``count`` colour bars and the time on a ``width`` x ``height`` image
    as the legend layout says: the bars stacked in their corner (the first nearest
    the image's edge it starts from), the time in its own. ``time_width(font)``:
    the time's text width [px] in a font of that height (default: a guess)."""
    s = layout(settings)
    unit = max(height / 1080.0, 0.4)
    font = 20.0 * unit * s["text"]
    time_font = 26.0 * unit * s["text"]
    margin = 28.0 * unit
    thick = 14.0 * unit * s["thickness"]
    gap = 6.0 * unit
    title_h = font * 1.25
    label_h = font * 1.25
    row = title_h + 2 * unit + thick + 2 * unit + label_h + gap
    length = s["length"] * width
    pad = 8.0 * unit if s["box"] else 0.0
    left = "left" in s["bars"]
    upper = "upper" in s["bars"]
    x = margin + pad if left else width - margin - pad - length
    total = count * row - (gap if count else 0.0)
    y0 = margin + pad if upper else height - margin - pad - total
    time_box = time_frame = None
    if time_shown and s["time"] != "off":
        guess = 7.5 * time_font
        tw = min(width * 0.5, time_width(time_font) if time_width else guess)
        th = time_font * 1.3
        tx = margin + pad if "left" in s["time"] else width - margin - pad - tw
        ty = margin + pad if "upper" in s["time"] else height - margin - pad - th
        if s["time"] == s["bars"] and count:  # same corner: the time first, bars after
            if upper:
                y0 += th + gap
            else:
                ty = y0 - gap - th - (pad if s["box"] else 0.0)
        time_box = Box(tx, ty, tw, th)
        if s["box"]:
            time_frame = Box(tx - pad, ty - pad, tw + 2 * pad, th + 2 * pad)
    titles, bars, labels = [], [], []
    for k in range(count):
        top = y0 + k * row
        titles.append(Box(x, top, length, title_h))
        bars.append(Box(x, top + title_h + 2 * unit, length, thick))
        labels.append(Box(x, top + title_h + 4 * unit + thick, length, label_h))
    frame = None
    if s["box"] and count:
        frame = Box(x - pad, y0 - pad, length + 2 * pad, total + 2 * pad)
    return Placed(unit, font, time_font, time_box, titles, bars, labels, frame, time_frame)


# ======================================================================== text
#: TeX names of the symbols titles may use ($\eta$ [m]), as text
SYMBOLS = {
    "alpha": "α",
    "beta": "β",
    "gamma": "γ",
    "delta": "δ",
    "epsilon": "ε",
    "zeta": "ζ",
    "eta": "η",
    "theta": "θ",
    "kappa": "κ",
    "lambda": "λ",
    "mu": "μ",
    "nu": "ν",
    "xi": "ξ",
    "pi": "π",
    "rho": "ρ",
    "sigma": "σ",
    "tau": "τ",
    "phi": "φ",
    "chi": "χ",
    "psi": "ψ",
    "omega": "ω",
    "Gamma": "Γ",
    "Delta": "Δ",
    "Theta": "Θ",
    "Lambda": "Λ",
    "Xi": "Ξ",
    "Pi": "Π",
    "Sigma": "Σ",
    "Phi": "Φ",
    "Psi": "Ψ",
    "Omega": "Ω",
    "nabla": "∇",
    "partial": "∂",
    "infty": "∞",
    "cdot": "·",
    "times": "×",
    "pm": "±",
    "degree": "°",
    "circ": "°",
}
_SUPER = str.maketrans("0123456789-+", "⁰¹²³⁴⁵⁶⁷⁸⁹⁻⁺")


def plain(text: str) -> str:
    """A title as plain text: the $TeX$ symbols as characters (η, Δ, m²), the
    rest as written."""

    def math(match: re.Match) -> str:
        inner = match.group(1)
        inner = re.sub(r"\\([A-Za-z]+)", lambda m: SYMBOLS.get(m.group(1), m.group(1)), inner)
        inner = re.sub(
            r"\^\{([^}]*)\}|\^(\S)", lambda m: (m.group(1) or m.group(2)).translate(_SUPER), inner
        )
        inner = re.sub(r"_\{([^}]*)\}", r"\1", inner)
        return inner.replace("{", "").replace("}", "").replace("_", "")

    return re.sub(r"\$([^$]*)\$", math, str(text))


# ======================================================================== line names
def line_names(text: str) -> dict[str, str]:
    """Names for a time series item's lines from its ``labels`` property: ``"P1 =
    gauge 1; P4 = gauge 4"`` (``;`` or new lines between them)."""
    names = {}
    for part in str(text or "").replace("\n", ";").split(";"):
        column, sep, name = part.partition("=")
        if sep and column.strip() and name.strip():
            names[column.strip()] = name.strip()
    return names


def line_name(item, column: str) -> str:
    """The name a line of a time series item goes by in legends."""
    if item is None:
        return column
    return line_names(item.properties.get("labels", "")).get(column, column)
