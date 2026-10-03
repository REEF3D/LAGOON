# SPDX-License-Identifier: GPL-3.0-or-later
"""Kinds of scene items and their editable properties.

Each kind lists its properties with defaults and limits. The properties panel in
the GUI is generated from these specs, and :func:`validate` keeps scripts and the
GUI to the same rules.

Some choices depend on the loaded data (which variables a dataset has). Such
properties name a ``provider``; the data manager lists the current choices, and
the core accepts any name so that a session can be built before its data exists.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

MATERIALS = ("steel", "concrete", "rock", "ice", "wood")
#: colour maps that were renamed or replaced; old sessions and scripts keep working
COLORMAP_ALIASES = {"cividis": "rainbow", "turbo": "rainbow", "jet": "rainbow"}

COLORMAPS = (
    "reef3d",
    "ocean",
    "tension",
    "lagoon_blue",
    "balance",
    "viridis",
    "rainbow",
    "sand",
    "greys",
)


@dataclass(frozen=True)
class PropertySpec:
    default: Any
    label: str
    choices: tuple[str, ...] | None = None
    minimum: float | None = None
    maximum: float | None = None
    #: name of a data-dependent choice list (``"variables"``, ``"datasets"``, ``"cases"``)
    provider: str | None = None
    tip: str = ""


def _data_source(role: str) -> dict[str, PropertySpec]:
    """Properties shared by all items that show a dataset of a case."""
    return {
        "case": PropertySpec("", "Case", provider="cases", tip="Case item (empty: first case)"),
        "dataset": PropertySpec(
            "", "Dataset", provider="datasets", tip=f"Output to show (empty: the {role} output)"
        ),
    }


def _exaggeration(tip: str) -> dict[str, PropertySpec]:
    return {
        "exaggeration": PropertySpec(
            1.0, "Vertical exaggeration", minimum=0.1, maximum=100.0, tip=tip
        )
    }


def _colouring(variable: str, colormap: str) -> dict[str, PropertySpec]:
    return {
        "variable": PropertySpec(variable, "Colour by", provider="variables"),
        "colormap": PropertySpec(colormap, "Colour map", choices=COLORMAPS),
        "auto_range": PropertySpec(True, "Automatic range"),
        "vmin": PropertySpec(0.0, "Range min"),
        "vmax": PropertySpec(1.0, "Range max"),
        "opacity": PropertySpec(1.0, "Opacity", minimum=0.0, maximum=1.0),
    }


@dataclass(frozen=True)
class KindSpec:
    label: str
    properties: dict[str, PropertySpec]


KINDS: dict[str, KindSpec] = {
    "case": KindSpec(
        "Case",
        {
            "path": PropertySpec("", "Path", tip="REEF3D case folder, output folder or file"),
            "print_interval": PropertySpec(
                0.0,
                "Output interval [s]",
                minimum=0.0,
                tip="Time between outputs; 0 = from ctrl.txt (P 30), else frame numbers",
            ),
        },
    ),
    "free_surface": KindSpec(
        "Free surface",
        {
            **_data_source("free surface"),
            **_colouring("eta", "reef3d"),
            **_exaggeration("Scales the surface elevation about the still water level"),
            "hide_dry": PropertySpec(
                True, "Hide dry land", tip="Leave out the surface where the depth is zero"
            ),
        },
    ),
    "bathymetry": KindSpec(
        "Bathymetry",
        {
            **_data_source("bed"),
            **_colouring("none", "sand"),
            **_exaggeration("Scales the bed height about the still water level"),
        },
    ),
    "volume": KindSpec(
        "Volume",
        {**_data_source("volume"), **_colouring("velocity", "ocean")},
    ),
    "series": KindSpec(
        "Time series",
        {
            "case": PropertySpec("", "Case", provider="cases", tip="Case item (empty: first case)"),
            "source": PropertySpec(
                "", "File", provider="series", tip="Gauge, 6DOF or force file of the case"
            ),
            "columns": PropertySpec(
                "", "Columns", tip="Comma-separated column names, e.g. P1, P4 or Fx, Fz"
            ),
            "gauges_3d": PropertySpec(
                True, "Gauges in 3D", tip="Show wave gauge positions in the 3D view"
            ),
        },
    ),
    "slice": KindSpec(
        "Slice",
        {
            **_data_source("volume"),
            "normal": PropertySpec("y", "Normal", choices=("x", "y", "z")),
            "position": PropertySpec(0.0, "Position [m]", tip="Where the plane cuts the axis"),
            **_colouring("velocity", "ocean"),
        },
    ),
    "gauge": KindSpec(
        "Gauge",
        {
            "x": PropertySpec(0.0, "x [m]"),
            "y": PropertySpec(0.0, "y [m]"),
            "show_in_plot": PropertySpec(True, "Show in plot"),
        },
    ),
    "body": KindSpec(
        "Bodies",
        {
            **_data_source("bodies"),
            "material": PropertySpec("steel", "Material", choices=MATERIALS),
            **_colouring("none", "viridis"),
        },
    ),
    "mooring": KindSpec(
        "Mooring lines",
        {
            **_data_source("mooring lines"),
            **_colouring("Tension", "tension"),
            "line_width": PropertySpec(3.0, "Line width [px]", minimum=0.5, maximum=20.0),
        },
    ),
    "structure": KindSpec(
        "Structures",
        {
            **_data_source("structures"),
            "material": PropertySpec("concrete", "Material", choices=MATERIALS),
            **_colouring("none", "greys"),
        },
    ),
}


def kind_spec(kind: str) -> KindSpec:
    try:
        return KINDS[kind]
    except KeyError:
        raise ValueError(f"unknown item kind {kind!r}; known: {', '.join(KINDS)}") from None


def default_properties(kind: str) -> dict[str, Any]:
    return {key: spec.default for key, spec in kind_spec(kind).properties.items()}


def validate(kind: str, key: str, value: Any) -> Any:
    """Check and coerce one property value; raise ``ValueError`` if it is not allowed."""
    specs = kind_spec(kind).properties
    if key not in specs:
        raise ValueError(f"{kind!r} has no property {key!r}; known: {', '.join(specs)}")
    spec = specs[key]
    default = spec.default
    if isinstance(default, bool):
        if not isinstance(value, bool):
            raise ValueError(f"{key!r} expects True or False, got {value!r}")
        return value
    if isinstance(default, int | float):
        if isinstance(value, bool) or not isinstance(value, int | float):
            raise ValueError(f"{key!r} expects a number, got {value!r}")
        value = type(default)(value)
        if spec.minimum is not None and value < spec.minimum:
            raise ValueError(f"{key!r} must be >= {spec.minimum}, got {value}")
        if spec.maximum is not None and value > spec.maximum:
            raise ValueError(f"{key!r} must be <= {spec.maximum}, got {value}")
        return value
    value = str(value)
    if key == "colormap":
        value = COLORMAP_ALIASES.get(value, value)
    if spec.choices is not None and value not in spec.choices:
        raise ValueError(f"{key!r} must be one of {', '.join(spec.choices)}, got {value!r}")
    return value
