# SPDX-License-Identifier: GPL-3.0-or-later
"""Kinds of scene items and their editable properties.

Each kind lists its properties with defaults and limits. The properties panel in
the GUI is generated from these specs, and :func:`validate` keeps scripts and the
GUI to the same rules. Phase 0 defines the kinds as placeholders; readers and
renderers attach to them in phases 1 and 2.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

COLORMAPS = ("lagoon_blue", "balance", "viridis", "cividis", "greys")


@dataclass(frozen=True)
class PropertySpec:
    default: Any
    label: str
    choices: tuple[str, ...] | None = None
    minimum: float | None = None
    maximum: float | None = None


@dataclass(frozen=True)
class KindSpec:
    label: str
    properties: dict[str, PropertySpec]


KINDS: dict[str, KindSpec] = {
    "case": KindSpec(
        "Case",
        {
            "path": PropertySpec("", "Path"),
            "solver": PropertySpec("NHFLOW", "Solver", choices=("CFD", "NHFLOW", "FNPF", "SFLOW")),
        },
    ),
    "free_surface": KindSpec(
        "Free surface",
        {
            "colormap": PropertySpec("lagoon_blue", "Colour map", choices=COLORMAPS),
            "variable": PropertySpec("eta", "Colour by", choices=("eta", "velocity", "none")),
            "opacity": PropertySpec(1.0, "Opacity", minimum=0.0, maximum=1.0),
            "exaggeration": PropertySpec(1.0, "Vertical exaggeration", minimum=0.1, maximum=100.0),
        },
    ),
    "bathymetry": KindSpec(
        "Bathymetry",
        {
            "colormap": PropertySpec("greys", "Colour map", choices=COLORMAPS),
            "opacity": PropertySpec(1.0, "Opacity", minimum=0.0, maximum=1.0),
        },
    ),
    "slice": KindSpec(
        "Slice",
        {
            "normal": PropertySpec("y", "Normal", choices=("x", "y", "z")),
            "position": PropertySpec(0.0, "Position [m]"),
            "variable": PropertySpec(
                "velocity", "Colour by", choices=("velocity", "pressure", "phi")
            ),
            "colormap": PropertySpec("balance", "Colour map", choices=COLORMAPS),
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
        "Body",
        {
            "material": PropertySpec(
                "steel", "Material", choices=("steel", "concrete", "rock", "ice")
            ),
            "opacity": PropertySpec(1.0, "Opacity", minimum=0.0, maximum=1.0),
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
    if spec.choices is not None and value not in spec.choices:
        raise ValueError(f"{key!r} must be one of {', '.join(spec.choices)}, got {value!r}")
    return value
