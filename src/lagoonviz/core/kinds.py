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
    "white",
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
    #: also edit with a slider: "log" (for factors such as exaggeration) or "linear"
    slider: str | None = None


def _data_source(role: str) -> dict[str, PropertySpec]:
    """Properties shared by all items that show a dataset of a case."""
    return {
        "case": PropertySpec("", "Case", provider="cases", tip="Case item (empty: first case)"),
        "dataset": PropertySpec(
            "", "Dataset", provider="datasets", tip=f"Output to show (empty: the {role} output)"
        ),
    }


def _contours(variable: str = "none") -> dict[str, PropertySpec]:
    return {
        "contour_by": PropertySpec(
            variable,
            "Contour lines of",
            provider="variables",
            tip="Draw contour lines of a variable on the surface (none: no lines)",
        ),
        "contour_levels": PropertySpec(
            10, "Contour lines", minimum=2, maximum=100, tip="About how many lines (round levels)"
        ),
    }


def _vectors() -> dict[str, PropertySpec]:
    return {
        "vectors_of": PropertySpec(
            "none",
            "Vectors of",
            provider="vectors",
            tip="Draw arrows of a vector variable, e.g. the velocity (none: no arrows)",
        ),
        "vector_count": PropertySpec(
            400, "Arrows", minimum=10, maximum=20000, tip="About how many arrows, evenly spread"
        ),
        "vector_scale": PropertySpec(
            1.0,
            "Arrow length",
            minimum=0.1,
            maximum=10.0,
            slider="log",
            tip="1: the largest arrows reach the next arrow",
        ),
    }


def _exaggeration(tip: str) -> dict[str, PropertySpec]:
    return {
        "exaggeration": PropertySpec(
            1.0, "Vertical scale", minimum=0.1, maximum=100.0, tip=tip, slider="log"
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


#: groups of the "Case set-up" item (lagoonviz.core.data.setup.GROUPS)
SETUP_GROUPS = {
    "domain": "Domain",
    "sides": "Sides",
    "water": "Still water",
    "zones": "Relaxation zones",
    "waves": "Waves",
    "gauges": "Gauges",
    "probes": "Probes",
    "sections": "Sections",
    "forces": "Forces",
    "bodies": "Floating bodies",
    "structures": "Structures",
    "regions": "Regions",
    "solids": "Solids",
    "bed": "Bed",
}
#: groups the item starts with switched off (solids are drawn as a structure item)
_SETUP_OFF = ("solids",)
_SETUP_TIPS = {
    "domain": "The domain (B 10 of control.txt)",
    "sides": "The kinds of the domain's sides (C 11 - C 16): inflow, outflow, waves, beach",
    "water": "The still water level (F 60 of ctrl.txt)",
    "zones": "Relaxation zones of wave generation and numerical beach (B 96, B 107, B 108)",
    "waves": "Wave direction (B 105), focus point (B 81), wavemakers (X 170 - X 172)",
    "gauges": "Wave gauges and lines (P 50 - P 58, P 121 - P 147)",
    "probes": "Point, line and velocity probes (P 61 - P 67, P 88, P 144 - P 148)",
    "sections": "Discharge sections and flow files (P 167, P 168, P 230, P 240)",
    "forces": "Force boxes (P 81) and Morison cylinders (P 85)",
    "bodies": "Floating bodies, centre of gravity, moorings (X 23, X 110 - X 164, X 311 - X 322)",
    "structures": "Direct forcing (A 581 - A 589) and porous (B 210 - B 270) structures",
    "regions": "Refinement boxes, wind areas and other regions (A 276, A 372, F 72, S 74)",
    "solids": "Outlines of DIVEMesh's solids (S); the solids themselves are a structure item",
    "bed": "The bed's topo objects (T) and the geo.dat points",
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
            "exaggeration": PropertySpec(
                1.0,
                "Vertical scale",
                minimum=0.1,
                maximum=100.0,
                slider="log",
                tip="Scales heights about the still water level, for the whole case",
            ),
            "sync_exaggeration": PropertySpec(
                True,
                "Same for all items",
                tip="Use the case's vertical scale for every item (surface, bed, bodies, "
                "slices, gauges); off: each surface and bed has its own",
            ),
        },
    ),
    "free_surface": KindSpec(
        "Free surface",
        {
            **_data_source("free surface"),
            **_colouring("eta", "reef3d"),
            **_contours(),
            **_vectors(),
            **_exaggeration("Scales the surface elevation about the still water level"),
            "hide_dry": PropertySpec(
                True, "Hide dry land", tip="Leave out the surface where the depth is zero"
            ),
        },
    ),
    "amr": KindSpec(
        "AMR surface",
        {
            **_data_source("AMR surface"),
            **_colouring("eta", "reef3d"),
            **_contours(),
            **_exaggeration("Scales the surface elevation about the still water level"),
            "outlines": PropertySpec(
                "dark",
                "Patch outlines",
                choices=("dark", "white", "off"),
                tip="Draw the borders of the refined patches (colour by 'level' to see "
                "the refinement levels)",
            ),
            "hide_dry": PropertySpec(
                True, "Hide dry land", tip="Leave out the surface where the depth is zero"
            ),
        },
    ),
    "bathymetry": KindSpec(
        "Bed",
        {
            **_data_source("bed"),
            **_colouring("none", "sand"),
            **_contours(),
            **_exaggeration("Scales the bed height about the still water level"),
            "hide_wet": PropertySpec(
                False,
                "Hide wet land",
                tip="Leave out the bed under still water, so only dry land shows",
            ),
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
            **_contours(),
            **_vectors(),
        },
    ),
    "iso_surface": KindSpec(
        "Iso-surface",
        {
            **_data_source("volume"),
            "iso_by": PropertySpec(
                "none",
                "Iso-surface of",
                provider="variables",
                tip="The variable whose surface of equal value is drawn",
            ),
            "iso_value": PropertySpec(0.0, "Value", tip="Where the surface lies"),
            **_colouring("none", "ocean"),
        },
    ),
    "streamlines": KindSpec(
        "Streamlines",
        {
            **_data_source("volume"),
            "streamlines_of": PropertySpec(
                "velocity", "Streamlines of", provider="vectors", tip="The vector field to follow"
            ),
            "lines_in": PropertySpec(
                "volume",
                "Lines in",
                choices=("volume", "seed plane", "free surface"),
                tip="Through the volume; in the seed plane only (as on a slice); or on the "
                "free surface (its horizontal flow)",
            ),
            "seed_normal": PropertySpec(
                "y", "Seeds on plane", choices=("x", "y", "z"), tip="The plane the lines start on"
            ),
            "seed_position": PropertySpec(0.0, "Plane at [m]", tip="Where the plane cuts the axis"),
            "seeds": PropertySpec(60, "Lines", minimum=1, maximum=2000, tip="About how many lines"),
            **_colouring("velocity", "ocean"),
            "line_width": PropertySpec(2.0, "Line width [px]", minimum=0.5, maximum=20.0),
        },
    ),
    "profile": KindSpec(
        "Wave profile",
        {
            "case": PropertySpec("", "Case", provider="cases", tip="Case item (empty: first case)"),
            "source": PropertySpec(
                "", "Profiles", provider="profiles", tip="WSFLINE output of the case"
            ),
            "lines": PropertySpec("", "Lines", tip="Comma-separated line numbers (empty: all)"),
            "theory": PropertySpec(
                True, "Wave theory", tip="Also draw the wave theory (FNPF), dashed"
            ),
            "profile_3d": PropertySpec(
                True, "Profile in 3D", tip="Draw the profile as a line in the 3D view"
            ),
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
    "dem": KindSpec(
        "DEM elements",
        {
            **_data_source("DEM"),
            "material": PropertySpec("rock", "Material", choices=MATERIALS),
            **_colouring("none", "viridis"),
            "trail_outputs": PropertySpec(
                0,
                "Trail [outputs]",
                minimum=0,
                maximum=500,
                tip="Path of each element's centre (by its id) over this many outputs before",
            ),
        },
    ),
    "ice": KindSpec(
        "Ice floes",
        {
            **_data_source("ice"),
            "material": PropertySpec("ice", "Material", choices=MATERIALS),
            **_colouring("none", "viridis"),
        },
    ),
    "rods": KindSpec(
        "Rod trees",
        {
            **_data_source("rod tree"),
            "material": PropertySpec("wood", "Material", choices=MATERIALS),
            **_colouring("none", "viridis"),
        },
    ),
    "setup": KindSpec(
        "Case set-up",
        {
            "case": PropertySpec("", "Case", provider="cases", tip="Case item (empty: first case)"),
            **{
                f"show_{group}": PropertySpec(
                    group not in _SETUP_OFF, label, tip=_SETUP_TIPS.get(group, "")
                )
                for group, label in SETUP_GROUPS.items()
            },
            "labels": PropertySpec(
                True, "Labels", tip="Numbers of gauges and probes, kinds of sides"
            ),
        },
    ),
    "particles": KindSpec(
        "Particles",
        {
            **_data_source("particle"),
            **_colouring("velocity", "viridis"),
            "point_size": PropertySpec(
                4.0, "Point size [px]", minimum=1.0, maximum=40.0, tip="Size on screen"
            ),
            "trail_outputs": PropertySpec(
                0,
                "Trail [outputs]",
                minimum=0,
                maximum=500,
                tip="Path of each particle (by its id) over this many outputs before; 0: none",
            ),
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
