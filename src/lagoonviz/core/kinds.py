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


#: maps of the "Background map" item (lagoonviz.core.data.background.SOURCE_CHOICES)
BASEMAP_SOURCES = (
    "Kartverket topo",
    "Kartverket grey",
    "Kartverket nautical chart",
    "Sentinel-2 (satellite)",
    "OpenStreetMap",
    "GeoTIFF file",
    "Tile server URL",
)
#: background maps that were renamed; old sessions, scripts and case folders keep working
BASEMAP_ALIASES = {"Sentinel-2 2017 (satellite)": "Sentinel-2 (satellite)"}
BASEMAP_PLACEMENTS = ("below the model", "at the still water level")


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


def _grid_lines(what: str) -> dict[str, PropertySpec]:
    return {
        "grid_lines": PropertySpec(
            "off",
            "Grid lines",
            choices=("off", "dark", "white"),
            tip=f"The edges of the cells {what}: the grid the solver computed on",
        )
    }


def _colouring(variable: str, colormap: str) -> dict[str, PropertySpec]:
    return {
        "variable": PropertySpec(variable, "Colour by", provider="variables"),
        "colormap": PropertySpec(colormap, "Colour map", choices=COLORMAPS),
        "auto_range": PropertySpec(True, "Automatic range"),
        "range_from": PropertySpec(
            "1st - 99th percentile",
            "Range from",
            choices=("1st - 99th percentile", "lowest to highest"),
            tip="The automatic range: the 1st to the 99th percentile, symmetric around zero "
            "when the values change sign (strong colours, still water white), or the "
            "lowest to the highest value, as ParaView (every extreme in range)",
        ),
        "vmin": PropertySpec(0.0, "Range min"),
        "vmax": PropertySpec(1.0, "Range max"),
        "range_percent": PropertySpec(
            100.0,
            "Range [%]",
            minimum=1.0,
            maximum=500.0,
            tip="The automatic range's min and max times this: 100 % the lowest to the "
            "highest value, 95 % 0.95 × min to 0.95 × max (values beyond take the end "
            "colours). Fit range to this frame uses it too",
        ),
        "opacity": PropertySpec(1.0, "Opacity", minimum=0.0, maximum=1.0),
        **_legend(),
    }


def _legend() -> dict[str, PropertySpec]:
    """The item's colour bar (lagoonviz.core.legend): the 3D view's, films', figures'."""
    from lagoonviz.core.legend import NUMBER_FORMATS

    return {
        "legend": PropertySpec(
            True,
            "Colour bar",
            tip="Show this item's colour bar (3D view, films, saved images, figures)",
        ),
        "legend_title": PropertySpec(
            "",
            "Bar title",
            tip=r"The colour bar's title (empty: name · variable); $\eta$ [m] for symbols",
        ),
        "legend_format": PropertySpec(
            "auto",
            "Bar numbers",
            choices=NUMBER_FORMATS,
            tip="How the colour bar's numbers are written",
        ),
        "legend_ticks": PropertySpec(
            0,
            "Bar ticks",
            minimum=0,
            maximum=11,
            tip="Numbers along the colour bar: 0 automatic (the 3D view and films: its "
            "ends; figures: round numbers), 1 the ends and the middle, 2 to 11 that "
            "many, evenly spaced from end to end",
        ),
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
    "curvgrid": "Curvilinear grid",
    "river": "River drawn",
}
#: faces of the domain the "DIVEMesh grid" item draws its lines on
GRID_FACES = ("bottom and back", "bottom", "back", "outside")
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
    "structures": "Direct forcing (A 581 - A 589) and porous (B 210 - B 291) structures",
    "regions": "Refinement boxes, wind areas and other regions (A 276, A 372, F 72, S 74)",
    "solids": "Outlines of DIVEMesh's solids (S); the solids themselves are a structure item",
    "bed": "The bed's topo objects (T); geo.dat's points are an item of their own",
    "curvgrid": "The curvilinear grid of curvgrid.dat (G 50 2) or blocks.dat (G 50 1), on the bed",
    "river": "The river drawn on the map (river.json), before its corridor is made",
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
            **_grid_lines("of the surface"),
            **_contours(),
            **_vectors(),
            **_exaggeration("Scales the surface elevation about the still water level"),
            "hide_dry": PropertySpec(
                True,
                "Hide dry land",
                tip="Show only the wet free surface: leave it out over dry land (FNPF: no "
                "water depth; NHFLOW, SFLOW: dry cells, wetdry)",
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
            "patches": PropertySpec(
                True,
                "AMR patches",
                tip="Show the refined patches; off: the base grid (level 0) alone",
            ),
            "outlines": PropertySpec(
                "dark",
                "Patch outlines",
                choices=("dark", "white", "off"),
                tip="Draw the borders of the refined patches (colour by 'level' to see "
                "the refinement levels)",
            ),
            **_grid_lines("of every level"),
            "hide_dry": PropertySpec(
                True,
                "Hide dry land",
                tip="Show only the wet free surface: leave it out over dry land (FNPF: no "
                "water depth; NHFLOW, SFLOW: dry cells, wetdry)",
            ),
        },
    ),
    "bathymetry": KindSpec(
        "Bed",
        {
            **_data_source("bed"),
            **_colouring("none", "sand"),
            **_grid_lines("of the bed"),
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
        {
            **_data_source("volume"),
            **_colouring("velocity", "ocean"),
            **_grid_lines("on the volume's outside"),
        },
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
            "labels": PropertySpec(
                "",
                "Line names",
                tip="Names for the legends of the plots and their figures, e.g. "
                "P1 = gauge 1; P4 = gauge 4 (empty: the column names)",
            ),
            "plot_legend": PropertySpec(
                "above the plot",
                "Legend",
                choices=("above the plot", "in the plot", "off"),
                tip="Where the Plots panel shows the lines' names (figures: inside, "
                "where they cover least; off: none)",
            ),
            "legend_values": PropertySpec(
                True, "Values at the cursor", tip="The legend gives each line's value at the time"
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
            **_grid_lines("the slice cuts"),
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
                tip="The variable whose surface of equal value is drawn (of a free surface "
                "or bed: its line)",
            ),
            "iso_value": PropertySpec(0.0, "Value", tip="Where the surface lies"),
            **_colouring("none", "ocean"),
        },
    ),
    "iso_volume": KindSpec(
        "Iso-volume",
        {
            **_data_source("volume"),
            "iso_by": PropertySpec(
                "none",
                "Iso-volume of",
                provider="variables",
                tip="The variable whose range is drawn (phi from 0 up: the water of a CFD run; "
                "of a free surface or bed: the part of it in the range)",
            ),
            "iso_min": PropertySpec(0.0, "From", tip="The part of the volume from this value"),
            "iso_max": PropertySpec(1.0, "To", tip="… up to this value"),
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
        "Free surface line",
        {
            "case": PropertySpec("", "Case", provider="cases", tip="Case item (empty: first case)"),
            "source": PropertySpec(
                "",
                "Lines from",
                provider="profiles",
                tip="Free surface lines of the case (WSFLINE, WSFLINE_Y, time-averaged)",
            ),
            "lines": PropertySpec("", "Lines", tip="Comma-separated line numbers (empty: all)"),
            "theory": PropertySpec(
                True, "Wave theory", tip="Also draw the wave theory (FNPF), dashed"
            ),
            "profile_3d": PropertySpec(
                True, "Lines in 3D", tip="Draw the free surface lines in the 3D view"
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
    "corridor": KindSpec(
        "River corridor",
        {
            "case": PropertySpec("", "Case", provider="cases", tip="Case item (empty: first case)"),
            "show_points": PropertySpec(
                True, "Points", tip="The points that shape the banks (dragged in the 2D view)"
            ),
            "show_grid": PropertySpec(
                True, "Grid", tip="The curvilinear grid the corridor makes (Make grid writes it)"
            ),
        },
    ),
    "multiblock": KindSpec(
        "Multi-block grid",
        {
            "case": PropertySpec("", "Case", provider="cases", tip="Case item (empty: first case)"),
            "show_points": PropertySpec(
                True,
                "Points",
                tip="The corners and the points that shape the edges (dragged in the 2D view)",
            ),
            "show_grid": PropertySpec(
                True, "Grid", tip="The cells of the blocks (Make grid writes them)"
            ),
            "show_quality": PropertySpec(
                True,
                "Quality",
                tip="Fair (orange), poor (red) and folded (dark red) cells, as dots",
            ),
            "show_issues": PropertySpec(
                True,
                "Rules",
                tip="Where a rule of DIVEMesh and REEF3D is broken (red), or warned about",
            ),
        },
    ),
    "grid": KindSpec(
        "DIVEMesh grid",
        {
            "case": PropertySpec("", "Case", provider="cases", tip="Case item (empty: first case)"),
            "faces": PropertySpec(
                "bottom and back",
                "Faces",
                choices=GRID_FACES,
                tip="The faces of the domain the grid lines are drawn on",
            ),
            "every": PropertySpec(
                0,
                "Every n-th line",
                minimum=0,
                maximum=1000,
                tip="0: as many as read well (a few hundred per face); 1: every grid line",
            ),
            "partition": PropertySpec(
                True, "MPI partition", tip="The planes between the MPI ranks (M 10)"
            ),
        },
    ),
    "basemap": KindSpec(
        "Background map",
        {
            "case": PropertySpec("", "Case", provider="cases", tip="Case item (empty: first case)"),
            "source": PropertySpec(
                "Kartverket topo",
                "Map",
                choices=BASEMAP_SOURCES,
                tip="Map tiles from a server (cached on disk), or a GeoTIFF of yours; the "
                "case needs a georeference (Set-up, Map)",
            ),
            "file": PropertySpec(
                "",
                "File or URL",
                tip="GeoTIFF file (an orthophoto, a chart), or a tile server URL with "
                "{z}, {x} and {y}",
            ),
            "margin": PropertySpec(
                0.5,
                "Margin",
                minimum=0.0,
                maximum=10.0,
                tip="How far the map reaches round the domain, in domain sizes",
            ),
            "placement": PropertySpec(
                "below the model",
                "Placed",
                choices=BASEMAP_PLACEMENTS,
                tip="Under the bed, or at the still water level round the domain (in 2D "
                "it lies under everything)",
            ),
            "opacity": PropertySpec(1.0, "Opacity", minimum=0.0, maximum=1.0),
            "pixels": PropertySpec(
                2048, "Resolution [px]", minimum=256, maximum=8192, tip="Along the longer side"
            ),
        },
    ),
    "geodat": KindSpec(
        "geo.dat points",
        {
            **_data_source("geo.dat"),
            **_colouring("elevation", "viridis"),
            "point_size": PropertySpec(
                5.0,
                "Point size [px]",
                minimum=1.0,
                maximum=40.0,
                tip="Size on screen: zoom in to see the spacing of the points",
            ),
        },
    ),
    "seastate": KindSpec(
        "Sea state",
        {
            **_data_source("sea state"),
            **_colouring("Hs", "rainbow"),
            **_grid_lines("of the map"),
            **_contours(),
            **_vectors(),
            "hide_dry": PropertySpec(
                True, "Hide dry land", tip="Leave out the map where SEASTATE has no water (wet)"
            ),
        },
    ),
    "statistics": KindSpec(
        "Statistics",
        {
            "case": PropertySpec("", "Case", provider="cases", tip="Case item (empty: first case)"),
            "source": PropertySpec(
                "",
                "Of",
                provider="datasets",
                tip="The surface output the statistics are of (free surface, bed; empty: the "
                "free surface)",
            ),
            "start": PropertySpec(0.0, "From [s]", minimum=0.0, tip="Start of the window"),
            "end": PropertySpec(
                0.0, "To [s]", minimum=0.0, tip="End of the window; 0: the last output"
            ),
            "of_variables": PropertySpec(
                "",
                "Variables",
                tip="The variables, separated by commas (empty: eta and velocity; a bed's "
                "elevation): their max, min, mean, time of the maximum and change",
            ),
            "wet_depth": PropertySpec(
                0.01, "Wet from [m]", minimum=0.0, tip="A water depth above this is wet"
            ),
            **_colouring("Hs", "rainbow"),
            **_contours(),
            **_vectors(),
            "hide_dry": PropertySpec(
                True, "Hide never wet", tip="Leave out where the water never came (ever_wet 0)"
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
    elif key == "source" and kind == "basemap":
        value = BASEMAP_ALIASES.get(value, value)
    if spec.choices is not None and value not in spec.choices:
        raise ValueError(f"{key!r} must be one of {', '.join(spec.choices)}, got {value!r}")
    return value
