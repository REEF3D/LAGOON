# SPDX-License-Identifier: GPL-3.0-or-later
# Architect: Hans Bihs
"""Publication figures: the plots, the wave profiles and the 3D view as paper figures.

Drawn again with matplotlib (not a screenshot of the panels): a given width in mm,
fonts in points, vector output (PDF, SVG, EPS) or PNG/TIFF at a given resolution,
axis labels with units, LaTeX in labels ($\\eta$ through matplotlib's mathtext, or a
real LaTeX installation with ``usetex``). The 3D view is rendered off screen at
the figure's resolution and placed with its colour bars and, in the 2D top view,
with x and y axes in metres.

    from lagoonviz.render.figure import FigureSpec, series_figure, save_figure
    figure = series_figure(session, FigureSpec(width_mm=90))
    save_figure(figure, "gauges.pdf", FigureSpec(width_mm=90))
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from lagoonviz.core import Session
from lagoonviz.core.colormaps import SERIES_PALETTES, lut
from lagoonviz.core.data.manager import Trace

MM = 1 / 25.4  # inches per millimetre
#: journal column widths [mm]
WIDTHS = {"single column": 90.0, "1.5 columns": 140.0, "double column": 190.0}
FORMATS = (".pdf", ".svg", ".eps", ".png", ".tif", ".tiff", ".jpg")
PROBE_COLOR = "#e07b39"


@dataclass
class FigureSpec:
    """Size and look of a figure."""

    width_mm: float = 90.0
    height_mm: float | None = None  # None: from the content
    dpi: int = 300  # raster output and the 3D view's pixels
    font_size: float = 8.0  # points
    font_family: str = "sans-serif"
    usetex: bool = False  # labels through a LaTeX installation
    title: str = ""
    time_cursor: bool = True  # a line at the scene's time in time-series figures
    palette: list[str] = field(default_factory=lambda: list(SERIES_PALETTES["light"]))

    def rc(self) -> dict:
        """matplotlib settings for this figure (used in a context, nothing global)."""
        return {
            "font.size": self.font_size,
            "font.family": self.font_family,
            "axes.titlesize": self.font_size,
            "axes.labelsize": self.font_size,
            "xtick.labelsize": self.font_size * 0.9,
            "ytick.labelsize": self.font_size * 0.9,
            "legend.fontsize": self.font_size * 0.9,
            "axes.linewidth": 0.6,
            "lines.linewidth": 1.0,
            "xtick.major.width": 0.6,
            "ytick.major.width": 0.6,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "legend.frameon": False,
            "mathtext.fontset": "dejavusans" if self.font_family == "sans-serif" else "stix",
            "text.usetex": self.usetex,
            "savefig.dpi": self.dpi,
            "pdf.fonttype": 42,  # editable text in Illustrator and Inkscape
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }


def _figure(spec: FigureSpec, height_mm: float):
    from matplotlib.figure import Figure

    height = spec.height_mm or height_mm
    return Figure(figsize=(spec.width_mm * MM, height * MM), dpi=spec.dpi, layout="constrained")


def _context(spec: FigureSpec):
    import matplotlib

    return matplotlib.rc_context(spec.rc())


def colormap(name: str):
    """A LAGOON colour map for matplotlib."""
    from matplotlib.colors import ListedColormap

    return ListedColormap(lut(name, 256), name=name)


def _label(name: str, unit: str) -> str:
    return f"{name} [{unit}]" if unit else name


# ======================================================================== time series
def series_rows(
    session: Session, item_ids: list[str] | None = None
) -> list[tuple[str, list[Trace]]]:
    """(item id, traces) of the visible time-series items (as the Plots panel)."""
    data = session.data
    rows = []
    for item_id in data.series_items():
        item = session.scene.get(item_id)
        if item_ids is not None and item_id not in item_ids:
            continue
        if item_ids is None and not item.visible:
            continue
        traces = data.traces(item_id)
        if traces:
            rows.append((item_id, traces))
    return rows


def series_figure(
    session: Session,
    spec: FigureSpec | None = None,
    rows: list[tuple[str, list[Trace]]] | None = None,
    window: tuple[float, float] | None = None,
):
    """The time series as stacked panels sharing the time axis, one per item, its
    lines in the plot colours, a legend when there are several."""
    spec = spec or FigureSpec()
    rows = series_rows(session) if rows is None else rows
    if not rows:
        raise ValueError("no time series to draw: add a gauge or force file to the scene")
    with _context(spec):
        figure = _figure(spec, 28.0 * len(rows) + 14.0)
        axes = figure.subplots(len(rows), 1, sharex=True, squeeze=False)[:, 0]
        seconds = session.data.time_in_seconds()
        for ax, (item_id, traces) in zip(axes, rows, strict=True):
            units = {t.unit for t in traces}
            for index, trace in enumerate(traces):
                color = (
                    PROBE_COLOR if item_id == "probe" else spec.palette[index % len(spec.palette)]
                )
                ax.plot(trace.time, trace.values, color=color, label=trace.name)
            if len(traces) == 1:
                ax.set_ylabel(_label(traces[0].name, traces[0].unit))
            else:
                ax.set_ylabel(
                    _label(_row_name(session, item_id), units.pop() if len(units) == 1 else "")
                )
                ax.legend(loc="best", ncols=min(len(traces), 4))
            if spec.time_cursor and seconds:
                ax.axvline(session.scene.time, color="0.4", linewidth=0.6, linestyle="--")
            ax.margins(x=0)
        if window is not None:
            axes[0].set_xlim(*window)
        axes[-1].set_xlabel("t [s]" if seconds else "output")
        if spec.title:
            figure.suptitle(spec.title)
    return figure


def _row_name(session: Session, item_id: str) -> str:
    if item_id == "probe":
        return "probe"
    return session.scene.get(item_id).name if item_id in session.scene else item_id


# ======================================================================== wave profiles
def profile_figure(session: Session, spec: FigureSpec | None = None, time: float | None = None):
    """The wave profiles at a time: η along each line, the wave theory dashed."""
    spec = spec or FigureSpec()
    data = session.data
    items = [i for i in data.profile_items() if session.scene.get(i).visible]
    lines = [(item_id, line) for item_id in items for line in data.profile_lines(item_id, time)]
    if not lines:
        raise ValueError("no wave profiles to draw: add a wave profile item to the scene")
    with _context(spec):
        figure = _figure(spec, 50.0)
        ax = figure.subplots()
        for index, (_item, line) in enumerate(lines):
            color = spec.palette[index % len(spec.palette)]
            label = f"y = {line.y:g} m" if len(lines) > 1 else r"$\eta$"
            ax.plot(line.x, line.eta, color=color, label=label)
            if line.theory is not None:
                ax.plot(line.x, line.theory, color=color, linestyle="--", linewidth=0.8)
        if any(line.theory is not None for _, line in lines):
            ax.plot([], [], color="0.3", linestyle="--", linewidth=0.8, label="theory")
        ax.set_xlabel("x [m]")
        ax.set_ylabel(r"$\eta$ [m]")
        ax.margins(x=0)
        if len(lines) > 1 or any(line.theory is not None for _, line in lines):
            ax.legend(loc="best")
        when = session.scene.time if time is None else time
        ax.set_title(spec.title or f"t = {when:.2f} s", loc="left")
    return figure


# ======================================================================== 3D view
@dataclass
class ColourBar:
    label: str
    colormap: str
    vmin: float
    vmax: float


def colour_bars(session: Session) -> list[ColourBar]:
    """The colour scales of the visible coloured items, as the view shows them."""
    data = session.data
    bars = []
    for item_id in data.data_items():
        item = session.scene.get(item_id)
        variable = str(item.properties.get("variable", "none"))
        if not item.visible or variable == "none":
            continue
        mesh = data.load(item_id)
        if mesh is None:
            continue
        try:
            vmin, vmax = data.value_range(item_id, mesh)
        except KeyError:
            continue
        bars.append(
            ColourBar(f"{item.name}: {variable}", str(item.properties["colormap"]), vmin, vmax)
        )
    return bars


def view_extent(camera, size: tuple[int, int]) -> tuple[float, float, float, float] | None:
    """x and y range seen by an orthographic camera looking down (2D top view), or
    None for any other camera."""
    if type(camera).__name__ != "OrthographicCamera":
        return None
    direction = np.asarray(camera.world.forward, dtype=float)
    if abs(abs(direction[2]) - 1.0) > 1e-6:
        return None  # not looking straight down
    width, height = float(camera.width), float(camera.height)
    aspect = size[0] / max(size[1], 1)
    if width / max(height, 1e-12) < aspect:
        width = height * aspect
    else:
        height = width / aspect
    zoom = float(getattr(camera, "zoom", 1.0) or 1.0)
    width, height = width / zoom, height / zoom
    cx, cy = float(camera.world.position[0]), float(camera.world.position[1])
    return cx - width / 2, cx + width / 2, cy - height / 2, cy + height / 2


def view_figure(
    session: Session,
    spec: FigureSpec | None = None,
    style: str = "studio",
    view: str | None = None,
    image: np.ndarray | None = None,
    extent: tuple[float, float, float, float] | None = None,
):
    """The 3D view at the scene's time with the colour bars of the coloured items
    under it; in the 2D top view with x and y axes in metres. ``image`` (RGBA)
    skips the rendering (tests, or a view rendered before)."""
    spec = spec or FigureSpec()
    bars = colour_bars(session)
    width_in = spec.width_mm * MM
    image_height_mm = spec.height_mm or spec.width_mm * 9 / 16
    if image is None:
        from lagoonviz.render.movie import OffscreenView

        pixels = (round(width_in * spec.dpi), round(image_height_mm * MM * spec.dpi))
        offscreen = OffscreenView(session, pixels, style, view)
        image = offscreen.render(session.scene.time)
        if extent is None and session.scene.plan_view:
            extent = view_extent(offscreen.scene.camera, pixels)
    bar_mm = 11.0 * len(bars)
    with _context(spec):
        figure = _figure(spec, image_height_mm + bar_mm + (8.0 if extent else 0.0))
        rows = [image_height_mm] + [9.0] * len(bars)
        grid = figure.add_gridspec(len(rows), 1, height_ratios=rows)
        ax = figure.add_subplot(grid[0])
        if extent is not None:
            ax.imshow(image, extent=extent, origin="upper", interpolation="lanczos")
            ax.set_xlabel("x [m]")
            ax.set_ylabel("y [m]")
            ax.set_aspect("equal")
        else:
            ax.imshow(image, interpolation="lanczos")
            ax.set_axis_off()
        when = session.scene.time
        seconds = session.data.time_in_seconds()
        ax.set_title(
            spec.title or (f"t = {when:.2f} s" if seconds else f"output {round(when)}"), loc="left"
        )
        from matplotlib.cm import ScalarMappable
        from matplotlib.colors import Normalize

        for index, bar in enumerate(bars):
            cax = figure.add_subplot(grid[index + 1])
            mappable = ScalarMappable(Normalize(bar.vmin, bar.vmax), colormap(bar.colormap))
            colourbar = figure.colorbar(mappable, cax=cax, orientation="horizontal")
            colourbar.set_label(bar.label)
            colourbar.outline.set_linewidth(0.5)
    return figure


# ======================================================================== saving
def save_figure(figure, path: str | Path, spec: FigureSpec | None = None) -> Path:
    """Write a figure; the format follows the extension (PDF, SVG, EPS: vectors)."""
    spec = spec or FigureSpec()
    path = Path(path).expanduser()
    if path.suffix.lower() not in FORMATS:
        path = path.with_suffix(".pdf")
    path.parent.mkdir(parents=True, exist_ok=True)
    with _context(spec):
        figure.savefig(path, dpi=spec.dpi)
    return path
