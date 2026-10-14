# SPDX-License-Identifier: GPL-3.0-or-later
# Architect: Hans Bihs
"""Publication figures: the plots, the free surface lines and the 3D view as paper figures.

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
from lagoonviz.core.legend import ColourBar, format_number, line_name
from lagoonviz.core.legend import colour_bars as legend_bars

MM = 1 / 25.4  # inches per millimetre
#: journal column widths [mm]
WIDTHS = {"single column": 90.0, "1.5 columns": 140.0, "double column": 190.0}
FORMATS = (".pdf", ".svg", ".eps", ".png", ".tif", ".tiff", ".jpg")
PROBE_COLOR = "#e07b39"

#: house styles: fonts, line widths and line colours of a kind of figure
REEF3D_LINES = ["#2a5aa0", "#de5a45", "#3a9a5b", "#c9a227", "#8e5bb0", "#4f8fcf", "#6b6b6b"]
STYLES: dict[str, dict] = {
    # the group's papers: REEF3D blue and red first, sans serif, thin lines
    "REEF3D": {
        "font_family": "sans-serif",
        "font_size": 8.0,
        "line_width": 1.0,
        "palette": REEF3D_LINES,
        "grid": False,
    },
    # journals that set text in a serif face: black first, then the REEF3D colours
    "Journal (serif)": {
        "font_family": "serif",
        "font_size": 8.0,
        "line_width": 0.8,
        "palette": ["#000000", "#c4302b", "#2a5aa0", "#3a9a5b", "#8e5bb0", "#808080"],
        "grid": False,
    },
    # slides: large text, thick lines, a light grid
    "Presentation": {
        "font_family": "sans-serif",
        "font_size": 14.0,
        "line_width": 2.0,
        "palette": REEF3D_LINES,
        "grid": True,
    },
}


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
    line_width: float = 1.0  # points
    grid: bool = False
    style: str = ""

    @classmethod
    def from_style(cls, name: str, **overrides) -> FigureSpec:
        """A figure in a house style (``STYLES``), with some settings changed."""
        values = dict(STYLES.get(name, STYLES["REEF3D"]))
        values["palette"] = list(values["palette"])
        values.update({k: v for k, v in overrides.items() if v is not None})
        return cls(style=name, **values)

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
            "axes.linewidth": 0.6 * max(self.line_width, 1.0),
            "lines.linewidth": self.line_width,
            "axes.grid": self.grid,
            "grid.linewidth": 0.4,
            "grid.color": "0.85",
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
            item = _row_item(session, item_id)
            for index, trace in enumerate(traces):
                color = (
                    PROBE_COLOR if item_id == "probe" else spec.palette[index % len(spec.palette)]
                )
                ax.plot(trace.time, trace.values, color=color, label=line_name(item, trace.name))
            if len(traces) == 1:
                ax.set_ylabel(_label(line_name(item, traces[0].name), traces[0].unit))
            else:
                ax.set_ylabel(
                    _label(_row_name(session, item_id), units.pop() if len(units) == 1 else "")
                )
                place_legend(ax, _row_legend(item), columns=min(len(traces), 4))
            if spec.time_cursor and seconds:
                ax.axvline(session.scene.time, color="0.4", linewidth=0.6, linestyle="--")
            ax.margins(x=0)
        if window is not None:
            axes[0].set_xlim(*window)
        axes[-1].set_xlabel("t [s]" if seconds else "output")
        if spec.title:
            figure.suptitle(spec.title)
    return figure


def _row_item(session: Session, item_id: str):
    """The time series item of a row of plots (None: the probe's)."""
    return session.scene.get(item_id) if item_id in session.scene else None


def _row_legend(item) -> str:
    """Where a quick figure puts a time series item's legend: inside, or off as the
    item's Plots legend is."""
    if item is not None and item.properties.get("plot_legend") == "off":
        return "off"
    return "best"


def _row_name(session: Session, item_id: str) -> str:
    if item_id == "probe":
        return "probe"
    return session.scene.get(item_id).name if item_id in session.scene else item_id


# ======================================================================== free surface lines
def profile_figure(session: Session, spec: FigureSpec | None = None, time: float | None = None):
    """The free surface lines at a time: η along each line, the wave theory dashed."""
    spec = spec or FigureSpec()
    data = session.data
    items = [i for i in data.profile_items() if session.scene.get(i).visible]
    lines = [(item_id, line) for item_id in items for line in data.profile_lines(item_id, time)]
    if not lines:
        raise ValueError("no free surface lines to draw: add a free surface line item to the scene")
    with _context(spec):
        figure = _figure(spec, 50.0)
        ax = figure.subplots()
        for index, (_item, line) in enumerate(lines):
            color = spec.palette[index % len(spec.palette)]
            label = f"{line.across} = {line.y:g} m" if len(lines) > 1 else r"$\eta$"
            ax.plot(line.x, line.eta, color=color, label=label)
            if line.theory is not None:
                ax.plot(line.x, line.theory, color=color, linestyle="--", linewidth=0.8)
        if any(line.theory is not None for _, line in lines):
            ax.plot([], [], color="0.3", linestyle="--", linewidth=0.8, label="theory")
        ax.set_xlabel(f"{', '.join(sorted({line.axis for _, line in lines}))} [m]")
        ax.set_ylabel(r"$\eta$ [m]")
        ax.margins(x=0)
        if len(lines) > 1 or any(line.theory is not None for _, line in lines):
            ax.legend(loc="best")
        when = session.scene.time if time is None else time
        ax.set_title(spec.title or f"t = {when:.2f} s", loc="left")
    return figure


# ======================================================================== spectra
def spectrum_figure(
    session: Session,
    spec: FigureSpec | None = None,
    rows: list[tuple[str, list[Trace]]] | None = None,
    start: float | None = None,
    end: float | None = None,
):
    """Variance density spectra of the time series (wave gauges): one panel per
    item, a line per column with its Hm0 and Tp in the legend."""
    from lagoonviz.core.analysis import spectrum, wave_parameters

    spec = spec or FigureSpec()
    rows = series_rows(session) if rows is None else rows
    if not rows:
        raise ValueError("no time series to analyse: add a wave gauge file to the scene")
    with _context(spec):
        figure = _figure(spec, 34.0 * len(rows) + 14.0)
        axes = figure.subplots(len(rows), 1, sharex=True, squeeze=False)[:, 0]
        highest = 0.0
        for ax, (item_id, traces) in zip(axes, rows, strict=True):
            units = {t.unit for t in traces}
            item = _row_item(session, item_id)
            for index, trace in enumerate(traces):
                try:
                    f, s = spectrum(trace.time, trace.values, start=start, end=end)
                except ValueError:
                    continue
                waves = wave_parameters(f, s)
                color = (
                    PROBE_COLOR if item_id == "probe" else spec.palette[index % len(spec.palette)]
                )
                unit = f" {trace.unit}" if trace.unit else ""
                label = (
                    f"{line_name(item, trace.name)}: $H_{{m0}}$ = {waves.hm0:.3g}{unit}, "
                    f"$T_p$ = {waves.tp:.3g} s"
                )
                ax.plot(f, s, color=color, label=label)
                # up to where the spectrum has fallen to half a percent of its peak
                above = np.flatnonzero(s > 0.005 * s.max()) if s.size and s.max() > 0 else []
                if len(above):
                    highest = max(highest, 1.3 * float(f[above[-1]]))
            unit = units.pop() if len(units) == 1 else ""
            ax.set_ylabel(f"S [{unit}$^2$/Hz]" if unit else "S [1/Hz]")
            top = max((line.get_ydata().max() for line in ax.get_lines()), default=1.0)
            legend = _row_legend(item)
            ax.set_ylim(0, (1.45 if legend != "off" else 1.05) * top)  # room for the legend
            place_legend(ax, "upper right" if legend != "off" else "off")
            ax.set_title(_row_name(session, item_id), loc="left")
        if highest > 0:
            axes[0].set_xlim(0, highest)
        axes[-1].set_xlabel("f [Hz]")
        if spec.title:
            figure.suptitle(spec.title)
    return figure


# ======================================================================== validation
def validation_figure(
    trace: Trace,
    measured_time: np.ndarray,
    measured: np.ndarray,
    spec: FigureSpec | None = None,
    measured_name: str = "measured",
    start: float | None = None,
    end: float | None = None,
):
    """A computed series against measurements: both over time where they overlap,
    with RMSE, bias, correlation and Willmott's skill in the corner."""
    from lagoonviz.core.analysis import compare

    spec = spec or FigureSpec()
    stats = compare(trace.time, trace.values, measured_time, measured, start, end)
    with _context(spec):
        figure = _figure(spec, 55.0)
        ax = figure.subplots()
        keep = (measured_time >= stats.start) & (measured_time <= stats.end)
        model = (trace.time >= stats.start) & (trace.time <= stats.end)
        ax.plot(
            measured_time[keep],
            measured[keep],
            color="0.15",
            linestyle="none",
            marker="o",
            markersize=max(1.5, spec.line_width * 1.6),
            markerfacecolor="none",
            markeredgewidth=0.5 * spec.line_width,
            label=measured_name,
        )
        ax.plot(
            trace.time[model],
            trace.values[model],
            color=spec.palette[0],
            label=f"{trace.name} (REEF3D)",
        )
        unit = f" {trace.unit}" if trace.unit else ""
        # the legend above the plot, clear of the data; the numbers in a box
        ax.text(
            0.01,
            0.03,
            f"RMSE = {stats.rmse:.3g}{unit}, bias = {stats.bias:.2g}{unit}, "
            f"r = {stats.r:.3f}, d = {stats.skill:.3f}",
            transform=ax.transAxes,
            fontsize=spec.font_size * 0.85,
            va="bottom",
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.85, "pad": 1.5},
        )
        ax.legend(loc="lower left", bbox_to_anchor=(0.0, 1.0), ncols=2, borderaxespad=0.2)
        ax.set_xlabel("t [s]")
        ax.set_ylabel(_label(trace.name, trace.unit))
        ax.margins(x=0)
        if spec.title:
            figure.suptitle(spec.title)
    figure.comparison = stats
    return figure


# ======================================================================== 3D view
def colour_bars(session: Session) -> list[ColourBar]:
    """The colour bars of the visible coloured items that have one, as the view shows
    them (lagoonviz.core.legend)."""
    return legend_bars(session)


def draw_bar(figure, cax, bar: ColourBar, orientation: str = "horizontal") -> None:
    """A colour bar in its axes: the item's colour map and range, its title, and its
    numbers as the item asks (ticks, format)."""
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize
    from matplotlib.ticker import FuncFormatter

    mappable = ScalarMappable(Normalize(bar.vmin, bar.vmax), colormap(bar.colormap))
    colourbar = figure.colorbar(mappable, cax=cax, orientation=orientation)
    colourbar.set_label(bar.label)
    colourbar.outline.set_linewidth(0.5)
    if bar.count > 0:
        colourbar.set_ticks([value for value, _ in bar.ticks])
        colourbar.set_ticklabels([text for _, text in bar.ticks])
    elif bar.number_format != "auto":
        how = bar.number_format
        colourbar.formatter = FuncFormatter(lambda value, _pos: format_number(value, how))
        colourbar.update_ticks()


def place_legend(
    ax,
    place: str = "auto",
    entries: int | None = None,
    columns: int = 0,
    size: float | None = None,
    frame: bool = False,
) -> None:
    """A line panel's legend where ``place`` says (LEGEND_PLACES): "auto" inside where
    it covers least, when there are several lines; outside right, above or below the
    axes; "off" none. ``columns`` 0: one, or a row of up to four above and below."""
    handles, labels = ax.get_legend_handles_labels()
    count = len(labels) if entries is None else entries
    if place == "off" or not labels or (place == "auto" and count < 2):
        return
    options: dict = {"frameon": bool(frame)}
    if size:
        options["fontsize"] = size
    across = columns or (min(len(labels), 4) if place in ("above", "below") else 1)
    options["ncols"] = max(int(across), 1)
    if place == "outside right":
        options.update(loc="center left", bbox_to_anchor=(1.01, 0.5), borderaxespad=0.0)
    elif place == "above":
        options.update(loc="lower right", bbox_to_anchor=(1.0, 1.0), borderaxespad=0.2)
    elif place == "below":
        # under the tick labels and the x label: their height in points, down from
        # the axes' lower edge
        from matplotlib import rcParams
        from matplotlib.transforms import ScaledTranslation

        def points(key: str) -> float:
            size = rcParams[key]
            return float(size) if isinstance(size, int | float) else 0.9 * rcParams["font.size"]

        drop = 1.5 * (points("xtick.labelsize") + points("axes.labelsize")) + 6.0
        figure = ax.get_figure()
        below = ax.transAxes + ScaledTranslation(0, -drop / 72, figure.dpi_scale_trans)
        options.update(
            loc="upper center", bbox_to_anchor=(0.5, 0.0), bbox_transform=below, borderaxespad=0
        )
    else:
        options["loc"] = "best" if place == "auto" else place
    ax.legend(handles, labels, **options)


def view_extent(camera, size: tuple[int, int]) -> tuple[float, float, float, float] | None:
    """x and y range seen by an orthographic camera looking down (2D top view), or
    None for any other camera."""
    if float(getattr(camera, "fov", 0.0) or 0.0) > 0:
        return None  # perspective (LAGOON's orthographic camera is fov 0)
    direction = np.asarray(camera.world.forward, dtype=float)
    if abs(abs(direction[2]) - 1.0) > 1e-6:
        return None  # not looking straight down
    right = getattr(camera.world, "right", (1.0, 0.0, 0.0))
    if abs(float(right[0]) - 1.0) > 1e-6:
        return None  # turned (north up over a turned model): no x and y axes
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
        offscreen.credit = False  # in the axes, below
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
        map_credits(ax, session)
        when = session.scene.time
        seconds = session.data.time_in_seconds()
        ax.set_title(
            spec.title or (f"t = {when:.2f} s" if seconds else f"output {round(when)}"), loc="left"
        )
        for index, bar in enumerate(bars):
            draw_bar(figure, figure.add_subplot(grid[index + 1]), bar)
    return figure


# ======================================================================== templates
def template_spec(template, dpi: int = 300, **overrides) -> FigureSpec:
    """The figure settings of a template (its style, size and title)."""
    return FigureSpec.from_style(
        template.style,
        width_mm=template.width_mm,
        height_mm=template.height_mm,
        font_size=template.font_size,
        title=template.title,
        dpi=dpi,
        **overrides,
    )


#: natural heights of line panels [mm]; views take 9/16 of their width
PANEL_HEIGHTS = {"time": 30.0, "spectrum": 34.0, "profile": 32.0}
#: a colour bar under a view panel [mm]: the bar, and with its ticks and label
BAR_MM = 3.0
BAR_ROOM_MM = 13.0
#: a colour bar right of a view panel [mm]: the bar, its ticks and its title
BAR_SIDE_MM = 16.0


def template_figure(
    session: Session, template, spec: FigureSpec | None = None, images: dict | None = None
):
    """A figure from a template (``lagoonviz.core.figure_template``): its panels in a
    grid of ``columns``, row by row. Time panels over time (measured lines as points),
    spectrum panels with Hm0 and Tp, free surface line panels along x at their moment, 3D
    view panels rendered off screen at their moment and camera, with colour bars.
    ``images`` (panel index -> RGBA) skips rendering a view (tests, previews). What
    could not be drawn is named in the figure's ``missing`` list."""
    from lagoonviz.core.figure_template import layout

    spec = spec or template_spec(template)
    if not template.panels:
        raise ValueError("the figure has no panels: drag a time series onto it")
    missing: list[str] = []
    rows = layout(template)
    columns = max(int(template.columns or 1), 1)
    cell_mm = spec.width_mm / columns
    bars = colour_bars(session)

    def natural(panel, span: int) -> float:
        if panel.kind == "view":
            below = panel.colour_bars and getattr(panel, "bar_place", "below") == "below"
            extra = BAR_ROOM_MM * len(bars) if below else 0.0
            return cell_mm * span * 9 / 16 + extra + (14.0 if panel.view == "top" else 6.0)
        return PANEL_HEIGHTS.get(panel.kind, 30.0)

    heights = [max(natural(template.panels[i], span) for i, _, span in row) for row in rows]
    with _context(spec):
        figure = _figure(spec, sum(heights) + 12.0 + (6.0 if spec.title else 0.0))
        # room for the titles over images: their axes shrink to the picture's aspect
        # after the layout, which then misses a title by a few points
        figure.get_layout_engine().set(h_pad=0.06)
        grid = figure.add_gridspec(len(rows), columns, height_ratios=heights)
        panel_axes: dict[int, object] = {}
        for r, row in enumerate(rows):
            for index, column, span in row:
                panel = template.panels[index]
                cell = grid[r, column : column + span]
                try:
                    if panel.kind == "view":
                        ax, default = _view_panel(
                            figure,
                            cell,
                            session,
                            panel,
                            spec,
                            cell_mm * span,
                            bars if panel.colour_bars else [],
                            (images or {}).get(index),
                        )
                    else:
                        ax = figure.add_subplot(cell)
                        default = _lines_panel(ax, session, panel, spec, missing)
                except Exception as error:  # noqa: BLE001 (named in the figure's list)
                    missing.append(f"panel {index + 1}: {error}")
                    ax = figure.add_subplot(cell)
                    ax.set_axis_off()
                    default = ""
                title = panel.title or default
                if template.letters:
                    title = f"({_letter(index)}) {title}".strip()
                if title:
                    ax.set_title(title, loc="left")
                for note in panel.notes:
                    ax.text(
                        note.x,
                        note.y,
                        note.text,
                        transform=ax.transAxes,
                        fontsize=note.size or spec.font_size,
                        va="top",
                        ha="left",
                    )
                panel_axes[index] = ax
        # one column: time panels share their time axis, the x label only under the lowest
        if columns == 1:
            times = [panel_axes[i] for i, p in enumerate(template.panels) if p.kind == "time"]
            for other in times[1:]:
                other.sharex(times[0])
            for ax in times[:-1]:
                ax.set_xlabel("")
        if spec.title:
            figure.suptitle(spec.title)
    figure.missing = missing
    return figure


def _letter(index: int) -> str:
    """a, b, ..., z, aa, ab, ..."""
    letters = ""
    index += 1
    while index:
        index, rest = divmod(index - 1, 26)
        letters = chr(97 + rest) + letters
    return letters


def _lines_panel(ax, session: Session, panel, spec: FigureSpec, missing: list[str]) -> str:
    """The lines of a time, spectrum or free surface line panel; returns its default title."""
    from lagoonviz.core.analysis import compare, spectrum, wave_parameters
    from lagoonviz.core.figure_template import resolve

    when = session.scene.time if panel.time is None else float(panel.time)
    drawn = []
    for index, line in enumerate(panel.lines):
        try:
            data = resolve(session, line, when)
        except (KeyError, OSError, ValueError) as error:
            missing.append(f"{line.title()}: {error}")
            continue
        theory = getattr(line, "theory", False)
        color = line.color or (
            "0.15"
            if line.measured
            else "0.35"
            if theory
            else spec.palette[index % len(spec.palette)]
        )
        linestyle = line.linestyle
        if theory and linestyle == "-":
            linestyle = "--"
        style = {
            "color": color,
            "linestyle": "none" if linestyle == "none" else linestyle,
            "label": data.name if getattr(line, "legend", True) else "_nolegend_",
        }
        if line.width:
            style["linewidth"] = line.width
        elif theory:
            style["linewidth"] = 0.8 * spec.line_width
        if line.marker:
            style.update(marker=line.marker, markersize=3, markerfacecolor="none")
        elif line.measured and line.linestyle in ("none", "-"):
            style.update(
                linestyle="none",
                marker="o",
                markersize=2.5,
                markerfacecolor="none",
                markeredgewidth=0.5,
            )
        if panel.kind == "spectrum":
            try:
                f, s = spectrum(data.time, data.values, start=panel.start, end=panel.end)
            except ValueError as error:
                missing.append(f"{line.title()}: {error}")
                continue
            waves = wave_parameters(f, s)
            if getattr(line, "legend", True):
                style["label"] = (
                    f"{data.name}: $H_{{m0}}$ = {waves.hm0:.3g}, $T_p$ = {waves.tp:.3g} s"
                )
            ax.plot(f, s, **style)
        else:
            keep = np.ones(data.time.size, bool)
            if panel.kind == "time":  # profiles: the x range is set on the axis
                if panel.start is not None:
                    keep &= data.time >= panel.start
                if panel.end is not None:
                    keep &= data.time <= panel.end
            ax.plot(data.time[keep], data.values[keep], **style)
        drawn.append(data)
    title = ""
    units = {d.unit for d in drawn if d.unit}
    if panel.kind == "spectrum":
        ax.set_xlabel("f [Hz]")
        ax.set_ylabel(panel.ylabel or (f"S [{units.pop()}$^2$/Hz]" if len(units) == 1 else "S"))
        ax.set_ylim(bottom=0)
    elif panel.kind == "profile":
        ax.set_xlabel("x [m]")
        ax.set_ylabel(panel.ylabel or r"$\eta$ [m]")
        ax.margins(x=0)
        if panel.start is not None or panel.end is not None:
            ax.set_xlim(panel.start, panel.end)
        seconds = session.data.time_in_seconds()
        title = f"t = {when:.2f} s" if seconds else f"output {round(when)}"
    else:
        default = (
            _label(drawn[0].name, drawn[0].unit)
            if len(drawn) == 1
            else (f"[{units.pop()}]" if len(units) == 1 else "")
        )
        ax.set_ylabel(panel.ylabel or default)
        ax.set_xlabel("t [s]" if session.data.time_in_seconds() else "output")
        ax.margins(x=0)
        if panel.compare:
            computed = [d for d in drawn if not d.line.measured]
            measured = [d for d in drawn if d.line.measured]
            if computed and measured:
                try:
                    stats = compare(
                        computed[0].time,
                        computed[0].values,
                        measured[0].time,
                        measured[0].values,
                        panel.start,
                        panel.end,
                    )
                    ax.text(
                        0.01,
                        0.03,
                        f"RMSE = {stats.rmse:.3g}, bias = {stats.bias:.2g}, "
                        f"r = {stats.r:.3f}, d = {stats.skill:.3f}",
                        transform=ax.transAxes,
                        fontsize=spec.font_size * 0.85,
                        va="bottom",
                        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.85, "pad": 1.5},
                    )
                except ValueError as error:
                    missing.append(f"comparison: {error}")
    if panel.logy:
        ax.set_yscale("log")
    if panel.ymin is not None or panel.ymax is not None:
        ax.set_ylim(panel.ymin, panel.ymax)
    place_legend(
        ax,
        getattr(panel, "legend", "auto"),
        len(drawn),
        getattr(panel, "legend_columns", 0),
        getattr(panel, "legend_size", None),
        getattr(panel, "legend_frame", False),
    )
    return title


def _view_panel(figure, cell, session: Session, panel, spec, width_mm, bars, image):
    """A 3D view panel: the image (rendered off screen at the panel's moment and
    camera, unless given), its colour bars below it, right of it or inside it (the
    panel's ``bar_place``). Returns (axes, default title)."""
    place = getattr(panel, "bar_place", "below")
    length = min(max(float(getattr(panel, "bar_length", 0.7) or 0.7), 0.1), 1.0)
    side = place == "right" and bool(bars)
    image_w_mm = width_mm - (BAR_SIDE_MM * len(bars) if side else 0.0)
    image_mm = image_w_mm * 9 / 16
    bar_axes = []
    if bars and place == "below":
        sub = cell.subgridspec(1 + len(bars), 1, height_ratios=[image_mm] + [BAR_MM] * len(bars))
        ax = figure.add_subplot(sub[0])
        bar_axes = [figure.add_subplot(sub[k + 1]) for k in range(len(bars))]
    elif side:
        sub = cell.subgridspec(
            1, 1 + len(bars), width_ratios=[image_w_mm] + [BAR_SIDE_MM] * len(bars)
        )
        ax = figure.add_subplot(sub[0, 0])
        bar_axes = [figure.add_subplot(sub[0, k + 1]) for k in range(len(bars))]
    else:
        ax = figure.add_subplot(cell)
    when = session.scene.time if panel.time is None else float(panel.time)
    extent = None
    if image is None:
        image, extent = render_view(
            session,
            (round(image_w_mm * MM * spec.dpi), round(image_mm * MM * spec.dpi)),
            time=when,
            camera=panel.camera,
            top=panel.view == "top",
            look=panel.look or "studio",
        )
    if extent is not None:
        ax.imshow(image, extent=extent, origin="upper", interpolation="lanczos")
        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")
        ax.set_aspect("equal")
    else:
        ax.imshow(image, interpolation="lanczos")
        ax.set_axis_off()
    map_credits(ax, session)
    if place == "below":
        for room, bar in zip(bar_axes, bars, strict=True):
            room.set_axis_off()  # the bar sits centred in its row, narrower than the image
            draw_bar(figure, room.inset_axes([(1 - length) / 2, 0.0, length, 1.0]), bar)
    elif side:
        for room, bar in zip(bar_axes, bars, strict=True):
            room.set_axis_off()
            cax = room.inset_axes([0.0, (1 - length) / 2, 0.22, length])
            draw_bar(figure, cax, bar, "vertical")
    elif bars:
        _bars_inside(ax, figure, bars, place, length, spec, image_mm)
    seconds = session.data.time_in_seconds()
    return ax, (f"t = {when:.2f} s" if seconds else f"output {round(when)}")


def _bars_inside(ax, figure, bars, place: str, length: float, spec, image_mm: float) -> None:
    """Colour bars stacked in a corner of a view, on a light box."""
    from matplotlib.patches import Rectangle

    font_mm = spec.font_size * 0.3528
    bar_h = BAR_MM / image_mm  # of the axes' height
    row = (BAR_MM + 2.6 * font_mm + 1.0) / image_mm
    width = min(length, 0.9) * 0.6  # inside: shorter than below the image
    pad = 0.02
    left = "left" in place
    upper = "upper" in place
    x = pad + 0.02 if left else 1 - pad - 0.02 - width
    total = row * len(bars)
    y_top = 1 - pad if upper else pad + total
    ax.add_patch(
        Rectangle(
            (x - 0.045, y_top - total - 0.01),
            width + 0.09,
            total + 0.01,
            transform=ax.transAxes,
            facecolor="white",
            alpha=0.75,
            edgecolor="none",
            zorder=3,
        )
    )
    for k, bar in enumerate(bars):
        # title above the bar, numbers under it: the bar in the row's middle
        y = y_top - (k + 1) * row + (1.3 * font_mm + 0.5) / image_mm
        cax = ax.inset_axes([x, y, width, bar_h], zorder=4)
        draw_bar(figure, cax, bar)
        cax.xaxis.set_label_position("top")


def map_credits(ax, session: Session) -> None:
    """The credits the background maps' sources ask for, small in the lower right
    corner of a view."""
    credits = session.data.backgrounds.attributions()
    if not credits:
        return
    ax.text(
        0.995,
        0.005,
        " · ".join(credits),
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=4.5,
        color="#1c2330",
        bbox={"facecolor": "white", "alpha": 0.6, "edgecolor": "none", "pad": 0.6},
    )


def render_view(
    session: Session,
    pixels: tuple[int, int],
    time: float | None = None,
    camera: dict | None = None,
    top: bool = False,
    look: str = "studio",
):
    """The 3D view rendered off screen at a moment and camera (or from the top, with
    the extent in metres for axes); the session's time, camera and view are put back.
    Returns (RGBA image, extent or None)."""
    from lagoonviz.render.movie import OffscreenView

    scene = session.scene
    kept = (scene.time, scene.camera, scene.plan_view, scene.plan_side)
    try:
        if time is not None:
            session.set_time(float(time))
        if camera:
            session.set_camera(camera)
        if top:
            session.set_plan_view(True, "top")
        offscreen = OffscreenView(session, pixels, look, None)
        offscreen.credit = False  # the figure puts them in its axes
        image = offscreen.render(session.scene.time)
        extent = view_extent(offscreen.scene.camera, pixels) if session.scene.plan_view else None
        return image, extent
    finally:
        session.set_time(kept[0])
        session.set_camera(kept[1])
        session.set_plan_view(kept[2], kept[3])


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
