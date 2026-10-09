# SPDX-License-Identifier: GPL-3.0-or-later
# Architect: Hans Bihs
"""The set-up sketch: a case's input drawn in plan and side view, as a paper figure.

The marks of :mod:`lagoonviz.core.data.setup` (domain, sides, relaxation zones,
waves, gauges, probes, structures, bodies, refinement boxes) are drawn with
matplotlib over the bed (geo.dat, or the bed of the results): the plan view
(x, y) above with the bed in colour and the coastline at the still water level,
the side view (x, z) below as a section through the middle of the domain with
the bed and the water, then a table of the main facts, the levels of a refined
grid (G 1, G 10) and a legend. DIVEMesh's solids are filled in both views. A 2D
case (one cell in y) has the side view only; a long, flat domain gets a vertical
scale in the side view so its depth can be seen.

    lagoon-setup ~/runs/flume                  # flume-setup.pdf next to the case
    lagoon-setup ~/runs/flume sketch.png --width 190
    lagoon-setup ~/runs/flume --describe       # the facts as text
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from lagoonviz.core.data.setup import COLORS, SIDE_KINDS, SIDES, CaseSetup, read_setup
from lagoonviz.render.figure import MM, FigureSpec, _context, _figure, colormap, save_figure

#: the side view is stretched when the domain is this much longer than deep
FLAT = 8.0
#: groups the sketch leaves out (the bed's points: the bed is drawn as a raster)
QUIET = ("bed",)
#: the colours of the filled areas
SAND = "#d9c59a"
WATER = "#cfe8fb"
SOLID = "#3d434b"


def _setup_of(source, case: str | None) -> tuple[CaseSetup, np.ndarray | None]:
    """The set-up of a case folder, a CaseSetup, or the first (or named) case of a
    session, with the points of the results' bed when there is one."""
    if isinstance(source, CaseSetup):
        return source, None
    if isinstance(source, str | Path):
        setup = read_setup(source)
        if setup.domain is None or not (setup.path / "geo.dat").is_file():
            # no control.txt or no geo.dat: the extent and the bed of the results
            from lagoonviz.core import Session

            session = Session()
            try:
                case_id = session.open_case(setup.path)
            except (OSError, ValueError):
                return setup, None
            return session.data.case_setup(case_id) or setup, _results_bed(session, case_id)
        return setup, None
    data = source.data  # a session
    ids = [case] if case else [i.id for i in source.scene if i.kind == "case"]
    for item_id in ids:
        setup = data.case_setup(item_id)
        if setup is not None:
            return setup, _results_bed(source, item_id)
    raise ValueError("no case with input files (control.txt, ctrl.txt) in the session")


def _results_bed(session, case_id: str) -> np.ndarray | None:
    """The points of the bed a run wrote (its first output), None without one."""
    data = session.data
    for item in session.scene:
        if item.kind != "bathymetry":
            continue
        owner = data.case_item(item.id)
        if owner is None or owner.id != data.case_item(case_id).id:
            continue
        try:
            mesh = data.load(item.id, data.first_time(owner.id) or 0.0)
        except (OSError, ValueError, KeyError, IndexError):
            continue
        if mesh is not None and len(mesh.points):
            return mesh.points
    return None


def setup_figure(source, spec: FigureSpec | None = None, case: str | None = None):
    """The sketch of a case's set-up (``source``: a case folder, a CaseSetup or a
    session)."""
    setup, bed_points = _setup_of(source, case)
    spec = spec or FigureSpec.from_style("REEF3D")
    marks = [m for m in setup.marks() if m.group not in QUIET]
    bounds = setup.bounds()
    if bounds is None:
        raise ValueError(f"{setup.path}: nothing to draw (no domain, no gauges)")
    low, high = bounds
    span = np.maximum(high - low, 1e-9)
    plan = not setup.two_d and span[1] > 1e-6
    raster = None if setup.two_d else setup.bed_raster(bed_points)
    profile = _profile(setup, raster, bed_points)
    solids = setup.solid_mesh()
    refinement = setup.refinement()
    facts = setup.facts()
    # heights of the parts from the width and the domain's proportions
    width = spec.width_mm
    plan_h = width * min(max(span[1] / span[0], 0.12), 1.0) if plan else 0.0
    # the side view: the domain across, its heights (and the bed's, the solids')
    # with a margin, stretched by a round vertical scale when the domain is flat
    x_side = (low[0] - 0.02 * span[0], high[0] + 0.02 * span[0])
    z_side = _heights(setup, low, high, profile, solids)
    z_span = z_side[1] - z_side[0]
    scale = 1.0
    if span[0] / z_span > FLAT:
        scale = float(f"{span[0] / z_span / FLAT:.2g}")
    box = min(max(z_span * scale / (x_side[1] - x_side[0]), 0.08), 0.6)
    side_h = 0.85 * width * box + 14.0
    legend_h = 4.0 + 3.4 * ((len(_legend(marks, raster, solids, setup)) + 1) // 2)
    table_h = 3.6 * ((len(facts) + 1) // 2) + 2.0
    levels_h = (
        3.6 * (len(refinement.levels) + 1 + bool(refinement.flagged)) + 2.0 if refinement else 0.0
    )
    figure = _figure(spec, plan_h + side_h + table_h + levels_h + legend_h + 16.0)
    with _context(spec):
        rows = ([plan_h] if plan else []) + [side_h, table_h]
        rows += [levels_h] if refinement else []
        rows += [legend_h]
        grid = figure.add_gridspec(len(rows), 1, height_ratios=rows)
        index = 0
        if plan:
            ax = figure.add_subplot(grid[index])
            image = _bed_plan(ax, raster, setup.water_level)
            _solids(ax, solids, (0, 1))
            _draw(ax, marks, (0, 1))
            _side_labels(ax, setup)
            if refinement is not None:
                _level0_grid(ax, setup)
            ax.set_aspect("equal", adjustable="datalim")
            ax.set_xlabel("x [m]")
            ax.set_ylabel("y [m]")
            ax.set_title("Plan view", loc="left")
            if image is not None:
                bar = figure.colorbar(image, ax=ax, shrink=0.8, pad=0.02)
                bar.set_label("bed z [m]")
            index += 1
        ax = figure.add_subplot(grid[index])
        where = _bed_side(ax, profile, setup)
        _solids(ax, solids, (0, 2))
        _zones_side(ax, marks, setup, where)
        side_marks = [
            m for m in marks if m.group != "water" and not _y_side(m) and not _refinement_box(m)
        ]
        _draw(ax, side_marks, (0, 2))
        title = "Side view"
        details = []
        if where is not None and plan:
            details.append(f"at y = {where:.4g} m")
        if scale > 1.0:
            details.append(f"vertical scale ×{scale:g}")
        ax.set_xlim(*x_side)
        ax.set_ylim(*z_side)
        # the box's proportions make the vertical scale (heights x scale)
        ax.set_box_aspect(z_span * scale / (x_side[1] - x_side[0]))
        ax.set_title(title + (f" ({', '.join(details)})" if details else ""), loc="left")
        ax.set_xlabel("x [m]")
        ax.set_ylabel("z [m]")
        index += 1
        _table(figure.add_subplot(grid[index]), facts)
        index += 1
        if refinement is not None:
            _levels(figure.add_subplot(grid[index]), refinement)
            index += 1
        legend_ax = figure.add_subplot(grid[index])
        legend_ax.axis("off")
        handles = _legend(marks, raster, solids, setup)
        if handles:
            legend_ax.legend(handles=handles, loc="upper left", ncols=2, frameon=False)
        figure.suptitle(spec.title or f"Set-up of {setup.path.name}")
    return figure


# ------------------------------------------------------------------ the bed
def _heights(setup: CaseSetup, low, high, profile, solids) -> tuple[float, float]:
    """The heights the side view shows: the domain's, the bed's and the solids'."""
    z0, z1 = float(low[2]), float(high[2])
    if setup.water_level is not None:
        z1 = max(z1, setup.water_level)
    if profile is not None and np.isfinite(profile[1]).any():
        z0 = min(z0, float(np.nanmin(profile[1])))
        z1 = max(z1, float(np.nanmax(profile[1])))
    if solids is not None:
        z1 = max(z1, float(solids[0][:, 2].max()))
        z0 = min(z0, float(solids[0][:, 2].min()))
    pad = 0.06 * max(z1 - z0, 1e-6)
    return z0 - pad, z1 + pad


def _profile(
    setup: CaseSetup, raster, points
) -> tuple[np.ndarray, np.ndarray, float | None] | None:
    """The bed along x through the middle of the domain: (x, z, y of the section)."""
    if raster is not None:
        x, y, z = raster
        middle = 0.5 * (y[0] + y[-1]) if setup.domain is None else 0.5 * sum(setup.domain[2:4])
        row = int(np.argmin(np.abs(y - middle)))
        return x, z[row], float(y[row])
    if points is None:
        if not (setup.path / "geo.dat").is_file():
            return None
        points = setup.geo_points()
    points = np.asarray(points, float)
    if len(points) < 2:
        return None
    x0, x1 = points[:, 0].min(), points[:, 0].max()
    bins = int(np.clip(len(points) ** 0.5 * 2, 20, 400))
    edges = np.linspace(x0, x1, bins + 1)
    index = np.clip(np.searchsorted(edges, points[:, 0]) - 1, 0, bins - 1)
    sums = np.bincount(index, points[:, 2], bins)
    counts = np.bincount(index, None, bins)
    z = np.where(counts > 0, sums / np.maximum(counts, 1), np.nan)
    return 0.5 * (edges[1:] + edges[:-1]), z, None


def _bed_plan(ax, raster, water: float | None):
    """The bed in colour (LAGOON's sand map) with the coastline at the still water."""
    if raster is None:
        return None
    x, y, z = raster
    if not np.isfinite(z).any() or np.nanmax(z) - np.nanmin(z) < 1e-9 * max(x[-1] - x[0], 1.0):
        return None  # a flat bed: nothing to colour
    dx, dy = (x[1] - x[0]) / 2, (y[1] - y[0]) / 2
    image = ax.imshow(
        np.ma.masked_invalid(z),
        origin="lower",
        extent=(x[0] - dx, x[-1] + dx, y[0] - dy, y[-1] + dy),
        cmap=colormap("sand"),
        interpolation="bilinear",
        zorder=0,
    )
    if water is not None and np.nanmin(z) < water < np.nanmax(z):
        ax.contour(x, y, z, levels=[water], colors="#1c2330", linewidths=0.6, zorder=1)
    return image


def _bed_side(ax, profile, setup: CaseSetup) -> float | None:
    """The section: the bed filled with sand, the water above it to the still water."""
    if profile is None:
        return None
    x, z, where = profile
    ok = np.isfinite(z)
    if not ok.any():
        return None
    bottom = setup.domain[4] if setup.domain is not None else float(np.nanmin(z))
    bottom = min(bottom, float(np.nanmin(z)))
    water = setup.water_level
    if water is not None:
        ax.fill_between(
            x, np.where(ok, z, np.nan), water, where=ok & (z < water), color=WATER, lw=0, zorder=0
        )
        ax.axhline(water, color=COLORS["water"], lw=0.8, zorder=1)
    ax.fill_between(x, bottom, np.where(ok, z, np.nan), where=ok, color=SAND, lw=0, zorder=0.5)
    ax.plot(x, np.where(ok, z, np.nan), color="#8a6d3b", lw=0.6, zorder=1)
    return where


def _refinement_box(mark) -> bool:
    """A box of the mesh refinement (ctrl.txt G 10, G 11): drawn in plan only."""
    return mark.code in ("G 10", "G 11") and mark.file == "ctrl.txt"


def _y_side(mark) -> bool:
    """A side at y min or y max (C 12, C 13): seen from the side it is the whole box."""
    return mark.group == "sides" and mark.code in ("C 12", "C 13")


def _zones_side(ax, marks, setup: CaseSetup, where: float | None) -> None:
    """The relaxation zones the section crosses, hatched from the bed to the water."""
    from matplotlib.patches import Rectangle

    if setup.domain is None:
        return
    bottom = setup.domain[4]
    top = setup.water_level if setup.water_level is not None else setup.domain[5]
    middle = where if where is not None else 0.5 * (setup.domain[2] + setup.domain[3])
    for mark in marks:
        if mark.group != "zones" or mark.faces is None or not len(mark.faces):
            continue
        corners = mark.faces.reshape(-1, 3)
        if not corners[:, 1].min() - 1e-9 <= middle <= corners[:, 1].max() + 1e-9:
            continue
        x0, x1 = corners[:, 0].min(), corners[:, 0].max()
        if x1 - x0 >= 0.999 * (setup.domain[1] - setup.domain[0]):
            continue  # a zone along the whole side (at y min or max) is not across the section
        ax.add_patch(
            Rectangle(
                (x0, bottom),
                x1 - x0,
                top - bottom,
                facecolor="none",
                edgecolor=mark.color,
                hatch="////",
                lw=0,
                alpha=0.6,
                zorder=2,
            )
        )


def _solids(ax, solids, axes: tuple[int, int]) -> None:
    """DIVEMesh's solids filled: their triangles seen from above (or the side)."""
    if solids is None:
        return
    from matplotlib.collections import PolyCollection

    points, triangles = solids
    flat = points[np.asarray(triangles, np.int64)][:, :, list(axes)]
    ax.add_collection(PolyCollection(flat, facecolor=SOLID, edgecolor="none", zorder=3))


# ------------------------------------------------------------------ marks
def _draw(ax, marks, axes: tuple[int, int]) -> None:
    from matplotlib.collections import LineCollection, PolyCollection

    a, b = axes
    # areas first (the still water, relaxation zones), then the lines over them
    order = sorted(marks, key=lambda m: m.faces is None)
    for mark in order:
        if mark.faces is not None and len(mark.faces) and b == 1:
            if mark.group == "water":
                continue  # in plan its outline is the domain's, the bed shows the depth
            ax.add_collection(
                PolyCollection(
                    [_outline(mark.faces[:, :, [a, b]])],
                    facecolor="none",
                    edgecolor=mark.color,
                    hatch="////",
                    lw=0,
                    alpha=0.6,
                    zorder=2,
                )
            )
        if len(mark.segments):
            lines = mark.segments[:, :, [a, b]]
            width = 2.0 if mark.group == "sides" else 1.1 if mark.group == "domain" else 0.9
            style = "--" if mark.code == "G 11" and mark.file == "ctrl.txt" else "-"
            ax.add_collection(
                LineCollection(
                    lines, colors=mark.color, linewidths=width, linestyles=style, zorder=4
                )
            )
        if len(mark.points):
            ax.plot(
                mark.points[:, a], mark.points[:, b], "o", ms=2.5, color=mark.color, mew=0, zorder=5
            )
        for text, where in mark.labels:
            if mark.group in ("gauges", "probes", "forces", "sections", "waves"):
                ax.annotate(
                    text,
                    (where[a], where[b]),
                    xytext=(2, 2),
                    textcoords="offset points",
                    fontsize="x-small",
                    color=mark.color,
                    zorder=6,
                )
    ax.autoscale_view()
    ax.margins(0.03)


def _side_labels(ax, setup: CaseSetup) -> None:
    """The kind of each open side, written beside it: inflow with its discharge."""
    if setup.domain is None:
        return
    xs, xe, ys, ye, _, _ = setup.domain
    discharge = setup.ctrl.value("W 10")
    for side, kind in setup.sides.items():
        if kind not in (1, 2) or side > 4:
            continue
        name = SIDE_KINDS[kind][0]
        if kind == 1 and discharge:
            name += f", W 10 {discharge:g} m³/s"
        axis, at_end, _ = SIDES[side]
        if axis == 0:
            x, y = (xe if at_end else xs), 0.5 * (ys + ye)
            ha, rotation = ("right" if at_end else "left"), 90
        else:
            x, y = 0.5 * (xs + xe), (ye if at_end else ys)
            ha, rotation = "center", 0
        ax.annotate(
            name,
            (x, y),
            xytext=(
                -4 if ha == "right" else 4 if ha == "left" else 0,
                0 if axis == 0 else (-8 if at_end else 4),
            ),
            textcoords="offset points",
            ha=ha,
            va="center",
            rotation=rotation,
            fontsize="x-small",
            color="#1f7a46",
            bbox={"boxstyle": "round,pad=0.2", "fc": "white", "ec": "none", "alpha": 0.8},
            zorder=7,
        )


def _level0_grid(ax, setup: CaseSetup) -> None:
    """Faint lines of the level-0 grid of a refined case (every n-th, 80 at most)."""
    if setup.domain is None or setup.cells is None:
        return
    from matplotlib.collections import LineCollection

    xs, xe, ys, ye, _, _ = setup.domain
    nx, ny, _ = setup.cells
    step = max(1, int(np.ceil(max(nx, ny) / 80)))
    lines = [((x, ys), (x, ye)) for x in np.linspace(xs, xe, nx + 1)[::step]]
    lines += [((xs, y), (xe, y)) for y in np.linspace(ys, ye, ny + 1)[::step]]
    ax.add_collection(LineCollection(lines, colors="white", linewidths=0.25, alpha=0.5, zorder=1.5))


def _outline(triangles: np.ndarray) -> np.ndarray:
    """The outline of a flat convex area given as triangles (2D points), in order."""
    points = np.unique(np.round(triangles.reshape(-1, 2), 12), axis=0)
    centre = points.mean(axis=0)
    angle = np.arctan2(points[:, 1] - centre[1], points[:, 0] - centre[0])
    return points[np.argsort(angle)]


# ------------------------------------------------------------------ tables, legend
def _table(ax, facts: list[tuple[str, str]]) -> None:
    """The main facts in two columns of label and value."""
    ax.axis("off")
    half = (len(facts) + 1) // 2
    for column, rows in enumerate((facts[:half], facts[half:])):
        x = 0.0 if column == 0 else 0.53
        for row, (label, value) in enumerate(rows):
            y = 1.0 - (row + 0.5) / max(half, 1)
            ax.text(
                x,
                y,
                label + ":",
                weight="bold",
                fontsize="x-small",
                va="center",
                transform=ax.transAxes,
            )
            ax.text(x + 0.16, y, value, fontsize="x-small", va="center", transform=ax.transAxes)


def _levels(ax, refinement) -> None:
    """The levels of a refined grid: cell size, where, about how many columns."""
    ax.axis("off")
    rows = []
    total = 0
    for level in refinement.levels:
        total += level.columns
        rows.append(
            (
                f"Level {level.level}",
                f"{level.size:.3g} m",
                f"{level.where}: ≈ {_millions(level.columns)} columns",
            )
        )
    ux, uy = refinement.uniform_cells
    fewer = refinement.uniform / max(total, 1)
    rows.append(
        (
            "Uniform reference",
            f"{refinement.levels[-1].size:.3g} m",
            f"{ux} × {uy} = {_millions(refinement.uniform)} columns → refined ≈ {fewer:.2g}× fewer",
        )
    )
    if refinement.flagged:
        rows.append(("", "", "plus the cells the run flags: " + ", ".join(refinement.flagged)))
    n = len(rows)
    for row, (name, size, text) in enumerate(rows):
        y = 1.0 - (row + 0.5) / n
        ax.text(
            0.0,
            y,
            name,
            weight="bold",
            color="#2c4a7a",
            fontsize="small",
            va="center",
            transform=ax.transAxes,
        )
        ax.text(0.2, y, size, fontsize="small", va="center", transform=ax.transAxes)
        ax.text(0.32, y, text, fontsize="small", color="0.35", va="center", transform=ax.transAxes)


def _millions(count: int) -> str:
    if count >= 1e6:
        return f"{count / 1e6:.3g} M"
    if count >= 1e3:
        return f"{count / 1e3:.3g}k"
    return str(count)


def _legend(marks, raster=None, solids=None, setup: CaseSetup | None = None) -> list:
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    seen: dict[str, object] = {}
    if raster is not None and setup is not None and setup.water_level is not None:
        seen["coastline"] = Line2D(
            [],
            [],
            color="#1c2330",
            lw=0.8,
            label=f"coastline (still water z = {setup.water_level:g} m)",
        )
    sides: dict[str, list[str]] = {}
    for mark in marks:
        if mark.group == "sides":  # sides of one kind in one entry: C 12, C 13
            name = mark.legend.split(" (")[0]
            sides.setdefault(name, []).append(mark.code)
            label = f"{name} ({', '.join(sorted(sides[name]))})"
            if name in seen:
                seen[name].set_label(label)
            else:
                seen[name] = Line2D([], [], color=mark.color, lw=2.0, label=label)
            continue
        if mark.legend in seen:
            continue
        if mark.group == "water":
            seen[mark.legend] = Patch(facecolor=WATER, edgecolor=COLORS["water"], label=mark.legend)
        elif mark.faces is not None:
            seen[mark.legend] = Patch(
                facecolor="none", edgecolor=mark.color, hatch="////", label=mark.legend
            )
        elif len(mark.segments):
            style = "--" if mark.code == "G 11" and mark.file == "ctrl.txt" else "-"
            width = 2.0 if mark.group == "sides" else 1.2
            seen[mark.legend] = Line2D(
                [], [], color=mark.color, lw=width, ls=style, label=mark.legend
            )
        else:
            seen[mark.legend] = Line2D(
                [], [], color=mark.color, marker="o", ms=3, lw=0, label=mark.legend
            )
    if solids is not None:
        seen["solids"] = Patch(facecolor=SOLID, edgecolor="none", label="DIVEMesh solids")
    return list(seen.values())


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="lagoon-setup",
        description="Draw a REEF3D case's set-up (control.txt, ctrl.txt) in plan and side view.",
    )
    parser.add_argument("case", help="case folder with control.txt and/or ctrl.txt")
    parser.add_argument("output", nargs="?", help="figure file (default: <case>-setup.pdf)")
    parser.add_argument("--width", type=float, default=140.0, help="width in mm (default 140)")
    parser.add_argument(
        "--style", default="REEF3D", help="house style (REEF3D, Journal (serif), Presentation)"
    )
    parser.add_argument("--describe", action="store_true", help="print the facts and stop")
    args = parser.parse_args(argv)
    case = Path(args.case).expanduser()
    if not case.exists():
        print(f"{case} does not exist", file=sys.stderr)
        return 2
    setup, _ = _setup_of(case, None)
    if args.describe:
        print(setup.describe())
        return 0
    spec = FigureSpec.from_style(args.style, width_mm=args.width)
    figure = setup_figure(case, spec)
    output = Path(args.output) if args.output else case.parent / f"{case.resolve().name}-setup.pdf"
    path = save_figure(figure, output, spec)
    print(path)
    return 0


__all__ = ["MM", "main", "setup_figure"]

if __name__ == "__main__":
    raise SystemExit(main())
