# SPDX-License-Identifier: GPL-3.0-or-later
# Architect: Hans Bihs
"""The set-up sketch: a case's input drawn in plan and side view, as a paper figure.

The marks of :mod:`lagoonviz.core.data.setup` (domain, sides, relaxation zones,
waves, gauges, probes, structures, bodies, solids) are drawn with matplotlib: the
plan view (x, y) above, the side view (x, z) below, with a legend and a box of the
main facts. A 2D case (one cell in y) has the side view only. A long, flat domain
(a flume) gets a vertical scale in the side view so its depth can be seen.

    lagoon-setup ~/runs/flume                  # flume-setup.pdf next to the case
    lagoon-setup ~/runs/flume sketch.png --width 190
    lagoon-setup ~/runs/flume --describe       # the facts as text
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from lagoonviz.core.data.setup import CaseSetup, read_setup
from lagoonviz.render.figure import MM, FigureSpec, _context, _figure, save_figure

#: the side view is stretched when the domain is this much longer than deep
FLAT = 8.0
#: groups the sketch leaves out (the bed's points would be a cloud of dots)
QUIET = ("bed",)


def _setup_of(source, case: str | None) -> CaseSetup:
    """The set-up of a case folder, a CaseSetup, or the first (or named) case of a
    session."""
    if isinstance(source, CaseSetup):
        return source
    if isinstance(source, str | Path):
        return read_setup(source)
    data = source.data  # a session
    ids = [case] if case else [i.id for i in source.scene if i.kind == "case"]
    for item_id in ids:
        setup = data.case_setup(item_id)
        if setup is not None:
            return setup
    raise ValueError("no case with input files (control.txt, ctrl.txt) in the session")


def setup_figure(source, spec: FigureSpec | None = None, case: str | None = None):
    """The sketch of a case's set-up (``source``: a case folder, a CaseSetup or a
    session)."""
    setup = _setup_of(source, case)
    spec = spec or FigureSpec.from_style("REEF3D")
    marks = [m for m in setup.marks() if m.group not in QUIET]
    bounds = setup.bounds()
    if bounds is None:
        raise ValueError(f"{setup.path}: nothing to draw (no domain, no gauges)")
    low, high = bounds
    span = np.maximum(high - low, 1e-9)
    plan = not setup.two_d and span[1] > 1e-6
    # heights of the views from the width and the domain's proportions
    width = spec.width_mm
    plan_h = width * min(max(span[1] / span[0], 0.12), 1.2) if plan else 0.0
    scale = 1.0
    if span[0] / span[2] > FLAT:
        scale = span[0] / span[2] / FLAT
    side_h = width * min(max(span[2] * scale / span[0], 0.12), 0.8)
    legend_h = 6.0 + 3.6 * ((len({m.legend for m in marks}) + 1) // 2)
    figure = _figure(spec, plan_h + side_h + legend_h + 14.0)
    with _context(spec):
        rows = [plan_h, side_h, legend_h] if plan else [side_h, legend_h]
        grid = figure.add_gridspec(len(rows), 1, height_ratios=rows)
        index = 0
        if plan:
            ax = figure.add_subplot(grid[index])
            _draw(ax, marks, (0, 1))
            ax.set_aspect("equal", adjustable="datalim")
            ax.set_xlabel("x [m]")
            ax.set_ylabel("y [m]")
            ax.set_title("Plan view", loc="left")
            index += 1
        ax = figure.add_subplot(grid[index])
        _draw(ax, marks, (0, 2))
        if scale > 1.0:
            ax.set_aspect(scale, adjustable="datalim")
            ax.set_title(f"Side view (vertical scale {scale:.3g})", loc="left")
        else:
            ax.set_aspect("equal", adjustable="datalim")
            ax.set_title("Side view", loc="left")
        ax.set_xlabel("x [m]")
        ax.set_ylabel("z [m]")
        index += 1
        legend_ax = figure.add_subplot(grid[index])
        legend_ax.axis("off")
        handles = _legend(marks)
        if handles:
            legend_ax.legend(handles=handles, loc="upper left", ncols=2, frameon=False)
        if spec.title:
            figure.suptitle(spec.title)
    return figure


def _draw(ax, marks, axes: tuple[int, int]) -> None:
    from matplotlib.collections import LineCollection, PolyCollection

    a, b = axes
    # areas first (the still water, relaxation zones), then the lines over them
    order = sorted(marks, key=lambda m: m.faces is None)
    for mark in order:
        if mark.faces is not None and len(mark.faces) and b == 1:
            ax.add_collection(
                PolyCollection(
                    [_outline(mark.faces[:, :, [a, b]])], facecolor=mark.color, alpha=0.15, lw=0
                )
            )
            if mark.group == "water":
                continue  # in plan its outline is the domain's
        if len(mark.segments):
            lines = mark.segments[:, :, [a, b]]
            width = 1.1 if mark.group in ("domain", "sides") else 0.9
            ax.add_collection(LineCollection(lines, colors=mark.color, linewidths=width))
        if len(mark.points):
            ax.plot(mark.points[:, a], mark.points[:, b], "o", ms=2.5, color=mark.color, mew=0)
        for text, where in mark.labels:
            if mark.group in ("gauges", "probes", "forces", "sections"):
                ax.annotate(
                    text,
                    (where[a], where[b]),
                    xytext=(2, 2),
                    textcoords="offset points",
                    fontsize="x-small",
                    color=mark.color,
                )
    ax.autoscale_view()
    ax.margins(0.03)


def _outline(triangles: np.ndarray) -> np.ndarray:
    """The outline of a flat convex area given as triangles (2D points), in order."""
    points = np.unique(np.round(triangles.reshape(-1, 2), 12), axis=0)
    centre = points.mean(axis=0)
    angle = np.arctan2(points[:, 1] - centre[1], points[:, 0] - centre[0])
    return points[np.argsort(angle)]


def _legend(marks) -> list:
    from matplotlib.lines import Line2D

    seen: dict[str, object] = {}
    for mark in marks:
        if mark.legend in seen:
            continue
        if len(mark.segments) or mark.faces is not None:
            seen[mark.legend] = Line2D([], [], color=mark.color, lw=1.2, label=mark.legend)
        else:
            seen[mark.legend] = Line2D(
                [], [], color=mark.color, marker="o", ms=3, lw=0, label=mark.legend
            )
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
    setup = _setup_of(case, None)
    if args.describe:
        print(setup.describe())
        return 0
    spec = FigureSpec.from_style(args.style, width_mm=args.width)
    figure = setup_figure(setup, spec)
    output = Path(args.output) if args.output else case.parent / f"{case.resolve().name}-setup.pdf"
    path = save_figure(figure, output, spec)
    print(path)
    return 0


__all__ = ["MM", "main", "setup_figure"]

if __name__ == "__main__":
    raise SystemExit(main())
