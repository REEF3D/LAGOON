# SPDX-License-Identifier: GPL-3.0-or-later
# Architect: Hans Bihs
"""Figure templates: what a figure shows, kept apart from the run it shows.

A template names its lines by the time-series file key and column of a case
(``nhflow_wsf_hg``, ``P1``), so the same template draws the same figure for any
run of a set-up; measured data is named by its file and column. Templates are
JSON files (``*.lagoonfig``); the figure editor makes them, the figure renderer
(``lagoonviz.render.figure.template_figure``) draws them.

    template = FigureTemplate(panels=[Panel(lines=[Line("nhflow_wsf_hg", "P1")])])
    template.save("gauge.lagoonfig")
    traces = resolve(session, template.panels[0].lines[0])
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

FORMAT = "lagoon-figure"
VERSION = 1
SUFFIX = ".lagoonfig"
#: what a panel draws: lines over time, their spectra, or wave profiles along x
PANEL_KINDS = ("time", "spectrum")
LINE_STYLES = ("-", "--", ":", "-.", "none")
MARKERS = ("", "o", "s", "^", "x", ".")


@dataclass
class Line:
    """One line: a column of a case's time-series file, or of a measured file
    (``file`` set: a path, absolute or relative to the template)."""

    source: str  # time-series file key of the case (empty for a measured file)
    column: str
    file: str = ""  # measured data (text or CSV: time, then values)
    label: str = ""  # empty: the column's name
    color: str = ""  # empty: the style's colour for its place
    linestyle: str = "-"
    marker: str = ""
    width: float = 0.0  # 0: the style's line width

    @property
    def measured(self) -> bool:
        return bool(self.file)

    def title(self) -> str:
        if self.label:
            return self.label
        if self.measured:
            return f"{Path(self.file).stem} {self.column}"
        return self.column


@dataclass
class Panel:
    """One set of axes."""

    kind: str = "time"
    lines: list[Line] = field(default_factory=list)
    ylabel: str = ""  # empty: from the lines (name or unit)
    ymin: float | None = None
    ymax: float | None = None
    logy: bool = False
    start: float | None = None  # time window [s]: the x range of time panels,
    end: float | None = None  # the part of the run used for spectra
    compare: bool = False  # time panels: statistics of the first line against measured lines


@dataclass
class FigureTemplate:
    """A whole figure: its panels (stacked) and its look."""

    panels: list[Panel] = field(default_factory=list)
    style: str = "REEF3D"
    width_mm: float = 90.0
    height_mm: float | None = None
    font_size: float | None = None  # None: the style's
    title: str = ""

    # ------------------------------------------------------------- files
    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data = {"format": FORMAT, "version": VERSION, **data}
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> FigureTemplate:
        if data.get("format") != FORMAT:
            raise ValueError("not a LAGOON figure template")
        panels = []
        for p in data.get("panels", []):
            lines = [
                Line(**{k: v for k, v in line.items() if k in Line.__dataclass_fields__})
                for line in p.get("lines", [])
            ]
            values = {
                k: v for k, v in p.items() if k in Panel.__dataclass_fields__ and k != "lines"
            }
            if values.get("kind", "time") not in PANEL_KINDS:
                values["kind"] = "time"
            panels.append(Panel(lines=lines, **values))
        values = {k: v for k, v in data.items() if k in cls.__dataclass_fields__ and k != "panels"}
        return cls(panels=panels, **values)

    def save(self, path: str | Path) -> Path:
        path = Path(path).expanduser()
        if path.suffix != SUFFIX:
            path = path.with_suffix(SUFFIX)
        data = self.to_dict()
        # measured files next to the template are kept relative: the pair can move
        for panel in data["panels"]:
            for line in panel["lines"]:
                if line["file"]:
                    try:
                        line["file"] = str(
                            Path(line["file"]).resolve().relative_to(path.resolve().parent)
                        )
                    except ValueError:
                        pass
        path.write_text(json.dumps(data, indent=2), encoding="utf-8")
        return path

    @classmethod
    def load(cls, path: str | Path) -> FigureTemplate:
        path = Path(path).expanduser()
        template = cls.from_dict(json.loads(path.read_text(encoding="utf-8")))
        for panel in template.panels:
            for line in panel.lines:
                if line.file and not Path(line.file).is_absolute():
                    line.file = str((path.parent / line.file).resolve())
        return template


# ======================================================================== data
@dataclass
class LineData:
    """A line's values, ready to draw."""

    line: Line
    time: np.ndarray
    values: np.ndarray
    unit: str
    name: str


def series_columns(session) -> dict[str, tuple[str, list[str]]]:
    """The time-series files of the open cases: key -> (name, columns), for the
    editor's list of what can be drawn."""
    found: dict[str, tuple[str, list[str]]] = {}
    for item in session.scene:
        if item.kind != "case":
            continue
        case = session.data.case(item.id)
        if case is None:
            continue
        for key, series in case.series.items():
            if key in found:
                continue
            try:
                table = series.read()
            except (OSError, ValueError):
                continue
            found[key] = (series.name, list(table.columns))
    return found


def resolve(session, line: Line) -> LineData:
    """The values of a line: from the first open case that has its file, or from
    its measured file."""
    if line.measured:
        from lagoonviz.core.analysis import read_measurements

        time, columns = read_measurements(line.file)
        if line.column not in columns:
            raise KeyError(f"{Path(line.file).name}: no column {line.column!r}")
        return LineData(line, time, columns[line.column], "", line.title())
    for item in session.scene:
        if item.kind != "case":
            continue
        case = session.data.case(item.id)
        if case is None or line.source not in case.series:
            continue
        table = case.series[line.source].read()
        if line.column not in table.columns:
            raise KeyError(f"{line.source}: no column {line.column!r}")
        return LineData(
            line,
            table.time,
            table.columns[line.column],
            table.units.get(line.column, ""),
            line.title(),
        )
    raise KeyError(f"no open case has the time series {line.source!r}")
