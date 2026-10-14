# SPDX-License-Identifier: GPL-3.0-or-later
# Architect: Hans Bihs
"""Figure templates: what a figure shows, kept apart from the run it shows.

A template names its lines by the time-series file key and column of a case
(``nhflow_wsf_hg``, ``P1``) or the free surface line set and line number
(``nhflow_wsfline``, ``1``, ``1 theory``), so the same template draws the same
figure for any run of a set-up; measured data is named by its file and column.
Panels are time series, spectra, free surface lines or 3D views, laid out in a grid
of ``columns`` (a panel may span several), lettered (a), (b), ... if wanted, with
notes (text or ``$LaTeX$``) on them. Templates are JSON files (``*.lagoonfig``);
the figure editor makes them, the figure renderer
(``lagoonviz.render.figure.template_figure``) draws them.

    template = FigureTemplate(panels=[Panel(lines=[Line("nhflow_wsf_hg", "P1")])])
    template.save("gauge.lagoonfig")
    traces = resolve(session, template.panels[0].lines[0])
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

FORMAT = "lagoon-figure"
VERSION = 1
SUFFIX = ".lagoonfig"
#: what a panel draws: lines over time, their spectra, free surface lines along x, the 3D view
PANEL_KINDS = ("time", "spectrum", "profile", "view")
#: what a line's data is: a time series column or a free surface line line (measured
#: files go with either)
LINE_DATA = ("series", "profile")
#: 3D view panels: the view as it was (camera) or from the top with axes in metres
VIEWS = ("camera", "top")
#: where a line panel's legend goes: inside (matplotlib's places), outside, or off
LEGEND_PLACES = (
    "auto",
    "best",
    "upper right",
    "upper left",
    "lower left",
    "lower right",
    "center right",
    "center left",
    "upper center",
    "lower center",
    "outside right",
    "above",
    "below",
    "off",
)
#: where a view panel's colour bars go
BAR_PLACES = (
    "below",
    "right",
    "inside lower left",
    "inside lower right",
    "inside upper left",
    "inside upper right",
)
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
    data: str = "series"  # "series" (a time series column) or "profile" (a free surface line line)
    legend: bool = True  # named in the panel's legend

    @property
    def measured(self) -> bool:
        return bool(self.file)

    @property
    def theory(self) -> bool:
        """A free surface line line's wave theory (``"1 theory"``), not the computed one."""
        return self.data == "profile" and self.column.endswith("theory")

    def title(self) -> str:
        if self.label:
            return self.label
        if self.measured:
            return f"{Path(self.file).stem} {self.column}"
        if self.data == "profile":
            number = self.column.split()[0] if self.column else "?"
            return f"line {number} theory" if self.theory else f"line {number}"
        return self.column


@dataclass
class Note:
    """Text on a panel (``$LaTeX$`` allowed), placed in the panel's fractions: (0, 0)
    bottom left, (1, 1) top right."""

    text: str
    x: float = 0.03
    y: float = 0.95
    size: float | None = None  # points; None: the figure's font size


@dataclass
class Panel:
    """One set of axes."""

    kind: str = "time"
    lines: list[Line] = field(default_factory=list)
    ylabel: str = ""  # empty: from the lines (name or unit)
    ymin: float | None = None
    ymax: float | None = None
    logy: bool = False
    start: float | None = None  # time window [s]: the x range of time panels, the part
    end: float | None = None  # of the run for spectra; profile panels: the x range [m]
    compare: bool = False  # time panels: statistics of the first line against measured lines
    title: str = ""  # empty: none (profiles and views: their time)
    span: int = 1  # columns of the figure's grid the panel takes
    notes: list[Note] = field(default_factory=list)
    time: float | None = None  # profiles and views: the moment [s]; None: the scene's
    view: str = "camera"  # views: "camera" (as it was captured) or "top" (2D, axes in m)
    camera: dict | None = None  # views: the 3D camera; None: the session's
    look: str = ""  # views: the 3D look ("studio", ...); empty: the window's
    colour_bars: bool = True  # views: a colour bar per coloured item under the image
    # line panels: where the legend goes (LEGEND_PLACES; "auto": inside where it
    # covers least, when there are several lines), its columns (0: as fits), text
    # size [pt] (None: the figure's), a frame round it
    legend: str = "auto"
    legend_columns: int = 0
    legend_size: float | None = None
    legend_frame: bool = False
    # views: where the colour bars go (BAR_PLACES) and their length (of the image)
    bar_place: str = "below"
    bar_length: float = 0.7


@dataclass
class FigureTemplate:
    """A whole figure: its panels (stacked) and its look."""

    panels: list[Panel] = field(default_factory=list)
    style: str = "REEF3D"
    width_mm: float = 90.0
    height_mm: float | None = None
    font_size: float | None = None  # None: the style's
    title: str = ""
    columns: int = 1  # panels fill a grid of this many columns, row by row
    letters: bool = False  # (a), (b), ... before the panels' titles

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
            notes = [
                Note(**{k: v for k, v in note.items() if k in Note.__dataclass_fields__})
                for note in p.get("notes", [])
                if isinstance(note, dict) and "text" in note
            ]
            values = {
                k: v
                for k, v in p.items()
                if k in Panel.__dataclass_fields__ and k not in ("lines", "notes")
            }
            if values.get("kind", "time") not in PANEL_KINDS:
                values["kind"] = "time"
            if values.get("view", "camera") not in VIEWS:
                values["view"] = "camera"
            if values.get("legend", "auto") not in LEGEND_PLACES:
                values["legend"] = "auto"
            if values.get("bar_place", "below") not in BAR_PLACES:
                values["bar_place"] = "below"
            panels.append(Panel(lines=lines, notes=notes, **values))
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
    """A line's values, ready to draw: over time (series), or along x (profiles,
    ``time`` holding x)."""

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


def profile_columns(session) -> dict[str, tuple[str, list[tuple[str, str]]]]:
    """The free surface line sets of the open cases: key -> (name, [(column, label)]):
    each line (``"1"``) and, where the files have it (FNPF), its wave theory
    (``"1 theory"``)."""
    found: dict[str, tuple[str, list[tuple[str, str]]]] = {}
    for item in session.scene:
        if item.kind != "case":
            continue
        case = session.data.case(item.id)
        if case is None:
            continue
        for key, profiles in case.profiles.items():
            if key in found or not profiles.paths:
                continue
            try:
                first = profiles.at(float(profiles.times[0]) if len(profiles.times) else 0.0)
            except (OSError, ValueError):
                continue
            columns = []
            for number in sorted(first.lines):
                line = first.lines[number]
                columns.append((str(number), f"line {number} ({line.across} = {line.y:g} m)"))
                if line.theory is not None:
                    columns.append((f"{number} theory", f"line {number}, wave theory"))
            found[key] = (profiles.name, columns)
    return found


def resolve(session, line: Line, time: float | None = None) -> LineData:
    """The values of a line: from the first open case that has its file, or from
    its measured file. Free surface line lines are taken at ``time`` (None: the
    scene's)."""
    if line.data == "profile" and not line.measured:
        return _resolve_profile(session, line, session.scene.time if time is None else time)
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


def _resolve_profile(session, line: Line, time: float) -> LineData:
    number_text, _, _ = line.column.partition(" ")
    try:
        number = int(number_text)
    except ValueError:
        raise KeyError(f"{line.source}: no line {line.column!r}") from None
    for item in session.scene:
        if item.kind != "case":
            continue
        case = session.data.case(item.id)
        if case is None or line.source not in case.profiles:
            continue
        profile = case.profiles[line.source].at(float(time))
        if number not in profile.lines:
            raise KeyError(f"{line.source}: no line {number}")
        found = profile.lines[number]
        if line.theory:
            if found.theory is None:
                raise KeyError(f"{line.source}: line {number} has no wave theory")
            return LineData(line, found.x, found.theory, "m", line.label or "theory")
        name = line.label or (f"y = {found.y:g} m" if len(profile.lines) > 1 else r"$\eta$")
        return LineData(line, found.x, found.eta, "m", name)
    raise KeyError(f"no open case has the free surface lines {line.source!r}")


def layout(template: FigureTemplate) -> list[list[tuple[int, int, int]]]:
    """The grid of a template: rows of (panel index, first column, columns spanned);
    panels fill the rows in order, a panel that does not fit starts the next row."""
    columns = max(int(template.columns or 1), 1)
    rows: list[list[tuple[int, int, int]]] = []
    row: list[tuple[int, int, int]] = []
    used = 0
    for index, panel in enumerate(template.panels):
        span = min(max(int(panel.span or 1), 1), columns)
        if used + span > columns:
            rows.append(row)
            row, used = [], 0
        row.append((index, used, span))
        used += span
    if row:
        rows.append(row)
    return rows


# ======================================================================== editing
#: a note with its place: "0.6, 0.9: text"
_PLACED = re.compile(r"^\s*(-?[\d.]+)\s*,\s*(-?[\d.]+)\s*:\s*(.*)$")


def fits(line: Line, kind: str) -> bool:
    """Whether a line can go onto a panel of a kind: time series onto time and
    spectrum panels, free surface lines onto profile panels, measured files onto any."""
    if kind == "view":
        return False
    if line.measured:
        return True
    return (kind == "profile") == (line.data == "profile")


def notes_text(notes: list[Note]) -> str:
    """Notes as the editor shows them: one per line, "x, y: text" when placed."""
    out = []
    for k, note in enumerate(notes):
        default = Note(note.text, y=0.95 - 0.08 * k)
        same = abs(note.x - default.x) < 1e-9 and abs(note.y - default.y) < 1e-9
        out.append(note.text if same else f"{note.x:g}, {note.y:g}: {note.text}")
    return "\n".join(out)


def parse_notes(text: str) -> list[Note]:
    """The notes of the editor's text: a line each, "x, y: text" to place it (fractions
    of the panel), else under the one before."""
    notes = []
    for line in text.splitlines():
        if not line.strip():
            continue
        placed = _PLACED.match(line)
        if placed:
            notes.append(Note(placed.group(3), float(placed.group(1)), float(placed.group(2))))
        else:
            notes.append(Note(line.strip(), y=0.95 - 0.08 * len(notes)))
    return notes
