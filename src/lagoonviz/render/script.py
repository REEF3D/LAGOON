# SPDX-License-Identifier: GPL-3.0-or-later
# Architect: Hans Bihs
"""The Python of an output, for the recorded script: each figure, film and
screenshot made in the GUI is noted in the session (``Session.record_output``) as
the code that makes it again from a script, without the GUI.

    code = figure_code("series", "gauges.pdf", settings, time=12.0, items=["series-1"])
    session.record_output(code)

Every block starts ``if MAKE_OUTPUTS:``, so ``Session.from_script`` rebuilds the
scene without writing files; ``python script.py`` writes them. No Qt here.
"""

from __future__ import annotations

import pprint
from pathlib import Path
from typing import Any

#: what a figure's settings may set (``FigureSpec.from_style`` arguments)
SPEC_KEYS = ("width_mm", "height_mm", "dpi", "font_size", "font_family", "usetex", "title")
WIDTH = 92  # code lines are wrapped beyond this


def _value(value: Any, indent: int) -> str:
    text = repr(value)
    if len(text) + indent <= WIDTH or not isinstance(value, (dict, list, tuple)):
        return text
    lines = pprint.pformat(value, width=WIDTH - indent, sort_dicts=False).splitlines()
    return "\n".join([lines[0], *(" " * indent + line for line in lines[1:])])


def _wrapped(prefix: str, value: Any, indent: int = 4) -> str:
    """``prefix(value)``; a long value on lines of its own under the call."""
    one = f"{prefix}({value!r})"
    if indent + len(one) <= WIDTH:
        return one
    inner = indent + 4
    lines = pprint.pformat(value, width=WIDTH - inner, sort_dicts=False).splitlines()
    body = "".join(f"{' ' * inner}{line}\n" for line in lines)
    return f"{prefix}(\n{body}{' ' * indent})"


def call(name: str, *args: Any, indent: int = 4, raw: tuple[str, ...] = (), **kwargs: Any) -> str:
    """``name(args, key=value)`` as code; one argument per line when long. ``raw``
    are code already (names of variables, calls)."""
    parts = [*raw, *(repr(a) for a in args)]
    parts += [f"{k}={v!r}" for k, v in kwargs.items()]
    one = f"{name}({', '.join(parts)})"
    if indent + len(one) <= WIDTH:
        return one
    inner = indent + 4
    parts = [*raw, *(_value(a, inner) for a in args)]
    parts += [f"{k}={_value(v, inner + len(k) + 1)}" for k, v in kwargs.items()]
    body = "".join(f"{' ' * inner}{part},\n" for part in parts)
    return f"{name}(\n{body}{' ' * indent})"


def _block(comment: str, imports: list[str], body: list[str]) -> str:
    lines = [f"if MAKE_OUTPUTS:  # {comment}"]
    lines += [f"    {line}" for line in imports]
    lines += [f"    {line}" for line in body]
    return "\n".join(lines)


def _view(time: float, camera: dict | None, plan: tuple[bool, str] | None) -> list[str]:
    """Code that puts the session at the time and view of an output."""
    lines = [f"session.set_time({float(time)!r})"]
    if plan is not None:
        on, side = plan
        lines.append(f"session.set_plan_view({bool(on)!r}, side={side!r})")
    if camera:
        lines.append(_wrapped("session.set_camera", dict(camera)))
    return lines


def _spec(settings: dict) -> str:
    # what is not set, or the default, is left out
    values = {k: settings[k] for k in SPEC_KEYS if settings.get(k) not in (None, "", False)}
    return "spec = " + call("FigureSpec.from_style", settings.get("style", "REEF3D"), **values)


def _path(path: str | Path) -> str:
    return str(Path(path).expanduser())


# ======================================================================== figures
def figure_code(
    content: str,
    path: str | Path,
    settings: dict,
    *,
    time: float = 0.0,
    items: list[str] | None = None,
    window: tuple[float, float] | None = None,
    line: tuple[str, str] | None = None,
    camera: dict | None = None,
    plan: tuple[bool, str] | None = None,
    style: str = "studio",
) -> str:
    """File → Export figure. ``content`` as the dialog ("series", "spectra",
    "validation", "profiles", "view", "setup"); ``settings`` the dialog's (style, sizes,
    fonts, time window ``start``/``end``, ``measured``, ``measured_column``);
    ``items`` the plotted time-series items, ``window`` the plots' time range,
    ``line`` (item, column) of the validated line, ``camera``, ``plan`` and
    ``style`` the 3D view's."""
    imports = ["from lagoonviz.render.figure import FigureSpec, save_figure"]
    body = _view(time, camera if content == "view" else None, plan if content == "view" else None)
    body.append(_spec(settings))
    notes: list[str] = []
    shown = [i for i in (items or []) if i != "probe"]
    if items and len(shown) < len(items):
        notes.append("# the probe's line is drawn in the GUI only")
    start, end = settings.get("start"), settings.get("end")
    if content in ("series", "spectra"):
        name = "series_figure" if content == "series" else "spectrum_figure"
        imports[0] = (
            f"from lagoonviz.render.figure import FigureSpec, {name}, save_figure, series_rows"
        )
        body.append(f"rows = series_rows(session, {_value(shown, 31)})")
        parts = ["session", "spec", "rows=rows"]
        if content == "series" and window is not None:
            parts.append(f"window={tuple(float(v) for v in window)!r}")
        if content == "spectra":
            parts += [f"start={start!r}", f"end={end!r}"]
        body.append("figure = " + _raw_call(name, parts))
    elif content == "validation":
        if line is None or not settings.get("measured"):
            raise ValueError("a validation figure needs the line and the measured file")
        item, column = line
        measured = _path(settings["measured"])
        imports = [
            "from lagoonviz.core.analysis import read_measurements",
            "from lagoonviz.render.figure import FigureSpec, save_figure, validation_figure",
        ]
        body.append(
            f"trace = next(t for t in session.data.traces({item!r}) if t.name == {column!r})"
        )
        body.append(f"measured_time, measured = read_measurements({measured!r})")
        name = f"{Path(measured).stem} {settings['measured_column']}"
        body.append(
            "figure = "
            + _raw_call(
                "validation_figure",
                [
                    "trace",
                    "measured_time",
                    f"measured[{settings['measured_column']!r}]",
                    "spec",
                    repr(name),
                    repr(start),
                    repr(end),
                ],
            )
        )
    elif content == "profiles":
        imports[0] = "from lagoonviz.render.figure import FigureSpec, profile_figure, save_figure"
        body.append("figure = profile_figure(session, spec)")
    elif content == "view":
        imports[0] = "from lagoonviz.render.figure import FigureSpec, save_figure, view_figure"
        body.append(f"figure = view_figure(session, spec, style={style!r})")
    elif content == "setup":
        imports.append("from lagoonviz.render.setup_sketch import setup_figure")
        body.append("figure = setup_figure(session, spec)")
    else:
        raise ValueError(f"unknown figure content {content!r}")
    body += notes
    body.append(f"save_figure(figure, {_path(path)!r}, spec)")
    return _block(f"File > Export figure: {content}", imports, body)


def _raw_call(name: str, parts: list[str], indent: int = 4) -> str:
    one = f"{name}({', '.join(parts)})"
    if indent + len("figure = ") + len(one) <= WIDTH:
        return one
    body = "".join(f"{' ' * (indent + 4)}{part},\n" for part in parts)
    return f"{name}(\n{body}{' ' * indent})"


def template_code(template: dict, path: str | Path, dpi: int = 300, time: float = 0.0) -> str:
    """File → Figure editor → Export: the template written out in the script."""
    imports = [
        "from lagoonviz.core.figure_template import FigureTemplate",
        "from lagoonviz.render.figure import save_figure, template_figure, template_spec",
    ]
    body = _view(time, None, None)
    body.append("template = " + _wrapped("FigureTemplate.from_dict", compact_template(template)))
    body.append(f"spec = template_spec(template, dpi={int(dpi)})")
    body.append(f"save_figure(template_figure(session, template, spec), {_path(path)!r}, spec)")
    return _block("File > Figure editor: export", imports, body)


def compact_template(template: dict) -> dict:
    """A template's dict without the values that are the defaults (``from_dict``
    fills them in again): what the figure sets, readable in the script."""
    from dataclasses import MISSING, fields

    from lagoonviz.core.figure_template import FigureTemplate, Line, Panel

    def defaults(cls) -> dict:
        found = {}
        for f in fields(cls):
            if f.default is not MISSING:
                found[f.name] = f.default
            elif f.default_factory is not MISSING:
                found[f.name] = f.default_factory()
        return found

    def strip(values: dict, cls) -> dict:
        known = defaults(cls)
        return {k: v for k, v in values.items() if k not in known or known[k] != v}

    out = strip(dict(template), FigureTemplate)
    out["panels"] = [
        {
            **strip({k: v for k, v in panel.items() if k != "lines"}, Panel),
            "lines": [strip(line, Line) for line in panel.get("lines", [])],
        }
        for panel in template.get("panels", [])
    ]
    return out


# ======================================================================== films, screenshots
def film_code(
    path: str | Path,
    *,
    start: int | None = None,
    end: int | None = None,
    every: int = 1,
    size: tuple[int, int] = (1920, 1080),
    fps: float = 25.0,
    style: str = "studio",
    camera: dict | None = None,
    plan: tuple[bool, str] | None = None,
    overlay: bool = False,
    engine: str = "pygfx",
    samples: int = 64,
    foam: bool = True,
) -> str:
    """File → Export animation (``engine`` "cycles": path traced by Blender, with
    white water unless ``foam`` is off)."""
    imports = ["from lagoonviz.render.movie import frame_times, write_movie"]
    body = [line for line in _view(0.0, camera, plan) if not line.startswith("session.set_time")]
    body.append(f"times = frame_times(session, {start!r}, {end!r}, {int(every)!r})")
    if engine == "cycles":
        imports = [
            "from lagoonviz.render.movie import frame_times",
            "from lagoonviz.render.pathtrace import path_trace",
        ]
        body.append(
            _raw_call(
                "path_trace",
                [
                    "session",
                    repr(_path(path)),
                    "times",
                    repr(tuple(int(v) for v in size)),
                    repr(style),
                    repr(int(samples)),
                    repr(float(fps)),
                    *([] if foam else ["foam=False"]),
                ],
            )
        )
        return _block("File > Export animation (path traced with Blender)", imports, body)
    body.append(
        "write_movie("
        + f"session, {_path(path)!r}, times, {tuple(int(v) for v in size)!r}, {float(fps)!r}, "
        + f"{style!r})"
    )
    if len(body[-1]) + 4 > WIDTH:
        body[-1] = _raw_call(
            "write_movie",
            [
                "session",
                repr(_path(path)),
                "times",
                repr(tuple(int(v) for v in size)),
                repr(float(fps)),
                repr(style),
            ],
        )
    if overlay:
        body.append("# the time and colour bars drawn on the frames are added by the GUI only")
    return _block("File > Export animation", imports, body)


def cycles_still_code(
    path: str | Path,
    *,
    size: tuple[int, int],
    time: float = 0.0,
    style: str = "open_ocean",
    samples: int = 64,
    camera: dict | None = None,
    plan: tuple[bool, str] | None = None,
) -> str:
    """File → Render still with Cycles: the 3D view path traced by Blender."""
    imports = ["from lagoonviz.render.pathtrace import path_trace"]
    body = _view(time, camera, plan)
    body.append(
        _raw_call(
            "path_trace",
            [
                "session",
                repr(_path(path)),
                f"size={tuple(int(v) for v in size)!r}",
                f"look={style!r}",
                f"samples={int(samples)!r}",
            ],
        )
    )
    return _block("File > Render still with Cycles (path traced with Blender)", imports, body)


def screenshot_code(
    path: str | Path,
    *,
    size: tuple[int, int],
    time: float = 0.0,
    style: str = "studio",
    camera: dict | None = None,
    plan: tuple[bool, str] | None = None,
) -> str:
    """File → Save screenshot: in a script, the 3D view (the window's panels are
    the GUI's)."""
    imports = ["from lagoonviz.render.offscreen import render_session, write_png"]
    body = _view(time, camera, plan)
    size = tuple(int(v) for v in size)
    body.append(f"write_png({_path(path)!r}, render_session(session, {size!r}, style={style!r}))")
    if len(body[-1]) + 4 > WIDTH:
        body[-1] = f"image = render_session(session, {size!r}, style={style!r})"
        body.append(f"write_png({_path(path)!r}, image)")
    return _block("File > Save screenshot (the 3D view)", imports, body)
