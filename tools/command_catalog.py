# SPDX-License-Identifier: GPL-3.0-or-later
# Architect: Hans Bihs
"""The commands of ctrl.txt (REEF3D) and control.txt (DIVEMesh), read from the user
guides, for LAGOON's set-up editor:

    python tools/command_catalog.py ~/Codelite/UserGuides \
        --reef3d ~/Codelite/REEF3D --divemesh ~/Codelite/DIVEMesh

writes ``src/lagoonviz/core/data/catalog/reef3d.json`` and ``divemesh.json``. Each
command of the guides' reference chapters, e.g. ``\\item[P 51 *] {\\color{blue}
{double}} $x$, {\\color{blue} {double}} $y$ Wave gauge \\\\ ...`` with its
``description`` list of options and its default, becomes::

    "P 51": {"title": "Wave gauge", "repeat": true,
             "params": [{"name": "x", "type": "double"}, {"name": "y", "type": "double"}],
             "options": {}, "default": [], "text": "...", "modules": "SFNC"}

How many values a command takes, and whether it may be repeated, comes from the
programs' own readers (``src/read_control.cpp`` of REEF3D and DIVEMesh) when their
source trees are given: they are what counts. Commands the readers know and the
guides do not describe yet are added without a description.

Run it again when the guides or the readers change; ``tests/core/test_controlfile.py``
checks the files.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "src" / "lagoonviz" / "core" / "data" / "catalog"

GUIDES = {
    "reef3d": (
        "REEF3D_User_Guide/REEF3D-UserGuide.tex",
        r"\section{A :: Hydrodynamic models}",
        r"\chapter{Compiling the Code}",
        "REEF3D",
        "ctrl.txt",
    ),
    "divemesh": (
        "DIVEMesh_User_Guide/DIVEMesh-UserGuide.tex",
        r"\chapter{The",
        r"\end{document}",
        "DIVEMesh",
        "control.txt",
    ),
}

#: LaTeX symbols in parameter names and descriptions, as text
SYMBOLS = {
    r"\Delta": "Δ",
    r"\delta": "δ",
    r"\alpha": "α",
    r"\beta": "β",
    r"\gamma": "γ",
    r"\Gamma": "Γ",
    r"\epsilon": "ε",
    r"\varepsilon": "ε",
    r"\eta": "η",
    r"\theta": "θ",
    r"\Theta": "Θ",
    r"\kappa": "κ",
    r"\lambda": "λ",
    r"\mu": "μ",
    r"\nu": "ν",
    r"\xi": "ξ",
    r"\pi": "π",
    r"\rho": "ρ",
    r"\sigma": "σ",
    r"\tau": "τ",
    r"\phi": "φ",
    r"\varphi": "φ",
    r"\psi": "ψ",
    r"\omega": "ω",
    r"\Omega": "Ω",
    r"\chi": "χ",
    r"\zeta": "ζ",
    r"\partial": "∂",
    r"\infty": "∞",
    r"\geq": "≥",
    r"\leq": "≤",
    r"\ge": "≥",
    r"\le": "≤",
    r"\neq": "≠",
    r"\approx": "≈",
    r"\times": "×",
    r"\cdot": "·",
    r"\pm": "±",
    r"\circ": "°",
    r"\degree": "°",
    r"\rightarrow": "→",
    r"\to": "→",
    r"\sim": "~",
    r"\ldots": "…",
    r"\dots": "…",
    r"\cdots": "…",
    r"\quad": " ",
    r"\qquad": " ",
    r"\,": " ",
    r"\;": " ",
    r"\!": "",
    r"\%": "%",
    r"\&": "&",
    r"\_": "_",
    r"\#": "#",
    r"\$": "$",
    r"\{": "{",
    r"\}": "}",
    r"\\": " ",
    "~": " ",
    "--": "–",
}

_ITEM = re.compile(r"\\item\s*\[\s*([A-Z])\s+(\d+)\s*(\$?\s*\*\s*\$?)?\s*\]")
_TYPE = re.compile(r"\{\\color\{blue\}\s*\{\s*(int|double|char|string|text)\s*\}\s*\}")
_VALID = re.compile(r"\\item\s*\[\s*((?:\\(?:yes|no)\s*(?:\\quad)?\s*){4})\]")
_OPTION = re.compile(r"\\item\s*\[([^\]]*)\]\s*(.*)")


def plain(text: str) -> str:
    """LaTeX of a description as plain text: commands and math markup out."""
    text = re.sub(r"(?<!\\)%.*", "", text)  # comments
    text = re.sub(r"\\(?:citep?|citet|ref|eqref|label|cite)\{[^}]*\}", "", text)
    text = re.sub(r"\\(?:vspace|hspace|needspace)\*?\{[^}]*\}", "", text)
    text = re.sub(
        r"\\(?:mathrm|texttt|textbf|textit|text|emph|mathbf|mathit|bf|it|tt)(?![a-zA-Z])\s*",
        "",
        text,
    )
    text = re.sub(r"\\(?:frac)\{([^{}]*)\}\{([^{}]*)\}", r"(\1)/(\2)", text)
    text = re.sub(r"\\(?:sqrt)\{([^{}]*)\}", r"√(\1)", text)
    text = re.sub(r"\\(?:left|right|big|Big)\s*", "", text)
    text = re.sub(r"\\(?:overline|bar|hat|tilde|vec)\{([^{}]*)\}", r"\1", text)
    text = re.sub(r"\{\\color\{[a-z]+\}\s*\{([^{}]*)\}\s*\}", r"\1", text)
    text = re.sub(r"\\color\{[a-z]+\}", "", text)
    for symbol in sorted(SYMBOLS, key=len, reverse=True):
        if symbol.startswith("\\") and symbol[1:].isalpha():
            text = re.sub(re.escape(symbol) + r"(?![a-zA-Z])", SYMBOLS[symbol], text)
        else:
            text = text.replace(symbol, SYMBOLS[symbol])
    text = re.sub(r"_\{([^{}]*)\}", r"_\1", text)
    text = re.sub(r"\^\{([^{}]*)\}", r"^\1", text)
    text = re.sub(r"\\[a-zA-Z]+\*?", "", text)  # anything else
    text = text.replace("$", "").replace("{", "").replace("}", "")
    return re.sub(r"\s+", " ", text).strip()


def _params(head: str) -> tuple[list[dict], str]:
    """The values and the title of a command head: ``{int} N1, {double} x1 Title``
    (REEF3D, a type before each value) or ``{double} title; $x_{start}, x_{end}$``
    (DIVEMesh, one type, names after the semicolon)."""
    types = list(_TYPE.finditer(head))
    if not types:
        return [], plain(head)
    if ";" in head[types[-1].end() :] and len(types) == 1:
        kind = types[0].group(1)
        before, after = head[types[0].end() :].split(";", 1)
        names = [plain(n) for n in re.split(r",(?![^{]*\})", after) if plain(n)]
        # "each of the 4 points is given by ..." has no short names: the title says it
        if not names or any(len(n) > 24 for n in names):
            return [{"name": "", "type": kind, "count": 0}], plain(before + ";" + after)
        return [{"name": n, "type": kind} for n in names], plain(before)
    params = []
    title = ""
    for index, match in enumerate(types):
        end = types[index + 1].start() if index + 1 < len(types) else len(head)
        rest = head[match.end() : end]
        if index + 1 < len(types):
            # the value's name, up to its comma; text before the next type belongs to it
            params.append({"name": plain(rest.rstrip().rstrip(",")), "type": match.group(1)})
            continue
        # last value: its name is the first math or word group, the rest is the title
        rest = rest.strip()
        group = re.match(r"((?:\$[^$]*\$\s*,?\s*)+)(.*)", rest, re.S)
        if group and len(types) == 1 and "," in group.group(1):
            # one type for a list of values: ``{double} $x_1, y_1, z_1$, $x_2, ...$``
            names = [plain(n) for n in group.group(1).replace("$", "").split(",") if plain(n)]
            params.extend({"name": n, "type": match.group(1)} for n in names)
            title = group.group(2)
            break
        name_match = re.match(r"(\$[^$]*\$|[^\s,]+)\s*,?\s*(.*)", rest, re.S)
        if name_match is None:
            params.append({"name": "", "type": match.group(1)})
            break
        name, title = name_match.group(1), name_match.group(2)
        # a lone type before the title (``{int} Number of ...``): the value has no name
        if (
            not name.startswith("$")
            and len(types) == 1
            and not re.match(r"^[a-zA-Z0-9_]{1,3}$", name)
        ):
            params.append({"name": "", "type": match.group(1)})
            title = rest
        elif not title.strip() and not name.startswith("$") and len(types) == 1:
            params.append({"name": "", "type": match.group(1)})  # ``{int} DEM``: a title
            title = name
        else:
            params.append({"name": plain(name), "type": match.group(1)})
    return params, plain(title)


def _default_values(text: str) -> list[str]:
    """The default values, if the text is only numbers: ``0.0 ; 0.0`` or ``[4x] 0.0 ;
    0.0 ; 0.0`` (each repeated); else none (``na``, ``not set``, words)."""
    repeat = 1
    found = re.match(r"\[(\d+)\s*x\]\s*(.*)", text)
    if found:
        repeat, text = int(found.group(1)), found.group(2)
    tokens = [t for t in re.split(r"[;\s]+", text) if t]
    try:
        for token in tokens:
            float(token)
    except ValueError:
        return []
    return tokens * repeat


def _count(params: list[dict], default: list[str]) -> int | None:
    """How many values the command takes (None: not known, e.g. one type, no names
    and no default)."""
    if len(params) == 1 and not params[0]["name"]:
        if params[0].get("count") == 0 and not default:
            return None
        return max(1, len(default))
    return len(params) or None


def _split_items(body: str) -> list[tuple[int, re.Match]]:
    return [(m.start(), m) for m in _ITEM.finditer(body)]


def parse(text: str, start: str, end: str) -> dict:
    """The commands between the start and end markers of a guide."""
    begin = text.index(start)
    stop = text.find(end, begin)
    body = text[begin : stop if stop > 0 else len(text)]
    sections = {}
    for match in re.finditer(r"\\section\*?\{\s*([A-Z])\s*::\s*([^}]*)\}", body):
        sections[match.group(1)] = plain(match.group(2))
    commands: dict[str, dict] = {}
    items = _split_items(body)
    for index, (position, match) in enumerate(items):
        after = body[match.end() : items[index + 1][0] if index + 1 < len(items) else len(body)]
        if not _TYPE.match(after.lstrip()) and not after.lstrip().startswith("input for"):
            # "input for the three zones of B 101 8: {int} N1, ..." (DIVEMesh) is a
            # command too; any other item (a cross reference) is not
            if not _TYPE.search(after.split("\n", 1)[0]):
                continue
        code = f"{match.group(1)} {int(match.group(2))}"
        if code in commands:
            continue  # the first definition counts
        # the head: up to "\\" or the end of the line
        line_end = after.find("\n")
        head = after if line_end < 0 else after[:line_end]
        long = "" if line_end < 0 else after[line_end:]
        if "\\\\" in head:
            head, extra = head.split("\\\\", 1)
            long = extra + long
        # options and default in the description list that follows
        options: dict[str, str] = {}
        default: list[str] = []
        default_text = ""
        description = re.search(r"\\begin\{description\}(.*?)\\end\{description\}", long, re.S)
        text_part = long[: description.start()] if description else long
        if description:
            for option in _OPTION.finditer(description.group(1)):
                key, label = option.group(1).strip(), plain(option.group(2))
                if key.startswith("default"):
                    default_text = plain(label)
                    default = _default_values(default_text)
                elif key and not key.startswith("\\"):
                    options[plain(key)] = label
        # validity marks (REEF3D): the \yes/\no item just before this one
        valid = ""
        before = body[max(0, position - 400) : position]
        marks = list(_VALID.finditer(before))
        if marks:
            flags = re.findall(r"\\(yes|no)", marks[-1].group(1))
            valid = "".join(c for c, f in zip("SFNC", flags, strict=False) if f == "yes")
        head_text = head
        prefix = ""
        if not _TYPE.match(head.lstrip()):  # "input for the three zones of B 101 8: ..."
            first = _TYPE.search(head)
            prefix = plain(head[: first.start()]).rstrip(":") if first else ""
            head_text = head[first.start() :] if first else head
        params, title = _params(head_text)
        if prefix:
            title = prefix + (f"; {title}" if title else "")
        if not title:  # only the values are named (A 371: U_10, direction)
            title = ", ".join(p["name"] for p in params if p["name"])
        entry = {
            "title": title,
            "repeat": bool(match.group(3)),
            "params": params,
            "values": _count(params, default),
            "options": options,
            "default": default,
            "text": plain(text_part)[:1200],
        }
        if default_text and not default:
            entry["default_text"] = default_text
        if valid:
            entry["modules"] = valid
        commands[code] = entry
    return {"sections": sections, "commands": commands}


def read_source(path: Path) -> dict[str, dict]:
    """The commands a program's reader (``read_control.cpp``) knows: code -> how many
    values it reads and whether it may be repeated (read in the second pass into
    arrays, one per line)."""
    text = path.read_text(encoding="utf-8", errors="replace")
    text = re.sub(r"//[^\n]*", "", text)
    passes = [m.start() for m in re.finditer(r"while\s*\(\s*!\s*control\.eof\(\)\s*\)", text)]
    # the code each "case N:" belongs to: the letter of the "case 'X':" before it
    # (``case 'S': case 'O':`` read the same commands: the first letter counts)
    marks = []
    for match in re.finditer(r"case\s+'([A-Z])'\s*:|case\s+(\d+)\s*:", text):
        marks.append((match.start(), match.end(), match.group(1), match.group(2)))
    found: dict[str, dict] = {}
    letter = ""
    fresh = False  # a letter with no command yet: a following letter is an alias
    for index, (start, end, char, number) in enumerate(marks):
        if char:
            if not (fresh and letter):
                letter = char
            fresh = True
            continue
        fresh = False
        if not letter:
            continue
        stop = marks[index + 1][0] if index + 1 < len(marks) else len(text)
        body = text[end:stop]
        reads = 0
        for statement in re.finditer(r"((?:for\s*\([^)]*\)\s*)?)control\s*>>([^;]*);", body):
            names = [n.strip() for n in statement.group(2).split(">>")]
            if names == ["test"] or names[0] == "test":
                continue  # a look ahead (B 96), not a value
            loop = re.search(r"<\s*(\d+)", statement.group(1) or "")
            reads += len(names) * (int(loop.group(1)) if loop else 1)
        code = f"{letter} {int(number)}"
        second = len(passes) >= 2 and start > passes[1]
        entry = found.setdefault(code, {"values": 0, "repeat": False})
        entry["values"] = max(entry["values"], reads)
        entry["repeat"] = entry["repeat"] or second
    return found


#: commands the guides do not describe yet, as REEF3D's reader and vrans_ini.cpp
#: take them: (title, parameter names, modules, text)
_MATERIAL = ["n", "d_50", "α", "β"]
_VRANS = (
    "Porosity n, mean grain diameter d_50 in m and the linear and nonlinear "
    "resistance coefficients α and β follow the coordinates in m."
)
FROM_READER = {
    "B 270": (
        "VRANS porous box",
        ["x_start", "x_end", "y_start", "y_end", "z_start", "z_end", *_MATERIAL],
        "C",
        _VRANS,
    ),
    "B 274": (
        "VRANS porous vertical cylinder",
        ["x_center", "y_center", "z_start", "z_end", "radius", *_MATERIAL],
        "C",
        _VRANS,
    ),
    "B 281": (
        "VRANS porous wedge in x-direction",
        ["x_start", "x_end", "y_start", "y_end", "z_start", "z_end", *_MATERIAL],
        "C",
        "The top runs from z_start at x_start to z_end at x_end, the base at the lower. " + _VRANS,
    ),
    "B 282": (
        "VRANS porous wedge in y-direction",
        ["x_start", "x_end", "y_start", "y_end", "z_start", "z_end", *_MATERIAL],
        "C",
        "The top runs from z_start at y_start to z_end at y_end, the base at the lower. " + _VRANS,
    ),
    "B 291": (
        "VRANS porous plate in x-direction",
        ["x_start", "x_end", "y_start", "y_end", "z_start", "z_end", "d", *_MATERIAL],
        "C",
        "A band of thickness d in m above the line from z_start at x_start to z_end at "
        "x_end. " + _VRANS,
    ),
}


def describe_from_reader(commands: dict) -> dict:
    """Titles, names and texts for commands the guides do not describe yet
    (FROM_READER); the guide's own description, once there, wins."""
    for code, (title, names, modules, text) in FROM_READER.items():
        entry = commands.get(code)
        if entry is None or entry.get("documented", True):
            continue
        entry.update(
            title=title,
            params=[{"name": n, "type": "double"} for n in names],
            modules=modules,
            text=text + " Not in the user guide yet.",
        )
    return commands


def merge(catalog: dict, source: dict[str, dict]) -> dict:
    """The guide's commands with the reader's counts; commands only the reader knows
    are added, undocumented; commands the reader no longer has are marked."""
    commands = catalog["commands"]
    for code, entry in commands.items():
        known = source.get(code)
        if known is None:
            entry["in_program"] = False
            continue
        if known["values"]:
            entry["values"] = known["values"]
        entry["repeat"] = entry["repeat"] or known["repeat"]
    for code, known in source.items():
        if code in commands or not known["values"]:
            continue
        commands[code] = {
            "title": "",
            "repeat": known["repeat"],
            "params": [],
            "values": known["values"],
            "options": {},
            "default": [],
            "text": "Not in the user guide yet.",
            "documented": False,
        }
    describe_from_reader(commands)
    catalog["commands"] = dict(
        sorted(commands.items(), key=lambda kv: (kv[0].split()[0], int(kv[0].split()[1])))
    )
    return catalog


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("guides", nargs="?", default="~/Codelite/UserGuides")
    parser.add_argument("--reef3d", default="~/Codelite/REEF3D", help="REEF3D source tree")
    parser.add_argument("--divemesh", default="~/Codelite/DIVEMesh", help="DIVEMesh source tree")
    args = parser.parse_args(argv)
    guides = Path(args.guides).expanduser()
    trees = {"reef3d": Path(args.reef3d).expanduser(), "divemesh": Path(args.divemesh).expanduser()}
    OUT.mkdir(parents=True, exist_ok=True)
    for name, (path, start, end, program, file) in GUIDES.items():
        source = guides / path
        catalog = parse(source.read_text(encoding="utf-8", errors="replace"), start, end)
        catalog = {"program": program, "file": file, "source": path} | catalog
        reader = trees[name] / "src" / "read_control.cpp"
        if reader.is_file():
            catalog = merge(catalog, read_source(reader))
        else:
            print(f"{reader} not found: counts from the guide only")
        target = OUT / f"{name}.json"
        target.write_text(
            json.dumps(catalog, indent=1, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        print(f"{target.relative_to(ROOT)}: {len(catalog['commands'])} commands")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
