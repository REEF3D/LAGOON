# SPDX-License-Identifier: GPL-3.0-or-later
# Architect: Hans Bihs
"""The input files of a case: REEF3D's ``ctrl.txt`` and DIVEMesh's ``control.txt``.

A :class:`ControlFile` keeps every line of the file: the commands (``P 51 0.3 0.5``),
comments (``// gauges``), blank lines and anything it does not understand. It
answers what the set-up needs (``file.values("B 10")``) and edits commands in place,
so a file written back differs from the one read only in the commands that were
changed, added or removed::

    control = ControlFile.read("case/control.txt")
    control.set("B 1", [0.025])          # the first B 1, or a new line
    control.add("S 33", [5.0, 2.0, 0.5]) # one more cylinder
    control.write()

The :class:`Catalog` lists the commands of both programs as their user guides
describe them (``catalog/reef3d.json``, ``catalog/divemesh.json``, made by
``tools/command_catalog.py``): title, values, choices, default. :meth:`Catalog.check`
finds what REEF3D or DIVEMesh would stumble over (unknown commands, missing values).

Both programs read their file token by token: a capital letter and a number start a
command, its values follow, and a token starting with ``/`` makes the rest of the
line a comment.
"""

from __future__ import annotations

import copy
import json
import math
from dataclasses import dataclass, field
from functools import cache
from pathlib import Path

CTRL = "ctrl.txt"
CONTROL = "control.txt"

#: which program reads which file
PROGRAMS = {CTRL: "reef3d", CONTROL: "divemesh"}

_CATALOG_DIR = Path(__file__).resolve().parent / "catalog"


# ======================================================================== lines
@dataclass
class Line:
    """One line of a control file: a command (``code`` "P 51", its ``values`` as
    written) or not (``code`` None: a comment, a blank line, text)."""

    text: str
    code: str | None = None
    values: list[str] = field(default_factory=list)
    #: what follows the values from the first ``/`` on, with the space before it
    comment: str = ""

    @classmethod
    def parse(cls, text: str) -> Line:
        tokens = text.split()
        if len(tokens) < 2 or len(tokens[0]) != 1 or not tokens[0].isupper():
            return cls(text)
        try:
            number = int(tokens[1])
        except ValueError:
            return cls(text)
        values = []
        for token in tokens[2:]:
            if token.startswith("/"):
                break
            values.append(token)
        comment = ""
        slash = _comment_start(text)
        if slash >= 0:
            start = slash
            while start > 0 and text[start - 1] in " \t":
                start -= 1
            comment = text[start:]
        return cls(text, f"{tokens[0]} {number}", values, comment)

    def numbers(self) -> list[float] | None:
        """The values as numbers (None if one of them is not a number)."""
        try:
            return [float(v) for v in self.values]
        except ValueError:
            return None

    def rewrite(self) -> None:
        """The text from code, values and comment (after an edit)."""
        assert self.code is not None
        self.text = " ".join([self.code, *self.values]) + self.comment


def _comment_start(text: str) -> int:
    """Index of the first token that starts with ``/`` (-1: none)."""
    position = 0
    for token in text.split():
        position = text.index(token, position)
        if token.startswith("/"):
            return position
        position += len(token)
    return -1


def format_value(value, kind: str = "double") -> str:
    """A value as it goes into a control file: integers plain, decimals short but
    exact enough (10 significant digits)."""
    if isinstance(value, str):
        return value.strip()
    if kind == "int":
        return str(int(round(float(value))))
    value = float(value)
    if math.isfinite(value) and value == int(value) and abs(value) < 1e15:
        return f"{int(value)}.0"
    return f"{value:.10g}"


# ======================================================================== the file
class ControlFile:
    """The lines of a ``ctrl.txt`` or ``control.txt`` (see the module docstring)."""

    def __init__(self, path: str | Path, text: str = "") -> None:
        self.path = Path(path)
        self.lines: list[Line] = [Line.parse(t) for t in text.splitlines()]
        #: the file was found when read (an empty ControlFile stands for a missing one)
        self.exists = False
        #: counts edits: views compare it to know whether to draw again
        self.revision = 0
        #: text as last read or written (to tell whether there are unsaved edits)
        self._saved = self.text()

    @classmethod
    def read(cls, path: str | Path) -> ControlFile:
        path = Path(path)
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            return cls(path)
        file = cls(path, text)
        file.exists = True
        return file

    @property
    def name(self) -> str:
        return self.path.name

    @property
    def program(self) -> str:
        """ "reef3d" (ctrl.txt) or "divemesh" (control.txt)."""
        return PROGRAMS.get(self.path.name, "reef3d")

    def copy(self) -> ControlFile:
        return copy.deepcopy(self)

    # ------------------------------------------------------------- reading
    def __contains__(self, code: object) -> bool:
        return any(line.code == code for line in self.lines)

    def __bool__(self) -> bool:
        return any(line.code for line in self.lines)

    def codes(self) -> list[str]:
        """The commands in the file, each once, in order of their first line."""
        seen: dict[str, None] = {}
        for line in self.lines:
            if line.code:
                seen.setdefault(line.code, None)
        return list(seen)

    def entries(self, code: str | None = None) -> list[Line]:
        """The command lines (of one command, or all)."""
        return [line for line in self.lines if line.code and (code is None or line.code == code)]

    def values(self, code: str, occurrence: int = 0) -> list[float] | None:
        """The values of a command as numbers: of its first line, or the
        ``occurrence``-th (None: not given, or not numbers)."""
        lines = self.entries(code)
        if occurrence >= len(lines):
            return None
        return lines[occurrence].numbers()

    def all(self, code: str) -> list[list[float]]:
        """The values of every line of a repeated command (``P 51``: one per gauge)."""
        found = []
        for line in self.entries(code):
            numbers = line.numbers()
            if numbers is not None:
                found.append(numbers)
        return found

    def value(self, code: str, default: float | None = None, index: int = 0) -> float | None:
        """One value of a command (its first line), or ``default``."""
        numbers = self.values(code)
        if numbers is None or index >= len(numbers):
            return default
        return numbers[index]

    def int(self, code: str, default: int = 0, index: int = 0) -> int:
        number = self.value(code, None, index)
        return default if number is None else int(number)

    # ------------------------------------------------------------- editing
    def set(self, code: str, values, occurrence: int = 0, kinds=None) -> None:
        """Give a command these values: its ``occurrence``-th line, or a new line if
        there are fewer. ``kinds``: "int" or "double" per value (default: as the
        catalogue says)."""
        lines = self.entries(code)
        texts = self._texts(code, values, kinds)
        if occurrence < len(lines):
            line = lines[occurrence]
            if line.values == texts:
                return
            line.values = texts
            line.rewrite()
        else:
            self._insert(Line(" ".join([code, *texts]), code, texts))
        self.revision += 1

    def add(self, code: str, values, kinds=None) -> int:
        """One more line of a (repeated) command; returns its occurrence."""
        texts = self._texts(code, values, kinds)
        self._insert(Line(" ".join([code, *texts]), code, texts))
        self.revision += 1
        return len(self.entries(code)) - 1

    def remove(self, code: str, occurrence: int | None = None) -> int:
        """Remove a command's line (all of its lines with ``occurrence`` None); returns
        how many lines went."""
        lines = self.entries(code)
        gone = lines if occurrence is None else lines[occurrence : occurrence + 1]
        if not gone:
            return 0
        ids = {id(line) for line in gone}
        self.lines = [line for line in self.lines if id(line) not in ids]
        self.revision += 1
        return len(gone)

    def set_text(self, text: str) -> None:
        """Replace the whole file (the text editor)."""
        if text == self.text():
            return
        self.lines = [Line.parse(t) for t in text.splitlines()]
        self.revision += 1

    def _texts(self, code: str, values, kinds) -> list[str]:
        if kinds is None:
            info = catalog_for(self.path.name).get(code)
            kinds = info.kinds(len(values)) if info else ["double"] * len(values)
        elif isinstance(kinds, str):
            kinds = [kinds] * len(values)
        return [format_value(v, k) for v, k in zip(values, kinds, strict=False)]

    def _insert(self, line: Line) -> None:
        """A new command line: after the last line of the same command, else after the
        last command of the same letter, else at the end (before trailing blanks)."""
        assert line.code is not None
        letter = line.code.split()[0]
        after = -1
        for index, other in enumerate(self.lines):
            if other.code == line.code:
                after = index
        if after < 0:
            for index, other in enumerate(self.lines):
                if other.code and other.code.split()[0] == letter:
                    after = index
        if after < 0:
            after = len(self.lines) - 1
            while after >= 0 and not self.lines[after].text.strip():
                after -= 1
        self.lines.insert(after + 1, line)

    # ------------------------------------------------------------- writing
    def text(self) -> str:
        if not self.lines:
            return ""
        return "\n".join(line.text for line in self.lines) + "\n"

    @property
    def modified(self) -> bool:
        """Edited since it was read or last written."""
        return self.text() != self._saved

    def write(self, path: str | Path | None = None) -> Path:
        target = Path(path) if path is not None else self.path
        target.write_text(self.text(), encoding="utf-8")
        if path is None or target == self.path:
            self._saved = self.text()
            self.exists = True
        return target

    def mark_saved(self) -> None:
        self._saved = self.text()


def read_commands(path: str | Path) -> ControlFile:
    """A control file, empty if there is none."""
    return ControlFile.read(path)


# ======================================================================== the catalogue
@dataclass(frozen=True)
class Param:
    name: str
    type: str  # "int", "double", "char", ...


@dataclass(frozen=True)
class CommandInfo:
    """A command as the user guide describes it."""

    code: str
    title: str
    #: it may be given several times (one line per gauge, solid, ...)
    repeat: bool
    params: tuple[Param, ...]
    #: how many values it takes (None: not known)
    count: int | None
    #: choices of a switch: value -> meaning
    options: dict[str, str]
    default: tuple[str, ...]
    default_text: str
    text: str
    #: REEF3D: the modules it is used by, letters of "SFNC" (SFLOW, FNPF, NHFLOW, CFD)
    modules: str
    section: str

    @property
    def letter(self) -> str:
        return self.code.split()[0]

    @property
    def number(self) -> int:
        return int(self.code.split()[1])

    def kinds(self, n: int) -> list[str]:
        """The type of each of n values."""
        if not self.params:
            return ["double"] * n
        if len(self.params) == 1 and not self.params[0].name:
            return [self.params[0].type] * n
        kinds = [p.type for p in self.params]
        return (kinds + [kinds[-1]] * n)[:n]

    def names(self) -> list[str]:
        """A short name for each value (``x``, ``y``; ``value 1`` ... when unnamed)."""
        n = self.count or len(self.params) or 1
        if len(self.params) == n and all(p.name for p in self.params):
            return [p.name for p in self.params]
        if n == 1:
            return ["value"]
        return [f"value {i + 1}" for i in range(n)]

    def label(self) -> str:
        return f"{self.code} · {self.title}" if self.title else self.code

    def used_by(self, solver: str) -> bool:
        """REEF3D: whether the module (A 10: "SFLOW", "FNPF", "NHFLOW", "CFD") uses it."""
        if not self.modules or not solver:
            return True
        letter = {"SFLOW": "S", "FNPF": "F", "NHFLOW": "N", "CFD": "C"}.get(solver.upper())
        return letter is None or letter in self.modules


@dataclass(frozen=True)
class Problem:
    """Something REEF3D or DIVEMesh would stumble over, by line (1-based)."""

    line: int
    code: str
    message: str
    #: "error" (the program stops or misreads) or "warning" (ignored, or no effect)
    level: str = "warning"


class Catalog:
    """The commands of one program (see the module docstring)."""

    def __init__(self, data: dict) -> None:
        self.program: str = data.get("program", "")
        self.file: str = data.get("file", "")
        self.sections: dict[str, str] = dict(data.get("sections", {}))
        self.commands: dict[str, CommandInfo] = {}
        for code, entry in data.get("commands", {}).items():
            letter = code.split()[0]
            self.commands[code] = CommandInfo(
                code=code,
                title=entry.get("title", ""),
                repeat=bool(entry.get("repeat")),
                params=tuple(
                    Param(p.get("name", ""), p.get("type", "double"))
                    for p in entry.get("params", [])
                ),
                count=entry.get("values"),
                options=dict(entry.get("options", {})),
                default=tuple(entry.get("default", [])),
                default_text=entry.get("default_text", ""),
                text=entry.get("text", ""),
                modules=entry.get("modules", ""),
                section=self.sections.get(letter, ""),
            )

    @classmethod
    def load(cls, name: str) -> Catalog:
        """ "reef3d" or "divemesh" (empty if the file is missing)."""
        return _load(name)

    def get(self, code: str) -> CommandInfo | None:
        return self.commands.get(code)

    def __contains__(self, code: object) -> bool:
        return code in self.commands

    def __len__(self) -> int:
        return len(self.commands)

    def in_section(self, letter: str) -> list[CommandInfo]:
        return sorted(
            (c for c in self.commands.values() if c.letter == letter), key=lambda c: c.number
        )

    def search(self, text: str, solver: str = "") -> list[CommandInfo]:
        """Commands whose code, title or description has all the words of ``text``
        (code and title matches first)."""
        words = text.lower().split()
        if not words:
            return []
        exact = text.strip().upper()
        found = []
        for info in self.commands.values():
            head = f"{info.code} {info.title}".lower()
            if info.code == exact or info.code.replace(" ", "") == exact.replace(" ", ""):
                found.append((0, info))  # the command asked for, used by the module or not
            elif solver and not info.used_by(solver):
                continue
            elif all(w in head for w in words):
                found.append((1, info))
            elif all(w in (head + " " + info.text.lower()) for w in words):
                found.append((2, info))
        found.sort(key=lambda pair: (pair[0], pair[1].letter, pair[1].number))
        return [info for _, info in found]

    def check(self, file: ControlFile, solver: str = "") -> list[Problem]:
        """What the program would stumble over in a file."""
        problems = []
        seen: dict[str, int] = {}
        for number, line in enumerate(file.lines, start=1):
            stripped = line.text.strip()
            if line.code is None:
                if stripped and not stripped.startswith("/") and any(c.isupper() for c in stripped):
                    problems.append(
                        Problem(
                            number,
                            "",
                            "text without // before it: a capital letter in it starts a command",
                            "error",
                        )
                    )
                continue
            info = self.get(line.code)
            if info is None:
                problems.append(
                    Problem(
                        number, line.code, f"{line.code} is not a {self.program} command: ignored"
                    )
                )
                continue
            if solver and not info.used_by(solver):
                problems.append(
                    Problem(number, line.code, f"{line.code} has no effect in {solver}")
                )
            numbers = line.numbers()
            if numbers is None:
                problems.append(
                    Problem(number, line.code, f"{line.code}: a value is not a number", "error")
                )
            elif info.count is not None and len(numbers) < info.count:
                problems.append(
                    Problem(
                        number,
                        line.code,
                        f"{line.code} needs {info.count} value{'s' if info.count > 1 else ''}, "
                        f"has {len(numbers)}",
                        "error",
                    )
                )
            elif info.count is not None and len(numbers) > info.count:
                problems.append(
                    Problem(
                        number,
                        line.code,
                        f"{line.code} takes {info.count} value{'s' if info.count > 1 else ''}, "
                        f"has {len(numbers)}: the rest is read as the next command",
                        "error",
                    )
                )
            elif numbers is not None and info.options and len(numbers) == 1:
                key = format_value(numbers[0], "int")
                if key not in info.options and all(k.lstrip("-").isdigit() for k in info.options):
                    problems.append(
                        Problem(number, line.code, f"{line.code} {key}: not one of its choices")
                    )
            if line.code in seen and not info.repeat:
                problems.append(
                    Problem(
                        number,
                        line.code,
                        f"{line.code} is given again (line {seen[line.code]}): the last one counts",
                    )
                )
            seen.setdefault(line.code, number)
        return problems


@cache
def _load(name: str) -> Catalog:
    path = _CATALOG_DIR / f"{name}.json"
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        data = {}
    return Catalog(data)


def catalog_for(file_name: str) -> Catalog:
    """The catalogue of the program that reads a file (ctrl.txt: REEF3D, control.txt:
    DIVEMesh)."""
    return Catalog.load(PROGRAMS.get(Path(file_name).name, "reef3d"))
