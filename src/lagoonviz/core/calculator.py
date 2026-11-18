# SPDX-License-Identifier: GPL-3.0-or-later
# Architect: Hans Bihs
"""Calculated fields: new variables of a case made of the ones REEF3D wrote, as
ParaView's calculator makes them.

A case's *Calculated fields* (its item's ``calculated`` setting) define them, one
per line or separated by ``;``::

    nu_rel = eddyv / nu                 # the eddy viscosity over the water's viscosity
    speed = mag(velocity)
    ke = 0.5 * rho * (velocity.x**2 + velocity.y**2 + velocity.z**2)
    hot = where(nu_rel > 100, 1, 0)

A field is then a variable of every output that has what it is made of: to colour
by, for contour lines, iso-lines on slices, iso-surfaces and iso-volumes ("the
fluid where nu_rel is above 100": an iso-volume of nu_rel from 100). Fields may use
the ones defined before them.

In an expression: the variables by name (a space as ``_``: ``vorticity_x``), a
vector's components as ``velocity.x`` (``.y``, ``.z``), numbers, ``+ - * / **``,
comparisons (1 where true, 0 where not), ``and``, ``or``, ``not``; the functions
``sqrt abs exp log log10 sin cos tan arctan2 mag min max where clip``; the case's
constants ``nu`` and ``rho`` (water: W 2, W 1 of ctrl.txt), ``nu_air`` and ``rho_air``
(W 4, W 3), ``g`` (W 22) and ``pi``. Nothing else runs: an expression is checked
against these before it is evaluated.
"""

from __future__ import annotations

import ast
import logging
import math
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)

#: REEF3D's defaults (ini.cpp) of the constants an expression may use
DEFAULT_CONSTANTS = {
    "rho": 998.2,  # W 1
    "nu": 1.004e-6,  # W 2
    "rho_air": 1.205,  # W 3
    "nu_air": 1.41e-5,  # W 4
    "g": 9.81,  # |W 22|
    "pi": math.pi,
}
_FROM_CTRL = {"W 1": "rho", "W 2": "nu", "W 3": "rho_air", "W 4": "nu_air", "W 22": "g"}


def _mag(v):
    v = np.asarray(v, np.float64)
    return np.linalg.norm(v, axis=-1) if v.ndim == 2 else np.abs(v)


FUNCTIONS = {
    "sqrt": np.sqrt,
    "abs": np.abs,
    "exp": np.exp,
    "log": np.log,
    "log10": np.log10,
    "sin": np.sin,
    "cos": np.cos,
    "tan": np.tan,
    "arctan2": np.arctan2,
    "mag": _mag,
    "min": np.minimum,
    "max": np.maximum,
    "where": np.where,
    "clip": np.clip,
}
_COMPONENTS = {"x": 0, "y": 1, "z": 2}
_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_ALLOWED = (
    ast.Expression,
    ast.BinOp,
    ast.UnaryOp,
    ast.BoolOp,
    ast.Compare,
    ast.Call,
    ast.Name,
    ast.Load,
    ast.Attribute,
    ast.Constant,
    ast.Add,
    ast.Sub,
    ast.Mult,
    ast.Div,
    ast.Pow,
    ast.Mod,
    ast.FloorDiv,
    ast.USub,
    ast.UAdd,
    ast.Not,
    ast.And,
    ast.Or,
    ast.Lt,
    ast.LtE,
    ast.Gt,
    ast.GtE,
    ast.Eq,
    ast.NotEq,
)


@dataclass(frozen=True)
class Field:
    """A calculated field: its name, its expression and what it is made of."""

    name: str
    expression: str
    #: the names it reads (variables of the output, or fields defined before it)
    inputs: tuple[str, ...]


class CalculatorError(ValueError):
    """An expression that cannot be calculated."""


def parse(text: str) -> list[Field]:
    """The fields of a case's *Calculated fields* (module doc), in order; a line that
    is not one is reported (:class:`CalculatorError`)."""
    fields: list[Field] = []
    for raw in re.split(r"[;\n]", str(text or "")):
        line = raw.split("#")[0].strip()
        if not line:
            continue
        name, equals, expression = line.partition("=")
        name, expression = name.strip(), expression.strip()
        if not equals or not _NAME.match(name) or not expression:
            raise CalculatorError(f"{line!r}: write name = expression")
        if name in FUNCTIONS or name in DEFAULT_CONSTANTS:
            raise CalculatorError(f"{name!r} is a function or constant of the calculator")
        fields.append(Field(name, expression, _inputs(expression)))
    return fields


def parse_quietly(text: str) -> list[Field]:
    """:func:`parse`, the lines that are not fields left out (and logged)."""
    fields = []
    for raw in re.split(r"[;\n]", str(text or "")):
        try:
            fields += parse(raw)
        except CalculatorError as error:
            log.warning("calculated field not used: %s", error)
    return fields


def _tree(expression: str) -> ast.Expression:
    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError as error:
        raise CalculatorError(f"{expression!r}: {error.msg}") from None
    for node in ast.walk(tree):
        if not isinstance(node, _ALLOWED):
            raise CalculatorError(f"{expression!r}: {type(node).__name__} is not allowed")
        if isinstance(node, ast.Call) and (
            not isinstance(node.func, ast.Name) or node.func.id not in FUNCTIONS or node.keywords
        ):
            raise CalculatorError(f"{expression!r}: only the calculator's functions")
        if isinstance(node, ast.Attribute) and (
            node.attr not in _COMPONENTS or not isinstance(node.value, ast.Name)
        ):
            raise CalculatorError(f"{expression!r}: a component is .x, .y or .z of a variable")
        if isinstance(node, ast.Constant) and not isinstance(node.value, (int, float)):
            raise CalculatorError(f"{expression!r}: numbers only")
    return tree


def _inputs(expression: str) -> tuple[str, ...]:
    """The variables an expression reads (not the functions or constants)."""
    tree = _tree(expression)
    called = {n.func.id for n in ast.walk(tree) if isinstance(n, ast.Call)}
    names = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and node.id not in called:
            if node.id not in DEFAULT_CONSTANTS and node.id not in names:
                names.append(node.id)
    return tuple(names)


def array_of(name: str, available) -> str | None:
    """The array an expression's name stands for among ``available``: itself, or with
    its ``_`` as spaces (``vorticity_x``: "vorticity x")."""
    if name in available:
        return name
    spaced = name.replace("_", " ")
    return spaced if spaced in available else None


def usable(fields: list[Field], components: dict[str, int]) -> dict[str, int]:
    """The fields an output can calculate (all they read it has, or fields before
    them), and their components (1, or 3 for a vector)."""
    found: dict[str, int] = {}
    known = dict(components)
    for f in fields:
        if all(array_of(n, known) is not None for n in f.inputs):
            try:
                sample = {array_of(n, known): _sample(known[array_of(n, known)]) for n in f.inputs}
                value = evaluate(f, sample)
            except (CalculatorError, ValueError, TypeError, IndexError, FloatingPointError):
                continue
            count = 3 if np.ndim(value) == 2 and np.shape(value)[1] == 3 else 1
            found[f.name] = count
            known[f.name] = count
    return found


def _sample(count: int) -> np.ndarray:
    return np.ones((2, count)) if count > 1 else np.ones(2)


def needed(names, fields: list[Field], components: dict[str, int]) -> tuple[list[str], list[Field]]:
    """The output's own arrays to read for ``names`` (variables, fields), and the
    fields to calculate after, in their order."""
    by_name = {f.name: f for f in fields}
    arrays: list[str] = []
    wanted: set[str] = set()

    def visit(name: str) -> None:
        own = array_of(name, components)
        if own is not None and name not in by_name:
            if own not in arrays:
                arrays.append(own)
            return
        f = by_name.get(name)
        if f is None or name in wanted:
            return
        wanted.add(name)
        for n in f.inputs:
            visit(n)

    for name in names:
        visit(str(name).partition(".")[0])
    return arrays, [f for f in fields if f.name in wanted]


def evaluate(f: Field, data: dict[str, np.ndarray], constants: dict | None = None):
    """A field's values at the points of ``data`` (arrays by name)."""
    tree = _tree(f.expression)
    scope = {**DEFAULT_CONSTANTS, **(constants or {})}

    def value(node):
        if isinstance(node, ast.Expression):
            return value(node.body)
        if isinstance(node, ast.Constant):
            return float(node.value)
        if isinstance(node, ast.Name):
            own = array_of(node.id, data)
            if own is not None:
                return np.asarray(data[own], np.float64)
            if node.id in scope:
                return float(scope[node.id])
            raise CalculatorError(f"{f.name}: {node.id!r} is not there")
        if isinstance(node, ast.Attribute):
            vector = value(node.value)
            if np.ndim(vector) != 2:
                raise CalculatorError(f"{f.name}: {node.value.id!r} has no components")
            return vector[:, _COMPONENTS[node.attr]]
        if isinstance(node, ast.UnaryOp):
            v = value(node.operand)
            return (
                -v
                if isinstance(node.op, ast.USub)
                else (1.0 - _truth(v))
                if isinstance(node.op, ast.Not)
                else v
            )
        if isinstance(node, ast.BinOp):
            a, b = _paired(value(node.left), value(node.right))
            op = node.op
            if isinstance(op, ast.Add):
                return a + b
            if isinstance(op, ast.Sub):
                return a - b
            if isinstance(op, ast.Mult):
                return a * b
            if isinstance(op, ast.Div):
                return a / b
            if isinstance(op, ast.Pow):
                return a**b
            if isinstance(op, ast.Mod):
                return a % b
            return a // b
        if isinstance(node, ast.BoolOp):
            parts = [_truth(value(v)) for v in node.values]
            out = parts[0]
            for p in parts[1:]:
                out = out * p if isinstance(node.op, ast.And) else np.maximum(out, p)
            return out
        if isinstance(node, ast.Compare):
            left = value(node.left)
            out = 1.0
            for op, right in zip(node.ops, node.comparators, strict=True):
                r = value(right)
                a, b = _paired(left, r)
                test = {
                    ast.Lt: np.less,
                    ast.LtE: np.less_equal,
                    ast.Gt: np.greater,
                    ast.GtE: np.greater_equal,
                    ast.Eq: np.equal,
                    ast.NotEq: np.not_equal,
                }[type(op)](a, b)
                out = out * np.asarray(test, np.float64)
                left = r
            return out
        if isinstance(node, ast.Call):
            args = [value(a) for a in node.args]
            return FUNCTIONS[node.func.id](*args)
        raise CalculatorError(f"{f.name}: cannot calculate {type(node).__name__}")

    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        result = value(tree)
    return np.asarray(result, np.float64)


def _truth(v):
    return (np.asarray(v, np.float64) != 0).astype(np.float64)


def _paired(a, b):
    """A vector and a scalar per point side by side (each component times it)."""
    if np.ndim(a) == 2 and np.ndim(b) == 1:
        return a, np.asarray(b)[:, None]
    if np.ndim(b) == 2 and np.ndim(a) == 1:
        return np.asarray(a)[:, None], b
    return a, b


def add_fields(
    point_data: dict[str, np.ndarray], fields: list[Field], constants: dict | None = None
) -> dict[str, np.ndarray]:
    """``point_data`` with the fields calculated (as float32), in their order; a field
    whose values cannot be made is left out (and logged)."""
    if not fields:
        return point_data
    data = dict(point_data)
    n = len(next(iter(data.values()))) if data else 0
    for f in fields:
        try:
            values = evaluate(f, data, constants)
        except (CalculatorError, ValueError, TypeError, IndexError) as error:
            log.warning("%s not calculated: %s", f.name, error)
            continue
        if np.ndim(values) == 0:
            values = np.full(n, float(values))
        data[f.name] = np.asarray(values, np.float32)
    return data


def constants_of(case_dir: str | Path) -> dict[str, float]:
    """The constants of a case (its ctrl.txt's W 1 - W 4 and W 22, else REEF3D's
    defaults)."""
    found = dict(DEFAULT_CONSTANTS)
    for where in (Path(case_dir), *list(Path(case_dir).parents)[:1]):
        ctrl = where / "ctrl.txt"
        if not ctrl.is_file():
            continue
        try:
            text = ctrl.read_text(encoding="utf-8", errors="replace")
        except OSError:
            break
        for line in text.splitlines():
            words = line.split("//")[0].split()
            key = " ".join(words[:2])
            if key in _FROM_CTRL and len(words) > 2:
                try:
                    value = float(words[2])
                except ValueError:
                    continue
                found[_FROM_CTRL[key]] = abs(value) if key == "W 22" else value
        break
    return found
