# SPDX-License-Identifier: GPL-3.0-or-later
# Architect: Hans Bihs
"""Check a case's input before a run: ctrl.txt and control.txt together, the grid
DIVEMesh made for them, and what the run will cost (no Qt).

    report = check_case(case_dir)
    for problem in report.problems: problem.level, problem.where, problem.message
    report.facts   # [(name, text)]: cells, cells per process, time per iteration ...

What REEF3D or DIVEMesh would stop at comes first ("error"), then what is likely
not meant ("warning"), then hints ("note"). Each file's commands are checked with
the command catalog (unknown commands, missing values); across the files: the
number of processes (M 10) in both and the grid files DIVEMesh wrote for them, a
grid older than control.txt, no 3D output, no end. The cost comes from the grid
and, when the last run in the folder had the same grid, from that run's log.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from lagoonviz.core.data.controlfile import CONTROL, CTRL, Catalog
from lagoonviz.core.run_monitor import duration, read_tail

GRID_FOLDER = "DIVEMesh_Grid"
#: cells per process below which communication dominates, above which a step is slow
FEW_CELLS, MANY_CELLS = 5_000, 3_000_000
LEVELS = ("error", "warning", "note")


@dataclass
class Finding:
    level: str  # "error", "warning" or "note"
    where: str  # "ctrl.txt:12", "control.txt", "DIVEMesh_Grid", ...
    message: str


@dataclass
class InputReport:
    case_dir: Path
    solver: str = ""
    problems: list[Finding] = field(default_factory=list)
    facts: list[tuple[str, str]] = field(default_factory=list)

    @property
    def errors(self) -> list[Finding]:
        return [p for p in self.problems if p.level == "error"]

    @property
    def ok(self) -> bool:
        return not self.errors

    def summary(self) -> str:
        counts = {level: sum(p.level == level for p in self.problems) for level in LEVELS}
        if not self.problems:
            return "No problems found"
        parts = [f"{n} {level}{'s' if n != 1 else ''}" for level, n in counts.items() if n]
        return ", ".join(parts)

    def text(self) -> str:
        lines = [f"Input check of {self.case_dir}: {self.summary()}"]
        for problem in self.problems:
            lines.append(f"  {problem.level:7s} {problem.where}: {problem.message}")
        if self.facts:
            lines.append("")
            lines += [f"  {name}: {value}" for name, value in self.facts]
        return "\n".join(lines)


def _grid_files(case_dir: Path) -> dict[int, Path]:
    """DIVEMesh's grid file of each process (grid-000001.dat: process 1)."""
    folder = case_dir / GRID_FOLDER
    if not folder.is_dir():
        return {}
    found = {}
    for path in folder.glob("grid-*.dat"):
        number = path.stem[5:]
        if number.isdigit():
            found[int(number)] = path
    return found


def check_case(case_dir: str | Path) -> InputReport:
    """Check the input files of a case folder (see the module docstring)."""
    from lagoonviz.core.data.setup import read_setup

    case_dir = Path(case_dir).expanduser()
    report = InputReport(case_dir)
    add = report.problems.append
    has_ctrl, has_control = (case_dir / CTRL).is_file(), (case_dir / CONTROL).is_file()
    if not has_ctrl:
        add(Finding("error", CTRL, "missing: REEF3D needs it"))
    if not has_control:
        add(Finding("error", CONTROL, "missing: DIVEMesh needs it to make the grid"))
    if not (has_ctrl or has_control):
        return report
    setup = read_setup(case_dir)
    report.solver = setup.solver
    ctrl, control = setup.ctrl, setup.control

    # each file on its own: what the catalog knows
    for name, file, catalog in (
        (CTRL, ctrl, Catalog.load("reef3d")),
        (CONTROL, control, Catalog.load("divemesh")),
    ):
        if not file:
            continue
        for problem in catalog.check(file, report.solver if name == CTRL else ""):
            where = f"{name}:{problem.line}" if problem.line else name
            add(Finding(problem.level, where, problem.message))

    # the processes: the same in both files, and DIVEMesh's grid for each
    ranks_reef3d = ctrl.int("M 10", 1) if has_ctrl else None
    ranks_divemesh = control.int("M 10", 1) if has_control else None
    if ranks_reef3d is not None and ranks_divemesh is not None and ranks_reef3d != ranks_divemesh:
        add(
            Finding(
                "error",
                "M 10",
                f"ctrl.txt has M 10 {ranks_reef3d}, control.txt M 10 {ranks_divemesh}: REEF3D "
                "stops (the same number of processes in both, and for mpirun -np)",
            )
        )
    ranks = ranks_reef3d or ranks_divemesh or 1
    grids = _grid_files(case_dir)
    if has_control and not grids:
        add(Finding("error", GRID_FOLDER, "no grid: run DIVEMesh first"))
    elif grids:
        missing = [rank for rank in range(1, ranks + 1) if rank not in grids]
        if missing:
            add(
                Finding(
                    "error",
                    GRID_FOLDER,
                    f"no grid-{missing[0]:06d}.dat"
                    + (f" (and {len(missing) - 1} more)" if len(missing) > 1 else "")
                    + f": the grid was made for fewer processes than M 10 {ranks}; run "
                    "DIVEMesh again",
                )
            )
        extra = [rank for rank in grids if rank > ranks]
        if extra and not missing:
            add(
                Finding(
                    "warning",
                    GRID_FOLDER,
                    f"{len(extra)} grid file(s) beyond M 10 {ranks} (from an earlier "
                    "decomposition): unused",
                )
            )
        used = [grids[rank] for rank in range(1, ranks + 1) if rank in grids]
        oldest = min(g.stat().st_mtime for g in used) if used and not missing else 0.0
        for name in (CONTROL, "geo.dat", "solid.stl"):
            path = case_dir / name
            if oldest and path.is_file() and path.stat().st_mtime > oldest + 1.0:
                add(
                    Finding(
                        "error" if name == CONTROL else "warning",
                        name,
                        "changed after DIVEMesh made the grid: run DIVEMesh again",
                    )
                )

    if has_ctrl:
        _output_and_end(report, ctrl)
    _cost(report, setup, ranks)
    order = {level: k for k, level in enumerate(LEVELS)}
    report.problems.sort(key=lambda p: order.get(p.level, 9))
    return report


def _output_and_end(report: InputReport, ctrl) -> None:
    add = report.problems.append
    every = ctrl.value("P 20")
    interval = ctrl.value("P 30")
    store = ctrl.int("P 18", 0)
    if not (every and every > 0) and not (interval and interval > 0):
        add(
            Finding(
                "warning",
                CTRL,
                "no 3D output: neither P 20 (every n iterations) nor P 30 (every t seconds)",
            )
        )
    if store == 0 and ctrl.int("P 10", 1) == 0:
        add(Finding("warning", CTRL, "P 10 0 and no P 18: the run writes no 3D files"))
    end_time = ctrl.value("N 41")
    max_steps = ctrl.value("N 45")
    if not (end_time and 0 < end_time < 1e8) and not (max_steps and 0 < max_steps < 1e8):
        add(
            Finding(
                "warning",
                CTRL,
                "no end: neither N 41 (simulated time) nor N 45 (time steps) is set",
            )
        )
    if store == 0:
        add(
            Finding(
                "note",
                "P 18",
                "the LAGOON store is off: P 18 1 writes it instead of the VTK files (smaller, "
                "faster to open)",
            )
        )


def _cost(report: InputReport, setup, ranks: int) -> None:
    """The grid's size, per process, and the time per iteration from the last run
    in the folder (when it had the same grid)."""
    facts = report.facts
    if setup.cells is None:
        return
    nx, ny, nz = setup.cells
    if report.solver == "SFLOW":
        nz = 1
    cells = nx * ny * nz
    facts.append(("Grid", f"{nx} x {ny} x {nz} = {cells:,} cells"))
    per_rank = cells / max(ranks, 1)
    facts.append(("Per process", f"{per_rank:,.0f} cells on each of {ranks}"))
    if ranks > 1 and per_rank < FEW_CELLS:
        report.problems.append(
            Finding(
                "note",
                "M 10",
                f"only {per_rank:,.0f} cells per process: fewer processes may be as fast",
            )
        )
    if per_rank > MANY_CELLS:
        report.problems.append(
            Finding(
                "note",
                "M 10",
                f"{per_rank:,.0f} cells per process: more processes would make it faster",
            )
        )
    last = _last_run(report.case_dir)
    if last is None:
        return
    seconds, dt, last_cells, last_ranks = last
    if not last_cells:
        return
    scaled = seconds * (cells / last_cells) * (last_ranks / max(ranks, 1))
    facts.append(
        (
            "Time per iteration",
            f"about {scaled:.3g} s (the last run here: {seconds:.3g} s with "
            f"{last_cells:,} cells on {last_ranks})",
        )
    )
    end_time = setup.ctrl.value("N 41")
    if last_cells == cells and dt and end_time and 0 < end_time < 1e8:
        facts.append(
            (
                "Run time",
                f"about {duration(end_time / dt * scaled)} for N 41 {end_time:g} s (at the "
                f"last run's time step {dt:.3g} s)",
            )
        )


def _last_run(case_dir: Path) -> tuple[float, float, int, int] | None:
    """(wall seconds per iteration, time step, cells, processes) of the last run
    in the folder, from REEF3D_Log and its run log; None if not known."""
    import numpy as np

    for folder, name in (("REEF3D_Log", "REEF3D_mainlog.dat"),):
        path = case_dir / folder / name
        tail = read_tail(path)
        if tail is None or tail.columns.shape[1] < 3:
            continue
        cells = 0
        try:
            with path.open(errors="replace") as stream:
                for _ in range(12):
                    line = stream.readline()
                    if line.startswith("number of cells:"):
                        cells = int(float(line.split(":")[1]))
                        break
        except (OSError, ValueError):
            pass
        ranks = _last_ranks(case_dir) or 1
        seconds = float(np.median(tail.columns[-200:, 2]))
        dt = float(np.median(tail.columns[-200:, 0]))
        return seconds, dt, cells, ranks
    return None


def _last_ranks(case_dir: Path) -> int | None:
    runs = sorted((case_dir / "REEF3D_Case").glob("REEF3D_*_run.jsonl"))
    latest = None
    for path in runs:
        try:
            for line in path.read_text(errors="replace").splitlines():
                if '"type":"run"' not in line.replace(" ", ""):
                    continue
                record = json.loads(line)
                if latest is None or record.get("started", "") > latest.get("started", ""):
                    latest = record
        except (OSError, ValueError):
            continue
    if latest is None:
        return None
    try:
        return int(latest.get("ranks"))
    except (TypeError, ValueError):
        return None
