# SPDX-License-Identifier: GPL-3.0-or-later
# Architect: Hans Bihs
"""How a REEF3D run is going: read from its run log (REEF3D_Case/REEF3D_<SOLVER>_run.jsonl)
and its log files (REEF3D_Log/REEF3D_mainlog.dat, REEF3D_maxlog.dat; SFLOW:
REEF3D_SFLOW_Log/REEF3D_SFLOW_mainlog.dat). NumPy only, no Qt: the Run monitor panel
asks for :func:`run_status` every few seconds.

    status = run_status(case_dir)
    status.iteration, status.time, status.dt, status.eta, status.warnings

The main log has one line per iteration (P 12): iteration, time step, simulation
time, wall time of the iteration, then solver-specific columns; the max log the
largest velocities. Only the tail of a long log is read.
"""

from __future__ import annotations

import difflib
import math
import time as clock
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np

#: how much of a log's end is read (bytes): some thousand iterations
TAIL = 400_000
#: iterations over which speed and the usual time step are taken
WINDOW = 50

SOLVER_LOGS = {
    "SFLOW": ("REEF3D_SFLOW_Log", "REEF3D_SFLOW_mainlog.dat", "REEF3D_SFLOW_maxlog.dat"),
}
DEFAULT_LOGS = ("REEF3D_Log", "REEF3D_mainlog.dat", "REEF3D_maxlog.dat")


@dataclass
class LogTail:
    """The columns of the last lines of a log: iteration, then the others."""

    iteration: np.ndarray
    columns: np.ndarray  # (n, k): the columns after the iteration
    modified: float  # the file's modification time (epoch seconds)


@dataclass
class RunStatus:
    """What the Run monitor shows about the latest run of a case."""

    case_dir: Path
    run_id: str = ""
    solver: str = ""
    started: datetime | None = None
    #: "running", "finished", "stopped" (ended otherwise), "not responding" (no end
    #: written, the log not changed for long: killed, or hanging), "no run"
    state: str = "no run"
    ended: str = ""  # the end record's status, as REEF3D wrote it
    iteration: int | None = None
    time: float | None = None  # simulation time [s]
    dt: float | None = None
    end_time: float | None = None  # N 41 of the control file the run started with
    max_iterations: int | None = None  # N 45
    seconds_per_iteration: float | None = None  # wall time, the last WINDOW iterations
    simulated_per_hour: float | None = None  # simulation seconds per wall hour
    elapsed: float | None = None  # wall seconds since the start
    eta: float | None = None  # wall seconds to the end (N 41 or N 45, whichever first)
    log_age: float | None = None  # seconds since the main log was written
    outputs: int = 0
    last_output_time: float | None = None
    max_velocity: float | None = None  # the largest of umax, vmax, wmax, now
    #: the time step and the max velocity over the last iterations, for a small plot
    history: dict[str, np.ndarray] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    #: ctrl.txt changed since the run started: the differing lines (unified diff)
    ctrl_changes: list[str] = field(default_factory=list)

    @property
    def progress(self) -> float | None:
        """0..1 of the way to the end: the nearer of N 41 (time) and N 45 (iterations)."""
        parts = []
        if self.time is not None and _set(self.end_time):
            parts.append(self.time / self.end_time)
        if self.iteration is not None and _set(self.max_iterations):
            parts.append(self.iteration / self.max_iterations)
        return max(0.0, min(1.0, max(parts))) if parts else None


def _set(limit: float | None) -> bool:
    """Whether an end (N 41, N 45) was set, not left at a huge default."""
    return limit is not None and 0 < limit < 1e8


# ----------------------------------------------------------------- reading
def read_tail(path: Path, tail: int = TAIL) -> LogTail | None:
    """The numeric lines of the end of a log (iteration first); None if none."""
    try:
        modified = path.stat().st_mtime
        with path.open("rb") as stream:
            size = stream.seek(0, 2)
            stream.seek(max(0, size - tail))
            text = stream.read().decode("utf-8", errors="replace")
    except OSError:
        return None
    lines = text.splitlines()
    if len(text) >= tail and lines:
        lines = lines[1:]  # the first line read is likely cut
    rows = []
    width = None
    for line in lines:
        fields = line.replace("|", " ").split()
        if not fields or not _integer(fields[0]):
            continue
        try:
            values = [float(f) for f in fields]
        except ValueError:
            continue
        if width is None:
            width = len(values)
        if len(values) < width:
            continue  # being written
        rows.append(values[:width])
    if not rows:
        return None
    table = np.asarray(rows, np.float64)
    return LogTail(table[:, 0].astype(np.int64), table[:, 1:], modified)


def _integer(token: str) -> bool:
    return token.lstrip("-").isdigit()


def read_ctrl_value(text: str, key: str, index: int = 0) -> float | None:
    """A value of a ctrl.txt line (``key`` as "N 41"), the last line naming it."""
    found = None
    letter, number = key.split()
    for line in text.splitlines():
        words = line.split("//")[0].split()
        if len(words) > 2 + index and words[0] == letter and words[1] == number:
            try:
                found = float(words[2 + index])
            except ValueError:
                pass
    return found


def ctrl_changes(current: str, started: str) -> list[str]:
    """The lines of ctrl.txt that differ from the copy the run started with
    (comments and blank lines left out), as a unified diff without its header."""

    def lines(text: str) -> list[str]:
        out = []
        for line in text.splitlines():
            words = line.split("//")[0].split()
            if words:
                out.append(" ".join(words))
        return out

    diff = difflib.unified_diff(lines(started), lines(current), lineterm="", n=0)
    return [line for line in diff if line[:1] in "+-" and line[:3] not in ("---", "+++")]


# ----------------------------------------------------------------- the status
def _latest_run(case_dir: Path, solver: str = ""):
    from lagoon_format.run import read_case

    try:
        history = read_case(case_dir)
    except (OSError, ValueError):
        return None
    if history is None or not history.runs:
        return None
    runs = [r for r in history.runs if not solver or r.solver.upper() == solver.upper()]
    return max(runs or history.runs, key=lambda r: r.started)


def _when(text: str) -> datetime | None:
    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None


def run_status(case_dir: str | Path, solver: str = "", now: float | None = None) -> RunStatus:
    """The latest run of a case folder (of ``solver``, if given) and how it is going."""
    case_dir = Path(case_dir)
    now = clock.time() if now is None else now
    status = RunStatus(case_dir)
    run = _latest_run(case_dir, solver)
    started_ctrl = ""
    if run is not None:
        status.run_id, status.solver = run.id, run.solver.upper()
        status.started = _when(run.started)
        status.outputs = len({o.step for o in run.outputs})
        if run.outputs:
            status.last_output_time = max(o.time for o in run.outputs)
        ctrl = run.record.get("ctrl")
        copy = ctrl.get("copy") if isinstance(ctrl, dict) else None
        if copy and (case_dir / copy).is_file():
            try:
                started_ctrl = (case_dir / copy).read_text(errors="replace")
            except OSError:
                started_ctrl = ""
    try:
        current_ctrl = (case_dir / "ctrl.txt").read_text(errors="replace")
    except OSError:
        current_ctrl = ""
    ctrl_text = started_ctrl or current_ctrl
    status.end_time = read_ctrl_value(ctrl_text, "N 41")
    iterations = read_ctrl_value(ctrl_text, "N 45")
    status.max_iterations = int(iterations) if iterations is not None else None
    if started_ctrl and current_ctrl:
        status.ctrl_changes = ctrl_changes(current_ctrl, started_ctrl)

    folder, main_name, max_name = SOLVER_LOGS.get(status.solver, DEFAULT_LOGS)
    main = read_tail(case_dir / folder / main_name)
    if main is not None and main.columns.shape[1] >= 3:
        dt, simtime, itertime = main.columns[:, 0], main.columns[:, 1], main.columns[:, 2]
        status.iteration = int(main.iteration[-1])
        status.time = float(simtime[-1])
        status.dt = float(dt[-1])
        status.log_age = max(0.0, now - main.modified)
        recent = slice(-WINDOW, None)
        per_iteration = float(np.mean(itertime[recent]))
        if per_iteration > 0 and np.isfinite(per_iteration):
            status.seconds_per_iteration = per_iteration
            rate = float(np.mean(dt[recent])) / per_iteration  # simulation s per wall s
            status.simulated_per_hour = rate * 3600.0
            remaining = []
            if _set(status.end_time) and rate > 0:
                remaining.append(max(0.0, status.end_time - status.time) / rate)
            if _set(status.max_iterations):
                remaining.append(max(0, status.max_iterations - status.iteration) * per_iteration)
            status.eta = min(remaining) if remaining else None
        status.history = {"iteration": main.iteration, "dt": dt, "itertime": itertime}
        _time_step_warnings(status, dt, simtime)
    if status.started is not None:
        status.elapsed = max(0.0, now - status.started.timestamp())
    maxima = read_tail(case_dir / folder / max_name, tail=TAIL // 4)
    if maxima is not None and maxima.columns.shape[1] >= 3:
        speeds = np.abs(maxima.columns[:, :3]).max(axis=1)
        status.max_velocity = float(speeds[-1])
        status.history["velocity_iteration"] = maxima.iteration
        status.history["velocity"] = speeds
        _velocity_warnings(status, speeds)

    status.state, status.ended = _state(run, status)
    if status.state != "running":
        status.eta = None
    if status.ctrl_changes:
        status.warnings.append(
            "ctrl.txt was changed after the run started: the run goes on with the copy in "
            "REEF3D_Case (see the changes below)"
        )
    return status


def _state(run, status: RunStatus) -> tuple[str, str]:
    if run is None and status.iteration is None:
        return "no run", ""
    if run is not None and run.end:
        ended = str(run.end.get("status", "ended"))
        return ("finished" if ended == "finished" else "stopped"), ended
    # no end record: going on, unless the log has been quiet for much longer than an
    # iteration takes (a run killed, or hanging)
    quiet = status.log_age if status.log_age is not None else 0.0
    usual = status.seconds_per_iteration or 1.0
    if quiet > max(300.0, 30.0 * usual):
        status.warnings.append(
            f"no iteration for {_duration(quiet)} (an iteration takes {usual:.3g} s): "
            "the run was killed, crashed or hangs"
        )
        return "not responding", ""
    return "running", ""


def _time_step_warnings(status: RunStatus, dt: np.ndarray, simtime: np.ndarray) -> None:
    if not np.all(np.isfinite(dt[-WINDOW:])) or not np.all(np.isfinite(simtime[-WINDOW:])):
        status.warnings.append("the time step or the simulation time is not a number (NaN)")
        return
    if dt.size > 2 * WINDOW:
        usual = float(np.median(dt[:-WINDOW]))
        now = float(np.mean(dt[-10:]))
        if usual > 0 and now < 0.1 * usual:
            status.warnings.append(
                f"the time step dropped to {now:.3g} s, a tenth of the usual {usual:.3g} s: "
                "the flow may be blowing up"
            )


def _velocity_warnings(status: RunStatus, speeds: np.ndarray) -> None:
    if not np.all(np.isfinite(speeds[-10:])):
        status.warnings.append("the largest velocity is not a number (NaN)")
        return
    if speeds.size > 2 * WINDOW:
        usual = float(np.median(speeds[:-WINDOW]))
        now = float(speeds[-1])
        if usual > 0 and now > 5.0 * usual:
            status.warnings.append(
                f"the largest velocity rose to {now:.3g} m/s, five times the usual {usual:.3g} m/s"
            )


def _duration(seconds: float) -> str:
    """A wall time as a person reads it: 45 s, 12 min, 3 h 20 min, 2 d 4 h."""
    if seconds is None or not math.isfinite(seconds):
        return "–"
    seconds = max(0.0, seconds)
    if seconds < 90:
        return f"{seconds:.0f} s"
    minutes = seconds / 60.0
    if minutes < 90:
        return f"{minutes:.0f} min"
    hours, minutes = divmod(int(round(minutes)), 60)
    if hours < 48:
        return f"{hours} h {minutes:02d} min"
    days, hours = divmod(hours, 24)
    return f"{days} d {hours} h"


duration = _duration


def finish_clock(eta: float | None, now: float | None = None) -> str:
    """When a run will end, as a local clock time (with the day if not today)."""
    if eta is None or not math.isfinite(eta):
        return "–"
    now = clock.time() if now is None else now
    end = datetime.fromtimestamp(now + eta)
    today = datetime.fromtimestamp(now)
    return end.strftime("%H:%M") if end.date() == today.date() else end.strftime("%a %d %b %H:%M")
