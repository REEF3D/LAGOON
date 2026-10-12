"""Reference reader for the REEF3D run log (``REEF3D_Case/REEF3D_<SOLVER>_run.jsonl``).

See ``RUN.md`` for the specification. Standard library only, MIT-licensed.

    python -m lagoon_format.run path/to/case      # summary and checks
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

FORMAT = "reef3d-run"
VERSION = 1
PATTERN = "REEF3D_*_run.jsonl"
FOLDER = "REEF3D_Case"


@dataclass
class Output:
    step: int
    time: float
    iteration: int | None
    streams: list[str]
    files: dict[str, str] = field(default_factory=dict)


@dataclass
class Run:
    """One run: its start record, its outputs and how it ended (if it did)."""

    id: str
    solver: str
    record: dict[str, Any]
    outputs: list[Output] = field(default_factory=list)
    end: dict[str, Any] | None = None

    @property
    def streams(self) -> dict[str, dict[str, Any]]:
        return self.record.get("streams", {})

    @property
    def georef(self):
        """The case's georeference the run was started with (a
        :class:`lagoon_format.geo.Georef`), or None."""
        from lagoon_format.geo import from_attributes

        return from_attributes(self.record)

    @property
    def started(self) -> str:
        return str(self.record.get("started", ""))

    @property
    def status(self) -> str:
        return str(self.end.get("status", "ended")) if self.end else "running or killed"

    def file_name(self, stream: str, output: Output) -> str | None:
        """File name of a frame stream at an output (``files`` override, else pattern)."""
        if stream in output.files:
            return output.files[stream]
        pattern = self.streams.get(stream, {}).get("files")
        if not pattern:
            return None
        return pattern.format(step=output.step, iteration=output.iteration or 0)

    def frames(self, stream: str) -> list[tuple[Output, Path]]:
        """(output, path relative to the case folder) of every frame of a stream."""
        folder = Path(self.streams.get(stream, {}).get("folder", ""))
        found = []
        for output in self.outputs:
            if stream in output.streams:
                name = self.file_name(stream, output)
                if name:
                    found.append((output, folder / name))
        return found


@dataclass
class RunLog:
    """All runs of a case folder, oldest first, with the problems found reading."""

    folder: Path
    runs: list[Run] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def run(self, run_id: str) -> Run | None:
        return next((r for r in self.runs if r.id == run_id), None)

    def owners(self) -> dict[Path, tuple[Run, Output, str]]:
        """Which run wrote each frame file that exists now: the last one listing it
        (a later run overwrites the files of an earlier one)."""
        owner: dict[Path, tuple[Run, Output, str]] = {}
        for run in sorted(self.runs, key=lambda r: r.started):
            for stream in run.streams:
                for output, path in run.frames(stream):
                    owner[path] = (run, output, stream)
        return owner

    def surviving(self, run: Run, stream: str) -> list[tuple[Output, Path]]:
        """Frames of a run's stream that are still on disk and still its own."""
        owners = self.owners()
        return [
            (output, path)
            for output, path in run.frames(stream)
            if owners.get(path, (None,))[0] is run and (self.folder / path).exists()
        ]


def read_log(path: str | Path, log: RunLog | None = None) -> RunLog:
    """Read one run log file (add to ``log`` if given)."""
    path = Path(path)
    case = path.parent.parent if path.parent.name == FOLDER else path.parent
    log = log or RunLog(case)
    runs = {r.id: r for r in log.runs}
    text = path.read_text(encoding="utf-8", errors="replace")
    lines = text.split("\n")
    if lines and lines[-1] != "":
        log.warnings.append(f"{path.name}: last line incomplete (run killed while writing?)")
    for number, line in enumerate(lines[:-1], start=1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as error:
            log.warnings.append(f"{path.name}:{number}: not JSON ({error.msg})")
            continue
        kind = record.get("type")
        if kind == "run":
            if record.get("format") != FORMAT:
                log.warnings.append(f"{path.name}:{number}: format {record.get('format')!r}")
            if int(record.get("version", 0)) > VERSION:
                log.warnings.append(
                    f"{path.name}:{number}: version {record.get('version')} is newer than "
                    f"{VERSION}; reading what is known"
                )
            run = Run(str(record.get("run", "")), str(record.get("solver", "")), record)
            if run.id in runs:
                log.warnings.append(f"{path.name}:{number}: run {run.id} started twice")
            runs[run.id] = run
            log.runs.append(run)
        elif kind == "stream":
            run = runs.get(str(record.get("run", "")))
            if run is None:
                log.warnings.append(f"{path.name}:{number}: stream of an unknown run")
                continue
            run.record.setdefault("streams", {})[str(record.get("name", ""))] = record.get(
                "stream", {}
            )
        elif kind == "output":
            run = runs.get(str(record.get("run", "")))
            if run is None:
                log.warnings.append(f"{path.name}:{number}: output of an unknown run")
                continue
            iteration = record.get("iteration")
            run.outputs.append(
                Output(
                    int(record["step"]),
                    float(record["time"]),
                    int(iteration) if iteration is not None else None,
                    [str(s) for s in record.get("streams", [])],
                    {str(k): str(v) for k, v in record.get("files", {}).items()},
                )
            )
        elif kind == "end":
            run = runs.get(str(record.get("run", "")))
            if run is not None:
                run.end = record
        # other types: ignored (newer writers)
    return log


def read_case(folder: str | Path) -> RunLog | None:
    """All run logs of a case folder, or None if it has none."""
    folder = Path(folder)
    paths = sorted((folder / FOLDER).glob(PATTERN)) + sorted(folder.glob(PATTERN))
    if not paths:
        return None
    log = RunLog(folder)
    for path in paths:
        read_log(path, log)
    log.runs.sort(key=lambda r: r.started)
    return log


def check(log: RunLog) -> list[str]:
    """Problems a writer should fix: times going back, unknown streams, missing files."""
    problems = list(log.warnings)
    owners = log.owners()
    for run in log.runs:
        # per stream: a store counts its volume output only at the next one (RUN.md)
        for stream in {s for o in run.outputs for s in o.streams}:
            times = [o.time for o in run.outputs if stream in o.streams]
            if any(b < a for a, b in zip(times, times[1:], strict=False)):
                problems.append(f"run {run.id}: output times of {stream} go backwards")
        for output in run.outputs:
            for stream in output.streams:
                if stream not in run.streams:
                    problems.append(f"run {run.id} step {output.step}: unknown stream {stream!r}")
        for stream in run.streams:
            for _output, path in run.frames(stream):
                if owners.get(path, (None,))[0] is run and not (log.folder / path).exists():
                    problems.append(f"run {run.id}: missing {path}")
                    break  # one per stream is enough
    return problems


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    folder = Path(argv[0] if argv else ".")
    log = read_case(folder)
    if log is None:
        print(f"no {FOLDER}/{PATTERN} in {folder}")
        return 1
    for run in log.runs:
        span = (
            f"t = {min(o.time for o in run.outputs):g} .. {max(o.time for o in run.outputs):g} s"
            if run.outputs
            else "no outputs"
        )
        print(f"{run.id}  {run.solver}  {len(run.outputs)} outputs, {span}, {run.status}")
        for stream, spec in run.streams.items():
            kept = len(log.surviving(run, stream)) if "files" in spec else "-"
            print(
                f"    {stream:10s} {spec.get('role', '?'):13s} {spec.get('folder', '')}  "
                f"frames on disk: {kept}"
            )
    problems = check(log)
    for problem in problems:
        print("problem:", problem)
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
