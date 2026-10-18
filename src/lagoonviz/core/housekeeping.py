# SPDX-License-Identifier: GPL-3.0-or-later
# Architect: Hans Bihs
"""What in a case folder belongs to which run, and putting away what is not the
run shown (no Qt).

A folder collects the output of every run started in it: REEF3D overwrites only
the files it writes again, so a shorter run leaves the later outputs of the one
before; another solver's run leaves its own folders; a new LAGOON store renames
the one before to ``*.lagoon-earlier``. :func:`inventory` sorts them into groups
(each with its files and size); :func:`archive` moves groups into
``LAGOON_archive/<date>/`` (keeping their paths, to be put back by hand),
:func:`remove` deletes them.

    groups = inventory(case_dir)
    archive([g for g in groups if g.suggested], case_dir)

The run shown (what the case opens with, lagoonviz.core.data.reef3d.Case.main_datasets)
and DIVEMesh's grid and input files are never offered.
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

ARCHIVE = "LAGOON_archive"


@dataclass
class Group:
    """Files that go together: one dataset of a run, or a whole store."""

    label: str
    kind: str  # "earlier run", "other run", "earlier store", "other store"
    paths: list[Path] = field(default_factory=list)
    run: str = ""  # the run id or the start time, if known
    #: offered ticked: output left over from earlier runs (not another solver's run,
    #: which may still be wanted)
    suggested: bool = False

    @property
    def size(self) -> int:
        total = 0
        for path in self.paths:
            total += _size(path)
        return total


def _size(path: Path) -> int:
    try:
        if path.is_dir():
            return sum(p.stat().st_size for p in path.rglob("*") if p.is_file())
        return path.stat().st_size
    except OSError:
        return 0


def human_size(size: int) -> str:
    value = float(size)
    for unit in ("B", "kB", "MB", "GB", "TB"):
        if value < 1000 or unit == "TB":
            return f"{value:.0f} {unit}" if unit == "B" else f"{value:.1f} {unit}"
        value /= 1000.0
    return f"{value:.1f} TB"


def _files(dataset) -> list[Path]:
    """Every file of a dataset's frames: the index files and their pieces."""
    found: dict[Path, None] = {}
    for frame in dataset.frames:
        for path in (frame.path, *frame.pieces):
            if isinstance(path, Path) and path.suffix and path.exists():
                found[path] = None
    return list(found)


def _stored(dataset) -> bool:
    return getattr(dataset, "store", None) is not None or _in_store(Path(dataset.folder))


def _in_store(path: Path) -> bool:
    return any(part.endswith(".lagoon") or ".lagoon-" in part for part in path.parts)


def inventory(case_dir: str | Path, case=None) -> list[Group]:
    """What in a case folder is not the run shown, grouped (see the module docstring).
    ``case``: the case as opened already (a lagoonviz Case), else it is opened."""
    from lagoonviz.core.data.lagoon import find_stores
    from lagoonviz.core.data.reef3d import open_case

    case_dir = Path(case_dir).resolve()
    if case is None:
        case = open_case(case_dir)
    main = list(case.main_datasets().values())
    shown_solvers = {d.solver for d in main}
    leads = [d for d in main if d.role not in ("solid", "topo", "geodat", "forcing")]
    groups: list[Group] = []
    # the files: as REEF3D wrote them (a store's output stands for its VTK files when
    # the case opens, so these would be missed)
    files_case = (
        open_case(case_dir, stores=False)
        if any(_stored(d) for d in main) or any(_stored(d) for d in case.datasets.values())
        else case
    )
    for dataset in files_case.datasets.values():
        if dataset.role in ("solid", "topo", "geodat", "forcing"):
            continue  # the case's own: DIVEMesh, geo.dat, forcing solids
        if dataset in main or any(dataset.same_run_as(lead) for lead in leads):
            continue
        files = [p for p in _files(dataset) if not _in_store(p)]
        if not files:
            continue
        other_solver = dataset.solver not in shown_solvers
        when = dataset.run_started[:16].replace("T", " ") if dataset.run_started else ""
        steps = f"outputs {dataset.steps[0]}–{dataset.last_step}"
        if dataset.earlier_run or not other_solver:
            kind = "earlier run"
        else:
            kind = "other run"
        groups.append(
            Group(
                label=f"{dataset.solver} {dataset.role.replace('_', ' ')}, {steps}"
                + (f" (run of {when})" if when else ""),
                kind=kind,
                paths=files,
                run=dataset.run_id or when,
                suggested=kind == "earlier run",
            )
        )
    current_stores = {
        Path(getattr(getattr(d, "store", None), "path", "")).resolve()
        for d in main
        if getattr(d, "store", None) is not None
    }
    for store in sorted(case_dir.glob("*.lagoon-earlier")):
        groups.append(
            Group(f"{store.name}", "earlier store", [store], suggested=True, run=store.name)
        )
    for store in find_stores(case_dir):
        if store.resolve() in current_stores:
            continue
        solver = store.stem.removeprefix("REEF3D_")
        if solver in shown_solvers:
            continue  # the run shown may read it
        groups.append(Group(f"{store.name} ({solver})", "other store", [store], run=store.name))
    return groups


def archive(groups: list[Group], case_dir: str | Path, when: datetime | None = None) -> Path:
    """Move the groups' files into ``LAGOON_archive/<date time>/`` of the case folder,
    keeping their paths there; the archive folder."""
    case_dir = Path(case_dir).resolve()
    stamp = (when or datetime.now()).strftime("%Y-%m-%d_%H%M%S")
    target = case_dir / ARCHIVE / stamp
    for group in groups:
        for path in group.paths:
            path = Path(path)
            if not path.exists():
                continue
            relative = path.resolve().relative_to(case_dir)
            destination = target / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(path), str(destination))
    _drop_empty_folders(groups, case_dir)
    return target


def remove(groups: list[Group], case_dir: str | Path) -> int:
    """Delete the groups' files (cannot be undone); the bytes freed."""
    case_dir = Path(case_dir).resolve()
    freed = 0
    for group in groups:
        for path in group.paths:
            path = Path(path)
            if not path.exists():
                continue
            if case_dir not in path.resolve().parents:
                raise ValueError(f"{path} is not in the case folder {case_dir}")
            freed += _size(path)
            if path.is_dir():
                shutil.rmtree(path)
            else:
                path.unlink()
    _drop_empty_folders(groups, case_dir)
    return freed


def _drop_empty_folders(groups: list[Group], case_dir: Path) -> None:
    """Output folders emptied by putting their files away go too."""
    folders = {Path(p).parent for g in groups for p in g.paths}
    for folder in sorted(folders, key=lambda f: len(f.parts), reverse=True):
        try:
            if folder != case_dir and folder.is_dir() and not any(folder.iterdir()):
                folder.rmdir()
        except OSError:
            pass
