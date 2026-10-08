# SPDX-License-Identifier: GPL-3.0-or-later
# Architect: Hans Bihs
"""Running DIVEMesh, REEF3D's grid generator, on a case folder.

DIVEMesh reads ``control.txt`` (and ``geo.dat``, ``solid.stl``) in the folder it is
started in and writes the grid REEF3D reads (``DIVEMesh_Grid``), a preview of it,
the solids, the bed and the MPI partition (``DIVEMesh_Paraview``) and its logs.
LAGOON shows those as the set-up's grid, solids and bed.

The program is found with :func:`find_divemesh`: ``LAGOON_DIVEMESH``, then
``DiveMESH`` on the PATH, then the usual places of a source build
(``~/Codelite/DIVEMesh/bin/DiveMESH``)::

    from lagoonviz.core.divemesh_run import run_divemesh
    code = run_divemesh("~/runs/flume", on_line=print)
"""

from __future__ import annotations

import os
import shutil
import subprocess
import time
from collections.abc import Callable
from pathlib import Path

#: names the DIVEMesh program goes by
NAMES = ("DiveMESH", "DIVEMesh", "divemesh")
#: where a source build puts it, relative to the home folder
BUILDS = (
    "Codelite/DIVEMesh/bin",
    "Codelite/DIVEMesh/Release",
    "DIVEMesh/bin",
    "DIVEMesh",
    "bin",
)
#: the output of the last run, in the case folder
LOG = Path("DIVEMesh_Log") / "DIVEMesh-run.log"


def find_divemesh(preferred: str | Path | None = None) -> Path | None:
    """The DIVEMesh program: ``preferred`` (a setting), ``LAGOON_DIVEMESH``, the PATH,
    then a source build in the home folder (None: not found)."""
    for given in (preferred, os.environ.get("LAGOON_DIVEMESH")):
        if given:
            path = Path(str(given)).expanduser()
            if path.is_file() and os.access(path, os.X_OK):
                return path
    for name in NAMES:
        found = shutil.which(name)
        if found:
            return Path(found)
    home = Path.home()
    for folder in BUILDS:
        for name in NAMES:
            path = home / folder / name
            if path.is_file() and os.access(path, os.X_OK):
                return path
    return None


def divemesh_command(executable: str | Path) -> list[str]:
    """The command line that runs DIVEMesh (it takes no arguments: it reads
    control.txt in the folder it is started in)."""
    return [str(Path(executable).expanduser())]


def run_divemesh(
    case_dir: str | Path,
    executable: str | Path | None = None,
    on_line: Callable[[str], None] | None = None,
    timeout: float | None = None,
) -> int:
    """Run DIVEMesh in a case folder and wait for it; each line it prints goes to
    ``on_line`` and to ``DIVEMesh_Log/DIVEMesh-run.log``. Returns its exit code."""
    case_dir = Path(case_dir).expanduser()
    if not (case_dir / "control.txt").is_file():
        raise FileNotFoundError(f"{case_dir} has no control.txt")
    program = find_divemesh(executable)
    if program is None:
        raise FileNotFoundError(
            "DIVEMesh was not found: set LAGOON_DIVEMESH to the program, or put DiveMESH "
            "on the PATH"
        )
    log = case_dir / LOG
    log.parent.mkdir(parents=True, exist_ok=True)
    start = time.monotonic()
    with log.open("w", encoding="utf-8") as out:
        process = subprocess.Popen(
            divemesh_command(program),
            cwd=case_dir,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            errors="replace",
        )
        assert process.stdout is not None
        for line in process.stdout:
            out.write(line)
            if on_line is not None:
                on_line(line.rstrip("\n"))
            if timeout is not None and time.monotonic() - start > timeout:
                process.kill()
                break
        code = process.wait()
    return code
