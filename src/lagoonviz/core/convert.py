# SPDX-License-Identifier: GPL-3.0-or-later
# Architect: Hans Bihs
"""REEF3D VTK output to a LAGOON store (lagoon-format SPEC.md, section 6).

The σ-grid volumes of FNPF and NHFLOW, CFD's Cartesian volume and the free surface
and bed surfaces of any solver are converted; each MPI piece becomes a block of the
store. Floating bodies keep their mesh once and a rigid motion per output; time
series and wave profiles become tables. Converting a case again appends the
outputs written since.

Command line::

    lagoon-convert CASE [--out STORE] [--outputs volume,free_surface,bed,bodies,...]
"""

from __future__ import annotations

import argparse
import re
import sys
import time as _time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from lagoon_format.bodies import NotRigid
from lagoon_format.store import Store, StoreError, StoreWriter, is_store

from lagoonviz import __version__
from lagoonviz.core.data.mesh import triangulate
from lagoonviz.core.data.reef3d import Case, Dataset, open_case, preferred, run_log
from lagoonviz.core.data.vtkxml import read_piece, read_time_value

#: roles that are converted, and the grid each becomes
ROLES = {"volume": "sigma", "free_surface": "surface", "bed": "surface"}
#: what else is converted (format 0.2): floating bodies (mesh once, a rigid motion
#: per output), time series tables and wave profiles
EXTRAS = ("bodies", "series", "profiles")
SIGMA_SOLVERS = ("FNPF", "NHFLOW")
#: solvers whose volume is converted: σ-grids, and CFD's Cartesian grid
VOLUME_SOLVERS = (*SIGMA_SOLVERS, "CFD")

Progress = Callable[[str, int, int], None]


class ConvertError(ValueError):
    """The output cannot be converted (not a structured grid, another run)."""


# ----------------------------------------------------------------- grid plans
@dataclass
class PiecePlan:
    """Where the points of one VTK piece go in its block, found once.

    ``order`` lists the piece's point indices in block order (level, y, x for
    volumes; y, x for surfaces); -1 marks a grid point the piece does not have.
    """

    start: tuple[int, ...]  # (i0, j0), or (i0, j0, k0) for a Cartesian block
    size: tuple[int, ...]  # (nx_b, ny_b), or (nx_b, ny_b, nz_b)
    order: np.ndarray  # (nz, ny_b, nx_b) or (ny_b, nx_b)
    points: int  # number of points in the piece (to check later frames)

    def take(self, values: np.ndarray) -> np.ndarray:
        """``values`` (per point) arranged on the block, NaN where there is no point."""
        values = np.asarray(values)
        present = self.order >= 0
        picked = values[np.where(present, self.order, 0)].astype(np.float32)
        if not present.all():
            picked[~present] = np.nan
        return picked


def _grid_lines(pieces: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """The distinct x and y of all points, to the millimetre (float32 coordinates of
    different MPI ranks differ in the last digits)."""
    xs = np.unique(np.concatenate([np.round(p[:, 0].astype(np.float64), 3) for p in pieces]))
    ys = np.unique(np.concatenate([np.round(p[:, 1].astype(np.float64), 3) for p in pieces]))
    return xs, ys


def _indices(points: np.ndarray, xs: np.ndarray, ys: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    i = np.searchsorted(xs, np.round(points[:, 0].astype(np.float64), 3))
    j = np.searchsorted(ys, np.round(points[:, 1].astype(np.float64), 3))
    return np.clip(i, 0, len(xs) - 1), np.clip(j, 0, len(ys) - 1)


def plan_volume(points: np.ndarray, xs: np.ndarray, ys: np.ndarray) -> tuple[PiecePlan, int]:
    """The block of a σ-grid piece: its columns and, in each, the points by height."""
    i, j = _indices(points, xs, ys)
    i0, j0 = int(i.min()), int(j.min())
    nx_b, ny_b = int(i.max()) - i0 + 1, int(j.max()) - j0 + 1
    column = (j - j0) * nx_b + (i - i0)
    counts = np.bincount(column, minlength=nx_b * ny_b)
    used = counts[counts > 0]
    nz = int(np.bincount(used).argmax())
    if nz < 2 or np.any(used != nz):
        raise ConvertError("the volume is not made of columns with the same number of levels")
    # by column, then in the file's order: REEF3D writes the levels bottom to top
    # (k outermost). Not by height: near the coast FNPF has columns whose levels
    # are not in rising order, and dry columns have all levels at one height.
    order = np.lexsort((np.arange(len(points)), column))
    plan = np.full((ny_b * nx_b, nz), -1, np.int64)
    filled = np.flatnonzero(counts > 0)
    plan[filled] = order.reshape(-1, nz)
    block = np.ascontiguousarray(plan.reshape(ny_b, nx_b, nz).transpose(2, 0, 1))
    return PiecePlan((i0, j0), (nx_b, ny_b), block, len(points)), nz


def plan_cartesian(points: np.ndarray, xs: np.ndarray, ys: np.ndarray, zs: np.ndarray) -> PiecePlan:
    """The block of a Cartesian piece (CFD), which MPI may split in z as well."""
    i, j = _indices(points, xs, ys)
    k = np.clip(np.searchsorted(zs, np.round(points[:, 2].astype(np.float64), 3)), 0, len(zs) - 1)
    i0, j0, k0 = int(i.min()), int(j.min()), int(k.min())
    nx_b, ny_b, nz_b = int(i.max()) - i0 + 1, int(j.max()) - j0 + 1, int(k.max()) - k0 + 1
    flat = np.full(nz_b * ny_b * nx_b, -1, np.int64)
    where = ((k - k0) * ny_b + (j - j0)) * nx_b + (i - i0)
    flat[where[::-1]] = np.arange(len(points))[::-1]  # the first point wins
    order = flat.reshape(nz_b, ny_b, nx_b)
    return PiecePlan((i0, j0, k0), (nx_b, ny_b, nz_b), order, len(points))


def plan_surface(points: np.ndarray, xs: np.ndarray, ys: np.ndarray) -> PiecePlan:
    """The block of a surface piece: one point per grid column (the first if two)."""
    i, j = _indices(points, xs, ys)
    i0, j0 = int(i.min()), int(j.min())
    nx_b, ny_b = int(i.max()) - i0 + 1, int(j.max()) - j0 + 1
    flat = np.full(ny_b * nx_b, -1, np.int64)
    where = (j - j0) * nx_b + (i - i0)
    flat[where[::-1]] = np.arange(len(points))[::-1]  # the first point wins
    return PiecePlan((i0, j0), (nx_b, ny_b), flat.reshape(ny_b, nx_b), len(points))


def nominal_sigma(columns: list[np.ndarray]) -> np.ndarray:
    """σ of the levels: the median over the wet columns of all blocks (z: levels first)."""
    samples = []
    for z in columns:
        z = z.reshape(z.shape[0], -1).astype(np.float64)
        depth = z[-1] - z[0]
        wet = np.isfinite(depth) & (depth > 1e-6)
        if wet.any():
            samples.append((z[:, wet] - z[0, wet]) / depth[wet])
    if not samples:
        nz = columns[0].shape[0]
        return np.linspace(0.0, 1.0, nz)
    sigma = np.median(np.concatenate(samples, axis=1), axis=1)
    sigma[0], sigma[-1] = 0.0, 1.0
    return sigma


def level_offsets(z: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    """``z_offset`` of a block (SPEC 3): what the σ-levels miss, zero within two
    float32 steps of z."""
    z = z.astype(np.float64)
    bed, top = z[0], z[-1]
    guess = bed[None] + sigma[:, None, None] * (top - bed)[None]
    offset = z - guess
    tolerance = 2.0 * np.spacing(np.abs(z).astype(np.float32)).astype(np.float64)
    offset[~(np.abs(offset) > tolerance)] = 0.0  # NaN (no point) becomes 0 as well
    return offset.astype(np.float32)


# ----------------------------------------------------------------- conversion
@dataclass
class Report:
    store: Path
    outputs: dict[str, int]  # outputs added per output group
    vtk_bytes: int
    seconds: float

    def summary(self) -> str:
        added = ", ".join(f"{name} +{n}" for name, n in self.outputs.items()) or "nothing new"
        size = store_size(self.store)
        ratio = f", {self.vtk_bytes / size:.1f}× smaller than the VTK files" if size else ""
        return (
            f"{self.store.name}: {added} in {self.seconds:.1f} s; "
            f"{size / 1e6:.1f} MB{ratio if self.vtk_bytes else ''}"
        )


def store_size(path: Path) -> int:
    return sum(p.stat().st_size for p in Path(path).rglob("*") if p.is_file())


def chosen_datasets(case: Case, roles=tuple(ROLES)) -> dict[str, Dataset]:
    """The datasets to convert, one per role: those LAGOON shows for the case (of
    several runs in a folder, the latest, unless it is only a short start)."""
    shown = case.main_datasets()
    lead = shown.get("free_surface") or shown.get("bed") or shown.get("volume")
    picked: dict[str, Dataset] = {}
    for role in roles:
        if role in shown:
            picked[role] = shown[role]
            continue
        options = [d for d in case.datasets.values() if d.role == role]
        if lead is not None:
            options = [d for d in options if d.solver == lead.solver]
        if role == "volume":
            options = [
                d for d in options if d.solver in VOLUME_SOLVERS and d.kind == "UnstructuredGrid"
            ]
        if options:
            picked[role] = preferred(options)
    return picked


def _frame_times(case: Case, dataset: Dataset) -> list[float]:
    if dataset.times is not None and len(dataset.times) == len(dataset.frames):
        return [float(t) for t in dataset.times]
    times = []
    for frame in dataset.frames:
        value = read_time_value(frame.pieces[0])
        if value is None and case.print_interval:
            value = frame.step * case.print_interval
        times.append(float(frame.step) if value is None else float(value))
    return times


def _run_record(case: Case, dataset: Dataset) -> dict | None:
    if not dataset.run_id:
        return None
    log = run_log(case.path)
    run = log.run(dataset.run_id) if log is not None else None
    return dict(run.record) if run is not None else {"type": "run", "run": dataset.run_id}


def convert(
    case_path: str | Path,
    out: str | Path | None = None,
    roles: tuple[str, ...] = (*ROLES, *EXTRAS),
    progress: Progress | None = None,
    shard_time: int = 16,
    limit: int | None = None,
) -> Report:
    """Convert (or bring up to date) the LAGOON store of a REEF3D case."""
    began = _time.perf_counter()
    case = open_case(case_path, stores=False)
    datasets = chosen_datasets(case, tuple(r for r in roles if r in ROLES))
    lead = next(iter(datasets.values()), None)
    bodies = None
    if "bodies" in roles:
        bodies = chosen_datasets(case, ("body",)).get("body")
        if bodies is not None and lead is not None and bodies.solver != lead.solver:
            bodies = None
        if lead is None and bodies is not None:
            lead = bodies
    solver = lead.solver if lead is not None else _series_solver(case)
    series = {k: f for k, f in case.series.items() if f.solver == solver and "series" in roles}
    profiles = {
        k: f for k, f in case.profiles.items() if f.solver == solver and "profiles" in roles
    }
    if not datasets and bodies is None and not series and not profiles:
        raise ConvertError(f"{case.path}: no volume, surface, body or time series to convert")
    path = Path(out) if out is not None else case.path / f"REEF3D_{solver}.lagoon"
    run = _run_record(case, lead) if lead is not None else None
    if is_store(path):
        existing = Store(path)
        if run and existing.run_id and existing.run_id != run.get("run"):
            raise ConvertError(
                f"{path.name} holds run {existing.run_id}, the case's outputs are of run "
                f"{run.get('run')}: move it away to convert the new run"
            )
        writer = StoreWriter.open(path)
    else:
        if path.exists() and any(path.iterdir()):
            raise ConvertError(f"{path}: exists and is not a LAGOON store")
        writer = StoreWriter.create(
            path, solver, run=run, created_by=f"lagoon-convert {__version__}", source="REEF3D VTK"
        )
    added: dict[str, int] = {}
    vtk_bytes = 0
    with ThreadPoolExecutor(8) as pool:
        for role, dataset in datasets.items():
            n, size = _convert_dataset(
                case, dataset, role, writer, pool, progress, shard_time, limit
            )
            added[role] = n
            vtk_bytes += size
        if bodies is not None:
            n, size = _convert_bodies(case, bodies, writer, pool, progress, limit)
            added["bodies"] = n
            vtk_bytes += size
    if series:
        added["series"] = _convert_series(series, writer)
    if profiles:
        added["profiles"] = _convert_profiles(profiles, writer, progress)
    writer.close()
    return Report(path, added, vtk_bytes, _time.perf_counter() - began)


def _series_solver(case: Case) -> str:
    solvers = [f.solver for f in case.series.values()] + [p.solver for p in case.profiles.values()]
    return max(set(solvers), key=solvers.count) if solvers else "REEF3D"


def _convert_series(files: dict, writer: StoreWriter) -> int:
    """The time series tables whose files changed since the last conversion."""
    store = Store(writer.path)
    written = 0
    for key, item in files.items():
        try:
            stat = item.path.stat()
        except OSError:
            continue
        old = store.tables.get(key)
        if (
            old is not None
            and old.info.get("source_bytes") == stat.st_size
            and old.info.get("source_mtime") == stat.st_mtime
        ):
            continue  # unchanged
        try:
            table = item.read()
        except (OSError, ValueError):
            continue
        writer.write_table(
            key,
            table.time,
            table.columns,
            name=item.name,
            kind=item.group,
            units=table.units,
            locations=table.locations,
            source=f"{item.path.parent.name}/{item.path.name}",
            extra={"source_bytes": stat.st_size, "source_mtime": stat.st_mtime},
        )
        written += 1
    return written


def _convert_profiles(sets: dict, writer: StoreWriter, progress: Progress | None) -> int:
    """Wave profiles written since the last conversion, appended."""
    from lagoonviz.core.data.profiles import read_profile

    store = Store(writer.path)
    added = 0
    for key, item in sets.items():
        old = store.profiles.get(key)
        last = float(old.times[-1]) if old is not None and len(old.times) else -np.inf
        todo = [(t, p) for t, p in zip(item.times, item.paths, strict=True) if t > last + 1e-9]
        if not todo:
            continue
        if old is None:
            first = read_profile(todo[0][1])
            lines = {
                n: (line.y, line.x, line.theory is not None) for n, line in first.lines.items()
            }
            profile_writer = writer.profile_writer(key, lines, source=item.folder.name)
        else:
            profile_writer = writer.profile_writer(key)
        for start in range(0, len(todo), 64):
            batch = todo[start : start + 64]
            profiles = []
            for _, path in batch:
                profile = read_profile(path)
                profiles.append({n: (ln.eta, ln.theory) for n, ln in profile.lines.items()})
            profile_writer.append([float(t) for t, _ in batch], profiles)
            added += len(batch)
            if progress is not None:
                progress("profiles", added, len(todo))
    return added


def _convert_dataset(case, dataset, role, writer, pool, progress, shard_time, limit):
    grid = ROLES[role]
    times = _frame_times(case, dataset)
    first = [read_piece(p, point_arrays=[], with_cells=False) for p in dataset.frames[0].pieces]
    xs, ys = _grid_lines([p.points for p in first])
    zs = None
    if grid == "sigma" and dataset.solver not in SIGMA_SOLVERS:
        grid = "cartesian"  # CFD: levels at fixed heights, MPI pieces split in z too
        zs = np.unique(
            np.concatenate([np.round(p.points[:, 2].astype(np.float64), 3) for p in first])
        )
        plans = [plan_cartesian(p.points, xs, ys, zs) for p in first]
    elif grid == "sigma":
        planned = [plan_volume(p.points, xs, ys) for p in first]
        plans = [plan for plan, _ in planned]
        if len({nz for _, nz in planned}) != 1:
            raise ConvertError("the pieces have different numbers of levels")
    else:
        plans = [plan_surface(p.points, xs, ys) for p in first]
    variables = dict(dataset.components)
    exists = role in writer.root
    if exists:
        out = writer.output(role, shard_time=shard_time)
        done = set(int(s) for s in out.output.steps)
        recorded = out.output.info.get("dataset")
        if recorded and recorded != dataset.key:
            raise ConvertError(
                f"{role}: the store holds {recorded}, the case now shows {dataset.key} "
                "(another run): move the store away to convert it"
            )
        sigma = out.output.sigma
        grid = out.output.grid
        blocks = [_extent(b, out.output.grid) for b in out.output.blocks]
        if blocks != [(tuple(p.start), tuple(p.size)) for p in plans] or len(out.output.x) != len(
            xs
        ):
            raise ConvertError(f"{role}: the grid differs from the store's")
    else:
        done = set()
        sigma = None
        if grid == "sigma":
            heights = [p.take(f.points[:, 2]) for p, f in zip(plans, first, strict=True)]
            sigma = nominal_sigma(heights)
        out = writer.add_output(
            role,
            grid,
            xs,
            ys,
            variables,
            [(p.start, p.size) for p in plans],
            sigma=sigma,
            z=zs,
            source=dataset.folder.name,
            shard_time=shard_time,
            attributes={"dataset": dataset.key},
        )
    todo = [(k, f) for k, f in enumerate(dataset.frames) if f.step not in done]
    if limit is not None:
        todo = todo[:limit]
    size = 0

    def read_frame(item):
        k, frame = item
        pieces = [read_piece(p, with_cells=False) for p in frame.pieces]
        return k, frame, pieces

    for count, (k, frame, pieces) in enumerate(_prefetched(pool, read_frame, todo), start=1):
        if len(pieces) != len(plans) or any(
            p.points.shape[0] != plan.points for p, plan in zip(pieces, plans, strict=True)
        ):
            raise ConvertError(f"{role}: output {frame.step} has another grid than the first")
        blocks = []
        for plan, piece in zip(plans, pieces, strict=True):
            z = plan.take(piece.points[:, 2])
            arrays = {name: plan.take(values) for name, values in piece.point_data.items()}
            if grid == "cartesian":
                pass  # the heights are the grid's (z of the levels)
            elif grid == "sigma":
                arrays["z_bed"], arrays["z_surface"] = z[0], z[-1]
                offset = level_offsets(z, sigma)
                if offset.any():
                    arrays["z_offset"] = offset
            else:
                arrays["z"] = z
            blocks.append(arrays)
        out.append(times[k], frame.step, blocks)
        size += sum(p.stat().st_size for p in frame.pieces)
        if progress is not None:
            progress(role, count, len(todo))
    out.flush()
    return len(todo), size


_BODY_FILE = re.compile(r"-(?P<body>\d+)-(?P<step>\d+)$")


def _body_number(path: Path, default: int) -> int:
    match = _BODY_FILE.search(path.stem)
    return int(match.group("body")) if match else default


def _read_bodies(frame) -> dict[int, tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]]:
    """The bodies of one output: number -> (vertices, triangles, point arrays)."""
    bodies = {}
    for k, path in enumerate(frame.pieces):
        piece = read_piece(path)
        triangles = triangulate(piece.connectivity, piece.offsets)
        bodies[_body_number(path, k)] = (piece.points, triangles, dict(piece.point_data))
    return bodies


def _convert_bodies(case, dataset, writer, pool, progress, limit) -> tuple[int, int]:
    """Floating bodies (SPEC 9): the mesh of each body once and its rigid motion per
    output; a body that turns out not to move rigidly is written again with its
    vertices per output."""
    key = dataset.key
    times = _frame_times(case, dataset)
    source = dataset.folder.name
    stored = Store(writer.path).bodies.get(key)
    rigid: dict[int, bool] | None = None
    if stored is not None:
        if stored.info.get("dataset", key) != key:
            raise ConvertError(f"bodies: the store holds {stored.info.get('dataset')}")
        rigid = {n: b.rigid for n, b in stored.bodies.items()}
    while True:
        if stored is None:
            first = _read_bodies(dataset.frames[0])
            if rigid is None:
                rigid = dict.fromkeys(first, True)
            body_writer = writer.body_writer(
                key,
                {
                    n: (
                        v,
                        tri,
                        {name: _components(a) for name, a in data.items()},
                        rigid.get(n, True),
                    )
                    for n, (v, tri, data) in first.items()
                },
                source=source,
                attributes={"dataset": key},
            )
            done: set[int] = set()
        else:
            body_writer = writer.body_writer(key)
            done = {int(s) for s in stored.steps}
        todo = [(k, f) for k, f in enumerate(dataset.frames) if f.step not in done]
        if limit is not None:
            todo = todo[:limit]
        size = 0
        try:
            for count, (k, frame, bodies) in enumerate(
                _prefetched(pool, lambda item: (*item, _read_bodies(item[1])), todo), start=1
            ):
                body_writer.append(
                    times[k], frame.step, {n: (v, data) for n, (v, _t, data) in bodies.items()}
                )
                size += sum(p.stat().st_size for p in frame.pieces)
                if progress is not None:
                    progress("bodies", count, len(todo))
        except NotRigid as bent:
            # write the set again from the first output, this body with its vertices
            if len(dataset.frames) < len(stored or ()):
                raise ConvertError(
                    f"bodies: {bent}; the VTK files converted before are gone"
                ) from bent
            body_writer.discard()  # nothing of this set is kept
            rigid[bent.number] = False
            stored = None
            continue
        except ValueError as error:
            raise ConvertError(f"bodies: {error}") from error
        body_writer.flush()
        return len(todo), size


def _components(values: np.ndarray) -> int:
    return int(values.shape[1]) if values.ndim > 1 else 1


def _extent(block, grid: str) -> tuple[tuple[int, ...], tuple[int, ...]]:
    if grid == "cartesian":
        return (*block.start, block.k0), (*block.size, block.nz)
    return tuple(block.start), tuple(block.size)


def _prefetched(pool, function, items, ahead: int = 3):
    """``map`` that reads at most ``ahead`` frames before they are needed (memory)."""
    pending = []
    items = iter(items)
    for item in items:
        pending.append(pool.submit(function, item))
        if len(pending) > ahead:
            yield pending.pop(0).result()
    for future in pending:
        yield future.result()


# ----------------------------------------------------------------- command line
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="lagoon-convert",
        description="Convert REEF3D VTU/VTP output to a LAGOON store (again: append new outputs)",
    )
    parser.add_argument("case", help="REEF3D case folder")
    parser.add_argument("--out", help="store to write (default: CASE/REEF3D_<SOLVER>.lagoon)")
    parser.add_argument(
        "--outputs",
        default=",".join((*ROLES, *EXTRAS)),
        help="what to convert, comma separated (default: everything: "
        "volume,free_surface,bed,bodies,series,profiles)",
    )
    parser.add_argument("--limit", type=int, help="convert at most this many outputs per stream")
    parser.add_argument("--shard-time", type=int, default=16, help="outputs per shard file")
    args = parser.parse_args(argv)
    roles = tuple(r.strip() for r in args.outputs.split(",") if r.strip())
    unknown = [r for r in roles if r not in ROLES and r not in EXTRAS]
    if unknown:
        parser.error(f"unknown outputs: {', '.join(unknown)}")
    last = [0.0]

    def progress(role: str, done: int, total: int) -> None:
        now = _time.perf_counter()
        if done == total or now - last[0] > 1.0:
            last[0] = now
            print(f"\r{role}: {done}/{total}", end="" if done < total else "\n", flush=True)

    try:
        report = convert(args.case, args.out, roles, progress, args.shard_time, args.limit)
    except (ConvertError, StoreError, FileNotFoundError, ValueError) as error:
        print(f"lagoon-convert: {error}", file=sys.stderr)
        return 1
    print(report.summary())
    return 0


if __name__ == "__main__":
    sys.exit(main())
