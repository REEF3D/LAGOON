# SPDX-License-Identifier: GPL-3.0-or-later
# Architect: Hans Bihs
"""How LAGOON copes with large cases: a synthetic LAGOON store as REEF3D writes
it (an FNPF-like free surface and bed on many MPI blocks, optionally a σ-volume),
then the time and memory to open it and to read, show and play its frames.

    python tools/benchmark_large.py make DIR [--nx 4000 --ny 2500 --frames 8 --blocks 16]
                                             [--levels 10]     # also a volume
    python tools/benchmark_large.py time DIR/Large.lagoon [--memory MODE] [--play 6]

A 4000 x 2500 surface is 10 million points, the free surface of a case with
100 million cells (10 levels).
"""

from __future__ import annotations

import argparse
import math
import os
import resource
import sys
import time
from pathlib import Path

import numpy as np

#: what REEF3D's FNPF writes on its free surface
SURFACE_VARIABLES = {
    "velocity": 3,
    "Fifsf": 1,
    "eta": 1,
    "depth": 1,
    "breaking": 1,
    "coastline": 1,
    "Hs": 1,
}


def peak_mb() -> float:
    """The process's peak memory so far (MB)."""
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return peak / (1 << 20) if sys.platform == "darwin" else peak / 1024


def _blocks(nx: int, ny: int, count: int) -> list[tuple[tuple[int, int], tuple[int, int]]]:
    """MPI-like blocks: a near-square split, neighbours sharing a line of points."""
    cols = max(1, round(math.sqrt(count * nx / ny)))
    rows = max(1, count // cols)
    xs = np.linspace(0, nx - 1, cols + 1).round().astype(int)
    ys = np.linspace(0, ny - 1, rows + 1).round().astype(int)
    blocks = []
    for r in range(rows):
        for c in range(cols):
            i0, i1 = xs[c], xs[c + 1] + 1
            j0, j1 = ys[r], ys[r + 1] + 1
            blocks.append(((int(i0), int(j0)), (int(i1 - i0), int(j1 - j0))))
    return blocks


def make(folder: Path, nx: int, ny: int, frames: int, blocks: int, levels: int) -> Path:
    from lagoon_format.store import StoreWriter

    folder.mkdir(parents=True, exist_ok=True)
    path = folder / "Large.lagoon"
    run = {"run": "bench-1", "started": "2026-10-14T06:00:00Z", "solver": "FNPF"}
    store = StoreWriter.create(path, solver="FNPF", run=run)
    x = np.linspace(0.0, 2.0 * nx, nx)
    y = np.linspace(0.0, 2.0 * ny, ny)
    parts = _blocks(nx, ny, blocks)
    surface = store.add_output("free_surface", "surface", x, y, SURFACE_VARIABLES, parts)
    bed = store.add_output("bed", "surface", x, y, {"depth": 1}, parts)
    volume = None
    if levels:
        sigma = np.linspace(0.0, 1.0, levels)
        volume = store.add_output(
            "volume", "sigma", x, y, {"velocity": 3, "pressure": 1}, parts, sigma=sigma
        )
    gx, gy = np.meshgrid(x.astype(np.float32), y.astype(np.float32))
    bottom = (-20.0 + 0.002 * gx + 3.0 * np.sin(gy / 300.0)).astype(np.float32)
    for frame in range(frames):
        t = 2.0 * frame
        eta = (1.5 * np.sin(gx / 40.0 - t) * np.cos(gy / 90.0)).astype(np.float32)
        wet = eta > bottom
        speed = np.stack([np.cos(gx / 40.0 - t), 0.2 * np.sin(gy / 90.0), 0 * gx], -1)
        fields = {
            "z": np.where(wet, eta, np.nan).astype(np.float32),
            "velocity": speed.astype(np.float32),
            "Fifsf": (eta * 3.0).astype(np.float32),
            "eta": eta,
            "depth": (-bottom).astype(np.float32),
            "breaking": (eta > 1.4).astype(np.float32),
            "coastline": wet.astype(np.float32),
            "Hs": np.abs(eta) * 2.0,
        }
        surface.append(t, frame, [_part(fields, p) for p in parts])
        if frame == 0:
            base = {"z": bottom, "depth": (-bottom).astype(np.float32)}
            bed.append(t, frame, [_part(base, p) for p in parts])
        if volume is not None:
            vol = []
            for start, size in parts:
                cut = (slice(start[1], start[1] + size[1]), slice(start[0], start[0] + size[0]))
                nlev = len(sigma)
                u = np.broadcast_to(speed[cut][None], (nlev, *speed[cut].shape))
                p = np.broadcast_to(
                    (1000.0 * (eta[cut] - bottom[cut]))[None], (nlev, *eta[cut].shape)
                )
                vol.append(
                    {
                        "z_bed": bottom[cut],
                        "z_surface": eta[cut],
                        "velocity": np.ascontiguousarray(u, np.float32),
                        "pressure": np.ascontiguousarray(p, np.float32),
                    }
                )
            volume.append(t, frame, vol)
        print(f"output {frame + 1}/{frames} written", flush=True)
    store.close()
    return path


def _part(fields: dict[str, np.ndarray], block) -> dict[str, np.ndarray]:
    (i0, j0), (w, h) = block
    return {name: v[j0 : j0 + h, i0 : i0 + w] for name, v in fields.items()}


def timed(label: str, function):
    start = time.perf_counter()
    result = function()
    print(f"{label:<44} {time.perf_counter() - start:7.3f} s   peak {peak_mb():7.0f} MB")
    return result


def measure(path: Path, mode: str | None, play: int) -> None:
    if mode:
        os.environ["LAGOON_MEMORY"] = mode
    from lagoonviz.core import Session, memory

    print(f"memory use: {memory.current().describe()}")
    session = timed("open the case", lambda: Session())
    session_open = timed("Session.open_case", lambda: session.open_case(path))
    del session_open
    water = next(i for i in session.scene if i.kind == "free_surface")
    bed = next(i for i in session.scene if i.kind == "bathymetry")
    data = session.data
    mesh = timed("first free surface frame (read)", lambda: data.load(water.id))
    print(f"  {len(mesh.points):,} points, {len(mesh.triangles):,} triangles")
    timed("  ... shown (display arrays)", lambda: data.display(water.id, mesh))
    timed("bed (read + display)", lambda: data.display(bed.id, data.load(bed.id)))
    times = data.timeline()[0]
    for n, when in enumerate(times[1 : 1 + play]):
        session.set_time(float(when))
        frame = timed(f"frame {n + 1}: read", lambda: data.load(water.id))
        timed(f"frame {n + 1}: display arrays", lambda f=frame: data.display(water.id, f))
    if memory.current().detail_points:
        data.moving = True  # playing: large surfaces coarser
        for n, when in enumerate(times[1 + play : 1 + 2 * play]):
            session.set_time(float(when))
            frame = timed(f"playing, frame {n + 1}: read", lambda: data.load(water.id))
            timed(f"playing, frame {n + 1}: display", lambda f=frame: data.display(water.id, f))
        print(f"  {len(frame.points):,} points while playing")
        data.moving = False
    print(f"frame cache: {data.cache.nbytes / 1e6:.0f} MB in {len(data.cache)} meshes")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    m = sub.add_parser("make")
    m.add_argument("folder", type=Path)
    m.add_argument("--nx", type=int, default=4000)
    m.add_argument("--ny", type=int, default=2500)
    m.add_argument("--frames", type=int, default=8)
    m.add_argument("--blocks", type=int, default=16)
    m.add_argument("--levels", type=int, default=0)
    t = sub.add_parser("time")
    t.add_argument("store", type=Path)
    t.add_argument("--memory", default=None)
    t.add_argument("--play", type=int, default=6)
    args = parser.parse_args()
    if args.command == "make":
        path = make(args.folder, args.nx, args.ny, args.frames, args.blocks, args.levels)
        print(f"written: {path}")
    else:
        measure(args.store, args.memory, args.play)


if __name__ == "__main__":
    main()
