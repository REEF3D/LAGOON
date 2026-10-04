# SPDX-License-Identifier: GPL-3.0-or-later
# Architect: Hans Bihs
"""Compare a LAGOON store with the VTU/VTP output it was converted from.

    python tools/benchmark_format.py CASE [--outputs 64] [--store PATH]

Converts the first outputs of CASE (if the store is not there yet) and reports
size, time to open, and the time to read a frame, a slice, the volume faces and a
probe series, from the VTK files and from the store.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from lagoonviz.core.convert import convert, store_size
from lagoonviz.core.data.lagoon import store_datasets
from lagoonviz.core.data.reef3d import open_case


def timed(function, repeat: int = 1):
    times = []
    result = None
    for _ in range(repeat):
        start = time.perf_counter()
        result = function()
        times.append(time.perf_counter() - start)
    return min(times), result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("case")
    parser.add_argument("--outputs", type=int, default=64)
    parser.add_argument("--store")
    args = parser.parse_args()
    case_dir = Path(args.case)
    store = Path(args.store) if args.store else case_dir / "benchmark.lagoon"
    if not store.exists():
        seconds, report = timed(lambda: convert(case_dir, store, limit=args.outputs))
        print(f"converted in {seconds:.1f} s: {report.summary()}")
    print(f"store size {store_size(store) / 1e6:.1f} MB")
    seconds, case = timed(lambda: open_case(case_dir, stores=False))
    print(f"open the VTK case: {seconds * 1000:.0f} ms ({len(case.datasets)} datasets)")
    seconds, stored = timed(lambda: store_datasets(store))
    print(f"open the store:    {seconds * 1000:.0f} ms")
    for lagoon in stored:
        vtk = next(d for d in case.datasets.values() if d.key == lagoon.key)
        compare(vtk, lagoon)


def compare(vtk, lagoon) -> None:
    n = len(lagoon)
    files = sum(p.stat().st_size for f in vtk.frames[:n] for p in f.pieces)
    size = sum(p.stat().st_size for p in lagoon.output.group.path.rglob("*") if p.is_file())
    ratio = files / max(size, 1)
    print(f"\n== {lagoon.key}: {n} outputs, VTK {files / 1e6:.0f} MB, store {size / 1e6:.0f} MB")
    print(f"  {ratio:.1f}x smaller")
    variable = "velocity" if "velocity" in lagoon.components else next(iter(lagoon.components))
    lo, hi = lagoon.bounds(0)
    middle = float((lo[0] + hi[0]) / 2)
    rows = [("frame (first read of each)", lambda d, k: d.read(k, [variable]))]
    if lagoon.role == "volume":
        rows.append(("x slice", lambda d, k: d.read_slice(k, 0, middle, [variable])))
    frames = list(range(min(n, 8)))
    for label, function in rows:
        line = []
        for name, dataset in (("VTK", vtk), ("store", lagoon)):
            start = time.perf_counter()
            for k in frames:
                function(dataset, k)
            line.append(f"{name} {1000 * (time.perf_counter() - start) / len(frames):6.0f} ms")
        print(f"  {label:28s} " + "   ".join(line))
    x, y = middle, float((lo[1] + hi[1]) / 2)
    a, (_, va) = timed(lambda: vtk.point_series(variable, x, y))
    b, (_, vb) = timed(lambda: lagoon.point_series(variable, x, y))
    print(f"  {'probe series':28s} VTK {a * 1000:6.0f} ms   store {b * 1000:6.0f} ms")
    print(f"  (VTK over {len(vtk)} outputs, store over {n})")


if __name__ == "__main__":
    main()
