# SPDX-License-Identifier: GPL-3.0-or-later
# Architect: Hans Bihs
"""The pictures of the documentation, made from a small example run.

    python tools/docs_images.py                 # all of them, into docs/images
    python tools/docs_images.py figure-series   # some of them

Runs before the documentation is built (.github/workflows/docs.yml), so the
pictures always show what the current LAGOON draws. The 3D pictures need pygfx
and a GPU (or Mesa's software Vulkan driver); the paper figures only matplotlib.
The example run is the synthetic NHFLOW case of the tests
(``tests/reef3d_fixtures.py``: waves over a sloping bed, gauges, wave profiles, a
moored floating body).
"""

from __future__ import annotations

import sys
import tempfile
from collections.abc import Callable
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "docs" / "images"
SIZE = (1200, 675)


def example_session(folder: Path):
    """The example run, opened with a slice and the gauges plotted."""
    sys.path.insert(0, str(ROOT / "tests"))
    from reef3d_fixtures import make_run

    from lagoonviz.core import Session

    run = make_run(
        folder / "flume",
        frames=6,
        print_interval=0.5,
        series=True,
        profiles=True,
        objects=True,
    )
    _irregular_series(run)
    session = Session()
    session.open_case(run)
    session.set_time(1.5)
    return session


def _irregular_series(run: Path, duration: float = 60.0, dt: float = 0.05) -> None:
    """Gauges, forces and heave of irregular waves (a JONSWAP-like sea, Hs 0.1 m,
    Tp 1.6 s) instead of the tests' few samples, so the figures look like a flume's."""
    import numpy as np
    from reef3d_fixtures import GAUGES

    rng = np.random.default_rng(3)
    t = np.arange(0.0, duration + dt / 2, dt)
    tp, hs, gamma = 1.6, 0.1, 3.3
    f = np.linspace(0.3, 2.0, 60)
    fp = 1 / tp
    sigma = np.where(f <= fp, 0.07, 0.09)
    peak = gamma ** np.exp(-((f - fp) ** 2) / (2 * sigma**2 * fp**2))
    shape = f**-5 * np.exp(-1.25 * (fp / f) ** 4) * peak
    integrate = getattr(np, "trapezoid", None) or np.trapz  # NumPy 2 renamed it
    density = shape * (hs / 4) ** 2 / integrate(shape, f)
    amplitude = np.sqrt(2 * density * (f[1] - f[0]))
    phase = rng.uniform(0, 2 * np.pi, f.size)
    k = (2 * np.pi * f) ** 2 / 9.81

    def eta(x: float, lag: float = 0.0) -> np.ndarray:
        arg = 2 * np.pi * f[None, :] * (t[:, None] - lag) - k[None, :] * x + phase[None, :]
        return (amplitude[None, :] * np.cos(arg)).sum(axis=1)

    solver = "NHFLOW"
    lines = [f"number of gauges:  {len(GAUGES)}", "", "x_coord     y_coord"]
    lines += [f"{i + 1}\t {x:g}\t {y:g}" for i, (x, y) in enumerate(GAUGES)]
    lines += ["", "", "time\t " + "\t ".join(f"P{i + 1}" for i in range(len(GAUGES)))]
    columns = [eta(x) for x, _ in GAUGES]
    for i, time in enumerate(t):
        lines.append(f"{time:.3f}\t " + "\t ".join(f"{c[i]:.6f}" for c in columns))
    (run / f"REEF3D_{solver}_WSF" / f"REEF3D-{solver}-WSF-HG.dat").write_text(
        "\n".join(lines) + "\n"
    )
    body = eta(5.0, lag=0.12)
    dof = run / f"REEF3D_{solver}_6DOF"
    forces = ["time \t Fx \t Fy \t Fz"]
    forces += [
        f"{time:.3f} \t {4000 * b:.3f} \t 0 \t {2000 + 6000 * b:.3f}"
        for time, b in zip(t, eta(5.0, lag=0.4), strict=True)
    ]
    (dof / "REEF3D_6DOF_forces_0.dat").write_text("\n".join(forces) + "\n")
    position = ["time \t XG \t YG \t ZG \t Phi \t Theta \t Psi"]
    position += [
        f"{time:.3f} \t 5 \t 2 \t {1.5 + 0.8 * b:.5f} \t 0 \t 0 \t 0"
        for time, b in zip(t, body, strict=True)
    ]
    (dof / "REEF3D_6DOF_position_0.dat").write_text("\n".join(position) + "\n")


def _view(name: str, style: str = "studio", view: str | None = None, setup=None):
    def make(session, path: Path) -> None:
        from lagoonviz.render.offscreen import render_session, write_png

        if setup is not None:
            setup(session)
        write_png(path, render_session(session, SIZE, view=view, style=style))

    return make


def _slice(session) -> None:
    session.add_slice("y")


def _figure(kind: str):
    def make(session, path: Path) -> None:
        from lagoonviz.render.figure import (
            FigureSpec,
            profile_figure,
            save_figure,
            series_figure,
            spectrum_figure,
        )

        spec = FigureSpec.from_style("REEF3D", width_mm=140, dpi=150, time_cursor=False)
        draw = {"series": series_figure, "spectra": spectrum_figure, "profiles": profile_figure}
        save_figure(draw[kind](session, spec), path, spec)

    return make


#: picture name (docs/images/<name>.png) -> how it is made
IMAGES: dict[str, Callable] = {
    "view-studio": _view("view-studio"),
    "view-ocean": _view("view-ocean", style="ocean"),
    "view-top": _view("view-top", view="2d"),
    "view-slice": _view("view-slice", setup=_slice),
    "look-open-ocean": _view("look-open-ocean", style="open_ocean"),
    "look-wave-flume": _view("look-wave-flume", style="wave_flume"),
    "look-underwater": _view("look-underwater", style="underwater"),
    "look-diorama": _view("look-diorama", style="diorama"),
    "look-diorama-classic": _view("look-diorama-classic", style="diorama_classic"),
    "figure-series": _figure("series"),
    "figure-spectra": _figure("spectra"),
    "figure-profiles": _figure("profiles"),
}


def main(argv: list[str]) -> int:
    names = argv or list(IMAGES)
    unknown = [n for n in names if n not in IMAGES]
    if unknown:
        print(f"unknown: {', '.join(unknown)}; known: {', '.join(IMAGES)}", file=sys.stderr)
        return 2
    OUT.mkdir(parents=True, exist_ok=True)
    failed = []
    with tempfile.TemporaryDirectory(prefix="lagoon-docs-") as folder:
        for name in names:
            session = example_session(Path(folder) / name)  # each picture from a fresh run
            path = OUT / f"{name}.png"
            try:
                IMAGES[name](session, path)
            except Exception as error:  # noqa: BLE001 (report all, then fail)
                failed.append(name)
                print(f"{name}: FAILED: {error}", file=sys.stderr)
                continue
            print(f"{name}: {path.relative_to(ROOT)}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
