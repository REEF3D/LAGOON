<p align="center"><img src="https://raw.githubusercontent.com/REEF3D/LAGOON/main/docs/logo.png" alt="LAGOON" width="520"></p>

**LAGOON: Lightweight Animation and Graphics for Open Ocean Numerics.**
An open-source desktop application for exploring, plotting and rendering ocean and
coastal simulations, in the spirit of ParaView and Blender but built for
hydrodynamics. Part of the [REEF3D](https://github.com/REEF3D) family.

LAGOON opens REEF3D runs (NHFLOW, FNPF, SFLOW and CFD output, AMR surfaces, the
LAGOON format), shows free surfaces, beds, volumes, slices, iso-surfaces,
streamlines, floating bodies, moorings, particles and ice in a GPU viewport, plays
them over time, and plots wave gauges, wave profiles, 6DOF forces and motions
linked to the 3D view. It makes paper figures in a journal's style, films with
camera paths, and path-traced showcase images and animations with Blender's
Cycles. Everything it does can be written out as a Python script and replayed
without a screen.

> Status: **1.0 release candidate.** Documentation:
> [reef3d.github.io/LAGOON](https://reef3d.github.io/LAGOON/); what is new: `CHANGELOG.md`.

<p align="center"><img src="https://raw.githubusercontent.com/REEF3D/LAGOON/main/docs/phase1_viewport.png" alt="LAGOON showing an NHFLOW free surface" width="720"></p>

## Install

On a Mac with Apple silicon, the simplest is the app: download
`LAGOON-<version>-macos-arm64.dmg` from the
[releases](https://github.com/REEF3D/LAGOON/releases) and drag LAGOON into
Applications (Python and LAGOON are inside).

With pip, in any Python 3.11 to 3.13 environment (macOS, Linux, Windows):

```bash
pip install lagoon-viz
lagoon                 # or: python -m lagoonviz
```

With conda or pixi, from conda-forge (from the 1.0 release on):

```bash
conda install -c conda-forge lagoon-viz
pixi add lagoon-viz
```

The 3D view needs a GPU with Metal (macOS), Vulkan (Linux) or Direct3D 12
(Windows); headless Linux machines can use Mesa's software Vulkan driver. The
LAGOON data format is a separate MIT-licensed package, `pip install lagoon-format`
(no dependencies), for codes that read or write it.

## Develop

With [pixi](https://pixi.sh) (identical environment on every machine):

```bash
pixi run lagoon        # start the application
pixi run test          # run the tests
pixi run lint          # ruff + layer rules
```

With pip, from a clone:

```bash
pip install -e ./lagoon-format -e ".[dev]"
pytest
```

On a Mac, `setup_lagoon.command` sets up everything (uv, Python, LAGOON, the
tests) and builds a `LAGOON.app`. How a release is made: `RELEASING.md`.

## Open a REEF3D run

*File → Open case…* and pick the folder the run wrote its `REEF3D_*` folders into
(or one of those folders, or a single `.pvtu`/`.pvtp` file). From a terminal:

```bash
lagoon ~/runs/seamount                    # a case folder
lagoon ~/runs/seamount/REEF3D_NHFLOW_VTP_FSF
lagoon figure.lagoon-session              # a saved session
```

In the viewport: left drag rotates, right drag (or Shift + drag) pans, the wheel
zooms. Blender's middle-mouse navigation works too, and the number keys choose views:
1 front, 3 side, 7 top (with Ctrl the opposite side), 5 perspective/orthographic,
Home frames everything. *Probe* (P) and a click on a surface list every value
there in a box beside the view, following the time. The timeline plays and scrubs through the frames; frames are
read in the background and kept in memory, so stepping back is instant.

*Add → Slice* cuts the volume output with an x, y or z plane; set its position and
the variable to colour by in the properties.

Wave gauges (`REEF3D_*_WSF`), 6DOF forces and motions (`REEF3D_*_6DOF`) and force
files of the run are plotted under the 3D view. The orange cursor is the current
time: click or drag in a plot to move the 3D view to that moment. Pick other files or
columns (e.g. `P1, P4` or `Fx, Fz`) in the properties of a *Time series* item.

The same view renders without a screen, e.g. on a cluster:

```bash
lagoon-render ~/runs/seamount frame.png --time 120 --size 1920x1080
lagoon-render figure.lagoon-session figure.png
```

For showcase images and films, Blender's Cycles path tracer renders the same scene
(reflections, refraction through the water, soft shadows) in one of the cinematic
looks. Install [Blender](https://www.blender.org) 4.2 or newer; LAGOON finds it in
`/Applications` or on the `PATH` (or set `LAGOON_BLENDER`) and runs it in the
background, without opening it. In the GUI: *File → Export animation → Renderer:
Blender Cycles*.

```bash
lagoon-render showcase.lagoon-session still.png --engine cycles --style open_ocean
lagoon-render showcase.lagoon-session film.mp4 --engine cycles --samples 256 --size 3840x2160
```

A stopped film goes on where it stopped when made again: frames whose data, camera
and settings are unchanged are kept (`film.cycles/` beside the film).

## Layout

| Path | What | License |
| --- | --- | --- |
| `src/lagoonviz/core` | Scene, commands, sessions, REEF3D readers. No Qt, no GPU. | GPLv3 |
| `src/lagoonviz/render` | pygfx 3D scene, offscreen rendering, figures, path tracing via Blender | GPLv3 |
| `src/lagoonviz/gui` | PySide6 application | GPLv3 |
| `src/lagoonviz/plugins` | Readers and tools for other codes (later) | GPLv3 |
| `lagoon-format/` | LAGOON data format: specification and reference reader/writer | MIT |

The layer rules (core never imports Qt, render never imports the GUI, the format
library never imports the application) are checked in CI with `lint-imports`.

## Scripting

Everything the GUI does goes through `lagoonviz.core.Session`, so it can be scripted
and replayed headless:

```python
from lagoonviz.core import Session

session = Session()
session.open_case("~/runs/seamount")            # adds the case, free surface and bed
session.set_property("free_surface-1", "exaggeration", 8.0)
session.set_time(120.0)
session.save("seamount.lagoon-session")
print(session.to_script())  # the history as a runnable script

mesh = session.data.load("free_surface-1")      # NumPy arrays of the current frame
print(mesh.points.shape, mesh.point_data["eta"].max())
```

In the application, *File → Export script* writes the same script, and the Python
console shows every command as it runs. The figures, films and screenshots made in
the session are in the script too, each after the change it followed: run it with
`python script.py` (for another run of the set-up, change the `open_case` path) to
make them again without the GUI. `Session.from_script` only rebuilds the scene.

## License

The LAGOON application is licensed under the GNU General Public License v3.0 or
later (`LICENSE`). The data format package in `lagoon-format/` is MIT-licensed
(`lagoon-format/LICENSE`) so that other codes can read and write the format freely.
