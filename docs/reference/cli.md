# Command line

## lagoon

The application.

```text
lagoon [PATH] [--version] [--dev-remote DIR]
```

| | |
| --- | --- |
| `PATH` | a session (`.lagoon-session`), a REEF3D case folder, an output folder or file, or a LAGOON store to open |
| `--version` | print the version |
| `--dev-remote DIR` | developer testing: take screenshots and simple actions from `DIR/request.json` |

`python -m lagoonviz` does the same.

## lagoon-render

Images and films without a screen: on a cluster, in a batch job, or for a figure
at a size the window does not have.

```text
lagoon-render SOURCE OUTPUT [options]
```

| | |
| --- | --- |
| `SOURCE` | a case folder, an output file, a `.lagoon-session` or a `.py` script |
| `OUTPUT` | an image (`.png`) or a film (`.mp4`, `.mov`) |
| `--time TIME` | the time (or output) to show |
| `--size SIZE` | image size, e.g. `1920x1080` (default `1600x900`) |
| `--view VIEW` | `perspective`, `top`, `front`, `right`, or the 2D views `2d`, `2d-front`, `2d-right` (default: the session's view) |
| `--style STYLE` | the look: `studio`, `ocean`, `open_ocean`, `wave_flume`, `underwater` |
| `--fps FPS` | films: frames per second (default 25) |
| `--start START` | films: the first output (default: the first with data) |
| `--end END` | films: the last output (default: the last) |
| `--every EVERY` | films: every n-th output |
| `--engine ENGINE` | `pygfx` (LAGOON's own renderer, the default) or `cycles` (path traced by Blender) |
| `--samples SAMPLES` | Cycles: samples per pixel (default 64) |
| `--blender BLENDER` | Cycles: the Blender executable (default: found) |
| `--fresh` | Cycles: render every frame again, instead of keeping the frames rendered before from the same data, camera and settings |
| `--white-water {none,light,medium,heavy}` | Cycles: how much foam, spray and bubbles where the free surface breaks (default medium) |
| `--no-foam` | Cycles: no white water (as `--white-water none`) |

```bash
lagoon-render ~/runs/flume view.png --time 120 --size 1920x1080
lagoon-render flume.lagoon-session flume.mp4 --start 100 --end 400 --every 2
lagoon-render flume.lagoon-session still.png --engine cycles --style open_ocean
```

## lagoon-convert

REEF3D's VTU/VTP output to a [LAGOON store](../guide/format.md). Run again, it
appends the outputs written since.

```text
lagoon-convert CASE [options]
```

| | |
| --- | --- |
| `CASE` | the REEF3D case folder |
| `--out OUT` | the store to write (default `CASE/REEF3D_<SOLVER>.lagoon`) |
| `--outputs OUTPUTS` | what to convert, comma separated: `volume`, `free_surface`, `bed`, `bodies`, `particles`, `dem`, `ice`, `rods`, `mooring`, `series`, `profiles`, `amr` (default: everything there is) |
| `--limit LIMIT` | convert at most this many outputs per stream |
| `--shard-time SHARD_TIME` | outputs per shard file (default 16) |

## lagoon-setup

A case's set-up, from its input files, in plan and side view: the domain and its
sides, relaxation zones, gauges, probes, bodies, structures, solids
([Case set-up](../guide/setup.md)).

```text
lagoon-setup CASE [OUTPUT] [options]
```

| | |
| --- | --- |
| `CASE` | the case folder, with `control.txt` and/or `ctrl.txt` |
| `OUTPUT` | the figure (default `<case>-setup.pdf` beside the case); PDF, SVG, EPS, PNG, TIFF |
| `--width WIDTH` | width in mm (default 140) |
| `--style STYLE` | `REEF3D` (default), `Journal (serif)`, `Presentation` |
| `--describe` | print the main facts of the input instead |

```bash
lagoon-setup ~/runs/flume
lagoon-setup ~/runs/flume flume.png --width 190 --style Presentation
lagoon-setup ~/runs/flume --describe
```

## Environment variables

| | |
| --- | --- |
| `LAGOON_BLENDER` | the Blender executable for path tracing |
| `LAGOON_DIVEMESH` | the DIVEMesh program (`DiveMESH`) the set-up runs |
| `LAGOON_LOG` | also write the application's log to this file |
| `LAGOON_NO_GPU` | start the window without the 3D view (e.g. over a remote desktop) |
