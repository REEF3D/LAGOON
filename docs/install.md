# Install

LAGOON runs on macOS, Linux and Windows with Python 3.11 to 3.13.

=== "pip"

    ```bash
    pip install lagoon-viz
    lagoon
    ```

    Best in a fresh environment (`python -m venv lagoon`, or
    `uv venv && uv pip install lagoon-viz`).

=== "conda / pixi"

    From conda-forge (from the 1.0 release on):

    ```bash
    conda install -c conda-forge lagoon-viz
    # or, in a pixi project
    pixi add lagoon-viz
    ```

=== "From the sources"

    ```bash
    git clone https://github.com/REEF3D/LAGOON.git
    cd LAGOON
    pixi run lagoon            # with pixi: the same environment everywhere
    # or with pip
    pip install -e ./lagoon-format -e ".[dev]"
    ```

    On a Mac, `bash setup_lagoon.command` installs everything (uv, Python, LAGOON),
    runs the tests and makes a `LAGOON.app`.

Three commands come with it:

| Command | What it does |
| --- | --- |
| `lagoon` | the application (also `python -m lagoonviz`) |
| `lagoon-render` | images and films without a screen, e.g. on a cluster |
| `lagoon-convert` | REEF3D output to a LAGOON store |

See [Command line](reference/cli.md) for their options.

## Graphics

The 3D view draws with WebGPU: Metal on macOS, Vulkan on Linux, Direct3D 12 on
Windows. Any GPU of the last ten years does. Headless Linux machines (clusters,
CI) render with Mesa's software Vulkan driver (`mesa-vulkan-drivers` on
Debian/Ubuntu); without a usable GPU the window opens with a note in place of the
3D view, and plots, figures and scripts still work.

## Optional

- **Films** (MP4) need `imageio-ffmpeg`, which pip installs with LAGOON; it brings
  its own ffmpeg.
- **Path tracing** needs [Blender](https://www.blender.org) 4.2 or newer. LAGOON
  finds it in Applications (macOS), on the `PATH`, or where `LAGOON_BLENDER`
  points. See [Cinematic looks and path tracing](guide/cinematic.md).
- **LaTeX** labels in paper figures work without LaTeX (matplotlib's mathtext);
  with a LaTeX installation, *Labels through LaTeX* sets them in real LaTeX.
- **The LAGOON format** alone, for other codes: `pip install lagoon-format` (no
  dependencies).
