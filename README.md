<p align="center"><img src="docs/logo.png" alt="LAGOON" width="520"></p>

**LAGOON: Lightweight Animation and Graphics for Open Ocean Numerics.**
An open-source desktop application for exploring, plotting and rendering ocean and
coastal simulations, in the spirit of ParaView and Blender but built for
hydrodynamics. Part of the [REEF3D](https://github.com/REEF3D) family.

> Status: **phase 0, foundations.** The application window, scene tree, properties,
> timeline and Python console exist; there is no data reading or 3D rendering yet.
> See the implementation plan for what comes next.

## Run it

With [pixi](https://pixi.sh) (recommended, identical environment on every machine):

```bash
pixi run lagoon        # start the application
pixi run test          # run the tests
pixi run lint          # ruff + layer rules
```

With pip in any Python 3.11+ environment:

```bash
pip install -e ./lagoon-format -e ".[dev]"
lagoon                 # or: python -m lagoonviz
pytest
```

## Layout

| Path | What | License |
| --- | --- | --- |
| `src/lagoonviz/core` | Scene, commands, sessions. No Qt, no GPU. | GPLv3 |
| `src/lagoonviz/render` | pygfx and matplotlib rendering (from phase 1) | GPLv3 |
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
fs = session.add_item("free_surface", colormap="balance")
session.set_property(fs, "exaggeration", 8.0)
session.save("seamount.lagoon-session")
print(session.to_script())  # the history as a runnable script
```

In the application, *File → Export script* writes the same script, and the Python
console shows every command as it runs.

## License

The LAGOON application is licensed under the GNU General Public License v3.0 or
later (`LICENSE`). The data format package in `lagoon-format/` is MIT-licensed
(`lagoon-format/LICENSE`) so that other codes can read and write the format freely.
