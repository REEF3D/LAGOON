# Contributing to LAGOON

## Ground rules

1. **The core never imports Qt or GPU libraries.** `lagoonviz.core` must run in a plain
   Python process on a cluster. `lint-imports` enforces this in CI.
2. **Every user action is a command.** GUI code calls `Session` methods
   (`add_item`, `set_property`, ...), never changes the scene directly. That keeps
   undo/redo, the console log and the script recorder correct.
3. **Logic belongs in the core, not in widgets.** If a widget computes something,
   move it to `core` (or `render`) and test it there.
4. **Every figure must be reproducible headless** from a saved session or a recorded
   script. New features come with a test that shows this.
5. **The format library stays independent.** `lagoon-format` (MIT) must not import
   `lagoonviz` (GPLv3).

## Workflow

```bash
pixi run test      # pytest, including GUI tests (offscreen)
pixi run lint      # ruff and the layer rules
pixi run lagoon    # start the application
```

Branch from `main`, keep pull requests small, and make sure CI passes on all three
platforms.

## Testing with real data and a real screen

The automated tests use small synthetic runs (`tests/reef3d_fixtures.py`) and render
offscreen. For real REEF3D output and the real window:

```bash
lagoon ~/runs/case                       # look at it
lagoon --dev-remote /tmp/lagoon-remote ~/runs/case
```

With `--dev-remote`, a tool can drive the running window through
`/tmp/lagoon-remote/request.json` (screenshots, frames, views, a few properties and
playback timing; see `lagoonviz/gui/devremote.py`) and read `result.json`. It cannot
run code. On macOS, `LAGOON_DEV=1 bash setup_lagoon.command CASE` does the same.

## Adding a scene item kind

1. Add a `KindSpec` with its properties to `src/lagoonviz/core/kinds.py`.
2. The properties panel is generated from the spec, so no GUI code is needed for
   simple properties.
3. Add a test in `tests/core`.

## Licensing of contributions

Contributions to the application are licensed under GPLv3-or-later, contributions to
`lagoon-format/` under MIT. A contributor agreement (to keep dual licensing possible)
is still to be decided with NTNU's technology transfer office before outside
contributions are accepted.
