# LAGOON architecture (phase 0)

```
lagoonviz.gui      PySide6 windows and panels      -> may import render, core
lagoonviz.render   pygfx / matplotlib (phase 1+)   -> may import core, never Qt
lagoonviz.core     scene, kinds, commands, session -> no Qt, no GPU, no plotting
lagoonviz.plugins  readers for other codes (later) -> may import core, render
lagoon_format      data format, MIT                -> never imports lagoonviz
```

## Data flow

1. A user action in the GUI (menu, scene tree, properties panel, console) calls a
   `Session` method, e.g. `session.set_property(id, "opacity", 0.5)`.
2. `Session` creates a command and runs it on the `CommandStack`.
3. The command changes the `Scene`; the scene notifies its observers.
4. `SessionBridge` turns core notifications into Qt signals; panels update.
5. The stack's history is the recorded script (`Session.to_script()`), shown live in
   the Python console.

## Key rules

- Item ids are deterministic (`<kind>-<n>`), including across undo, so recorded
  scripts replay to the same scene.
- Property values are validated in the core (`kinds.validate`), so scripts and the
  GUI follow the same rules. The properties panel is generated from the kind specs.
- Slider and spin-box drags fold into one undo step (`merge=True`, `end_merge()`).
- Time is view state, not an undo step.
- The scene tree updates entries in place on changes, never rebuilding the entry
  that is emitting a Qt signal.
