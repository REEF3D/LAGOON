# Architecture

```
lagoonviz.gui      PySide6 windows and panels      -> may import render, core
lagoonviz.render   pygfx scene, offscreen images,  -> may import core, never Qt
                   films, figures (matplotlib),
                   cinematic looks, path tracing
lagoonviz.core     scene, kinds, commands, session -> no Qt, no GPU, no plotting
                   camera paths, figure templates,
                   analysis, the converter
  core.data        VTK XML reader, REEF3D cases,
                   AMR, LAGOON stores, merging,
                   cache, loader                   -> NumPy only
lagoonviz.plugins  readers for other codes (later) -> may import core, render
lagoon_format      data format, MIT                -> never imports lagoonviz
```

Blender is never imported: `render.pathtrace` writes the scene to a folder and
runs Blender as a separate process on `render/blender_cycles.py`, a script that
uses only Blender's own Python.

## Data flow

1. A user action in the GUI (menu, scene tree, properties panel, console) calls a
   `Session` method, e.g. `session.set_property(id, "opacity", 0.5)`.
2. `Session` creates a command and runs it on the `CommandStack`.
3. The command changes the `Scene`; the scene notifies its observers.
4. `SessionBridge` turns core notifications into Qt signals; panels update.
5. The stack's history is the recorded script (`Session.to_script()`), shown live in
   the Python console.

## From files to pixels

1. `Session.open_case(path)` runs one `OpenCase` command: a `case` item (the path)
   and items for the latest run's free surface and bed. Only descriptions go into
   the scene; the data stays on disk.
2. `DataManager` (`session.data`) opens the case (`core.data.reef3d.open_case`):
   each `REEF3D_*` folder becomes a `Dataset` whose frames are the numbered files.
3. For an item and the scene time it builds a `FrameKey` (case, dataset, frame,
   arrays). The `FrameCache` holds recent frames; the `BackgroundLoader` reads
   missing ones in worker threads and prefetches ahead while playing.
4. A frame is read with `vtkxml.read_piece` per MPI piece (only the arrays needed)
   and merged with a `MergePlan` made once per dataset. Volumes keep only their
   outer surface. Normals are computed in the worker thread.
5. `DataManager.display` turns the mesh into plain arrays for an item's settings
   (exaggeration, colour range, colour map): a `DisplaySurface`.
6. `render.scene3d.SceneRenderer` puts those arrays into pygfx buffers (updated in
   place while the topology stays the same). The GUI viewport and
   `render.offscreen` use the same renderer, so screen and headless images match.
   Films (`render.movie`) and the 3D panels of figures (`render.figure`) render
   off screen the same way; path tracing (`render.pathtrace`) hands the same
   arrays to Blender.

## Key rules

- Item ids are deterministic (`<kind>-<n>`), including across undo, so recorded
  scripts replay to the same scene.
- Property values are validated in the core (`kinds.validate`), so scripts and the
  GUI follow the same rules. The properties panel is generated from the kind specs.
- Slider and spin-box drags fold into one undo step (`merge=True`, `end_merge()`).
- Time is view state, not an undo step.
- The scene tree updates entries in place on changes, never rebuilding the entry
  that is emitting a Qt signal.
- Readers and geometry live in the core and are tested with small synthetic runs
  written like REEF3D writes them (`tests/reef3d_fixtures.py`).
- The camera is view state stored in the scene (saved with the session, not undoable).
