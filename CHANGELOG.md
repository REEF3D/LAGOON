# Changelog

## 0.2.0.dev0 (phase 2, domain tools; in progress)

- Time series: REEF3D wave gauges (`REEF3D_*_WSF`), 6DOF forces, motions, velocities
  and mooring lines (`REEF3D_*_6DOF`) and force files are found in a case and read on
  first use (again when a running simulation appends to them). Units in column names
  and gauge positions are kept; REEF3D's `-1e20` and `nan` mean "no value".
- New item kind *Time series* (file and columns). Opening a case adds plots for the
  shown run's gauges, body forces and motion. A case with only time series opens too.
- Plot panel: one row per series, shared time axis, orange time cursor linked to the
  timeline and the 3D view; click or drag to set the time, wheel to zoom, right drag
  to pan, double-click to reset; legend with values at the cursor; min/max decimation.
- Slices through volume output (x, y or z plane), coloured by any variable, cut with
  marching tetrahedra in NumPy; vertical slices of σ-grids reuse their cut plan, so
  playing a slice costs only interpolation. *Add → Slice* cuts the middle of the volume.
- Click on a surface in the 3D view to read its values at that point; the marker
  stays and its values follow the timeline.
- Fresher look: light theme in REEF3D blues by default, dark theme under View → Theme
  (Fusion style on every platform), icons drawn in code, toolbar with icons and labels,
  larger player buttons, Space to play and pause.

## 0.1.0.dev0 (phase 1, viewer; in progress)

- REEF3D readers (`lagoonviz.core.data`), NumPy only: VTK XML `.vtu`/`.vtp` and the
  parallel `.pvtu`/`.pvtp`; raw appended, base64, zlib and ASCII data; arrays read on
  demand. Tolerates the oversized block header of REEF3D::FNPF's `elevation` array.
- MPI pieces are merged into one mesh (shared boundary points joined); volumes are
  reduced to their outer surface. Merge plans are made once per dataset.
- Case discovery: a case folder, a `REEF3D_*` output folder or a single file; free
  surface, bed and volume output of NHFLOW, FNPF and CFD; output interval from
  `ctrl.txt` (`P 30`). Folders holding several runs show the latest one.
- Frame cache (1 GB budget) and background loader with prefetch while playing.
- Scene: `Session.open_case`, `set_properties`, `fit_range`, saved camera; item kind
  `volume`; data-dependent choices (variables, datasets); colour maps as tables
  shared by all renderers.
- 3D viewport with pygfx (Metal, Vulkan, Direct3D 12): Blender-style navigation and
  view keys, colour bars, vertical exaggeration, automatic or fixed colour ranges.
- Timeline over the frames of the case, in seconds when the interval is known;
  play, step, scrub, playback speed. Scene tree groups items under their case.
- Headless rendering: `lagoon-render case/ out.png` and `render_session()`.
- Screenshots (View → Save screenshot, F12); `lagoon PATH` opens a case or session.

## 0.0.1.dev0 (phase 0, foundations)

- Repository layout: `lagoonviz` application (GPLv3) and `lagoon-format` (MIT).
- Qt-free core: scene with deterministic item ids, item kinds with validated properties,
  undoable commands, sessions saved as JSON, script recorder.
- Main window (PySide6): scene tree, viewport placeholder, properties panel generated
  from the item kinds, plots placeholder, timeline, Python console with command log;
  undo/redo, session open/save, script export, saved and resettable layout.
- Tests for core and GUI (offscreen), including the headless replay test.
- Layer rules checked with import-linter; CI on macOS, Linux and Windows.
