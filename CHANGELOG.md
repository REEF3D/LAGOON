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
- Floating bodies (`REEF3D_*_6DOF_VTP`), mooring lines with their tension
  (`REEF3D_*_6DOF_Mooring`, legacy VTK) and porous or forced structures
  (`REEF3D_*_VRANS_VTP`, `_FORCING_VTP`) are shown in 3D and move with the timeline;
  opening a case adds those of the shown run. Bodies take a material colour.
- Click on a surface in the 3D view to read its values at that point; the marker
  stays (same size at any zoom, drawn on top) and its values follow the timeline.
- Viewport in the style of the REEF3D result animations ("studio"): white background,
  a fixed sun from above plus a headlamp, matt surfaces, plain sand bed, free surface in
  the new REEF3D colour map (blue troughs, white still water, orange crests), magnitudes
  in "ocean" (white to REEF3D blue). The dark blue look stays under View → Viewport look;
  `lagoon-render --style ocean` too.
- Mooring lines are coloured per vertex by tension, in the new "tension" colour map
  (REEF3D blue to red, without the white middle so lines read on white). The REEF3D
  colour map ends in a clear red-orange and a mid blue instead of near-black; the
  sun is lower, so waves show their shape.
- View → Theme switches at once (no restart) and takes the 3D view along: light with
  the studio look, dark with the ocean look.
- REEF3D run log (`lagoon-format/RUN.md`, version 1): `REEF3D_Case/REEF3D_<SOLVER>_run.jsonl`
  states each run, the streams it writes (roles incl. floating bodies, DEM, CPM
  particles), and the time of every output; streams can register while running.
  Reference reader and checker: `python -m lagoon_format.run <case>`. The REEF3D
  writer (hans_dev patch, `runlog` class) was tested on the NHFLOW Stokes5
  tutorial with a restart; its log is a test fixture.
- Compare side by side (View → Compare side by side, Ctrl+Shift+2): two 3D views,
  each showing one case (chosen in its header); cameras move together and both
  follow the timeline. With two cases open, each view gets one.
- Wave profiles along lines (`REEF3D_*_WSFLINE`): new item kind *Wave profile*
  (lines, wave theory, profile in 3D). A *Wave profiles* tab next to the plots shows
  the elevation along x at the current time (FNPF's wave theory dashed) on a steady
  axis; the 3D view draws each line at its y, in the same colours. Files are timed
  by their `simtime`; a folder written by several runs shows the latest (unless it
  is a short start). Opening a case adds the profiles of the shown run.
- Wave gauges in 3D: each plotted gauge is a staff at its position with a dot at the
  water surface, in the colour of its plot line; the dot follows the timeline. Gauge
  files plot the first four gauges by default; *Gauges in 3D* turns them off.
- macOS: `setup_lagoon.command` builds `LAGOON.app` (`tools/make_macos_app.sh`), a thin
  bundle around the venv, so the menu bar and the Dock show LAGOON with its icon, and
  LAGOON can be started by double-click.
- Studio lighting is balanced for pygfx's physical lights, and the sun casts soft
  shadows (bodies on the water, land relief); the shadow map is fitted to what is
  shown. Crests of the REEF3D colour map end in a warm red.
- File → Export animation (Ctrl+Shift+E): films the 3D view as it is (camera, look)
  over a range of outputs into an MP4 (H.264, plays in QuickTime and PowerPoint), at
  a chosen size and frame rate, with the time and colour bars drawn in. Frames are
  rendered off screen while the next one is read. `lagoon-render case/ film.mp4
  --fps 25 --start 110 --end 400 --every 2` does the same without a window. Needs
  `imageio-ffmpeg` (now a dependency); without it numbered PNGs are written.
- Stronger 3D look: the sun stands across the wave crests (from -x, 31 degrees up),
  with more sun, less ambient light and less headlamp, and glossier water with a soft
  sun sheen. The "greys" colour map runs from dark to mid-light grey (no paper white).
- Bathymetry has a vertical exaggeration too, about the still water level (from the
  bed's still water depth), so bed and free surface can be exaggerated together.
- Colour maps are clamped at their ends: values at or above the top of the range
  wrapped round to the bottom colour before (dark spots on wave crests).
- A run that starts later on the timeline (outputs 0 to n overwritten by a restart)
  is hidden before its first output instead of showing its first frame frozen; the
  timeline (slider, first frame, playback loop) starts at the first output with data. The viewport header
  shows REEF3D's output number.
- Combo boxes in the property panel show their drop-down arrow again; spin boxes have
  clear up/down buttons and ignore the mouse wheel unless they have focus.
- Fixed: the renderer wrote into cached frames (pygfx buffers keep the array they are
  given), so changing the vertical exaggeration compounded, going back in time showed
  the last frame shown, and hidden dry land stuck to every frame. Buffers now get
  copies.
- Fixed: "frame all" and the shadow map use the shown data only; screenshots draw the
  view first, so they are never one change behind; gauges follow the case's
  visibility; a live theme switch recolours the viewport header; a bed from an
  earlier run shows at any time.
- New "rainbow" colour map (Tecplot's classic blue, cyan, green, yellow, red, with
  dark blue and dark red ends so the extremes keep a gradient) in place of cividis; "cividis",
  "jet" and "turbo" in older sessions and scripts map to it.
- Smoother playback: a playback clock that goes on as soon as the next frame is in
  (instead of a whole frame interval later), 4 reading threads prefetching 8 frames
  ahead, the shoreline's triangle list re-used while it stays put (no re-upload),
  gauges and scene bounds updated in place. The developer remote's `measure_play`
  reports where the time goes.
- Choosing a colour map for an item coloured by "none" (a plain sand bed) also
  colours it by its natural variable (a bed's elevation), in one undo step.
- Ctrl+C in the terminal closes LAGOON properly (it used to close only the 3D view
  and leave the terminal waiting).
- Move the view: Ctrl (or Cmd) + drag, Shift + drag or right drag; arrow keys move
  it by a tenth (Shift: a third).
- Fixed: the background loader could deadlock when a frame finished before its
  completion callback was registered (playing, prefetching).
- Animations are saved next to the case by default (or in ~/Movies); an unwritable
  folder is reported before rendering starts.
- Developer remote: `{"diagnose": true}` reports the renderer's items, bounds, camera
  and sun.
- Dry land is left out of the free surface (where its `depth` is zero, as FNPF writes
  it), so the bed shows there; *Hide dry land* turns it off. A case shows its solver's
  bed even when the bed was written by an earlier run.
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
