# Changelog

## Unreleased

### Case set-up

- **Set-up and results**: two buttons in the toolbar (*View → Set-up*, *View →
  Results*) switch between the input files of the cases and what the runs wrote.
  The set-up leaves the results out and rests the timeline; set-up items keep a
  check box of their own in each mode. A case that has not run yet opens in the
  set-up. `session.set_mode("setup")` in scripts; saved with the session.
- **File → New case…** starts a case in a folder (`control.txt` and `ctrl.txt`
  begun with a comment line) and opens it in the set-up.
- **A case that has not run yet** opens as its set-up from the input files
  (`control.txt`, `ctrl.txt`, `geo.dat`): DIVEMesh's solids as a structure and a
  *Case set-up* item ([Case set-up](guide/setup.md)).
- **Case set-up item** (also over a case with results, there switched off): the
  domain, the kinds of its sides (C 11 – C 16), the still water level (F 60),
  relaxation zones (B 96, B 107, B 108), wave direction, focus point and
  wavemakers, wave gauges and lines, probes, sections, force boxes, Morison
  cylinders, floating bodies with their centre of gravity and moorings, direct
  forcing and porous structures (A 581 – A 589, B 210 – B 270), regions, the
  bed's topo objects and geo.dat points; gauges and probes numbered as in the
  output files; each group can be switched off. Beside results, `ctrl.txt` may be
  REEF3D's copy in `REEF3D_Case`, and without `control.txt` the domain is the
  results' extent.
- **Set-up panel**: `control.txt` and `ctrl.txt` edited command by command: the
  commands by section, the selected one's values with what the user guide says
  (default, modules, choices), its signs lit in the 3D view; *Add* by number or
  words, *Remove*, the whole file as text. What REEF3D or DIVEMesh would stumble
  over is marked. Each change is an undo step and a script line
  (`session.set_input`, `add_input`, `remove_input`); *Save* writes the files,
  comments and order kept.
- **Run DIVEMesh** from LAGOON (`session.run_divemesh`): `control.txt` is saved,
  DIVEMesh runs in the case folder with its output in the panel, and its grid,
  solids and bed show in the set-up. The program is found as `LAGOON_DIVEMESH`,
  on the PATH or in `~/Codelite/DIVEMesh/bin`, or asked for once.
- **DIVEMesh grid** item: the grid lines on the faces of the domain (every n-th of
  a fine grid), the MPI partition planes; before DIVEMesh has run, the uniform grid
  of `control.txt`; cell counts and sizes in its properties.
- **Set-up sketch**: *File → Export figure → Case set-up* and `lagoon-setup CASE
  [OUTPUT]` draw plan and side view with a legend; `--describe` prints the main
  facts of the input (domain, grid, sides, waves, what does not fit together).
  `lagoonviz.core.data.setup.read_setup` gives the same in a script.
- **The commands of both programs** (`lagoonviz.core.data.controlfile`): every
  command of `ctrl.txt` and `control.txt` with its title, values, choices and
  default, from the REEF3D and DIVEMesh user guides and the programs' readers
  (`tools/command_catalog.py`); input files are read and written back line by
  line, comments and order kept.
- Back in the results, the Set-up panel makes way for *Properties*.
- **The bed in the 3D view** is LAGOON's light sand again in the cinematic looks
  (the dark rock stays in path-traced renders); the sliders in *Properties*
  (vertical scale, arrow length) have no play-range stripes.
- **Iso-surface** in the toolbar, beside *Slice*.
- **Automatic colour range** from the lowest to the highest value, as ParaView
  (it was the 1st to 99th percentile, symmetric around zero). Values all alike
  (a flat bed at the start of a scour run) no longer fix it; slices and
  iso-surfaces widen it to the frames shown, so a bed that scours later keeps
  its colours.
- **Opening a case** also adds its run's volume, switched off (*Volume* in the
  toolbar shows it).

### Fixes

- **A LAGOON store opened on its own** (`REEF3D_FNPF.lagoon`) showed nothing when
  its folder also held other output that could not be read: its frames were read
  from the whole folder instead of the store.
- A run that wrote only its volume (CFD) opened without DIVEMesh's solids.
- With DIVEMesh's solids in a case, the timeline went by output numbers instead
  of the run log's seconds (the free surface of a store showed the wrong output).
- `.pvtp`/`.pvtu` files whose pieces are gone (an earlier run's, the pieces
  removed by a run with P 18 1) are left out instead of stopping the case.

## 1.0.0rc2 (second release candidate)

The second candidate for 1.0: path traced pictures as ÆgirFrames renders them,
and faster; the 3D view in the same light; foam and spray; DIVEMesh solids; a
play range on the timeline; the sun where you want it; Turbo as the rainbow.

### Path tracing with Blender's Cycles

- **As ÆgirFrames renders**: clear water is tinted glass seen by the camera only
  (light and shadows pass through it to the bed), with a faint haze; the Open
  ocean, Wave flume and Diorama looks are a model floating in ÆgirFrames' sky
  picture (MIT, NTNU), which lights it, turned so its sun stands in the sun's
  direction, with Blender's AgX as it is. A bed in its default colour is
  ÆgirFrames' dark rock in Wave flume and Diorama; structures and bodies in their
  default colours are ÆgirFrames' red, half metallic and matte. Colours you choose
  are kept.
- **Open ocean** dresses a coast as land by its height above the still water
  level and its slope (sand under water, wet sand, dry sand, grass and heath,
  rock where it is steep) with fine relief; at the sides of the domain the land
  is cut off in rock down to a base and the water ends in clear walls, with small
  wind ripples, and the water thins out at the shore with broken foam off the
  beach.
- **Foam, spray and bubbles**: white water born where REEF3D flags breaking, on
  sharp fast crests and where the surface flow traps air, then spray flying, foam
  drifting and fading, bubbles rising (diffuse particles after Ihmsen et al.
  2012, scaled to the case), rendered as point clouds. *Foam and spray* in the
  export dialog, `--no-foam` on the command line.
- **Faster films, same pictures**: meshes whose triangles stay from frame to frame
  are moved in place instead of built again; materials alike within 3 % are
  shared and made when first used; smooth surfaces are subdivided (up to twice,
  for smooth crests) only as far as the picture can show; broken triangles are
  dropped on export, so Blender no longer checks every mesh: about three times
  faster in a test on a Mac. Each frame's line in `blender.log` says how long
  building and rendering took.

### The 3D view

- **As Blender renders it**: Open ocean, Wave flume and Diorama show the same sky
  picture in the 3D view (tone mapped as Cycles shows it, turned to the sun),
  lighting the model and reflected by the water, the sun at the picture's height;
  the same dark rock and red; Open ocean and Diorama stand on a block.
- **Diorama** look: the model on a stand, the bed a solid block, the water
  see-through with sides down to the bed. **Diorama classic** keeps it as it was
  first: a blue backdrop, LAGOON's own soft sky and a block of sand.
- **2D view lit as the 3D view**: from the top the items are still drawn flat and
  seen straight down, but shaded with their own shapes' normals in the same look
  and sun, so waves, the bed's relief and the water's glint read as in 3D; the
  headlamp no longer shines straight down (which laid a washed-out glare over
  every flat face).
- **Sun**: *View → Sun...* sets where the light comes from (direction and
  height); the 3D view, films and Cycles renders use it, the sky turns along, it
  is saved with the session and scripts set it with
  `Session.set_sun(azimuth, elevation)`.
- **Rainbow is Turbo**: the rainbow colour map is now Google's Turbo, as in
  SEASTATE's maps (blue through cyan, green and yellow to red, without Turbo's
  darkest ends, yellow and orange given more of the range than green), in place
  of the Tecplot-like rainbow; its smooth lightness shows no false bands. Saved
  sessions and scripts that name `rainbow`, `turbo` or `jet` get it.

### Installing

- **A Mac app**: `LAGOON.app` in a DMG with Python and LAGOON inside, nothing to
  install (Apple silicon); its program also runs `render` and `convert`. Built by
  `packaging/macos/build_app.sh`, signed and notarized when a Developer ID is set
  up (RELEASING.md).

### Runs and the timeline

- **DIVEMesh solids**: a case shows its solids (a monopile, a breakwater, the
  blocks under a beach) as a structure, read from DIVEMesh's
  `DIVEMesh_Paraview/REEF3D_Solid.vtp` or, without it, built from the S commands
  of `control.txt`; their faces in the bed are left out.
- **Play range**: two stripes beside the timeline's slider mark where playing
  starts and ends; dragged, they restrict playback, the export dialog and films
  to a part of the run (saved with the session, `Session.set_play_range`). A
  **repeat** button loops the range, or plays it once.
- **Removing a case removes its items**: its surfaces, volumes, slices, time
  series and the other items that show it go with it (one undo step).
- **A store opened on its own** shows only its solver's run (no gauges of other
  runs in the same folder); a store REEF3D has only just begun says it has no
  outputs yet instead of opening something else.
- **Opening REEF3D_Case opens the case**: REEF3D's own folder (the run log and
  control files) opened only the wave gauges; it now opens the whole case, named
  after the case folder.
- **Run log of outputs in a store** (`lagoon-format/RUN.md`): outputs REEF3D
  writes only into its store (`P 18 1`) are listed as streams of format `zarr`
  with the `group` in the store; times increase per stream (the store counts its
  volume at the next one), and `python -m lagoon_format.run` checks them that way.

## 1.0.0rc1 (first release candidate)

The first public release of LAGOON, the post-processor of REEF3D, as a candidate
for 1.0: everything of phases 0 to 5 below.

- **Open** REEF3D runs: NHFLOW, FNPF, SFLOW and CFD output (VTU/VTP, parallel
  pieces), AMR surfaces, wave gauges, wave profiles, 6DOF, forces, moorings,
  particles, DEM and ice; follow a running simulation; the run log.
- **Show** free surfaces, beds, volumes, slices, iso-surfaces, streamlines,
  vectors and contours in a GPU viewport with Blender-like navigation, linked to
  time series and spectra; compare two runs side by side; a full screen viewer.
- **Store** runs in the LAGOON format (`lagoon-format` 0.3, Zarr v3): several
  times smaller and faster to read, written by `lagoon-convert` or by REEF3D
  itself, read by LAGOON and by the ParaView plugin.
- **Publish**: paper figures in a journal's style from templates (time series,
  spectra, wave profiles, 3D views, panel letters and notes); films with camera
  keyframes and orbits; cinematic looks, and path-traced stills and films with
  Blender's Cycles.
- **Script**: every action goes through `Session`; *Export script* writes a
  runnable Python script, and `lagoon-render` renders sessions, scripts and
  cases without a screen.

Install with `pip install lagoon-viz` (Python 3.11 to 3.13; macOS, Linux,
Windows). Documentation (user guide, scripting, command line, the Session API, the
format specification) at https://reef3d.github.io/LAGOON/, built with MkDocs from
`docs/` (`pixi run docs`), its pictures made from an example run by
`tools/docs_images.py`.

## 0.5.0.dev0 (phase 5, cinematic rendering)

- Cinematic looks (*View → Viewport look*, `lagoon-render --style`): **Open
  ocean** (a sky with a low sun, reflected by the water), **Wave flume** (a lab
  hall with rows of ceiling lights) and **Underwater** (blue depth fog, Snell's
  window overhead). Physically based materials (metallic-roughness): smooth water
  that reflects its surroundings, sand and rock with a fine grain (tiled normal
  maps), concrete, steel, ice, wood (the material property of bodies, structures
  and DEM elements). The surroundings are generated (`lagoonviz.render.cinematic`,
  cube maps from the direction of each texel), nothing to download. The
  engineering looks (studio, ocean) are unchanged.
- Camera keyframes: the key button on the timeline (or *View → Camera path →
  Add camera keyframe*, Ctrl+K) keeps the 3D view at the current time; the
  slider marks the keys. With the camera path on (the button beside it), the 3D
  view and exported films follow a smooth path through the keys: positions on a
  Hermite curve with the keys' own spacing, rotations on aligned quaternions,
  field of view and zoom alike, easing in and out at the first and last key.
  Keys are saved with the session and written by *Export script*
  (`session.add_camera_key(time, state)`); `lagoonviz.core.camera_path` does the
  interpolation. The easing at the ends is done in time, so the path keeps its
  shape there. *Orbit around the scene* (`session.orbit_camera(degrees)`) makes
  the keyframes of one turn around the middle of the data over the timeline.
- Path tracing with Blender's Cycles (*File → Export animation → Renderer*,
  `lagoon-render --engine cycles`, `lagoonviz.render.pathtrace.path_trace`):
  stills and films of the 3D view in a cinematic look, rendered by headless
  Blender as a separate process; Blender is never opened, and LAGOON never imports
  it. LAGOON writes the scene (triangles per frame with the colours mapped, lines
  as thin tubes, particles as small spheres, the camera of each frame from the
  camera path, materials, the look's surroundings as an HDR panorama, the sun,
  fog around the scene under water); `blender_cycles.py` builds and renders it
  (GPU when there is one, denoised, AgX). Water refracts and lets the sun through
  to the bed. Frames are kept with a key of their data, camera and settings, so a
  stopped film goes on where it stopped. Needs Blender 4.2 or newer. Tuned for
  speed: 64 samples by default (denoised), adaptive sampling, few light bounces
  (light crosses the water sheet once or twice), indirect light clamped. Clear
  water goes to Blender as a closed body of water (surface, walls, floor) that
  refracts, reflects and absorbs light with depth, with fine ripples on it. *File →
  Render still with Cycles* path traces the 3D view as it is.

## 0.4.0.dev0 (phase 4, figure studio)

- Paper figures in the figure editor (phase 4 complete): wave profile panels (η
  along the WSFLINE lines at a chosen moment, FNPF's wave theory dashed) and 3D
  view panels (the view taken as it is, with its camera, time and look, or from
  the top with axes in metres; colour bars under the image) next to time series
  and spectra; a grid of columns in which a panel may span several; panel letters
  (a), (b), ...; per panel a title and notes (text or `$LaTeX$`, placed with
  "x, y: text"). Templates keep all of it; `template_figure` draws it (a view
  that cannot be rendered is left out and named), `render_view` renders the 3D
  view at a moment and camera and puts the session's view back.

- AMR output of REEF3D (FNPF, NHFLOW and SFLOW with `A 270`): `REEF3D_<SOLVER>_AMR`
  (a `.vtm` per output, a `.vtr` per rank and patch) opens with the case as an
  *AMR surface* item: the leaf surface (each grid on its node lattice, without
  the cells a finer patch covers), coloured by any field or by `level`, the
  borders of the refined patches drawn over it (dark, white or off), vertical
  scale, dry land hidden, contour lines, the probe from the finest grid. It
  takes the place of the run's free surface, whose level 0 it is.
  `lagoonviz.core.data.amr` reads the files.
- LAGOON format 0.3 (`lagoon-format` 0.3.0): AMR surfaces (SPEC.md section 11),
  the grids of every output one after another (a grid table, node coordinates,
  cell fields), written by `lagoon-convert` (`--outputs amr`) and by REEF3D with
  `P 18` (patch 0013); read by LAGOON, by `Store.amr` and by the ParaView plugin
  (rectilinear grids, covered cells hidden). On a synthetic 4-rank case the
  store is 13 times smaller than the ASCII `.vtr` files.

- *File → Export figure* (Ctrl+Shift+P): the plots, the wave profiles or the 3D
  view as a paper figure, drawn again with matplotlib: journal widths (90, 140,
  190 mm or any), fonts in points, PDF/SVG/EPS (editable text) or PNG/TIFF at a
  given resolution, axis labels with units, $LaTeX$ in labels (mathtext, or a
  LaTeX installation). The 3D view is rendered off screen at the figure's
  resolution, with a colour bar per coloured item; in the 2D top view with x and
  y axes in metres. A preview in the dialog. `lagoonviz.render.figure` does the
  same from Python.
- Fix: range min and max can be typed while the range is automatic (typing
  fixes the range).
- House styles for figures: REEF3D (sans serif, REEF3D blue and red first, thin
  lines), Journal (serif; black first) and Presentation (large text, thick
  lines, a grid); a style sets fonts, line widths and line colours, the dialog
  can still change the font.
- Figure templates: wave spectra of the plotted gauges (Welch, Hann windows; the
  legend gives Hm0 and Tp; a time window to leave out the start), and a
  comparison of a plotted line with measurements from a text or CSV file
  (computed line, measured points, RMSE, bias, correlation r and Willmott's
  skill d). `lagoonviz.core.analysis` has the numbers (spectrum,
  wave_parameters, compare, read_measurements).
- *File → Figure editor*: a figure built by hand. Drag time-series columns of the
  open cases (or of measured files) onto panels, or double-click them; time and
  spectrum panels, stacked; per line a label, colour, line style, marker and
  width; per panel the y label and range, a log axis, a time window and the
  statistics against the measured line; the figure's style, width, height, font
  size and title. The preview is drawn again after each change. Figures are kept
  as templates (`*.lagoonfig`, JSON) that name lines by file and column, so the
  same template draws the same figure for another run of the set-up; measured
  files next to the template are kept relative. `lagoonviz.core.figure_template`
  and `lagoonviz.render.figure.template_figure` do the same from Python.
- The recorded script (*File → Export script*) has the outputs too: each figure
  (Export figure, the figure editor), film and screenshot made is written into
  it after the change it followed, as the `lagoonviz.render` calls that make it
  again. They run when the script is run (`python script.py`), behind
  `MAKE_OUTPUTS`, so `Session.from_script` and `lagoon-render` still only rebuild
  the scene. In the script a screenshot is the 3D view, and films have no time
  and colour-bar overlay (drawn with Qt). `lagoonviz.render.script` writes the
  code; `Session.record_output` keeps it.

## 0.3.0.dev0 (phase 3, LAGOON format)

- LAGOON format 0.2 complete: `lagoon-format` 0.2.0 (SPEC.md 0.2, no longer a
  draft) and a user guide, `lagoon-format/GUIDE.md`: making stores (REEF3D's
  `P 18`, the converter), opening them (LAGOON, ParaView, Python, zarr-python) and
  the `P 18` text for REEF3D's user guide.
- LAGOON format 0.1 (`lagoon-format/SPEC.md`): one Zarr v3 store per run, one
  group per output (volume, free surface, bed), one block per MPI rank. σ-grids
  keep x, y and σ once and per output only the fields and two heights per column
  (`z_bed`, `z_surface`; `z_offset` for levels off their σ, empty for FNPF and NHFLOW).
  Chunks: shards of 16 outputs per block, inner chunks of one output, a few levels
  and a 64 × 64 tile; byte shuffle + gzip level 1. Readable with zarr-python and
  xarray.
- `lagoon_format.store` (reader and writer) and `lagoon_format.zarr3` (a small
  Zarr v3 implementation, NumPy and the standard library only; parallel chunk
  decoding).
- `lagoon-convert CASE` and *File → Convert to LAGOON format*: the case's VTU/VTP
  output to `REEF3D_<SOLVER>.lagoon` in the case folder; again: appends the outputs
  written since. Fields are kept bit for bit, heights to float32 precision. On the
  FNPF and NHFLOW reference outputs: 5.0× and 5.6× smaller than the VTU/VTP files.
- CFD volumes (format 0.1, section 3a): Cartesian grids keep the heights of their
  levels once (`z`); MPI blocks may split the levels too (`start` i0, j0, k0).
  The converter, the app and the REEF3D writer handle them.
- LAGOON format 0.2: time series tables (wave gauges, 6DOF forces and motions, force
  files; one column per quantity, 64k-row chunks) and wave profiles (WSFLINE, one
  array per line). The converter writes them (again when a file changed; profiles
  appended); a case reads them from the store while the text files are unchanged.
- LAGOON format 0.2: floating bodies (section 9). The mesh of each 6DOF body is
  kept once and per output only its rigid motion (translation and quaternion),
  fitted to the written vertices and checked (to 4 float32 steps); a body that
  does not move rigidly keeps its vertices per output. On the NHFLOW ship run
  (75,696 vertices, 697 outputs): 1420× smaller than the VTP files, within 0.06 mm.
- Particles: CFD's CPM particles (`REEF3D_CFD_CPM_Particle`, a file per MPI rank)
  and NHFLOW's particles open as a *Particles* item, drawn as points coloured by
  a variable (velocity by default), with their own output times (TimeValue).
- The converter leaves out surfaces that are not on a grid (CFD's free surface is
  an iso-surface) and says so, instead of building a grid of holes from them.
- 4K films: from 2560 px wide the Mac's video hardware (VideoToolbox) encodes
  when it is there, else libx264 "veryfast"; frames go to ffmpeg as RGBA without a
  copy; the status bar reports frames per second, the encoder and where the time
  went.
- Vertical scale: the Sync button sits under the slider (the slider has the whole
  width); the slider goes where it is clicked, as the timeline. Playback up to
  240 fps (a precise timer, reading a quarter second ahead).
- Particle trails: *Trail [outputs]* on Particles and DEM elements draws the path
  of each particle (DEM: each element's centre) over that many outputs before,
  joined by its `id`, in the item's colour; from the store or the VTK files.
- LAGOON format 0.2: particles and objects (section 10). CPM and NHFLOW particles,
  DEM elements, ice floes, rod trees and mooring lines are particle sets: the
  points of every output one after another, cells stored again only when they
  change; the converter writes and appends them, the app reads them from the
  store. Mooring lines of the NHFLOW ship run: bit for bit, 5.2× smaller, an
  output read in 0.1 ms instead of 1 ms.
- The theme switch repainted every widget of the application, including those of
  closed windows Qt was deleting, which crashed the test suite now and then; it
  now repaints only its own window.
- ParaView reader plugin (`lagoon-format/paraview/lagoon_reader.py`): a store's
  outputs as structured grids per block, bodies as triangle meshes, particles as
  points (vertex cells) and objects as polygons or lines, with all their times.
- A case with a store opens it instead of the VTU/VTP output it was made from
  (unless the VTU/VTP output is newer); a store also opens on its own. Volume
  faces, vertical slices and probes read only the levels and tiles they need.

## 0.2.0.dev0 (phase 2, domain tools)

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
- 2D from the front and the side: in 2D, *Top*, *Front* and *Side* (keys 7, 1, 3; Ctrl
  for the opposite side) turn the 2D view to that side; from the top everything is
  drawn flat as before, from the front or side it is an orthographic elevation of the
  shapes (with the vertical scale), still without rotation. *3D* leaves 2D. The side
  is saved with the session and filmed by the export (`lagoon-render --view 2d-front`).
- Probe time series: placing the orange dot on the free surface plots eta there over
  the whole run (at the nearest grid point; the coloured variable on other surfaces),
  as an orange "Probe" row in the plots, read in the background with a few bytes per
  frame. A box above the plots chooses what is shown: all plots, the probe, or one
  time series item.
- "Vertical exaggeration" is called "Vertical scale".
- Ice floes (`REEF3D_FNPF_ICE`, prisms of each floe) and rod trees (`REEF3D_RodTree`,
  corals or vegetation as rods with a radius, drawn as round tubes) are shown in 3D and
  move with the timeline, in ice and wood colour; their values per floe or rod (speed,
  contact force, colony, hydrodynamic force, ...) can colour them. Both write at their
  own interval: their times come from the files' `TimeValue`. The VTK reader reads
  line data and gives cell data to the points of each cell.
- DEM elements (`REEF3D_DEM_VTP`, one ASCII file per output with every element's
  triangles, velocity, id and whether it is resolved) are shown in 3D and move with the
  timeline; opening a case adds them, in rock colour. The DEM writes at its own
  interval (E 15): without a run log their times come from the files' `TimeValue`.
- Floating bodies (`REEF3D_*_6DOF_VTP`), mooring lines with their tension
  (`REEF3D_*_6DOF_Mooring`, legacy VTK) and porous or forced structures
  (`REEF3D_*_VRANS_VTP`, `_FORCING_VTP`) are shown in 3D and move with the timeline;
  opening a case adds those of the shown run. Bodies take a material colour.
- Click on a surface in the 3D view to read its values at that point; the marker
  stays (same size at any zoom, drawn on top) and its values follow the timeline. The
  values above the view start with the marker's orange dot.
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
- Full screen viewer (View → Full screen viewer, Cmd+Shift+F): only the 3D view and
  the player on the whole screen; Esc or "Exit full screen" brings the window back as
  it was.
- Automatic colour range: shown in the min/max boxes (read-only while automatic);
  found from frames spread over the run for surfaces (the first frame is often calm
  water); switching it off keeps the colours (the fixed range starts at the automatic
  one); switching it on finds it again.
- Reload (toolbar, File → Reload case, Cmd+R / F5): reads the case folders again,
  keeping items, time and camera; at the last frame it moves on to the new last
  frame. File → Follow running simulation reloads by itself when the output folders
  change (checked every 5 s, not while playing).
- LAGOON reads the run log when a case has one: every frame gets its simulation
  time from the log (no `P 30` needed; free surface and volume written at different
  intervals line up), files are assigned to the run that wrote them (a restart in
  the same folder is split by run, not guessed from file times), runs are labelled
  with their start, and the timeline steps through the logged output times. Cases
  without a log open as before. A small case written by the patched REEF3D (Stokes5
  tutorial with a restart) is a test case.
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
- Colour map and *Colour by* are independent again: choosing one never changes the
  other. Coloured by "none", a surface takes one colour of its colour map (sand: sand,
  greys: grey, white: white, others their middle colour), so a plain bed can still be
  recoloured; the range settings are greyed out. Bodies and structures take their
  material's colour.
- New colour map "white": grey (low) to white (high) when colouring by a variable, pure white
  when coloured by nothing.
- Regression suite for agreed behaviour: behaviour tests, text snapshots of the window
  (menus, toolbar, panels, properties of every kind) and reference images of the 3D
  view per platform; `pytest --update-snapshots` (`pixi run snapshots`) accepts an
  intended change. `lagoon-render --view 2d` renders the 2D view.
- Closing LAGOON in the full screen viewer keeps the panels for the next start.
- *Frame all* fits tall scenes (strong vertical exaggeration) in a wide window; they
  were cut off at the top and bottom. Flat scenes are framed as before.
- Playing with time series plots shown is fast again: the plots drew every line again at
  each frame (330 ms for the gauges of a long run, 3 fps); lines, grid and labels are now
  drawn once, and each frame draws only the cursor and the legend values.
- The timeline slider no longer jiggles while playing: the time label beside it took
  the width of each new text, so the slider changed width with every frame.
- Dragging the timeline slider while playing: playing goes on from where the slider
  is let go, at once. Frames the slider passed over are no longer all read one after
  the other (they queued up, and playing waited for them); a view asks only for the
  frame it shows now. Holding the slider pauses playing.
- The timeline slider goes where it is pressed and follows the mouse until let go,
  also while playing (pressed beside the handle, Qt's slider stepped towards the mouse
  again and again while playing moved it away: it jumped back and forth). Scrolling
  over the slider (two fingers on a trackpad) no longer moves the time.
- Opening a case shows its free surface in the properties panel (else its bed),
  not the last item added.
- Vertical exaggeration as a slider (log scale, 0.1 to 100, plus a number box) with a
  *Sync* button, on every item that has one and on the case. Sync on (the default):
  any of these sliders sets one value for the whole case, and free surface, bed,
  bodies and gauges all scale about the case's still water level. Sync off: each item
  has its own, starting from the case's value; one undo step either way.
- Bathymetry: *Hide wet land* leaves out the bed under still water, so only dry land
  shows (by the still water depth REEF3D writes with the bed).
- Rotation goes all the way round (pygfx stopped 1 degree short of the poles), always
  about the centre of the view. After the top view the scene no longer tips over when
  turned (the top view's "up" had become the orbit axis).
- Streamlines can also lie in their seed plane (*Lines in: seed plane*, the part of the
  flow across the plane left out, as on a slice) or on the free surface (its
  horizontal flow, from the free surface output, dry land left out). *Add →
  Streamlines* with a slice selected starts in its plane, with the free surface
  selected on the free surface.
- "Bathymetry" is called "Bed" (new items, menus, toolbar); sessions and scripts keep
  working.
- Streamlines through volume output (*Add → Streamlines*): lines along a vector field
  (the velocity) from seeds spread over a plane (x, y or z, through the middle at first),
  traced forwards and backwards until they leave the water, coloured by any variable
  (speed by default). REEF3D's volumes are columns (fixed levels for CFD, σ-levels for
  NHFLOW and FNPF), which makes finding points fast without a search tree; the volume
  of every MPI rank is joined first. Dry columns stop the lines. How the points form
  columns is worked out once per grid; later frames only pick their values, so
  playing with streamlines keeps up.
- Iso-surfaces of volume output (*Add → Iso-surface*): the surface where a variable has
  a value, e.g. REEF3D's level set `phi` at 0 (the free surface of a CFD run), a
  pressure or a speed; coloured by any variable, played like the rest. A new one
  starts on `phi` = 0 if the volume has it, else on the middle speed. Cut with
  marching tetrahedra like slices, with shared points merged and every triangle
  facing the rising field, so it shades smoothly (mixed windings shaded in stripes). At
  a time when the field does not reach the value, nothing is drawn.
- Vector arrows on the free surface and slices: *Vectors of* (e.g. the velocity), about
  how many arrows (spread evenly over the surface, also over the height of a vertical
  slice) and their length (1: the longest reach the next arrow; the scale stays put
  while playing). Vertical parts are exaggerated like the surface; in 2D the arrows
  lie flat on top. Filmed by the export too.
- Contour lines on the free surface, the bed and slices: *Contour lines of* (any
  variable, e.g. the bed's elevation or the surface's eta) and about how many lines.
  Levels are round numbers over the variable's automatic range and stay put while
  playing; the lines follow the surface as shown (exaggerated, or flat in 2D), dark on
  the studio look, light on the ocean look, and are filmed by the export too. Whole-
  number properties are edited without decimals.
- 2D view (button next to *Frame all*, key 2): everything drawn flat (the free surface
  as a plane coloured by its values over the bed, objects between, gauges and profiles
  on top; no shadows), seen from straight above, orthographic, no rotation; every drag
  pans, the wheel zooms. While it is on, 3D, front and side are greyed out. It is saved
  with the session and in scripts, both views follow it when comparing, and exported
  animations and `lagoon-render --view 2d` film it the same way.
- Studio look uses the ocean look's lights (headlamp, more sheen); the two looks now
  differ only in background. The sun is strong enough again for visible shadows.
- Ctrl+C in the terminal closes LAGOON properly (it used to close only the 3D view
  and leave the terminal waiting): the 3D view's library installed its own handler
  once its loop had started; LAGOON's is now put back.
- Move the view: Ctrl (or Cmd) + drag, Shift + drag or right drag; arrow keys move
  it by a tenth (Shift: a third).
- Fixed: the background loader could deadlock when a frame finished before its
  completion callback was registered (playing, prefetching).
- Faster animation export: frames are read several ahead in three threads and encoded
  while the next one is drawn (reading, drawing and encoding used to take turns), and
  H.264 uses the "fast" preset at the same quality setting. The log tells where the
  time went.
- Exported animations frame what the 3D view shows: the size has the view's shape
  (Full HD, HD or 4K wide); a 16:9 film of a differently shaped view showed more or
  less at the sides.
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

## 0.1.0.dev0 (phase 1, viewer)

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
