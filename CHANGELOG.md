# Changelog

## Unreleased

### Derived quantities

- Colour by (contour lines, vectors, slices, iso-surfaces) quantities worked out
  from the output: on free surfaces and AMR surfaces with a water depth
  `water_depth`, `froude`, `unit_discharge` (h·u, a vector), `hazard` (h·|u|) and
  `bed_shear` (Manning, the case's *Manning's n*); `vorticity` on surfaces;
  `vorticity` (the curl) and `q_criterion` through volumes, worked out in their
  cells before slicing, so iso-surfaces of Q show the vortices (VTU and LAGOON
  stores).
- The case's *Formulas*: your own, `name = expression; ...`, numpy expressions of
  the variables (`velocity.x`, `eta`, the built-ins, earlier formulas). Undo, the
  saved session and the script keep them; the case's information lists those that
  fit no output.

### The sea anywhere, rivers and tide gauges

- The Sea tab has a *Source*: MET Norway (NORA3, NorKyst, as before; Norwegian
  waters) or **Open-Meteo** (anywhere): ERA5 waves and wind from 1940 for the storms,
  the conditions and a JONSWAP wave generation (B 85 2, B 93, B 105), and
  Météo-France's sea level with the tide and its currents from 2022 for the
  conditions and the boundary hydrographs (the depth from the case's bed).
  *Automatic* takes MET Norway in its waters. An Open-Meteo API key for commercial
  use.
- *River (GloFAS)*: a river's daily discharge (Copernicus GloFAS through Open-Meteo,
  from 1984) at the corridor's upstream end or the inflow side, hourly over the
  window, as `hydrograph.dat` (ioFlow, B 60 2) or a side's `hydrograph_Q_<n>.dat`.
- *Tide gauges*: the nearest gauges of Kartverket, NOAA CO-OPS and the IOC network;
  their observed and predicted water level and the surge over the window as a time
  series to compare a run with, or the observed level as the water level at sides
  of the domain (`hydrograph_FSF_<n>.dat`, B 422).
- In a script: `find_storms(..., source=...)`, `write_waves`, `river_discharge`,
  `find_gauges`, `gauge_levels`, `gauge_boundaries`.

### A new SFLOW run, not the one before

- A LAGOON store REEF3D started on top of an earlier run's (SFLOW before REEF3D's
  fix of 2026-10-15: no volume output, so the store was not moved aside) shows the
  latest run only: an output the run log has from an earlier run into the store,
  and not from the latest (e.g. the earlier run's AMR surface), is left out, and
  the others count as the latest run's.

### The wheel scrolls the panels

- Over a number box, a choice list or a slider in the Properties and Set-up panels
  (any scrolled panel) the mouse wheel or trackpad scrolls the panel; the field
  takes the wheel only once clicked into. Scrolling down a panel no longer changes
  a value or an option by accident. The time slider keeps the wheel.

### Items added for the data chosen

- *Add → Iso-surface*, *Iso-volume*, *Free surface*, *Bed*, *Slice* and
  *Streamlines* list *Automatic* and every output of the open cases the item can
  show (a volume; for iso items also the free surface, the bed, an AMR surface):
  the item is made of the one chosen. The toolbar buttons have the same list on
  their arrow. In a script `session.add_of(kind, dataset, case=...)`.

### AMR: grid lines and patches shown or hidden

- An AMR surface has **Grid lines** (off, dark, white): the edges of the cells of
  every level, beside the patch outlines in their own colour.
- **AMR patches** off shows the base grid (level 0) alone, whole, without the
  refined patches; on again, the patches come back.
- Grid lines and patch outlines lie on the surface (they floated over it in a
  shallow case): lifted by 0.03 % of the model's size, not 0.25 %.
- SFLOW's AMR surface has the **velocity** (magnitude, x, y, z) made of the u, v, w
  it writes; also for a point's time series.
- SFLOW's AMR surface hides dry land by its wet cells (`wetdry`); its `WL` was
  taken for a water depth and kept dry land.

### Another case's geo.dat and place left out

- A case whose control.txt does not read geo.dat (no `G 10 1`) opens without the
  *geo.dat points* and without a background map: a geo.dat and georef.json left in
  the folder by another case (e.g. a home folder used for several cases) are not
  this case's. A map kept for the folder (`LAGOON_map.json`) still opens.

### Multi-block grids

- The Set-up panel has a **Blocks** tab: blocks of curvilinear cells that meet along
  their faces, node on node, for SFLOW and NHFLOW (`G 50 1`, DIVEMesh `R 20 1`).
  Start one block over the domain, draw one, or make the river corridor block 0;
  then, in the 2D top view, drag corners and points, cut edges and add blocks
  outside an edge. Cells go on through the blocks so the faces always match; each
  block is meshed by transfinite interpolation and Winslow smoothing, its wall nodes
  sliding along the walls.
- The faces of a block are walls or open: a discharge in through a -i face, a water
  level at a +i face (`G 54` for the blocks 1.., `B 60`, `W 10`, `F 62` for block 0).
- The tab lists the rules the grid breaks (faces that do not join up or do not
  match, folded cells, block 0's interfaces on part of a face, too few cells behind
  an interface, open faces in the wrong place, ...) and the set-up marks them where
  they are; it draws the blocks, their grids and the quality of the cells. **Make
  grid** waits for the errors, then writes `blocks.dat`, control.txt and ctrl.txt.
- Kept in `blocks.json`; a **Multi-block grid** item of the scene. In a script
  `session.start_multiblock()`, `extrude_block()`, `move_block_vertex()`,
  `set_block_face()`, `make_multiblock_grid()`.
- The catalogue knows `G 50 1`, `G 54` (REEF3D), `R 20` and `R 21` (DIVEMesh).
- Without a GPU, the set-up's placeholder view no longer stops on *North up*.

### Tidy river grids

- The River tab's grid can have its **cross-sections across the river**: straight
  across the line halfway between the banks where the banks allow, the cells along
  each bank evened out (no cell more than 1.2 times its neighbour, none shorter
  than 0.6 or longer than 1.6 times an even one), so it is neither skewed nor
  squeezed. A corridor along the thalweg starts so; **Tidy the grid** (under Follow
  thalweg) makes it so again after dragging, with a little smoothing;
  *Cross-sections* chooses it or the equal lengths along both banks of before.
- A corridor's grid of more than 400,000 nodes was not drawn (so the grid vanished
  after Make grid, the curvgrid.dat one no longer drawn beside it): it is now, up
  to 5 million nodes (every n-th line).
- One grid at a time: while there is a corridor the set-up draws its grid only,
  not curvgrid.dat's as well (which drew two grids after Make grid, and after a
  session was opened again); the River tab says when curvgrid.dat is another grid.
- **Follow thalweg** without F 60: the water at 0 where the bed's heights reach
  below and above 0 (a coast, an estuary), and no bank farther than 2.5 times the
  river's usual half width from the thalweg. On the Elbe at Hamburg the corridor
  follows the river (about 500 m wide) instead of the marshes (12 km).
- The bed from geo.dat is found in a second (only the points round the corridor;
  geo.dat on a regular grid read on it; very many points by their nearest): Make
  grid and Follow thalweg no longer hang on a large geo.dat.

### Turning the model on the map as you look

- The Map tab has **sliders** for the georeference's rotation (about the middle of
  the view) and for turning the domain (about its centre): while one is dragged the
  map and north turn at once (the map made before, turned; nothing fetched or
  written), and the turn is written when it is let go. From the top the view keeps
  north up, so the map stays and the model turns. **North up** sits beside the
  rotation too. In a script: `session.set_rotation(40.0, about=[x, y])`, and
  `session.try_rotation(40.0)` shows a rotation without keeping it.
- *Draw on the map*: the domain's edges and corners can be dragged (the cursor
  shows them); it stays a rectangle, snapped to the cell size.
- After the georeference changes, the map made before is shown turned to the new
  one until the new map is made, instead of the old one as it was.

### River corridors and their curvilinear grids

- The Set-up panel has a **River** tab: lay a river corridor and drag the points of
  its two banks in the 2D top view (12 per bank to start, 4 – 60). The banks are a
  spline through the points (centripetal Catmull-Rom) or a smoother B-spline the
  points pull on. Start one along the domain, draw it over an area, or let it
  **follow the thalweg** of the bed (geo.dat): along the deepest channel from the
  inflow to the outflow side, its banks at the water's edge plus a margin - then
  drag the points. The banks and their points are drawn in yellow.
- The corridor makes the curvilinear grid REEF3D's SFLOW and NHFLOW read with
  `G 50 2`: cells along and across, Winslow smoothing for bends, the bed sloping
  from the upstream end or from geo.dat. The set-up draws the grid as it changes and
  says if cells fold; **Make grid** writes `curvgrid.dat`, `G 50 2` and DIVEMesh's
  logical box. Kept in `corridor.json`; in a script `session.start_corridor()`,
  `move_corridor_point()`, `set_corridor_options()`, `make_corridor_grid()`.
- The corridor is a **River corridor** item of the scene: *Add → River corridor*
  lays one out, *Remove* takes it away (with corridor.json), *Reset* lays it out
  again as it was made (along the domain, over the area drawn or along the
  thalweg), dropping the dragged points. Its points and grid can be switched off.
- A case with `G 50 2` and its `curvgrid.dat` is drawn where its nodes are: the
  domain, its sides and the grid (*Curvilinear grid*), not DIVEMesh's logical box.

### Porous structures and solids in the set-up

- The Set-up panel has a **Porous** tab: add, draw and change the porous structures
  (VRANS) of a case. REEF3D::CFD: boxes (B 270), vertical cylinders (B 274),
  wedges along x and y (B 281, B 282) and plates (B 291), each with its porosity,
  d50, α and β. REEF3D::NHFLOW: boxes, cylinders, spheres and wedges (B 210 –
  B 218), one material for all of them (B 201, added with the first one). *Add*
  puts one in the middle of the domain from the bed to the still water, *Draw*
  drags its plan in the 2D top view; the selected one lights up in the 3D view.
  In a script: `session.add_structure("B 270", [...])`.
- A **Solids** tab beside it does the same for solids: DIVEMesh's boxes, cylinders,
  spheres and wedges (S 10, S 32, S 33, S 51, S 61, S 62 in control.txt, meshed by
  Run DIVEMesh) for every module, and NHFLOW's direct forcing solids (A 581 –
  A 588 in ctrl.txt). They reach from the bed to the domain's top.
- The set-up draws wedges (B 217, B 218, B 281, B 282, A 587, A 588) as wedges,
  and B 274 and B 291 too; the ctrl.txt tab names the values of B 270 – B 291.

### The frame number typed

- The player's frame number can be typed: click it, type the frame and press
  Return (`t = 12.50 s  (frame [25] / 120)`); a number past the end goes to the last
  frame.

### Following a running simulation

- The result no longer vanishes when reloading while REEF3D writes its LAGOON store:
  an output read just as REEF3D appended the next one to the same shard came back
  as missing values (the surface drawn as nothing until the time moved). The
  shard is now read again once its new index is written (milliseconds); one left
  half written is reported as an error instead of drawn as nothing.
- **Reload** (and following a run) keeps the results on screen until each item's
  new frame has been read: the 3D view used to be cleared at once, and stayed empty
  while a large run's frames were read again (for good if a read failed).

### Only the wet free surface

- **Hide dry land** now also works for NHFLOW and SFLOW (their free surface and AMR
  surface): it leaves out the surface over dry cells (`wetdry`), so only the wet
  free surface shows and the bed is seen on dry land. It is on by default, as it
  was for FNPF; switch it off in the item's properties to see the whole surface.

### SFLOW and curvilinear grids in LAGOON stores

- **Curvilinear grids** (REEF3D's `G 50`, NHFLOW and SFLOW) in LAGOON stores:
  format 0.5 keeps where every grid point lies (`xc`, `yc`, SPEC 2.2); LAGOON draws
  the free surface, bed and volume there, cuts slices through the grid's cells,
  and probes the nearest point. The ParaView reader too. The converter leaves out a
  curvilinear run's VTK output (it is on no grid lines): REEF3D writes its store.
- REEF3D writes **SFLOW's** free surface and bed into the store (`P 18`).

### Images of the 3D view

- **File → Export image…**: the 3D view alone, drawn again off screen at the size
  you choose (as on the screen, twice as sharp, Full HD, 4K or 8K wide, a width in
  pixels, or a width in cm at a print resolution), lines, markers and text keeping
  their share of the image; the time and colour bars on it if you like; PNG, JPEG
  or TIFF (the dpi kept). In scripts: `render_image(...)`; `render_session` and
  the films' `OffscreenView` take a `scale`.

### The bed next to the coast

- **No flat 1 m next to the coast**: a download's bed between the coastline and
  the sea source's trusted depths (1.5 of its cells out) is made again: *Near the
  coast* in the Map tab, *A beach profile (Dean)* (A x^(2/3), bent to meet the
  sea's depths; the default), *The smoothest bed* (creeks, harbours, mudflats) or
  *At most 1 m deep* as before. *Trust the sea from* also makes ETOPO's and GEBCO's
  flat shallows again (`make_geo_dat(..., nearshore=, dean_a=, trust_depth=)`).
- **Files of one's own over a download**: surveys, chart soundings (their zero
  below mean sea level), intertidal or coastal terrain models, laid over the
  download and blended in at their edges (*Over it*, `overlays=`).
- **ICESat-2's nearshore soundings**: NASA's ATL24 files, or downloaded for the
  domain with an Earthdata login (*ICESat-2 soundings*, `icesat2=True`), guide the
  profile near the coast. Reading them needs h5py (now in `lagoon-viz[geo]`).
- **A shallow coast is the sea, not a polder**: with no sea 10 m deep in the domain
  (Mumbai's west coast), the sea was taken for low ground behind dikes and kept at
  Copernicus' 0 m; Copernicus' water at 0 m where the sea source has the sea now
  joins it to the sea.
- A geo.dat made before is made again with the 1 m of before (*Make geo.dat
  again*); its recipe keeps the new settings.

### Satellite maps outside Europe

- **Sentinel-2 (satellite)** (was "Sentinel-2 2017"): the 2017 mosaic has images of
  Europe only (elsewhere a white land and a blue sea, so Mumbai showed none); outside
  Europe the map now comes from the 2016 world mosaic (also CC BY 4.0). Sessions,
  scripts and case folders with the old name keep working.

### Finding what is slow

- **Help → Record a performance report**: counts what every thread of LAGOON does
  while you use it; stopped, it saves a plain text report (per thread, where the
  time went), to send when LAGOON is sluggish.
- macOS: LAGOON started from Terminal (`setup_lagoon.command`) had no menu bar and
  no Dock icon and ran slowed down as a background process; it is now a regular
  application, brought to the front, with App Nap off while it runs.

### geo.dat as fine as the data

- **Resolution** in the Map tab: geo.dat as DIVEMesh's cells (B 1), *the finest the
  data has* (Kartverket's 1 m in Norway, Copernicus' 30 m on land, EMODnet's 115 m;
  a file's own cells or point spacing), or every so many metres; the tab says what
  the finest is over the domain and about how many points it makes. *Most points*
  (4 million by default, 0: no limit) keeps geo.dat to a size of your choice.
  In scripts `make_geo_dat(..., spacing="finest", most_points=...)`.

### Iso-surfaces and iso-volumes of the right data

- **Made of the data selected**: with a volume (or a slice or iso item of it)
  selected, a new iso-surface, iso-volume, slice or streamlines is made of that
  volume, not the case's first; with the free surface or the bed selected, an
  iso-surface or iso-volume of that surface. *Dataset* lists what each can show.
- **On the free surface and the bed**: an iso-surface is the line where the
  variable has the value (eta = 0: the still water line; depth contours), an
  iso-volume the part of the surface in the range; in 2D just over their surface.

### The map with the results

- **A case opened again has its background map**: the map is kept with the case
  (`LAGOON_map.json`: map, margin, placement, opacity, shown or not), so its
  results opened anew show it too. A georeferenced case that keeps none opens with
  the place's best map; a map taken away stays away.

### The coast of downloaded bathymetry

- **The coast where Copernicus' 30 m land has it**: *Best available* no longer lets
  EMODnet's 115 m or ETOPO's and GEBCO's 450 m cells smear the coast. Land takes
  Copernicus' heights, the sea the coarse depths held below zero next to the coast
  (0.1 m, deeper by 2 cm per metre, to 1 m), polders behind dikes their own heights.
- **EMODnet on mean sea level**: *LAT below MSL* in the Map tab (and
  `make_geo_dat(..., lat_below_msl=...)`) moves EMODnet's depths, measured below
  the lowest astronomical tide, onto the land's zero.

### geo.dat follows a turned model

- **geo.dat made before the model was turned** (or moved on the map, given another
  CRS or a larger domain) no longer fits: it is in model coordinates, and DIVEMesh
  meshed the bed as it lay before. LAGOON keeps how geo.dat was made
  (`LAGOON_geo.dat.json`); the Map tab says when it no longer fits, *Make geo.dat
  again* makes it from the same source and settings, and *Run DIVEMesh* asks to
  do that first (`session.geo_dat_state()`, `session.remake_geo_dat()`).

### Maps away from Europe

- **North up is true north**: the view turns the map north up wherever the case
  is, taking in its CRS's turn (grid convergence) at the middle of the domain;
  *North up* now turns a view whose map was not north up (e.g. Mumbai in a
  Norwegian UTM zone). A model lined up with a UTM grid away from the zone's
  meridian lies a few degrees turned in the north-up view, as it does on the map.
- **A place far from the case's CRS gets its own UTM zone** (WGS 84 outside
  Europe): Mumbai found for a case set up in Norway no longer keeps UTM 33, whose
  map there is turned and stretched.

### Legends

- **Each item's colour bar**: on or off, its title (`$\eta$ [m]`), how its numbers
  are written and how many ticks it has (*Colour bar*, *Bar title*, *Bar numbers*,
  *Bar ticks*); the 3D view, films and figures follow them.
- **View → Legend…**: where films put the colour bars and the time (a corner, or no
  time), the bars' length and thickness, the text size, a box behind them; the 3D
  view's bars on or off. Kept with the session (`session.set_legend`). Films made
  by scripts get the time and the colour bars too (`write_movie(..., legend=True)`).
- **Figure editor**: a panel's legend inside, outside right, above, below or off,
  its columns, text size and frame; lines left out of it; a 3D view's colour bars
  under, right of or inside the image, and their length.
- **Plots panel**: line names (`P1 = gauge 1`), the legend above the plot, in it or
  off, values at the cursor or not; the figures made from the plots use the names.

### The sea at the case (NORA3, NorKyst)

- **Sea tab** in the Set-up panel: MET Norway's hindcasts at a georeferenced case,
  read from thredds.met.no. *Find storms* lists the largest wave events of NORA3
  (from 1959) at the wave generation side; one chosen sets the time window.
- **The conditions as time series**: NORA3's Hs, Tp, direction and wind and
  NorKyst's water level and current (from 2017-02-20) over the window, as items,
  their tables in the case's `LAGOON_Sea` folder (directions also in the model's
  frame, for B 105).
- **Wave spectrum for REEF3D**: NORA3's 2D spectrum at the storm's peak as
  `spectrum-file-2d.dat`; ctrl.txt's wave generation set to it (B 85 11, B 92 31,
  B 93, B 105, B 130 1).
- **Boundary hydrographs**: NorKyst's water level and discharge along the sides of
  the domain as `hydrograph_FSF_<n>.dat` and `hydrograph_Q_<n>.dat`, with the
  patches in ctrl.txt (B 440, B 421, B 422).
- In scripts: `find_storms`, `sea_conditions`, `write_spectrum`,
  `write_boundaries`.

### Print-outs and several cases

- **Slices for every case**: with several cases open, new slices, iso-surfaces,
  iso-volumes and streamlines cut the case of the selected item (else the first
  case with a volume), not always the first case; putting a slice on another case
  moves it into that case's volume when its plane would miss it.
- **NHFLOW and FNPF set-up grid as ParaView shows it**: REEF3D scales DIVEMesh's
  heights to σ and stretches them from the bed to the still water (F 60); the
  *DIVEMesh grid* item, its partition planes and the domain are drawn there, the
  levels spaced as DIVEMesh spaced them, whatever heights B 10 gives.
- **REEF3D's names**: wave profiles are *Free surface lines* (P 52); the time
  series are named as the User Guide names the print-outs: Velocity probe, Point
  probe, Pressure probe, Force box, Morison force, Run-up gauges, Cylinder run-up,
  Bed level gauges, Wave gauges (theory), ...
- **More print-outs read**: the probes' tables headed `t` (NHFLOW's velocity and
  pressure probes, FNPF's, CFD's point probes, SFLOW's depth-averaged probes),
  CFD's pressure probes (no header), NHFLOW's run-up gauges (no time in the
  header), the bed level and bed shear stress gauges (`REEF3D_*_Sediment`), the
  wave theory's time series (`REEF3D_Log-Wave`), free surface lines along y (P 56,
  `WSFLINE_Y`) and NHFLOW's time-averaged free surface lines (P 146, P 147).

### Large cases

- **Streaming**, a fourth *View → Memory use* mode for very large cases (100
  million cells and more): frames are read as they are shown and few are kept;
  while playing or dragging the timeline, free surfaces from a LAGOON store of
  more than 2.5 million points are read at every 2nd, 3rd, ... grid line, and at
  full detail as soon as the time stands still. Films and stills are always at
  full detail. `LAGOON_MEMORY=streaming` for `lagoon-render` and scripts. Opening
  a case of more than 5 million points a frame in another mode, the status bar
  suggests it.
- **Performance (in memory)** reads the whole play range ahead while playing
  when it fits its budget, so playing runs from memory from the second time round.
- **Reading ahead within the budget**: playing and films never read further ahead
  than the frame cache holds; frames of 300 MB read eight ahead into a 1 GB cache
  used to push out each other before being shown, and were read again.
- **Switched-off items are not read**: the volume a case opens with (switched
  off) was read with every frame while playing and filming.
- **Faster frames from LAGOON stores**: the normals of a surface in single
  precision on several threads (10x faster), points and triangles made in place,
  the blocks of an array decoded without threads within threads, fields that are
  the same in every output (FNPF's still water depth, its coastline) read once,
  and the variables read while the points are made. A frame of the 1.85 million
  point FNPF case: 0.44 s → 0.23 s; of a 10 million point surface: 2.5 s → 0.8 s;
  the first frame 8 s → 2 s (the colour range sampled from tiles, not whole
  outputs).
- **AMR with thousands of patches**: the cells a finer patch covers are found
  from the overlapping patches only, the normals from each grid's slopes: a frame
  with 3000 patches 20 s → 1.7 s.
- `tools/benchmark_large.py` writes a synthetic store of any size and times
  opening, reading, showing and playing it.

### Fixes

- **Only the case's own solids**: a `REEF3D_Solid.vtp` another case's DIVEMesh
  run left in the folder (e.g. a monopile case run before an FNPF case in the
  home folder) is no longer opened as a structure. The solids are passed over
  when `control.txt` has no solid commands, when DIVEMesh has run since without
  writing them, or when they lie outside the domain of the run shown (run log);
  solids older than `control.txt` are built from it. DIVEMesh's left-over bed
  likewise.
- **No crash when opening a large case**: Python's garbage collector ran in a
  loader thread while the window deleted a panel's widgets and walked a half-deleted
  Qt object (a segmentation fault in `SbkObject_tp_traverse`). It now runs on the
  GUI thread only.

### Rendering

- **North in 3D**: turning a georeferenced model (*Turn*, Shift + drag) turns the
  3D view with it about the model's middle, so the background map stays where it
  is and the model and its grid turn, as in the 2D view.
- **North up** is a button (and ++n++) that turns any view, 2D or 3D, so north
  points up the screen (the model's y without a georeference); it was a switch
  that did something only from the top and for a turned model.
- **White water in Cycles**: *View → White water in Cycles* (none, light, medium,
  heavy) and the export dialog choose how much foam, spray and bubbles path-traced
  stills and films show; medium, the new default, is about half the white water
  of before (heavy). `--white-water` for `lagoon-render`, `foam="light"` in scripts.
- **Dragging the timeline** is smooth again: a time step no longer draws again
  what does not change with time (the bed, solids, geo.dat's points, the set-up,
  the background maps).

### Slices, iso-volumes and grid lines

- **Iso-volume** (*Add → Iso-volume*, a button in the toolbar): the part of the
  volume where a variable lies in a range, drawn closed (the iso-surfaces at both
  bounds and the volume's outside clipped to the range); a CFD run starts with
  phi ≥ 0, its water. `session.add_iso_volume("phi", 0.0, 1.0)`.
- **Grid lines** on the free surface, the bed and the volume (*Grid lines*: off,
  dark, white): the edges of their cells, from the VTP/VTU files or a store's grid.
- **The outer surface of volumes faces outwards again** (the grid lines of slices
  had replaced the cell faces it is made from: two of a hexahedron's faces turned
  inwards, shading them dark).
- **Grid lines** on slices: the edges of the cells the slice cuts (dark or white),
  from the VTU cells or straight from a LAGOON store's grid.

### Colours

- **Strong colours again**: the automatic range is again the 1st to the 99th
  percentile, symmetric around zero when the values change sign (as before 9
  October); *Range from → lowest to highest* gives ParaView's range.
- **Range [%]** in the properties of every coloured item: the automatic range's
  min and max times a percentage (95 %: 0.95 × min to 0.95 × max); *Fit range to
  this frame* uses it too. `session.set_property(item, "range_percent", 95)`.

### Speed or memory

- **Large cases open and play faster again**: grid lines are made only when they
  are switched on (they were made for every surface and volume as it was read: 4 s
  for a 1681 × 1101 free surface, at every opening and reload, and for every frame
  of a store's volume); a store's surfaces take their normals from the grid's
  slopes (ten times faster than from the triangles); the colour range samples a
  store's variable alone, without points and normals.
- **View → Memory use**: *Performance* (frames kept up to a quarter of the
  computer's memory, more read ahead, more threads), *Balanced* (as before) or *Low
  memory* (256 MB of frames, little read ahead, a volume's cells not kept: large
  cases fit). Kept for the next start; `LAGOON_MEMORY` for scripts and
  `lagoon-render`.

### Opening output

- **SEASTATE**: its maps of the sea state (`REEF3D_SEASTATE_VTP`) open as a *Sea
  state* item: Hs on the rainbow map, arrows of the mean wave direction, dry land
  left out; every parameter it writes to colour by, and vectors made from them
  (wave direction, Hs along it, wind, wave force, Stokes transport). Its AMR
  output opens as an AMR surface coloured by Hs, with SEASTATE's levels and times
  (its FieldData in the Piece, the time as TIME).
- **Results opened from the set-up are shown**: opening a case that has results
  while LAGOON is in the set-up (a case set up in LAGOON, then run) switches to the
  results; before, its free surface and bed were opened but not drawn.
- **File → New case** in a LAGOON store (`REEF3D_*.lagoon`) is refused: it wrote
  `control.txt` and `ctrl.txt` into the store.
- **File → Open output file** lists a folder's files grouped into series, as
  ParaView does (`REEF3D-NHFLOW-FSF-..pvtp`, with the number of files, their
  numbers and size); the pieces of parallel output are left out. Opening a series
  opens it with its case.

### Maps and the place of a case

- **A case started on the map**: *File → New case from map…* finds the place (a
  name from Kartverket or OpenStreetMap, or latitude, longitude), starts a draft
  case there with a domain and a map; *Create case…* puts it in its folder when
  it is ready. `session.new_case_from_map(...)`, `place_case(...)`,
  `create_case(...)`.
- **Place search**: a name as typed, else names beginning so, else alike (a
  misspelling); *Nidelva, Trondheim* (name, municipality) narrows it; the places
  near the case come first, towns and fjords before farms.
- **Georeference**: a case's `georef.json` (CRS, origin, rotation, vertical datum)
  says where its model coordinates lie; *Set-up → Map* edits it (*Find* a place
  by name or latitude, longitude), `session.set_georef(...)`, `session.turn_model(...)`.
  LAGOON transforms WGS 84, ETRS89, Web Mercator, UTM and NTM itself, any CRS with
  pyproj (`pip install "lagoon-viz[geo]"`) ([Maps](guide/maps.md)).
- **Background map** item: Kartverket's maps and nautical chart, Sentinel-2
  cloudless (satellite), OpenStreetMap, a tile server or a GeoTIFF of yours, warped
  into the model's frame and drawn under the results and the set-up in the 3D and
  2D views, figures, films and the set-up sketch, with the credit its source asks
  for. Tiles are cached on disk. In the 2D top view the map follows the view
  (finer when zoomed in, the new area when zoomed out or panned).
- **The domain on the map**: drag it in the 2D view (B 10), Shift + drag turns the
  model on the map.
- **Bathymetry from the web**: *Make geo.dat* needs no file: Kartverket's
  national topobathy model (Norway, land and sea), EMODnet's terrain model
  (European seas), the Copernicus DEM GLO-30 (land, the world, 30 m), NOAA's
  ETOPO 2022 and GEBCO 2020 (the world, 450 m) are downloaded for the domain, the finest first;
  `session.make_geo_dat("best")`. With no case open the Map tab starts at
  Trondheim on Kartverket's map.
- **geo.dat points** item: the points of a case's `geo.dat` as DIVEMesh reads
  them, a dot each coloured by height, with their number, spacing, extent and
  heights in the properties; a set-up item with a check box of its own in each
  mode, drawn again when *Make geo.dat* writes the file. (The *Case set-up* item
  no longer draws them in its *Bed* group.)
- **Bathymetry to geo.dat**: a GeoTIFF, ASCII grid or x y z soundings interpolated
  (TIN, inverse distance, nearest) onto a grid over the domain and written as
  geo.dat in model coordinates with G 10 1; `session.make_geo_dat(...)`.
- LAGOON now needs SciPy and Pillow (both were there with Matplotlib's usual
  installs).

### LAGOON format 0.4

- The root attribute `georef` (section 1.2) and the run record's `georef`
  (RUN.md); `lagoon_format.geo` reads, writes and transforms it; `Store.georef`,
  `StoreWriter.create(georef=...)`, `Run.georef`. `lagoon-convert` copies the
  case's georeference into the stores it makes. lagoon-format 0.4.0.

## 1.0.0rc4 (fourth release candidate)

The fourth candidate for 1.0: the set-up sketch with room for its tables, also as
a narrow figure, and without the files other cases left in the folder.

### Case set-up

- **Set-up sketch**: narrow figures take the facts and the legend in one column,
  and the tables no longer overlap or squeeze the views; *Export figure* makes a
  set-up at least a column and a half wide (140 mm). A geo.dat, bed or solid left
  from another case in the folder is not drawn.

## 1.0.0rc3 (third release candidate)

The third candidate for 1.0: the set-up of a case beside its results (the input
files edited in LAGOON, DIVEMesh run from it, its grid shown), REEF3D's own
LAGOON stores read as they grow, colours scaled as in ParaView.

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
  `lagoonviz.core.data.setup.read_setup` gives the same in a script. The plan
  shows the bed in colour (geo.dat or the results' bed) with the coastline, the
  side view a section with the bed and the water; DIVEMesh's solids filled,
  relaxation zones hatched, inflow and outflow written beside their sides; a
  table of the main facts; for a refined grid (G 1) its boxes (G 10) and a table
  of the levels with about how many columns each has, against a uniform grid.
- **Mesh refinement boxes** (ctrl.txt G 10, G 11) in the *Case set-up* item; the
  σ-grids of FNPF, NHFLOW and SFLOW reach from the bed to the still water when
  B 10 gives placeholder heights.
- **The commands of both programs** (`lagoonviz.core.data.controlfile`): every
  command of `ctrl.txt` and `control.txt` with its title, values, choices and
  default, from the REEF3D and DIVEMesh user guides and the programs' readers
  (`tools/command_catalog.py`); input files are read and written back line by
  line, comments and order kept.
- Back in the results, the Set-up panel makes way for *Properties*.

### The 3D view and the scene

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
- An empty `.lagoon` folder (REEF3D has not written into it yet) says so.
- `lagoon-format` 0.3.1: the run log check takes each output stream's times on
  their own (a store counts its volume output only at the next one).

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
