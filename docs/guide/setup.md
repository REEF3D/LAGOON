# Case set-up

LAGOON shows a case before it has run: open a folder with DIVEMesh's
`control.txt` and REEF3D's `ctrl.txt` (*File → Open case…*) and you see what they
set up. The same signs can be laid over the results of a run, and drawn as a
paper figure in plan and side view.

## Set-up and results

LAGOON works in one of two modes, switched with the two buttons in the toolbar
(*View → Set-up*, ++ctrl+shift+u++, and *View → Results*, ++ctrl+shift+r++):

- **Set-up**: the input files of the cases. The *Case set-up* items and, once
  DIVEMesh has run, its grid are shown; what the runs wrote is left out (listed
  greyed in the scene) and the timeline rests. Each case with input files gets a
  *Case set-up* item when it has none.
- **Results**: what the runs wrote, as always. The *Case set-up* items are there
  too, switched off: tick one to see where the gauges and zones were. The Set-up
  panel makes way for *Properties* (*View → Set-up* brings it back).

Set-up items keep a check box of their own in each mode, so switching back and
forth shows each mode as you left it. A case that has not run yet opens in the
set-up. The switch is a command like any other: it can be undone, and a script
switches with `session.set_mode("setup")` or `session.set_mode("results")`.

## Starting a case

*File → Open case…* opens a folder that has input files or output. For a case
that does not exist yet, *File → New case…* asks for a folder (make one in the
dialog with *New Folder*), starts its `control.txt` and `ctrl.txt` with a comment
line, and opens it in the set-up. Add DIVEMesh's commands in the Set-up panel's
`control.txt` tab (the domain B 10, the cell size B 1, the sides C 11 – C 16),
then REEF3D's in the `ctrl.txt` tab, and *Save*. In a script:
`session.new_case("~/runs/flume")`.

*File → New case from map…* starts a case at a place on the Earth instead: a
draft with a domain and a background map, which you move and turn on the map, give
a bathymetry from the web and then put in its folder with *Create case…*
([Starting a case on the map](maps.md#starting-a-case-on-the-map)). With no case
open, the *Map* tab of the Set-up panel starts one at Trondheim.

## A case that has not run yet

A case folder without output opens as its set-up:

- the **domain** (B 10 of `control.txt`), its grid (B 1 or B 2) and the kinds of
  its sides (C 11 – C 16): inflow, outflow, wave generation, numerical beach,
  symmetry plane, wall;
- the **still water level** (F 60 of `ctrl.txt`), the **relaxation zones** of wave
  generation and numerical beach (B 96, or B 107 and B 108), the wave direction
  (B 105), focus point (B 81) and wavemakers (X 170 – X 172);
- **gauges** and **probes** (P 50 – P 67, P 88, P 121 – P 148), numbered as in the
  output files, discharge **sections** (P 167, P 168), **force** boxes (P 81) and
  Morison cylinders (P 85);
- floating **bodies** (X 110 – X 164), their centre of gravity (X 23) and
  moorings (X 311, X 312), nets (X 322);
- direct forcing and porous **structures** (A 581 – A 589, B 210 – B 291), and
  other regions (A 276, A 372, F 72, S 74);
- DIVEMesh's **solids** (S), as a structure item, and the **bed**: its topo
  objects (T);
- the points of **`geo.dat`**, as an item of their own (see below).

The vertical scale of the case applies to the set-up as to any other item
([Vertical scale](scene.md#vertical-scale)), and the 2D button draws it flat from
above. A long, flat domain such as a flume is easiest to read from the side
(the 2D view's *front* side) with a vertical scale.

## The Case set-up item

A case that has run gets the same item when its folder has input files, switched
off: tick it in the scene to see where the gauges and zones were over the results.
`ctrl.txt` may also be the copy REEF3D keeps in `REEF3D_Case`, and without
`control.txt` (or B 10 in it) the domain is the extent of the results. *Add → Case
set-up* adds another one. Each group can be switched off in its properties:

| Property | Shows | Commands |
| --- | --- | --- |
| Domain | the domain's box | B 10 |
| Sides | the kinds of the domain's sides, labelled | C 11 – C 16 |
| Still water | the still water level, a faint plane | F 60 |
| Relaxation zones | wave generation and numerical beach zones | B 96, B 107, B 108 |
| Waves | wave direction, focus point, wavemakers | B 105, B 81, X 170 – X 172 |
| Gauges | wave gauges and the lines of free surface lines, numbered | P 50 – P 58, P 121 – P 147 |
| Probes | point, line and velocity probes, numbered | P 61 – P 67, P 88, P 144 – P 148 |
| Sections | discharge sections and flow files | P 167, P 168, P 230, P 240 |
| Forces | force boxes and Morison cylinders | P 81, P 85 |
| Floating bodies | bodies, centre of gravity, moorings, nets | X 23, X 110 – X 164, X 311 – X 322 |
| Structures | direct forcing and porous structures | A 581 – A 589, B 210 – B 291 |
| Regions | refinement boxes, wind areas, prescribed water levels | A 276, A 277, A 372, F 72, S 74 |
| Solids | outlines of DIVEMesh's solids (off: they are a structure item) | S |
| Bed | DIVEMesh's topo objects | T |
| Labels | the numbers of gauges and probes, the kinds of the sides | |

Commands LAGOON cannot place (an unknown shape, too few values) are listed in the
set-up's notes (`lagoon-setup CASE --describe`).

## The points of geo.dat

A case with a `geo.dat` gets a **geo.dat points** item in the scene, beside
*Case set-up* and *DIVEMesh grid*: every point of the file as DIVEMesh reads it
(scaled with G 12, z turned with G 23), a dot each, coloured by its height
(*Colour by* `elevation` or `depth`, any colour map). Zoom in, in the 2D view from
the top, to see how far apart they lie; *Point size* makes the dots smaller or
larger. Its properties say how many points there are, their **spacing** (the
grid's dx and dy for a regular grid such as the one *Make geo.dat* writes, else
the median distance to the nearest point, for soundings), and their extent and
heights. Like the other set-up items it has a check box of its own in each mode:
on in the set-up, off over the results until ticked. *Make geo.dat* adds it and
draws it again whenever it writes the file.

## Editing the input files

The **Set-up** panel (beside *Properties*; it comes forward in the set-up) edits a
case's input files: one tab for `control.txt` (DIVEMesh), one for `ctrl.txt`
(REEF3D). Each lists the commands of its file by section (*B · Boundary*,
*P · Printer*, ...); a command given several times (gauges, solids) has a line
for each, numbered as in the output files.

- **Select** a command to see its values, ready to change, with what the user
  guide says about it: its default, which REEF3D modules use it, whether it may be
  given several times. Its signs light up in the 3D view (yellow), so a gauge or a
  zone can be checked where it is. Choices (C 11 – C 16, A 10, ...) are picked
  from a list.
- **Add** finds any command of the program by its number (`P 61`) or by words
  (`wave gauge`, `relaxation`); it is added with its default values, after the
  other commands of its letter. *Add another* adds one more line of a command
  that may be given several times.
- **Remove** takes the selected line out.
- **Text** edits the whole file as text (*Apply text*).

Commands the program would stumble over are marked: red for what stops it or is
misread (a missing value, text with a capital letter and no `//`), orange for what
it ignores (an unknown command, one given twice, one the module does not use);
the tooltip says why.

Every change is an undo step and a line of the session's script
(`session.set_input("ctrl.txt", "P 51", [3.0, 2.5], occurrence=1)`), and the 3D
view follows at once. The files themselves are written with *Save* (or *File →
Save input files*); a tab with edits not saved yet has a `*`, and LAGOON asks
before closing. Comments, blank lines and the order of the file are kept: only
the lines that changed are rewritten.

## Porous structures

The panel's *Porous* tab holds the porous structures of a case (VRANS:
breakwaters, rubble mounds, porous cylinders), as the `ctrl.txt` lines that make
them:

| Module | Structures | Material |
| --- | --- | --- |
| CFD | box (B 270), vertical cylinder (B 274), wedge along x (B 281) or y (B 282), plate along x (B 291) | on each line: n, d50, α, β |
| NHFLOW | box (B 210), cylinder along y (B 212), vertical cylinder (B 213), sphere (B 215), wedge along x (B 217) or y (B 218) | one for all: B 201 n d50 α β |

- **Add** puts the kind chosen in the middle of the domain (in a 2D case across
  it), from the bed to the still water level (F 60), of rubble: n = 0.4, d50 =
  0.05 m, α = 500, β = 2.0. For NHFLOW, B 201 comes with the first structure.
- **Draw** drags its plan in the 2D top view of the 3D view (Esc stops): a box or
  wedge over the area, the largest cylinder or sphere in it, snapped to the cells
  (B 1). It reaches from the bed to the still water.
- **Select** one to see its shape and material, ready to change; it lights up in
  the 3D view. **Remove** takes it out.

A wedge's top runs from its first height at its start to its second at its end,
its base at the lower of the two (a slope rising along x: z start at the bed,
z end at the crest). A plate is a band of the thickness given above that line.
n is the porosity, d50 the mean stone diameter, α and β the linear and nonlinear
resistance coefficients of the Forchheimer terms. SFLOW and FNPF have no porous
structures. Each change is an undo step, saved with the input files; in a script:
`session.add_structure("B 270", [30, 40, 0, 1, 0, 0.6, 0.4, 0.05, 500, 2])` (no
values: the default place and material).

## Solids

The *Solids* tab works the same way for solids:

| Module | Solids | File |
| --- | --- | --- |
| every module (DIVEMesh) | box (S 10), vertical cylinder over the whole height (S 33), cylinder along y across the domain (S 32), sphere (S 51), wedge along x (S 61) or y (S 62) | `control.txt` |
| NHFLOW (direct forcing) | box (A 581), cylinder along y (A 583), vertical cylinder (A 584), sphere (A 586), wedge along x (A 587) or y (A 588) | `ctrl.txt` |

*Add* and *Draw* make a solid from the bed to the domain's top (for NHFLOW, whose
domain ends at the still water, up to half the water depth above it). DIVEMesh's
solids are meshed by *Run DIVEMesh*; NHFLOW takes them as A 580 says (0: they raise
the bed, 1: immersed). In a script: `session.add_structure("S 33", [12.0, 2.5, 0.4])`.

## A river corridor

The panel's *River* tab lays a river corridor: two banks shaped by points you drag,
and the curvilinear grid between them that REEF3D's SFLOW and NHFLOW read with
`G 50 2` (`curvgrid.dat`).

- **Start** lays a straight corridor along the domain (a tenth in from each side
  across), **Draw** one over an area you drag in the 2D top view; it flows along the
  area's longer side, with 12 points on each bank.
- **Follow thalweg** (with geo.dat) lays the corridor along the river in the bed:
  the thalweg, the path along the deepest channel, from the deepest point of the
  inflow side (C 1x = 1) to the deepest of the outflow side (C 1x = 2; without
  them, from end to end of the domain's longer axis), smoothed. Its banks lie where
  the water reaches across it (*Water level*: F 60 by default) plus the *Margin*,
  smoothed along the river, with *Points per bank* points to drag afterwards. The
  bed of its grid comes from geo.dat.
- **Drag points** (on after Start, Draw and Follow thalweg): press on a point of a bank in the 2D
  top view and drag it; the bank follows at once, drawn over everything while you
  drag. Pressing anywhere else pans the view as always; Esc stops. Each point moved
  is an undo step.
- **Points per bank** (4 – 60) spaces that many points along the banks as they are
  now: more for a winding river, fewer for a calm one.
- **Curve**: *through the points* (a centripetal Catmull-Rom spline: the bank passes
  every point, without loops or overshoot between close points) or *smooth* (a
  clamped cubic B-spline the points pull on; it passes the first and the last).
  The banks and their points are drawn in yellow.
- **Reverse flow** swaps upstream and downstream. The bank on the left looking
  downstream is the *left* bank; "upstream" is written at the corridor's start.

The grid has *Cells along* x *Cells across* cells: each bank is cut into equal
lengths and the nodes between them lie on straight lines across. *Smoothing* runs
that many sweeps of Winslow's elliptic grid equations over the inner nodes (the
banks and the two ends stay): smoother, more even cells in bends. The set-up draws
the grid as it changes, and the tab says how large its cells are, their smallest
angle and whether any fold (REEF3D stops on a folded cell: move the points, use
fewer cells along or more smoothing). The bed (zb at the nodes) slopes down from the
upstream end (*Bed at upstream end*, *Bed slope* in m/m), or comes from geo.dat's
points.

**Make grid** writes `curvgrid.dat` (`ni nj`, then x y zb of every node, i fastest),
sets `G 50 2` in ctrl.txt and makes DIVEMesh's grid the logical box of the
curvilinear one (B 10's x and y and B 1: ni x nj cells; its heights kept), one undo
step. Then *Save* and *Run DIVEMesh*. The sides keep their numbers: 1 is the
upstream end (i = 0), 4 the downstream end, 3 and 2 the banks. With `G 50 2`, the
set-up draws the domain, its sides and the grid (*Curvilinear grid*, on the bed)
where the nodes are, not DIVEMesh's logical box.

The corridor is kept in `corridor.json` in the case folder. In a script:

```python
session.start_corridor([0, 200, 0, 20], points=12)   # or along the thalweg:
session.auto_corridor(margin=5.0, points=12)
session.move_corridor_point("left", 5, 90.0, 28.0)
session.set_corridor_options(points=16, curve="bspline", cells_along=100,
                             cells_across=10, smoothing=100, slope=0.001)
session.make_corridor_grid()
```

## On the map

The panel's *Map* tab places the case on the Earth: its georeference
(`georef.json`), the domain drawn on a background map, and `geo.dat` made from a
bathymetry survey ([Maps and the place of a case](maps.md)).

## The sea at the case

The panel's *Sea* tab reads the sea at a georeferenced case from MET Norway's
hindcasts on thredds.met.no: **NORA3** (waves and wind, 3 km, hourly, from 1959,
a month or two behind) and **NorKyst** (water level and current, 800 m, hourly,
from 2017-02-20: v2 to 2023, v3 from 2024). It reads at the middle of the wave
generation side (C 1x = 6), else at the domain's centre; the tab says where.

- *Find storms*: the largest wave events at the case in the years chosen (NORA3's
  Hs, at least three days apart), with Tp, where they come from and the wind.
  Choosing one sets the time window (a day before its peak to 12 hours after) and
  the hour of the spectrum (its peak). Reading a year takes some seconds.
- *Show the conditions*: NORA3's Hs, Tp, direction and wind and NorKyst's water
  level and depth-averaged current over the window, as time series items (*NORA3
  waves …*, *NorKyst sea …*). They are tables in the case's `LAGOON_Sea` folder,
  time 0 at the window's start (UTC, written in the header); `beta` is where the
  waves go in the model (from its x-axis, as B 105), `u`, `v` the current in the
  model's frame, `level` the still water level (F 60) plus NorKyst's zeta.
- *Write spectrum-file-2d.dat*: NORA3's 2D wave spectrum (30 frequencies, 24
  directions) at the hour chosen as REEF3D's `spectrum-file-2d.dat` in the case
  folder: the directions within ±90° of the mean, turned into the model's frame
  (its rotation and the grid's north), the energy per rad/s and rad. With *Set
  ctrl.txt to them*, the wave generation follows it: B 85 11, B 92 31, B 93 Hs Tp,
  B 105 the mean direction (its origin kept), B 130 1.
- *Write boundary hydrographs*: NorKyst's water level and discharge along the
  sides of the domain over the window: per side `hydrograph_FSF_<n>.dat` (time,
  water level) and `hydrograph_Q_<n>.dat` (time, discharge into the domain), n = 1
  x min, 2 y max, 3 y min, 4 x max; the sides more than half in NorKyst's water,
  or those chosen. The discharge is the depth-averaged current across the side
  times the water depth (NorKyst's, plus zeta), the level the still water level
  plus the mean zeta along it. A *Boundaries* time series shows them all. With *Set
  ctrl.txt to them*: B 440 n n (the side as a patch), B 421 n 1, B 422 n 1. The
  User Guide gives patchBC for SFLOW and CFD.

ctrl.txt changes are one undo step each and are saved with the input files; the
script repeats them as `set_input_text`. In a script: `session.find_storms(2015,
2024)`, `session.sea_conditions("2020-01-05T00:00Z", "2020-01-06T00:00Z")`,
`session.write_spectrum("2020-01-05T04:00Z")`,
`session.write_boundaries(start, end, sides=[1, 4])`.

The data are MET Norway's under NLOD 2.0 / CC BY 4.0: name MET Norway (NORA3,
NorKyst) when you publish results made with them.

## Running DIVEMesh

*Run DIVEMesh* (in the Set-up panel, the toolbar in the set-up, *File → Run
DIVEMesh*) saves `control.txt` and runs DIVEMesh in the case folder; its output
shows in the panel's *DIVEMesh* tab and is kept in
`DIVEMesh_Log/DIVEMesh-run.log`. When it has finished, the set-up shows what it
made: its grid, the solids (`REEF3D_Solid.vtp`) and the bed (`REEF3D_Topo.vtp`,
as a *Bed* item of the set-up).

LAGOON looks for the program as `LAGOON_DIVEMESH`, `DiveMESH` on the PATH, then
in a source build (`~/Codelite/DIVEMesh/bin/DiveMESH` from make,
`~/Codelite/DIVEMesh/src/Release/DIVEMesh` from CodeLite); if none is there, it
asks for it once and remembers it.

## DIVEMesh's grid

The *DIVEMesh grid* item draws the grid lines on the faces of the domain: by
default the bottom and the two faces at the back of the usual view (*Faces*:
bottom and back, bottom, back, outside). A fine grid is drawn with every n-th
line, so a face has a few hundred lines at most (*Every n-th line*: 0 for that,
1 for every line); the planes between the MPI ranks of DIVEMesh's partition
(M 10) are drawn in orange. Before DIVEMesh has run, the item shows the uniform
grid `control.txt` sets up (B 10 with B 1 or B 2); a stretched grid (B 101 –
B 103) shows once DIVEMesh has made it. The item's properties give the number of
cells and the smallest and largest cell sizes along x, y and z.

FNPF and NHFLOW scale DIVEMesh's heights to σ (0 – 1) and stretch them from the bed
to the water. Their grid, and the domain, are drawn from the bed's lowest point
(geo.dat, else B 10's bottom) to the still water level (F 60), with the levels
spaced as DIVEMesh spaced them: as ParaView shows the results at the start, whatever
heights B 10 gives (0 – 1 m, or a box higher than the water).

## The set-up sketch

*File → Export figure → Case set-up* draws the set-up as a paper figure:

- the **plan view**: the bed in colour (geo.dat, or the bed of the results) with
  its scale and the coastline at the still water level; the sides (inflow with its
  discharge W 10, outflow written beside them), relaxation zones hatched, gauges
  and probes numbered, DIVEMesh's solids filled; for a refined grid (G 1) the
  refinement boxes (G 10) and faint lines of the level-0 grid;
- the **side view**: a section through the middle of the domain, the bed filled
  with sand and the water above it, the zones it crosses hatched; a long, flat
  domain gets a vertical scale, given in the title (only the side view for a 2D
  case);
- a **table of the main facts**: module, domain, cells, still water, waves,
  generation and absorption, inflow, simulated time, output interval, refinement,
  processes;
- for a refined grid, **the levels**: their cell size and about how many columns
  each has (level 0 the whole domain, the finest the boxes, those between with a
  nesting buffer of G 3 cells), against a uniform grid as fine as the finest
  level; cells the run flags (G 12, G 20 - G 23) come on top;
- the legend.

For FNPF, NHFLOW and SFLOW, whose σ-grids reach from the bed to the free surface,
the heights come from the bed (geo.dat) and the still water level, whatever
heights B 10 gives. The same from the command line:

```bash
lagoon-setup ~/runs/flume                  # flume-setup.pdf beside the case
lagoon-setup ~/runs/flume sketch.png --width 190
lagoon-setup ~/runs/flume --describe       # the main facts as text
```

`--describe` lists the files found, the module (A 10), the domain and grid, the
kinds of the sides, the still water level and waves, how many gauges, probes and
other signs there are, and what does not fit together (M 10 differs between
`control.txt` and `ctrl.txt`: REEF3D would stop).

In a script, `lagoonviz.core.data.setup.read_setup` gives the set-up of a folder
and `lagoonviz.render.setup_sketch.setup_figure` its sketch:

```python
from lagoonviz.core.data.setup import read_setup
from lagoonviz.render.figure import FigureSpec, save_figure
from lagoonviz.render.setup_sketch import setup_figure

setup = read_setup("~/runs/flume")
print(setup.describe())
save_figure(setup_figure(setup, FigureSpec(width_mm=140)), "flume-setup.pdf")
```

A session edits the input files and runs DIVEMesh the same way the panel does:

```python
from lagoonviz.core import Session

session = Session()
session.open_case("~/runs/flume")                    # not run yet: the set-up
session.set_input("control.txt", "B 1", [0.025])      # finer cells
session.add_input("ctrl.txt", "P 51", [12.0, 0.5])    # one more wave gauge
session.save_inputs()
session.run_divemesh(on_line=print)                   # its grid, solids and bed
```
