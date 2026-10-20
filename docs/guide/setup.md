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

The third button, *2D plots* (++ctrl+shift+d++), puts the figure editor in place
of the 3D view without changing the mode of the session ([2D plots](plots2d.md)).

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
  the water reaches across it (*Water level*: F 60 by default; without F 60, 0 when
the bed's heights reach below and above 0 - a coast, an estuary - else halfway up
the banks) plus the *Margin*, never more than 2.5 times the river's usual half
width from the thalweg (a marsh or a harbour basin does not widen it),
  smoothed along the river, with *Points per bank* points to drag afterwards. The
  bed of its grid comes from geo.dat.
- **Draw the river** (before there is a bed: a new case started on the map): click
  the river in the 2D top view over the map as a line, point by point from upstream;
  click the last point again (a double click) or press Enter to end, Backspace
  takes the last point back, Esc stops, and dragging away from the points pans the
  view as always. The line is drawn in yellow, "upstream" at its start, and kept in
  `river.json`. **Continue drawing** takes it up again: a click adds a point at the
  downstream end, a click on a segment puts a point into it, a point is dragged to
  move it, and a right click (or Alt + click) on a point removes it.
  **Domain and geo.dat** sets the domain (B 10's x and y) around the river, a
  *River width* beyond its line and on DIVEMesh's cells, and makes geo.dat there
  with the Map tab's source (downloaded, or the file chosen there). **Create
  river** then makes the corridor along the line, *River width* wide, and its grid
  (20 sweeps of smoothing); with geo.dat it follows the river in the bed, as *Fit
  to the bed* does (the *Margin*, *Water level* and *Points per bank* below). Draw
  or continue the river again and *Create river* makes the corridor anew.
- **Fit to the bed** (with geo.dat) makes the corridor as it is - created along the
  river, laid over an area or dragged - follow the river in the bed: the thalweg within the
  corridor, from the deepest point of its upstream end to the deepest of its
  downstream end, the banks where the water reaches (*Water level*) plus the
  *Margin*, at most half as far again as the corridor was wide. Its points per bank,
  curve and grid's cells stay; the bed of its grid comes from geo.dat.
- **Drag points** (on after Start, Draw, Fit to the bed and Follow thalweg): press on a point of a bank in the 2D
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

The grid has *Cells along* x *Cells across* cells, its nodes on straight
cross-sections between the banks. *Cross-sections*: **square to the river** (from
points spaced equally along the line halfway between the banks, straight across it
to both banks; where they would cross on the inside of a tight bend they are spaced
evenly there), or **equal lengths** along both banks, paired (skewed in bends where
one bank is longer). Along the banks, square cross-sections are evened out so that no cell is much
shorter or longer than its neighbour (at most 1.2 times) or than an even one (0.6
to 1.6 times): across the river where the banks allow, never squeezed or
stretched. A corridor along the thalweg starts so; **Tidy the grid** (under
*Follow thalweg*) makes the grid nice again after points were dragged: these
cross-sections and a little smoothing (20 sweeps). While there is a corridor, the
set-up draws its grid only (not curvgrid.dat's as well); the River tab says when
curvgrid.dat is another grid (*Make grid* writes this one). *Smoothing* runs
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

The corridor is an item of the scene, **River corridor** (*Add → River corridor*
lays one out at once: along the thalweg with geo.dat, else along the domain). It
draws the banks and their points in yellow and the grid it makes; *Points* and
*Grid* in its properties switch them off. Selecting it in the set-up brings the
River tab forward. **Remove** (in the scene, or in the River tab) takes the
corridor away with its `corridor.json`; **Reset** (in its properties, or in the
River tab) lays it out again the way it was made - along the domain, over the area
drawn, along the river drawn on the map, along the thalweg with its margin, or
fitted to the bed from any of these - dropping the dragged points and keeping the
grid's settings. Each is one undo step.

The corridor is kept in `corridor.json` in the case folder. In a script:

```python
session.start_corridor([0, 200, 0, 20], points=12)   # or along the thalweg:
session.auto_corridor(margin=5.0, points=12)
# or the river drawn on the map first, the domain and the bed made for it:
session.draw_river([(0, 92), (125, 106), (250, 96), (400, 90)])
session.corridor_domain(width=60.0)   # B 10 around it; then make_geo_dat("best")
session.create_river(60.0, points=12, margin=2.0)   # fitted to the bed
session.move_corridor_point("left", 5, 90.0, 28.0)
session.set_corridor_options(points=16, curve="bspline", cells_along=100,
                             cells_across=10, smoothing=100, slope=0.001)
session.reset_corridor()     # as it was made, the dragged points dropped
session.make_corridor_grid()
```

## A multi-block grid

The panel's *Blocks* tab lays a multi-block grid: blocks of curvilinear cells that
meet along their faces, node on node, for REEF3D's SFLOW and NHFLOW (`G 50 1`,
DIVEMesh `R 20 1`, `blocks.dat`). Block 0 is the grid of the run (DIVEMesh's
logical box); the blocks 1.. lie beside it, a river reach and the harbour it runs
into, say.

The topology is kept in `blocks.json` in the case folder: the *corners*, the
*edges* between two corners (cells, a grading ratio, straight or curved through
their points) and the *blocks*, each with its four faces `i0`, `i1`, `j0`, `j1`
(-i, +i, -j, +j) as chains of edges (`"-e"` runs edge `e` backwards). Blocks that
share an edge share its nodes: that is the interface. Each block is meshed by
transfinite interpolation from its faces, then *Smoothing* sweeps of Winslow's
equations; the nodes of a wall slide along it, so cells stay square at the banks (not on a
face that is partly an interface, a river joining it: there they stay even).

- **Start** lays one block over the domain, **Draw** one over an area you drag in
  the 2D top view (a further one when there are blocks), **From corridor** makes
  the River tab's corridor block 0. **Remove** takes the blocks away with
  `blocks.json`.
- In the 2D top view: **Drag points** moves corners (the points of their edges
  go along, the more the nearer they are) and the points of curved edges;
  **Cut edge** cuts the edge clicked in two at that point (a new corner);
  **Add block** puts a block outside the edge clicked, *Depth* deep with *Cells*
  across, sharing that edge. Esc stops. Each change is an undo step.
- **Block**: its name, smoothing and four faces. A face shows its edges, cells and
  interfaces; a face without an interface is a wall, or open: *Inflow* (a
  discharge, m3/s) on the -i face, *Outflow* (a water level, m) on the +i face.
  **Block 0** makes the block chosen the grid of the run; **Remove block**.
- **Edge**: cells (they go on to the opposite face and through the blocks beyond,
  so the counts always match), curve and grading. **Finer** and **Coarser** double
  or halve every edge. **Pick edge** (in the 2D view) chooses the edge clicked.
- **Import blocks.dat**: a grid made elsewhere (a script, another program) as the
  case's blocks, to go on with here. Each face is cut where another block's face
  starts or ends on it; the pieces are edges *through their nodes* (the nodes kept
  exactly; more or fewer cells keep their grading), shared where two faces have the
  same nodes: the interfaces. The boundaries come from control.txt (C 11 – C 14 of
  block 0: an inflow with W 10, an outflow with F 62, other codes *open*) and the G 54
  lines of ctrl.txt. The inside of each block is made again (2000 sweeps of
  smoothing). Without geo.dat the nodes' bed is written as geo.dat, so *Make grid*
  gives it back.

### Coasts and estuaries

The faces of the blocks of a coast or an estuary lie on land a little beyond the
water, smooth, and the shore inside them is wetted and dried. Three tools lay them:

- **Circular arc** (an edge's *Curve*): through the edge's corners and one point,
  the open sea's side of a block round a mouth. Made an arc, a straight edge bulges
  out of its block by a quarter of its length; drag the point.
- **Spacing**: *By the cells at its ends* spaces an edge's nodes by the length of
  its first and last cell (*At the start*, *At the end*), growing evenly from one to
  the other (60 m at a mouth to 1.5 km offshore, say), or, with a *Largest* cell, by
  *Growth* from each end up to it (fine at both ends of a bay). The cells follow from
  the edge's length; *And the edges across the blocks* gives the edges whose cells
  go with it the same spacing, turned their way, and they all take the most cells
  any of their spacings asks for. After corners were moved, **Cells from spacing**
  sets the counts again. The tab says how long the end cells are now.
- **Along the water** (with geo.dat): **Follow the water** lays the edge on land,
  *Margin* m from the water below *Water below* (F 60), bending no tighter than
  *Bend radius* (land spits and inlets narrower than twice it are passed over, never
  nearer the water than the margin, to about a third of the smaller of margin and
  radius). It runs along the shore between the points nearest its corners, the way
  round nearer to where the edge is now; *Its corners onto the shore* moves them
  there (off: they stay, the edge joining the shore from them). Dragging one of its
  corners lays it again, so a corner slides along the coast; dragging a point of it
  shapes it by hand from then on (*Stop following* too).

A face of block 0 can be **open**: its condition is set in ctrl.txt by hand (a tide
edge, B 520 with a background): DIVEMesh's side keeps its code (C 11 2), no W 10 or
F 62. The other blocks' open faces are an inflow or an outflow (G 54).

**Attach river** (the *River* box) puts a river block on a part of the chosen
edge (one on the outline): along the river drawn in the River tab (*Draw the
river*; else the corridor's line), from upstream to where it meets the edge. The
edge is cut there, *Cells across* of its cells wide (or as many as make *Width*),
and the river block runs upstream from the cut, its banks parallel to the river
and bending onto the cut's ends, cells *Cell length* long (twice those across when
not given). Its i runs downstream: the *Discharge* comes in at its -i face (G 54).
Its banks are edges to drag, space or lay along the water like any other.

### Pieces for the processes

REEF3D runs each block 1.. on one process, and shares block 0 out over all of them
(DIVEMesh). A large block beside a small block 0 then keeps one process busy while
the others wait. *Pieces* writes a block as that many blocks in blocks.dat, cut in
equal parts along i or j; the pieces meet node for node, the G 54 lines name the
pieces at the faces (an inflow on the first piece, an outflow on the last). The
blocks stay one block here, to edit. A block with an inflow or outflow is cut along
i; pieces need 5 cells at least along the cut.

**Balance** cuts the blocks 1.. for the processes of M 10. Each block gets about
as many pieces as fit a process beside its share of block 0, then one more cut at
a time where it helps most, until the busiest process has at most 5 % more cells
than the mean (each piece also costs some exchange). Under *Rules* the tab says
how the processes share the cells, as REEF3D does it: block 0 with its ring evenly
over all, then the pieces, the largest first, each onto the process with the
fewest cells. The set-up draws the cuts as lines and names a block's pieces.

So a coast with an estuary is laid out in a few steps: one block over the sea, its
corners where the coast meets the open sea and on the banks of the mouth, its open
side an arc, its two sides along the water, spaced from fine at the mouth to coarse
offshore; then *Add block* at the mouth for the narrows and again for the bay, its
far corners on the river, its sides along the water; *Attach river* on the bay's
far side; *Balance* for the processes.

The tab lists the **rules** the grid breaks, and the set-up marks them where they
are (red: errors, Make grid waits; orange: warnings):

- the edges of a face must join up, corner to corner, and opposite faces have the
  same number of cells; no folded or left-handed cells;
- block 0's interfaces cover whole faces (blocks 1.. may meet several blocks on a
  face), and the block behind an interface is at least 5 cells deep (8 next to
  block 0: DIVEMesh `R 21`, a warning below that);
- inflow only on a -i face, outflow only on a +i face, never on an interface, and
  with a value; block 0 has at most one of each;
- warnings: corners with 3 or 5 blocks around them, poor cells.

The set-up draws the blocks in yellow with their corners and names, the grid of
each and the quality of its cells (*good*: angles of 45° or more and a size jump
of at most 1.3 to the neighbour; *fair*: 30°, 2; *poor*; *folded*).

**Make grid** writes `blocks.dat` (the number of blocks, then per block `ni nj`
and x y zb of every node, i fastest; the bed from geo.dat or flat), `R 20 1` and
block 0's box and sides (B 10's x and y, C 11 – C 14) in control.txt, and in
ctrl.txt `G 50 1`, one `G 54 b t v` per open face of the blocks 1.. and, for an
open face of block 0, `B 60 1` with `W 10` or `F 62`: one undo step. Then *Save*
and *Run DIVEMesh*.

The blocks are an item of the scene, **Multi-block grid** (*Add → Multi-block
grid*); *Points*, *Grid*, *Quality* and *Rules* in its properties switch the marks
off. In a script:

```python
session.start_multiblock([0, 100, 0, 50], cells=(20, 10))   # or:
session.multiblock_from_corridor()
session.extrude_block("b0_e", 60.0, cells=12)     # a block outside edge b0_e
session.split_block_edge("b0_n", 50.0, 50.0)
session.move_block_vertex("b0_ne", 102.0, 55.0)
session.set_block_edge("b0_e", cells=16, curve="through", ratio=1.0)
session.set_block_face(0, "i0", "inflow", 50.0)
session.set_block_face(1, "i1", "outflow", 2.5)
session.set_block_face(0, "i0", "open", 2)        # a tide edge: C 11 2, B 520 by hand
session.set_block_edge("b0_w", curve="arc")
session.set_block_spacing("b0_s", 1500.0, 60.0)   # first and last cell (m), across
session.set_block_spacing("b1_n", 150.0, 150.0, largest=300.0, growth=1.05)
session.follow_water("b0_n", margin=800.0, radius=3000.0)   # with geo.dat
session.fit_block_cells()
session.import_blocks(smoothing=2000)              # the case's blocks.dat
session.attach_river("b2_n", cells=16, step=100.0, discharge=336.0)
session.set_block_options(2, pieces=3, split="i")  # written as 3 blocks
session.balance_blocks()                           # for M 10's processes
print(session.multiblock_balance().text())
issues, quality = session.multiblock_check()
session.make_multiblock_grid()
```

## On the map

The panel's *Map* tab places the case on the Earth: its georeference
(`georef.json`), the domain drawn on a background map, and `geo.dat` made from a
bathymetry survey ([Maps and the place of a case](maps.md)).

## The sea at the case

The panel's *Sea* tab reads the sea at a georeferenced case, for its waves and its
boundaries, and the water measured nearby, to compare a run with. *Source* chooses
where the sea comes from (*Automatic*: MET Norway in its waters, Open-Meteo
elsewhere):

- **MET Norway** (thredds.met.no; Norwegian waters): **NORA3** (waves and wind, 3 km,
  hourly, from 1959, a month or two behind, with 2D spectra) and **NorKyst** (water
  level and current, 800 m, hourly, from 2017-02-20: v2 to 2023, v3 from 2024).
- **Open-Meteo** (open-meteo.com; anywhere): **ERA5** waves and wind (ECMWF's
  reanalysis, 0.5°, hourly, from 1940, a few days behind; no 2D spectra) and
  **Météo-France's** sea level with the tide and the surge and its surface currents
  (SMOC, 8 km, hourly, from 2022). Free for non-commercial use; for commercial use
  give the key of an Open-Meteo plan in *API key* (kept in LAGOON's settings).

It reads at the middle of the wave generation side (C 1x = 6), else at the domain's
centre; the tab says where and from which source.

- *Find storms*: the largest wave events at the case in the years chosen (Hs, at
  least three days apart), with Tp, where they come from and the wind. Choosing one
  sets the time window (a day before its peak to 12 hours after) and the hour of the
  spectrum (its peak). Reading a year takes some seconds.
- *Show the conditions*: Hs, Tp, direction and wind and the water level and current
  over the window, as time series items (*NORA3 waves …*, *NorKyst sea …*, or *ERA5
  waves …*, *Open-Meteo sea …*). They are tables in the case's `LAGOON_Sea` folder,
  time 0 at the window's start (UTC, written in the header); `beta` is where the
  waves go in the model (from its x-axis, as B 105), `u`, `v` the current in the
  model's frame, `level` the still water level (F 60) plus zeta. NorKyst's current
  is depth-averaged, Open-Meteo's at the surface.
- *Write spectrum-file-2d.dat* (MET Norway): NORA3's 2D wave spectrum (30
  frequencies, 24 directions) at the hour chosen as REEF3D's `spectrum-file-2d.dat`
  in the case folder: the directions within ±90° of the mean, turned into the
  model's frame (its rotation and the grid's north), the energy per rad/s and rad.
  With *Set ctrl.txt to them*, the wave generation follows it: B 85 11, B 92 31, B
  93 Hs Tp, B 105 the mean direction (its origin kept), B 130 1.
- *Set the waves (JONSWAP)* (Open-Meteo): ERA5's Hs, Tp and mean direction at the
  hour; with *Set ctrl.txt to them* a JONSWAP spectrum: B 85 2, B 92 31, B 93 Hs Tp,
  B 105 the direction (B 88, the peak enhancement, and the spreading B 130 stay as
  they are). Where Open-Meteo serves no peak period, ERA5's mean period stands for
  Tp (the table says so).
- *Write boundary hydrographs*: the water level and discharge along the sides of
  the domain over the window: per side `hydrograph_FSF_<n>.dat` (time, water level)
  and `hydrograph_Q_<n>.dat` (time, discharge into the domain), n = 1 x min, 2 y
  max, 3 y min, 4 x max; the sides more than half in the water, or those chosen. The
  discharge is the current across the side times the water depth, the level the
  still water level plus the mean zeta along it. NorKyst gives its own depth;
  with Open-Meteo the depth comes from the case's bed (geo.dat, else the bottom of
  the domain, B 10), and its sea level and current, asked about every 4 km along
  the sides (its resolution), are interpolated between. A *Boundaries* time series
  shows them all. With *Set ctrl.txt to them*: B 440 n n (the side as a patch), B
  421 n 1, B 422 n 1. The User Guide gives patchBC for SFLOW and CFD.

### A river

*River (GloFAS)*: the daily discharge of the river at the case from **GloFAS** (the
Copernicus flood service, 5 km, from 1984; through Open-Meteo, anywhere), hourly
between the days over the window, time 0 at its start. It is read at the upstream
end of the case's river corridor (the end nearer an inflow side), else at the middle
of the inflow side (C 1x = 1), else at the domain's centre; the tab says where, and
the message which GloFAS cell answered and how far it is. As `hydrograph.dat` for
ioFlow (B 60 2), or as `hydrograph_Q_<n>.dat` for a side as a patch (B 440, B 421).
A *GloFAS river* time series shows it. GloFAS is a model on a 5 km river network:
small rivers may not be in it, and a point near a confluence may take the
neighbour.

### Tide gauges

*Find gauges* lists the tide gauges nearest to the case (300 km at most) of
**Kartverket** (Norway: observations and the tide predicted, relative to mean sea
level, every 10 minutes), **NOAA CO-OPS** (USA: observations every 6 minutes and
predictions, relative to MSL) and the **IOC** Sea Level Station Monitoring Facility
(the world: raw observations of many agencies, without a datum: given about their
mean over the window, not quality-controlled). *Show the measured water level*: the
observed, the predicted and the surge (observed minus predicted) over the window,
and `level` the still water level plus the observed, as a time series (*NOAA gauge
8518750 …*) to compare the run's gauges with. *Use it as the water level at the
sides*: the observed level (gaps filled linearly) as `hydrograph_FSF_<n>.dat` at the
sides chosen above (B 440, B 422).

ctrl.txt changes are one undo step each and are saved with the input files; the
script repeats them as `set_input_text`. In a script: `session.find_storms(2015,
2024, source="Open-Meteo")`, `session.sea_conditions("2020-01-05T00:00Z",
"2020-01-06T00:00Z")`, `session.write_spectrum("2020-01-05T04:00Z")`,
`session.write_waves("2023-03-10T06:00Z")`, `session.write_boundaries(start, end,
sides=[1, 4])`, `session.river_discharge(start, end, target="B 60")`,
`session.find_gauges()`, `session.gauge_levels("NOAA:8518750", start, end)`,
`session.gauge_boundaries("Kartverket:TRD", start, end, [4])`.

Name the data when you publish results made with them: MET Norway (NORA3,
NorKyst; NLOD 2.0 / CC BY 4.0); Open-Meteo.com (CC BY 4.0) with Copernicus/ECMWF
(ERA5, GloFAS) and Météo-France / Copernicus Marine (sea level, currents);
Kartverket (NLOD 2.0), NOAA CO-OPS, the IOC Sea Level Station Monitoring Facility.

## Checking the input

*Check* (in the Set-up panel) looks at the input before a run, as it is saved:

- each file's commands, as for the editor (unknown commands, missing values, a
  command with no effect in the solver);
- the number of processes (M 10) in both files, and DIVEMesh's grid file for each
  (`DIVEMesh_Grid/grid-000001.dat` ...): a grid made for fewer processes, none at
  all, or a `control.txt`, `geo.dat` or `solid.stl` changed after DIVEMesh made it;
- no 3D output (neither P 20 nor P 30), no end (neither N 41 nor N 45);
- what the run will cost: the cells, per process (a note for very few or very many),
  and, when the last run in the folder had the same grid, the time per iteration and
  for N 41.

Errors (REEF3D or DIVEMesh would stop) come first, then warnings and notes.
`session.check_input()` gives the same in a script.

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
