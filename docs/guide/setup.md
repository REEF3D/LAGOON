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
- direct forcing and porous **structures** (A 581 – A 589, B 210 – B 270), and
  other regions (A 276, A 372, F 72, S 74);
- DIVEMesh's **solids** (S), as a structure item, and the **bed**: its topo
  objects (T) and the points of `geo.dat` (G 10).

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
| Gauges | wave gauges and the lines of wave profiles, numbered | P 50 – P 58, P 121 – P 147 |
| Probes | point, line and velocity probes, numbered | P 61 – P 67, P 88, P 144 – P 148 |
| Sections | discharge sections and flow files | P 167, P 168, P 230, P 240 |
| Forces | force boxes and Morison cylinders | P 81, P 85 |
| Floating bodies | bodies, centre of gravity, moorings, nets | X 23, X 110 – X 164, X 311 – X 322 |
| Structures | direct forcing and porous structures | A 581 – A 589, B 210 – B 270 |
| Regions | refinement boxes, wind areas, prescribed water levels | A 276, A 277, A 372, F 72, S 74 |
| Solids | outlines of DIVEMesh's solids (off: they are a structure item) | S |
| Bed | DIVEMesh's topo objects and the geo.dat points | T, G 10 |
| Labels | the numbers of gauges and probes, the kinds of the sides | |

Commands LAGOON cannot place (an unknown shape, too few values) are listed in the
set-up's notes (`lagoon-setup CASE --describe`).

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
the heights come from the bed (geo.dat) and the still water level when B 10 gives
placeholder heights (0 - 1 m). The same from the command line:

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
