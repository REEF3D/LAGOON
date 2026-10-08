# Case set-up

LAGOON shows a case before it has run: open a folder with DIVEMesh's
`control.txt` and REEF3D's `ctrl.txt` (*File → Open case…*) and you see what they
set up. The same signs can be laid over the results of a run, and drawn as a
paper figure in plan and side view.

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

*Add → Case set-up* puts the same signs over a case that has run, e.g. to see
where the gauges and zones were. Each group can be switched off in its
properties:

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

## The set-up sketch

*File → Export figure → Case set-up* draws the set-up as a paper figure: the plan
view above, the side view below (only the side view for a 2D case), with a
legend. The side view of a long, flat domain gets a vertical scale, given in its
title. The same from the command line:

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
