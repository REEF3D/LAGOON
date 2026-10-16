# Opening runs

## A case folder

*File → Open case…* takes the folder a REEF3D run wrote into (with its
`REEF3D_*` folders), one of those output folders, or a LAGOON store. *File → Open
output file…* opens one output series: it lists a folder's files as ParaView
does, the numbered files of a series as one line (`REEF3D-NHFLOW-FSF-..pvtp`,
with the number of files, their first and last number and their size) and the
pieces of parallel output (the `.vtp`/`.vtu` next to their `.pvtp`/`.pvtu`) left
out. Double-click a folder to go in (*Up* goes out), a series or a file to open
it with its case. A run can also be opened from a terminal:

```bash
lagoon ~/runs/flume                            # a case folder
lagoon ~/runs/flume/REEF3D_NHFLOW_VTP_FSF      # one output folder
lagoon ~/runs/flume/REEF3D_NHFLOW.lagoon       # a LAGOON store
lagoon figure.lagoon-session                   # a saved session
```

LAGOON reads the output of NHFLOW, FNPF, SFLOW and CFD: the free surface and bed
(VTP), the volume (VTU), AMR surfaces (`REEF3D_<SOLVER>_AMR`), floating bodies and
their mooring lines (6DOF), structures, DEM elements, ice floes, rod trees and
particles, and the text files of wave gauges (`WSF`), free surface lines (`WSFLINE`),
6DOF motions and forces. Each MPI rank's piece is read and joined; the parallel
files (`.pvtu`, `.pvtp`) say which pieces belong together.

A **CFD** run has no free surface file: it opens with its free surface as an
iso-surface of the volume, *Free surface*, where the level set `phi` is 0 (coloured
by the velocity), the volume itself switched off, and its floating bodies.

With **adaptive mesh refinement**, CFD writes the refined blocks of every output in
`REEF3D_CFD_AMR`: an index per output (`REEF3D-CFD-AMR-<output>.vtm`) and its blocks,
3D grids of half, a quarter, ... of the volume's cell size. LAGOON joins them to the
volume (its label says *with AMR*): the volume's cells under a block are left out,
and so are a block's cells under a finer block. The free surface, slices,
iso-surfaces and iso-volumes are then as fine as the refinement, e.g. around a
floating body; *Colour by* `level` shows where (0: the volume's own cells), and the
free surface's *Grid lines* (dark or white) draw where it crosses the cells' faces,
of every level. The item **AMR blocks** (in the scene, switched off) draws the outlines of the blocks,
coloured by their level. Only the blocks an output's index lists are read: a file
left over from an earlier run is not. Streamlines and time series at a point come
from the volume's own cells.

**SEASTATE**, REEF3D's spectral wave model, writes maps of the sea state
(`REEF3D_SEASTATE_VTP`): they open as a *Sea state* item, coloured by the
significant wave height `Hs` on the rainbow map (as SEASTATE's own maps), with
arrows where the waves go and dry land left out. *Colour by* offers `Tm01`,
`Tm-10`, `Tp`, `dir`, `spread`, `depth` and whatever the run wrote besides (the
wind `U10x`, `U10y`; coupled to a flow solver, the wave forces `Fx`, `Fy` and the
Stokes transport `MStx`, `MSty`; with rollers `roller`, `Dr`, `Dw`); *Vectors of*
`wave_direction` (unit arrows), `wave_vector` (as long as Hs), `wind`,
`wave_force` and `stokes_transport`, which LAGOON makes from them. SEASTATE's AMR
output (`REEF3D_SEASTATE_AMR`) opens as an AMR surface coloured by Hs, its
refined patches outlined, at the times SEASTATE wrote.

Opening a case adds the case and its main items to the scene: the free surface
(or the AMR surface, when the run wrote one) and the bed, the objects of the same
run (bodies, mooring lines, structures, DEM elements, ice floes, rod trees,
particles), and its time series and free surface lines. The run's volume is there too,
switched off: tick it (or *Volume* in the toolbar) to show it. When a folder
holds the output of several runs, the run written last is shown. Slices and
iso-surfaces are in the toolbar, streamlines in the *Add* menu.

The solids of the case (a monopile, a breakwater, the blocks that shape a beach)
come from DIVEMesh: LAGOON reads `DIVEMesh_Paraview/REEF3D_Solid.vtp`, which
DIVEMesh writes with every solid of `control.txt`, STL geometries included.
Without it, the common solids are built from `control.txt` itself (boxes,
cylinders, cones, spheres, wedges and polyhedra: S 10, S 32, S 33, S 41 – S 43,
S 51, S 61 – S 63, S 81 – S 84). They are shown as one structure, *DIVEMesh
solids*; the faces that lie in the bed REEF3D writes are left out.

Only the case's own solids are opened. DIVEMesh leaves `REEF3D_Solid.vtp` alone
when a case has no solids, so in a folder where several cases run (your home
folder, say) the file can be another case's. LAGOON passes it over when:

- `control.txt` has no solid commands (S): the case has none;
- DIVEMesh has run since without writing it (its partition and grid preview in
  `DIVEMesh_Paraview` are more than an hour newer);
- the solids lie outside the domain of the run shown (from the run log).

When `control.txt` has changed since DIVEMesh wrote the file, its solids are built
from `control.txt` instead. DIVEMesh's bed (`REEF3D_Topo.vtp`) left over from an
earlier DIVEMesh run is passed over in the same way.

## Time

The timeline steps through REEF3D's outputs. Their times come from the run log
when REEF3D wrote one (`REEF3D_<SOLVER>_run.jsonl`), else from the output interval
in `ctrl.txt` (`P 30`); without either, the outputs are numbered. The case's
*Output interval* property sets the interval by hand.

## Several runs

Open more cases into the same session; each item has a *Case* property naming the
case it shows. *View → Compare side by side* (++ctrl+shift+2++) shows two cases in
two 3D views with linked cameras and time ([The 3D view](viewport.md#compare)).

## A running simulation

*File → Reload case* (++f5++) reads the case folders again: new outputs of a
running simulation appear, and the scene, time and camera stay. *File → Follow
running simulation* does this by itself whenever the simulation writes, and keeps
the timeline at the newest output.

## LAGOON stores

A run converted to the [LAGOON format](format.md) (*File → Convert to LAGOON
format…*, or `lagoon-convert`) is several times smaller and opens faster. A store
in the case folder (`REEF3D_<SOLVER>.lagoon`) is read in place of the VTK files.
REEF3D can also write the store itself during the run (`P 18 1`). A store opened
on its own shows only its solver's run. REEF3D adds an output to the store when it
writes the next one: a store opened just after the run started has no outputs yet;
open it again, or reload, once the run has written more.
