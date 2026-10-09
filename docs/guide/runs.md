# Opening runs

## A case folder

*File → Open case…* takes the folder a REEF3D run wrote into (with its
`REEF3D_*` folders), one of those output folders, or a LAGOON store. *File → Open
output file…* takes a single `.pvtu`, `.pvtp`, `.vtu` or `.vtp` file and the series
it belongs to. A run can also be opened from a terminal:

```bash
lagoon ~/runs/flume                            # a case folder
lagoon ~/runs/flume/REEF3D_NHFLOW_VTP_FSF      # one output folder
lagoon ~/runs/flume/REEF3D_NHFLOW.lagoon       # a LAGOON store
lagoon figure.lagoon-session                   # a saved session
```

LAGOON reads the output of NHFLOW, FNPF, SFLOW and CFD: the free surface and bed
(VTP), the volume (VTU), AMR surfaces (`REEF3D_<SOLVER>_AMR`), floating bodies and
their mooring lines (6DOF), structures, DEM elements, ice floes, rod trees and
particles, and the text files of wave gauges (`WSF`), wave profiles (`WSFLINE`),
6DOF motions and forces. Each MPI rank's piece is read and joined; the parallel
files (`.pvtu`, `.pvtp`) say which pieces belong together.

Opening a case adds the case and its main items to the scene: the free surface
(or the AMR surface, when the run wrote one) and the bed, the objects of the same
run (bodies, mooring lines, structures, DEM elements, ice floes, rod trees,
particles), and its time series and wave profiles. The run's volume is there too,
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
