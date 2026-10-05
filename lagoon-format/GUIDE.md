# Using LAGOON stores

A LAGOON store (`REEF3D_<SOLVER>.lagoon`) holds a REEF3D run in one folder: the
volume, the free surface and the bed, time series, wave profiles, floating bodies,
particles and objects, and the refined surfaces of AMR runs, each stored once in a compact, chunked form (Zarr v3, see
[SPEC.md](SPEC.md)). It replaces the many VTU, VTP and `.dat` files of a run:
4 to 9 times smaller for grids, often a thousand times for floating bodies,
and much faster to open, slice and probe.

## Making a store

### While REEF3D runs (`P 18`)

| `P 18` | |
|--------|---|
| 0 | no store (default) |
| 1 | the store **instead of** the VTU/VTP files of what it holds |
| 2 | the store **as well as** the VTU/VTP files |

REEF3D writes `./REEF3D_<SOLVER>.lagoon` next to its other output:

- the **volume** (FNPF, NHFLOW: σ-grid; CFD: Cartesian grid), when the VTU output
  is on (`P 10`); each MPI rank writes its block in the background, so the run does
  not wait for it;
- the **free surface** and **bed** (FNPF, NHFLOW; CFD: the topography as bed), when
  their VTP output is on;
- **floating bodies** (6DOF): the mesh once and the motion of each output;
- **NHFLOW particles** (with `L 62` 1 or 3), **DEM elements** and **FNPF ice floes**;
- the **AMR surface** of FNPF, NHFLOW and SFLOW with refinement (`A 270`): level 0
  and the refined patches of every output, gathered from all ranks (`P 18 1`
  leaves out the `.vtr` and `.vtm` files of `REEF3D_<SOLVER>_AMR`).

A run never adds to the store of an earlier run: that store is renamed to
`REEF3D_<SOLVER>-<run>.lagoon-earlier` when the new run starts. Moving or tilted
grids (`B 180`, `B 191`, `B 192`) are not written to the store. CFD's free
surface (an iso-surface of the level set, not a grid), CPM particles, mooring
lines, rod trees and the time series files are not written by REEF3D; the
converter adds them (CPM, mooring lines, rod trees, time series) after the run.

If the store cannot be written (a rank that cannot write its part, a grid the
store cannot hold), that output stops going into the store on all ranks and
REEF3D writes its VTU/VTP (and AMR `.vtr`/`.vtm`) files again for the rest of the
run, also with `P 18 1`.

### From the VTK files (converter)

In LAGOON: *File → Convert to LAGOON format* converts the case that is open. Or
on the command line:

```sh
lagoon-convert CASE                         # everything there is
lagoon-convert CASE --outputs volume,bodies # some outputs only
lagoon-convert CASE --out other.lagoon      # another place
```

Converting again appends what the run has written since (a run that still goes
on). A store of another run is not appended to: move it away first. Fields are
kept bit for bit, heights and body vertices to float32 precision.

## Opening a store

- **LAGOON**: open the case folder as usual. A store replaces the VTU/VTP output
  it holds, unless the VTK files have newer outputs (a run still going on, not
  converted again); the item labels say "(LAGOON)". A store folder can also be
  opened on its own.
- **ParaView**: *Tools → Manage Plugins → Load New* and pick
  `paraview/lagoon_reader.py`; then open the store's top `zarr.json` and choose
  "LAGOON store". Grids come as structured blocks, bodies as triangles, particles
  as points, objects as polygons or lines, AMR surfaces as rectilinear grids per
  level and patch (the cells a finer patch covers hidden), all on the time bar.
- **Python** (`pip install lagoon-format`, NumPy only):

```python
from lagoon_format.store import Store

store = Store("REEF3D_NHFLOW.lagoon")
print(store.solver, store.run_id, list(store.outputs))

volume = store["volume"]
volume.times                            # time of every output [s]
u = volume.read("velocity", 10)         # (nz, ny, nx, 3) at output 10
z = volume.heights(10)                  # (nz, ny, nx) heights of the points
eta = store["free_surface"].series("eta", i=40, j=12)   # one point, all outputs

gauges = store.tables["nhflow_wsf_hg"]  # a time series file
gauges.time, gauges.column("P1")

profiles = store.profiles["nhflow_wsfline"]      # wave profiles along lines
bodies = store.bodies["nhflow_body"]
vertices, triangles = bodies.mesh(10)            # all bodies at output 10
particles = store.particles["nhflow_particles"]
particles.positions(10), particles.point_field("velocity", 10)
amr = store.amr["nhflow_amr"]                     # AMR surface
for grid in amr.grids(10, ["eta"]):              # level 0 of every rank, then the patches
    grid.level, grid.x, grid.y, grid.fields["eta"]   # nodes, (ny, nx) cell values
```

- **zarr-python** reads the arrays directly (the byte shuffle needs `numcodecs`):

```python
import zarr

root = zarr.open_group("REEF3D_NHFLOW.lagoon", mode="r")
root.attrs["lagoon"]                         # what the store holds
root["volume/blocks/b0000/velocity"][10]     # one MPI block at output 10
```

## For REEF3D's user guide

**P 18** *int* — LAGOON store of the output. 0: off (default); 1: the store
`REEF3D_<SOLVER>.lagoon` instead of the VTU/VTP files of the volume, free
surface, bed, 6DOF bodies, NHFLOW particles, DEM elements and FNPF ice floes,
and of the AMR output (`.vtr`, `.vtm`) of FNPF, NHFLOW and SFLOW; 2: the store
as well as these files. The store follows the LAGOON data format 0.3 (Zarr v3)
and opens in LAGOON, in ParaView with the LAGOON reader plugin and in Python with
`lagoon-format` or `zarr`.
