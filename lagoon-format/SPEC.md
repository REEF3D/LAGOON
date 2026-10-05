# LAGOON data format specification

Version: 0.2 (draft for review)
Status: σ-grid volumes (FNPF, NHFLOW), Cartesian volumes (CFD), structured
surfaces (free surface, bed), time series (wave gauges, 6DOF, forces), wave
profiles, floating bodies, particles (CPM, NHFLOW) and objects (DEM elements, ice
floes, rod trees, mooring lines) are specified and implemented
(`lagoon_format.store`, converter `lagoon-convert`).

Changes in 0.2: time series tables (section 7), wave profiles (section 8),
floating bodies (section 9), particles and objects (section 10). A 0.1 reader
opens 0.2 stores and ignores the new groups.

## Goals

- **Small.** A REEF3D VTU file stores every point's coordinates and every cell's
  connectivity in every output. A structured grid needs neither: the horizontal
  grid and the σ-levels are stored once, and each output only adds the fields and
  the two heights per column (bed and free surface). With byte shuffling and zlib,
  a σ-grid output is 4 to 9 times smaller than the VTU.
- **Fast to open and to read in parts.** Opening a store reads a few small JSON
  files, whatever the number of outputs. A probe, a slice or a free surface reads
  only the chunks it needs.
- **Written in parallel.** Every MPI rank writes its own block of the grid, with no
  communication and no shared files.
- **Open.** The store is a standard [Zarr v3](https://zarr-specs.readthedocs.io/en/latest/v3/core/index.html)
  hierarchy (directories, JSON metadata, binary chunk files), readable with
  zarr-python and xarray as well as with the reference reader, which needs only
  NumPy and the Python standard library. This specification is MIT-licensed.

## Run metadata

What a run is, what it writes and the time of every output is specified in
[RUN.md](RUN.md) (the REEF3D run log). A store holds the outputs of **one run**;
the run's id and its run record are kept in the root attributes, the time of
every output in the `time` array of each output group.

## 1. The store

A store is a directory named `<name>.lagoon` holding a Zarr v3 hierarchy. The
converter and the REEF3D writer name it `REEF3D_<SOLVER>.lagoon` and put it in
the case folder, next to the `REEF3D_*` output folders, e.g.

```
case/
  ctrl.txt
  REEF3D_Case/REEF3D_FNPF_run.jsonl
  REEF3D_FNPF.lagoon/
    zarr.json                    root group
    volume/                      output group (the VTU output)
      zarr.json
      time/ step/                (nt,) per output
      x/ y/ sigma/               the grid, once
      blocks/
        zarr.json
        b0000/                   block group (MPI rank 0's part)
          zarr.json
          z_bed/ z_surface/      (nt, ny, nx) per output
          velocity/ Fi/ ...      (nt, nz, ny, nx[, c]) per output
        b0001/ ...
    free_surface/                output group (the FSF VTP output)
    bed/                         output group (the BED VTP output)
    series/                      time series tables (section 7)
      nhflow_wsf_hg/             one table per file: time/, values/
    profiles/                    wave profiles along lines (section 8)
      nhflow_wsfline/            time/, line_1/x eta theory, ...
    bodies/                      floating bodies (section 9)
      nhflow_body/               time/, step/, body_0/vertices triangles translation rotation
    particles/                   particles and objects (section 10)
      cfd_particles/             time/, step/, point_end/, position/, point_data/velocity ...
```

Group and array names are case sensitive and use only `[A-Za-z0-9_.-]`.

### 1.1 Root group

`zarr.json` of the root is a Zarr group whose attributes hold the key `lagoon`:

```json
{"zarr_format": 3, "node_type": "group",
 "attributes": {
   "lagoon": {
     "format": "lagoon", "version": "0.1",
     "solver": "FNPF",
     "outputs": ["volume", "free_surface", "bed"],
     "created_by": "lagoon-convert 0.1 (lagoon-format 0.1.0)",
     "created": "2026-10-04T12:00:00Z",
     "source": "REEF3D VTK output"
   },
   "reef3d_run": {"type": "run", "run": "20261003T154312Z-7f3a", "solver": "FNPF", "...": "..."}
 }}
```

- `version`: `major.minor` of this specification. A reader accepts stores of the
  same major version and ignores attributes, groups and arrays it does not know.
- `outputs`: the output groups in the store (section 2).
- `series`, `profiles`, `bodies`, `particles`: the time series tables, wave
  profile sets, body sets and particle sets (sections 7 to 10), optional.
- `reef3d_run`: the `run` record of the run log (RUN.md), unchanged, when known.
- `units`: optional, as in RUN.md (`{"length": "m", "time": "s", ...}`); SI if absent.

## 2. Output groups

One group per output stream of the run, named by its role (the same names as
`role` in RUN.md): `volume` (VTU), `free_surface` (FSF VTP), `bed` (BED VTP).
Its attributes:

```json
{"lagoon": {
   "role": "volume", "grid": "sigma",
   "shape": {"x": 351, "y": 301, "level": 11},
   "variables": {"velocity": {"components": 3, "units": "m/s"},
                 "Fi": {"components": 1, "units": "m2/s"},
                 "elevation": {"components": 1, "units": "m"}},
   "blocks": 4,
   "source": "REEF3D_FNPF_VTU"}}
```

- `grid`: `sigma` (section 3), `cartesian` (section 3a) or `surface` (section 4).
- `variables`: the point fields, in the order the writer chose; `components` is 1
  for scalars and 3 for vectors (x, y, z). `units` is optional.
- `blocks`: the number of block groups `blocks/b0000` ... (four digits, more if
  needed, numbered from 0, usually the MPI rank).
- `source`: optional, where the data came from; `dataset`: optional, the name the
  converting application gave the source output (LAGOON: `fnpf_volume`, ...).

### 2.1 Time

| array  | shape  | dtype   | content |
|--------|--------|---------|---------|
| `time` | (nt,)  | float64 | simulation time of each output [s] |
| `step` | (nt,)  | int64   | REEF3D's output number (the number in the VTK file names) |

Outputs are in the order they were written, `time` rising. **`time` is the
authority on how many outputs exist**: a writer writes an output's data first and
extends `time` and `step` last, so a reader that uses `nt = len(time)` never sees
an output whose data is incomplete. Data arrays may be longer than `time`.

### 2.2 Grid coordinates

| array   | shape  | dtype   | content |
|---------|--------|---------|---------|
| `x`     | (nx,)  | float64 | x of the grid lines, rising |
| `y`     | (ny,)  | float64 | y of the grid lines, rising |
| `sigma` | (nz,)  | float64 | σ of the levels, rising from 0 (bed) to 1 (free surface); σ-grids only |

The horizontal grid does not change during a run. A grid with one point in y (a
2D flume written with one cell) has `ny = 2` as REEF3D writes it.

### 2.3 Blocks

The grid is split into rectangular blocks in x and y, one per MPI rank. A block
is a group `blocks/bNNNN` with the attributes

```json
{"lagoon": {"start": [0, 0], "size": [176, 151], "rank": 0}}
```

- `start`: `[i0, j0]`, the index of the block's first point in `x` and `y`.
- `size`: `[nx_b, ny_b]`, its number of points in x and y.
- `rank`: optional, the MPI rank that wrote it.

Blocks may overlap: REEF3D writes the points on the boundary between two ranks in
both. Readers take a point from the block with the **lowest number** that holds
it. Points of the global grid in no block have no value (NaN).

All arrays of a block cover the block only (shapes use `nx_b`, `ny_b`) and have
`dimension_names`, so they open in xarray with named dimensions:

| dimension   | meaning |
|-------------|---------|
| `time`      | output (index into the group's `time`) |
| `level`     | σ-level (index into `sigma`) |
| `y`, `x`    | grid point within the block |
| `component` | vector component x, y, z |

Arrays are stored in C order with x fastest, as REEF3D writes its points.

## 3. σ-grids (`grid: "sigma"`): FNPF and NHFLOW volumes

Each column (x_i, y_j) has `nz` points between the bed and the free surface. The
height of level k is

```
z = z_bed + sigma[k] * (z_surface - z_bed) + z_offset[k]
```

Arrays of a block:

| array       | shape                    | dtype   | content |
|-------------|--------------------------|---------|---------|
| `z_bed`     | (nt, ny_b, nx_b)         | float32 | z of the lowest level (k = 0) |
| `z_surface` | (nt, ny_b, nx_b)         | float32 | z of the highest level (k = nz − 1) |
| `z_offset`  | (nt, nz, ny_b, nx_b)     | float32 | optional: correction of the level heights, see below |
| *field*     | (nt, nz, ny_b, nx_b)     | float32 | scalar field, one per scalar in `variables` |
| *field*     | (nt, nz, ny_b, nx_b, 3)  | float32 | vector field |

- `z_bed` is stored per output because the bed may move (sediment transport). It
  compresses to almost nothing when it does not.
- A dry column has all levels on the bed: `z_surface = z_bed`. REEF3D also writes
  columns with `z_surface < z_bed` (near the coast, FNPF): their levels fall, and
  readers must not assume that levels rise.
- **`z_offset`** is for solvers whose levels are not exactly at `sigma[k]` in every
  column (a solver that stretches its levels differently in some columns). It holds
  `z − (z_bed + sigma[k] (z_surface − z_bed))`, set to exactly 0 where that is
  below the tolerance of the writer (the converter: two float32 steps of z), so in
  the usual case it is all zeros. Its `fill_value` is 0, so tiles of zeros are not
  stored at all and an absent `z_offset` means 0 everywhere.
- `sigma` is the writer's nominal σ of each level (the converter: the median over
  the wet columns of the first output).

## 3a. Cartesian grids (`grid: "cartesian"`): CFD volumes

The levels lie at the same heights in every column and do not move, so their
heights are a coordinate array of the output group, like `x` and `y`:

| array | shape  | dtype   | content |
|-------|--------|---------|---------|
| `z`   | (nz,)  | float64 | z of the levels, rising |

A block holds only the fields, `(nt, nz_b, ny_b, nx_b[, 3])`; there are no
per-output heights. MPI splits CFD grids in z as well, so a block of a Cartesian
grid may hold a range of levels: its `start` and `size` then have three entries,
`[i0, j0, k0]` and `[nx_b, ny_b, nz_b]` (two entries: all levels). Overlapping
blocks are resolved as in 2.3 (the lowest numbered wins). Points inside solid bodies keep the values the solver writes (REEF3D's
VTU output gives them values too).

## 4. Structured surfaces (`grid: "surface"`): free surface and bed

One point per grid column (no `sigma`, no `level` dimension):

| array   | shape                   | dtype   | content |
|---------|-------------------------|---------|---------|
| `z`     | (nt, ny_b, nx_b)        | float32 | z of the surface point (the VTP point's z) |
| *field* | (nt, ny_b, nx_b[, 3])   | float32 | as in the VTP, e.g. `eta`, `depth`, `velocity` |

The surface is drawn as the quadrilaterals between neighbouring grid points that
all have a value.

REEF3D writes the store itself with `P 18` (1: instead of the VTU/VTP files of
what it holds, 2: as well as them). It writes the free surface and bed (FNPF, NHFLOW, and
CFD's topography as `bed`): each rank's VTP piece becomes its block; `P 18 1`
leaves out the surface VTP files.

## 5. Chunks, shards and codecs

Every data array of a block uses one chunk layout:

- **Shard** (Zarr's `chunk_grid`, one file): `T` outputs of the whole block, shape
  `(T, nz, NY, NX[, 3])`, where `NY`, `NX` are `ny_b`, `nx_b` rounded up to a
  multiple of the inner chunk. The converter uses `T = 16`; a writer that writes one
  output at a time may use `T = 1`.
- **Inner chunk** (inside the shard, Zarr's `sharding_indexed` codec): one output,
  a few levels and a tile of the block, shape `(1, cz, cy, cx[, 3])` (`(1, cy, cx[, 3])`
  for surfaces). The writers' rule: `cy` at most 64 and `cx` at most 64 (more when the
  block is narrow in y, up to 4096 / `cy`), both chosen so the tiles split the block
  evenly; `cz` = 1 when a tile holds at least 2048 points, else as many levels as make
  it hold 2048. A probe, a slice or the faces of a volume decompress only the levels
  and tiles they touch. Readers take the chunk shapes from the metadata.
- **Inner codecs**: `bytes` (little endian), `numcodecs.shuffle` (`elementsize` 4:
  the first bytes of all values, then the second bytes, ...), `gzip` (level 1).
  Shuffling makes the slowly varying exponents and leading mantissa bytes of a
  smooth field compress well; level 1 is nearly as small as level 9 and much faster.
- **Index**: at the end of the shard, codecs `bytes` (little endian) only.
- `fill_value`: `NaN` for float arrays (`0` for `z_offset`), `0` for integers. A chunk that is not
  written (e.g. a tile with no points) reads as fill values.
- Coordinate and time arrays (`x`, `y`, `sigma`, `time`, `step`) are single
  uncompressed chunks (`time` and `step` may use chunks of 4096 outputs).
- Chunk keys use the default encoding, e.g. `velocity/c/0/0/0/0/0`.

Example `zarr.json` of a block array:

```json
{"zarr_format": 3, "node_type": "array",
 "shape": [5024, 11, 151, 176, 3], "data_type": "float32",
 "chunk_grid": {"name": "regular", "configuration": {"chunk_shape": [16, 11, 153, 177, 3]}},
 "chunk_key_encoding": {"name": "default", "configuration": {"separator": "/"}},
 "fill_value": "NaN",
 "codecs": [{"name": "sharding_indexed", "configuration": {
     "chunk_shape": [1, 1, 51, 59, 3],
     "codecs": [{"name": "bytes", "configuration": {"endian": "little"}},
                {"name": "numcodecs.shuffle", "configuration": {"elementsize": 4}},
                {"name": "gzip", "configuration": {"level": 1}}],
     "index_codecs": [{"name": "bytes", "configuration": {"endian": "little"}}],
     "index_location": "end"}}],
 "dimension_names": ["time", "level", "y", "x", "component"],
 "attributes": {"units": "m/s"}}
```

`numcodecs.shuffle` is the name zarr-python (with numcodecs) uses for this codec;
it is the only codec outside the Zarr v3 core.

### 5.1 Writing while the run goes on

- Each rank writes its block's shard files. A shard of `T` outputs is written
  once at the end of its `T` outputs, or grows: the new inner chunk is written
  where the old index began and the index (with the new entry) after it.
  Entries of chunks not yet written are `2^64 − 1` (offset and size), as Zarr
  specifies for empty chunks.
- A block's array `zarr.json` files are rewritten with the new `shape[0]` by the
  rank that owns the block (write to a temporary file, then rename).
- Rank 0 extends `time` and `step` (and their `shape`) last.

## 6. Converting REEF3D VTK output

`lagoon-convert <case>` (LAGOON application) writes `REEF3D_<SOLVER>.lagoon`:

- **Volume (VTU)**: the grid lines are the distinct x and y of the points (to the
  millimetre, as float32 coordinates of different ranks differ in the last
  digits); each VTU piece becomes a block. A column's points, in the order of the
  file (REEF3D writes the levels from the bed up), are its levels; `z_bed` and
  `z_surface` are the first and the last; `sigma` and `z_offset` follow
  (section 3). (Not sorted by height: near the coast FNPF writes columns whose
  levels do not rise.)
- **Surfaces (VTP)**: each point becomes the grid point of its column. A surface
  whose points are not on a grid (CFD's free surface, an iso-surface of the level
  set) is left out.
- The time of every output comes from the run log, else from the `TimeValue`
  field data, else from `P 30` (output interval) times the output number.
- The fields are stored as they are in the VTK files; `z` of the points is kept
  through `z_bed`, `z_surface`, `z_offset` (volume) or `z` (surfaces).
- Converting again appends the outputs written since (a running simulation); a
  store of another run is not appended to.
- **Time series** (`.dat` tables of the solver's folders) become tables (section 7);
  a table whose file has changed since is written again.
- **Wave profiles** (WSFLINE) become profile sets (section 8); converting again
  appends the profiles written since.
- **Floating bodies** (6DOF VTP, one file per body and output) become a body set
  (section 9): the first output's vertices and triangles, then a fitted rigid
  motion per output. A body whose vertices are once not where the motion puts
  them is written again, from the first output, with its vertices per output.
- **Particles and objects** (CPM particles `REEF3D_CFD_CPM_Particle`, NHFLOW
  particles, DEM elements, ice floes, rod trees, mooring lines) become particle
  sets (section 10): the files of an output (MPI ranks, lines) are joined in the
  order of their numbers, their cells renumbered; converting again appends.

## 7. Time series tables

Gauges, forces and motions are tables: rows at times, one column per quantity
(REEF3D's `.dat` files: wave gauges `REEF3D_*_WSF`, `REEF3D_*_6DOF` forces,
positions, velocities and mooring forces, force files). Each is a group
`series/<key>`, the key named after the file (`nhflow_wsf_hg`,
`nhflow_6dof_forces_0`), with the attributes

```json
{"lagoon": {"kind": "table", "name": "Wave gauges", "group": "wave gauges",
            "columns": ["P1", "P2", "P3"], "units": {"P1": "m"},
            "locations": {"P1": [2.5, 0.5], "P2": [5.0, 0.5]},
            "rows": 20001, "source": "REEF3D_NHFLOW_WSF/REEF3D-NHFLOW-WSF-HG.dat"}}
```

- `columns`: the quantities in the order of `values`; `units` optional per column.
- `locations`: optional, where a column was measured (x, y[, z]).
- `group`: what kind of table (`wave gauges`, `6DOF`, `forces`, `probes`, ...).
- `source`: optional, the file it was converted from; the converter also keeps its
  size and modification time (`source_bytes`, `source_mtime`) to know when the
  file has grown.

| array    | shape       | dtype   | content |
|----------|-------------|---------|---------|
| `time`   | (n,)        | float64 | time of each row [s] |
| `values` | (n, ncols)  | float64 | the columns; NaN where the file has no value (REEF3D's ±1e20) |

Chunks: 65536 rows of one column (a plot reads only the columns it shows), codecs
`bytes`, `numcodecs.shuffle` (`elementsize` 8), `gzip` level 1.

## 8. Wave profiles

The free surface along lines at many times (REEF3D's `REEF3D_*_WSFLINE` files,
one file per time) is a group `profiles/<key>` with

```json
{"lagoon": {"kind": "profiles", "lines": {"1": {"y": 0.005, "points": 1200}},
            "source": "REEF3D_FNPF_WSFLINE"}}
```

| array            | shape        | dtype   | content |
|------------------|--------------|---------|---------|
| `time`           | (nt,)        | float64 | time of each profile [s]; counts the profiles |
| `line_<n>/x`     | (points,)    | float64 | position along line n |
| `line_<n>/eta`   | (nt, points) | float32 | computed free surface elevation |
| `line_<n>/theory`| (nt, points) | float32 | optional: wave theory (FNPF) |

Chunks: 16 times of a line; codecs as for fields (`elementsize` 4). As for
outputs, `time` is extended last when profiles are appended.

## 9. Floating bodies

REEF3D writes the whole triangle mesh of every 6DOF body at every output
(`REEF3D_*_6DOF_VTP/REEF3D-6DOF-<body>-<output>.vtp`, each triangle with its own
three points). A body moves rigidly, so a store keeps the mesh once and a rigid
motion per output. The bodies of one output stream are a group
`bodies/<key>`, the key named as LAGOON names the output (`nhflow_body`):

```json
{"lagoon": {"kind": "bodies", "bodies": [0, 1], "dataset": "nhflow_body",
            "source": "REEF3D_NHFLOW_6DOF_VTP"}}
```

| array    | shape | dtype   | content |
|----------|-------|---------|---------|
| `time`   | (nt,) | float64 | time of each output [s]; counts the outputs, extended last |
| `step`   | (nt,) | int64   | REEF3D output number |

and per body a group `body_<n>` with `{"lagoon": {"rigid": true, "vertices": 75696,
"fields": {}, "max_error": 6.1e-05}}`:

| array         | shape                         | dtype   | content |
|---------------|-------------------------------|---------|---------|
| `triangles`   | (ntri, 3)                     | int32   | vertex indices (polygons fan-triangulated) |
| `vertices`    | rigid: (nv, 3); else (nt, nv, 3) | float32 | rigid: the reference vertices (converter: those of the first output; REEF3D: relative to the centre of gravity at the start); not rigid: per output |
| `translation` | (nt, 3)                       | float64 | rigid only: t [m] |
| `rotation`    | (nt, 4)                       | float64 | rigid only: unit quaternion (w, x, y, z), w ≥ 0 |
| `<field>`     | (nt, nv) or (nt, nv, c)       | float32 | optional point arrays per output (`fields`: name → components) |

The vertices of a rigid body at output t are `R(q_t) v + t_t` (computed in
float64, then rounded to float32), `v` the stored vertices and `R(q)` the
rotation matrix of the quaternion. The motion is fitted to the written vertices
(least squares, Kabsch) and accepted when no vertex is further than 4 float32
steps of the largest coordinate from where it was written; `max_error` is the
largest distance of any accepted output [m]. Otherwise the body is not rigid
(`"rigid": false`) and keeps its vertices per output.

Chunks: 4096 outputs of the motion arrays, one output of vertices and fields;
codecs as for fields (`elementsize` 4 or 8).

REEF3D writes body sets itself with `P 18` (rank 0, `lagoon_bodies`): the mesh
relative to the centre of gravity at the start, the centre of gravity as
translation and its rotation matrix as quaternion, checked against the vertices
at every output; `P 18 1` leaves out the 6DOF VTP files.

## 10. Particles and objects

Particle clouds (CFD's CPM particles, NHFLOW's particles) and objects made of
points (DEM elements, ice floes, rod trees, mooring lines) have a different
number of points at every output. A *particle set* `particles/<key>` keeps the
points of all outputs one after another:

```json
{"lagoon": {"kind": "particles", "dataset": "cfd_particles", "role": "particles",
            "source": "REEF3D_CFD_CPM_Particle", "cells": null,
            "point_fields": {"velocity": {"components": 3, "data_type": "float32",
                                          "array": "velocity"},
                             "fluid velocity": {"components": 3, "data_type": "float32",
                                                "array": "fluid_velocity"}},
            "cell_fields": {}}}
```

- `cells`: `null` (points only), `"polys"` (polygons: DEM elements, ice floes) or
  `"lines"` (rods, mooring lines).
- `point_fields`, `cell_fields`: the arrays as written (name → components, data
  type, and the array that holds them: the name with characters other than
  `[A-Za-z0-9_.-]` replaced by `_`).
- `role`: what LAGOON shows it as (`particles`, `dem`, `ice`, `rods`, `mooring`).

| array                 | shape        | dtype   | content |
|-----------------------|--------------|---------|---------|
| `time`                | (nt,)        | float64 | time of each output [s]; counts the outputs, extended last |
| `step`                | (nt,)        | int64   | output number |
| `point_end`           | (nt,)        | int64   | end of each output's points: output t has points `point_end[t-1]` … `point_end[t]-1` |
| `position`            | (np, 3)      | float32 | points of all outputs [m] |
| `point_data/<array>`  | (np[, c])    | as written | point fields |
| `cell_set`            | (nt,)        | int64   | with cells: the cell set of each output |
| `cell_sets/cell_end`, `cell_sets/connectivity_end` | (ns,) | int64 | end of each cell set's cells and corners |
| `cell_sets/offsets`   | (nc,)        | int32   | VTK offsets (end of each cell's corners) within its set |
| `cell_sets/connectivity` | (nk,)     | int32   | corners, as the output's point numbers (0 is its first point) |
| `cell_data_end`       | (nt,)        | int64   | with cell fields: end of each output's cells in the cell fields |
| `cell_data/<array>`   | (ncd[, c])   | as written | cell fields (one value per floe, per rod) |

A cell set is stored only when an output's cells differ from those of the output
before; objects that keep their shape (DEM elements, rods, mooring lines) have
one. The growing arrays are sharded: shards of 1,048,576 rows, inner chunks of
65,536 rows, codecs as for fields. Writers keep rows and write them in large
pieces; `time` is extended last.

REEF3D writes particle sets itself with `P 18` (rank 0, `lagoon_particles`):
NHFLOW's particles (`nhflow_particles`), DEM elements (`dem_dem`) and FNPF's ice
floes (`fnpf_ice`). It writes every output at once, the inner chunk being filled
again with each output, so a store can be read during a run; `P 18 1` leaves out
their VTP files.

## 11. To be specified

- Particle tracks by particle id (from `id` fields) for path lines.
