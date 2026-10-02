# LAGOON data format specification

Version: 0.0 (draft stub, not yet usable)

## Scope

One dataset per simulation run, stored as a Zarr store. Structured grids are
stored as coordinate arrays, not as cells:

- Cartesian grids: 1D coordinates x, y, z
- σ-grids: 1D coordinates x, y, σ, plus bathymetry d(x, y) and free surface
  η(x, y, t); the vertical position follows from z = σ (η + d) − d
- Bodies: geometry stored once (triangle mesh), plus a pose per time step
- Gauges, probes and forces: time series in the same store
- Metadata: solver, version, case settings (ctrl.txt), units

Chunks map onto the MPI decomposition so that every rank writes its own chunks.

## To be specified (phase 3)

- Group and array names, attribute conventions (CF-style names and units)
- Chunking rules and compression
- Body, particle and rod-tree layouts
- Versioning and compatibility rules
