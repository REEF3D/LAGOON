# lagoon-format

The LAGOON data format: a structured, chunked dataset layout (Zarr v3) for ocean and
coastal simulation output, designed for Cartesian and σ-grids, floating bodies,
particles and gauges.

This package holds

- the format **specification** ([SPEC.md](SPEC.md), version 0.2: σ-grid volumes of
  FNPF and NHFLOW, Cartesian volumes of CFD, structured free surface and bed
  surfaces, time series tables, wave profiles, floating bodies, particles and
  objects such as DEM elements and mooring lines);
- the **reference reader and writer** (`lagoon_format.store`), with a small Zarr v3
  implementation (`lagoon_format.zarr3`) that needs only NumPy and the standard
  library; stores also open in zarr-python and xarray;
- the **run log** specification ([RUN.md](RUN.md): which outputs belong to which
  REEF3D run, and their times) with its reference reader
  (`python -m lagoon_format.run <case>` checks a case).

```python
from lagoon_format.store import Store

store = Store("case/REEF3D_FNPF.lagoon")
volume = store["volume"]
u = volume.read("velocity", 10)              # (nz, ny, nx, 3) at output 10
eta = store["free_surface"].series("eta", i=40, j=12)   # over all outputs
gauges = store.tables["fnpf_wsf_hg"]; p1 = gauges.column("P1")   # a wave gauge
vertices, triangles = store.bodies["nhflow_body"].mesh(10)       # bodies at output 10
xyz = store.particles["cfd_particles"].positions(10)              # particles at output 10
```

REEF3D output is converted with `lagoon-convert CASE` (LAGOON application);
REEF3D can also write stores itself during a run (`P 18`, see the REEF3D patch).

**ParaView**: load `paraview/lagoon_reader.py` with *Tools → Manage Plugins → Load
New*, then open the `zarr.json` at the top of a store and choose "LAGOON store".

It is MIT-licensed so that other codes and tools can adopt the format freely; the
LAGOON application itself is GPLv3.
