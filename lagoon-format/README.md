# lagoon-format

The LAGOON data format: a structured, chunked dataset layout (Zarr) for ocean and
coastal simulation output, designed for Cartesian and σ-grids, floating bodies,
particles and gauges.

This package holds the format **specification** (`SPEC.md`), the **run log**
specification (`RUN.md`: which outputs belong to which REEF3D run, and their times)
with its reference reader (`python -m lagoon_format.run <case>` checks a case), and
the reference reader/writer. It is MIT-licensed so that other codes and tools can adopt the
format freely; the LAGOON application itself is GPLv3.

Status: phase 0. The specification is a stub; the layout is designed in phase 3
of the LAGOON implementation plan.
