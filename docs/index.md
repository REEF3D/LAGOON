# LAGOON

**Lightweight Animation and Graphics for Open Ocean Numerics**: the post-processor
of [REEF3D](https://github.com/REEF3D). A desktop application for looking at,
plotting and rendering ocean and coastal simulations, in the spirit of ParaView
and Blender but made for hydrodynamics.

![The example flume in the open ocean look](images/look-open-ocean.png)

<div class="grid cards" markdown>

- **Open a run** — point LAGOON at a REEF3D case folder: free surface, bed,
  volume, wave gauges, wave profiles, floating bodies and moorings appear by
  themselves. [Opening runs](guide/runs.md)
- **Place** — start a case on the map, draw its domain there and make DIVEMesh's
  `geo.dat` from bathymetry downloaded for it (Norway, Europe, the world); maps
  and satellite images under the results. [Maps](guide/maps.md)
- **Explore** — a GPU 3D view with Blender's navigation, slices, iso-surfaces,
  streamlines, contour lines and vectors, all linked to the time series under it.
  [The 3D view](guide/viewport.md)
- **Publish** — paper figures in a journal's style, drawn again with matplotlib at
  the width of a column; templates make the same figure for the next run.
  [Paper figures](guide/figures.md)
- **Show** — films with smooth camera paths, cinematic looks, and path-traced
  images and films through Blender's Cycles.
  [Films](guide/films.md) · [Path tracing](guide/cinematic.md)
- **Script** — everything the window does is a Python command; a session writes
  itself out as a script that makes the same figures and films without a screen.
  [Scripting](scripting.md)
- **Store** — the LAGOON format keeps a run several times smaller and opens it
  faster, written by `lagoon-convert` or by REEF3D itself.
  [The LAGOON format](guide/format.md)

</div>

## Quick start

```bash
pip install lagoon-viz
lagoon ~/runs/flume          # a REEF3D case folder
```

Then [First steps](guide/first-steps.md) walks through a run.

## What LAGOON reads

REEF3D's NHFLOW, FNPF, SFLOW and CFD output (VTU/VTP files, one per MPI rank and
output, and their parallel `.pvtu`/`.pvtp` files), AMR surfaces, the wave gauge,
wave profile, 6DOF, force and mooring files, DEM, ice and particle output, the
run log, and LAGOON stores. Other codes can write the
[LAGOON format](format/SPEC.md) to be read the same way.

## License

LAGOON is free software under the GNU General Public License v3.0 or later. The
LAGOON format library (`lagoon-format`) is MIT-licensed, so that any code may read
and write the format.
