# The LAGOON format

A LAGOON store (`REEF3D_<SOLVER>.lagoon`) holds a whole REEF3D run in one folder:
volume, free surface and bed, time series, wave profiles, floating bodies,
particles and objects, and AMR surfaces. It replaces the many VTU, VTP and `.dat`
files of a run: 4 to 9 times smaller for grids, often a thousand times for
floating bodies, and much faster to open, step through and probe. It is Zarr
version 3, so other tools can read it too.

## Making a store

- **In LAGOON**: *File → Convert to LAGOON format…* converts the open case in the
  background and opens it again from the store.
- **On the command line**:

    ```bash
    lagoon-convert ~/runs/flume                          # everything there is
    lagoon-convert ~/runs/flume --outputs volume,bodies  # some outputs only
    ```

    Converting again appends what a running simulation has written since.
- **By REEF3D itself**, while it runs: `P 18 1` writes the store instead of the
  VTU/VTP files, `P 18 2` as well as them.

## Opening a store

Open the case folder as usual: a store replaces the output it holds, unless the
VTK files have newer outputs (a run that goes on and was not converted again).
The items then say "(LAGOON)". A store folder can also be opened on its own.

ParaView reads stores with the plugin `lagoon-format/paraview/lagoon_reader.py`
(*Tools → Manage Plugins → Load New*), and Python with the `lagoon-format`
package (`pip install lagoon-format`, NumPy only):

```python
from lagoon_format.store import Store

store = Store("REEF3D_NHFLOW.lagoon")
u = store["volume"].read("velocity", 10)       # (nz, ny, nx, 3) at output 10
gauges = store.tables["nhflow_wsf_hg"]
gauges.time, gauges.column("P1")
```

More in the [format guide](../format/GUIDE.md), the
[specification](../format/SPEC.md) and the [run log](../format/RUN.md).
