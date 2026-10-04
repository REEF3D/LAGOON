# SPDX-License-Identifier: MIT
# Architect: Hans Bihs
"""ParaView reader plugin for LAGOON stores (lagoon-format SPEC.md).

Load it in ParaView with Tools > Manage Plugins > Load New (this file), then open
the ``zarr.json`` at the top of a store (``REEF3D_FNPF.lagoon/zarr.json``) and pick
"LAGOON store". Each output (volume, free surface, bed) becomes a group of
structured grids, one per block, with the store's times on the time bar.

The plugin needs NumPy (ParaView brings it) and the ``lagoon_format`` package: it
is found next to this file (``../src``) or wherever Python finds it.
"""

from __future__ import annotations

import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
if (_HERE.parent / "src" / "lagoon_format").is_dir():
    sys.path.insert(0, str(_HERE.parent / "src"))

import numpy as np  # noqa: E402
from lagoon_format.store import Store  # noqa: E402


def frame_index(times: np.ndarray, t: float) -> int | None:
    """The output shown at time ``t``: the last one at or before it."""
    if not len(times):
        return None
    index = int(np.searchsorted(times, t + 1e-9, side="right")) - 1
    return max(index, 0)


def structured_block(output, block, t: int) -> tuple[tuple[int, int, int], np.ndarray, dict]:
    """One block at output ``t`` as a structured grid: dimensions (nx, ny, nz),
    points (n, 3) with x fastest, then y, then the level, and the point arrays."""
    (i0, j0), (i1, j1) = block.start, block.stop
    x = output.x[i0:i1]
    y = output.y[j0:j1]
    z = output.heights(t, block)
    if z.ndim == 2:
        z = z[None]
    nz = z.shape[0]
    gx = np.broadcast_to(x[None, None, :], z.shape)
    gy = np.broadcast_to(y[None, :, None], z.shape)
    points = np.stack([gx.ravel(), gy.ravel(), z.ravel()], axis=1).astype(np.float32)
    arrays = {}
    for name, components in output.variables.items():
        values = output.read_block(name, t, block)
        arrays[name] = values.reshape(-1, components) if components > 1 else values.ravel()
    return (len(x), len(y), nz), points, arrays


def all_times(store: Store) -> list[float]:
    times = set()
    for output in store.outputs.values():
        times.update(float(t) for t in output.times)
    return sorted(times)


try:
    from paraview.util.vtkAlgorithm import (
        VTKPythonAlgorithmBase,
        smdomain,
        smhint,
        smproperty,
        smproxy,
    )
except ImportError:  # outside ParaView: only the functions above
    smproxy = None

if smproxy is not None:
    from vtkmodules.util.numpy_support import numpy_to_vtk
    from vtkmodules.vtkCommonCore import vtkPoints
    from vtkmodules.vtkCommonDataModel import vtkMultiBlockDataSet, vtkStructuredGrid
    from vtkmodules.vtkCommonExecutionModel import vtkStreamingDemandDrivenPipeline as _sddp

    @smproxy.reader(
        name="LAGOONStoreReader",
        label="LAGOON store",
        extensions="json",
        file_description="LAGOON store (zarr.json at the top of a .lagoon folder)",
    )
    class LAGOONStoreReader(VTKPythonAlgorithmBase):
        def __init__(self):
            super().__init__(nInputPorts=0, nOutputPorts=1, outputType="vtkMultiBlockDataSet")
            self._path = None
            self._store = None

        @smproperty.stringvector(name="FileName")
        @smdomain.filelist()
        @smhint.filechooser(extensions="json", file_description="LAGOON store (zarr.json)")
        def SetFileName(self, name):
            path = Path(name)
            if path.name == "zarr.json":
                path = path.parent
            if path != self._path:
                self._path = path
                self._store = None
                self.Modified()

        def _open(self) -> Store:
            if self._store is None:
                self._store = Store(self._path)
            else:
                self._store.refresh()  # a run still writing
            return self._store

        @smproperty.doublevector(
            name="TimestepValues", information_only="1", si_class="vtkSITimeStepsProperty"
        )
        def GetTimestepValues(self):
            return all_times(self._open()) if self._path else []

        def RequestInformation(self, request, inInfoVec, outInfoVec):
            info = outInfoVec.GetInformationObject(0)
            info.Remove(_sddp.TIME_STEPS())
            info.Remove(_sddp.TIME_RANGE())
            times = all_times(self._open())
            for t in times:
                info.Append(_sddp.TIME_STEPS(), t)
            if times:
                info.Append(_sddp.TIME_RANGE(), times[0])
                info.Append(_sddp.TIME_RANGE(), times[-1])
            return 1

        def RequestData(self, request, inInfoVec, outInfoVec):
            store = self._open()
            info = outInfoVec.GetInformationObject(0)
            times = all_times(store)
            t = info.Get(_sddp.UPDATE_TIME_STEP()) if info.Has(_sddp.UPDATE_TIME_STEP()) else None
            if t is None:
                t = times[0] if times else 0.0
            result = vtkMultiBlockDataSet.GetData(outInfoVec, 0)
            result.SetNumberOfBlocks(len(store.outputs))
            for number, (name, output) in enumerate(store.outputs.items()):
                group = vtkMultiBlockDataSet()
                index = frame_index(output.times, t)
                if index is not None:
                    group.SetNumberOfBlocks(len(output.blocks))
                    for b, block in enumerate(output.blocks):
                        group.SetBlock(b, _grid(*structured_block(output, block, index)))
                        group.GetMetaData(b).Set(vtkMultiBlockDataSet.NAME(), f"block {b}")
                result.SetBlock(number, group)
                result.GetMetaData(number).Set(vtkMultiBlockDataSet.NAME(), name)
            return 1

    def _grid(dims, points, arrays) -> vtkStructuredGrid:
        grid = vtkStructuredGrid()
        grid.SetDimensions(*dims)
        vtk_points = vtkPoints()
        vtk_points.SetData(numpy_to_vtk(np.ascontiguousarray(points), deep=True))
        grid.SetPoints(vtk_points)
        for name, values in arrays.items():
            array = numpy_to_vtk(np.ascontiguousarray(values, dtype=np.float32), deep=True)
            array.SetName(name)
            grid.GetPointData().AddArray(array)
        return grid
