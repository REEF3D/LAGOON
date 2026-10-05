# SPDX-License-Identifier: MIT
# Architect: Hans Bihs
"""ParaView reader plugin for LAGOON stores (lagoon-format SPEC.md).

Load it in ParaView with Tools > Manage Plugins > Load New (this file), then open
the ``zarr.json`` at the top of a store (``REEF3D_FNPF.lagoon/zarr.json``) and pick
"LAGOON store". Each output (volume, free surface, bed) becomes a group of
structured grids, one per block; floating bodies become polygon meshes, particles
points (with vertex cells) and objects (DEM elements, ice floes, rods, mooring
lines) polygons or lines; an AMR surface a group of rectilinear grids (level 0 and
the patches) whose covered cells are hidden (``vtkGhostType``), so only the leaf
cells show; the store's times are on the time bar.

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
from lagoon_format.amr import covered  # noqa: E402
from lagoon_format.store import Store  # noqa: E402

#: vtkDataSetAttributes::HIDDENCELL: cells ParaView does not draw
HIDDEN_CELL = 32


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


def body_mesh(bodies, t: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """All bodies of a body set at output ``t``: points, VTK connectivity and offsets
    of their triangles, and point arrays."""
    points, triangles = bodies.mesh(t)
    arrays = {}
    for name in sorted({n for b in bodies.bodies.values() for n in b.fields}):
        parts = [b.field(name, t) for _, b in sorted(bodies.bodies.items()) if name in b.fields]
        if len(parts) == len(bodies.bodies):
            arrays[name] = np.concatenate(parts)
    connectivity = np.asarray(triangles, np.int64).ravel()
    offsets = np.arange(3, len(connectivity) + 1, 3, dtype=np.int64)
    return points, connectivity, offsets, arrays


def particle_cells(particles, t: int) -> tuple[str, np.ndarray, np.ndarray, dict, dict]:
    """Output ``t`` of a particle set: the kind of cells (``verts`` for points only,
    one vertex cell per point, so ParaView draws them), connectivity, offsets,
    point arrays and cell arrays."""
    n = particles.count(t)
    points = {name: particles.point_field(name, t) for name in particles.point_fields}
    if not particles.cells:
        return (
            "verts",
            np.arange(n, dtype=np.int64),
            np.arange(1, n + 1, dtype=np.int64),
            points,
            {},
        )
    connectivity, offsets = particles.cell_set(t)
    cells = {name: particles.cell_field(name, t) for name in particles.cell_fields}
    return particles.cells, connectivity, offsets, points, cells


def amr_grids(amr, t: int) -> list[tuple[object, dict, np.ndarray]]:
    """The grids of an AMR set's output t: (grid, cell arrays with ``level``, ghost
    flags hiding the cells a finer grid covers), level 0 first."""
    grids = amr.grids(t)
    found = []
    for grid in grids:
        cells = {
            name: values.reshape(grid.nx * grid.ny, *values.shape[2:])
            for name, values in grid.fields.items()
        }
        cells["level"] = np.full(grid.nx * grid.ny, grid.level, np.int32)
        ghosts = np.where(covered(grid, grids).ravel(), HIDDEN_CELL, 0).astype(np.uint8)
        found.append((grid, cells, ghosts))
    return found


def all_times(store: Store) -> list[float]:
    times = set()
    for output in store.outputs.values():
        times.update(float(t) for t in output.times)
    for group in (*store.bodies.values(), *store.particles.values(), *store.amr.values()):
        times.update(float(t) for t in group.times)
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
    from vtkmodules.util.numpy_support import numpy_to_vtk, numpy_to_vtkIdTypeArray
    from vtkmodules.vtkCommonCore import vtkPoints
    from vtkmodules.vtkCommonDataModel import (
        vtkCellArray,
        vtkMultiBlockDataSet,
        vtkPolyData,
        vtkRectilinearGrid,
        vtkStructuredGrid,
    )
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
            result.SetNumberOfBlocks(
                len(store.outputs) + len(store.bodies) + len(store.particles) + len(store.amr)
            )
            number = -1
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
            for key, bodies in store.bodies.items():
                number += 1
                index = frame_index(bodies.times, t)
                if index is not None:
                    points, connectivity, offsets, arrays = body_mesh(bodies, index)
                    result.SetBlock(number, _poly(points, "polys", connectivity, offsets, arrays))
                result.GetMetaData(number).Set(vtkMultiBlockDataSet.NAME(), key)
            for key, particles in store.particles.items():
                number += 1
                index = frame_index(particles.times, t)
                if index is not None:
                    kind, connectivity, offsets, arrays, cells = particle_cells(particles, index)
                    poly = _poly(
                        particles.positions(index), kind, connectivity, offsets, arrays, cells
                    )
                    result.SetBlock(number, poly)
                result.GetMetaData(number).Set(vtkMultiBlockDataSet.NAME(), key)
            for key, amr in store.amr.items():
                number += 1
                group = vtkMultiBlockDataSet()
                index = frame_index(amr.times, t)
                if index is not None:
                    grids = amr_grids(amr, index)
                    group.SetNumberOfBlocks(len(grids))
                    for g, (grid, cells, ghosts) in enumerate(grids):
                        group.SetBlock(g, _rectilinear(grid, cells, ghosts))
                        name = f"level {grid.level}" + (
                            f" rank {grid.rank}" if grid.rank >= 0 else ""
                        )
                        group.GetMetaData(g).Set(vtkMultiBlockDataSet.NAME(), name)
                result.SetBlock(number, group)
                result.GetMetaData(number).Set(vtkMultiBlockDataSet.NAME(), key)
            return 1

    def _vtk_array(name, values):
        values = np.ascontiguousarray(values)
        if values.dtype.kind == "f":
            values = values.astype(np.float32, copy=False)
        array = numpy_to_vtk(values, deep=True)
        array.SetName(name)
        return array

    def _poly(points, kind, connectivity, offsets, arrays, cells=None) -> vtkPolyData:
        poly = vtkPolyData()
        vtk_points = vtkPoints()
        vtk_points.SetData(numpy_to_vtk(np.ascontiguousarray(points, np.float32), deep=True))
        poly.SetPoints(vtk_points)
        cell_array = vtkCellArray()
        starts = np.concatenate(([0], np.asarray(offsets, np.int64)))
        cell_array.SetData(  # VTK's offsets start with 0
            numpy_to_vtkIdTypeArray(starts.astype(np.int64), deep=True),
            numpy_to_vtkIdTypeArray(np.asarray(connectivity, np.int64), deep=True),
        )
        {"polys": poly.SetPolys, "lines": poly.SetLines, "verts": poly.SetVerts}[kind](cell_array)
        for name, values in arrays.items():
            poly.GetPointData().AddArray(_vtk_array(name, values))
        for name, values in (cells or {}).items():
            poly.GetCellData().AddArray(_vtk_array(name, values))
        return poly

    def _rectilinear(grid, cells, ghosts) -> vtkRectilinearGrid:
        out = vtkRectilinearGrid()
        out.SetDimensions(grid.nx + 1, grid.ny + 1, 1)
        out.SetXCoordinates(numpy_to_vtk(np.ascontiguousarray(grid.x, np.float64), deep=True))
        out.SetYCoordinates(numpy_to_vtk(np.ascontiguousarray(grid.y, np.float64), deep=True))
        out.SetZCoordinates(numpy_to_vtk(np.zeros(1, np.float64), deep=True))
        for name, values in cells.items():
            out.GetCellData().AddArray(_vtk_array(name, values))
        ghost = numpy_to_vtk(ghosts, deep=True)
        ghost.SetName("vtkGhostType")
        out.GetCellData().AddArray(ghost)
        return out

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
