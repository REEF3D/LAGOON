# REEF3D run log: which outputs belong to which run, at which time

Version: 1 (draft for review)
Status: proposed. LAGOON reads it when present and falls back to its heuristics
otherwise. The REEF3D side is a small patch: one line of text per output.

## Why

A REEF3D case folder holds the outputs of every run that was started in it.
Nothing in the files says which run wrote them, or at which time: LAGOON has to
guess from modification times, from overwritten outputs (a restart writes from
output 0 again), and from `P 30` in `ctrl.txt`. Guessing fails for variable
output intervals, for runs without `P 30`, for gauges and profiles written by
another run than the surfaces, and for two solvers in one folder.

The run log states it. It is meant for every output REEF3D writes today
(VTU/VTP, legacy VTK, `.dat` tables), and its content is the metadata layer of
the LAGOON format (Zarr, phase 3), so it is written once and kept.

## Two parts

1. **A run log per solver**: `REEF3D_<SOLVER>_run.jsonl` in the case folder (the
   folder that holds the `REEF3D_*` output folders). One JSON object per line.
2. **Run and time inside every VTK file**: three `FieldData` arrays, so a file
   that left its folder still says where it belongs.

## 1. The run log

### File

- Name: `REEF3D_<SOLVER>_run.jsonl`, e.g. `REEF3D_NHFLOW_run.jsonl`.
  `<SOLVER>` is the same word as in the output folders (`FNPF`, `NHFLOW`, `CFD`,
  `SFLOW`, ...).
- Encoding: UTF-8, [JSON Lines](https://jsonlines.org): one JSON object per line,
  `\n` at the end of every line.
- **Append only.** Rank 0 opens the file in append mode, writes a whole line and
  flushes. Lines are never rewritten, so a run that crashes leaves a valid log.
- A run started again in the same folder appends to the same file. The file is
  the history of the folder.
- Every object has a `"type"`. Readers ignore types and keys they do not know,
  and a last line without `\n` (a run killed while writing).

### Record `run`: when a run starts

Written once, before the first output.

```json
{"type": "run", "format": "reef3d-run", "version": 1,
 "run": "20261003T154312Z-7f3a",
 "solver": "NHFLOW", "reef3d": "25.02", "git": "a1b2c3d",
 "started": "2026-10-03T15:43:12Z", "host": "node017", "ranks": 4,
 "case": "floating_wind_turbine",
 "ctrl": {"file": "ctrl.txt", "sha256": "9c1e..."},
 "restart": {"from": "20261002T081500Z-c2d1", "time": 120.0},
 "units": {"length": "m", "time": "s", "mass": "kg", "force": "N"},
 "still_water_level": 56.0,
 "domain": {"min": [0.0, 0.0, 0.0], "max": [1760.0, 1440.0, 120.0]},
 "streams": { ... }}
```

| key | | meaning |
|---|---|---|
| `format`, `version` | required | `"reef3d-run"`, `1` |
| `run` | required | Unique id of this run: start time (UTC, `YYYYMMDDTHHMMSSZ`) and a random hex suffix. Sorts by time. |
| `solver` | required | As in the folder names |
| `reef3d`, `git` | optional | Version string and commit of the executable |
| `started` | required | Start time, ISO 8601 UTC |
| `host`, `ranks` | optional | Machine, number of MPI ranks |
| `case` | optional | A name for the case (default: the folder name) |
| `ctrl` | recommended | Control file and its SHA-256: same hash = same set-up |
| `restart` | optional | The run and time this one continues from (hot start) |
| `units` | optional | Default SI as shown |
| `still_water_level` | recommended | z of the still water surface |
| `domain` | optional | Bounding box of the domain |
| `streams` | required | What the run writes, see below |

### `streams`: what the run writes

A map from a short stream name (unique within the run) to its description.

```json
"streams": {
  "fsf":    {"role": "free_surface", "folder": "REEF3D_NHFLOW_VTP_FSF",
             "files": "REEF3D-NHFLOW-FSF-{step:08d}.pvtp", "format": "vtp",
             "pieces": 4, "variables": {"eta": {"unit": "m"},
                                        "velocity": {"components": 3, "unit": "m/s"}}},
  "bed":    {"role": "bed", "folder": "REEF3D_NHFLOW_VTP_BED",
             "files": "REEF3D-NHFLOW-BED-{step:08d}.pvtp", "format": "vtp",
             "pieces": 4, "static": true},
  "volume": {"role": "volume", "folder": "REEF3D_NHFLOW_VTU",
             "files": "REEF3D-NHFLOW-{step:08d}.pvtu", "format": "vtu", "pieces": 4},
  "body0":  {"role": "body", "object": 0, "folder": "REEF3D_NHFLOW_6DOF_VTP",
             "files": "REEF3D-6DOF-0-{step:06d}.vtp", "format": "vtp"},
  "moor1":  {"role": "mooring", "object": 1, "folder": "REEF3D_NHFLOW_6DOF_Mooring",
             "files": "REEF3D-Mooring-00000001-{step:06d}.vtk", "format": "vtk-legacy",
             "first_step": 1},
  "vrans":  {"role": "structure", "folder": "REEF3D_NHFLOW_VRANS_VTP",
             "file": "REEF3D-NHFLOW-VRANS.vtp", "format": "vtp", "static": true},
  "wsf":    {"role": "gauges", "folder": "REEF3D_NHFLOW_WSF",
             "file": "REEF3D-NHFLOW-WSF-HG.dat", "format": "dat",
             "gauges": [{"name": "P1", "x": 100.0, "y": 720.0}]},
  "lines":  {"role": "profiles", "folder": "REEF3D_NHFLOW_WSFLINE",
             "files": "REEF3D-NHFLOW-wsfline-{iteration:08d}.dat", "format": "dat",
             "lines": [{"number": 1, "y": 720.0}]},
  "forces0":{"role": "forces", "object": 0, "folder": "REEF3D_NHFLOW_6DOF",
             "file": "REEF3D_6DOF_forces_0.dat", "format": "dat"},
  "motion0":{"role": "motion", "object": 0, "folder": "REEF3D_NHFLOW_6DOF",
             "file": "REEF3D_6DOF_position_0.dat", "format": "dat"}
}
```

| key | | meaning |
|---|---|---|
| `role` | required | `free_surface`, `bed`, `volume`, `body`, `mooring`, `structure`, `particles`, `gauges`, `profiles`, `forces`, `motion`, `probes`, or a name of the solver's own (readers show unknown roles as generic data) |
| `folder` | required | Output folder, relative to the case folder |
| `files` | frames | File name pattern with `{step}` (output number) or `{iteration}`, Python/`fmt` style: `{step:08d}`. For parallel VTK it names the `.pvtu`/`.pvtp` file |
| `file` | tables, static | One file: a table that grows over the run (gauges, forces, motions), or static geometry (a structure) |
| `format` | required | `vtu`, `vtp`, `vtk-legacy`, `dat` (more later, e.g. `zarr`) |
| `pieces` | optional | MPI pieces per frame |
| `variables` | optional | Point arrays: `components` (default 1), `unit`, `description` |
| `static` | optional | `true` if the data does not change during the run (a fixed bed) |
| `object` | optional | Index of the body, mooring line or structure |
| `first_step` | optional | First output this stream writes, if not 0 |
| `gauges`, `lines` | optional | Positions of gauges and profile lines |

### Record `output`: every time the run writes output

One line per output step, after its files are complete (closed on all ranks).

```json
{"type": "output", "run": "20261003T154312Z-7f3a", "step": 110, "time": 27.5,
 "iteration": 5500, "streams": ["fsf", "volume", "body0", "moor1"]}
```

| key | | meaning |
|---|---|---|
| `run` | required | The run id |
| `step` | required | Output number, the `{step}` in the file names |
| `time` | required | Simulation time of this output, in seconds |
| `iteration` | optional | Solver iteration |
| `streams` | required | The frame streams written at this step (tables are not listed) |
| `files` | optional | Map stream → file name, when a name does not follow its pattern |

Streams that write at their own pace (profiles every N iterations) get their own
`output` lines with only those streams. Tables (`file`) are not listed: they grow
on their own, and their rows carry their time.

### Record `end`: when a run stops

```json
{"type": "end", "run": "20261003T154312Z-7f3a", "step": 696, "time": 174.0,
 "status": "finished", "wall_seconds": 51234}
```

`status`: `finished`, `stopped` (by the user or a time limit) or `error`. No
`end` record means the run is still going, or was killed; readers can tell the
two apart by the age of the last line.

## 2. Run and time inside VTK files

Every `.vtu`, `.vtp`, `.pvtu` and `.pvtp` file gets field data before its first
`<Piece>` (in the parallel files: inside `<PUnstructuredGrid>`/`<PPolyData>`):

```xml
<FieldData>
  <DataArray type="Float64" Name="TimeValue" NumberOfTuples="1" format="ascii">27.5</DataArray>
  <DataArray type="Int32" Name="Step" NumberOfTuples="1" format="ascii">110</DataArray>
  <DataArray type="String" Name="RunId" NumberOfTuples="1" format="ascii">20261003T154312Z-7f3a</DataArray>
</FieldData>
```

`TimeValue` is the name ParaView already uses: with it, ParaView shows the real
simulation time instead of the file index, so this helps every user. Legacy VTK
(mooring lines) gets the same as a `FIELD FieldData 3` block.

## How readers use it

1. Read every `REEF3D_*_run.jsonl` in the case folder. Each `run` record is a run.
2. A file belongs to the **last** run whose `output` records list it: a later
   run overwrites the files of an earlier one, and the log says which survived.
   Runs of which no file survived are history only.
3. Times come from `output` records (`step` → `time`), so variable output
   intervals and missing `P 30` need no special care.
4. Gauges, profiles, forces and motions belong to the run that lists them; their
   file holds the data of the run that wrote it last.
5. A file without a run log is read as today (modification times, `ctrl.txt`).
   A file whose `RunId` does not match the log was written by another tool or
   copied in; the reader warns and uses the file's own `TimeValue`.

## Relation to the LAGOON format (Zarr, phase 3)

The LAGOON format stores a run as one Zarr store. The run log is its metadata
layer, unchanged:

- the `run` record (with `streams`) is the root attribute `reef3d_run`;
- the `output` records are the arrays `/outputs/step`, `/outputs/time`,
  `/outputs/iteration` along the time dimension;
- each stream is a group, chunked in time and by MPI rank.

Writing the run log now therefore costs nothing later: the same records move
into the store, and REEF3D keeps writing them for VTK output as long as VTK is
written.

## Writing it in REEF3D (sketch)

Rank 0 only, after the output of a step is complete on all ranks:

```cpp
// once, at start
runlog.open(case_dir + "/REEF3D_" + solver + "_run.jsonl", std::ios::app);
runlog << run_record_json() << '\n' << std::flush;
// at every output
runlog << "{\"type\":\"output\",\"run\":\"" << run_id << "\",\"step\":" << step
       << ",\"time\":" << std::setprecision(17) << simtime
       << ",\"iteration\":" << iteration << ",\"streams\":[" << streams << "]}\n"
       << std::flush;
// at the end
runlog << "{\"type\":\"end\",...}\n" << std::flush;
```

Times are written with 17 significant digits (round trip for doubles).

## Versioning

`version` is an integer. New optional keys or record types keep the version;
a change that old readers would misread raises it. Readers accept any version
they know and warn on a higher one, reading what they understand.
