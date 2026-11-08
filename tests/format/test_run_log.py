"""The REEF3D run log (lagoon-format RUN.md): reading, ownership after restarts, checks."""

import json

from lagoon_format.run import check, main, read_case

STREAMS = {
    "fsf": {
        "role": "free_surface",
        "folder": "REEF3D_NHFLOW_VTP_FSF",
        "files": "REEF3D-NHFLOW-FSF-{step:08d}.pvtp",
        "format": "vtp",
        "pieces": 2,
    },
    "wsf": {
        "role": "gauges",
        "folder": "REEF3D_NHFLOW_WSF",
        "file": "REEF3D-NHFLOW-WSF-HG.dat",
        "format": "dat",
    },
}


def _write_case(root, crash=False):
    """A run of 6 outputs, then a restart that wrote outputs 0-1 again."""
    folder = root / "REEF3D_NHFLOW_VTP_FSF"
    folder.mkdir(parents=True)
    lines = []
    for run_id, started, steps, dt in (
        ("20261001T080000Z-aaaa", "2026-10-01T08:00:00Z", range(6), 0.5),
        ("20261003T154312Z-bbbb", "2026-10-03T15:43:12Z", range(2), 0.25),
    ):
        lines.append(
            {
                "type": "run",
                "format": "reef3d-run",
                "version": 1,
                "run": run_id,
                "solver": "NHFLOW",
                "started": started,
                "streams": STREAMS,
            }
        )
        for step in steps:
            (folder / f"REEF3D-NHFLOW-FSF-{step:08d}.pvtp").write_text("x")
            lines.append(
                {
                    "type": "output",
                    "run": run_id,
                    "step": step,
                    "time": step * dt,
                    "iteration": step * 10,
                    "streams": ["fsf"],
                }
            )
    lines.append({"type": "end", "run": "20261001T080000Z-aaaa", "status": "finished"})
    text = "".join(json.dumps(line) + "\n" for line in lines)
    if crash:
        text += '{"type": "output", "run": "2026'  # killed while writing
    (root / "REEF3D_Case").mkdir()
    (root / "REEF3D_Case" / "REEF3D_NHFLOW_run.jsonl").write_text(text)
    return root


def test_runs_outputs_and_times(tmp_path):
    log = read_case(_write_case(tmp_path / "case"))
    first, second = log.runs
    assert first.id.endswith("aaaa") and first.status == "finished"
    assert second.status == "running or killed"
    assert [o.time for o in first.outputs] == [0.0, 0.5, 1.0, 1.5, 2.0, 2.5]
    assert first.frames("fsf")[3][1].name == "REEF3D-NHFLOW-FSF-00000003.pvtp"


def test_a_restart_owns_the_files_it_overwrote(tmp_path):
    log = read_case(_write_case(tmp_path / "case"))
    first, second = log.runs
    assert [o.step for o, _ in log.surviving(first, "fsf")] == [2, 3, 4, 5]
    assert [o.step for o, _ in log.surviving(second, "fsf")] == [0, 1]
    owner, output, stream = log.owners()[first.frames("fsf")[0][1]]
    assert owner is second and output.time == 0.0 and stream == "fsf"
    assert check(log) == []


def test_damage_is_reported_not_fatal(tmp_path):
    root = _write_case(tmp_path / "case", crash=True)
    (root / "REEF3D_NHFLOW_VTP_FSF" / "REEF3D-NHFLOW-FSF-00000004.pvtp").unlink()
    log = read_case(root)
    assert len(log.runs) == 2
    problems = check(log)
    assert any("incomplete" in p for p in problems)
    assert any("missing" in p and "00000004" in p for p in problems)
    assert read_case(tmp_path) is None  # no log: readers fall back


def test_command_line(tmp_path, capsys):
    assert main([str(_write_case(tmp_path / "case"))]) == 0
    out = capsys.readouterr().out
    assert "6 outputs, t = 0 .. 2.5 s, finished" in out and "frames on disk: 4" in out


def test_the_example_in_the_specification_reads():
    from pathlib import Path

    from lagoon_format.run import read_log

    example = Path(__file__).parents[2] / "lagoon-format" / "examples" / "REEF3D_NHFLOW_run.jsonl"
    log = read_log(example)
    (run,) = log.runs
    assert run.solver == "NHFLOW" and run.status == "finished" and not log.warnings
    assert run.streams["moor0"]["first_step"] == 1
    assert [o.time for o in run.outputs] == [0.0, 0.25, 0.5]
    assert run.file_name("body0", run.outputs[2]) == "REEF3D-6DOF-0-000002.vtp"


def test_streams_registered_while_running(tmp_path):
    """REEF3D's writers register a stream when they first write it."""
    root = tmp_path / "case"
    (root / "REEF3D_Case").mkdir(parents=True)
    (root / "REEF3D_NHFLOW_6DOF_VTP").mkdir()
    run = "20261003T190624Z-5dbb"
    records = [
        {
            "type": "run",
            "format": "reef3d-run",
            "version": 1,
            "run": run,
            "solver": "NHFLOW",
            "started": "2026-10-03T19:06:24Z",
            "streams": {},
        },
        {
            "type": "stream",
            "run": run,
            "name": "body0",
            "stream": {
                "role": "floating_body",
                "folder": "REEF3D_NHFLOW_6DOF_VTP",
                "files": "REEF3D-6DOF-0-{step:06d}.vtp",
                "format": "vtp",
            },
        },
    ]
    for step in range(3):
        (root / "REEF3D_NHFLOW_6DOF_VTP" / f"REEF3D-6DOF-0-{step:06d}.vtp").write_text("x")
        records.append(
            {
                "type": "output",
                "run": run,
                "step": step,
                "time": 0.5 * step,
                "iteration": step,
                "streams": ["body0"],
            }
        )
    text = "".join(json.dumps(r) + "\n" for r in records)
    (root / "REEF3D_Case" / "REEF3D_NHFLOW_run.jsonl").write_text(text)
    log = read_case(root)
    (only,) = log.runs
    assert only.streams["body0"]["role"] == "floating_body"
    assert [o.time for o, _ in log.surviving(only, "body0")] == [0.0, 0.5, 1.0]
    assert check(log) == []


def test_a_log_written_by_reef3d():
    """Written by the patched REEF3D (hans_dev) on the NHFLOW Stokes5 tutorial:
    a run of 2.2 s, then a restart of 1.2 s in the same folder."""
    from pathlib import Path

    from lagoon_format.run import read_log

    path = (
        Path(__file__).parents[2]
        / "lagoon-format/examples/reef3d_stokes5_restart/REEF3D_Case/REEF3D_NHFLOW_run.jsonl"
    )
    log = read_log(path)
    assert not log.warnings
    first, second = log.runs
    assert first.status == second.status == "finished"
    assert set(first.streams) == {"wsf", "volume", "fsf", "bed", "wsfline"}
    assert first.streams["fsf"]["files"] == "REEF3D-NHFLOW-FSF-{step:08d}.pvtp"
    assert second.record["still_water_level"] == 4.01
    times = [o.time for o in second.outputs if "fsf" in o.streams]
    assert times == sorted(times) and times[0] == 0.0
    owners = log.owners()
    fsf0 = Path("REEF3D_NHFLOW_VTP_FSF/REEF3D-NHFLOW-FSF-00000000.pvtp")
    assert owners[fsf0][0] is second  # the restart overwrote output 0


def test_outputs_in_a_store(tmp_path):
    """P 18 1: the outputs go into the store, the log lists them as zarr streams. The
    volume is counted at the next volume output, after surface outputs of later times
    (lines as REEF3D's runlog::stored writes them)."""
    root = tmp_path / "case"
    (root / "REEF3D_Case").mkdir(parents=True)
    run = "20261007T082100Z-ef6a"

    def stream(name, group):
        spec = {"role": group, "folder": "REEF3D_CFD.lagoon", "group": group}
        return {"type": "stream", "run": run, "name": name, "stream": spec | {"format": "zarr"}}

    def output(name, step, time):
        return {
            "type": "output",
            "run": run,
            "step": step,
            "time": time,
            "iteration": int(time * 400),
            "streams": [name],
        }

    records = [
        {
            "type": "run",
            "format": "reef3d-run",
            "version": 1,
            "run": run,
            "solver": "CFD",
            "streams": {},
        },
        stream("lagoon_bed", "bed"),
        output("lagoon_bed", 0, 0.0),
        output("lagoon_bed", 1, 0.5),
        stream("lagoon_volume", "volume"),
        output("lagoon_volume", 0, 0.0),
        output("lagoon_bed", 2, 1.0),
        output("lagoon_volume", 1, 0.5),
        output("lagoon_volume", 2, 1.0),
        {"type": "end", "run": run, "time": 1.0, "status": "finished"},
    ]
    log_file = root / "REEF3D_Case" / "REEF3D_CFD_run.jsonl"
    log_file.write_text("".join(json.dumps(r) + "\n" for r in records))
    log = read_case(root)
    (found,) = log.runs
    assert found.streams["lagoon_volume"]["group"] == "volume"
    assert [o.time for o in found.outputs if "lagoon_volume" in o.streams] == [0.0, 0.5, 1.0]
    assert found.frames("lagoon_volume") == []  # no files: nothing on disk to own
    assert check(log) == []  # times go back only across streams
    assert main([str(root)]) == 0
    records.append(output("lagoon_volume", 3, 0.25))
    log_file.write_text("".join(json.dumps(r) + "\n" for r in records))
    assert any("lagoon_volume go backwards" in p for p in check(read_case(root)))


def test_a_growing_log_is_read_on_from_where_it_was(tmp_path):
    """A running simulation's log only grows: read again, only what was written since
    is read (opening a case read a long log a dozen times, a second each), the
    half-written last line too once it is complete; a log written anew (a new run in
    its place, shorter) is read from the start."""
    import lagoon_format.run as run_module

    root = _write_case(tmp_path / "case", crash=True)
    path = root / "REEF3D_Case" / "REEF3D_NHFLOW_run.jsonl"
    first = read_case(root)
    assert len(first.runs[1].outputs) == 2
    assert any("incomplete" in w for w in first.warnings)
    known = run_module._GROWING[str(path)]
    offset = known.offset
    with path.open("a") as stream:  # the half line completed, and one more output
        stream.write(
            '0Q", "step": 9, "time": 9.0}\n'.replace("0Q", "1003T154312Z-bbbb")
            + json.dumps(
                {
                    "type": "output",
                    "run": "20261003T154312Z-bbbb",
                    "step": 2,
                    "time": 0.5,
                    "streams": ["fsf"],
                }
            )
            + "\n"
        )
    second = read_case(root)
    assert known.offset > offset and run_module._GROWING[str(path)] is known
    outputs = second.runs[1].outputs
    assert [o.step for o in outputs][-1] == 2 and len(outputs) == 4
    assert not any("incomplete" in w for w in second.warnings)
    # written anew: read from the start
    _write_case(tmp_path / "other")
    path.write_text((tmp_path / "other" / "REEF3D_Case" / "REEF3D_NHFLOW_run.jsonl").read_text())
    third = read_case(root)
    assert [len(r.outputs) for r in third.runs] == [6, 2]
