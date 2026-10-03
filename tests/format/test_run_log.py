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
    (root / "REEF3D_NHFLOW_run.jsonl").write_text(text)
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
