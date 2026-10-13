"""Numbered output files grouped into series, as ParaView lists them."""

from lagoonviz.core.data.file_groups import file_groups


def test_a_folder_of_parallel_output(tmp_path):
    folder = tmp_path / "REEF3D_NHFLOW_VTP_FSF"
    folder.mkdir()
    for step in range(3):
        (folder / f"REEF3D-NHFLOW-FSF-{step:08d}.pvtp").write_text("x" * 10)
        for piece in range(2):
            (folder / f"REEF3D-NHFLOW-FSF-{piece:05d}-{step:08d}.vtp").write_text("y")
    (folder / "notes.txt").write_text("not output")
    (folder / "REEF3D-NHFLOW-FSF-mean.pvtp").write_text("z")
    (folder / "sub").mkdir()
    (folder / ".hidden.pvtp").write_text("h")
    groups = file_groups(folder)
    assert [(g.kind, g.label) for g in groups] == [
        ("folder", "sub"),
        ("series", "REEF3D-NHFLOW-FSF-..pvtp"),
        ("file", "REEF3D-NHFLOW-FSF-mean.pvtp"),
    ]
    series = groups[1]
    assert len(series.paths) == 3 and (series.first, series.last) == (0, 2)
    assert series.path.name == "REEF3D-NHFLOW-FSF-00000000.pvtp"
    assert series.describe() == "3 files, 0 – 2, 30 B"


def test_serial_files_are_series_too(tmp_path):
    for step in (10, 2, 1):
        (tmp_path / f"REEF3D-6DOF-0-{step:06d}.vtp").write_text("a")
    (tmp_path / "REEF3D-CFD-000001.vtu").write_text("b")
    groups = file_groups(tmp_path)
    assert [g.label for g in groups] == ["REEF3D-6DOF-0-..vtp", "REEF3D-CFD-000001.vtu"]
    assert [p.name[-10:] for p in groups[0].paths] == ["000001.vtp", "000002.vtp", "000010.vtp"]
    assert file_groups(tmp_path / "missing") == []
