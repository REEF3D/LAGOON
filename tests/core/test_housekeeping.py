"""Putting away what in a case folder is not the run shown."""

import os
import shutil
import time

from reef3d_fixtures import make_run

from lagoonviz.core.housekeeping import ARCHIVE, archive, human_size, inventory, remove


def _age(folder, seconds, names=None):
    when = time.time() - seconds
    for path in folder.rglob("*"):
        if path.is_file() and (names is None or any(n in path.name for n in names)):
            os.utime(path, (when, when))


def mixed_folder(tmp_path):
    """A CFD run going on (outputs 0, 1) over the leftovers of a longer CFD run
    (outputs 2, 3), an NHFLOW run of the day before and an earlier store."""
    root = make_run(tmp_path / "case", solver="CFD", frames=4, volume=True)
    for name in ("REEF3D_CFD_VTP_FSF", "REEF3D_CFD_VTP_BED"):
        shutil.rmtree(root / name)
    (root / "ctrl.txt").write_text("A 10 6\nP 30 0.5\n")
    _age(root / "REEF3D_CFD_VTU", 7200, ["-00000002", "-00000003"])  # left over
    make_run(root, solver="NHFLOW", frames=2)
    _age(root, 86400, ["NHFLOW"])
    earlier = root / "REEF3D_CFD-20261016T212950Z-f289.lagoon-earlier"
    earlier.mkdir()
    (earlier / "zarr.json").write_text("{}")
    return root


def test_the_inventory(tmp_path):
    root = mixed_folder(tmp_path)
    groups = {(g.kind, g.label.split(",")[0]): g for g in inventory(root)}
    leftovers = groups[("earlier run", "CFD volume")]
    assert leftovers.suggested and "outputs 2–3" in leftovers.label
    assert all("-0000000" in p.name for p in leftovers.paths)
    steps = {p.name.split("-")[2][:8] for p in leftovers.paths}
    assert steps == {"00000002", "00000003"} and any(p.suffix == ".pvtu" for p in leftovers.paths)
    nhflow = [g for (kind, _), g in groups.items() if kind == "other run"]
    assert {g.label.split(",")[0] for g in nhflow} == {
        "NHFLOW free surface", "NHFLOW bed", "NHFLOW volume"
    }  # fmt: skip
    assert not any(g.suggested for g in nhflow)  # another solver's run: not ticked
    store = groups[("earlier store", "REEF3D_CFD-20261016T212950Z-f289.lagoon-earlier")]
    assert store.suggested and store.size > 0
    # the run shown is never offered
    offered = {p for g in groups.values() for p in g.paths}
    assert not any("-00000000" in p.name and "CFD" in p.name for p in offered)


def test_archive_and_remove(tmp_path):
    root = mixed_folder(tmp_path)
    groups = inventory(root)
    ticked = [g for g in groups if g.suggested]
    target = archive(ticked, root)
    assert target.parent == root / ARCHIVE
    assert (target / "REEF3D_CFD_VTU" / "REEF3D-CFD-00000002.pvtu").is_file()
    assert not (root / "REEF3D_CFD_VTU" / "REEF3D-CFD-00000002.pvtu").exists()
    assert (root / "REEF3D_CFD_VTU" / "REEF3D-CFD-00000001.pvtu").is_file()  # the run shown
    assert not any(root.glob("*.lagoon-earlier"))
    others = [g for g in inventory(root) if g.kind == "other run"]
    freed = remove(others, root)
    assert freed > 0 and not (root / "REEF3D_NHFLOW_VTU").exists()  # emptied folders go
    assert [g.kind for g in inventory(root)] == []


def test_sizes():
    assert human_size(512) == "512 B" and human_size(1_500_000) == "1.5 MB"
