"""File → Open output file: the files of a folder grouped into series."""

from lagoonviz.gui.open_output import OpenOutputDialog


def test_the_dialog_lists_series_and_opens_one(qtbot, tmp_path):
    folder = tmp_path / "REEF3D_NHFLOW_VTP_FSF"
    folder.mkdir()
    for step in range(4):
        (folder / f"REEF3D-NHFLOW-FSF-{step:08d}.pvtp").write_text("x")
        (folder / f"REEF3D-NHFLOW-FSF-00000-{step:08d}.vtp").write_text("y")
    dialog = OpenOutputDialog(tmp_path)
    qtbot.addWidget(dialog)
    names = [dialog.tree.topLevelItem(i).text(0) for i in range(dialog.tree.topLevelItemCount())]
    assert names == ["REEF3D_NHFLOW_VTP_FSF"]
    dialog._activate(dialog.tree.topLevelItem(0))  # into the folder
    assert dialog.folder == folder
    item = dialog.tree.currentItem()
    assert item.text(0) == "REEF3D-NHFLOW-FSF-..pvtp" and item.text(1).startswith("4 files")
    dialog._activate(item)
    assert dialog.chosen == folder / "REEF3D-NHFLOW-FSF-00000000.pvtp"
