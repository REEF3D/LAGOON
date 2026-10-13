# SPDX-License-Identifier: GPL-3.0-or-later
# Architect: Hans Bihs
"""File → Open output file: a folder's output files grouped into series, as ParaView
lists them (``REEF3D-NHFLOW-FSF-..pvtp  ·  121 files, 0 – 120``).

Double-click a folder to go in, *Up* to go out; double-click a series (or a file)
to open it: the series it belongs to opens, with its case.
"""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QHBoxLayout,
    QHeaderView,
    QLineEdit,
    QPushButton,
    QStyle,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
)

from lagoonviz.core.data.file_groups import FileGroup, file_groups

ROLE = Qt.UserRole


class OpenOutputDialog(QDialog):
    """Pick a series of output files (or one file) in a folder (module doc)."""

    def __init__(self, start: str | Path, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Open REEF3D output")
        self.resize(720, 480)
        self.chosen: Path | None = None
        self.folder = Path(start).expanduser()
        if not self.folder.is_dir():
            self.folder = Path.home()

        self.place = QLineEdit()
        self.place.setToolTip("The folder shown (type a path and press Return)")
        self.place.returnPressed.connect(lambda: self.show_folder(self.place.text()))
        up = QPushButton("Up")
        up.setToolTip("The folder above")
        up.clicked.connect(lambda: self.show_folder(self.folder.parent))
        browse = QPushButton("Browse…")
        browse.clicked.connect(self._browse)
        top = QHBoxLayout()
        top.addWidget(up)
        top.addWidget(self.place, 1)
        top.addWidget(browse)

        self.tree = QTreeWidget()
        self.tree.setHeaderLabels(["Name", "Files"])
        self.tree.setRootIsDecorated(False)
        self.tree.setUniformRowHeights(True)
        self.tree.header().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        self.tree.header().setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        self.tree.itemDoubleClicked.connect(lambda item, _c: self._activate(item))
        self.tree.currentItemChanged.connect(lambda *_: self._update_buttons())

        self.buttons = QDialogButtonBox(QDialogButtonBox.Open | QDialogButtonBox.Cancel)
        self.buttons.accepted.connect(lambda: self._activate(self.tree.currentItem()))
        self.buttons.rejected.connect(self.reject)

        layout = QVBoxLayout(self)
        layout.addLayout(top)
        layout.addWidget(self.tree, 1)
        layout.addWidget(self.buttons)
        self.show_folder(self.folder)

    def show_folder(self, folder: str | Path) -> None:
        folder = Path(folder).expanduser()
        if not folder.is_dir():
            self.place.setText(str(self.folder))
            return
        self.folder = folder
        self.place.setText(str(folder))
        self.tree.clear()
        style = self.style()
        icons = {
            "folder": style.standardIcon(QStyle.SP_DirIcon),
            "series": style.standardIcon(QStyle.SP_FileDialogDetailedView),
            "file": style.standardIcon(QStyle.SP_FileIcon),
        }
        for group in file_groups(folder):
            item = QTreeWidgetItem(
                [group.label, group.describe() if group.kind != "folder" else ""]
            )
            item.setIcon(0, icons[group.kind])
            item.setData(0, ROLE, group)
            if group.kind == "series":
                item.setToolTip(0, f"{group.paths[0].name} … {group.paths[-1].name}")
            self.tree.addTopLevelItem(item)
        first_series = next(
            (
                self.tree.topLevelItem(i)
                for i in range(self.tree.topLevelItemCount())
                if self.tree.topLevelItem(i).data(0, ROLE).kind != "folder"
            ),
            None,
        )
        self.tree.setCurrentItem(first_series or self.tree.topLevelItem(0))
        self._update_buttons()

    def _group(self, item) -> FileGroup | None:
        return item.data(0, ROLE) if item is not None else None

    def _update_buttons(self) -> None:
        group = self._group(self.tree.currentItem())
        button = self.buttons.button(QDialogButtonBox.Open)
        button.setEnabled(group is not None)
        button.setText("Go in" if group is not None and group.kind == "folder" else "Open")

    def _activate(self, item) -> None:
        group = self._group(item)
        if group is None:
            return
        if group.kind == "folder":
            self.show_folder(group.path)
            return
        self.chosen = group.path
        self.accept()

    def _browse(self) -> None:
        chosen = QFileDialog.getExistingDirectory(self, "Folder of the output", str(self.folder))
        if chosen:
            self.show_folder(chosen)


def choose_output(start: str | Path, parent=None) -> Path | None:
    """The output file chosen (the first of a series), or None."""
    dialog = OpenOutputDialog(start, parent)
    if dialog.exec() == QDialog.Accepted:
        return dialog.chosen
    return None
