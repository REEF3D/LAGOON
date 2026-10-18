# SPDX-License-Identifier: GPL-3.0-or-later
# Architect: Hans Bihs
"""File → Clean up case folder: what in the folder is not the run shown, by run,
with its size; ticked groups are moved to LAGOON_archive/<date>/ or deleted
(lagoonviz.core.housekeeping). Output left over from earlier runs and earlier
stores come ticked; another solver's run is listed, not ticked."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QDialog,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QMessageBox,
    QPushButton,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from lagoonviz.core.housekeeping import ARCHIVE, Group, archive, human_size, inventory


class HousekeepingDialog(QDialog):
    def __init__(self, case_dir: str | Path, case=None, parent: QWidget | None = None):
        super().__init__(parent)
        self.case_dir = Path(case_dir)
        self.setWindowTitle(f"Clean up {self.case_dir.name}")
        self.resize(760, 420)
        #: set after archiving or deleting: what was done, for the window to reload
        self.done_text = ""
        self.groups: list[Group] = inventory(self.case_dir, case)

        intro = QLabel(
            "Output in this folder that is not the run shown: left over from earlier runs "
            "(REEF3D overwrites only what it writes again), other solvers' runs and earlier "
            f"LAGOON stores. Archiving moves them into <code>{ARCHIVE}/</code> of the folder, "
            "their paths kept."
        )
        intro.setWordWrap(True)
        intro.setTextFormat(Qt.RichText)
        self.tree = QTreeWidget()
        self.tree.setHeaderLabels(["What", "Kind", "Files", "Size"])
        self.tree.setRootIsDecorated(False)
        header = self.tree.header()
        header.setSectionResizeMode(0, QHeaderView.Stretch)
        for column in (1, 2, 3):
            header.setSectionResizeMode(column, QHeaderView.ResizeToContents)
        for group in self.groups:
            row = QTreeWidgetItem(
                [group.label, group.kind, str(len(group.paths)), human_size(group.size)]
            )
            row.setFlags(row.flags() | Qt.ItemIsUserCheckable)
            row.setCheckState(0, Qt.Checked if group.suggested else Qt.Unchecked)
            row.setToolTip(0, "\n".join(str(p) for p in group.paths[:12]))
            row.setTextAlignment(3, Qt.AlignRight)
            self.tree.addTopLevelItem(row)
        self.tree.itemChanged.connect(lambda *_: self._summarize())
        self.summary = QLabel()
        self.archive_button = QPushButton(f"Move to {ARCHIVE}")
        self.archive_button.clicked.connect(self.archive_ticked)
        self.delete_button = QPushButton("Delete…")
        self.delete_button.clicked.connect(self.delete_ticked)
        close = QPushButton("Close")
        close.clicked.connect(self.reject)
        buttons = QHBoxLayout()
        buttons.addWidget(self.summary, 1)
        buttons.addWidget(self.archive_button)
        buttons.addWidget(self.delete_button)
        buttons.addWidget(close)
        layout = QVBoxLayout(self)
        layout.addWidget(intro)
        layout.addWidget(self.tree, 1)
        layout.addLayout(buttons)
        if not self.groups:
            self.tree.setVisible(False)
            intro.setText("Nothing to clean up: everything in this folder is the run shown.")
        self._summarize()

    def ticked(self) -> list[Group]:
        return [
            group
            for index, group in enumerate(self.groups)
            if self.tree.topLevelItem(index).checkState(0) == Qt.Checked
        ]

    def _summarize(self) -> None:
        ticked = self.ticked()
        size = sum(g.size for g in ticked)
        self.summary.setText(
            f"{len(ticked)} of {len(self.groups)} ticked, {human_size(size)}" if self.groups else ""
        )
        for button in (self.archive_button, self.delete_button):
            button.setEnabled(bool(ticked))

    def archive_ticked(self) -> Path | None:
        ticked = self.ticked()
        if not ticked:
            return None
        target = archive(ticked, self.case_dir)
        self.done_text = f"Moved {len(ticked)} group(s) to {target.relative_to(self.case_dir)}"
        self.accept()
        return target

    def delete_ticked(self, confirm: bool = True) -> int:
        from lagoonviz.core.housekeeping import remove

        ticked = self.ticked()
        if not ticked:
            return 0
        size = human_size(sum(g.size for g in ticked))
        if confirm:
            answer = QMessageBox.question(
                self,
                "Delete for good?",
                f"Delete {len(ticked)} group(s), {size}, from {self.case_dir}? This cannot be "
                f"undone (Move to {ARCHIVE} keeps them).",
            )
            if answer != QMessageBox.Yes:
                return 0
        freed = remove(ticked, self.case_dir)
        self.done_text = f"Deleted {len(ticked)} group(s), {human_size(freed)}"
        self.accept()
        return freed
