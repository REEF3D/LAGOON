# SPDX-License-Identifier: GPL-3.0-or-later
# Architect: Hans Bihs
"""Set-up panel: a case's input files edited command by command, and DIVEMesh run.

Two tabs, ``control.txt`` (DIVEMesh) and ``ctrl.txt`` (REEF3D), list the commands
of each file by section (B · Boundary, P · Printer, ...). Selecting one shows what
the user guide says about it and its values, ready to change; its signs light up
in the 3D view. *Add* finds any command of the program by number or by words
("wave gauge"), *Remove* takes a line out, and *Text* edits the file as text.
Every change is a command of the session (undo, script); the files are written
with *Save*. *Run DIVEMesh* saves control.txt and runs DIVEMesh in the case folder,
its output in the third tab; its grid, solids and bed then show in the set-up.
"""

from __future__ import annotations

import logging
from pathlib import Path

from PySide6.QtCore import QProcess, QSettings, QStringListModel, Qt, Signal
from PySide6.QtGui import QBrush, QColor, QFontDatabase, QTextCursor
from PySide6.QtWidgets import (
    QComboBox,
    QCompleter,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QScrollArea,
    QSplitter,
    QStackedWidget,
    QTabWidget,
    QToolButton,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from lagoonviz.core.data.controlfile import CommandInfo, catalog_for, format_value
from lagoonviz.core.divemesh_run import LOG, divemesh_command, find_divemesh
from lagoonviz.gui.bridge import SessionBridge
from lagoonviz.gui.theme import COLORS

log = logging.getLogger(__name__)

FILES = ("control.txt", "ctrl.txt")
TITLES = {"control.txt": "control.txt · DIVEMesh", "ctrl.txt": "ctrl.txt · REEF3D"}
KEY_ROLE = Qt.UserRole


class InputEditor(QWidget):
    """One input file of the current case: its commands, the selected one's values."""

    #: (code, occurrence) of the selected line, or ("", 0)
    selected = Signal(str, int)

    def __init__(self, panel: SetupPanel, file: str):
        super().__init__(panel)
        self.panel = panel
        self.session = panel.session
        self.file = file
        self.catalog = catalog_for(file)
        self._key: tuple[str, int] = ("", 0)
        #: the line whose values the form shows, and its fields
        self._shown: tuple[str, int] | None = None
        self._fields: list[QWidget] = []

        # ---- the row of buttons: add a command, remove a line, text
        self.find = QLineEdit()
        self.find.setPlaceholderText("Add a command: its number or words (wave gauge)")
        self.find.setClearButtonEnabled(True)
        entries = [f"{c.code} · {c.title}" for c in self.catalog.commands.values()]
        self.completer = QCompleter(QStringListModel(entries, self), self)
        self.completer.setCaseSensitivity(Qt.CaseInsensitive)
        self.completer.setFilterMode(Qt.MatchContains)
        self.completer.setMaxVisibleItems(16)
        self.find.setCompleter(self.completer)
        self.completer.activated.connect(self._add_from)
        self.find.returnPressed.connect(lambda: self._add_from(self.find.text()))
        self.add_button = QPushButton("Add")
        self.add_button.setToolTip("Add the command typed beside (a new line of it)")
        self.add_button.clicked.connect(lambda: self._add_from(self.find.text()))
        self.remove_button = QPushButton("Remove")
        self.remove_button.setToolTip("Remove the selected line from the file")
        self.remove_button.clicked.connect(self.remove_selected)
        self.text_button = QToolButton()
        self.text_button.setText("Text")
        self.text_button.setCheckable(True)
        self.text_button.setToolTip("Edit the whole file as text")
        self.text_button.toggled.connect(self._show_text)
        row = QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        row.addWidget(self.find, 1)
        row.addWidget(self.add_button)
        row.addWidget(self.remove_button)
        row.addWidget(self.text_button)

        # ---- the commands, by section, and the selected one's values
        self.tree = QTreeWidget()
        self.tree.setObjectName(f"commands_{file.split('.')[0]}")
        self.tree.setHeaderLabels(["Command", "Values"])
        self.tree.setRootIsDecorated(True)
        self.tree.setUniformRowHeights(True)
        self.tree.itemSelectionChanged.connect(self._on_select)
        self.details = QScrollArea()
        self.details.setWidgetResizable(True)
        split = QSplitter(Qt.Vertical)
        split.addWidget(self.tree)
        split.addWidget(self.details)
        split.setStretchFactor(0, 3)
        split.setStretchFactor(1, 2)

        # ---- the file as text
        self.text = QPlainTextEdit()
        self.text.setFont(QFontDatabase.systemFont(QFontDatabase.FixedFont))
        self.text.setLineWrapMode(QPlainTextEdit.NoWrap)
        self.apply_text = QPushButton("Apply text")
        self.apply_text.clicked.connect(self._apply_text)
        text_page = QWidget()
        text_layout = QVBoxLayout(text_page)
        text_layout.setContentsMargins(0, 0, 0, 0)
        text_layout.addWidget(self.text)
        text_layout.addWidget(self.apply_text, 0, Qt.AlignRight)

        self.pages = QStackedWidget()
        self.pages.addWidget(split)
        self.pages.addWidget(text_page)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.addLayout(row)
        layout.addWidget(self.pages, 1)

    # ------------------------------------------------------------- the file
    def control(self):
        """The file being edited (None without a case)."""
        case = self.panel.case_id
        if not case or case not in self.session.scene:
            return None
        try:
            return self.session.input_file(self.file, case)
        except (OSError, ValueError):
            return None

    def refresh(self) -> None:
        """The list again from the file, the selection kept."""
        control = self.control()
        key = self._key
        self.tree.blockSignals(True)
        self.tree.clear()
        if control is None:
            self.tree.blockSignals(False)
            self._show_details(None, None)
            return
        problems = {}
        for problem in self.catalog.check(control, self.panel.solver()):
            problems.setdefault(problem.line, []).append(problem)
        sections: dict[str, QTreeWidgetItem] = {}
        counts: dict[str, int] = {}
        chosen = None
        for number, line in enumerate(control.lines, start=1):
            if line.code is None:
                continue
            letter = line.code.split()[0]
            section = sections.get(letter)
            if section is None:
                title = self.catalog.sections.get(letter, "")
                section = QTreeWidgetItem([f"{letter} · {title}" if title else letter, ""])
                section.setFlags(Qt.ItemIsEnabled)
                self.tree.addTopLevelItem(section)
                sections[letter] = section
            occurrence = counts.get(line.code, 0)
            counts[line.code] = occurrence + 1
            info = self.catalog.get(line.code)
            label = line.code + (f" · {info.title}" if info and info.title else "")
            if info is not None and info.repeat:
                label += f"  #{occurrence + 1}"
            entry = QTreeWidgetItem([label, " ".join(line.values)])
            entry.setData(0, KEY_ROLE, (line.code, occurrence))
            tip = info.text if info is not None else "Not a command of " + self.catalog.program
            found = problems.get(number, [])
            if found:
                colour = "#e5484d" if any(p.level == "error" for p in found) else "#d9822b"
                for column in (0, 1):
                    entry.setForeground(column, QBrush(QColor(colour)))
                tip = "\n".join(p.message for p in found) + ("\n\n" + tip if tip else "")
            entry.setToolTip(0, tip)
            entry.setToolTip(1, line.text)
            section.addChild(entry)
            if (line.code, occurrence) == key:
                chosen = entry
        self.tree.expandAll()
        self.tree.resizeColumnToContents(0)
        if chosen is not None:
            self.tree.setCurrentItem(chosen)
        self.tree.blockSignals(False)
        if chosen is not None:
            if self._shown == key and self._fields:
                self._update_fields(*key)  # in place: the field being typed in keeps focus
            else:
                self._show_details(*key)
        else:
            self._key = ("", 0)
            self._show_details(None, None)
        if self.text_button.isChecked() and not self.text.hasFocus():
            self.text.setPlainText(control.text())

    # ------------------------------------------------------------- selection
    def _on_select(self) -> None:
        items = self.tree.selectedItems()
        key = items[0].data(0, KEY_ROLE) if items else None
        if not key:
            self._key = ("", 0)
            self._show_details(None, None)
        else:
            self._key = (str(key[0]), int(key[1]))
            self._show_details(*self._key)
        self.selected.emit(*self._key)

    def select(self, code: str, occurrence: int = 0) -> None:
        self._key = (code, occurrence)
        self.refresh()
        self.selected.emit(*self._key)

    def _show_details(self, code: str | None, occurrence: int | None) -> None:
        """The selected line: what it is, its values to change, what the guide says."""
        body = QWidget()
        form = QFormLayout(body)
        self._fields = []
        control = self.control()
        self._shown = (code, occurrence) if code is not None else None
        if code is None or control is None:
            hint = QLabel("Select a command to see and change its values.")
            hint.setEnabled(False)
            hint.setWordWrap(True)
            form.addRow(hint)
            self.details.setWidget(body)
            self.remove_button.setEnabled(False)
            return
        self.remove_button.setEnabled(True)
        info = self.catalog.get(code)
        lines = control.entries(code)
        values = lines[occurrence].values if occurrence < len(lines) else []
        title = QLabel(f"<b>{info.label() if info else code}</b>")
        title.setWordWrap(True)
        form.addRow(title)
        if info is None:
            note = QLabel(f"Not a command of {self.catalog.program}: it is ignored.")
            note.setWordWrap(True)
            form.addRow(note)
        names = info.names() if info else [f"value {i + 1}" for i in range(len(values))]
        count = len(names) if info and info.count else max(len(values), len(names))
        kinds = info.kinds(count) if info else ["double"] * count
        for index in range(count):
            name = names[index] if index < len(names) else f"value {index + 1}"
            value = values[index] if index < len(values) else ""
            default = info.default[index] if info and index < len(info.default) else ""
            field = self._field(info, value, default, kinds[index], count)
            self._fields.append(field)
            form.addRow(name or "value", field)
        if info is not None and info.repeat:
            again = QPushButton("Add another")
            again.setToolTip(f"One more line of {code} (another one of these)")
            again.clicked.connect(lambda: self._add_line(code, values))
            form.addRow("", again)
        if info is not None:
            facts = []
            if info.default_text or info.default:
                facts.append("Default: " + (info.default_text or " ".join(info.default)))
            if info.modules:
                names = {"S": "SFLOW", "F": "FNPF", "N": "NHFLOW", "C": "CFD"}
                facts.append("Used by: " + ", ".join(names[m] for m in info.modules))
            if info.repeat:
                facts.append("May be given several times, one line each")
            if facts:
                fact = QLabel("\n".join(facts))
                fact.setWordWrap(True)
                fact.setEnabled(False)
                form.addRow(fact)
            if info.text:
                text = QLabel(info.text)
                text.setWordWrap(True)
                text.setTextInteractionFlags(Qt.TextSelectableByMouse)
                form.addRow(text)
        self.details.setWidget(body)

    def _update_fields(self, code: str, occurrence: int) -> None:
        """The form's fields from the file again (after an edit, an undo)."""
        control = self.control()
        lines = control.entries(code) if control is not None else []
        values = lines[occurrence].values if occurrence < len(lines) else []
        for index, field in enumerate(self._fields):
            value = values[index] if index < len(values) else ""
            if field.hasFocus():
                continue
            field.blockSignals(True)
            if isinstance(field, QComboBox):
                found = field.findData(value)
                if found >= 0:
                    field.setCurrentIndex(found)
                else:
                    field.setEditText(value)
            else:
                field.setText(value)
            field.blockSignals(False)

    def _field(self, info: CommandInfo | None, value: str, default: str, kind: str, count: int):
        if info is not None and info.options and count == 1:
            box = QComboBox()
            box.setEditable(True)  # a value the guide does not list can still be typed
            for key, label in info.options.items():
                box.addItem(f"{key} · {label}", key)
            index = box.findData(value)
            if index >= 0:
                box.setCurrentIndex(index)
            else:
                box.setEditText(value)
            box.activated.connect(lambda _i: self._commit())
            box.lineEdit().editingFinished.connect(self._commit)
            return box
        edit = QLineEdit(value)
        if default:
            edit.setPlaceholderText(default)
        edit.setToolTip("An integer" if kind == "int" else "A number")
        edit.editingFinished.connect(self._commit)
        return edit

    def _field_values(self) -> list[str]:
        """The values in the form: a choice's value ("2" of "2 · SFLOW"), or as typed."""
        return [
            (field.currentText() if isinstance(field, QComboBox) else field.text())
            .split("·")[0]
            .strip()
            for field in self._fields
        ]

    # ------------------------------------------------------------- edits
    def _commit(self) -> None:
        """The values of the form into the file (a command of the session)."""
        code, occurrence = self._key
        control = self.control()
        if not code or control is None:
            return
        values = self._field_values()
        while values and not values[-1]:
            values.pop()  # trailing empty fields: not given
        if any(not v for v in values):
            self.panel.status(f"{code}: every value up to the last must be given", error=True)
            return
        try:
            numbers = [float(v) for v in values]
        except ValueError:
            self.panel.status(f"{code}: the values must be numbers", error=True)
            return
        lines = control.entries(code)
        if occurrence < len(lines) and lines[occurrence].numbers() == numbers:
            return
        info = self.catalog.get(code)
        kinds = info.kinds(len(numbers)) if info else ["double"] * len(numbers)
        texts = [format_value(n, k) for n, k in zip(numbers, kinds, strict=False)]
        self.session.set_input(self.file, code, texts, occurrence, case=self.panel.case_id)

    def _add_from(self, text: str) -> None:
        text = str(text).split("·")[0].strip()
        if not text:
            return
        found = self.catalog.search(text, self.panel.solver())
        info = found[0] if found else None
        if info is None:
            self.panel.status(f"No {self.catalog.program} command matches {text!r}", error=True)
            return
        control = self.control()
        if control is None:
            return
        if info.code in control and not info.repeat:
            self.select(info.code, 0)  # given once only: show the one there is
            self.panel.status(f"{info.code} is in the file already: change its values")
            self.find.clear()
            return
        values = list(info.default) if info.default else ["0"] * (info.count or 1)
        self._add_line(info.code, values)
        self.find.clear()

    def _add_line(self, code: str, values) -> None:
        occurrence = self.session.add_input(self.file, code, list(values), case=self.panel.case_id)
        self.select(code, occurrence)

    def remove_selected(self) -> None:
        code, occurrence = self._key
        if code:
            self.session.remove_input(self.file, code, occurrence, case=self.panel.case_id)
            self._key = ("", 0)
            self.refresh()
            self.selected.emit("", 0)

    def _show_text(self, on: bool) -> None:
        self.pages.setCurrentIndex(1 if on else 0)
        control = self.control()
        if on and control is not None:
            self.text.setPlainText(control.text())

    def _apply_text(self) -> None:
        if self.control() is not None:
            self.session.set_input_text(self.file, self.text.toPlainText(), case=self.panel.case_id)


class SetupPanel(QWidget):
    """The set-up of the current case: its input files and DIVEMesh (module doc)."""

    def __init__(
        self, bridge: SessionBridge, settings: QSettings | None = None, parent=None
    ) -> None:
        super().__init__(parent)
        self.setObjectName("setupPanel")
        self.bridge = bridge
        self.session = bridge.session
        self.settings = settings if settings is not None else QSettings()
        self.case_id = ""
        self.process: QProcess | None = None
        self._log_file = None

        self.case_box = QComboBox()
        self.case_box.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLengthWithIcon)
        self.case_box.setMinimumContentsLength(12)
        self.case_box.currentIndexChanged.connect(self._case_chosen)
        self.save_button = QPushButton("Save")
        self.save_button.setToolTip("Write the edited input files of the case")
        self.save_button.clicked.connect(self.save)
        self.run_button = QPushButton("Run DIVEMesh")
        self.run_button.setToolTip(
            "Save control.txt and run DIVEMesh in the case folder: the grid, solids and bed"
        )
        self.run_button.clicked.connect(self.run_divemesh)
        top = QHBoxLayout()
        top.setContentsMargins(4, 4, 4, 0)
        top.addWidget(QLabel("Case"))
        top.addWidget(self.case_box, 1)
        top.addWidget(self.save_button)
        top.addWidget(self.run_button)

        self.tabs = QTabWidget()
        self.editors = {name: InputEditor(self, name) for name in FILES}
        for name, editor in self.editors.items():
            self.tabs.addTab(editor, TITLES[name])
            editor.selected.connect(lambda code, n, f=name: self._highlight(f, code, n))
        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setFont(QFontDatabase.systemFont(QFontDatabase.FixedFont))
        self.log.setPlaceholderText("DIVEMesh's output appears here when it runs.")
        self.tabs.addTab(self.log, "DIVEMesh")
        self.tabs.currentChanged.connect(self._tab_changed)

        self.message = QLabel()
        self.message.setWordWrap(True)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addLayout(top)
        layout.addWidget(self.tabs, 1)
        layout.addWidget(self.message)

        bridge.sceneChanged.connect(self._on_scene)
        self._fill_cases()

    # ------------------------------------------------------------- cases
    def solver(self) -> str:
        setup = self.session.data.case_setup(self.case_id) if self.case_id else None
        return setup.solver if setup is not None else ""

    def _fill_cases(self) -> None:
        cases = [(i.name, i.id) for i in self.session.scene if i.kind == "case"]
        box = self.case_box
        box.blockSignals(True)
        box.clear()
        for name, item_id in cases:
            box.addItem(name, item_id)
        index = box.findData(self.case_id)
        if index < 0 and cases:
            index = 0
        box.setCurrentIndex(index)
        box.blockSignals(False)
        self.case_id = box.currentData() or ""
        self.refresh()

    def _case_chosen(self, _index: int) -> None:
        self.case_id = self.case_box.currentData() or ""
        self.refresh()

    def refresh(self) -> None:
        has_case = bool(self.case_id)
        for editor in self.editors.values():
            editor.setEnabled(has_case)
            editor.refresh()
        modified = self.session.inputs_modified(self.case_id) if has_case else []
        for index, name in enumerate(FILES):
            self.tabs.setTabText(index, TITLES[name] + (" *" if name in modified else ""))
        self.save_button.setEnabled(bool(modified))
        running = self.process is not None
        self.run_button.setEnabled(has_case and not running)
        self.run_button.setText("DIVEMesh running…" if running else "Run DIVEMesh")
        if not has_case:
            self.status("Open a case folder (File → Open case…) to edit its input files.")
        elif not self.message.text():
            self._describe()

    def _describe(self) -> None:
        setup = self.session.data.case_setup(self.case_id)
        if setup is None:
            self.status("No input files yet: add commands and save to write them.")
            return
        facts = setup.describe().split("\n")[1:4]
        self.status(" · ".join(f.strip() for f in facts))

    def status(self, text: str, error: bool = False) -> None:
        colour = "#e5484d" if error else COLORS["dim"]
        self.message.setStyleSheet(f"color: {colour};")
        self.message.setText(text)

    def _on_scene(self, event: str, item_id: str) -> None:
        if event in ("added", "removed", "reset") or (
            event == "changed"
            and item_id in self.session.scene
            and self.session.scene.get(item_id).kind == "case"
        ):
            self._fill_cases()
        elif event == "inputs" and item_id == self.case_id:
            self.message.clear()
            self.refresh()

    def _highlight(self, file: str, code: str, occurrence: int) -> None:
        if self.case_id:
            self.session.data.set_highlight(self.case_id, file, code, occurrence)

    def _tab_changed(self, index: int) -> None:
        if index >= len(FILES) and self.case_id:
            self.session.data.set_highlight(self.case_id, "", "")

    # ------------------------------------------------------------- saving
    def save(self) -> list[Path]:
        if not self.case_id:
            return []
        written = self.session.save_inputs(self.case_id)
        if written:
            self.status("Saved " + ", ".join(p.name for p in written))
        self.refresh()
        return written

    def ask_to_save(self, parent=None) -> bool:
        """Before closing: save the edited input files of every case, or not. False:
        cancelled."""
        cases = [i.id for i in self.session.scene if i.kind == "case"]
        modified = [(c, self.session.inputs_modified(c)) for c in cases]
        modified = [(c, files) for c, files in modified if files]
        if not modified:
            return True
        names = ", ".join(f"{self.session.scene.get(c).name}: {' '.join(f)}" for c, f in modified)
        answer = QMessageBox.question(
            parent or self,
            "Edited input files",
            f"Input files have edits that are not saved ({names}). Save them?",
            QMessageBox.Save | QMessageBox.Discard | QMessageBox.Cancel,
            QMessageBox.Save,
        )
        if answer == QMessageBox.Cancel:
            return False
        if answer == QMessageBox.Save:
            for case_id, _ in modified:
                self.session.save_inputs(case_id)
        return True

    # ------------------------------------------------------------- DIVEMesh
    def divemesh_program(self, ask: bool = True) -> Path | None:
        """The DIVEMesh program: the one chosen before, else found; else asked for."""
        saved = str(self.settings.value("divemesh/program", "") or "")
        program = find_divemesh(saved or None)
        if program is None and ask:
            chosen, _ = QFileDialog.getOpenFileName(
                self, "Where is DIVEMesh? (the DiveMESH program)", str(Path.home())
            )
            if chosen:
                program = find_divemesh(chosen)
        if program is not None:
            self.settings.setValue("divemesh/program", str(program))
        return program

    def run_divemesh(self) -> bool:
        """Save control.txt and run DIVEMesh on the case (in the background)."""
        if not self.case_id or self.process is not None:
            return False
        folder = self.session.data.case_folder(self.case_id)
        control = self.session.input_file("control.txt", self.case_id)
        if folder is None or not control:
            self.status("The case has no control.txt: add DIVEMesh's commands first", True)
            return False
        if control.modified:
            control.write()  # DIVEMesh reads the file
            self.session.scene.inputs_changed(self.case_id)
        program = self.divemesh_program()
        if program is None:
            self.status("DIVEMesh was not found: choose the DiveMESH program to run it", True)
            return False
        self.log.clear()
        self.tabs.setCurrentWidget(self.log)
        try:
            (folder / LOG).parent.mkdir(parents=True, exist_ok=True)
            self._log_file = (folder / LOG).open("w", encoding="utf-8")
        except OSError:
            self._log_file = None
        process = QProcess(self)
        process.setWorkingDirectory(str(folder))
        process.setProcessChannelMode(QProcess.MergedChannels)
        command = divemesh_command(program)
        process.setProgram(command[0])
        process.setArguments(command[1:])
        process.readyReadStandardOutput.connect(self._read_output)
        process.finished.connect(self._finished)
        process.errorOccurred.connect(self._failed)
        self.process = process
        self._running_case = self.case_id
        self.log.appendPlainText(f"$ {program}   (in {folder})")
        self.status("DIVEMesh is running…")
        process.start()
        self.refresh()
        return True

    def _read_output(self) -> None:
        if self.process is None:
            return
        text = bytes(self.process.readAllStandardOutput()).decode("utf-8", "replace")
        if self._log_file is not None:
            self._log_file.write(text)
        self.log.moveCursor(QTextCursor.End)
        self.log.insertPlainText(text)
        self.log.ensureCursorVisible()

    def _finished(self, code: int, _status=None) -> None:
        self._read_output()
        if self._log_file is not None:
            self._log_file.close()
            self._log_file = None
        self.process = None
        case_id = self._running_case
        if code == 0 and case_id in self.session.scene:
            self.session.divemesh_done(case_id)
            self.status("DIVEMesh finished: its grid, solids and bed are shown")
        else:
            self.status(f"DIVEMesh stopped with code {code}: see its output", True)
        self.refresh()

    def _failed(self, error) -> None:
        if self.process is None or error != QProcess.FailedToStart:
            return
        self.log.appendPlainText(f"DIVEMesh could not be started: {self.process.errorString()}")
        if self._log_file is not None:
            self._log_file.close()
            self._log_file = None
        self.process = None
        self.status("DIVEMesh could not be started (see its output)", True)
        self.refresh()
