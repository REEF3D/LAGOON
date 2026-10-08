"""Agreed structure of the window (part 2 of the regression suite).

Each test writes what the user sees as plain text and compares it with the agreed
snapshot in ``tests/snapshots/gui``: menus and shortcuts, toolbar, panels, and the
properties panel of every item kind. A change shows as a diff; accept an intended
one with ``pytest --update-snapshots`` (see ``tests/snapshot_tools.py``).
"""

import sys

import pytest
from describe import (
    describe_docks,
    describe_layout,
    describe_menus,
    describe_properties,
    describe_toolbar,
)
from PySide6.QtCore import QSettings

from lagoonviz.core import Session
from lagoonviz.core.kinds import KINDS
from lagoonviz.gui.main_window import MainWindow


@pytest.fixture
def window(qtbot, tmp_path):
    settings = QSettings(str(tmp_path / "settings.ini"), QSettings.IniFormat)
    win = MainWindow(Session(), settings=settings)
    win.ask_before_discard = False
    qtbot.addWidget(win)
    win.resize(1400, 900)
    win.show()
    return win


def test_menus(window, snapshots):
    # shortcuts follow each platform's conventions (Redo, Refresh, ...): one per platform
    snapshots.check_text(f"gui/menus-{sys.platform}", describe_menus(window))


def test_toolbar_and_panels(window, snapshots):
    text = "TOOLBAR\n" + describe_toolbar(window)
    text += "\n\nPANELS\n" + describe_docks(window)
    text += "\n\nPLAYER\n" + "\n".join(describe_layout(window.timeline.layout()))
    snapshots.check_text("gui/toolbar-and-panels", text)


def test_full_screen_layout(window, snapshots):
    window.toggle_full_screen(True)
    text = "PANELS\n" + describe_docks(window)
    text += "\n\nPLAYER\n" + "\n".join(describe_layout(window.timeline.layout()))
    window.toggle_full_screen(False)
    text += "\n\nBACK\n" + describe_docks(window)
    snapshots.check_text("gui/full-screen", text)


@pytest.mark.parametrize("kind", list(KINDS))
def test_properties_of_every_kind(window, reef3d_run, snapshots, kind):
    window.open_case(reef3d_run)
    session = window.session
    # an item the case shows (the volume it opened with is switched off: a new one)
    item_id = next((i.id for i in session.scene if i.kind == kind and i.visible), None)
    if item_id is None:
        item_id = session.add_item(kind)
    window.scene_tree.select(item_id)
    assert window.properties.item_id == item_id
    text = describe_properties(window.properties, {str(reef3d_run): "<case>"})
    snapshots.check_text(f"gui/properties-{kind}", text)
