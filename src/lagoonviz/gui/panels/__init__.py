# SPDX-License-Identifier: GPL-3.0-or-later
"""Dockable panels of the main window."""

from lagoonviz.gui.panels.console import ConsolePanel
from lagoonviz.gui.panels.plots import PlotsPanel
from lagoonviz.gui.panels.properties import PropertiesPanel
from lagoonviz.gui.panels.scene_tree import SceneTreePanel
from lagoonviz.gui.panels.timeline import TimelinePanel
from lagoonviz.gui.panels.viewport import ViewportPlaceholder

__all__ = [
    "ConsolePanel",
    "PlotsPanel",
    "PropertiesPanel",
    "SceneTreePanel",
    "TimelinePanel",
    "ViewportPlaceholder",
]
