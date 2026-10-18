# SPDX-License-Identifier: GPL-3.0-or-later
"""Dockable panels of the main window."""

from lagoonviz.gui.panels.console import ConsolePanel
from lagoonviz.gui.panels.plots import PlotsPanel
from lagoonviz.gui.panels.profiles import ProfilesPanel
from lagoonviz.gui.panels.properties import PropertiesPanel
from lagoonviz.gui.panels.run_monitor import RunMonitorPanel
from lagoonviz.gui.panels.scene_tree import SceneTreePanel
from lagoonviz.gui.panels.setup_editor import SetupPanel
from lagoonviz.gui.panels.timeline import TimelinePanel
from lagoonviz.gui.panels.viewport import ViewportPlaceholder, create_viewport

__all__ = [
    "ConsolePanel",
    "PlotsPanel",
    "ProfilesPanel",
    "PropertiesPanel",
    "RunMonitorPanel",
    "SceneTreePanel",
    "SetupPanel",
    "TimelinePanel",
    "ViewportPlaceholder",
    "create_viewport",
]
