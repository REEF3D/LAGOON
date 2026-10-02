# SPDX-License-Identifier: GPL-3.0-or-later
"""LAGOON core: data model, scene, commands and sessions.

The core never imports Qt or GPU libraries, so it runs in scripts, notebooks and
headless on clusters. This rule is checked in CI (``lint-imports``).
"""

from lagoonviz.core.kinds import KINDS
from lagoonviz.core.scene import Scene, SceneItem
from lagoonviz.core.session import SESSION_EXTENSION, Session

__all__ = ["KINDS", "SESSION_EXTENSION", "Scene", "SceneItem", "Session"]
