# SPDX-License-Identifier: GPL-3.0-or-later
"""Icons and images shipped with the application."""

from __future__ import annotations

from pathlib import Path

_HERE = Path(__file__).resolve().parent


def resource_path(name: str) -> Path:
    return _HERE / name


def app_icon_path(platform: str | None = None) -> Path:
    """The application's icon as it is shown while it runs: on macOS the rounded
    square of the app bundle's icon (packaging/macos/make_icon.py; Qt puts the window
    icon into the Dock as it is, so the square one stood out among the other apps),
    elsewhere the square one."""
    import sys

    platform = sys.platform if platform is None else platform
    return _HERE / ("icon_macos.png" if platform == "darwin" else "icon.png")
