# SPDX-License-Identifier: GPL-3.0-or-later
"""Icons and images shipped with the application."""

from __future__ import annotations

from pathlib import Path

_HERE = Path(__file__).resolve().parent


def resource_path(name: str) -> Path:
    return _HERE / name
