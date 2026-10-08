# SPDX-License-Identifier: GPL-3.0-or-later
# Architect: Hans Bihs
"""The entry point of the standalone LAGOON.app (packaging/macos/LAGOON.spec).

Double-clicked, it starts the window. From a terminal the same program also renders
and converts, so the app needs no Python of its own on the machine:

    /Applications/LAGOON.app/Contents/MacOS/LAGOON                 # the window
    /Applications/LAGOON.app/Contents/MacOS/LAGOON ~/runs/flume    # opens a case
    /Applications/LAGOON.app/Contents/MacOS/LAGOON render SOURCE OUTPUT [options]
    /Applications/LAGOON.app/Contents/MacOS/LAGOON convert CASE [options]
    /Applications/LAGOON.app/Contents/MacOS/LAGOON setup CASE [OUTPUT] [options]

Started from the Finder there is no terminal for its messages, so they go to
~/Library/Logs/LAGOON/lagoon.log (unless LAGOON_LOG says otherwise).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

COMMANDS = ("render", "convert", "setup")


def _log_file() -> None:
    if os.environ.get("LAGOON_LOG"):
        return
    folder = Path.home() / "Library" / "Logs" / "LAGOON"
    try:
        folder.mkdir(parents=True, exist_ok=True)
        os.environ["LAGOON_LOG"] = str(folder / "lagoon.log")
    except OSError:
        pass


def main() -> int:
    # old macOS versions add a process serial number when the Finder starts an app
    args = [a for a in sys.argv[1:] if not a.startswith("-psn_")]
    if args and args[0] in COMMANDS:
        command, rest = args[0], args[1:]
        sys.argv = [f"lagoon-{command}", *rest]
        if command == "render":
            from lagoonviz.render.offscreen import main as run
        elif command == "setup":
            from lagoonviz.render.setup_sketch import main as run
        else:
            from lagoonviz.core.convert import main as run
        return int(run(rest) or 0)
    _log_file()
    from lagoonviz.gui.app import main as window

    return int(window(args) or 0)


if __name__ == "__main__":
    sys.exit(main())
