# SPDX-License-Identifier: GPL-3.0-or-later
"""Application entry point: ``lagoon [session-file]``."""

from __future__ import annotations

import argparse
import sys

from PySide6.QtGui import QIcon
from PySide6.QtWidgets import QApplication

import lagoonviz
from lagoonviz.core import Session
from lagoonviz.gui.main_window import MainWindow
from lagoonviz.gui.resources import resource_path


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    parser = argparse.ArgumentParser(prog="lagoon", description="LAGOON desktop application")
    parser.add_argument("session", nargs="?", help="session file to open")
    parser.add_argument("--version", action="version", version=f"LAGOON {lagoonviz.__version__}")
    args = parser.parse_args(argv)

    app = QApplication.instance() or QApplication([sys.argv[0]])
    app.setApplicationName("LAGOON")
    app.setOrganizationName("REEF3D")
    app.setApplicationVersion(lagoonviz.__version__)
    app.setWindowIcon(QIcon(str(resource_path("icon.png"))))

    window = MainWindow(Session())
    if args.session:
        window.open_session(args.session)
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
