# SPDX-License-Identifier: GPL-3.0-or-later
"""Application entry point: ``lagoon [session file | case folder | output file]``."""

from __future__ import annotations

import argparse
import logging
import os
import signal
import sys
from pathlib import Path

from PySide6.QtCore import QSettings, QTimer
from PySide6.QtGui import QIcon
from PySide6.QtWidgets import QApplication

import lagoonviz
from lagoonviz.core import Session
from lagoonviz.gui.macos import set_app_name
from lagoonviz.gui.main_window import MainWindow
from lagoonviz.gui.resources import resource_path
from lagoonviz.gui.theme import DEFAULT_THEME, apply_theme


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    parser = argparse.ArgumentParser(prog="lagoon", description="LAGOON desktop application")
    parser.add_argument(
        "path", nargs="?", help="session file, REEF3D case folder or output file to open"
    )
    parser.add_argument(
        "--dev-remote",
        metavar="DIR",
        help="developer testing: take screenshots and simple actions from DIR/request.json",
    )
    parser.add_argument("--version", action="version", version=f"LAGOON {lagoonviz.__version__}")
    args = parser.parse_args(argv)
    setup_logging()

    set_app_name("LAGOON")  # before Qt starts: the macOS menu bar reads it once
    app = QApplication.instance() or QApplication(["LAGOON"])
    app.setApplicationName("LAGOON")
    app.setOrganizationName("REEF3D")
    app.setApplicationVersion(lagoonviz.__version__)
    app.setWindowIcon(QIcon(str(resource_path("icon.png"))))
    apply_theme(app, str(QSettings().value("view/theme", DEFAULT_THEME)))

    window = MainWindow(Session())
    window.show()
    if args.dev_remote:
        from lagoonviz.gui.devremote import DevRemote

        window.dev_remote = DevRemote(window, args.dev_remote)
    if args.path:
        path = Path(args.path).expanduser()
        if path.suffix == ".lagoon-session":
            window.open_session(path)
        else:
            window.open_case(path)
    _quit_on_ctrl_c(app, window)
    return app.exec()


def _quit_on_ctrl_c(app: QApplication, window: MainWindow) -> None:
    """Ctrl+C in the terminal closes LAGOON properly.

    The render canvas library installs its own handler, which closes only the 3D
    view and leaves the window (and the terminal) waiting; this one replaces it. A
    timer wakes Python regularly, so the signal is seen while Qt is idle.
    """

    def stop(*_args) -> None:
        window.ask_before_discard = False
        window.close()
        app.quit()

    signal.signal(signal.SIGINT, stop)
    app._signal_timer = QTimer()  # kept on the app so it lives as long
    app._signal_timer.timeout.connect(lambda: None)
    app._signal_timer.start(250)


def setup_logging() -> None:
    """Log to stderr, and to the file in ``LAGOON_LOG`` if it is set."""
    handlers: list[logging.Handler] = [logging.StreamHandler()]
    target = os.environ.get("LAGOON_LOG")
    if target:
        handlers.append(logging.FileHandler(target, mode="w", encoding="utf-8"))
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s: %(message)s",
        handlers=handlers,
        force=True,
    )
    logging.getLogger("lagoon").info("LAGOON %s, Python %s", lagoonviz.__version__, sys.version)


if __name__ == "__main__":
    raise SystemExit(main())
