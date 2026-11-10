# SPDX-License-Identifier: GPL-3.0-or-later
"""``python -m lagoonviz`` starts the LAGOON application."""

if __name__ == "__main__":  # not when a process LAGOON starts imports it again
    from lagoonviz.gui.app import main

    raise SystemExit(main())
