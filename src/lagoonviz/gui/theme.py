# SPDX-License-Identifier: GPL-3.0-or-later
"""LAGOON's look: a calm light theme in REEF3D blues (dark on request), the same
on every platform.

Qt's Fusion style plus a palette and a small style sheet. Colours used by custom
painted widgets (plots, colour bars, viewport header) come from :data:`COLORS`, so
everything changes together.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

from PySide6.QtCore import QPointF, Qt
from PySide6.QtGui import QColor, QFont, QImage, QPainter, QPalette, QPen
from PySide6.QtWidgets import QApplication

from lagoonviz.core.colormaps import SERIES_PALETTES

THEMES: dict[str, dict[str, str]] = {
    # default: bright panels, REEF3D blue accents
    "light": {
        "window": "#e9edf2",
        "panel": "#f6f8fa",
        "base": "#ffffff",
        "alternate": "#f1f4f7",
        "raised": "#e2e8ef",
        "border": "#cfd7e0",
        "text": "#1e2833",
        "dim": "#6a7684",
        "accent": "#2b6fb6",
        "accent_dark": "#2b4c7e",  # REEF3D blue
        "selection": "#cfe2f6",
        "selection_text": "#12233a",
        "pressed": "#bcd6f2",
        "cursor": "#e07b39",  # REEF3D orange: the current time everywhere
        "viewport_top": "#ffffff",
        "viewport_bottom": "#e6eaef",
    },
    "dark": {
        "window": "#1d2228",
        "panel": "#242a31",
        "base": "#181c21",
        "alternate": "#20262c",
        "raised": "#2d343c",
        "border": "#363e48",
        "text": "#dfe5ec",
        "dim": "#8a95a2",
        "accent": "#4f8fcf",  # REEF3D light blue
        "accent_dark": "#2b4c7e",  # REEF3D blue
        "selection": "#2b4c7e",
        "selection_text": "#ffffff",
        "pressed": "#2b4c7e",
        "cursor": "#e07b39",
        "viewport_top": "#3b5f8f",
        "viewport_bottom": "#1c3354",
    },
}

DEFAULT_THEME = "light"
#: the 3D view's look that goes with each theme
THEME_LOOKS = {"light": "studio", "dark": "ocean"}

#: the current theme's colours; widgets look them up when they paint
COLORS: dict[str, str] = dict(THEMES[DEFAULT_THEME])
SERIES_COLORS: list[str] = list(SERIES_PALETTES[DEFAULT_THEME])

STYLE_SHEET = """
QMainWindow, QDialog {{ background: {window}; }}
QMainWindow::separator {{ background: {window}; width: 4px; height: 4px; }}
QMainWindow::separator:hover {{ background: {accent}; }}
QDockWidget {{ color: {dim}; font-weight: 600; }}
QDockWidget::title {{
    background: {window}; padding: 6px 10px 4px 10px; text-align: left;
}}
QDockWidget > QWidget {{ background: {panel}; border-top: 1px solid {border}; }}
QToolBar {{
    background: {window}; border: none; padding: 4px 6px; spacing: 2px;
}}
QToolBar::separator {{ background: {border}; width: 1px; margin: 4px 6px; }}
QToolButton {{
    color: {text}; background: transparent; border: 1px solid transparent;
    border-radius: 5px; padding: 4px 8px;
}}
QToolButton:hover {{ background: {raised}; border-color: {border}; }}
QToolButton:pressed, QToolButton:checked {{ background: {pressed}; border-color: {accent}; }}
QToolButton:disabled {{ color: #59626d; }}
QMenuBar {{ background: {window}; color: {text}; }}
QMenuBar::item:selected {{ background: {raised}; border-radius: 4px; }}
QMenu {{ background: {panel}; color: {text}; border: 1px solid {border}; padding: 4px; }}
QMenu::item {{ padding: 5px 22px 5px 14px; border-radius: 4px; }}
QMenu::item:selected {{ background: {selection}; color: {selection_text}; }}
QMenu::separator {{ height: 1px; background: {border}; margin: 4px 6px; }}
QTreeWidget, QTreeView, QListView {{
    background: {panel}; border: none; outline: none;
    show-decoration-selected: 1;
}}
QTreeView::item {{ padding: 3px 2px; border-radius: 4px; }}
QTreeView::item:selected {{ background: {selection}; color: {selection_text}; }}
QTreeView::item:hover:!selected {{ background: {raised}; }}
QHeaderView::section {{
    background: {panel}; color: {dim}; border: none; border-bottom: 1px solid {border};
    padding: 4px 6px;
}}
QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox, QPlainTextEdit, QTextEdit {{
    background: {base}; color: {text}; border: 1px solid {border}; border-radius: 5px;
    padding: 3px 6px; selection-background-color: {selection};
    selection-color: {selection_text};
}}
QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus {{
    border-color: {accent};
}}
QComboBox::drop-down {{
    subcontrol-origin: padding; subcontrol-position: center right; width: 22px; border: none;
}}
QComboBox::down-arrow {{ image: url("{arrow}"); width: 10px; height: 10px; }}
QSpinBox, QDoubleSpinBox {{ padding-right: 22px; }}
QSpinBox::up-button, QDoubleSpinBox::up-button {{
    subcontrol-origin: border; subcontrol-position: top right; width: 20px;
    border: none; border-left: 1px solid {border}; border-top-right-radius: 5px;
}}
QSpinBox::down-button, QDoubleSpinBox::down-button {{
    subcontrol-origin: border; subcontrol-position: bottom right; width: 20px;
    border: none; border-left: 1px solid {border}; border-bottom-right-radius: 5px;
}}
QSpinBox::up-button:hover, QDoubleSpinBox::up-button:hover,
QSpinBox::down-button:hover, QDoubleSpinBox::down-button:hover {{ background: {raised}; }}
QSpinBox::up-arrow, QDoubleSpinBox::up-arrow {{
    image: url("{arrow_up}"); width: 8px; height: 8px;
}}
QSpinBox::down-arrow, QDoubleSpinBox::down-arrow {{
    image: url("{arrow}"); width: 8px; height: 8px;
}}
QComboBox QAbstractItemView {{
    background: {panel}; border: 1px solid {border}; selection-background-color: {selection};
    selection-color: {selection_text};
}}
QPushButton {{
    background: {raised}; color: {text}; border: 1px solid {border}; border-radius: 5px;
    padding: 5px 12px;
}}
QPushButton:hover {{ border-color: {accent}; }}
QPushButton:pressed {{ background: {pressed}; }}
QCheckBox {{ color: {text}; spacing: 6px; }}
QSlider::groove:horizontal {{ height: 4px; background: {border}; border-radius: 2px; }}
QSlider::sub-page:horizontal {{ background: {accent}; border-radius: 2px; }}
QSlider::handle:horizontal {{
    background: {cursor}; width: 12px; height: 12px; margin: -5px 0; border-radius: 6px;
}}
QScrollBar:vertical, QScrollBar:horizontal {{ background: transparent; width: 10px; height: 10px; }}
QScrollBar::handle {{
    background: {border}; border-radius: 4px; min-height: 24px; min-width: 24px;
}}
QScrollBar::add-line, QScrollBar::sub-line {{ width: 0; height: 0; }}
QStatusBar {{ background: {window}; color: {dim}; }}
QToolTip {{
    background: {raised}; color: {text}; border: 1px solid {border}; padding: 4px;
}}
QLabel {{ color: {text}; }}
QLabel:disabled {{ color: {dim}; }}
QScrollArea {{ border: none; background: {panel}; }}
QScrollArea > QWidget > QWidget {{ background: {panel}; }}
"""


def apply_theme(app: QApplication, name: str = DEFAULT_THEME) -> str:
    """Give the application LAGOON's look; ``name`` is "light" (default) or "dark".

    Call it before the main window is built: icons and painted widgets read the
    colours when they are created or painted.
    """
    if name not in THEMES:
        name = DEFAULT_THEME
    COLORS.clear()
    COLORS.update(THEMES[name])
    SERIES_COLORS[:] = SERIES_PALETTES[name]
    app.setStyle("Fusion")
    palette = QPalette()
    c = {key: QColor(value) for key, value in COLORS.items()}
    palette.setColor(QPalette.Window, c["window"])
    palette.setColor(QPalette.WindowText, c["text"])
    palette.setColor(QPalette.Base, c["base"])
    palette.setColor(QPalette.AlternateBase, c["alternate"])
    palette.setColor(QPalette.ToolTipBase, c["raised"])
    palette.setColor(QPalette.ToolTipText, c["text"])
    palette.setColor(QPalette.Text, c["text"])
    palette.setColor(QPalette.Button, c["raised"])
    palette.setColor(QPalette.ButtonText, c["text"])
    palette.setColor(QPalette.BrightText, QColor("#ffffff"))
    palette.setColor(QPalette.Highlight, c["selection"])
    palette.setColor(QPalette.HighlightedText, c["selection_text"])
    palette.setColor(QPalette.Link, c["accent"])
    palette.setColor(QPalette.PlaceholderText, c["dim"])
    for role in (QPalette.WindowText, QPalette.Text, QPalette.ButtonText):
        palette.setColor(QPalette.Disabled, role, c["dim"])
    app.setPalette(palette)
    app.setStyleSheet(
        STYLE_SHEET.format(
            **COLORS,
            arrow=_arrow_image(COLORS["dim"]),
            arrow_up=_arrow_image(COLORS["dim"], up=True),
        )
    )
    font = QFont(app.font())
    if font.pointSizeF() > 0:
        font.setPointSizeF(max(font.pointSizeF(), 12.0))
    app.setFont(font)
    return name


def _arrow_image(color: str, up: bool = False) -> str:
    """A small chevron (down, or up) for combo and spin boxes, as a PNG the style
    sheet can load.

    Styling a combo box's drop-down in a style sheet removes Qt's own arrow, so
    without this the boxes look like text fields. A 2x image is written next to it
    for sharp arrows on Retina screens.
    """
    folder = (
        Path(tempfile.gettempdir()) / f"lagoon-theme-{os.getuid() if hasattr(os, 'getuid') else 0}"
    )
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"arrow-{'up' if up else 'down'}-{color.lstrip('#')}.png"
    for scale, target in ((1, path), (2, path.with_name(path.stem + "@2x.png"))):
        if target.exists():
            continue
        image = QImage(10 * scale, 10 * scale, QImage.Format_ARGB32)
        image.fill(Qt.transparent)
        painter = QPainter(image)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.scale(scale, scale)
        painter.setPen(QPen(QColor(color), 1.6, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))
        if up:
            painter.drawPolyline([QPointF(1.5, 6.5), QPointF(5.0, 3.0), QPointF(8.5, 6.5)])
        else:
            painter.drawPolyline([QPointF(1.5, 3.5), QPointF(5.0, 7.0), QPointF(8.5, 3.5)])
        painter.end()
        image.save(str(target))
    return path.as_posix()
