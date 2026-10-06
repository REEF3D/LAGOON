# SPDX-License-Identifier: GPL-3.0-or-later
"""LAGOON's icons, drawn as simple line art with QPainter.

Drawing them in code keeps them sharp at any screen scale, matches the theme's
text colour, and needs no image files or SVG module.
"""

from __future__ import annotations

from collections.abc import Callable

from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QColor, QIcon, QPainter, QPainterPath, QPen, QPixmap

from lagoonviz.gui.theme import COLORS

SIZE = 64  # drawn on a 64 x 64 grid, scaled by Qt

Draw = Callable[[QPainter], None]


def _wave(p: QPainter, y: float, amplitude: float = 6.0) -> None:
    path = QPainterPath(QPointF(8, y))
    for i in range(4):
        x0 = 8 + i * 12
        path.cubicTo(x0 + 4, y - amplitude, x0 + 8, y - amplitude, x0 + 12, y)
        amplitude = -amplitude
    p.drawPath(path)


def _folder(p: QPainter) -> None:
    path = QPainterPath(QPointF(8, 18))
    path.lineTo(24, 18)
    path.lineTo(29, 24)
    path.lineTo(56, 24)
    path.lineTo(56, 50)
    path.lineTo(8, 50)
    path.closeSubpath()
    p.drawPath(path)
    p.save()
    p.setPen(QPen(QColor(COLORS["accent"]), 4, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))
    _wave(p, 39, 4)
    p.restore()


def _document(p: QPainter) -> None:
    path = QPainterPath(QPointF(16, 8))
    path.lineTo(38, 8)
    path.lineTo(50, 20)
    path.lineTo(50, 56)
    path.lineTo(16, 56)
    path.closeSubpath()
    p.drawPath(path)
    p.drawLine(QPointF(38, 8), QPointF(38, 20))
    p.drawLine(QPointF(38, 20), QPointF(50, 20))


def _open(p: QPainter) -> None:
    _document(p)
    p.drawLine(QPointF(24, 36), QPointF(42, 36))
    p.drawLine(QPointF(24, 44), QPointF(36, 44))


def _save(p: QPainter) -> None:
    p.drawRoundedRect(QRectF(10, 10, 44, 44), 5, 5)
    p.drawRect(QRectF(20, 10, 22, 14))
    p.drawRect(QRectF(18, 34, 28, 20))


def _arrow(p: QPainter, flip: bool) -> None:
    path = QPainterPath(QPointF(48, 46))
    path.cubicTo(48, 22, 30, 18, 14, 26)
    p.save()
    if flip:
        p.translate(SIZE, 0)
        p.scale(-1, 1)
    p.drawPath(path)
    p.drawLine(QPointF(14, 26), QPointF(24, 16))
    p.drawLine(QPointF(14, 26), QPointF(26, 32))
    p.restore()


def _free_surface(p: QPainter) -> None:
    _wave(p, 26, 8)
    p.save()
    p.setPen(QPen(QColor(COLORS["dim"]), 3, Qt.DashLine, Qt.RoundCap))
    p.drawLine(QPointF(8, 46), QPointF(56, 46))
    p.restore()


def _bathymetry(p: QPainter) -> None:
    path = QPainterPath(QPointF(6, 50))
    path.cubicTo(18, 50, 20, 30, 32, 30)
    path.cubicTo(44, 30, 46, 44, 58, 44)
    p.drawPath(path)
    p.drawLine(QPointF(6, 56), QPointF(58, 56))


def _volume(p: QPainter) -> None:
    front = QRectF(10, 22, 32, 32)
    p.drawRect(front)
    p.drawLine(QPointF(10, 22), QPointF(22, 10))
    p.drawLine(QPointF(42, 22), QPointF(54, 10))
    p.drawLine(QPointF(42, 54), QPointF(54, 42))
    p.drawLine(QPointF(22, 10), QPointF(54, 10))
    p.drawLine(QPointF(54, 10), QPointF(54, 42))


def _slice(p: QPainter) -> None:
    _volume(p)
    p.save()
    p.setPen(QPen(QColor(COLORS["cursor"]), 4, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))
    path = QPainterPath(QPointF(26, 54))
    path.lineTo(26, 22)
    path.lineTo(38, 10)
    path.lineTo(38, 42)
    path.closeSubpath()
    p.drawPath(path)
    p.restore()


def _series(p: QPainter) -> None:
    p.drawLine(QPointF(10, 8), QPointF(10, 54))
    p.drawLine(QPointF(10, 54), QPointF(58, 54))
    p.save()
    p.setPen(QPen(QColor(COLORS["cursor"]), 4, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))
    path = QPainterPath(QPointF(14, 40))
    for x, y in ((22, 24), (30, 36), (40, 16), (48, 30), (56, 22)):
        path.lineTo(x, y)
    p.drawPath(path)
    p.restore()


def _play(p: QPainter) -> None:
    path = QPainterPath(QPointF(20, 12))
    path.lineTo(50, 32)
    path.lineTo(20, 52)
    path.closeSubpath()
    p.fillPath(path, p.pen().color())


def _pause(p: QPainter) -> None:
    color = p.pen().color()
    p.fillRect(QRectF(18, 12, 9, 40), color)
    p.fillRect(QRectF(37, 12, 9, 40), color)


def _step(p: QPainter, forward: bool, to_end: bool) -> None:
    p.save()
    if not forward:
        p.translate(SIZE, 0)
        p.scale(-1, 1)
    color = p.pen().color()
    path = QPainterPath(QPointF(18, 14))
    path.lineTo(42, 32)
    path.lineTo(18, 50)
    path.closeSubpath()
    p.fillPath(path, color)
    if to_end:
        p.fillRect(QRectF(44, 14, 6, 36), color)
    p.restore()


def _frame_all(p: QPainter) -> None:
    for x, y, dx, dy in ((10, 10, 1, 1), (54, 10, -1, 1), (10, 54, 1, -1), (54, 54, -1, -1)):
        p.drawLine(QPointF(x, y), QPointF(x + 12 * dx, y))
        p.drawLine(QPointF(x, y), QPointF(x, y + 12 * dy))
    p.drawEllipse(QRectF(24, 24, 16, 16))


def _reload(p: QPainter) -> None:
    # a circular arrow: most of a circle, with the head at its open end
    p.drawArc(QRectF(14, 14, 36, 36), 60 * 16, 290 * 16)
    p.drawLine(QPointF(41, 16), QPointF(50, 14))
    p.drawLine(QPointF(41, 16), QPointF(44, 25))


def _camera(p: QPainter) -> None:
    p.drawRoundedRect(QRectF(8, 20, 48, 32), 6, 6)
    p.drawRect(QRectF(22, 12, 18, 8))
    p.drawEllipse(QRectF(22, 26, 20, 20))


def _diamond(p: QPainter, x: float, y: float, r: float, fill: bool) -> None:
    path = QPainterPath(QPointF(x, y - r))
    path.lineTo(x + r, y)
    path.lineTo(x, y + r)
    path.lineTo(x - r, y)
    path.closeSubpath()
    if fill:
        p.fillPath(path, p.pen().color())
    p.drawPath(path)


def _keyframe(p: QPainter) -> None:
    # a keyframe diamond with a plus: add a camera key here
    _diamond(p, 28, 34, 18, False)
    p.drawLine(QPointF(50, 8), QPointF(50, 24))
    p.drawLine(QPointF(42, 16), QPointF(58, 16))


def _camera_path(p: QPainter) -> None:
    # a smooth path through two keyframes
    path = QPainterPath(QPointF(8, 50))
    path.cubicTo(24, 50, 22, 14, 40, 14)
    path.cubicTo(50, 14, 52, 22, 58, 22)
    p.drawPath(path)
    _diamond(p, 14, 48, 7, True)
    _diamond(p, 40, 14, 7, True)


def _repeat(p: QPainter) -> None:
    # two arrows chasing each other round a loop
    top = QPainterPath(QPointF(14, 34))
    top.lineTo(14, 26)
    top.quadTo(14, 18, 22, 18)
    top.lineTo(48, 18)
    p.drawPath(top)
    p.drawLine(QPointF(48, 18), QPointF(41, 11))
    p.drawLine(QPointF(48, 18), QPointF(41, 25))
    bottom = QPainterPath(QPointF(50, 30))
    bottom.lineTo(50, 38)
    bottom.quadTo(50, 46, 42, 46)
    bottom.lineTo(16, 46)
    p.drawPath(bottom)
    p.drawLine(QPointF(16, 46), QPointF(23, 39))
    p.drawLine(QPointF(16, 46), QPointF(23, 53))


DRAWINGS: dict[str, Draw] = {
    "open_case": _folder,
    "open": _open,
    "save": _save,
    "undo": lambda p: _arrow(p, False),
    "redo": lambda p: _arrow(p, True),
    "free_surface": _free_surface,
    "bathymetry": _bathymetry,
    "volume": _volume,
    "slice": _slice,
    "series": _series,
    "play": _play,
    "pause": _pause,
    "next": lambda p: _step(p, True, False),
    "previous": lambda p: _step(p, False, False),
    "last": lambda p: _step(p, True, True),
    "first": lambda p: _step(p, False, True),
    "frame_all": _frame_all,
    "screenshot": _camera,
    "reload": _reload,
    "keyframe": _keyframe,
    "camera_path": _camera_path,
    "repeat": _repeat,
}

_cache: dict[tuple[str, str], QIcon] = {}


def icon(name: str, color: str | None = None) -> QIcon:
    """The icon ``name`` in the theme's text colour (or ``color``)."""
    color = color or COLORS["text"]
    key = (name, color)
    if key not in _cache:
        result = QIcon()
        for scale in (1, 2):
            pixmap = QPixmap(SIZE * scale // 2, SIZE * scale // 2)
            pixmap.fill(Qt.transparent)
            painter = QPainter(pixmap)
            painter.setRenderHint(QPainter.Antialiasing)
            painter.scale(scale / 2, scale / 2)
            pen = QPen(QColor(color), 4, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin)
            painter.setPen(pen)
            DRAWINGS[name](painter)
            painter.end()
            result.addPixmap(pixmap)
        _cache[key] = result
    return _cache[key]
