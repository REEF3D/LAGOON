# SPDX-License-Identifier: GPL-3.0-or-later
"""Timeline: play, step and scrub through the frames of the loaded cases.

With a case loaded the slider moves frame by frame; the time is shown in seconds
when the output interval is known (``ctrl.txt`` or the case's property), else as
the frame number. Playback waits for each frame to be loaded before moving on and
asks the loader to read the next frames ahead.

Camera keyframes: the key button stores the 3D view at this time, the slider marks
the keys, and the path button makes the views and films follow the smooth camera
path through them (``lagoonviz.core.camera_path``).

The play range: two thick stripes on the slider, at its ends unless moved, mark
where playing starts and ends (dragged with the mouse; saved with the session, and
the export dialog starts from it). The repeat button plays the range again and
again; off, playing stops at its end.
"""

from __future__ import annotations

import time
from collections.abc import Callable

import numpy as np
from PySide6.QtCore import QPointF, QRectF, QSize, Qt, QTimer, Signal
from PySide6.QtGui import QColor, QPainter, QPainterPath
from PySide6.QtWidgets import (
    QDoubleSpinBox,
    QHBoxLayout,
    QLabel,
    QSlider,
    QSpinBox,
    QStyle,
    QStyleOptionSlider,
    QToolButton,
    QWidget,
)

from lagoonviz.core.data.manager import BackgroundLoader
from lagoonviz.gui.bridge import SessionBridge
from lagoonviz.gui.icons import icon
from lagoonviz.gui.perf import PERF
from lagoonviz.gui.theme import COLORS

#: slider resolution without data (time in seconds up to "End")
STEPS = 1000
DEFAULT_FPS = 12
#: the fastest playback offered (ProMotion screens draw 120 frames per second)
MAX_FPS = 240


#: how near (pixels) a press must be to a range stripe to drag it
STRIPE_REACH = 7


class ClickSlider(QSlider):
    """A slider that goes where it is pressed and follows the mouse until let go.

    Qt's own slider, pressed beside its handle, steps a page towards the mouse and
    keeps stepping while the button is held. While playing, the handle moves away
    with every frame, so it stepped back again and again: the handle jumped back and
    forth and never got anywhere. Here the left button always drags: the handle goes
    under the mouse at once, the slider counts as held (``isSliderDown``) until the
    button is let go, and nothing steps on its own.
    """

    #: a range stripe is being dragged: the range (start, end) in slider values
    rangeMoved = Signal(int, int)
    #: a range stripe was let go
    rangeReleased = Signal()

    def __init__(self, *args) -> None:
        super().__init__(*args)
        self._dragging = False
        #: slider values marked with a small diamond (the camera keyframes)
        self.markers: list[int] = []
        #: the play range in slider values (None: the whole slider)
        self.play_range: tuple[int, int] | None = None
        self._stripe: int | None = None  # 0 or 1 while a stripe is dragged

    def bounds(self) -> tuple[int, int]:
        """The play range in slider values, within the slider."""
        low, high = self.play_range or (self.minimum(), self.maximum())
        low = min(max(low, self.minimum()), self.maximum())
        high = min(max(high, low), self.maximum())
        return low, high

    def set_play_range(self, play_range: tuple[int, int] | None) -> None:
        self.play_range = None if play_range is None else (int(play_range[0]), int(play_range[1]))
        self.update()

    def _stripe_at(self, x: float) -> int | None:
        """The range stripe under a horizontal position (the nearer one), or None."""
        low, high = self.bounds()
        distances = [abs(x - self.x_of(low)), abs(x - self.x_of(high))]
        nearest = (
            0
            if distances[0] < distances[1] or (distances[0] == distances[1] and x < self.x_of(low))
            else 1
        )
        return nearest if distances[nearest] <= STRIPE_REACH else None

    def set_markers(self, values: list[int]) -> None:
        self.markers = sorted(set(int(v) for v in values))
        self.update()

    def _groove(self) -> tuple[float, float, QStyleOptionSlider]:
        """Where the handle's centre is at the minimum, and how far it can travel."""
        option = QStyleOptionSlider()
        self.initStyleOption(option)
        groove = self.style().subControlRect(QStyle.CC_Slider, option, QStyle.SC_SliderGroove, self)
        handle = self.style().subControlRect(QStyle.CC_Slider, option, QStyle.SC_SliderHandle, self)
        return groove.x() + handle.width() / 2, max(groove.width() - handle.width(), 1), option

    def x_of(self, value: int) -> float:
        """The horizontal position of a slider value in the widget."""
        start, span, option = self._groove()
        return start + QStyle.sliderPositionFromValue(
            self.minimum(), self.maximum(), int(value), round(span), option.upsideDown
        )

    def paintEvent(self, event) -> None:  # noqa: N802 (Qt naming)
        super().paintEvent(event)
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        self._paint_range(painter)
        if not self.markers:
            painter.end()
            return
        colour = QColor(COLORS["cursor"])
        r = 4.5
        y = self.height() - r - 1  # under the groove, clear of the handle
        for value in self.markers:
            if not self.minimum() <= value <= self.maximum():
                continue
            x = self.x_of(value)
            path = QPainterPath(QPointF(x, y - r))
            path.lineTo(x + r, y)
            path.lineTo(x, y + r)
            path.lineTo(x - r, y)
            path.closeSubpath()
            painter.fillPath(path, colour)
        painter.end()

    def _paint_range(self, painter: QPainter) -> None:
        """The play range: outside it the slider is dimmed, at its ends thick stripes."""
        low, high = self.bounds()
        x0, x1 = self.x_of(low), self.x_of(high)
        shade = QColor(COLORS["text"])
        shade.setAlphaF(0.12)
        if low > self.minimum():
            painter.fillRect(QRectF(0, 2, x0, self.height() - 4), shade)
        if high < self.maximum():
            painter.fillRect(QRectF(x1, 2, self.width() - x1, self.height() - 4), shade)
        stripe = QColor(COLORS["accent"])
        for x in (x0, x1):
            painter.fillRect(QRectF(x - 2.5, 2, 5, self.height() - 4), stripe)

    def wheelEvent(self, event) -> None:  # noqa: N802 (Qt naming)
        event.ignore()  # scrolling (two fingers on a trackpad) must not move the time

    def mousePressEvent(self, event) -> None:  # noqa: N802 (Qt naming)
        if event.button() != Qt.LeftButton:
            super().mousePressEvent(event)
            return
        self._stripe = self._stripe_at(event.position().x())
        if self._stripe is not None:  # a range stripe: drag it, the time stays
            event.accept()
            return
        self._dragging = True
        self.setSliderDown(True)  # first, so playing leaves the handle alone
        self.setSliderPosition(self.value_at(event.position().x()))
        event.accept()

    def mouseMoveEvent(self, event) -> None:  # noqa: N802 (Qt naming)
        if self._stripe is not None:
            value = self.value_at(event.position().x())
            low, high = self.bounds()
            if self._stripe == 0:
                low = min(value, high)
            else:
                high = max(value, low)
            if (low, high) != self.bounds():
                self.play_range = (low, high)
                self.update()
                self.rangeMoved.emit(low, high)
            event.accept()
            return
        if not self._dragging:
            super().mouseMoveEvent(event)
            return
        self.setSliderPosition(self.value_at(event.position().x()))
        event.accept()

    def mouseReleaseEvent(self, event) -> None:  # noqa: N802 (Qt naming)
        if self._stripe is not None and event.button() == Qt.LeftButton:
            self._stripe = None
            self.rangeReleased.emit()
            event.accept()
            return
        if not self._dragging or event.button() != Qt.LeftButton:
            super().mouseReleaseEvent(event)
            return
        self._dragging = False  # the handle stays where the press or the last move put it
        self.setSliderDown(False)
        event.accept()

    def value_at(self, x: float) -> int:
        """The slider value under a horizontal position in the widget."""
        start, span, option = self._groove()
        return QStyle.sliderValueFromPosition(
            self.minimum(), self.maximum(), round(x - start), round(span), option.upsideDown
        )


class TimelinePanel(QWidget):
    #: the key button: store the 3D view as a camera keyframe at this time
    keyRequested = Signal()

    def __init__(
        self,
        bridge: SessionBridge,
        parent: QWidget | None = None,
        loader: BackgroundLoader | None = None,
        busy: Callable[[], bool] | None = None,
    ):
        super().__init__(parent)
        self.setObjectName("timeline")
        self.session = bridge.session
        self.loader = loader
        self.busy = busy or (lambda: False)
        self.n_frames = 0
        self.times = np.zeros(0)
        self.steps = np.zeros(0, int)
        self.first_frame = 0  # outputs before it hold no data of the shown runs
        self.dt = 1.0
        self.unit = "frame"

        def tool(name: str, tip: str, slot) -> QToolButton:
            button = QToolButton()
            button.setIcon(icon(name))
            button.setIconSize(QSize(22, 22))
            button.setMinimumSize(38, 36)
            button.setToolTip(tip)
            button.setAutoRaise(True)
            if slot is not None:
                button.clicked.connect(slot)
            return button

        self.first_button = tool("first", "Start of the play range", self.go_to_start)
        self.prev_button = tool("previous", "Previous frame", lambda: self.step(-1))
        self.play_button = QToolButton()
        self.play_button.setText("Play")
        self.play_button.setIcon(icon("play", COLORS["cursor"]))
        self.play_button.setIconSize(QSize(28, 28))
        self.play_button.setMinimumSize(48, 40)
        self.play_button.setToolTip("Play / pause (Space in the viewport)")
        self.play_button.setCheckable(True)
        self.play_button.setAutoRaise(True)
        self.play_button.toggled.connect(self._toggle_play)
        self.next_button = tool("next", "Next frame", lambda: self.step(1))
        self.last_button = tool("last", "End of the play range", self.go_to_end)
        self.repeat_button = tool(
            "repeat", "Repeat: play the range again and again (off: stop at its end)", None
        )
        self.repeat_button.setCheckable(True)
        self.repeat_button.setChecked(True)

        self.key_button = tool(
            "keyframe", "Camera keyframe: the 3D view at this time (Ctrl+K)", self.keyRequested.emit
        )
        self.path_button = tool("camera_path", "Follow the camera path through the keyframes", None)
        self.path_button.setCheckable(True)
        self.path_button.toggled.connect(self._toggle_path)

        self.slider = ClickSlider(Qt.Horizontal)
        self.slider.setRange(0, STEPS)
        self.slider.valueChanged.connect(self._on_slider)
        # let go: the time is where the handle is, whatever happened meanwhile
        self.slider.sliderReleased.connect(lambda: self._on_slider(self.slider.value()))
        # the play range: dragged on the slider, kept in the session when let go
        self.slider.rangeReleased.connect(self._store_range)

        self.label = QLabel()
        self.label.setMinimumWidth(150)

        self.fps = QSpinBox()
        self.fps.setRange(1, MAX_FPS)
        self.fps.setValue(DEFAULT_FPS)
        self.fps.setSuffix(" fps")
        self.fps.setToolTip("Playback speed in frames per second")

        self.end_label = QLabel("End")
        self.end = QDoubleSpinBox()
        self.end.setRange(0.1, 1e7)
        self.end.setValue(60.0)
        self.end.setSuffix(" s")
        self.end.setToolTip("End time when no case is loaded")
        self.end.valueChanged.connect(lambda _: self._on_end())

        layout = QHBoxLayout(self)
        for widget in (self.first_button, self.prev_button, self.play_button, self.next_button):
            layout.addWidget(widget)
        layout.addWidget(self.last_button)
        layout.addWidget(self.repeat_button)
        layout.addWidget(self.key_button)
        layout.addWidget(self.path_button)
        layout.addWidget(self.slider, 1)
        layout.addWidget(self.label)
        layout.addWidget(self.fps)
        layout.addWidget(self.end_label)
        layout.addWidget(self.end)
        # shown in the full screen viewer, where there is no toolbar or menu to leave by
        self.exit_full_screen = QToolButton()
        self.exit_full_screen.setText("Exit full screen")
        self.exit_full_screen.setToolTip("Back to the window (Esc)")
        self.exit_full_screen.hide()
        layout.addWidget(self.exit_full_screen)

        # playback: one single-shot timer, scheduled after each step. Waiting for a
        # frame re-checks every few milliseconds, so playing goes on as soon as the
        # data is there instead of a whole frame interval later.
        self.timer = QTimer(self)
        self.timer.setSingleShot(True)
        self.timer.setTimerType(Qt.PreciseTimer)  # a few ms matter at 120 fps
        self.timer.timeout.connect(self._tick)
        self._last_step = 0.0
        bridge.timeChanged.connect(self._show_time)
        bridge.sceneChanged.connect(self._on_scene)
        self.update_range()

    def _on_scene(self, event: str, _item: str) -> None:
        if event == "play_range":
            self.show_play_range()
        elif event in ("camera_keys", "sun"):
            self.update_keys()
        else:
            self.update_range()

    # ------------------------------------------------------------- range
    @property
    def has_frames(self) -> bool:
        return self.n_frames > 0

    @property
    def end_time(self) -> float:
        if self.has_frames:
            return float(self.times[-1])
        return self.end.value()

    def update_range(self) -> None:
        """Follow the frames of the loaded cases: REEF3D's output numbers (times from
        the output interval), or the times of the run log when REEF3D wrote one."""
        times, steps, unit = self.session.data.timeline()
        self.times = np.asarray(times, dtype=np.float64)
        self.steps = np.asarray(steps, dtype=int)
        n = int(self.steps[-1]) + 1 if self.steps.size else 0
        self.n_frames = n
        self.first_frame = int(self.steps[0]) if self.steps.size else 0
        self.unit = unit
        dt = self.session.data.print_interval()
        self.dt = dt or 1.0
        self.end.setVisible(not n)
        self.end_label.setVisible(not n)
        for button in (self.first_button, self.prev_button, self.next_button, self.last_button):
            button.setEnabled(self.steps.size > 1)
        self.slider.blockSignals(True)
        self.slider.setRange(self.first_frame, max(n - 1, 0) if n else STEPS)
        self.slider.setPageStep(max(1, self.steps.size // 20) if n else 50)
        self.slider.blockSignals(False)
        self._fit_label()
        self._show_time(self.session.scene.time)
        self.update_keys()
        self.show_play_range()

    # ------------------------------------------------------------- play range
    def time_of(self, value: int) -> float:
        """The time of a slider value: its frame's, or its share of the end time."""
        if self.has_frames:
            frame = min(max(int(value), self.first_frame), self.n_frames - 1)
            return float(self.times[frame - self.first_frame])
        return value / STEPS * self.end_time

    def show_play_range(self) -> None:
        """The stripes where the session's play range is (at the ends without one)."""
        played = self.session.scene.play_range
        if played is None:
            self.slider.set_play_range(None)
            return
        self.slider.set_play_range((self.slider_value(played[0]), self.slider_value(played[1])))

    def play_bounds(self) -> tuple[int, int]:
        """The play range in slider values (frames)."""
        return self.slider.bounds()

    def _store_range(self) -> None:
        """A stripe let go: the range into the session (None when it is all of it)."""
        low, high = self.slider.bounds()
        if (low, high) == (self.slider.minimum(), self.slider.maximum()):
            self.session.set_play_range(None)
        else:
            self.session.set_play_range(self.time_of(low), self.time_of(high))

    def go_to_start(self) -> None:
        self._go_to_value(self.play_bounds()[0])

    def go_to_end(self) -> None:
        self._go_to_value(self.play_bounds()[1])

    def _go_to_value(self, value: int) -> None:
        if self.has_frames:
            self.go_to_frame(value)
        else:
            self.session.set_time(self.time_of(value))

    def update_keys(self) -> None:
        """Mark the camera keyframes on the slider; the path button follows the scene."""
        keys = self.session.scene.camera_keys
        self.slider.set_markers([self.slider_value(k["time"]) for k in keys])
        self.path_button.blockSignals(True)
        self.path_button.setChecked(bool(self.session.scene.camera_path and keys))
        self.path_button.blockSignals(False)
        self.path_button.setEnabled(bool(keys))

    def _toggle_path(self, on: bool) -> None:
        self.session.set_camera_path(on)

    def slider_value(self, time: float) -> int:
        """The slider value of a time: its frame, or its share of the end time."""
        if self.has_frames:
            return self.frame_of(time)
        return round(min(max(time / self.end_time, 0.0), 1.0) * STEPS)

    def key_here(self, time: float | None = None) -> float | None:
        """The time of the camera keyframe on the same slider step as ``time``
        (default: now), or None."""
        time = self.session.scene.time if time is None else time
        here = self.slider_value(time)
        for key in self.session.scene.camera_keys:
            if self.slider_value(key["time"]) == here:
                return float(key["time"])
        return None

    def _on_end(self) -> None:
        self._show_time(self.session.scene.time)
        self.update_keys()

    def _fit_label(self) -> None:
        """A fixed width for the time label, as wide as its longest text. When it took
        the width of each text, the slider beside it changed width with every frame and
        its handle jiggled back and forth while playing."""
        end = max(self.end_time, 1.0)
        last = max(self.n_frames - 1, 1)
        digits = "8" * len(f"{end:.2f}".split(".")[0])
        sample = (
            f"t = {digits}.88 s  (frame {'8' * len(str(last))} / {'8' * len(str(last))})"
            if self.unit == "s" or not self.has_frames
            else f"frame {'8' * len(str(last))} / {'8' * len(str(last))}"
        )
        self.label.setFixedWidth(self.label.fontMetrics().horizontalAdvance(sample) + 12)

    # ------------------------------------------------------------- time
    def _index_of(self, time: float) -> int:
        """Index of the timeline frame nearest to a time."""
        right = int(np.searchsorted(self.times, time))
        if right <= 0:
            return 0
        if right >= self.times.size:
            return self.times.size - 1
        left = right - 1
        return left if time - self.times[left] <= self.times[right] - time else right

    def frame_of(self, time: float) -> int:
        if not self.has_frames:
            return 0
        return int(self.steps[self._index_of(time)])

    def go_to_frame(self, frame: int) -> None:
        if self.has_frames:
            frame = min(max(int(frame), self.first_frame), self.n_frames - 1)
            self.session.set_time(float(self.times[frame - self.first_frame]))

    def step(self, frames: int) -> None:
        self.go_to_frame(self.frame_of(self.session.scene.time) + frames)

    def _on_slider(self, value: int) -> None:
        if self.has_frames:
            self.go_to_frame(value)
        else:
            self.session.set_time(value / STEPS * self.end_time)

    def _show_time(self, time: float) -> None:
        held = self.slider.isSliderDown()  # the handle stays under the mouse
        self.slider.blockSignals(True)
        if self.has_frames:
            frame = self.frame_of(time)
            if not held:
                self.slider.setValue(frame)
            where = f"frame {frame} / {self.n_frames - 1}"
            self.label.setText(f"t = {time:.2f} s  ({where})" if self.unit == "s" else where)
        else:
            if not held:
                self.slider.setValue(round(min(time / self.end_time, 1.0) * STEPS))
            self.label.setText(f"t = {time:.2f} s")
        self.slider.blockSignals(False)

    def refresh_icons(self) -> None:
        """Draw the buttons again in the current theme's colours."""
        for button, name in (
            (self.first_button, "first"),
            (self.prev_button, "previous"),
            (self.next_button, "next"),
            (self.last_button, "last"),
            (self.key_button, "keyframe"),
            (self.path_button, "camera_path"),
            (self.repeat_button, "repeat"),
        ):
            button.setIcon(icon(name))
        playing = self.play_button.isChecked()
        self.play_button.setIcon(icon("pause" if playing else "play", COLORS["cursor"]))

    # ------------------------------------------------------------- playback
    def _toggle_play(self, playing: bool) -> None:
        self.play_button.setText("Pause" if playing else "Play")
        self.play_button.setIcon(icon("pause" if playing else "play", COLORS["cursor"]))
        if playing:
            low, high = self.play_bounds()
            here = self.slider_value(self.session.scene.time)
            if not low <= here < high:  # outside the range, or at its end: from its start
                self.go_to_start()
            self._last_step = 0.0
            self.timer.start(0)
        else:
            self.timer.stop()

    def _tick(self) -> None:
        if not self.play_button.isChecked():
            return
        if self.slider.isSliderDown():
            # the user holds the slider: playing waits and goes on from where it is let go
            self.timer.start(20)
            return
        if self.busy():
            PERF.count("tick waiting for data")
            self.timer.start(4)
            return
        interval = 1.0 / max(self.fps.value(), 1)
        wait = self._last_step + interval - time.monotonic()
        if wait > 0.002:
            self.timer.start(max(int(wait * 1000), 1))
            return
        self._last_step = time.monotonic()
        self._advance()
        self.timer.start(max(int(interval * 1000) - 2, 1))

    def _advance(self) -> None:
        if self.busy() or self.slider.isSliderDown():
            return  # wait until the current frame is on screen, or the slider is let go
        PERF.count("frames advanced")
        low, high = self.play_bounds()
        if self.has_frames:
            frame = self.frame_of(self.session.scene.time) + 1
            if frame > high or frame < low:
                if not self.repeat_button.isChecked():
                    self.play_button.setChecked(False)  # once through: stop at the end
                    return
                frame = low
            self.go_to_frame(frame)
            if self.loader is not None:
                self.loader.prefetch(
                    self.session.data.data_items(),
                    self.session.scene.time,
                    frames_ahead=max(8, self.fps.value() // 4),  # a quarter second
                    times=self.times,
                )
            return
        step = self.end_time / STEPS * 2
        time = self.session.scene.time + step
        if time > self.time_of(high) + 1e-9:
            if not self.repeat_button.isChecked():
                self.play_button.setChecked(False)
                return
            time = self.time_of(low)
        self.session.set_time(time)
