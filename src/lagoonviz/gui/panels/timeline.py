# SPDX-License-Identifier: GPL-3.0-or-later
"""Timeline: play, step and scrub through the frames of the loaded cases.

With a case loaded the slider moves frame by frame; the time is shown in seconds
when the output interval is known (``ctrl.txt`` or the case's property), else as
the frame number. Playback waits for each frame to be loaded before moving on and
asks the loader to read the next frames ahead. Keyframes follow in phase 5.
"""

from __future__ import annotations

import time
from collections.abc import Callable

from PySide6.QtCore import QSize, Qt, QTimer
from PySide6.QtWidgets import (
    QDoubleSpinBox,
    QHBoxLayout,
    QLabel,
    QSlider,
    QSpinBox,
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


class TimelinePanel(QWidget):
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
            button.clicked.connect(slot)
            return button

        self.first_button = tool("first", "First frame", lambda: self.go_to_frame(0))
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
        self.last_button = tool("last", "Last frame", lambda: self.go_to_frame(self.n_frames - 1))

        self.slider = QSlider(Qt.Horizontal)
        self.slider.setRange(0, STEPS)
        self.slider.valueChanged.connect(self._on_slider)

        self.label = QLabel()
        self.label.setMinimumWidth(150)

        self.fps = QSpinBox()
        self.fps.setRange(1, 60)
        self.fps.setValue(DEFAULT_FPS)
        self.fps.setSuffix(" fps")
        self.fps.setToolTip("Playback speed in frames per second")

        self.end_label = QLabel("End")
        self.end = QDoubleSpinBox()
        self.end.setRange(0.1, 1e7)
        self.end.setValue(60.0)
        self.end.setSuffix(" s")
        self.end.setToolTip("End time when no case is loaded")
        self.end.valueChanged.connect(lambda _: self._show_time(self.session.scene.time))

        layout = QHBoxLayout(self)
        for widget in (self.first_button, self.prev_button, self.play_button, self.next_button):
            layout.addWidget(widget)
        layout.addWidget(self.last_button)
        layout.addWidget(self.slider, 1)
        layout.addWidget(self.label)
        layout.addWidget(self.fps)
        layout.addWidget(self.end_label)
        layout.addWidget(self.end)

        # playback: one single-shot timer, scheduled after each step. Waiting for a
        # frame re-checks every few milliseconds, so playing goes on as soon as the
        # data is there instead of a whole frame interval later.
        self.timer = QTimer(self)
        self.timer.setSingleShot(True)
        self.timer.timeout.connect(self._tick)
        self._last_step = 0.0
        bridge.timeChanged.connect(self._show_time)
        bridge.sceneChanged.connect(lambda *_: self.update_range())
        self.update_range()

    # ------------------------------------------------------------- range
    @property
    def has_frames(self) -> bool:
        return self.n_frames > 0

    @property
    def end_time(self) -> float:
        if self.has_frames:
            return max(self.n_frames - 1, 0) * self.dt
        return self.end.value()

    def update_range(self) -> None:
        """Follow the frames of the loaded cases."""
        n = self.session.data.n_frames()
        dt = self.session.data.print_interval()
        self.n_frames = n
        self.first_frame = min(self.session.data.first_output(), max(n - 1, 0))
        self.dt = dt or 1.0
        self.unit = "s" if dt else "frame"
        self.end.setVisible(not n)
        self.end_label.setVisible(not n)
        for button in (self.first_button, self.prev_button, self.next_button, self.last_button):
            button.setEnabled(n > 1)
        self.slider.blockSignals(True)
        self.slider.setRange(self.first_frame, max(n - 1, 0) if n else STEPS)
        self.slider.setPageStep(max(1, n // 20) if n else 50)
        self.slider.blockSignals(False)
        self._show_time(self.session.scene.time)

    # ------------------------------------------------------------- time
    def frame_of(self, time: float) -> int:
        return int(min(max(round(time / self.dt), self.first_frame), max(self.n_frames - 1, 0)))

    def go_to_frame(self, frame: int) -> None:
        if self.has_frames:
            frame = min(max(frame, self.first_frame), self.n_frames - 1)
            self.session.set_time(frame * self.dt)

    def step(self, frames: int) -> None:
        self.go_to_frame(self.frame_of(self.session.scene.time) + frames)

    def _on_slider(self, value: int) -> None:
        if self.has_frames:
            self.session.set_time(value * self.dt)
        else:
            self.session.set_time(value / STEPS * self.end_time)

    def _show_time(self, time: float) -> None:
        self.slider.blockSignals(True)
        if self.has_frames:
            frame = self.frame_of(time)
            self.slider.setValue(frame)
            where = f"frame {frame} / {self.n_frames - 1}"
            self.label.setText(f"t = {time:.2f} s  ({where})" if self.unit == "s" else where)
        else:
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
        ):
            button.setIcon(icon(name))
        playing = self.play_button.isChecked()
        self.play_button.setIcon(icon("pause" if playing else "play", COLORS["cursor"]))

    # ------------------------------------------------------------- playback
    def _toggle_play(self, playing: bool) -> None:
        self.play_button.setText("Pause" if playing else "Play")
        self.play_button.setIcon(icon("pause" if playing else "play", COLORS["cursor"]))
        if playing:
            self._last_step = 0.0
            self.timer.start(0)
        else:
            self.timer.stop()

    def _tick(self) -> None:
        if not self.play_button.isChecked():
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
        if self.busy():
            return  # wait until the current frame is on screen
        PERF.count("frames advanced")
        if self.has_frames:
            frame = self.frame_of(self.session.scene.time) + 1
            if frame >= self.n_frames:
                frame = self.first_frame
            self.go_to_frame(frame)
            if self.loader is not None:
                self.loader.prefetch(
                    self.session.data.data_items(), self.session.scene.time, frames_ahead=8
                )
            return
        step = self.end_time / STEPS * 2
        time = self.session.scene.time + step
        if time > self.end_time:
            time = 0.0
        self.session.set_time(time)
