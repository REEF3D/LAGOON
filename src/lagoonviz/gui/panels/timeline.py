# SPDX-License-Identifier: GPL-3.0-or-later
"""Timeline: play and scrub through time. Keyframes for camera paths follow in phase 5."""

from __future__ import annotations

from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import (
    QDoubleSpinBox,
    QHBoxLayout,
    QLabel,
    QSlider,
    QToolButton,
    QWidget,
)

from lagoonviz.gui.bridge import SessionBridge

STEPS = 1000
FRAME_MS = 40


class TimelinePanel(QWidget):
    def __init__(self, bridge: SessionBridge, parent: QWidget | None = None):
        super().__init__(parent)
        self.setObjectName("timeline")
        self.session = bridge.session

        self.play_button = QToolButton()
        self.play_button.setText("Play")
        self.play_button.setCheckable(True)
        self.play_button.toggled.connect(self._toggle_play)

        self.slider = QSlider(Qt.Horizontal)
        self.slider.setRange(0, STEPS)
        self.slider.valueChanged.connect(self._on_slider)

        self.end = QDoubleSpinBox()
        self.end.setRange(0.1, 1e7)
        self.end.setValue(60.0)
        self.end.setSuffix(" s")
        self.end.setToolTip("End time (taken from the loaded case from phase 1 on)")
        self.end.valueChanged.connect(lambda _: self._show_time(self.session.scene.time))

        self.label = QLabel()
        self.label.setMinimumWidth(90)

        layout = QHBoxLayout(self)
        layout.addWidget(self.play_button)
        layout.addWidget(self.slider, 1)
        layout.addWidget(self.label)
        layout.addWidget(QLabel("End"))
        layout.addWidget(self.end)

        self.timer = QTimer(self)
        self.timer.setInterval(FRAME_MS)
        self.timer.timeout.connect(self._advance)
        bridge.timeChanged.connect(self._show_time)
        self._show_time(self.session.scene.time)

    @property
    def end_time(self) -> float:
        return self.end.value()

    def _on_slider(self, value: int) -> None:
        self.session.set_time(value / STEPS * self.end_time)

    def _show_time(self, time: float) -> None:
        self.label.setText(f"t = {time:.2f} s")
        self.slider.blockSignals(True)
        self.slider.setValue(round(min(time / self.end_time, 1.0) * STEPS))
        self.slider.blockSignals(False)

    def _toggle_play(self, playing: bool) -> None:
        self.play_button.setText("Pause" if playing else "Play")
        if playing:
            self.timer.start()
        else:
            self.timer.stop()

    def _advance(self) -> None:
        step = self.end_time / STEPS * 2
        time = self.session.scene.time + step
        if time > self.end_time:
            time = 0.0
        self.session.set_time(time)
