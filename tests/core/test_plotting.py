import numpy as np
import pytest

from lagoonviz.core.plotting import nice_step, nice_ticks, tick_label, value_range


def test_nice_ticks():
    assert nice_ticks(0, 10) == [0, 2, 4, 6, 8, 10]
    assert nice_ticks(-0.37, 0.41) == pytest.approx([-0.2, 0.0, 0.2, 0.4])
    assert nice_ticks(0, 174) == [0, 50, 100, 150]
    assert nice_ticks(5, 5) == [5]
    assert nice_step(0.0) == 1.0


def test_tick_labels():
    assert tick_label(0.4, 0.2) == "0.4"
    assert tick_label(150, 50) == "150"
    assert tick_label(0.25, 0.05) == "0.25"
    assert tick_label(2.5e6, 5e5) == "2.5e6"
    assert tick_label(0, 1) == "0"


def test_value_range():
    low, high = value_range([np.array([0.0, 1.0, np.nan]), np.array([-1.0])])
    assert low < -1 and high > 1
    assert value_range([np.array([np.nan])]) == (-1.0, 1.0)
    low, high = value_range([np.array([2.0, 2.0])])
    assert low < 2 < high
