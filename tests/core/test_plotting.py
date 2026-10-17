import numpy as np
import pytest

from lagoonviz.core.plotting import (
    nice_step,
    nice_ticks,
    number_text,
    parse_number,
    partial_number,
    range_step,
    tick_label,
    value_range,
)


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


def test_numbers_typed_in_range_boxes():
    """Colour range bounds keep significant digits: k and epsilon are not 0.000."""
    assert number_text(1.154173e-06) == "1.15417e-06"
    assert number_text(4.2154685e-10) == "4.21547e-10"
    assert number_text(0.00015690869) == "0.000156909"
    assert number_text(2.448) == "2.448"
    assert number_text(-17.0) == "-17"
    assert number_text(0.0) == "0"
    assert number_text(1e30) == "1e30"
    for value in (1.154173e-06, 4.2154685e-10, 0.00015690869, 2.448, -17.0):
        assert parse_number(number_text(value)) == pytest.approx(value, rel=1e-5)
    assert parse_number("1,5") == 1.5 and parse_number("x") is None
    assert parse_number("nan") is None
    for text in ("", "-", "1.", "2e", "2e-", "-.5", "3e-07"):
        assert partial_number(text), text
    for text in ("abc", "1..2", "e5x"):
        assert not partial_number(text), text
    assert range_step(0.0, 2.448) == pytest.approx(0.1)
    assert range_step(7.2e-7, 1.15e-6) == pytest.approx(1e-8)
    assert range_step(5.0, 5.0) == pytest.approx(0.1)
    assert range_step(0.0, 0.0) == 0.1
