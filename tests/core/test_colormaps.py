# Architect: Hans Bihs
"""Colour maps: the rainbow is Turbo, as SEASTATE's maps (without its darkest ends,
yellow and orange wider)."""

import numpy as np

from lagoonviz.core.colormaps import lut


def test_rainbow_is_turbo():
    table = lut("rainbow", 256)[:, :3]
    # Turbo's own darkest ends cut off: blue and red, not near black
    assert np.allclose(table[0], np.array([0x45, 0x59, 0xCB]) / 255, atol=1e-6)
    assert np.allclose(table[-1], np.array([0xC3, 0x25, 0x03]) / 255, atol=1e-6)
    assert table[0].max() > 0.6 and table[-1].max() > 0.6
    # lightness rises to the yellow-green middle and falls again, without bands
    light = table @ np.array([0.2126, 0.7152, 0.0722])
    top = int(np.argmax(light))
    assert 90 < top < 150
    assert np.all(np.diff(light[: top - 5]) > -2e-3) and np.all(np.diff(light[top + 5 :]) < 2e-3)
    assert np.allclose(lut("turbo"), lut("rainbow"))  # the old names lead here
    try:
        from matplotlib import colormaps
    except ImportError:
        return
    knots = np.array([(0, 0.10), (0.38, 0.42), (0.47, 0.56), (0.875, 0.80), (1, 0.90)])
    along = np.interp(np.linspace(0, 1, 256), knots[:, 0], knots[:, 1])
    reference = colormaps["turbo"](along)[:, :3]
    assert np.abs(table - reference).max() < 6 / 255
