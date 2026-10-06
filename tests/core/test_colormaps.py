# Architect: Hans Bihs
"""Colour maps: the rainbow is Turbo, as SEASTATE's maps (without its darkest ends)."""

import numpy as np

from lagoonviz.core.colormaps import lut


def test_rainbow_is_turbo():
    table = lut("rainbow", 256)[:, :3]
    # Turbo's own darkest ends cut off: blue and red, not near black
    assert np.allclose(table[0], np.array([0x42, 0x4B, 0xB5]) / 255, atol=1e-6)
    assert np.allclose(table[-1], np.array([0xB7, 0x1D, 0x02]) / 255, atol=1e-6)
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
    reference = colormaps["turbo"](np.linspace(0.08, 0.92, 256))[:, :3]
    assert np.abs(table - reference).max() < 6 / 255
