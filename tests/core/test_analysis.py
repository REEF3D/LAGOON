"""Wave spectra and comparison with measurements."""

import numpy as np
import pytest

from lagoonviz.core.analysis import (
    compare,
    read_measurements,
    spectrum,
    uniform,
    wave_parameters,
)


def test_a_regular_wave():
    """A sine of amplitude a and period T: Hm0 = 4 sqrt(a²/2), Tp = T, and the
    area under the spectrum is the variance."""
    rng = np.random.default_rng(0)
    time = np.sort(rng.uniform(0, 200, 8000))  # uneven steps, as REEF3D writes them
    eta = 0.05 * np.sin(2 * np.pi * time / 2.0) + 0.3  # a still water offset
    f, s = spectrum(time, eta, segments=8)
    waves = wave_parameters(f, s)
    assert waves.hm0 == pytest.approx(4 * np.sqrt(0.05**2 / 2), rel=0.03)
    assert waves.tp == pytest.approx(2.0, rel=0.03)
    assert waves.tm01 == pytest.approx(2.0, rel=0.05)
    integrate = getattr(np, "trapezoid", None) or np.trapz
    assert integrate(s, f) == pytest.approx(0.05**2 / 2, rel=0.03)


def test_two_components_and_a_window():
    time = np.arange(0, 300, 0.05)
    eta = 0.04 * np.sin(2 * np.pi * time / 4.0) + 0.02 * np.sin(2 * np.pi * time / 1.5)
    f, s = spectrum(time, eta, start=20.0)
    waves = wave_parameters(f, s)
    assert waves.tp == pytest.approx(4.0, rel=0.05)
    m0 = 0.04**2 / 2 + 0.02**2 / 2
    assert waves.hm0 == pytest.approx(4 * np.sqrt(m0), rel=0.05)
    assert waves.tm02 < waves.tm01 < waves.tp


def test_uniform_drops_nans_and_repeats():
    t, v, dt = uniform([0, 1, 1, 2, np.nan, 3, 4], [0, 1, 1, 2, 5, np.nan, 4])
    assert dt == 1 and list(t) == [0, 1, 2, 3, 4] and v[3] == pytest.approx(3.0)
    with pytest.raises(ValueError):
        spectrum([0, 1], [0, 1])


def test_comparison_with_measurements():
    t_lab = np.arange(0, 20, 0.1)
    lab = 0.05 * np.sin(t_lab)
    t_model = np.arange(-1, 15, 0.013)
    same = compare(t_model, 0.05 * np.sin(t_model), t_lab, lab)
    assert same.rmse < 1e-4 and same.skill == pytest.approx(1, abs=1e-4)
    assert same.r == pytest.approx(1, abs=1e-6) and same.end <= 15
    shifted = compare(t_model, 0.05 * np.sin(t_model) + 0.01, t_lab, lab, start=2)
    assert shifted.bias == pytest.approx(0.01, abs=1e-4) and shifted.start >= 2
    assert 0.5 < shifted.skill < 1
    with pytest.raises(ValueError):
        compare(t_model, t_model, t_lab + 100, lab)


def test_reading_measurement_files(tmp_path):
    path = tmp_path / "flume.dat"
    path.write_text("# flume run 12\ntime WG1 WG2\n0 0.1 0.2\n0.5 0.15 0.25\n1.0 0.2 0.3\n")
    time, columns = read_measurements(path)
    assert list(time) == [0, 0.5, 1.0] and list(columns) == ["WG1", "WG2"]
    path.write_text("0,1.5;2\n1,2.5;3\n")
    time, columns = read_measurements(path)
    assert list(columns) == ["1", "2"] and columns["2"][1] == 3
    path.write_text("nothing here\n")
    with pytest.raises(ValueError):
        read_measurements(path)
