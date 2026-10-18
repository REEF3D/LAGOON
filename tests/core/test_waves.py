"""Wave analysis: zero crossings, reflection, response (RAO)."""

import numpy as np
import pytest

from lagoonviz.core.waves import G, reflection, transfer, wavenumber, zero_crossing


def test_zero_crossing_of_regular_waves():
    t = np.linspace(0.0, 100.0, 20001)
    eta = 0.05 * np.sin(2 * np.pi * t / 2.0) + 0.01  # H 0.1 m, T 2 s, a mean offset
    waves = zero_crossing(t, eta)
    assert waves.n == pytest.approx(49, abs=1)
    assert waves.h13 == pytest.approx(0.1, rel=1e-3) and waves.hmax == pytest.approx(0.1, rel=1e-3)
    assert waves.t13 == pytest.approx(2.0, rel=1e-3) and waves.tmean == pytest.approx(2.0, rel=1e-3)
    window = zero_crossing(t, eta, start=10.0, end=20.0)
    assert window.n in (4, 5)


def test_zero_crossing_of_a_rayleigh_sea():
    """Narrow-band random waves: H1/3 near Hm0 = 4 sqrt(m0)."""
    rng = np.random.default_rng(1)
    t = np.arange(0.0, 3000.0, 0.05)
    f = np.linspace(0.45, 0.55, 60)
    a = np.full(f.size, 0.02)
    eta = sum(
        ai * np.cos(2 * np.pi * fi * t + p)
        for ai, fi, p in zip(a, f, rng.uniform(0, 2 * np.pi, f.size), strict=True)
    )
    waves = zero_crossing(t, eta)
    hm0 = 4 * np.sqrt(np.sum(a**2) / 2)
    assert waves.h13 == pytest.approx(hm0, rel=0.08)


def test_wavenumber():
    k = wavenumber([0.5, 1.0], 0.6)
    omega = 2 * np.pi * np.array([0.5, 1.0])
    np.testing.assert_allclose(omega**2, G * k * np.tanh(k * 0.6), rtol=1e-10)
    assert wavenumber(1.0, 1000.0)[0] == pytest.approx((2 * np.pi) ** 2 / G, rel=1e-6)


def synthetic_flume(kr=0.3, depth=0.6, gauges=(5.0, 5.3, 5.75)):
    """Irregular waves going +x, a part reflected back: eta at three gauges."""
    rng = np.random.default_rng(2)
    t = np.arange(0.0, 400.0, 0.02)
    f = np.linspace(0.4, 1.2, 80)
    a = 0.01 * np.exp(-(((f - 0.7) / 0.2) ** 2))
    k = wavenumber(f, depth)
    phases = rng.uniform(0, 2 * np.pi, f.size)
    back = rng.uniform(0, 2 * np.pi, f.size)
    series = []
    for x in gauges:
        eta = np.zeros_like(t)
        for ai, fi, ki, p, q in zip(a, f, k, phases, back, strict=True):
            w = 2 * np.pi * fi
            eta += ai * np.cos(ki * x - w * t + p) + kr * ai * np.cos(-ki * x - w * t + q)
        series.append(eta)
    hm0 = 4 * np.sqrt(np.sum(a**2) / 2)
    return t, series, list(gauges), depth, hm0


def test_reflection_three_gauges():
    t, series, x, depth, hm0 = synthetic_flume(kr=0.3)
    r = reflection(t, series, x, depth, segments=16)
    assert r.method == "Mansard and Funke"
    assert r.kr == pytest.approx(0.3, abs=0.03)
    assert r.hm0_incident == pytest.approx(hm0, rel=0.06)


def test_reflection_two_gauges_and_full_absorption():
    t, series, x, depth, _hm0 = synthetic_flume(kr=0.0, gauges=(5.0, 5.25))
    r = reflection(t, series, x, depth, segments=16)
    assert r.method == "Goda and Suzuki" and r.kr < 0.05
    with pytest.raises(ValueError):
        reflection(t, series[:1], x[:1], depth)


def test_transfer_of_a_lagging_scaled_response():
    rng = np.random.default_rng(3)
    t = np.arange(0.0, 600.0, 0.05)
    eta = rng.standard_normal(t.size)
    eta = np.convolve(eta, np.ones(10) / 10, mode="same")  # band-limited noise
    lag = 0.25
    heave = 0.8 * np.interp(t - lag, t, eta)
    rao = transfer(t, eta, heave, segments=16)
    band = (rao.frequency > 0.05) & (rao.frequency < 1.0)
    assert np.allclose(rao.gain[band], 0.8, rtol=0.05)
    assert np.all(rao.coherence[band] > 0.95)
    expected = 360 * rao.frequency[band] * lag
    np.testing.assert_allclose(rao.phase[band], expected, atol=6)


def test_the_wave_figures():
    from lagoonviz.core.analysis import spectrum
    from lagoonviz.render.figure import FigureSpec
    from lagoonviz.render.wave_figures import reflection_figure, response_figure, spectra_figure

    t, series, x, depth, _hm0 = synthetic_flume(kr=0.3)
    f, s = spectrum(t, series[0])
    figure = spectra_figure(
        [("P1", f, s), ("P1 measured", f, 0.9 * s)], FigureSpec(), styles=["-", "--"]
    )
    assert len(figure.axes[0].get_lines()) == 2
    r = reflection(t, series, x, depth)
    title = reflection_figure(r).axes[0].get_legend().get_title().get_text()
    assert title.startswith("$K_r$ = 0.") and title.endswith(f"{r.kr:.3f}")
    rao = transfer(t, series[0], 0.5 * series[1])
    assert len(response_figure(rao, unit="m/m").axes) == 2
