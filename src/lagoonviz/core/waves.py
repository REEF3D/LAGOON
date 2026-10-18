# SPDX-License-Identifier: GPL-3.0-or-later
# Architect: Hans Bihs
"""Wave analysis of gauge time series (NumPy only, no drawing): wave statistics by
zero crossings, reflection by the method of Mansard and Funke (three gauges; two:
Goda and Suzuki), and the response (RAO) of a floating body's motion to the waves.
The spectra and spectral parameters are lagoonviz.core.analysis's.

    waves = zero_crossing(time, eta, start=10, end=60)      # H1/3, Hmax, T1/3 ...
    r = reflection(time, [eta1, eta2, eta3], [x1, x2, x3], depth=0.6)
    r.kr, r.hm0_incident, r.frequency, r.incident, r.reflected
    rao = transfer(time, eta_gauge, heave)                   # gain, phase, coherence

All take an uneven time step (REEF3D writes gauges at every time step): the series
are put on the median step first (analysis.uniform).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from lagoonviz.core.analysis import uniform

G = 9.81


# ----------------------------------------------------------------- time windows
def _window(time, values, start, end):
    time, values, dt = uniform(time, values)
    keep = np.ones(time.size, bool)
    if start is not None:
        keep &= time >= start
    if end is not None:
        keep &= time <= end
    return time[keep], values[keep], dt


def _together(time, series, start, end):
    """Several series on one even time step over the times they all have."""
    dt = min(float(np.median(np.diff(np.unique(np.asarray(t, float))))) for t in time)
    low = max(float(np.nanmin(t)) for t in time)
    high = min(float(np.nanmax(t)) for t in time)
    if start is not None:
        low = max(low, start)
    if end is not None:
        high = min(high, end)
    if not high > low:
        raise ValueError("the series have no times in common in the window")
    even = np.arange(low, high + 0.5 * dt, dt)
    out = []
    for t, v in zip(time, series, strict=True):
        t = np.asarray(t, float)
        v = np.asarray(v, float)
        ok = np.isfinite(t) & np.isfinite(v)
        order = np.argsort(t[ok])
        out.append(np.interp(even, t[ok][order], v[ok][order]))
    return even, out, dt


# ----------------------------------------------------------------- zero crossings
@dataclass
class ZeroCrossing:
    """Individual waves between zero down-crossings (about the mean)."""

    heights: np.ndarray  # crest to trough of each wave
    periods: np.ndarray
    n: int
    h13: float  # mean of the highest third (significant wave height)
    hmax: float
    hmean: float
    hrms: float
    t13: float  # mean period of the highest third
    tmean: float


def zero_crossing(
    time: np.ndarray,
    eta: np.ndarray,
    start: float | None = None,
    end: float | None = None,
    up: bool = False,
) -> ZeroCrossing:
    """The waves of a surface elevation series between its zero down-crossings
    (``up``: up-crossings), about the mean of the window; crossing times
    interpolated between samples."""
    time, eta, _dt = _window(time, eta, start, end)
    eta = eta - eta.mean()
    sign = eta > 0
    if up:
        edges = np.flatnonzero(~sign[:-1] & sign[1:])
    else:
        edges = np.flatnonzero(sign[:-1] & ~sign[1:])
    if edges.size < 2:
        nan = float("nan")
        return ZeroCrossing(np.zeros(0), np.zeros(0), 0, nan, nan, nan, nan, nan, nan)
    # the crossing time between samples k and k + 1
    a, b = eta[edges], eta[edges + 1]
    crossing = time[edges] + (time[edges + 1] - time[edges]) * a / (a - b)
    heights, periods = [], []
    for k in range(edges.size - 1):
        piece = eta[edges[k] + 1 : edges[k + 1] + 1]
        if piece.size == 0:
            continue
        heights.append(float(piece.max() - piece.min()))
        periods.append(float(crossing[k + 1] - crossing[k]))
    heights = np.asarray(heights)
    periods = np.asarray(periods)
    order = np.argsort(heights)[::-1]
    third = order[: max(1, heights.size // 3)]
    return ZeroCrossing(
        heights=heights,
        periods=periods,
        n=int(heights.size),
        h13=float(heights[third].mean()),
        hmax=float(heights.max()),
        hmean=float(heights.mean()),
        hrms=float(np.sqrt(np.mean(heights**2))),
        t13=float(periods[third].mean()),
        tmean=float(periods.mean()),
    )


# ----------------------------------------------------------------- dispersion
def wavenumber(frequency: np.ndarray | float, depth: float) -> np.ndarray:
    """k of linear waves of ``frequency`` [Hz] in water ``depth`` [m]:
    (2 pi f)² = g k tanh(k h), by Newton's method."""
    f = np.atleast_1d(np.asarray(frequency, np.float64))
    omega = 2.0 * np.pi * f
    k = np.where(f > 0, omega**2 / G, 0.0)
    k = np.maximum(k, omega / np.sqrt(G * max(depth, 1e-9)))  # shallow water start
    for _ in range(50):
        tanh = np.tanh(k * depth)
        fk = G * k * tanh - omega**2
        dfk = G * tanh + G * k * depth * (1.0 - tanh**2)
        step = np.where(dfk > 0, fk / np.where(dfk > 0, dfk, 1.0), 0.0)
        k = np.maximum(k - step, 0.0)
        if np.all(np.abs(step) <= 1e-12 * np.maximum(k, 1e-12)):
            break
    return k


# ----------------------------------------------------------------- reflection
@dataclass
class Reflection:
    """Incident and reflected waves separated from gauges along the wave direction."""

    frequency: np.ndarray
    incident: np.ndarray  # variance density [m²/Hz]
    reflected: np.ndarray
    valid: np.ndarray  # frequencies where the gauge spacing allows the separation
    hm0_incident: float
    hm0_reflected: float
    kr: float  # reflection coefficient: sqrt of the energies' ratio (in the valid band)
    method: str


def _segments(n: int, segments: int) -> tuple[int, list[int]]:
    length = max(int(2 * n / (max(segments, 1) + 1)), 8)
    step = max(length // 2, 1)
    return length, list(range(0, n - length + 1, step))


def reflection(
    time: list[np.ndarray] | np.ndarray,
    series: list[np.ndarray],
    positions: list[float],
    depth: float,
    start: float | None = None,
    end: float | None = None,
    segments: int = 8,
    band: tuple[float, float] | None = None,
) -> Reflection:
    """Incident and reflected spectra from two or three gauges at ``positions``
    along the wave direction (x, increasing with it), in water ``depth``: for every
    frequency the incident and reflected Fourier amplitudes that fit the gauges best
    (Mansard and Funke 1980; with two gauges exactly: Goda and Suzuki 1976), averaged
    over Welch segments. Frequencies where the gauges' spacing is near a multiple of
    half a wave length cannot be separated: left out of the band (``valid``).
    ``band``: the frequencies the energies are summed over (default: those valid
    between 0.05 and 3 times the peak)."""
    if len(series) not in (2, 3) or len(positions) != len(series):
        raise ValueError("reflection needs two or three gauges and their positions")
    times = time if isinstance(time, list | tuple) else [time] * len(series)
    even, values, dt = _together(times, series, start, end)
    x = np.asarray(positions, np.float64) - float(positions[0])
    n = even.size
    length, starts = _segments(n, segments)
    if not starts:
        raise ValueError("too few values in the time window")
    window = np.hanning(length)
    scale = 1.0 / (np.sum(window**2) / dt)
    frequency = np.fft.rfftfreq(length, dt)
    k = wavenumber(frequency, depth)
    phase = np.exp(-1j * np.outer(k, x))  # incident: e^{-ikx} at +omega (numpy's FFT sign)
    incident = np.zeros(frequency.size)
    reflected = np.zeros(frequency.size)
    # the least squares solution per frequency: [e^-ikx, e^+ikx] [a_I, a_R] = X
    matrices = np.stack([phase, np.conj(phase)], axis=2)  # (nf, gauges, 2)
    pinv = np.linalg.pinv(matrices)  # (nf, 2, gauges)
    for first in starts:
        spectra = np.stack(
            [np.fft.rfft((v[first : first + length] - v[first : first + length].mean()) * window)
             for v in values],
            axis=1,
        )  # (nf, gauges)  # fmt: skip
        amplitudes = np.einsum("fij,fj->fi", pinv, spectra)
        incident += np.abs(amplitudes[:, 0]) ** 2
        reflected += np.abs(amplitudes[:, 1]) ** 2
    incident *= scale / len(starts)
    reflected *= scale / len(starts)
    incident[1:-1] *= 2.0
    reflected[1:-1] *= 2.0
    # separable: the gauge pairs' spacings away from multiples of half a wave length
    spacing = np.abs(x[:, None] - x[None, :])[np.triu_indices(len(x), 1)]
    worst = np.max(np.abs(np.sin(np.outer(k, spacing))), axis=1)
    valid = (frequency > 0) & (worst > 0.1)
    if band is None:
        peak = float(frequency[valid][np.argmax(incident[valid])]) if valid.any() else 0.0
        band = (0.05 * peak, 3.0 * peak) if peak > 0 else (0.0, float(frequency[-1]))
    inside = valid & (frequency >= band[0]) & (frequency <= band[1])
    integrate = getattr(np, "trapezoid", None) or np.trapz
    if inside.sum() >= 2:
        m0_i = float(integrate(incident[inside], frequency[inside]))
        m0_r = float(integrate(reflected[inside], frequency[inside]))
    else:
        m0_i = m0_r = float("nan")
    return Reflection(
        frequency=frequency,
        incident=incident,
        reflected=reflected,
        valid=valid,
        hm0_incident=4.0 * math.sqrt(m0_i) if m0_i >= 0 else float("nan"),
        hm0_reflected=4.0 * math.sqrt(m0_r) if m0_r >= 0 else float("nan"),
        kr=math.sqrt(m0_r / m0_i) if m0_i > 0 else float("nan"),
        method="Mansard and Funke" if len(series) == 3 else "Goda and Suzuki",
    )


# ----------------------------------------------------------------- response
@dataclass
class Transfer:
    """How an output (a body's motion) answers an input (the waves), per frequency."""

    frequency: np.ndarray
    gain: np.ndarray  # |output| / |input| (RAO), e.g. m/m or deg/m
    phase: np.ndarray  # output after input [deg]
    coherence: np.ndarray  # 0..1: how much of the output the input explains
    input_density: np.ndarray
    output_density: np.ndarray


def transfer(
    time_in: np.ndarray,
    values_in: np.ndarray,
    values_out: np.ndarray,
    time_out: np.ndarray | None = None,
    start: float | None = None,
    end: float | None = None,
    segments: int = 8,
) -> Transfer:
    """The transfer function from an input series to an output series by Welch's
    cross-spectra: gain |S_xy| / S_xx (the RAO for waves in, motion out), phase and
    coherence |S_xy|² / (S_xx S_yy)."""
    time_out = time_in if time_out is None else time_out
    even, (x, y), dt = _together([time_in, time_out], [values_in, values_out], start, end)
    length, starts = _segments(even.size, segments)
    if not starts:
        raise ValueError("too few values in the time window")
    window = np.hanning(length)
    scale = 1.0 / (np.sum(window**2) / dt)
    sxx = np.zeros(length // 2 + 1)
    syy = np.zeros(length // 2 + 1)
    sxy = np.zeros(length // 2 + 1, complex)
    for first in starts:
        a = np.fft.rfft((x[first : first + length] - x[first : first + length].mean()) * window)
        b = np.fft.rfft((y[first : first + length] - y[first : first + length].mean()) * window)
        sxx += np.abs(a) ** 2
        syy += np.abs(b) ** 2
        sxy += np.conj(a) * b
    sxx, syy, sxy = (s * scale / len(starts) for s in (sxx, syy, sxy))
    with np.errstate(divide="ignore", invalid="ignore"):
        gain = np.where(sxx > 0, np.abs(sxy) / sxx, np.nan)
        coherence = np.where(sxx * syy > 0, np.abs(sxy) ** 2 / (sxx * syy), np.nan)
    phase = -np.degrees(np.angle(sxy))  # positive: the output comes later
    one_sided = np.ones_like(sxx)
    one_sided[1:-1] = 2.0
    return Transfer(
        frequency=np.fft.rfftfreq(length, dt),
        gain=gain,
        phase=phase,
        coherence=coherence,
        input_density=sxx * one_sided,
        output_density=syy * one_sided,
    )
