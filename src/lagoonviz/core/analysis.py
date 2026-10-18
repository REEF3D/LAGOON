# SPDX-License-Identifier: GPL-3.0-or-later
# Architect: Hans Bihs
"""Analysis of time series for figures: wave spectra and comparison with
measurements. NumPy only; no drawing here.

    f, s = spectrum(time, eta)                 # one-sided variance density [m²/Hz]
    waves = wave_parameters(f, s)              # Hm0, Tp, Tm01, Tm02
    stats = compare(t_model, eta_model, t_lab, eta_lab)   # RMSE, bias, r, skill
    t, columns = read_measurements("flume.dat")
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np


# ======================================================================== spectra
def uniform(time: np.ndarray, values: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """The series on an even time step (the median one; REEF3D writes gauges at
    every time step, which varies): (time, values, dt). NaNs are left out first."""
    time = np.asarray(time, np.float64)
    values = np.asarray(values, np.float64)
    keep = np.isfinite(time) & np.isfinite(values)
    time, values = time[keep], values[keep]
    if time.size < 4:
        raise ValueError("too few values for a spectrum")
    order = np.argsort(time, kind="stable")
    time, values = time[order], values[order]
    time, unique = np.unique(time, return_index=True)
    values = values[unique]
    dt = float(np.median(np.diff(time)))
    if not dt > 0:
        raise ValueError("the times do not advance")
    even = np.arange(time[0], time[-1] + 0.5 * dt, dt)
    return even, np.interp(even, time, values), dt


def spectrum(
    time: np.ndarray,
    values: np.ndarray,
    segments: int = 8,
    start: float | None = None,
    end: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """One-sided variance density spectrum by Welch's method: ``segments`` Hann
    windowed pieces overlapping by half, the mean removed from each; between
    ``start`` and ``end`` (default: all). Returns frequency [Hz] and density
    [unit²/Hz], so that the area under it is the variance."""
    time, values, dt = uniform(time, values)
    keep = np.ones(time.size, bool)
    if start is not None:
        keep &= time >= start
    if end is not None:
        keep &= time <= end
    values = values[keep]
    n = values.size
    if n < 8:
        raise ValueError("too few values in the time window for a spectrum")
    length = max(int(2 * n / (max(segments, 1) + 1)), 8)
    step = max(length // 2, 1)
    window = np.hanning(length)
    scale = 1.0 / (np.sum(window**2) / dt)  # density: unit²/Hz
    total = np.zeros(length // 2 + 1)
    count = 0
    for first in range(0, n - length + 1, step):
        piece = values[first : first + length]
        piece = (piece - piece.mean()) * window
        total += np.abs(np.fft.rfft(piece)) ** 2
        count += 1
    density = total / max(count, 1) * scale
    density[1:-1] *= 2.0  # one-sided (DC and Nyquist once)
    if length % 2:
        density[-1] *= 2.0
    frequency = np.fft.rfftfreq(length, dt)
    return frequency, density


@dataclass
class WaveParameters:
    """Spectral wave parameters."""

    hm0: float  # significant wave height 4 sqrt(m0) [m]
    tp: float  # peak period [s]
    tm01: float  # mean period m0/m1 [s]
    tm02: float  # zero-crossing period sqrt(m0/m2) [s]


def wave_parameters(
    frequency: np.ndarray, density: np.ndarray, low: float = 0.0, high: float | None = None
) -> WaveParameters:
    """Hm0, Tp, Tm01 and Tm02 from a spectrum, its moments between ``low`` and
    ``high`` Hz (default: above 0)."""
    f = np.asarray(frequency, np.float64)
    s = np.asarray(density, np.float64)
    band = (f > max(low, 0.0)) & (f <= (high if high is not None else np.inf))
    f, s = f[band], s[band]
    if f.size < 2:
        return WaveParameters(float("nan"), float("nan"), float("nan"), float("nan"))
    integrate = getattr(np, "trapezoid", None) or np.trapz  # NumPy 1 and 2
    m0, m1, m2 = (integrate(s * f**k, f) for k in (0, 1, 2))
    peak = f[int(np.argmax(s))]
    return WaveParameters(
        hm0=float(4.0 * np.sqrt(max(m0, 0.0))),
        tp=float(1.0 / peak) if peak > 0 else float("nan"),
        tm01=float(m0 / m1) if m1 > 0 else float("nan"),
        tm02=float(np.sqrt(m0 / m2)) if m2 > 0 else float("nan"),
    )


# ======================================================================== comparison
@dataclass
class Comparison:
    """How well a computed series matches a measured one, over their common times."""

    rmse: float  # root mean square difference
    bias: float  # mean of computed minus measured
    r: float  # correlation coefficient
    skill: float  # Willmott's index of agreement d (1: perfect)
    n: int  # values compared
    start: float
    end: float
    #: the time shift [s] of the computed series that matches the measured best (by
    #: cross-correlation; positive: the computed one comes later)
    lag: float = float("nan")


def compare(
    t_model: np.ndarray,
    model: np.ndarray,
    t_measured: np.ndarray,
    measured: np.ndarray,
    start: float | None = None,
    end: float | None = None,
) -> Comparison:
    """The computed series at the measured times (linear interpolation), compared
    where both exist (and between ``start`` and ``end``)."""
    t_model = np.asarray(t_model, np.float64)
    model = np.asarray(model, np.float64)
    t_measured = np.asarray(t_measured, np.float64)
    measured = np.asarray(measured, np.float64)
    ok = np.isfinite(t_model) & np.isfinite(model)
    t_model, model = t_model[ok], model[ok]
    order = np.argsort(t_model)
    t_model, model = t_model[order], model[order]
    if t_model.size < 2:
        raise ValueError("no computed values to compare")
    low = max(t_model[0], start if start is not None else -np.inf)
    high = min(t_model[-1], end if end is not None else np.inf)
    keep = np.isfinite(t_measured) & np.isfinite(measured)
    keep &= (t_measured >= low) & (t_measured <= high)
    if keep.sum() < 2:
        raise ValueError("the computed and measured series do not overlap in time")
    obs = measured[keep]
    sim = np.interp(t_measured[keep], t_model, model)
    diff = sim - obs
    mean_obs = obs.mean()
    spread = np.sum((np.abs(sim - mean_obs) + np.abs(obs - mean_obs)) ** 2)
    r = float(np.corrcoef(sim, obs)[0, 1]) if np.std(sim) > 0 and np.std(obs) > 0 else float("nan")
    return Comparison(
        rmse=float(np.sqrt(np.mean(diff**2))),
        bias=float(diff.mean()),
        r=r,
        skill=float(1.0 - np.sum(diff**2) / spread) if spread > 0 else float("nan"),
        n=int(keep.sum()),
        start=float(t_measured[keep][0]),
        end=float(t_measured[keep][-1]),
        lag=float(_lag(t_measured[keep], sim, obs)),
    )


def _lag(time: np.ndarray, sim: np.ndarray, obs: np.ndarray) -> float:
    """The shift of ``sim`` that correlates best with ``obs`` (both at ``time``),
    within a quarter of their span; parabolic refinement between samples."""
    if time.size < 16:
        return float("nan")
    dt = float(np.median(np.diff(time)))
    if not dt > 0:
        return float("nan")
    even = np.arange(time[0], time[-1], dt)
    a = np.interp(even, time, sim)
    b = np.interp(even, time, obs)
    a, b = a - a.mean(), b - b.mean()
    if not (np.any(a) and np.any(b)):
        return float("nan")
    n = even.size
    size = 1 << int(np.ceil(np.log2(2 * n)))
    corr = np.fft.irfft(np.fft.rfft(a, size) * np.conj(np.fft.rfft(b, size)), size)
    reach = max(1, n // 4)
    lags = np.concatenate([corr[: reach + 1], corr[-reach:]])
    shifts = np.concatenate([np.arange(reach + 1), np.arange(-reach, 0)])
    k = int(np.argmax(lags))
    best = float(shifts[k])
    left, right = lags[k - 1] if k > 0 else None, lags[(k + 1) % lags.size]
    if left is not None:
        curve = left - 2 * lags[k] + right
        if curve < 0:
            best += 0.5 * (left - right) / curve
    return best * dt


def match_columns(
    model: list[str], measured: list[str], mapping: str = ""
) -> list[tuple[str, str]]:
    """Which measured column goes with which computed line: (computed, measured).

    ``mapping``: "P1 = 2; P4 = G4" (computed = measured, separated by ";" or ",").
    Without it: the same names (case and spaces aside), then the measured columns
    left over in order for the computed lines left over (a file without a header:
    its first value column for the first line plotted, ...)."""
    pairs = []
    if mapping.strip():
        by_name = {m.strip().lower(): m for m in measured}
        for part in re.split(r"[;,]", mapping):
            if "=" not in part:
                continue
            left, right = (x.strip() for x in part.split("=", 1))
            column = by_name.get(right.lower())
            if left in model and column is not None:
                pairs.append((left, column))
        return pairs
    key = {m.replace(" ", "").lower(): m for m in measured}
    used = set()
    for name in model:
        column = key.get(name.replace(" ", "").lower())
        if column is not None:
            pairs.append((name, column))
            used.add(column)
    rest_model = [n for n in model if n not in {p[0] for p in pairs}]
    rest_measured = [m for m in measured if m not in used]
    if len(pairs) < len(model):
        pairs += list(zip(rest_model, rest_measured, strict=False))
    order = {name: k for k, name in enumerate(model)}
    return sorted(pairs, key=lambda p: order[p[0]])


_NUMBER = re.compile(r"^[-+]?(\d+\.?\d*|\.\d+)([eE][-+]?\d+)?$")


def read_measurements(path: str | Path) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """A measured time series file: the first column the time, the others values;
    separated by spaces, tabs, commas or semicolons; lines starting with ``#`` or
    ``%`` and a header line of names are allowed. Returns (time, {name: values});
    without a header the columns are called "1", "2", ..."""
    names: list[str] | None = None
    rows: list[list[float]] = []
    for line in Path(path).read_text(errors="replace").splitlines():
        text = line.strip()
        if not text or text[0] in "#%":
            continue
        fields = [f for f in re.split(r"[\s,;]+", text) if f]
        if all(_NUMBER.match(f) for f in fields):
            rows.append([float(f) for f in fields])
        elif not rows and names is None:
            names = fields
    if not rows:
        raise ValueError(f"{Path(path).name}: no numbers found")
    width = min(len(r) for r in rows)
    if width < 2:
        raise ValueError(f"{Path(path).name}: needs a time column and a value column")
    table = np.array([r[:width] for r in rows])
    if names is None or len(names) < width:
        names = ["time", *(str(i) for i in range(1, width))]
    return table[:, 0], {names[i]: table[:, i] for i in range(1, width)}
