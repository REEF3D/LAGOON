# SPDX-License-Identifier: GPL-3.0-or-later
# Architect: Hans Bihs
"""Figures of the wave analysis (matplotlib): spectra with their parameters, incident
and reflected spectra, the response (RAO) of a motion to the waves. They take the
results of lagoonviz.core.analysis and lagoonviz.core.waves, so a script can draw
them as the Wave analysis window does:

    f, s = spectrum(t, eta)
    figure = spectra_figure([("P1", f, s)], FigureSpec())
    save_figure(figure, "spectra.pdf")
"""

from __future__ import annotations

import numpy as np

from lagoonviz.render.figure import MM, FigureSpec, _context


def _figure(spec: FigureSpec, height_mm: float, rows: int = 1):
    from matplotlib.figure import Figure

    height = spec.height_mm or height_mm
    figure = Figure(figsize=(spec.width_mm * MM, height * MM), dpi=spec.dpi, layout="constrained")
    axes = figure.subplots(rows, 1, sharex=True, squeeze=False)[:, 0]
    return figure, axes


def spectra_figure(
    lines: list[tuple[str, np.ndarray, np.ndarray]],
    spec: FigureSpec | None = None,
    high: float | None = None,
    styles: list[str] | None = None,
):
    """Variance density spectra: (label, frequency, density) per line; up to
    ``high`` Hz (default: where the largest falls below a thousandth of its peak)."""
    spec = spec or FigureSpec()
    with _context(spec):
        figure, (ax,) = _figure(spec, 55.0)
        top = 0.0
        for index, (label, f, s) in enumerate(lines):
            style = (styles or [])[index] if styles and index < len(styles) else "-"
            color = spec.palette[index % len(spec.palette)]
            ax.plot(f, s, color=color, linestyle=style, label=label)
            if len(s) and np.nanmax(s) > 0:
                peak = float(np.nanmax(s))
                above = np.flatnonzero(s > 1e-3 * peak)
                top = max(top, float(f[above[-1]]) if above.size else float(f[-1]))
        ax.set_xlim(0, high or (1.1 * top if top > 0 else None))
        ax.set_ylim(bottom=0)
        ax.set_xlabel("f [Hz]")
        ax.set_ylabel("S [m²/Hz]")
        if len(lines) > 1:
            ax.legend(frameon=False)
        if spec.title:
            figure.suptitle(spec.title)
    return figure


def reflection_figure(result, spec: FigureSpec | None = None):
    """Incident and reflected spectra; the frequencies the gauges cannot separate
    shaded."""
    spec = spec or FigureSpec()
    f = result.frequency
    with _context(spec):
        figure, (ax,) = _figure(spec, 55.0)
        ax.plot(f, result.incident, color=spec.palette[0], label="incident")
        ax.plot(f, result.reflected, color=spec.palette[1 % len(spec.palette)], label="reflected")
        invalid = ~result.valid
        if invalid.any():
            top = float(np.nanmax(result.incident)) if result.incident.size else 1.0
            ax.fill_between(f, 0, top, where=invalid, color="0.9", step="mid", linewidth=0)
        peak = float(np.nanmax(result.incident)) if result.incident.size else 0.0
        above = np.flatnonzero(result.incident > 1e-3 * peak) if peak > 0 else np.zeros(0, int)
        if above.size:
            ax.set_xlim(0, 1.1 * float(f[above[-1]]))
        ax.set_ylim(bottom=0)
        ax.set_xlabel("f [Hz]")
        ax.set_ylabel("S [m²/Hz]")
        ax.legend(frameon=False, title=f"$K_r$ = {result.kr:.3f}")
        if spec.title:
            figure.suptitle(spec.title)
    return figure


def response_figure(result, spec: FigureSpec | None = None, unit: str = "", coherence: float = 0.5):
    """The response of a motion to the waves: gain (RAO) and phase over frequency,
    where the coherence is at least ``coherence`` (the rest faint)."""
    spec = spec or FigureSpec()
    f = result.frequency
    good = (result.coherence >= coherence) & (f > 0)
    with _context(spec):
        figure, (gain_ax, phase_ax) = _figure(spec, 75.0, rows=2)
        color = spec.palette[0]
        gain_ax.plot(f, result.gain, color=color, alpha=0.25)
        gain_ax.plot(f, np.where(good, result.gain, np.nan), color=color)
        phase_ax.plot(f, np.where(good, result.phase, np.nan), color=color)
        if good.any():
            band = f[good]
            gain_ax.set_xlim(0, 1.2 * float(band.max()))
            gain_ax.set_ylim(0, 1.15 * float(np.nanmax(result.gain[good])))
        gain_ax.set_ylabel(f"RAO [{unit}]" if unit else "RAO")
        phase_ax.set_ylabel("phase [°]")
        phase_ax.set_xlabel("f [Hz]")
        if spec.title:
            figure.suptitle(spec.title)
    return figure
