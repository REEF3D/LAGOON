# SPDX-License-Identifier: GPL-3.0-or-later
# Architect: Hans Bihs
"""Ask the real services of the Sea tab once and say what each answers: Open-Meteo
(ERA5 waves, sea level and currents, GloFAS), Kartverket, NOAA CO-OPS and IOC.

    python tools/check_sea_services.py            # Trondheim and New York
    python tools/check_sea_services.py 63.44 10.4 # a place of your own (lat lon)

The tests use stand-ins for these services; this checks that the services still
answer as LAGOON reads them (a variable renamed, a format changed).
"""

from __future__ import annotations

import datetime as dt
import sys
import traceback

import numpy as np

from lagoonviz.core.geo import gauges, openmeteo


def check(label, work) -> None:
    try:
        print(f"OK    {label}: {work()}")
    except Exception as error:  # noqa: BLE001 (a report)
        print(f"FAIL  {label}: {type(error).__name__}: {error}")
        traceback.print_exc(limit=2, file=sys.stdout)


def place(lat: float, lon: float) -> None:
    print(f"\n--- {lat:.3f} N {lon:.3f} E")
    meteo = openmeteo.OpenMeteo()
    end = dt.datetime.now(dt.UTC) - dt.timedelta(days=10)
    start = end - dt.timedelta(days=2)

    def waves():
        s = meteo.waves(lat, lon, start, end)
        return (
            f"{len(s.time)} hours at {s.point.lat:.2f} N {s.point.lon:.2f} E ({s.source}); "
            f"Hs {np.nanmax(s.columns['Hs']):.2f} m, Tp {np.nanmean(s.columns['Tp']):.1f} s, "
            f"wind {np.nanmean(s.columns['wind']):.1f} m/s"
        )

    def sea():
        s = meteo.sea(lat, lon, start, end)
        return (
            f"{len(s.time)} hours; zeta {np.nanmin(s.columns['zeta']):.2f} to "
            f"{np.nanmax(s.columns['zeta']):.2f} m, "
            f"current {np.nanmean(s.columns['current']):.2f} m/s"
        )

    def river():
        s = meteo.discharge(lat, lon, start - dt.timedelta(days=30), end)
        q = np.nanmean(s.columns["Q"])
        return f"{len(s.time)} days at {s.point.lat:.3f} N {s.point.lon:.3f} E; Q {q:.1f} m3/s"

    check("Open-Meteo ERA5 waves and wind", waves)
    check("Open-Meteo sea level and currents", sea)
    check("Open-Meteo GloFAS discharge", river)
    for name in gauges.SOURCES:

        def gauge(name=name):
            found = gauges.nearest(lat, lon, (name,), count=1, radius_km=500.0)
            if not found:
                return "no gauge within 500 km"
            s = gauges.levels(found[0], start, start + dt.timedelta(hours=12))
            cols = ", ".join(f"{k} {np.nanmean(v):+.2f} m" for k, v in s.columns.items())
            return f"{found[0].describe()}: {len(s.time)} values; {cols}"

        check(f"{name} tide gauge", gauge)


def main(argv: list[str]) -> None:
    if len(argv) >= 2:
        place(float(argv[0]), float(argv[1]))
    else:
        place(63.44, 10.38)  # Trondheim
        place(40.45, -73.85)  # New York


if __name__ == "__main__":
    main(sys.argv[1:])
