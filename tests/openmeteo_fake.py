"""Stand-ins for Open-Meteo's APIs and the tide gauge services: answers made from the
request's parameters, as the real services lay them out."""

from __future__ import annotations

import datetime as dt
import json
import math
import urllib.parse

import numpy as np

STORM = dt.datetime(2023, 3, 10, 6, tzinfo=dt.UTC)


def _hours(params) -> np.ndarray:
    a = dt.datetime.fromisoformat(params["start_date"]).replace(tzinfo=dt.UTC)
    b = dt.datetime.fromisoformat(params["end_date"]).replace(tzinfo=dt.UTC)
    return np.arange(a.timestamp(), b.timestamp() + 86400.0, 3600.0)


def wave_height(t) -> np.ndarray:
    """1.5 m, a storm of 6 m at STORM."""
    t = np.asarray(t, float)
    return 1.5 + 4.5 * np.exp(-(((t - STORM.timestamp()) / (12 * 3600.0)) ** 2))


def tide(t) -> np.ndarray:
    return 0.7 * np.sin(2 * math.pi * (np.asarray(t, float) / 3600.0) / 12.42)


class FakeOpenMeteo:
    """marine (ERA5 waves; sea level and currents), archive (wind), flood (GloFAS)."""

    def __init__(self, peak_period: bool = True, land_from: float | None = None) -> None:
        self.peak_period = peak_period
        #: places west of this longitude are on land for the sea level (None: none)
        self.land_from = land_from
        self.asked: list[str] = []

    def __call__(self, address: str) -> bytes:
        self.asked.append(address)
        parts = urllib.parse.urlsplit(address)
        params = dict(urllib.parse.parse_qsl(parts.query))
        api = parts.netloc.split(".")[0].replace("customer-", "").replace("-api", "")
        lats = [float(v) for v in params["latitude"].split(",")]
        lons = [float(v) for v in params["longitude"].split(",")]
        answers = [self._one(api, params, la, lo) for la, lo in zip(lats, lons, strict=True)]
        if any("error" in a for a in answers):
            return json.dumps(answers[0]).encode()
        return json.dumps(answers if len(answers) > 1 else answers[0]).encode()

    def _one(self, api, params, lat, lon) -> dict:
        base = {"latitude": round(lat, 2), "longitude": round(lon, 2), "utc_offset_seconds": 0}
        if api == "flood":
            days = np.arange(
                dt.datetime.fromisoformat(params["start_date"]).replace(tzinfo=dt.UTC).timestamp(),
                dt.datetime.fromisoformat(params["end_date"]).replace(tzinfo=dt.UTC).timestamp()
                + 1.0,
                86400.0,
            )
            q = 120.0 + 10.0 * (days - days[0]) / 86400.0  # rising 10 m3/s a day
            return {**base, "daily": {"time": days.tolist(), "river_discharge": q.tolist()},
                    "daily_units": {"river_discharge": "m³/s"}}  # fmt: skip
        hourly_names = params["hourly"].split(",")
        t = _hours(params)
        hourly: dict = {"time": t.tolist()}
        if api == "archive" and params.get("models") == "cerra":  # 12 m/s to mid 2021
            end = dt.datetime(2021, 7, 1, tzinfo=dt.UTC).timestamp()
            hourly["wind_speed_10m"] = [12.0 if s < end else None for s in t]
            hourly["wind_direction_10m"] = [260.0 if s < end else None for s in t]
            return {**base, "hourly": hourly}
        if api == "archive":
            hourly["wind_speed_10m"] = (10.0 + 0 * t).tolist()
            hourly["wind_direction_10m"] = (250.0 + 0 * t).tolist()
            return {**base, "hourly": hourly}
        if "wave_peak_period" in hourly_names and not self.peak_period:
            return {"error": True, "reason": "Cannot initialize SurfaceVariable from invalid "
                    "String value wave_peak_period"}  # fmt: skip
        units = {}
        for name in hourly_names:
            if name == "wave_height":
                hourly[name] = wave_height(t).tolist()
            elif name in ("wave_peak_period", "wave_period"):
                hourly[name] = (8.0 + wave_height(t)).tolist()
            elif name == "wave_direction":
                hourly[name] = (300.0 + 0 * t).tolist()  # from the north-west
            elif name == "sea_level_height_msl":
                land = self.land_from is not None and lon < self.land_from
                hourly[name] = [None] * len(t) if land else tide(t).tolist()
            elif name == "ocean_current_velocity":
                hourly[name] = (3.6 + 0 * t).tolist()  # 1 m/s, in km/h
                units[name] = "km/h"
            elif name == "ocean_current_direction":
                hourly[name] = (90.0 + 0 * t).tolist()  # going east
        return {**base, "hourly": hourly, "hourly_units": units}


class FakeGauges:
    """Kartverket (XML), NOAA CO-OPS and IOC (JSON): one station each near the case."""

    def __init__(self, lat: float, lon: float) -> None:
        self.lat, self.lon = lat, lon
        self.asked: list[str] = []

    def __call__(self, address: str) -> bytes:
        self.asked.append(address)
        parts = urllib.parse.urlsplit(address)
        params = dict(urllib.parse.parse_qsl(parts.query))
        if "kartverket" in parts.netloc:
            return self._kartverket(params).encode()
        if "mdapi" in parts.path:
            stations = [
                {"id": "8518750", "name": "The Battery", "lat": self.lat + 0.05, "lng": self.lon},
                {"id": "9414290", "name": "San Francisco", "lat": 37.8, "lng": -122.5},
            ]
            return json.dumps({"count": 2, "stations": stations}).encode()
        if "tidesandcurrents" in parts.netloc:
            a = dt.datetime.strptime(params["begin_date"], "%Y%m%d %H:%M").replace(tzinfo=dt.UTC)
            b = dt.datetime.strptime(params["end_date"], "%Y%m%d %H:%M").replace(tzinfo=dt.UTC)
            t = np.arange(a.timestamp(), b.timestamp() + 1.0, 360.0)
            surge = 0.2 if params["product"] == "water_level" else 0.0
            rows = [
                {"t": dt.datetime.fromtimestamp(s, dt.UTC).strftime("%Y-%m-%d %H:%M"),
                 "v": f"{tide(s) + surge:.3f}"}
                for s in t
            ]  # fmt: skip
            key = "data" if params["product"] == "water_level" else "predictions"
            return json.dumps({key: rows}).encode()
        if "ioc-sealevelmonitoring" in parts.netloc:
            if params["query"] == "stationlist":
                return json.dumps(
                    [
                        {
                            "Code": "batt",
                            "Location": "Battery",
                            "Lat": self.lat - 0.1,
                            "Lon": self.lon,
                            "countryname": "USA",
                        }
                    ]  # fmt: skip
                ).encode()
            a = dt.datetime.fromisoformat(params["timestart"]).replace(tzinfo=dt.UTC)
            b = dt.datetime.fromisoformat(params["timestop"]).replace(tzinfo=dt.UTC)
            t = np.arange(a.timestamp(), b.timestamp() + 1.0, 60.0)
            rows = [
                {"slevel": float(3.0 + tide(s)), "stime": dt.datetime.fromtimestamp(s, dt.UTC)
                 .strftime("%Y-%m-%d %H:%M:%S"), "sensor": "rad"}
                for s in t
            ] + [{"slevel": -999, "stime": f"{a:%Y-%m-%d %H:%M:%S}", "sensor": "prs"}]  # fmt: skip
            return json.dumps(rows).encode()
        raise OSError(f"404: {address}")

    def _kartverket(self, params) -> str:
        if params["tide_request"] == "stationlist":
            return (
                '<?xml version="1.0"?><tide><stationinfo>'
                f'<station name="Oslo" code="OSL" latitude="{self.lat + 0.02}" '
                f'longitude="{self.lon}" type="PERM"/>'
                "</stationinfo></tide>"
            )
        a = dt.datetime.fromisoformat(params["fromtime"]).replace(tzinfo=dt.UTC)
        b = dt.datetime.fromisoformat(params["totime"]).replace(tzinfo=dt.UTC)
        t = np.arange(a.timestamp(), b.timestamp() + 1.0, 600.0)

        def rows(flag, add):
            return "".join(
                f'<waterlevel value="{100 * (tide(s) + add):.1f}" '
                f'time="{dt.datetime.fromtimestamp(s, dt.UTC):%Y-%m-%dT%H:%M:%S}+00:00" '
                f'flag="{flag}"/>'
                for s in t
            )

        return (
            '<?xml version="1.0"?><tide><stationdata><location name="Oslo" code="OSL">'
            f'<data type="observation" unit="cm" reflevelcode="MSL">{rows("obs", 0.1)}</data>'
            f'<data type="prediction" unit="cm" reflevelcode="MSL">{rows("pre", 0.0)}</data>'
            "</location></stationdata></tide>"
        )
