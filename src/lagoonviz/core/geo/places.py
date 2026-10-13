# SPDX-License-Identifier: GPL-3.0-or-later
# Architect: Hans Bihs
"""Finding a place: "63.44, 10.40", or a name ("Nidelva", "Trondheim havn").

Coordinates are read as latitude, longitude in degrees (also with N/S/E/W, or
"lon=…, lat=…"). Names are looked up in Kartverket's open place-name service
(sentralt stedsnavnregister, Norway) and in OpenStreetMap's Nominatim (the world);
one request per search, as their terms ask (no search while typing), with
LAGOON's name.
"""

from __future__ import annotations

import json
import os
import re
import urllib.parse
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass

KARTVERKET = "https://ws.geonorge.no/stedsnavn/v1/navn?{query}"
NOMINATIM = "https://nominatim.openstreetmap.org/search?{query}"


@dataclass(frozen=True)
class Place:
    name: str
    lat: float
    lon: float
    #: what it is ("Elv", "river", "coordinates"), where ("Trondheim"), whose list
    kind: str = ""
    where: str = ""
    source: str = ""

    def label(self) -> str:
        parts = [self.name]
        details = ", ".join(p for p in (self.kind, self.where) if p)
        if details:
            parts.append(f"({details})")
        return " ".join(parts) + f"  ·  {self.lat:.4f}, {self.lon:.4f}"


_NUMBER = r"[-+]?\d+(?:\.\d+)?"


def parse_coordinates(text: str) -> Place | None:
    """Latitude and longitude in degrees from text ("63.44, 10.40", "63.44N 10.4E",
    "10.4E 63.44N", "lat 63.44 lon 10.40", "63,44 10,40"), or None."""
    text = text.strip().replace("°", " ")
    if re.fullmatch(r"[-+]?\d+,\d+\s+[-+]?\d+,\d+", text):  # decimal commas
        text = text.replace(",", ".")
    named = {
        key.lower()[:3]: float(value)
        for key, value in re.findall(
            rf"\b(latitude|lat|longitude|long|lon|lng)\s*[=:]?\s*({_NUMBER})", text, re.I
        )
    }
    if "lat" in named and ("lon" in named or "lng" in named):
        lat, lon = named["lat"], named.get("lon", named.get("lng"))
    else:
        pattern = rf"({_NUMBER})\s*([NSEWnsew](?![A-Za-z]))?"
        found = re.findall(pattern, text)
        rest = re.sub(pattern, "", text)
        if len(found) != 2 or re.search(r"[^\s,;]", rest):
            return None
        values = []
        for number, hemisphere in found:
            value = float(number)
            hemisphere = hemisphere.upper()
            if hemisphere in ("S", "W"):
                value = -abs(value)
            values.append((value, hemisphere))
        (a, ha), (b, hb) = values
        lat, lon = (b, a) if ha in ("E", "W") or hb in ("N", "S") else (a, b)
    if not (-90 <= lat <= 90 and -180 <= lon <= 180):
        return None
    return Place(f"{lat:.5f}, {lon:.5f}", lat, lon, "coordinates", "", "")


#: Fetch signature: ``fetch(url) -> bytes``
Fetch = Callable[[str], bytes]


def _http(url: str, timeout: float = 15.0) -> bytes:
    if os.environ.get("LAGOON_OFFLINE"):
        raise OSError("offline (LAGOON_OFFLINE)")
    from lagoonviz import __version__

    request = urllib.request.Request(
        url,
        headers={
            "User-Agent": f"LAGOON/{__version__} (REEF3D post-processor; "
            "+https://github.com/REEF3D/LAGOON)",
            "Accept-Language": "nb,en",
        },
    )
    with urllib.request.urlopen(request, timeout=timeout) as answer:  # noqa: S310
        return answer.read()


def search_kartverket(text: str, fetch: Fetch = _http, limit: int = 10) -> list[Place]:
    query = urllib.parse.urlencode(
        {"sok": text, "fuzzy": "true", "utkoordsys": "4258", "treffPerSide": limit}
    )
    data = json.loads(fetch(KARTVERKET.format(query=query)).decode("utf-8"))
    found = []
    for entry in data.get("navn", []):
        point = entry.get("representasjonspunkt") or {}
        try:
            lon, lat = float(point["øst"]), float(point["nord"])
        except (KeyError, TypeError, ValueError):
            continue
        towns = ", ".join(k.get("kommunenavn", "") for k in entry.get("kommuner") or [])
        found.append(
            Place(
                str(entry.get("skrivemåte", text)),
                lat,
                lon,
                str(entry.get("navneobjekttype", "")),
                towns,
                "Kartverket",
            )
        )
    return found


def search_nominatim(text: str, fetch: Fetch = _http, limit: int = 10) -> list[Place]:
    query = urllib.parse.urlencode({"q": text, "format": "jsonv2", "limit": limit})
    data = json.loads(fetch(NOMINATIM.format(query=query)).decode("utf-8"))
    found = []
    for entry in data if isinstance(data, list) else []:
        try:
            lat, lon = float(entry["lat"]), float(entry["lon"])
        except (KeyError, TypeError, ValueError):
            continue
        name = str(entry.get("name") or entry.get("display_name", text).split(",")[0])
        where = ", ".join(str(entry.get("display_name", "")).split(", ")[1:3])
        found.append(Place(name, lat, lon, str(entry.get("type", "")), where, "OpenStreetMap"))
    return found


def search(text: str, fetch: Fetch = _http) -> list[Place]:
    """Places for a search: the coordinates typed, else names from Kartverket
    (Norway) and OpenStreetMap, Kartverket's first. Raises OSError when neither
    service answers."""
    text = text.strip()
    if not text:
        return []
    typed = parse_coordinates(text)
    if typed is not None:
        return [typed]
    found: list[Place] = []
    errors = []
    for service in (search_kartverket, search_nominatim):
        try:
            found += service(text, fetch)
        except (OSError, ValueError) as error:
            errors.append(f"{service.__name__.split('_')[1]}: {error}")
    if not found and errors:
        raise OSError("place search: " + "; ".join(errors))
    # the same place from both services once (within ~100 m)
    unique: list[Place] = []
    for place in found:
        if not any(abs(place.lat - p.lat) < 1e-3 and abs(place.lon - p.lon) < 2e-3 for p in unique):
            unique.append(place)
    return unique
