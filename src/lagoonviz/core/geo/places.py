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
    #: OpenStreetMap's importance of the place (0 - 1), 0 for others
    importance: float = 0.0

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


def search_kartverket(
    text: str, fetch: Fetch = _http, limit: int = 15, fuzzy: bool = False
) -> list[Place]:
    query = urllib.parse.urlencode(
        {
            "sok": text,
            "fuzzy": "true" if fuzzy else "false",
            "utkoordsys": "4258",
            "treffPerSide": limit,
        }
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


def _kartverket(name: str, fetch: Fetch) -> list[Place]:
    """Kartverket's names: as written, else beginning so ("Munkh" finds Munkholmen),
    else alike (fuzzy: a misspelling)."""
    for text, fuzzy in ((name, False), (name.rstrip("*") + "*", False), (name, True)):
        found = search_kartverket(text, fetch, fuzzy=fuzzy)
        if found:
            return found
    return []


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
        try:
            importance = float(entry.get("importance", 0.0) or 0.0)
        except (TypeError, ValueError):
            importance = 0.0
        found.append(
            Place(name, lat, lon, str(entry.get("type", "")), where, "OpenStreetMap", importance)
        )
    return found


#: kinds of places a search most likely means (Kartverket's, OpenStreetMap's)
_MAIN_KINDS = {
    "by": 30,
    "tettsted": 28,
    "city": 30,
    "town": 28,
    "kommune": 25,
    "administrative": 22,
    "bydel": 20,
    "village": 18,
    "suburb": 16,
    "havn": 15,
    "harbour": 15,
    "fjord": 15,
    "øy": 14,
    "island": 14,
    "elv": 12,
    "river": 12,
    "innsjø": 12,
    "lake": 12,
    "bukt": 10,
    "bay": 10,
    "nes": 8,
    "fjell": 8,
    "peak": 8,
}


def _score(place: Place, name: str, qualifier: str, near) -> float:
    """How likely a place is the one meant: the name as typed, the qualifier (the
    municipality, "Nidelva, Trondheim"), its kind, close to where the case is."""
    import math

    wanted = name.casefold()
    have = place.name.casefold()
    score = 100.0 if have == wanted else 40.0 if have.startswith(wanted) else 0.0
    if qualifier:
        q = qualifier.casefold()
        score += 80.0 if q in place.where.casefold() or q in have else -60.0
    score += _MAIN_KINDS.get(place.kind.casefold(), 0)
    score += 30.0 * place.importance
    if near is not None:
        lat0, lon0 = near
        dy = (place.lat - lat0) * 111.0
        dx = (place.lon - lon0) * 111.0 * math.cos(math.radians(lat0))
        score -= min(math.hypot(dx, dy) / 25.0, 40.0)  # nearby first: 1 point per 25 km
    return score


def search(text: str, fetch: Fetch = _http, near=None) -> list[Place]:
    """Places for a search, the likeliest first: the coordinates typed, else names
    from Kartverket (Norway) and OpenStreetMap. "Nidelva, Trondheim" asks for the
    Nidelva in Trondheim; ``near`` (lat, lon), e.g. where the case is, puts the
    places close to it first. Raises OSError when neither service answers."""
    text = text.strip()
    if not text:
        return []
    typed = parse_coordinates(text)
    if typed is not None:
        return [typed]
    name, _, qualifier = (part.strip() for part in text.partition(","))
    found: list[Place] = []
    errors = []
    for label, service in (
        ("Kartverket", lambda: _kartverket(name, fetch)),
        ("OpenStreetMap", lambda: search_nominatim(text, fetch)),
    ):
        try:
            found += service()
        except (OSError, ValueError) as error:
            errors.append(f"{label}: {error}")
    if not found and errors:
        raise OSError("place search: " + "; ".join(errors))
    found.sort(key=lambda p: -_score(p, name, qualifier, near))
    # the same place from both services once (within ~100 m)
    unique: list[Place] = []
    for place in found:
        if not any(abs(place.lat - p.lat) < 1e-3 and abs(place.lon - p.lon) < 2e-3 for p in unique):
            unique.append(place)
    return unique
