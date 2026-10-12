# Maps and the place of a case

REEF3D computes in model coordinates. A **georeference** says where they lie on
the Earth; with it LAGOON lays a map or a satellite image under the results and
the set-up, the domain can be drawn on the map, and a bathymetry survey becomes
DIVEMesh's `geo.dat`.

## The georeference

A case keeps it in `georef.json` in its folder:

```json
{"version": 1, "crs": "EPSG:25833", "origin": [270000.0, 7040000.0],
 "rotation_deg": 30.0, "vertical_datum": "EPSG:5941", "z_offset": 0.0}
```

- `crs`: the coordinate reference system of the map, an EPSG code (`EPSG:25833`
  is ETRS89 / UTM zone 33N) or WKT;
- `origin`: where the model's (0, 0) lies in it;
- `rotation_deg`: the angle from east to the model's x-axis, counter-clockwise
  (the panel also shows it as a compass bearing);
- `vertical_datum` and `z_offset`: the datum of the heights (`EPSG:5941`, NN2000;
  or `LAT`, `MSL`) and the height of the model's z = 0 in it.

REEF3D copies the file into its run log and into the LAGOON stores it writes
(`lagoon-convert` too), and DIVEMesh into its grid (`grid-geometry.dat`), so the
results keep their place when they leave the case folder
([format specification, 1.2](../format/SPEC.md)).

LAGOON transforms WGS 84 and ETRS89 longitude and latitude, Web Mercator, UTM and
the Norwegian NTM zones itself. For other CRSs install pyproj:
`pip install "lagoon-viz[geo]"` (it brings rasterio as well, which reads every
GeoTIFF and only the part of a large one that is needed).

## The Map tab of the set-up

*Set-up → Map*:

- **Georeference**: the CRS, the origin, the rotation, the vertical datum;
  *Apply* writes `georef.json`, *Remove* takes it away. *Place at* puts the
  centre of the domain at a latitude and longitude (`63.44, 10.40`), choosing the
  UTM zone when no CRS is set yet: the quickest way to start.
- **Domain**: *Draw on the map* turns the view to 2D from the top; drag the
  domain's rectangle there (B 10, snapped to the cell size B 1). **Shift + drag**
  turns the model on the map about the domain's centre: the domain stays
  axis-aligned in model coordinates and the map turns under it. Esc stops.
  Or type the extent (*Set B 10*) or the rotation (*Turn*).
- **Background map**: *Show map* adds a *Background map* item for the case.
- **Bathymetry**: choose a GeoTIFF, an Esri ASCII grid (`.asc`) or an x y z text
  file (soundings: spaces, tabs, commas or semicolons; a header line is skipped).
  Give its CRS if the file does not name one (`EPSG:4326` for longitude and
  latitude, `model` for model coordinates) and tick *depths* when the values are
  positive down. *Make geo.dat* carries it onto a grid over the domain (every B 1
  metres, two cells beyond the domain) and writes `geo.dat` in model coordinates,
  with `G 10 1` in `control.txt`. A raster is sampled bilinearly; scattered points
  are interpolated linearly on their triangulation (*TIN*), by inverse distance or
  from the nearest point; outside the data the nearest height is taken.
  Then *Run DIVEMesh*.

Every step is a command of the session: it can be undone, and a script does the
same:

```python
session.set_georef("EPSG:25833", origin=[270000.0, 7040000.0], rotation_deg=30.0)
session.set_domain(0.0, 2000.0, 0.0, 800.0)
session.turn_model(35.0)  # about the domain's centre
session.make_geo_dat("~/surveys/nidelva.tif")
session.add_background_map("Sentinel-2 2017 (satellite)", margin=1.0)
```

## Background maps

A *Background map* item shows a map round its case's domain, under the results
and the set-up, in the 3D view, the 2D view, paper figures, films and the set-up
sketch. Its properties:

| property | |
|---|---|
| Map | the source: Kartverket's topographic map, grey map or nautical chart (Norway), Sentinel-2 cloudless 2017 (satellite, Europe and the world), OpenStreetMap, a GeoTIFF of yours (an orthophoto, a chart) or a tile server (an XYZ URL with `{z}`, `{x}`, `{y}`) |
| File or URL | the GeoTIFF, or the tile server's URL |
| Margin | how far the map reaches round the domain, in domain sizes |
| Placed | below the model (a floor under the bed), or at the still water level round the domain; in 2D from the top it lies under everything |
| Opacity, Resolution | how strongly, and with how many pixels along the longer side |

Map tiles are fetched once and kept in a cache (`~/Library/Caches/LAGOON/tiles`
on a Mac; `LAGOON_CACHE` moves it). LAGOON asks the servers with its own name and
only for the tiles over the area shown, as their terms ask.

Every map shows the credit its source asks for: under the 3D view, in the corner
of films, in the corner of figures. Kartverket's maps are CC BY 4.0 (© Kartverket),
the Sentinel-2 cloudless 2016/2017 mosaic is CC BY 4.0 (EOX, modified Copernicus
Sentinel data), OpenStreetMap is ODbL (© OpenStreetMap contributors). Images whose
terms do not allow such use (Google, Esri) are not offered; a GeoTIFF of your own
is drawn with no credit.
