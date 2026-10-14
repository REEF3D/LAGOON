# Maps and the place of a case

REEF3D computes in model coordinates. A **georeference** says where they lie on
the Earth; with it LAGOON lays a map or a satellite image under the results and
the set-up, the domain can be drawn on the map, and a bathymetry survey becomes
DIVEMesh's `geo.dat`.

## Starting a case on the map

With no case open, the *Map* tab starts one at Trondheim on Kartverket's map:
*Find* takes it anywhere else.

*File → New case from map…* asks where: a place name (*Nidelva*, *Trondheim
havn*; see [Finding a place](#finding-a-place)) or a latitude and longitude
(`63.44, 10.40`). LAGOON starts a draft case there: a
domain of 1000 × 500 m centred on the place, in its UTM zone, with a background
map, in the set-up's 2D view and the *Map* tab. Then

1. drag the domain where it belongs (*Draw on the map*; Shift + drag turns it),
2. add the bathymetry (*Make geo.dat*) and the commands of `control.txt` and
   `ctrl.txt`,
3. *Create case…* asks for a folder (new or empty) and puts the case there: its
   input files, `georef.json`, `geo.dat` and what DIVEMesh made. From then on the
   case is that folder.

Until then the draft lives in LAGOON's drafts folder
(`~/Library/Application Support/LAGOON/Drafts` on a Mac; `LAGOON_DRAFTS` moves
it), so nothing is lost when LAGOON closes: open it again from there. In a script:

```python
case = session.new_case_from_map("Nidelva", length=2000.0)
session.place_case((63.44, 10.40))   # move it
session.create_case("~/runs/nidelva")
```

## Finding a place

A place is found by its name or its latitude and longitude:

- **Names** come from Kartverket's place names in Norway (sentralt
  stedsnavnregister: rivers, bays, islets, skerries, farms, towns) and from
  OpenStreetMap's Nominatim for the whole world. When the name as typed is not
  there, LAGOON looks for names beginning so (*Munkh* finds Munkholmen) and then
  for names alike (*Trondhem* finds Trondheim).
- **Name, municipality** narrows it: *Nidelva, Trondheim* (or *Nidelva, Arendal*)
  when several places have the name.
- The likeliest place comes first: the name as typed before names beginning so,
  towns and fjords before farms, and **the places near the case** before those
  far away (near Trondheim when there is no case yet). The list under the field
  (or *Which one?* for a new case) offers the others.
- **Coordinates**: `63.44, 10.40`, `63.44N 10.4E`, `lat 63.44 lon 10.40`, `63,44
  10,40`; latitude first unless N/S/E/W or *lat*/*lon* say otherwise.

One request goes to each service per search (none while typing), with LAGOON's
name, as their terms ask. Offline, type the latitude and longitude.

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

- **Find the place**: a place name (*Nidelva, Trondheim*: name, municipality) or
  a latitude and longitude, then *Find*: the centre of the domain goes there (in
  the place's UTM zone when the case has no CRS yet), and the case gets a
  background map. When several places have the name, the list under the field
  offers the others, those near the case first.
- **Georeference**: the CRS, the origin, the rotation, the vertical datum;
  *Apply* writes `georef.json`, *Remove* takes it away.
- **Domain**: *Draw on the map* turns the view to 2D from the top; drag the
  domain's rectangle there (B 10, snapped to the cell size B 1). **Shift + drag**
  turns the model on the map about the domain's centre: the domain turns, the map
  stays north up (B 10 keeps its numbers: the model's own axes turn). Esc stops.
  Or type the extent (*Set B 10*) or the rotation (*Turn*).
- **Background map**: *Show map* adds a *Background map* item for the case. The
  case keeps it (`LAGOON_map.json` in its folder, as last set: map, margin,
  placement, opacity, shown or not): opened again, its results in a new window or
  from a script, it has its map. A georeferenced case that keeps none opens with
  the place's best map (Kartverket's in Norway, else OpenStreetMap); one whose map
  was taken away opens without.
- **Bathymetry**: *From* **Download** needs no file: LAGOON asks the web for the
  bathymetry of the domain and its margin. *Best available* takes, the finest
  first, each filling what the ones before had no data for: Kartverket's
  national topobathy terrain model in Norway (land and sea in one model, 1 m
  where surveyed, else 50 m; heights in NN2000), EMODnet's Digital Terrain Model
  at sea in Europe (about 115 m, heights to LAT), the Copernicus DEM GLO-30 on
  land anywhere (30 m, heights to EGM2008; its 1° tiles are some 30 MB each,
  fetched once) NOAA's ETOPO 2022 for the rest of the world, land and sea
  (15 arc-seconds, about 450 m, EGM2008; at sea built on GEBCO 2022), and GEBCO
  2020 (15 arc-seconds, public domain) last. Each can be chosen alone too.
  **The coast** of *Best available* lies where Copernicus' 30 m land has it (land
  above 0), not where a coarse sea source smears it over its 115 or 450 m cells:
  land takes Copernicus' heights; the sea (low ground joined to the open sea) the
  sea sources' depths, at least 0.1 m deep at the coast and 2 cm deeper per metre
  out, to 1 m, so a coarse cell over the coast no longer puts land into the sea or
  sea onto the land. Low ground behind dikes (polders) keeps Copernicus' heights.
  Kartverket's model, where it has data, goes over all of them.
  **Near the coast** a coarse sea source is a guess: its first 1.5 cells out (about
  700 m for ETOPO and GEBCO, 170 m for EMODnet) mix land and sea, and ETOPO and
  GEBCO give tidal flats, harbours and creeks a flat 1 or 2 m. *Near the coast*
  makes the bed there again, between the coastline and the sea's depths further
  out: *A beach profile (Dean)*, the default, is A x^(2/3) deep at x metres from
  the coast (*Dean's A*: about 0.21 D^0.48 for a grain size D in mm, 0.1 for
  0.2 mm sand), bent to meet the sea's depths; *The smoothest bed* goes from 0 m at
  the coast to the sea's depths, for creeks, harbours and mudflats, whose bed is no
  beach; *At most 1 m deep* keeps the coarse depths, held 0.1 to 1 m deep next to
  the coast, as LAGOON did before (and as a geo.dat made before is made again).
  *Trust the sea from* also makes the sea source's shallows again (Mumbai: 2.5 m,
  over ETOPO's flat 1 and 2 m).
  **Over it** go files of yours, the lower in the list over the higher, blended in
  over five cells at their edges, and met by the profile near the coast: surveys,
  chart soundings (*depths*; *Its zero below MSL*: the chart's datum, about half
  the spring tidal range), an intertidal terrain model from satellite waterlines,
  DeltaDTM (30 m coastal terrain without buildings and trees, CC BY 4.0). ICESat-2
  files (NASA's ATL24, `.h5`, from the NSIDC Data Access Tool or Earthdata Search)
  and *ICESat-2 soundings*, downloaded for the domain (a free NASA Earthdata login:
  `EARTHDATA_TOKEN`, or `~/.netrc` with `machine urs.earthdata.nasa.gov`; reading
  them needs h5py, in `lagoon-viz[geo]`), give the sea floor along the satellite's
  tracks to 20 to 40 m in clear water (a few metres in turbid), heights over EGM2008
  as Copernicus' and ETOPO's: too sparse to lie over the sea, they guide the
  profile near the coast. EMODnet's depths
  are below the lowest astronomical tide (LAT): *LAT below MSL* (from tide tables
  or the chart's datum notes, about half the spring tidal range) puts them on the
  land's zero. Or *A file of mine…*: a GeoTIFF, an Esri ASCII grid
  (`.asc`) or an x y z text file (soundings: spaces, tabs, commas or semicolons;
  a header line is skipped). Give its CRS if the file does not name one
  (`EPSG:4326` for longitude and latitude, `model` for model coordinates) and
  tick *depths* when the values are positive down. *Make geo.dat* asks where to
  store `geo.dat` (a draft case started on the map goes to that folder, with its
  input files and `georef.json`; for a case in its own folder the dialog starts
  there, and a place elsewhere gets a copy: DIVEMesh reads the one in the case
  folder), carries the bathymetry onto a grid over the domain (two cells beyond
  it) as fine as *Resolution* says: as DIVEMesh's cells (B 1, the default), *the
  finest the data has* (Kartverket's 1 m in Norway, Copernicus' 30 m on land,
  EMODnet's 115 m, ETOPO's and GEBCO's 450 m; a file's own cells or the distance
  between its points; the tab says what that is over the domain), or every so many
  metres; at most *Most points* (4 million by default, 0: no limit), coarser where
  the domain would need more (DIVEMesh reads, bins and interpolates every point and
  writes them out again for ParaView; geo.dat takes some 35 bytes a point), and writes it in model coordinates, with `G 10 1`
  in `control.txt`. A progress bar shows how far it is; *Cancel* keeps the
  geo.dat there was. The *geo.dat points* item then shows the points written
  (zoom in to see their spacing). A raster
  is sampled bilinearly; scattered points are interpolated linearly on their
  triangulation (*TIN*), by inverse distance or from the nearest point; outside
  the data the nearest height is taken. Downloads are kept in the cache.
  Then *Run DIVEMesh*.
- **Turned or moved afterwards**: `geo.dat` is in model coordinates, the bed as it
  lay when it was made. Turning the model (*Turn*, Shift + drag), moving it on the
  map, another CRS or a larger domain leave it as it was, and DIVEMesh would mesh
  the bed of before. LAGOON keeps how `geo.dat` was made (`LAGOON_geo.dat.json`
  beside it: its source, spacing, interpolation, and the georeference and domain
  it was made for), so the Map tab then says it no longer fits, with *Make geo.dat
  again* (the same source and settings, for the model as it lies now), and *Run
  DIVEMesh* asks to make it again first. A geo.dat from elsewhere is not checked.

Every step is a command of the session: it can be undone, and a script does the
same:

```python
session.set_georef("EPSG:25833", origin=[270000.0, 7040000.0], rotation_deg=30.0)
session.set_domain(0.0, 2000.0, 0.0, 800.0)
session.turn_model(35.0)  # about the domain's centre
session.make_geo_dat("best")  # downloaded; or a file: "~/surveys/nidelva.tif"
session.make_geo_dat("best", spacing="finest")  # as fine as the data is
session.make_geo_dat("best", nearshore="smooth", trust_depth=2.5)  # creeks, mudflats
session.make_geo_dat(
    "best",
    overlays=[{"path": "~/charts/soundings.xyz", "crs": "EPSG:4326", "depths": True,
               "below_msl": 2.6}, "~/icesat2/"],  # a chart, then ICESat-2 files
    icesat2=True,  # and ICESat-2 downloaded for the domain
)
session.geo_dat_state()  # why geo.dat no longer fits the case, "" if it does
session.remake_geo_dat()  # from the same source, for the model as it lies now
session.add_background_map("Sentinel-2 (satellite)", margin=1.0)
```

## Background maps

A *Background map* item shows a map round its case's domain, under the results
and the set-up (each mode keeps its own view: the set-up its 2D map, the results
their 3D view, the map under both), in the 3D view, the 2D view, paper figures, films and the set-up
sketch. Its properties:

| property | |
|---|---|
| Map | the source: Kartverket's topographic map, grey map or nautical chart (Norway), Sentinel-2 cloudless (satellite: 2017 in Europe, 2016 elsewhere, where the 2017 mosaic has no images), OpenStreetMap, a GeoTIFF of yours (an orthophoto, a chart) or a tile server (an XYZ URL with `{z}`, `{x}`, `{y}`) |
| File or URL | the GeoTIFF, or the tile server's URL |
| Margin | how far the map reaches round the domain, in domain sizes |
| Placed | below the model (a floor under the bed), or at the still water level round the domain; in 2D from the top it lies under everything |
| Opacity, Resolution | how strongly, and with how many pixels along the longer side |

The 2D view from the top shows north up: the map lies as on a map, and a turned
model is drawn turned. Turning the model (*Turn*, Shift + drag) turns the domain
and its grid, never the map, in 2D and in 3D. *North up* in the view's bar (or
++n++) turns any view, 2D or 3D, back so that north points up the screen (in 3D:
away from you, the tilt kept), after you have turned it about. North is true north
at the middle of the domain: a CRS's grid north is true north only on its central
meridian (in a UTM zone up to a few degrees off at its edges), so a model lined up
with the CRS's grid lies a little turned in the north-up view, as on the map.

Finding a place far from the case's CRS (Mumbai for a case set up in Norway)
gives the case the place's UTM zone, as a CRS meant for one area turns and
stretches the map far away from it (WGS 84 UTM outside Europe, ETRS89 UTM in it);
the Map tab says so. A CRS chosen in the Map tab before the place is found is
kept when it is meant for the place.

In the 2D view from the top the map follows the view: zoom in and it is fetched
again in finer detail, zoom out or pan and it covers the new area (once the view
rests; the map before stays until the new one is there). In 3D it covers the
domain and its margin.

While a map is fetched, a bar above the view counts its tiles. Map tiles are
fetched once and kept in a cache (`~/Library/Caches/LAGOON/tiles`
on a Mac; `LAGOON_CACHE` moves it). LAGOON asks the servers with its own name and
only for the tiles over the area shown, as their terms ask.

Every map shows the credit its source asks for: under the 3D view, in the corner
of films, in the corner of figures. Kartverket's maps are CC BY 4.0 (© Kartverket),
the Sentinel-2 cloudless 2016/2017 mosaic is CC BY 4.0 (EOX, modified Copernicus
Sentinel data), OpenStreetMap is ODbL (© OpenStreetMap contributors). Images whose
terms do not allow such use (Google, Esri) are not offered; a GeoTIFF of your own
is drawn with no credit.
