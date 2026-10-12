# SPDX-License-Identifier: GPL-3.0-or-later
# Architect: Hans Bihs
"""The Earth under a case: its georeference, map tiles, bathymetry.

- :mod:`~lagoonviz.core.geo.crs`: from one CRS to another (pyproj, or LAGOON's own
  for WGS 84, ETRS89, Web Mercator, UTM and NTM)
- :mod:`~lagoonviz.core.geo.tiles`: map tile sources, their terms, the disk cache
- :mod:`~lagoonviz.core.geo.raster`: GeoTIFF, ASCII grids and x y z points
- :mod:`~lagoonviz.core.geo.basemap`: background maps warped into the model frame
- :mod:`~lagoonviz.core.geo.bathy`: bathymetry onto the grid, and geo.dat

The georeference itself is ``lagoon_format.geo`` (it belongs to the data format).
"""
