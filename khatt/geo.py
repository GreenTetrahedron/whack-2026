"""Projections, distances and H3 helpers.

Storage is WGS84 (EPSG:4326); metric work happens in UTM 36N (EPSG:32636).
h3 functions take (lat, lng); GeoJSON / deck.gl use [lng, lat].
"""

from __future__ import annotations

from functools import cache

import h3
import numpy as np
from pyproj import Transformer

WGS84 = "EPSG:4326"
UTM36N = "EPSG:32636"
EARTH_RADIUS_M = 6_371_008.8


@cache
def _to_utm() -> Transformer:
    return Transformer.from_crs(WGS84, UTM36N, always_xy=True)


@cache
def _from_utm() -> Transformer:
    return Transformer.from_crs(UTM36N, WGS84, always_xy=True)


def to_utm(lat, lng) -> tuple[np.ndarray, np.ndarray]:
    """(lat, lng) degrees → (easting_m, northing_m)."""
    east, north = _to_utm().transform(np.asarray(lng, dtype=float), np.asarray(lat, dtype=float))
    return np.asarray(east), np.asarray(north)


def from_utm(east, north) -> tuple[np.ndarray, np.ndarray]:
    """(easting_m, northing_m) → (lat, lng) degrees."""
    lng, lat = _from_utm().transform(np.asarray(east, dtype=float), np.asarray(north, dtype=float))
    return np.asarray(lat), np.asarray(lng)


def haversine_m(lat1, lng1, lat2, lng2) -> np.ndarray:
    lat1, lng1, lat2, lng2 = (np.radians(np.asarray(v, dtype=float)) for v in (lat1, lng1, lat2, lng2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lng2 - lng1) / 2) ** 2
    return 2 * EARTH_RADIUS_M * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


def cell(lat: float, lng: float, res: int) -> str:
    return h3.latlng_to_cell(float(lat), float(lng), res)


def cell_center(cell_id: str) -> tuple[float, float]:
    """Returns (lat, lng)."""
    lat, lng = h3.cell_to_latlng(cell_id)
    return lat, lng


def cells_for_points(lat, lng, res: int) -> np.ndarray:
    lat = np.asarray(lat, dtype=float)
    lng = np.asarray(lng, dtype=float)
    return np.array([h3.latlng_to_cell(a, b, res) for a, b in zip(lat, lng)], dtype=object)


def bbox_cells(bbox: tuple[float, float, float, float], res: int) -> list[str]:
    """All cells whose centre lies in bbox = (lng_min, lat_min, lng_max, lat_max)."""
    lng_min, lat_min, lng_max, lat_max = bbox
    outer = [(lat_min, lng_min), (lat_min, lng_max), (lat_max, lng_max), (lat_max, lng_min)]
    return sorted(h3.polygon_to_cells(h3.LatLngPoly(outer), res))


def lnglat(lat: float, lng: float) -> list[float]:
    """GeoJSON / deck.gl order."""
    return [float(lng), float(lat)]
