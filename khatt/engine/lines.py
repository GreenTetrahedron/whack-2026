"""Line / Station specs, station snapping and the per-line ride-time matrix."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from khatt.config import Params
from khatt.geo import haversine_m


@dataclass
class LineSpec:
    line_id: str
    name: str
    stations: list[tuple[float, float]]  # (lat, lng), in order, >= 2
    headway_min: float | None = None
    speed_kmh: float | None = None
    dwell_s: float | None = None

    def resolved(self, params: Params) -> LineSpec:
        """Fill missing operating values from params (central or overridden)."""
        return LineSpec(
            line_id=str(self.line_id),
            name=self.name or str(self.line_id),
            stations=[(float(lat), float(lng)) for lat, lng in self.stations],
            headway_min=float(self.headway_min if self.headway_min else params["line.headway_min"]),
            speed_kmh=float(self.speed_kmh if self.speed_kmh else params["line.speed_kmh"]),
            dwell_s=float(self.dwell_s if self.dwell_s is not None else params["line.dwell_s"]),
        )


@dataclass
class Station:
    line_id: str
    idx: int  # position on its line
    lat: float
    lng: float
    hex_idx: int = -1  # row in hex_index.json
    node: int = -1  # walk-graph node (CSR index), -1 if none within range
    snap_m: float = float("nan")  # distance to that node
    access_min: float = 0.0  # extra walk to the cell centre (stations outside modelled cells)
    global_idx: int = -1  # position in the joint station list
    extra: dict = field(default_factory=dict)


def validate_lines(lines: list[LineSpec], params: Params) -> None:
    if not lines:
        return
    ids = [ln.line_id for ln in lines]
    if len(set(ids)) != len(ids):
        raise ValueError(f"Duplicate line_id in {ids}")
    total = 0
    for ln in lines:
        if len(ln.stations) < 2:
            raise ValueError(f"Line '{ln.name}' needs at least 2 stations")
        for lat, lng in ln.stations:
            if not (-90 <= lat <= 90 and -180 <= lng <= 180):
                raise ValueError(f"Line '{ln.name}': invalid coordinate ({lat}, {lng})")
        if ln.headway_min is not None and ln.headway_min <= 0:
            raise ValueError(f"Line '{ln.name}': headway must be > 0")
        if ln.speed_kmh is not None and ln.speed_kmh <= 0:
            raise ValueError(f"Line '{ln.name}': speed must be > 0")
        total += len(ln.stations)
    limit = int(params["line.max_total_stations"])
    if total > limit:
        raise ValueError(f"{total} stations in total; the limit is {limit}")


def cumulative_distance_m(stations: list[tuple[float, float]]) -> np.ndarray:
    lat = np.array([s[0] for s in stations], dtype=float)
    lng = np.array([s[1] for s in stations], dtype=float)
    seg = haversine_m(lat[:-1], lng[:-1], lat[1:], lng[1:])
    return np.concatenate([[0.0], np.cumsum(seg)])


def ride_matrix(stations: list[tuple[float, float]], speed_kmh: float, dwell_s: float) -> np.ndarray:
    """ride[i, j] = dist(i, j) / speed + dwell × (intermediate stops), minutes, symmetric."""
    cum = cumulative_distance_m(stations)
    m_per_min = speed_kmh * 1000.0 / 60.0
    dist = np.abs(cum[:, None] - cum[None, :])
    stops = np.abs(np.arange(len(stations))[:, None] - np.arange(len(stations))[None, :])
    intermediate = np.maximum(stops - 1, 0)
    ride = dist / m_per_min + intermediate * (dwell_s / 60.0)
    np.fill_diagonal(ride, 0.0)
    return ride.astype(np.float64)


def line_length_m(stations: list[tuple[float, float]]) -> float:
    return float(cumulative_distance_m(stations)[-1])
