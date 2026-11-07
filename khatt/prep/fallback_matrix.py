"""Scipy fallback for the baseline matrix (CLAUDE.md §6, 08 fallback).

Nodes = res-8 cells + one node per (route, stop). Edges:
  cell ↔ neighbouring cell     walk: centroid distance × circuity / speed
  cell → route-stop            walk access + wait (headway/2 × irregularity[mode])
  route-stop → cell            walk egress
  route-stop k → k+1           ride time
Dijkstra from every cell (in chunks) → T0[N, N], minutes, 999 unreachable, diag = intracell.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import h3
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra
from scipy.spatial import cKDTree

from khatt.config import Params
from khatt.geo import haversine_m, to_utm

MIN_EDGE = 1e-6  # scipy drops zero-weight edges


@dataclass
class Route:
    route_id: str
    mode: str  # metro | bus | microbus
    headway_min: float
    stop_lat: list[float]
    stop_lng: list[float]
    ride_min: list[float] | None = None  # len = stops - 1; None → distance / speed_kmh
    speed_kmh: float = 20.0
    bidirectional: bool = False
    stop_ids: list[str] = field(default_factory=list)

    def segment_minutes(self) -> np.ndarray:
        if self.ride_min is not None:
            return np.maximum(np.asarray(self.ride_min, float), MIN_EDGE)
        lat = np.asarray(self.stop_lat)
        lng = np.asarray(self.stop_lng)
        d = haversine_m(lat[:-1], lng[:-1], lat[1:], lng[1:])
        return np.maximum(d / (self.speed_kmh * 1000 / 60), MIN_EDGE)


def _dedupe_min(rows, cols, vals, n):
    rows = np.asarray(rows, np.int64)
    cols = np.asarray(cols, np.int64)
    vals = np.maximum(np.asarray(vals, float), MIN_EDGE)
    key = rows * n + cols
    order = np.lexsort((vals, key))
    key, vals = key[order], vals[order]
    first = np.concatenate([[True], key[1:] != key[:-1]])
    key, vals = key[first], vals[first]
    return csr_matrix((vals, (key // n, key % n)), shape=(n, n))


def build_graph(hex_ids, hex_lat, hex_lng, routes: list[Route], params: Params):
    N = len(hex_ids)
    speed = params.walk_m_per_min
    circuity = float(params["walk.circuity"])
    radius = float(params["routing.stop_access_radius_m"])
    x, y = to_utm(hex_lat, hex_lng)
    xy = np.column_stack([np.atleast_1d(x), np.atleast_1d(y)])
    pos = {h: k for k, h in enumerate(hex_ids)}
    rows, cols, vals = [], [], []

    for k, h in enumerate(hex_ids):
        for nb in h3.grid_disk(h, 1):
            j = pos.get(nb)
            if j is None or j == k:
                continue
            rows.append(k)
            cols.append(j)
            vals.append(np.hypot(*(xy[k] - xy[j])) * circuity / speed)

    tree = cKDTree(xy)
    next_node = N
    for route in routes:
        n_stops = len(route.stop_lat)
        if n_stops < 2:
            continue
        irregularity = float(params.get(f"routing.irregularity_factor.{route.mode}", 1.0))
        wait = route.headway_min / 2.0 * irregularity
        sx, sy = to_utm(route.stop_lat, route.stop_lng)
        sxy = np.column_stack([np.atleast_1d(sx), np.atleast_1d(sy)])
        seg = route.segment_minutes()
        directions = [np.arange(n_stops)]
        if route.bidirectional:
            directions.append(np.arange(n_stops)[::-1])
        for order in directions:
            nodes = np.arange(next_node, next_node + n_stops)
            next_node += n_stops
            for pos_k, stop_k in enumerate(order):
                near = tree.query_ball_point(sxy[stop_k], r=radius)
                if not near:
                    near = [int(tree.query(sxy[stop_k])[1])]
                for c in near:
                    walk = np.hypot(*(sxy[stop_k] - xy[c])) * circuity / speed
                    rows += [c, nodes[pos_k]]
                    cols += [nodes[pos_k], c]
                    vals += [walk + wait, walk]
                if pos_k + 1 < n_stops:
                    nxt = order[pos_k + 1]
                    rows.append(nodes[pos_k])
                    cols.append(nodes[pos_k + 1])
                    vals.append(seg[min(stop_k, nxt)])
    return _dedupe_min(rows, cols, vals, next_node), N


def build_fallback_T0(hex_ids, hex_lat, hex_lng, routes: list[Route], params: Params,
                      chunk: int = 256) -> np.ndarray:
    G, N = build_graph(hex_ids, hex_lat, hex_lng, routes, params)
    unreachable = float(params["routing.unreachable_min"])
    T0 = np.empty((N, N), np.float32)
    for start in range(0, N, chunk):
        idx = np.arange(start, min(start + chunk, N))
        D = dijkstra(G, directed=True, indices=idx)[:, :N]
        D[~np.isfinite(D)] = unreachable
        T0[idx] = np.minimum(D, unreachable)
    np.fill_diagonal(T0, float(params["access.intracell_min"]))
    return T0
