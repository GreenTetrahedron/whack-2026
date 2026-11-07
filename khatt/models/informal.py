"""Model 4 — microbus routes and street vendors (CLAUDE.md §8). Vendor numbers are estimates."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial import cKDTree

from khatt.config import Params
from khatt.geo import to_utm


@dataclass
class InformalResult:
    share_at_risk: np.ndarray  # per route
    status: np.ndarray  # competing | feeder | unaffected
    vendor_pitches: np.ndarray  # per new station
    vendor_pitches_lost: float


def route_vertex_tree(world):
    """KD-tree over all microbus route vertices (UTM), cached on the world."""
    cached = getattr(world, "_route_tree", None)
    if cached is not None:
        return cached
    xs, ys, rows = [], [], []
    for r, geom in enumerate(world.microbus.geometry):
        if geom is None or geom.is_empty:
            continue
        coords = np.asarray(geom.coords)
        x, y = to_utm(coords[:, 1], coords[:, 0])
        xs.append(np.atleast_1d(x))
        ys.append(np.atleast_1d(y))
        rows.append(np.full(len(coords), r))
    if xs:
        tree = cKDTree(np.column_stack([np.concatenate(xs), np.concatenate(ys)]))
        out = (tree, np.concatenate(rows))
    else:
        out = (None, np.zeros(0, np.int64))
    world._route_tree = out
    return out


def run(world, params: Params, overlay, station_x: np.ndarray, station_y: np.ndarray,
        exits: np.ndarray) -> InformalResult:
    n_routes = len(world.microbus)
    rows, a, b = world.microbus_pairs(int(params["informal.max_pair_gap"]))
    share = np.zeros(n_routes)
    if len(rows) and overlay.S:
        t0, t1 = overlay.pairs(a, b)
        at_risk = (t1 < t0 - float(params["informal.switch_margin_min"])).astype(float)
        count = np.bincount(rows, minlength=n_routes)
        share = np.divide(np.bincount(rows, weights=at_risk, minlength=n_routes), count,
                          out=np.zeros(n_routes), where=count > 0)

    competing = share >= float(params["informal.compete_threshold"])
    feeder = np.zeros(n_routes, bool)
    tree, vertex_row = route_vertex_tree(world)
    if tree is not None and len(station_x):
        hits = tree.query_ball_point(np.column_stack([station_x, station_y]),
                                     r=float(params["informal.feeder_radius_m"]))
        for h in hits:
            feeder[vertex_row[h]] = True
    status = np.where(competing, "competing", np.where(feeder, "feeder", "unaffected"))

    pitches = exits / 1000.0 * float(params["informal.vendors_per_1000_exits"])
    lost = float((share[competing]).sum() * float(params["informal.vendors_per_route"]))
    return InformalResult(share, status.astype(object), pitches, lost)
