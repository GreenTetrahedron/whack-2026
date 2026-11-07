"""Toy networks for engine tests: a small base graph plus a brute-force line expansion."""

from __future__ import annotations

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra

from khatt.config import Params
from khatt.engine.lines import LineSpec, cumulative_distance_m
from khatt.engine.multiline import joint_rstar, wait_board
from khatt.engine.overlay import overlay
from khatt.geo import haversine_m


def toy_base(n: int = 24, seed: int = 0, k: int = 3):
    """Random points near Cairo joined to k nearest neighbours (symmetric), slow 'walk' edges."""
    rng = np.random.default_rng(seed)
    lat = 30.00 + rng.random(n) * 0.06
    lng = 31.20 + rng.random(n) * 0.06
    d = haversine_m(lat[:, None], lng[:, None], lat[None, :], lng[None, :])
    w = np.zeros((n, n))
    for i in range(n):
        for j in np.argsort(d[i])[1 : k + 1]:
            minutes = d[i, j] / 75.0  # 4.5 km/h
            w[i, j] = w[j, i] = minutes
    # a few fast "bus" shortcuts so the baseline is not pure walking
    for _ in range(3):
        a, b = rng.choice(n, 2, replace=False)
        w[a, b] = w[b, a] = d[a, b] / 300.0
    T0 = dijkstra(csr_matrix(w), directed=False).astype(np.float32)
    T0[~np.isfinite(T0)] = 999.0
    return lat, lng, w, T0


def toy_line(line_id: str, nodes: list[int], lat, lng, headway=4.0, speed=35.0, dwell=30.0):
    return LineSpec(
        line_id=line_id,
        name=f"Line {line_id}",
        stations=[(float(lat[k]), float(lng[k])) for k in nodes],
        headway_min=headway,
        speed_kmh=speed,
        dwell_s=dwell,
    )


def engine_T1(T0, lines, station_hex, params: Params):
    if not lines:
        return T0.copy()
    rstar = joint_rstar(lines, station_hex, T0, params)
    wb = wait_board(lines, params)
    return overlay(T0, T0[:, station_hex], T0[station_hex, :], wb, rstar)


def brute_force_T1(w, lines, line_nodes, params: Params):
    """Dijkstra on base graph + per-station (departing, arriving) on-board states."""
    n = w.shape[0]
    edges = [(i, j, w[i, j]) for i in range(n) for j in range(n) if w[i, j] > 0]
    wb_all = wait_board(lines, params)
    next_id = n
    g = 0
    for ln, nodes in zip(lines, line_nodes):
        m = len(nodes)
        dep = list(range(next_id, next_id + m))
        arr = list(range(next_id + m, next_id + 2 * m))
        next_id += 2 * m
        cum = cumulative_distance_m(ln.stations)
        m_per_min = ln.speed_kmh * 1000 / 60
        for k in range(m):
            edges.append((nodes[k], dep[k], wb_all[g + k]))  # wait + board
            edges.append((arr[k], nodes[k], 0.0))  # alight
            edges.append((arr[k], dep[k], ln.dwell_s / 60))  # stay on board through k
            if k + 1 < m:
                t = (cum[k + 1] - cum[k]) / m_per_min
                edges.append((dep[k], arr[k + 1], t))
                edges.append((dep[k + 1], arr[k], t))
        g += m
    size = next_id
    src = [e[0] for e in edges]
    dst = [e[1] for e in edges]
    # tiny epsilon keeps zero-cost alight edges in the sparse matrix
    val = [max(e[2], 1e-12) for e in edges]
    G = csr_matrix((val, (src, dst)), shape=(size, size))
    D = dijkstra(G, directed=True, indices=np.arange(n))[:, :n]
    D[~np.isfinite(D)] = 999.0
    return np.minimum(D, 999.0).astype(np.float32)
