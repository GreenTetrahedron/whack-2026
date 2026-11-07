"""Walking catchments on the in-memory walk graph (CSR + numba Dijkstra with cutoff)."""

from __future__ import annotations

import heapq
import threading
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numba import njit
from scipy.spatial import cKDTree

from khatt.geo import to_utm


@njit(cache=True)
def _dijkstra_cutoff(indptr, indices, weights, source, cutoff, dist):
    """Settled (node, time) within `cutoff`. `dist` is an all-inf workspace, restored on exit."""
    heap = [(0.0, np.int64(source))]
    touched = [np.int64(source)]
    dist[source] = 0.0
    out_nodes = []
    out_dist = []
    while len(heap) > 0:
        d, u = heapq.heappop(heap)
        if d > dist[u]:
            continue
        out_nodes.append(u)
        out_dist.append(d)
        for k in range(indptr[u], indptr[u + 1]):
            v = np.int64(indices[k])
            nd = d + weights[k]
            if nd <= cutoff and nd < dist[v]:
                if dist[v] == np.inf:
                    touched.append(v)
                dist[v] = nd
                heapq.heappush(heap, (nd, v))
    for t in touched:
        dist[t] = np.inf
    return np.array(out_nodes, dtype=np.int64), np.array(out_dist, dtype=np.float64)


class WalkGraph:
    """Directed walk graph in CSR form. Node coordinates are UTM 36N metres."""

    def __init__(self, indptr, indices, weights, x, y, node_r9, node_ids=None):
        self.indptr = np.asarray(indptr, np.int64)
        self.indices = np.asarray(indices, np.int64)
        self.weights = np.asarray(weights, np.float64)
        self.x = np.asarray(x, np.float64)
        self.y = np.asarray(y, np.float64)
        self.node_r9 = np.asarray(node_r9, np.int64)  # row in r9 table, -1 if none
        self.node_ids = np.asarray(node_ids if node_ids is not None else np.arange(len(self.x)))
        self.n = len(self.x)
        self._tree = cKDTree(np.column_stack([self.x, self.y])) if self.n else None
        self._dist = np.full(self.n, np.inf)
        self._lock = threading.Lock()
        self._cache: dict[tuple[int, float], tuple[np.ndarray, np.ndarray]] = {}

    # ---------- construction / persistence ----------
    @classmethod
    def from_edges(cls, x, y, src, dst, walk_min, node_r9, node_ids=None) -> WalkGraph:
        n = len(x)
        order = np.lexsort((dst, src))
        src, dst, walk_min = np.asarray(src)[order], np.asarray(dst)[order], np.asarray(walk_min)[order]
        indptr = np.zeros(n + 1, np.int64)
        np.add.at(indptr, np.asarray(src) + 1, 1)
        indptr = np.cumsum(indptr)
        return cls(indptr, dst, walk_min, x, y, node_r9, node_ids)

    @classmethod
    def from_networkx(cls, G, node_r9_map: dict | None = None) -> WalkGraph:
        """G: projected OSMnx graph (x, y in EPSG:32636) with edge attribute `walk_min`."""
        node_ids = np.array(list(G.nodes))
        pos = {nid: k for k, nid in enumerate(node_ids)}
        x = np.array([G.nodes[n]["x"] for n in node_ids], float)
        y = np.array([G.nodes[n]["y"] for n in node_ids], float)
        src, dst, w = [], [], []
        for u, v, data in G.edges(data=True):
            src.append(pos[u])
            dst.append(pos[v])
            w.append(float(data["walk_min"]))
        node_r9 = np.full(len(node_ids), -1, np.int64)
        if node_r9_map:
            for k, nid in enumerate(node_ids):
                node_r9[k] = node_r9_map.get(nid, -1)
        return cls.from_edges(x, y, np.array(src), np.array(dst), np.array(w), node_r9, node_ids)

    def save(self, path: Path) -> None:
        np.savez_compressed(
            path, indptr=self.indptr, indices=self.indices, weights=self.weights,
            x=self.x, y=self.y, node_r9=self.node_r9, node_ids=self.node_ids,
        )

    @classmethod
    def load(cls, path: Path) -> WalkGraph:
        z = np.load(path)
        return cls(z["indptr"], z["indices"], z["weights"], z["x"], z["y"], z["node_r9"], z["node_ids"])

    # ---------- queries ----------
    def snap(self, lat, lng) -> tuple[np.ndarray, np.ndarray]:
        """Nearest node for each point → (node index, distance m). -1 / inf if graph empty."""
        lat = np.atleast_1d(np.asarray(lat, float))
        lng = np.atleast_1d(np.asarray(lng, float))
        if self._tree is None:
            return np.full(len(lat), -1, np.int64), np.full(len(lat), np.inf)
        east, north = to_utm(lat, lng)
        dist, idx = self._tree.query(np.column_stack([np.atleast_1d(east), np.atleast_1d(north)]))
        return np.asarray(idx, np.int64), np.asarray(dist, float)

    def snap_xy(self, east, north) -> tuple[np.ndarray, np.ndarray]:
        dist, idx = self._tree.query(np.column_stack([east, north]))
        return np.asarray(idx, np.int64), np.asarray(dist, float)

    def reach(self, node: int, cutoff_min: float) -> tuple[np.ndarray, np.ndarray]:
        """(nodes, walk minutes) reachable from `node` within cutoff. Cached by node id."""
        key = (int(node), float(cutoff_min))
        hit = self._cache.get(key)
        if hit is not None:
            return hit
        with self._lock:
            result = _dijkstra_cutoff(
                self.indptr, self.indices, self.weights, np.int64(node), float(cutoff_min), self._dist
            )
        if len(self._cache) > 20_000:
            self._cache.clear()
        self._cache[key] = result
        return result


@dataclass
class Catchment:
    station: int  # global station index
    node: int
    nodes: np.ndarray  # reached walk nodes
    node_min: np.ndarray  # walk minutes to those nodes
    r9_idx: np.ndarray  # reached r9 cells (unique)
    r9_min: np.ndarray  # min walk minutes to each r9 cell


def reached_cells(graph: WalkGraph, nodes: np.ndarray, times: np.ndarray):
    """Collapse reached nodes to r9 cells (min time per cell)."""
    cells = graph.node_r9[nodes]
    keep = cells >= 0
    cells, times = cells[keep], times[keep]
    if len(cells) == 0:
        return np.zeros(0, np.int64), np.zeros(0, float)
    order = np.lexsort((times, cells))
    cells, times = cells[order], times[order]
    first = np.concatenate([[True], cells[1:] != cells[:-1]])
    return cells[first], times[first]


def catchments(graph: WalkGraph, station_nodes: np.ndarray, cutoff_min: float) -> list[Catchment]:
    out = []
    for s, node in enumerate(station_nodes):
        if node < 0:
            empty = np.zeros(0, np.int64)
            out.append(Catchment(s, -1, empty, np.zeros(0), empty, np.zeros(0)))
            continue
        nodes, times = graph.reach(int(node), cutoff_min)
        r9, r9_min = reached_cells(graph, nodes, times)
        out.append(Catchment(s, int(node), nodes, times, r9, r9_min))
    return out


def assign_nearest(catches: list[Catchment], n_cells: int, cutoff_min: float,
                   priority: list | None = None):
    """Each r9 cell goes to the single nearest station (no double counting).

    Ties go to the station that comes first in `priority` order (a sortable key per catchment,
    e.g. (line_id, idx)), so the result does not depend on the order lines were drawn.
    Returns owner[n_cells] (station index or -1) and owner_min[n_cells].
    """
    owner = np.full(n_cells, -1, np.int64)
    best = np.full(n_cells, np.inf)
    order = range(len(catches)) if priority is None else sorted(
        range(len(catches)), key=lambda k: priority[k])
    for k in order:
        c = catches[k]
        if len(c.r9_idx) == 0:
            continue
        mask = c.r9_min <= cutoff_min
        cells, t = c.r9_idx[mask], c.r9_min[mask]
        better = t < best[cells]
        best[cells[better]] = t[better]
        owner[cells[better]] = c.station
    return owner, best
