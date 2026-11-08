"""Walk graph from a local OSM .pbf (pyosmium) instead of the Overpass API.

Same way filter as OSMnx's network_type="walk". The graph is simplified: only intersections and
way endpoints become nodes; edge length is the summed segment length along the way. Every edge is
walkable both ways. Only the largest connected component is kept.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from khatt.geo import haversine_m, to_utm

# OSMnx "walk" filter (osmnx/_overpass.py, network_type="walk")
EXCLUDED_HIGHWAY = re.compile(
    r"abandoned|bus_guideway|construction|cycleway|motor|no|planned|platform|proposed|raceway|razed"
)


def walkable(tags) -> bool:
    hw = tags.get("highway")
    if hw is None or EXCLUDED_HIGHWAY.search(hw):
        return False
    if tags.get("area") == "yes" or tags.get("foot") == "no":
        return False
    return tags.get("access") != "private" and tags.get("service") != "private"


@dataclass
class WalkEdges:
    node_ids: np.ndarray  # OSM ids of kept nodes
    lat: np.ndarray
    lng: np.ndarray
    src: np.ndarray  # rows into node arrays, both directions included
    dst: np.ndarray
    length_m: np.ndarray


def build_walk_edges(pbf: Path) -> WalkEdges:
    import osmium

    # pass 1: how many walkable ways use each node; way endpoints are always kept
    uses: dict[int, int] = {}
    for way in osmium.FileProcessor(str(pbf), osmium.osm.WAY):
        if not walkable(way.tags):
            continue
        refs = [n.ref for n in way.nodes]
        if len(refs) < 2:
            continue
        for r in refs:
            uses[r] = uses.get(r, 0) + 1
        uses[refs[0]] += 1  # endpoints count as junctions
        uses[refs[-1]] += 1

    # pass 2: walk each way, emit an edge between consecutive kept nodes
    index: dict[int, int] = {}
    lat_l, lng_l, src, dst, length = [], [], [], [], []

    def node_row(ref: int, la: float, lo: float) -> int:
        k = index.get(ref)
        if k is None:
            k = index[ref] = len(lat_l)
            lat_l.append(la)
            lng_l.append(lo)
        return k

    processor = osmium.FileProcessor(str(pbf), osmium.osm.NODE | osmium.osm.WAY).with_locations()
    for way in processor:
        if not isinstance(way, osmium.osm.Way) or not walkable(way.tags):
            continue
        pts = [(n.ref, n.location.lat, n.location.lon) for n in way.nodes if n.location.valid()]
        if len(pts) < 2:
            continue
        refs = [p[0] for p in pts]
        la = np.array([p[1] for p in pts])
        lo = np.array([p[2] for p in pts])
        seg = haversine_m(la[:-1], lo[:-1], la[1:], lo[1:])
        start = 0
        acc = 0.0
        for i in range(1, len(refs)):
            acc += seg[i - 1]
            if uses.get(refs[i], 0) >= 2 or i == len(refs) - 1:
                a = node_row(refs[start], la[start], lo[start])
                b = node_row(refs[i], la[i], lo[i])
                if a != b:
                    src.append(a)
                    dst.append(b)
                    length.append(acc)
                start, acc = i, 0.0
    src_a, dst_a = np.array(src, np.int64), np.array(dst, np.int64)
    return WalkEdges(
        node_ids=np.array(list(index.keys()), np.int64),
        lat=np.array(lat_l), lng=np.array(lng_l),
        src=np.concatenate([src_a, dst_a]), dst=np.concatenate([dst_a, src_a]),
        length_m=np.tile(np.array(length), 2),
    )


def largest_component(e: WalkEdges) -> WalkEdges:
    n = len(e.node_ids)
    g = coo_matrix((np.ones(len(e.src)), (e.src, e.dst)), shape=(n, n))
    _, labels = connected_components(g, directed=False)
    keep_label = np.bincount(labels).argmax()
    keep = labels == keep_label
    new = -np.ones(n, np.int64)
    new[keep] = np.arange(keep.sum())
    ok = keep[e.src] & keep[e.dst]
    return WalkEdges(e.node_ids[keep], e.lat[keep], e.lng[keep], new[e.src[ok]], new[e.dst[ok]],
                     e.length_m[ok])


def walk_graph_arrays(pbf: Path, m_per_min: float):
    """→ (WalkEdges of the largest component, x, y, walk_min)."""
    e = largest_component(build_walk_edges(pbf))
    x, y = to_utm(e.lat, e.lng)
    return e, np.asarray(x), np.asarray(y), e.length_m / m_per_min
