"""Walk graph → data/processed/walk_csr.npz + walk_nodes_r9.parquet (CLAUDE.md §6, 06).

Default: built locally from data/raw/cairo.osm.pbf (pyosmium, OSMnx "walk" filter). The public
Overpass API is often at capacity for a bbox this size; `--overpass` uses OSMnx's
graph_from_bbox instead and writes walk_graph.pkl as in the original spec.
"""

from __future__ import annotations

import argparse
import pickle

import numpy as np
import pandas as pd

from khatt.engine.catchment import WalkGraph
from khatt.geo import cells_for_points, haversine_m
from khatt.prep.common import PROCESSED, RAW, decision, require, setup, summary

CHECK_POINTS = {"Zamalek": (30.0626, 31.2197), "Roda": (30.0170, 31.2290)}
CHECK_RADIUS_M = 300


def check_islands(lat, lng, log) -> None:
    for name, (la, lo) in CHECK_POINTS.items():
        if haversine_m(lat, lng, la, lo).min() > CHECK_RADIUS_M:
            log.warning("%s is not in the largest connected walk component", name)
            decision("06_walk_graph", f"{name} missing from the walk graph component")


def from_pbf(params, log):
    from khatt.prep.osm_clip import clip_highways
    from khatt.prep.walk_osm import walk_graph_arrays

    pbf = RAW / "cairo.osm.pbf"
    if not pbf.exists():
        full = require(RAW / "egypt-latest.osm.pbf", "run pipeline/00_download.py --only osm_pbf")
        clip_highways(full, pbf, tuple(params["area.bbox"]))
    e, x, y, walk_min = walk_graph_arrays(pbf, params.walk_m_per_min)
    check_islands(e.lat, e.lng, log)
    graph = WalkGraph.from_edges(x, y, e.src, e.dst, walk_min, np.full(len(x), -1), e.node_ids)
    graph.save(PROCESSED / "walk_csr.npz")
    (PROCESSED / "walk_graph.pkl").unlink(missing_ok=True)  # would be stale
    return e.node_ids, e.lat, e.lng, len(e.src)


def from_overpass(params, log):
    import osmnx as ox

    ox.settings.requests_timeout = 600
    G = ox.graph_from_bbox(tuple(params["area.bbox"]), network_type="walk")
    G = ox.truncate.largest_component(G, strongly=True)
    lat = np.array([d["y"] for _, d in G.nodes(data=True)])
    lng = np.array([d["x"] for _, d in G.nodes(data=True)])
    node_ids = np.array(list(G.nodes))
    check_islands(lat, lng, log)
    G = ox.project_graph(G, to_crs="EPSG:32636")
    for _, _, d in G.edges(data=True):
        d["walk_min"] = float(d["length"]) / params.walk_m_per_min
    with open(PROCESSED / "walk_graph.pkl", "wb") as fh:
        pickle.dump(G, fh, protocol=pickle.HIGHEST_PROTOCOL)
    (PROCESSED / "walk_csr.npz").unlink(missing_ok=True)  # rebuilt from the pickle on load
    return node_ids, lat, lng, G.number_of_edges()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--overpass", action="store_true", help="download via OSMnx/Overpass")
    args = ap.parse_args()
    log, params, t0 = setup("06_walk_graph")
    source = "Overpass (OSMnx)" if args.overpass else "local OSM extract (pyosmium)"
    node_ids, lat, lng, n_edges = (from_overpass if args.overpass else from_pbf)(params, log)
    nodes = pd.DataFrame({"node_id": node_ids,
                          "h3_r9": cells_for_points(lat, lng, int(params["area.h3_res_fine"]))})
    nodes.to_parquet(PROCESSED / "walk_nodes_r9.parquet", index=False)
    summary("06_walk_graph", [f"source: {source}",
                              f"{len(node_ids)} nodes, {n_edges} directed edges (largest component)",
                              f"walk speed {params['walk.speed_kmh']} km/h",
                              f"res-9 cells touched: {nodes['h3_r9'].nunique()}"], t0)


if __name__ == "__main__":
    main()
