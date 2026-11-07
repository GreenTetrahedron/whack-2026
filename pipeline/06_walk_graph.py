"""OSMnx walk graph → data/processed/walk_graph.pkl + walk_nodes_r9.parquet (CLAUDE.md §6, 06)."""

from __future__ import annotations

import pickle

import numpy as np
import pandas as pd

from khatt.geo import cells_for_points, haversine_m
from khatt.prep.common import PROCESSED, decision, setup, summary

CHECK_POINTS = {"Zamalek": (30.0626, 31.2197), "Roda": (30.0170, 31.2290)}
CHECK_RADIUS_M = 300


def main() -> None:
    import osmnx as ox

    log, params, t0 = setup("06_walk_graph")
    ox.settings.requests_timeout = 600
    ox.settings.overpass_settings = '[out:json][timeout:{timeout}]{maxsize}'
    ox.settings.overpass_memory = 2 * 1024**3
    G = ox.graph_from_bbox(tuple(params["area.bbox"]), network_type="walk")
    log.info("downloaded: %d nodes, %d edges", len(G), G.number_of_edges())
    G = ox.truncate.largest_component(G, strongly=True)
    lat = np.array([d["y"] for _, d in G.nodes(data=True)])
    lng = np.array([d["x"] for _, d in G.nodes(data=True)])
    node_ids = np.array(list(G.nodes))
    for name, (la, lo) in CHECK_POINTS.items():
        if haversine_m(lat, lng, la, lo).min() > CHECK_RADIUS_M:
            log.warning("%s is not in the largest strongly connected walk component", name)
            decision("06_walk_graph", f"{name} missing from the walk graph component")
    G = ox.project_graph(G, to_crs="EPSG:32636")
    m_per_min = params.walk_m_per_min
    for _, _, d in G.edges(data=True):
        d["walk_min"] = float(d["length"]) / m_per_min
    with open(PROCESSED / "walk_graph.pkl", "wb") as fh:
        pickle.dump(G, fh, protocol=pickle.HIGHEST_PROTOCOL)
    nodes = pd.DataFrame({"node_id": node_ids,
                          "h3_r9": cells_for_points(lat, lng, int(params["area.h3_res_fine"]))})
    nodes.to_parquet(PROCESSED / "walk_nodes_r9.parquet", index=False)
    (PROCESSED / "walk_csr.npz").unlink(missing_ok=True)  # rebuilt from the new graph on load
    summary("06_walk_graph", [f"{len(G)} nodes, {G.number_of_edges()} edges (largest SCC)",
                              f"walk speed {params['walk.speed_kmh']} km/h",
                              f"res-9 cells touched: {nodes['h3_r9'].nunique()}"], t0)


if __name__ == "__main__":
    main()
