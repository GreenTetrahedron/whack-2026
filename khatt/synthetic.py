"""Synthetic Cairo-shaped `data/processed` set for tests and for front-end work before the
real pipeline has run. Every file follows the CLAUDE.md §5 contracts; the numbers are invented.

    python -m khatt.synthetic data/synthetic --rings 31
"""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import geopandas as gpd
import h3
import networkx as nx
import numpy as np
import pandas as pd
from shapely.geometry import LineString, Point

from khatt.config import Params
from khatt.engine.catchment import WalkGraph
from khatt.geo import cells_for_points, from_utm, to_utm
from khatt.prep.catchments import station_catchment_pop
from khatt.prep.fallback_matrix import Route, build_fallback_T0
from khatt.prep.spending import allocate, normalise_wealth

CENTER_LAT, CENTER_LNG = 30.045, 31.235  # Downtown Cairo
CELL_SPACING_KM = 0.8  # res-8 centre-to-centre
PER_CAPITA = {  # EGP/person/yr, invented
    "Cairo": {"daily": 9000.0, "comp": 2600.0, "eat": 1800.0},
    "Giza": {"daily": 8000.0, "comp": 2200.0, "eat": 1400.0},
    "Qalyubia": {"daily": 7000.0, "comp": 1700.0, "eat": 900.0},
}


def _governorate(lat, lng):
    return np.where(lng < 31.215, "Giza", np.where(lat > 30.10, "Qalyubia", "Cairo"))


def _line_stations(lat0, lng0, lat1, lng1, spacing_km=1.2):
    x0, y0 = to_utm(lat0, lng0)
    x1, y1 = to_utm(lat1, lng1)
    length = float(np.hypot(x1 - x0, y1 - y0))
    n = max(2, round(length / (spacing_km * 1000)) + 1)
    t = np.linspace(0, 1, n)
    lat, lng = from_utm(x0 + t * (x1 - x0), y0 + t * (y1 - y0))
    return np.asarray(lat), np.asarray(lng)


def make_synthetic_world(out_dir: Path | str, *, rings: int = 10, seed: int = 0,
                         n_streets: int | None = None, n_microbus: int | None = None,
                         walk_spacing_m: float = 150.0, params: Params | None = None) -> Path:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    params = params or Params.load(calibrated=False)
    rng = np.random.default_rng(seed)

    # ---------- hexes ----------
    cells = sorted(h3.grid_disk(h3.latlng_to_cell(CENTER_LAT, CENTER_LNG, 8), rings))
    N = len(cells)
    pos = {c: k for k, c in enumerate(cells)}
    lat = np.array([h3.cell_to_latlng(c)[0] for c in cells])
    lng = np.array([h3.cell_to_latlng(c)[1] for c in cells])
    x, y = to_utm(lat, lng)
    cx, cy = to_utm(CENTER_LAT, CENTER_LNG)
    r_km = np.hypot(x - cx, y - cy) / 1000
    pop = (45000 * np.exp(-r_km / 5) * rng.lognormal(0, 0.35, N) + 500) * 0.737
    gov = _governorate(lat, lng)
    wealth = normalise_wealth(rng.lognormal(0, 0.3, N) * (1 + 0.5 * np.exp(-r_km / 4)), pop)
    spend = allocate(pop, wealth, gov, PER_CAPITA)
    hexes = pd.DataFrame({
        "h3": cells, "idx": np.arange(N, dtype=np.int32), "lat": lat, "lng": lng,
        "governorate": gov, "pop": pop.astype(np.float32), "wealth": wealth,
        "spend_daily": spend["daily"], "spend_comp": spend["comp"], "spend_eat": spend["eat"],
        # deterministic (no rng draw) so the rest of the synthetic world is unchanged
        "n_places": np.round(300 * np.exp(-r_km / 3) + pop / 2000).astype(np.int32),
    })

    # ---------- existing metro lines ----------
    reach_km = rings * CELL_SPACING_KM * 0.85
    dlat, dlng = reach_km / 111.0, reach_km / 96.0
    line_ends = {
        "1": (CENTER_LAT - dlat, CENTER_LNG - 0.2 * dlng, CENTER_LAT + dlat, CENTER_LNG + 0.1 * dlng),
        "2": (CENTER_LAT - 0.7 * dlat, CENTER_LNG - 0.7 * dlng, CENTER_LAT + 0.8 * dlat,
              CENTER_LNG + 0.3 * dlng),
        "3": (CENTER_LAT + 0.1 * dlat, CENTER_LNG - dlng, CENTER_LAT - 0.1 * dlat, CENTER_LNG + dlng),
    }
    metro_routes, st_rows, features = [], [], []
    for line, (a, b, c, d) in line_ends.items():
        s_lat, s_lng = _line_stations(a, b, c, d)
        metro_routes.append(Route(f"metro:L{line}", "metro", 4.0, list(s_lat), list(s_lng),
                                  speed_kmh=35.0, bidirectional=True))
        for k, (la, ln) in enumerate(zip(s_lat, s_lng)):
            st_rows.append({"stop_id": f"metro:L{line}-{k}", "line": f"Line {line}",
                            "name": f"L{line} station {k}", "lat": la, "lng": ln})
        features.append({
            "type": "Feature", "properties": {"line": f"Line {line}"},
            "geometry": {"type": "LineString", "coordinates": [[float(o), float(t)]
                         for t, o in zip(s_lat, s_lng)]},
        })

    # ---------- microbus routes ----------
    n_microbus = n_microbus or max(20, N // 25)
    mb_rows = []
    mb_routes = []
    attempts = 0
    while len(mb_rows) < n_microbus and attempts < n_microbus * 20:
        attempts += 1
        a, b = rng.choice(N, 2, replace=False)
        path = h3.grid_path_cells(cells[a], cells[b])
        if not (8 <= len(path) <= 45) or any(p not in pos for p in path):
            continue
        idx = [pos[p] for p in path]
        headway = float(rng.uniform(3, 15))
        rid = f"bus:mb{len(mb_rows)}"
        mb_rows.append({
            "route_id": rid, "agency": "Microbus", "headway_min": headway,
            "stop_hex_idx": np.array(idx, np.int32),
            "geometry": LineString([(lng[k], lat[k]) for k in idx]),
        })
        mb_routes.append(Route(rid, "microbus", headway, list(lat[idx]), list(lng[idx]),
                               speed_kmh=15.0, bidirectional=True))
    microbus = gpd.GeoDataFrame(mb_rows, geometry="geometry", crs="EPSG:4326")

    # ---------- baseline matrix (scipy fallback) ----------
    T0 = build_fallback_T0(cells, lat, lng, metro_routes + mb_routes, params)

    # ---------- streets ----------
    n_streets = n_streets or max(30, N // 10)
    weights = pop ** 1.2
    chosen = rng.choice(N, n_streets, replace=False, p=weights / weights.sum())
    jitter = rng.normal(0, 60, (n_streets, 2))
    s_lat, s_lng = from_utm(x[chosen] + jitter[:, 0], y[chosen] + jitter[:, 1])
    s_lat, s_lng = np.asarray(s_lat), np.asarray(s_lng)
    centrality = 1 + 2 * np.exp(-r_km[chosen] / 2.5)
    n_daily = rng.poisson(25, n_streets) + 5
    n_comp = rng.poisson(10 * centrality) + 1
    n_eat = rng.poisson(8 * centrality) + 2
    total = n_daily + n_comp + n_eat
    r8 = cells_for_points(s_lat, s_lng, 8)
    hex_idx = np.array([pos.get(c, chosen[k]) for k, c in enumerate(r8)], np.int32)
    r8 = np.array([cells[k] for k in hex_idx], dtype=object)
    streets = pd.DataFrame({
        "street_id": np.arange(n_streets, dtype=np.int32),
        "name": [f"Cluster {k}" for k in range(n_streets)],
        "lat": s_lat, "lng": s_lng, "h3_r8": r8,
        "h3_r9": cells_for_points(s_lat, s_lng, 9), "hex_idx": hex_idx,
        "n_daily": n_daily.astype(np.int32), "n_comp": n_comp.astype(np.int32),
        "n_eat": n_eat.astype(np.int32),
        "n_unbranded": rng.binomial(total, 0.8).astype(np.int32),
        "footprint_m2": (total * rng.lognormal(4.5, 0.3, n_streets)).astype(np.float32),
        "A_daily": n_daily.astype(np.float32), "A_comp": n_comp.astype(np.float32),
        "A_eat": n_eat.astype(np.float32),
    })

    # ---------- walk graph (lattice) ----------
    radius_m = (rings + 0.5) * CELL_SPACING_KM * 1000
    grid = np.arange(-radius_m, radius_m + 1, walk_spacing_m)
    gx, gy = np.meshgrid(grid, grid, indexing="ij")
    inside = np.hypot(gx, gy) <= radius_m
    ids = -np.ones(gx.shape, np.int64)
    ids[inside] = np.arange(inside.sum())
    node_x = (gx[inside] + cx).ravel()
    node_y = (gy[inside] + cy).ravel()
    src, dst = [], []
    for di, dj in [(1, 0), (0, 1)]:
        a = ids[: ids.shape[0] - di, : ids.shape[1] - dj]
        b = ids[di:, dj:]
        ok = (a >= 0) & (b >= 0)
        src += list(a[ok]) + list(b[ok])
        dst += list(b[ok]) + list(a[ok])
    walk_min = np.full(len(src), walk_spacing_m / params.walk_m_per_min)
    n_lat, n_lng = from_utm(node_x, node_y)
    node_r9_cells = cells_for_points(n_lat, n_lng, 9)

    G = nx.MultiDiGraph(crs="EPSG:32636")
    for k in range(len(node_x)):
        G.add_node(k, x=float(node_x[k]), y=float(node_y[k]))
    for u, v, w in zip(src, dst, walk_min):
        G.add_edge(int(u), int(v), length=float(walk_spacing_m), walk_min=float(w))
    with open(out / "walk_graph.pkl", "wb") as fh:
        pickle.dump(G, fh, protocol=pickle.HIGHEST_PROTOCOL)
    pd.DataFrame({"node_id": np.arange(len(node_x)), "h3_r9": node_r9_cells}).to_parquet(
        out / "walk_nodes_r9.parquet", index=False)

    # ---------- r9 cells ----------
    r9_cells, r9_pop, r9_parent = [], [], []
    for k, c in enumerate(cells):
        kids = sorted(h3.cell_to_children(c, 9))
        r9_cells += kids
        r9_pop += [pop[k] / len(kids)] * len(kids)
        r9_parent += [k] * len(kids)
    r9 = pd.DataFrame({"h3": r9_cells, "pop": np.array(r9_pop, np.float32),
                       "parent_r8_idx": np.array(r9_parent, np.int32)})
    r9_row = {c: k for k, c in enumerate(r9_cells)}
    node_r9 = np.array([r9_row.get(c, -1) for c in node_r9_cells], np.int64)
    walk = WalkGraph.from_edges(node_x, node_y, np.array(src), np.array(dst), walk_min, node_r9)

    # ---------- existing stations ----------
    existing = pd.DataFrame(st_rows)
    existing["h3_r8"] = cells_for_points(existing["lat"], existing["lng"], 8)
    ex_x, ex_y = to_utm(existing["lat"], existing["lng"])
    _, nearest = cKDTree_query(x, y, ex_x, ex_y)
    existing["hex_idx"] = np.array([pos.get(c, nearest[k]) for k, c in
                                    enumerate(existing["h3_r8"])], np.int32)
    pop10, _ = station_catchment_pop(walk, r9["pop"].to_numpy(), existing["lat"], existing["lng"],
                                     float(params["footfall.catchment_min"]))
    existing["catchment_pop10"] = pop10

    # ---------- write ----------
    (out / "hex_index.json").write_text(json.dumps(cells))
    hexes.to_parquet(out / "hexes_r8.parquet", index=False)
    streets.to_parquet(out / "streets.parquet", index=False)
    gpd.GeoDataFrame(streets, geometry=[Point(a, b) for a, b in zip(s_lng, s_lat)],
                     crs="EPSG:4326").to_file(out / "streets.geojson", driver="GeoJSON")
    np.save(out / "T0.npy", T0)
    r9.to_parquet(out / "r9_near.parquet", index=False)
    existing.to_parquet(out / "existing_stations.parquet", index=False)
    microbus.to_parquet(out / "microbus_routes.parquet", index=False)
    (out / "existing_lines.geojson").write_text(json.dumps(
        {"type": "FeatureCollection", "features": features}))
    (out / "SYNTHETIC").write_text("Invented data for tests and front-end work. Not Cairo.\n")
    return out


def cKDTree_query(x, y, qx, qy):
    from scipy.spatial import cKDTree

    tree = cKDTree(np.column_stack([x, y]))
    return tree.query(np.column_stack([np.atleast_1d(qx), np.atleast_1d(qy)]))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("out", nargs="?", default="data/synthetic")
    ap.add_argument("--rings", type=int, default=31, help="31 rings ≈ 2,977 cells")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--walk-spacing", type=float, default=200.0)
    args = ap.parse_args()
    out = make_synthetic_world(args.out, rings=args.rings, seed=args.seed,
                               walk_spacing_m=args.walk_spacing)
    print(f"Synthetic world written to {out}")


if __name__ == "__main__":
    main()
