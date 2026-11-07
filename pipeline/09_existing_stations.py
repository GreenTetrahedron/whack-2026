"""Existing metro/LRT/monorail stations + 10-min walking catchments (CLAUDE.md §6, 09).

Writes data/processed/existing_stations.parquet and existing_lines.geojson.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from khatt.data import _load_walk
from khatt.geo import cells_for_points, to_utm
from khatt.prep import gtfs
from khatt.prep.catchments import station_catchment_pop
from khatt.prep.common import INTERIM, PROCESSED, decision, require, setup, summary


def rail_stations(feed: gtfs.Feed) -> tuple[pd.DataFrame, list[dict]]:
    routes = feed["routes.txt"]
    rail = routes[[gtfs.route_mode(t) == "metro" for t in routes["route_type"]]]
    trips = feed["trips.txt"][feed["trips.txt"]["route_id"].isin(rail["route_id"])]
    st = feed["stop_times.txt"][feed["stop_times.txt"]["trip_id"].isin(trips["trip_id"])].copy()
    st["seq"] = pd.to_numeric(st["stop_sequence"], errors="coerce")
    stops = feed["stops.txt"].set_index("stop_id")
    # station = parent_station when given
    parent = stops["parent_station"] if "parent_station" in stops else pd.Series(dtype=str)
    rows, features = [], []
    name_col = "route_short_name" if "route_short_name" in rail else "route_id"
    for _, r in rail.iterrows():
        rt = trips[trips["route_id"] == r["route_id"]]
        if rt.empty:
            continue
        seqs = st[st["trip_id"].isin(rt["trip_id"])].sort_values(["trip_id", "seq"])
        longest = seqs.groupby("trip_id").size().idxmax()
        order = seqs[seqs["trip_id"] == longest]["stop_id"].tolist()
        coords = []
        for sid in order:
            pid = parent.get(sid, "") if len(parent) else ""
            key = pid if pid and pid in stops.index else sid
            s = stops.loc[key]
            lat, lng = float(s["stop_lat"]), float(s["stop_lon"])
            coords.append([lng, lat])
            rows.append({"stop_id": key, "line": str(r[name_col] or r["route_id"]),
                         "name": str(s.get("stop_name", key)), "lat": lat, "lng": lng})
        features.append({"type": "Feature", "properties": {"line": str(r[name_col])},
                         "geometry": {"type": "LineString", "coordinates": coords}})
    df = pd.DataFrame(rows).drop_duplicates(subset=["stop_id", "line"]).reset_index(drop=True)
    return df, features


def main() -> None:
    _log, params, t0 = setup("09_existing_stations")
    frames, features = [], []
    for name in ("metro", "manual", "bus"):
        f = INTERIM / "gtfs" / f"{name}.zip"
        if f.exists():
            df, feats = rail_stations(gtfs.read_feed(f))
            frames.append(df)
            features += feats
    if not frames:
        require(INTERIM / "gtfs" / "metro.zip", "run pipeline/07_gtfs_clean.py")
    st = pd.concat(frames, ignore_index=True).drop_duplicates(subset=["stop_id", "line"])
    hex_ids = json.loads(require(PROCESSED / "hex_index.json", "run 07").read_text())
    pos = {h: k for k, h in enumerate(hex_ids)}
    hexes = pd.read_parquet(PROCESSED / "hexes_r8.parquet").sort_values("idx")
    st["h3_r8"] = cells_for_points(st["lat"], st["lng"], 8)
    from scipy.spatial import cKDTree

    hx, hy = to_utm(hexes["lat"], hexes["lng"])
    sx, sy = to_utm(st["lat"], st["lng"])
    _, nearest = cKDTree(np.column_stack([hx, hy])).query(np.column_stack([sx, sy]))
    st["hex_idx"] = [pos.get(c, int(n)) for c, n in zip(st["h3_r8"], nearest)]
    st["hex_idx"] = st["hex_idx"].astype("int32")

    r9 = pd.read_parquet(require(PROCESSED / "r9_near.parquet", "run 07"))
    walk = _load_walk(PROCESSED, r9)
    if walk is None:
        require(PROCESSED / "walk_graph.pkl", "run pipeline/06_walk_graph.py")
    pop10, snap_m = station_catchment_pop(walk, r9["pop"].to_numpy(), st["lat"], st["lng"],
                                          float(params["footfall.catchment_min"]))
    st["catchment_pop10"] = pop10.astype("float32")
    far = int((snap_m > float(params["line.max_snap_m"])).sum())
    if far:
        decision("09_existing_stations", f"{far} existing stations > {params['line.max_snap_m']} m "
                 "from the walk network")
    st = st[["stop_id", "line", "name", "lat", "lng", "h3_r8", "hex_idx", "catchment_pop10"]]
    st.to_parquet(PROCESSED / "existing_stations.parquet", index=False)
    (PROCESSED / "existing_lines.geojson").write_text(json.dumps(
        {"type": "FeatureCollection", "features": features}))
    summary("09_existing_stations", [
        f"{len(st)} stations on lines {sorted(st['line'].unique())}",
        f"catchment population (10 min, no double counting): {st['catchment_pop10'].sum():,.0f}",
        f"stations > {params['line.max_snap_m']} m from walk network: {far}",
    ], t0)


if __name__ == "__main__":
    main()
