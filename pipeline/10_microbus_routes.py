"""Microbus / paratransit routes from the bus feed → data/processed/microbus_routes.parquet."""

from __future__ import annotations

import json

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import LineString, Point

from khatt.data import collapse_repeats
from khatt.geo import cells_for_points, to_utm
from khatt.prep import gtfs
from khatt.prep.common import INTERIM, PROCESSED, decision, require, setup, summary


def load_shapes(feed) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """shape_id → (lat, lng) arrays in shape_pt_sequence order."""
    shapes = feed.get("shapes.txt")
    if shapes is None or shapes.empty:
        return {}
    s = shapes.assign(seq=pd.to_numeric(shapes["shape_pt_sequence"], errors="coerce"),
                      lat=pd.to_numeric(shapes["shape_pt_lat"], errors="coerce"),
                      lng=pd.to_numeric(shapes["shape_pt_lon"], errors="coerce"))
    s = s.dropna(subset=["seq", "lat", "lng"]).sort_values(["shape_id", "seq"])
    return {sid: (g["lat"].to_numpy(), g["lng"].to_numpy()) for sid, g in s.groupby("shape_id")}


def route_line(stop_lat, stop_lng, shape, max_gap_m: float) -> tuple[LineString, float | None]:
    """The road trace if every stop lies within max_gap_m of it, else straight stop-to-stop
    segments. Returns (line, worst stop-to-trace distance in m or None if no trace)."""
    stops = LineString(list(zip(stop_lng, stop_lat)))
    if shape is None or len(shape[0]) < 2:
        return stops, None
    sx, sy = to_utm(shape[0], shape[1])
    trace = LineString(list(zip(np.atleast_1d(sx), np.atleast_1d(sy))))
    px, py = to_utm(np.asarray(stop_lat), np.asarray(stop_lng))
    worst = max(trace.distance(Point(x, y)) for x, y in zip(np.atleast_1d(px), np.atleast_1d(py)))
    if worst > max_gap_m:
        return stops, float(worst)
    return LineString(list(zip(shape[1], shape[0]))), float(worst)


def main() -> None:
    _log, params, t0 = setup("10_microbus_routes")
    feed = gtfs.read_feed(require(INTERIM / "gtfs" / "bus.zip", "run pipeline/07_gtfs_clean.py"))
    hm, tm = params["routing.departure_time"].split(":")
    reps = gtfs.representative_routes(feed, int(hm) * 3600 + int(tm) * 60,
                                      int(params["routing.departure_time_window_min"]) * 60,
                                      float(params["routing.default_headway_min"]))
    micro = [r for r in reps if r.mode == "microbus"]
    if not micro:
        decision("10_microbus_routes", "no agency matched the microbus pattern; all non-rail "
                 "bus-feed routes treated as microbus/paratransit")
        micro = [r for r in reps if r.mode != "metro"]
    agencies = feed.get("agency.txt")
    routes = feed["routes.txt"].set_index("route_id")
    a_name = dict(zip(agencies["agency_id"], agencies["agency_name"])) if agencies is not None \
        else {}
    pos = {h: k for k, h in enumerate(json.loads((PROCESSED / "hex_index.json").read_text()))}
    shapes = load_shapes(feed)
    max_gap = float(params["informal.shape_max_stop_gap_m"])
    n_traced, n_far, gaps = 0, 0, []
    rows = []
    for r in micro:
        rid = r.route_id.split("|")[0]
        cells = cells_for_points(r.stop_lat, r.stop_lng, 8)
        idx = collapse_repeats(np.array([pos.get(c, -1) for c in cells], np.int64))
        idx = idx[idx >= 0].astype(np.int32)
        if len(idx) < 2:
            continue
        agency_id = routes.at[rid, "agency_id"] if "agency_id" in routes.columns else ""
        line, worst = route_line(r.stop_lat, r.stop_lng, shapes.get(r.shape_id), max_gap)
        if worst is not None:
            gaps.append(worst)
            n_traced += worst <= max_gap
            n_far += worst > max_gap
        rows.append({"route_id": r.route_id, "agency": str(a_name.get(agency_id, agency_id)),
                     "headway_min": float(r.headway_min), "stop_hex_idx": idx,
                     "stop_lat": np.asarray(r.stop_lat, np.float64),
                     "stop_lng": np.asarray(r.stop_lng, np.float64),
                     "geometry": line})
    gdf = gpd.GeoDataFrame(rows, geometry="geometry", crs="EPSG:4326")
    gdf.to_parquet(PROCESSED / "microbus_routes.parquet", index=False)
    decision("10_microbus_routes", f"route lines follow the GTFS road trace (shapes.txt) for "
             f"{n_traced} of {len(gdf)} routes; {n_far} traces had a stop > {max_gap:.0f} m away "
             f"and {len(gdf) - n_traced - n_far} had none, so those use straight stop-to-stop "
             "lines. The model uses the stops either way (stop_lat/stop_lng, stop_hex_idx)")
    summary("10_microbus_routes", [
        f"{len(gdf)} microbus route-directions (of {len(reps)} bus-feed route-directions)",
        f"median headway {gdf['headway_min'].median():.1f} min" if len(gdf) else "none",
        (f"drawn along road traces: {n_traced}; straight (trace off by > {max_gap:.0f} m): "
         f"{n_far}; median worst stop-to-trace gap "
         f"{np.median(gaps) if gaps else float('nan'):.0f} m"),
        f"agencies: {gdf['agency'].value_counts().head(5).to_dict() if len(gdf) else {}}",
    ], t0)


if __name__ == "__main__":
    main()
