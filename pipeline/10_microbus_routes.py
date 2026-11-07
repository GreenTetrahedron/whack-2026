"""Microbus / paratransit routes from the bus feed → data/processed/microbus_routes.parquet."""

from __future__ import annotations

import json

import geopandas as gpd
import numpy as np
from shapely.geometry import LineString

from khatt.data import collapse_repeats
from khatt.geo import cells_for_points
from khatt.prep import gtfs
from khatt.prep.common import INTERIM, PROCESSED, decision, require, setup, summary


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
    rows = []
    for r in micro:
        rid = r.route_id.split("|")[0]
        cells = cells_for_points(r.stop_lat, r.stop_lng, 8)
        idx = collapse_repeats(np.array([pos.get(c, -1) for c in cells], np.int64))
        idx = idx[idx >= 0].astype(np.int32)
        if len(idx) < 2:
            continue
        agency_id = routes.at[rid, "agency_id"] if "agency_id" in routes.columns else ""
        rows.append({"route_id": r.route_id, "agency": str(a_name.get(agency_id, agency_id)),
                     "headway_min": float(r.headway_min), "stop_hex_idx": idx,
                     "geometry": LineString(list(zip(r.stop_lng, r.stop_lat)))})
    gdf = gpd.GeoDataFrame(rows, geometry="geometry", crs="EPSG:4326")
    gdf.to_parquet(PROCESSED / "microbus_routes.parquet", index=False)
    summary("10_microbus_routes", [
        f"{len(gdf)} microbus route-directions (of {len(reps)} bus-feed route-directions)",
        f"median headway {gdf['headway_min'].median():.1f} min" if len(gdf) else "none",
        f"agencies: {gdf['agency'].value_counts().head(5).to_dict() if len(gdf) else {}}",
    ], t0)


if __name__ == "__main__":
    main()
