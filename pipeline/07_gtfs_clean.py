"""Clean GTFS feeds, add the manual feed, then finalise the cell index (CLAUDE.md §6, 07).

Writes data/interim/gtfs/{bus,metro,manual}.zip and the processed files that depend on the final
index: hex_index.json, hexes_r8.parquet, streets.parquet/.geojson, r9_near.parquet.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from datetime import datetime

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import Point

from khatt.config import ROOT
from khatt.contracts import check_processed
from khatt.geo import cells_for_points
from khatt.prep import gtfs
from khatt.prep.common import INTERIM, PROCESSED, RAW, decision, require, setup, summary
from khatt.prep.streets import finalize_dtypes

FEEDS = {"bus": RAW / "gtfs_bus.zip", "metro": RAW / "gtfs_metro.zip"}
VALIDATOR = RAW / "gtfs-validator.jar"


def validate(name, path, log) -> None:
    out = INTERIM / "validation" / name
    if not (shutil.which("java") and VALIDATOR.exists()):
        log.info("gtfs-validator skipped for %s (needs java + %s)", name, VALIDATOR)
        return
    out.mkdir(parents=True, exist_ok=True)
    subprocess.run(["java", "-jar", str(VALIDATOR), "-i", str(path), "-o", str(out)], check=False)


def clean_feed(name, path, params, log) -> tuple[gtfs.Feed, dict]:
    feed = gtfs.read_feed(path)
    stats = {}
    validate(name, path, log)
    bad = gtfs.non_monotonic_trips(feed)
    gtfs.drop_trips(feed, bad)
    stats["dropped_trips"] = len(bad)
    swapped, outside = gtfs.fix_stop_coordinates(feed, params["area.bbox"])
    stats["swapped_stops"], stats["stops_outside_bbox"] = swapped, outside
    if "frequencies.txt" in feed and "exact_times" in feed["frequencies.txt"]:
        stats["headway_based"] = int((feed["frequencies.txt"]["exact_times"].isin(["", "0"])).sum())
    return feed, stats


def main() -> None:
    log, params, t0 = setup("07_gtfs_clean")
    day = datetime.strptime(params["routing.departure_date"], "%Y-%m-%d").date()
    out_dir = INTERIM / "gtfs"
    out_dir.mkdir(parents=True, exist_ok=True)
    cleaned: dict[str, gtfs.Feed] = {}
    notes = []
    for name, path in FEEDS.items():
        require(path, f"run pipeline/00_download.py --only gtfs_{name}")
        feed, stats = clean_feed(name, path, params, log)
        notes.append(f"{name}: {stats}")
        if not gtfs.has_service_on(feed, day):
            log.warning("%s feed has no service on %s (real feed dates are not changed)", name, day)
            decision("07_gtfs_clean", f"{name} feed has no service on {day}")
        cleaned[name] = feed

    # metro routes duplicated in the bus feed → drop from bus
    bus_routes = cleaned["bus"]["routes.txt"]
    dup = bus_routes.loc[pd.to_numeric(bus_routes["route_type"], errors="coerce") == 1, "route_id"]
    if len(dup):
        gtfs.drop_routes(cleaned["bus"], dup)
        decision("07_gtfs_clean", f"dropped {len(dup)} metro routes from the bus feed (in metro feed)")

    manual, warnings = gtfs.manual_feed(ROOT / "config" / "manual_lines.yaml",
                                        params["routing.departure_date"])
    for w in warnings:
        log.warning(w)
        decision("07_gtfs_clean", w)
    if manual is not None:
        cleaned["manual"] = manual
    for name, feed in cleaned.items():
        prefixed = gtfs.prefix_ids(feed, f"{name}:") if name != "manual" else feed
        gtfs.write_feed(prefixed, out_dir / f"{name}.zip")

    # ---------------- finalise the cell index ----------------
    grid = pd.read_parquet(require(INTERIM / "spending_r8.parquet", "run pipeline/03_spending.py"))
    streets = pd.read_parquet(require(INTERIM / "streets_raw.parquet",
                                      "run pipeline/05_streets.py"))
    stop_cells = set()
    for feed in cleaned.values():
        s = feed["stops.txt"]
        stop_cells |= set(cells_for_points(pd.to_numeric(s["stop_lat"]),
                                           pd.to_numeric(s["stop_lon"]), 8))
    min_pop = float(params["area.min_cell_pop"])
    keep = ((grid["pop"] > min_pop) | grid["h3"].isin(set(streets["h3_r8"]))
            | grid["h3"].isin(stop_cells))
    dropped_pop = grid.loc[~keep, "pop"].sum() / max(grid["pop"].sum(), 1)
    decision("07_gtfs_clean", f"cells need pop > {min_pop:g} (or a street/stop); "
             f"{dropped_pop:.2%} of population is in dropped cells")
    hexes = grid[keep].sort_values("h3").reset_index(drop=True)
    hexes.insert(1, "idx", np.arange(len(hexes), dtype=np.int32))
    hex_ids = hexes["h3"].tolist()
    pos = {h: k for k, h in enumerate(hex_ids)}
    streets = streets[streets["h3_r8"].isin(pos)].reset_index(drop=True)
    streets["street_id"] = np.arange(len(streets))
    streets = finalize_dtypes(streets, pos)

    pop9 = pd.read_parquet(require(INTERIM / "pop_r9.parquet", "run pipeline/02_population.py"))
    import h3

    pop9["parent"] = [h3.cell_to_parent(c, 8) for c in pop9["h3"]]
    r9 = pop9[pop9["parent"].isin(pos)].copy()
    r9["parent_r8_idx"] = r9["parent"].map(pos).astype("int32")
    r9 = r9[["h3", "pop", "parent_r8_idx"]].reset_index(drop=True)
    r9["pop"] = r9["pop"].astype("float32")

    # all mapped places (any category, incl. "other"): stand-in for jobs in the zones model
    places = pd.read_parquet(require(INTERIM / "shops.parquet", "run pipeline/04_shops.py"),
                             columns=["lat", "lng"])
    place_cells = pd.Series(cells_for_points(places["lat"], places["lng"], 8)).value_counts()
    hexes["n_places"] = hexes["h3"].map(place_cells).fillna(0).astype("int32")
    kept = int(hexes["n_places"].sum())
    decision("07_gtfs_clean", f"n_places: {kept:,} of {len(places):,} mapped places fall in kept "
             "cells (jobs stand-in for the station zones model)")

    cols = ["h3", "idx", "lat", "lng", "governorate", "pop", "wealth", "spend_daily",
            "spend_comp", "spend_eat", "n_places"]
    (PROCESSED / "hex_index.json").write_text(json.dumps(hex_ids))
    hexes[cols].to_parquet(PROCESSED / "hexes_r8.parquet", index=False)
    streets.to_parquet(PROCESSED / "streets.parquet", index=False)
    gpd.GeoDataFrame(streets, geometry=[Point(x, y) for x, y in zip(streets.lng, streets.lat)],
                     crs="EPSG:4326").to_file(PROCESSED / "streets.geojson", driver="GeoJSON")
    r9.to_parquet(PROCESSED / "r9_near.parquet", index=False)
    problems = [p for p in check_processed(PROCESSED) if "T0" not in p]
    if problems:
        log.warning("contract problems: %s", problems)
    summary("07_gtfs_clean", notes[:2] + [
        f"cells kept: {len(hexes)} of {len(grid)} (pop>{min_pop:g} | street | stop)",
        f"streets: {len(streets)}; r9 cells: {len(r9)}; manual feed: {manual is not None}",
    ], t0)


if __name__ == "__main__":
    main()
