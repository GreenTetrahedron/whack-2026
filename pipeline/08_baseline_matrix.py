"""Baseline door-to-door matrix T0 (walk + public transport, weekday 08:00) → data/processed/T0.npy.

r5py in a child process with a time limit (KHATT_BASELINE_TIMEOUT_MIN, default 90). If r5py
fails or times out, the scipy fallback graph is used (same output contract).

    python pipeline/08_baseline_matrix.py [--fallback] [--exclude-feed manual] [--out T0.npy]
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from khatt.engine.catchment import WalkGraph
from khatt.geo import from_utm, to_utm
from khatt.prep import gtfs
from khatt.prep.common import INTERIM, PROCESSED, RAW, decision, require, setup, summary
from khatt.prep.fallback_matrix import build_fallback_T0


def osm_extract(bbox) -> Path:
    """The bbox clip (R5 rejects all-of-Egypt extracts); built here if 00 has not made it."""
    clip = RAW / "cairo.osm.pbf"
    full = RAW / "egypt-latest.osm.pbf"
    if not clip.exists() and full.exists():
        from khatt.prep.osm_clip import clip_highways

        clip_highways(full, clip, tuple(bbox))
    return clip


def routing_points(lat: np.ndarray, lng: np.ndarray, graph: WalkGraph | None,
                   max_m: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Where r5py starts and ends each cell's trips: the cell centroid moved onto the nearest
    node of the main connected walk network (06). Left at the centroid, R5 links some cells to a
    disconnected fragment (a park path, a pier) and cannot route them at all. Cells farther than
    max_m from that network keep their centroid. Returns (lat, lng, moved_m)."""
    if graph is None or not graph.n:
        return lat, lng, np.zeros(len(lat))
    x, y = to_utm(lat, lng)
    node, dist = graph.snap_xy(np.asarray(x), np.asarray(y))
    nlat, nlng = (np.asarray(a) for a in from_utm(graph.x[node], graph.y[node]))
    move = dist <= max_m
    return (np.where(move, nlat, lat), np.where(move, nlng, lng), np.where(move, dist, 0.0))


def r5py_worker(osm: str, feeds: list[str], hexes_path: str, params_tree: dict,
                out_path: str, max_memory: str) -> None:
    sys.argv += ["--max-memory", max_memory]  # r5py reads its JVM heap from argv, before import
    import geopandas as gpd
    import r5py

    from khatt.config import Params

    params = Params(params_tree)
    hexes = pd.read_parquet(hexes_path)
    pts = gpd.GeoDataFrame({"id": hexes["idx"].to_numpy()},
                           geometry=gpd.points_from_xy(hexes["lng"], hexes["lat"]), crs="EPSG:4326")
    net = r5py.TransportNetwork(osm, feeds)
    dep = datetime.strptime(f"{params['routing.departure_date']} {params['routing.departure_time']}",
                            "%Y-%m-%d %H:%M")
    ttm = r5py.TravelTimeMatrix(
        net, origins=pts, destinations=pts, departure=dep,
        departure_time_window=timedelta(minutes=int(params["routing.departure_time_window_min"])),
        transport_modes=[r5py.TransportMode.TRANSIT, r5py.TransportMode.WALK],
        percentiles=[50], speed_walking=float(params["walk.speed_kmh"]),
        snap_to_network=int(params["routing.r5_snap_radius_m"]),
        max_time=timedelta(minutes=int(params["spending.max_trip_min"]) * 2),
    )
    df = pd.DataFrame(ttm)
    n = len(hexes)
    T0 = np.full((n, n), float(params["routing.unreachable_min"]), np.float32)
    ok = df["travel_time"].notna()
    T0[df.loc[ok, "from_id"].to_numpy(int), df.loc[ok, "to_id"].to_numpy(int)] = \
        df.loc[ok, "travel_time"].to_numpy(float)
    np.save(out_path, T0)


def run_r5py(feeds, params, hexes: pd.DataFrame, out: Path, timeout_min: float, log) -> bool:
    osm = osm_extract(params["area.bbox"])
    if not osm.exists():
        log.warning("no OSM extract for r5py")
        return False
    csr = PROCESSED / "walk_csr.npz"
    graph = WalkGraph.load(csr) if csr.exists() else None
    if graph is None:
        log.warning("no walk graph (run 06): r5py routes from cell centroids")
    lat, lng, moved = routing_points(hexes["lat"].to_numpy(), hexes["lng"].to_numpy(), graph,
                                     float(params["routing.r5_snap_radius_m"]))
    points = INTERIM / "routing_points.parquet"
    points.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"idx": hexes["idx"].to_numpy(), "lat": lat, "lng": lng,
                  "moved_m": moved}).to_parquet(points, index=False)
    log.info("routing points: median move %.0f m, p90 %.0f m, %d cells kept at centroid",
             np.median(moved), np.percentile(moved, 90), int((moved == 0).sum()))
    max_memory = os.environ.get("R5PY_MAX_MEMORY", "10G")
    ctx = mp.get_context("spawn")
    proc = ctx.Process(target=r5py_worker, args=(str(osm), [str(f) for f in feeds],
                                                str(points), params.tree, str(out), max_memory))
    proc.start()
    proc.join(timeout_min * 60)
    if proc.is_alive():
        proc.terminate()
        proc.join()
        log.warning("r5py exceeded %.0f min — switching to the scipy fallback", timeout_min)
        decision("08_baseline_matrix", f"r5py timed out after {timeout_min:.0f} min; fallback used")
        return False
    if proc.exitcode != 0 or not out.exists():
        log.warning("r5py failed (exit %s) — switching to the scipy fallback", proc.exitcode)
        decision("08_baseline_matrix", f"r5py failed (exit {proc.exitcode}); fallback used")
        return False
    return True


def run_fallback(feeds, params, hexes: pd.DataFrame, log) -> np.ndarray:
    hm, tm = params["routing.departure_time"].split(":")
    dep_s = int(hm) * 3600 + int(tm) * 60
    window_s = int(params["routing.departure_time_window_min"]) * 60
    routes = []
    for f in feeds:
        routes += gtfs.representative_routes(gtfs.read_feed(f), dep_s, window_s,
                                             float(params["routing.default_headway_min"]))
    log.info("fallback graph: %d cells, %d route-directions", len(hexes), len(routes))
    return build_fallback_T0(hexes["h3"].tolist(), hexes["lat"].to_numpy(),
                             hexes["lng"].to_numpy(), routes, params)


def check_service(feeds, params, log) -> list[str]:
    """Share of each feed's trips running in the departure window; aborts below
    routing.min_service_share (a feed with no service there would silently become walk-only)."""
    hm, tm = params["routing.departure_time"].split(":")
    start = int(hm) * 3600 + int(tm) * 60
    end = start + int(params["routing.departure_time_window_min"]) * 60
    floor = float(params["routing.min_service_share"])
    lines, bad = [], []
    for f in feeds:
        running, total = gtfs.trips_running(gtfs.read_feed(f), start, end)
        share = running / total if total else 0.0
        lines.append(f"{f.stem}: {running}/{total} trips run in the window ({share:.0%})")
        log.info(lines[-1])
        if share < floor:
            bad.append(f.stem)
    if bad:
        raise SystemExit(f"{', '.join(bad)}: fewer than {floor:.0%} of trips run "
                         f"{params['routing.departure_time']} + {params['routing.departure_time_window_min']}"
                         " min. Move routing.departure_time inside the feeds' service hours.")
    return lines


def fill_from_fallback(T0: np.ndarray, F: np.ndarray, unreachable: float,
                       max_fill_min: float) -> tuple[np.ndarray, int, int]:
    """Patch r5py gaps with the scipy fallback.

    * cells r5py could not route at all (whole row unreachable apart from the diagonal) take
      their row and column from F;
    * any other pair r5py left unreachable takes F where F <= max_fill_min (a trip the models use);
      r5py has no route there because the cell is linked to a disconnected street fragment.
    Everything r5py did route is kept. Returns (T0, n_dead_cells, n_pairs_filled).
    """
    T0 = T0.copy()
    off_diag = (T0 >= unreachable).sum(axis=1) - (np.diag(T0) >= unreachable)
    dead = np.flatnonzero(off_diag >= T0.shape[0] - 1)
    T0[dead, :] = F[dead, :]
    T0[:, dead] = F[:, dead]
    gap = (T0 >= unreachable) & (F <= max_fill_min)
    T0[gap] = F[gap]
    return T0, len(dead), int(gap.sum())


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--fallback", action="store_true", help="skip r5py")
    ap.add_argument("--exclude-feed", default="", help="e.g. manual (backtest variant)")
    ap.add_argument("--out", default=str(PROCESSED / "T0.npy"))
    args = ap.parse_args()
    log, params, t0 = setup("08_baseline_matrix")
    hexes = pd.read_parquet(require(PROCESSED / "hexes_r8.parquet",
                                    "run pipeline/07_gtfs_clean.py")).sort_values("idx")
    exclude = {x for x in args.exclude_feed.split(",") if x}
    feeds = [f for f in sorted((INTERIM / "gtfs").glob("*.zip")) if f.stem not in exclude]
    if not feeds:
        require(INTERIM / "gtfs" / "bus.zip", "run pipeline/07_gtfs_clean.py")
    service = check_service(feeds, params, log)
    out = Path(args.out)
    timeout = float(os.environ.get("KHATT_BASELINE_TIMEOUT_MIN", "90"))
    method = "r5py"
    if args.fallback or not run_r5py(feeds, params, hexes, out, timeout, log):
        method = "scipy fallback"
        T0 = run_fallback(feeds, params, hexes, log)
    else:
        T0 = np.load(out)
    unreachable = float(params["routing.unreachable_min"])
    if method == "r5py":
        np.save(out.with_name(out.stem + "_r5py.npy"), T0)  # raw r5py, before gap filling
        r5_reach = float((T0 < unreachable).mean())
        F = run_fallback(feeds, params, hexes, log)
        T0, n_dead, n_pairs = fill_from_fallback(T0, F, unreachable,
                                                 float(params["spending.max_trip_min"]))
        method = (f"r5py ({r5_reach:.1%} pairs routed); scipy fallback for {n_dead} unroutable "
                  f"cells and {n_pairs} other pairs <= {params['spending.max_trip_min']} min")
        decision("08_baseline_matrix", method)
    np.fill_diagonal(T0, float(params["access.intracell_min"]))
    T0 = np.where(np.isfinite(T0), T0, unreachable).astype(np.float32)
    np.save(out, T0)
    meta = {"method": method, "feeds": [f.name for f in feeds], "service": service,
            "departure": f"{params['routing.departure_date']} {params['routing.departure_time']}",
            "built": time.ctime(),
            "runtime_s": round(time.perf_counter() - t0, 1)}
    out.with_suffix(".json").write_text(json.dumps(meta, indent=2))
    decision("08_baseline_matrix", f"T0 built with {method} from {meta['feeds']}")
    reach = (T0 < unreachable).mean()
    summary("08_baseline_matrix", [f"method: {method}", f"T0 shape {T0.shape}, float32",
                                   (f"reachable pairs: {reach:.1%}; median "
                                   f"{np.median(T0[T0 < unreachable]):.0f} min")], t0)


if __name__ == "__main__":
    main()
