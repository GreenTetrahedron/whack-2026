"""GTFS helpers: read/write, checks, ID prefixing, manual feeds, representative routes (§6, 07)."""

from __future__ import annotations

import io
import re
import zipfile
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from khatt.geo import haversine_m
from khatt.prep.fallback_matrix import Route

ID_COLUMNS = {
    "agency.txt": ["agency_id"],
    "routes.txt": ["route_id", "agency_id"],
    "trips.txt": ["route_id", "trip_id", "service_id", "shape_id", "block_id"],
    "stops.txt": ["stop_id", "parent_station", "zone_id"],
    "stop_times.txt": ["trip_id", "stop_id"],
    "frequencies.txt": ["trip_id"],
    "calendar.txt": ["service_id"],
    "calendar_dates.txt": ["service_id"],
    "shapes.txt": ["shape_id"],
    "transfers.txt": ["from_stop_id", "to_stop_id"],
    "fare_rules.txt": ["route_id", "fare_id"],
    "fare_attributes.txt": ["fare_id", "agency_id"],
}
METRO_TYPES = {1, 2, 12}  # subway/metro, rail (LRT), monorail
MICROBUS_PATTERN = re.compile(r"micro|paratransit|informal|shared|minibus|suzuki|tomnaya|mashroo",
                              re.IGNORECASE)

Feed = dict[str, pd.DataFrame]


# ------------------------------------------------------------------ io
def read_feed(path: Path) -> Feed:
    path = Path(path)
    feed: Feed = {}
    if path.is_dir():
        for f in path.glob("*.txt"):
            feed[f.name] = pd.read_csv(f, dtype=str, keep_default_na=False, encoding="utf-8-sig")
        return feed
    with zipfile.ZipFile(path) as z:
        for name in z.namelist():
            base = Path(name).name
            if base.endswith(".txt"):
                with z.open(name) as fh:
                    feed[base] = pd.read_csv(fh, dtype=str, keep_default_na=False,
                                             encoding="utf-8-sig")
    return feed


def write_feed(feed: Feed, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as z:
        for name, df in feed.items():
            buf = io.StringIO()
            df.to_csv(buf, index=False)
            z.writestr(name, buf.getvalue())


def to_seconds(t) -> float:
    """'HH:MM:SS' (hours may exceed 24) → seconds; NaN for blanks."""
    if t is None:
        return np.nan
    s = str(t).strip()
    if not s:
        return np.nan
    h, m, sec = s.split(":")
    return int(h) * 3600 + int(m) * 60 + int(sec)


# ------------------------------------------------------------------ checks / fixes
def non_monotonic_trips(feed: Feed) -> list[str]:
    st = feed.get("stop_times.txt")
    if st is None or st.empty:
        return []
    df = st[["trip_id", "stop_sequence", "arrival_time", "departure_time"]].copy()
    df["seq"] = pd.to_numeric(df["stop_sequence"], errors="coerce")
    df["arr"] = df["arrival_time"].map(to_seconds)
    df["dep"] = df["departure_time"].map(to_seconds)
    df = df.sort_values(["trip_id", "seq"])
    df["t"] = df["arr"].fillna(df["dep"])
    df["prev"] = df.groupby("trip_id")["dep"].shift().fillna(df.groupby("trip_id")["t"].shift())
    bad = df[(df["t"] < df["prev"]) | (df["dep"] < df["arr"])]
    return sorted(bad["trip_id"].unique().tolist())


def fix_stop_coordinates(feed: Feed, bbox) -> tuple[int, int]:
    """Swap lat/lng for stops outside bbox when the swap puts them inside.

    Returns (n_swapped, n_still_outside)."""
    stops = feed["stops.txt"]
    lat = pd.to_numeric(stops["stop_lat"], errors="coerce")
    lng = pd.to_numeric(stops["stop_lon"], errors="coerce")
    lng_min, lat_min, lng_max, lat_max = bbox

    def inside(la, lo):
        return (la >= lat_min) & (la <= lat_max) & (lo >= lng_min) & (lo <= lng_max)

    out = ~inside(lat, lng)
    swap = out & inside(lng, lat)
    stops.loc[swap, ["stop_lat", "stop_lon"]] = stops.loc[swap, ["stop_lon", "stop_lat"]].values
    return int(swap.sum()), int((out & ~swap).sum())


def drop_trips(feed: Feed, trip_ids) -> None:
    trip_ids = set(trip_ids)
    for name in ("trips.txt", "stop_times.txt", "frequencies.txt"):
        if name in feed:
            feed[name] = feed[name][~feed[name]["trip_id"].isin(trip_ids)].reset_index(drop=True)


def drop_routes(feed: Feed, route_ids) -> None:
    route_ids = set(route_ids)
    trips = feed.get("trips.txt", pd.DataFrame(columns=["trip_id", "route_id"]))
    drop_trips(feed, trips.loc[trips["route_id"].isin(route_ids), "trip_id"])
    feed["routes.txt"] = feed["routes.txt"][~feed["routes.txt"]["route_id"].isin(route_ids)]


def prefix_ids(feed: Feed, prefix: str) -> Feed:
    out = {}
    for name, df in feed.items():
        df = df.copy()
        for col in ID_COLUMNS.get(name, []):
            if col in df.columns:
                mask = df[col].astype(str).str.len() > 0
                df.loc[mask, col] = prefix + df.loc[mask, col].astype(str)
        out[name] = df
    return out


def has_service_on(feed: Feed, day: date) -> bool:
    ymd = day.strftime("%Y%m%d")
    weekday = day.strftime("%A").lower()
    cal = feed.get("calendar.txt")
    active = set()
    if cal is not None and not cal.empty:
        ok = (cal["start_date"] <= ymd) & (cal["end_date"] >= ymd) & (cal[weekday] == "1")
        active |= set(cal.loc[ok, "service_id"])
    dates = feed.get("calendar_dates.txt")
    if dates is not None and not dates.empty:
        today = dates[dates["date"] == ymd]
        active |= set(today.loc[today["exception_type"] == "1", "service_id"])
        active -= set(today.loc[today["exception_type"] == "2", "service_id"])
    trips = feed.get("trips.txt")
    return bool(active) and trips is not None and trips["service_id"].isin(active).any()


# ------------------------------------------------------------------ manual feed
def _hms(seconds: float) -> str:
    seconds = round(seconds)
    return f"{seconds // 3600:02d}:{seconds % 3600 // 60:02d}:{seconds % 60:02d}"


def manual_feed(yaml_path: Path, service_date: str, prefix: str = "manual:") -> tuple[Feed | None,
                                                                                         list[str]]:
    """config/manual_lines.yaml → GTFS (frequency-based, both directions). Lines whose stations
    lack coordinates are skipped with a warning."""
    spec = yaml.safe_load(Path(yaml_path).read_text()) or {}
    warnings: list[str] = []
    day = datetime.strptime(service_date, "%Y-%m-%d").date()
    routes, stops, trips, stop_times, freqs = [], {}, [], [], []
    for line in spec.get("lines", []) or []:
        rid = str(line["route_id"])
        st = [s for s in line.get("stations", []) or []
              if s.get("lat") not in (None, "") and s.get("lng") not in (None, "")]
        if len(st) < 2 or len(st) != len(line.get("stations", []) or []):
            warnings.append(f"manual line {rid}: station coordinates missing; skipped")
            continue
        routes.append({"route_id": rid, "agency_id": "manual", "route_short_name": rid,
                       "route_long_name": line.get("name", rid),
                       "route_type": str(line.get("route_type", 1))})
        lat = np.array([float(s["lat"]) for s in st])
        lng = np.array([float(s["lng"]) for s in st])
        ids = []
        for s, la, lo in zip(st, lat, lng):
            sid = s.get("stop_id") or f"{rid}-{len(stops)}"
            stops.setdefault(sid, {"stop_id": sid, "stop_name": s.get("name", sid),
                                   "stop_lat": f"{la:.6f}", "stop_lon": f"{lo:.6f}"})
            ids.append(sid)
        seg_min = haversine_m(lat[:-1], lng[:-1], lat[1:], lng[1:]) / (
            float(line.get("speed_kmh", 35)) * 1000 / 60)
        dwell = float(line.get("dwell_s", 30))
        for direction, order in enumerate([np.arange(len(ids)), np.arange(len(ids))[::-1]]):
            tid = f"{rid}-d{direction}"
            trips.append({"route_id": rid, "service_id": "manual-all", "trip_id": tid,
                          "direction_id": str(direction)})
            t = 5 * 3600.0
            for seq, k in enumerate(order):
                if seq > 0:
                    prev = order[seq - 1]
                    t += seg_min[min(prev, k)] * 60
                arr = t
                dep = t + (dwell if 0 < seq < len(order) - 1 else 0)
                stop_times.append({"trip_id": tid, "arrival_time": _hms(arr),
                                   "departure_time": _hms(dep), "stop_id": ids[k],
                                   "stop_sequence": str(seq + 1)})
                t = dep
            freqs.append({"trip_id": tid, "start_time": "05:00:00", "end_time": "24:00:00",
                          "headway_secs": str(int(float(line.get("headway_min", 5)) * 60)),
                          "exact_times": "0"})
    if not routes:
        return None, warnings
    cal = {"service_id": "manual-all",
           **{d: "1" for d in ("monday", "tuesday", "wednesday", "thursday", "friday",
                               "saturday", "sunday")},
           "start_date": (day - timedelta(days=366)).strftime("%Y%m%d"),
           "end_date": (day + timedelta(days=366)).strftime("%Y%m%d")}
    feed = {
        "agency.txt": pd.DataFrame([{"agency_id": "manual", "agency_name": "Manual (Khatt)",
                                     "agency_url": "https://example.org",
                                     "agency_timezone": "Africa/Cairo"}]),
        "routes.txt": pd.DataFrame(routes),
        "stops.txt": pd.DataFrame(list(stops.values())),
        "trips.txt": pd.DataFrame(trips),
        "stop_times.txt": pd.DataFrame(stop_times),
        "frequencies.txt": pd.DataFrame(freqs),
        "calendar.txt": pd.DataFrame([cal]),
    }
    return prefix_ids(feed, prefix), warnings


# ------------------------------------------------------------------ routes for fallback / 10
def route_mode(route_type, agency_name: str = "") -> str:
    try:
        rt = int(route_type)
    except (TypeError, ValueError):
        rt = 3
    if rt in METRO_TYPES or 100 <= rt < 200 or 400 <= rt < 500:
        return "metro"
    if MICROBUS_PATTERN.search(agency_name or ""):
        return "microbus"
    return "bus"


def _headway_at(freq: pd.DataFrame | None, trip_ids, at_s: float) -> float | None:
    if freq is None or freq.empty:
        return None
    f = freq[freq["trip_id"].isin(trip_ids)]
    if f.empty:
        return None
    start = f["start_time"].map(to_seconds)
    end = f["end_time"].map(to_seconds)
    cover = f[(start <= at_s) & (end > at_s)]
    use = cover if not cover.empty else f
    return float(pd.to_numeric(use["headway_secs"], errors="coerce").min() / 60.0)


def representative_routes(feed: Feed, departure_s: float, window_s: float,
                          default_headway: float) -> list[Route]:
    """One Route per (route, direction): the most common stop pattern, its ride times and the
    headway around the departure time (frequencies.txt, else trip count in the window)."""
    routes = feed["routes.txt"]
    trips = feed["trips.txt"].copy()
    st = feed["stop_times.txt"].copy()
    stops = feed["stops.txt"].set_index("stop_id")
    freq = feed.get("frequencies.txt")
    agency = feed.get("agency.txt")
    agency_name = {}
    if agency is not None and "agency_id" in agency.columns:
        agency_name = dict(zip(agency["agency_id"], agency.get("agency_name", agency["agency_id"])))
    st["seq"] = pd.to_numeric(st["stop_sequence"], errors="coerce")
    st = st.sort_values(["trip_id", "seq"])
    patterns = st.groupby("trip_id")["stop_id"].apply(tuple)
    trips["pattern"] = trips["trip_id"].map(patterns)
    trips = trips.dropna(subset=["pattern"])
    if "direction_id" not in trips.columns:
        trips["direction_id"] = "0"
    out: list[Route] = []
    for (rid, direction), g in trips.groupby(["route_id", "direction_id"]):
        r = routes[routes["route_id"] == rid]
        if r.empty:
            continue
        r = r.iloc[0]
        pattern = g["pattern"].value_counts().index[0]
        if len(pattern) < 2 or any(s not in stops.index for s in pattern):
            continue
        same = g[g["pattern"] == pattern]["trip_id"]
        trip = st[st["trip_id"] == same.iloc[0]]
        arr = trip["arrival_time"].map(to_seconds).to_numpy()
        dep = trip["departure_time"].map(to_seconds).to_numpy()
        dep = np.where(np.isnan(dep), arr, dep)
        arr = np.where(np.isnan(arr), dep, arr)
        diffs = np.diff(arr) / 60.0
        ride = ([float(v) for v in diffs]
                if np.isfinite(diffs).all() and (diffs >= 0).all() else None)
        headway = _headway_at(freq, set(same), departure_s)
        if headway is None:
            first = st[st["trip_id"].isin(same)].groupby("trip_id")["departure_time"].first()
            t = first.map(to_seconds)
            n = int(((t >= departure_s - window_s / 2) & (t < departure_s + window_s / 2)).sum())
            headway = window_s / 60.0 / n if n else default_headway
        a_name = agency_name.get(r.get("agency_id", ""), "")
        mode = route_mode(r.get("route_type", 3), a_name)
        out.append(Route(
            route_id=f"{rid}|{direction}", mode=mode, headway_min=float(headway),
            stop_lat=[float(stops.at[s, "stop_lat"]) for s in pattern],
            stop_lng=[float(stops.at[s, "stop_lon"]) for s in pattern],
            ride_min=ride, speed_kmh=35.0 if mode == "metro" else 15.0,
            stop_ids=list(pattern),
        ))
    return out
