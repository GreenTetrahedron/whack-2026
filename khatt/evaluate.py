"""Orchestrate engine + 4 models for a set of drawn lines → JSON-ready result dict (§10)."""

from __future__ import annotations

import time
from dataclasses import dataclass

import h3
import numpy as np

from khatt import attribution as attribution_mod
from khatt.config import CATEGORIES, Params
from khatt.engine.catchment import Catchment, catchments
from khatt.engine.lines import LineSpec, Station, line_length_m, validate_lines
from khatt.engine.multiline import joint_rstar, stations_of, wait_board
from khatt.engine.overlay import Overlay
from khatt.geo import to_utm
from khatt.models import footfall, informal, rents, spending, zones

OUTSIDE_AREA_M = 1000.0  # station further than this from any modelled cell centre → warning


@dataclass
class Scenario:
    lines: list[LineSpec]
    stations: list[Station]
    station_hex: np.ndarray
    station_x: np.ndarray
    station_y: np.ndarray
    access: np.ndarray  # per station, minutes
    overlay: Overlay
    warnings: list[str]


# ---------------------------------------------------------------- setup
def snap_stations(world, lines: list[LineSpec], params: Params) -> tuple[list[Station], list[str]]:
    stations = stations_of(lines)
    warnings: list[str] = []
    if not stations:
        return stations, warnings
    lat = np.array([s.lat for s in stations])
    lng = np.array([s.lng for s in stations])
    x, y = to_utm(lat, lng)
    dist, nearest = world.hex_tree.query(np.column_stack([np.atleast_1d(x), np.atleast_1d(y)]))
    pos = _hex_pos(world)
    if world.walk is not None and world.walk.n:
        nodes, snap_m = world.walk.snap(lat, lng)
    else:
        nodes, snap_m = np.full(len(stations), -1), np.full(len(stations), np.inf)
    max_snap = float(params["line.max_snap_m"])
    m_per_min = params.walk_m_per_min
    circuity = float(params["walk.circuity"])
    names = {ln.line_id: ln.name for ln in lines}
    for k, st in enumerate(stations):
        cell = h3.latlng_to_cell(st.lat, st.lng, 8)
        st.hex_idx = int(pos.get(cell, nearest[k]))
        st.node = int(nodes[k])
        st.snap_m = float(snap_m[k])
        # outside the modelled cells: walk to the nearest cell centre on access and egress
        st.access_min = 0.0 if cell in pos else float(dist[k]) * circuity / m_per_min
        label = f"Station {st.idx + 1} of '{names[st.line_id]}'"
        if cell not in pos and dist[k] > OUTSIDE_AREA_M:
            warnings.append(f"{label} is outside the modelled area ({dist[k]:.0f} m from the "
                            "nearest cell); results there are unreliable")
        if world.walk is not None and st.snap_m > max_snap:
            warnings.append(f"{label} is {st.snap_m:.0f} m from the walking network")
    return stations, warnings


def _hex_pos(world) -> dict[str, int]:
    pos = getattr(world, "_hex_pos", None)
    if pos is None:
        pos = {h: k for k, h in enumerate(world.hex_ids)}
        world._hex_pos = pos
    return pos


def build_scenario(world, specs: list[LineSpec], params: Params) -> Scenario:
    lines = [s.resolved(params) for s in specs]
    validate_lines(lines, params)
    stations, warnings = snap_stations(world, lines, params)
    station_hex = np.array([s.hex_idx for s in stations], np.int64)
    access = np.array([s.access_min for s in stations], float)
    if stations:
        ov = scenario_overlay(world, lines, station_hex, access, params)
        sx, sy = to_utm([s.lat for s in stations], [s.lng for s in stations])
    else:
        ov = Overlay(world.T0, station_hex, np.zeros(0), np.zeros((0, 0)))
        sx, sy = np.zeros(0), np.zeros(0)
    return Scenario(lines, stations, station_hex, np.atleast_1d(sx), np.atleast_1d(sy), access,
                    ov, warnings)


def scenario_overlay(world, lines: list[LineSpec], station_hex: np.ndarray, access: np.ndarray,
                     params: Params) -> Overlay:
    rstar = joint_rstar(lines, station_hex, world.T0, params, access=access)
    return Overlay(world.T0, station_hex, wait_board(lines, params) + access, rstar)


def station_catchments(world, sc: Scenario, params: Params,
                       cutoff: float | None = None) -> list[Catchment]:
    """Walking catchment of each station. Default cutoff covers footfall + pass-by."""
    m_per_min = params.walk_m_per_min
    if cutoff is None:
        cutoff = max(float(params["footfall.catchment_min"]),
                     float(params["footfall.passby_radius_m"]) / m_per_min)
    if world.walk is not None and world.walk.n:
        nodes = np.array([s.node for s in sc.stations], np.int64)
        return catchments(world.walk, nodes, cutoff)
    # no walk graph: straight-line distance × circuity over r9 cells
    rx, ry = world.r9_xy()
    circuity = float(params["walk.circuity"])
    out = []
    for s in range(len(sc.stations)):
        minutes = np.hypot(rx - sc.station_x[s], ry - sc.station_y[s]) * circuity / m_per_min
        keep = np.flatnonzero(minutes <= cutoff)
        out.append(Catchment(s, -1, np.zeros(0, np.int64), np.zeros(0), keep, minutes[keep]))
    return out


def street_walk_minutes(world, sc: Scenario, catches: list[Catchment], params: Params):
    """Per station: (street rows, walk minutes) for streets within the pass-by radius."""
    m_per_min = params.walk_m_per_min
    radius_m = float(params["footfall.passby_radius_m"])
    limit = radius_m / m_per_min
    out = []
    xy = np.column_stack([world.street_x, world.street_y])
    for s, c in enumerate(catches):
        near = np.flatnonzero(np.hypot(xy[:, 0] - sc.station_x[s], xy[:, 1] - sc.station_y[s])
                              <= radius_m)
        if len(near) == 0:
            out.append((near, np.zeros(0)))
            continue
        if len(c.nodes):
            order = np.argsort(c.nodes)
            sorted_nodes = c.nodes[order]
            want = world.street_node[near]
            at = np.clip(np.searchsorted(sorted_nodes, want), 0, len(sorted_nodes) - 1)
            found = sorted_nodes[at] == want
            minutes = np.where(found, c.node_min[order][at], np.inf)
        elif world.walk is None:
            d = np.hypot(xy[near, 0] - sc.station_x[s], xy[near, 1] - sc.station_y[s])
            minutes = d * float(params["walk.circuity"]) / m_per_min
        else:
            minutes = np.full(len(near), np.inf)
        keep = minutes <= limit
        out.append((near[keep], minutes[keep]))
    return out


def destinations(world, sc: Scenario, params: Params):
    """Street cells plus res-8 cells within rents.radius_m of any new station."""
    street_cells = np.unique(world.street_hex)
    if len(sc.stations):
        hits = world.hex_tree.query_ball_point(np.column_stack([sc.station_x, sc.station_y]),
                                               r=float(params["rents.radius_m"]))
        rent_cells = np.unique(np.concatenate([np.asarray(h, np.int64) for h in hits]
                                              + [sc.station_hex]))
    else:
        rent_cells = np.zeros(0, np.int64)
    dest = np.union1d(street_cells, rent_cells).astype(np.int64)
    return dest, rent_cells


def baseline_huff(world, params: Params, T0_streets: np.ndarray) -> spending.Huff:
    """Baseline Huff depends only on T0 and params: cache it on the world."""
    cache = world.__dict__.setdefault("_huff_cache", {})
    key = params.key()
    if key not in cache:
        if len(cache) >= 4:
            cache.clear()
        cache[key] = spending.run_huff(world.spend, world.street_A, T0_streets, params)
    return cache[key]


# ---------------------------------------------------------------- main entry
def evaluate(world, specs: list[LineSpec], params: Params, attribution: bool = False) -> dict:
    t_start = time.perf_counter()
    timings: dict[str, float] = {}

    def tick(name: str, t0: list[float] = [t_start]) -> None:  # noqa: B006
        now = time.perf_counter()
        timings[name] = round((now - t0[0]) * 1000, 1)
        t0[0] = now

    sc = build_scenario(world, specs, params)
    tick("engine_setup")
    dest, rent_cells = destinations(world, sc, params)
    T0_d, T1_d = sc.overlay.to(dest)
    street_cols = np.searchsorted(dest, world.street_hex)
    rent_cols = np.searchsorted(dest, rent_cells)
    T0_s, T1_s = T0_d[:, street_cols], T1_d[:, street_cols]
    tick("overlay")

    base = baseline_huff(world, params, T0_s)
    sp = spending.run(world, params, T0_s, T1_s, base=base)
    tick("spending")

    catches = station_catchments(world, sc, params)
    walk_min = street_walk_minutes(world, sc, catches, params)
    priority = [(s.line_id, s.idx) for s in sc.stations]
    ff = footfall.run(world, params, catches, sp.use_share, walk_min, priority)
    tick("footfall")

    R0_total = sp.base.R.sum(axis=1)
    dR_total = sp.dR.sum(axis=1)
    rt = rents.run(world, params, rent_cells, T0_d[:, rent_cols], T1_d[:, rent_cols],
                   sp.scen.S, R0_total, dR_total, ff.passby_egp)
    tick("rents")

    inf = informal.run(world, params, sc.overlay, sc.station_x, sc.station_y, ff.riders_daily)
    tick("informal")

    walk_zone = station_catchments(world, sc, params, float(params["zones.walk_max_min"]))
    zn = zones.run(world, params, sc, walk_zone)
    tick("zones")

    result = assemble(world, params, sc, sp, ff, rt, inf, rent_cells, zn)
    report_rows = np.array([s["_row"] for s in result["streets"]], np.int64)
    for s in result["streets"]:
        del s["_row"]
    if attribution and len(sc.lines) >= 2:
        rows = [[k for k, s in enumerate(sc.stations) if s.line_id == ln.line_id]
                for ln in sc.lines]
        att = attribution_mod.run(
            world, params, sc.lines, [sc.station_hex[r] for r in rows],
            [sc.access[r] for r in rows], base, dest, street_cols, dR_total,
            world.streets["street_id"].to_numpy(), report_rows)
        result["synergy"] = att["synergy"]
        result["attribution"] = att["attribution"]
        result["warnings"] += att["warnings"]
        tick("attribution")
    result["timings_ms"] = timings
    result["elapsed_ms"] = round((time.perf_counter() - t_start) * 1000, 1)
    return result


# ---------------------------------------------------------------- output
def _pct(num, den):
    return np.divide(num, den, out=np.zeros_like(np.asarray(num, float)),
                     where=np.asarray(den) > 0) * 100.0


def assemble(world, params: Params, sc: Scenario, sp, ff, rt, inf, rent_cells,
             zn=None) -> dict:
    if zn is None:
        zn = zones.empty(len(sc.stations))
    st = world.streets
    R0 = sp.base.R
    dR = sp.dR
    R0_total = R0.sum(axis=1)
    dR_total = dR.sum(axis=1)
    dR_pct = _pct(dR_total, R0_total)
    by_cat_pct = _pct(dR, R0)

    if len(sc.stations):
        sxy = np.column_stack([sc.station_x, sc.station_y])
        d_near = np.min(np.hypot(world.street_x[:, None] - sxy[None, :, 0],
                                 world.street_y[:, None] - sxy[None, :, 1]), axis=1)
    else:
        d_near = np.full(len(st), np.inf)
    local_loser = (dR_total < 0) & (d_near <= float(params["spending.local_radius_m"]))

    min_pct = float(params["output.min_abs_dR_pct"])
    report = (np.abs(dR_pct) > min_pct) | (rt.risk != "green") | (ff.passby_egp > 0)
    F = spending.flow_matrix(world, sp)
    own_flows = spending.street_flows(world, F, np.flatnonzero(report),
                                      int(params["spending.n_street_flows"]),
                                      float(params["spending.flow_min_share"]))
    streets = []
    for k in np.flatnonzero(report):
        streets.append({
            "_row": int(k),
            "street_id": int(st["street_id"].iat[k]),
            "name": str(st["name"].iat[k]),
            "lat": float(st["lat"].iat[k]),
            "lng": float(st["lng"].iat[k]),
            "R0_egp": round(float(R0_total[k])),
            "dR_egp": round(float(dR_total[k])),
            "dR_pct": round(float(dR_pct[k]), 3),
            "by_category": {c: round(float(by_cat_pct[k, i]), 3) for i, c in enumerate(CATEGORIES)},
            "by_category_egp": {c: round(float(dR[k, i])) for i, c in enumerate(CATEGORIES)},
            "passby_egp": round(float(ff.passby_egp[k])),
            "rent_pct": round(float(rt.street_rent[k] * 100), 3),
            "rent_pct_opening": round(float(rt.street_rent_opening[k] * 100), 3),
            "revenue_pct": round(float(rt.revenue[k] * 100), 3),
            "risk": str(rt.risk[k]),
            "risk_opening": str(rt.risk_opening[k]),
            "local_loser": bool(local_loser[k]),
            "unbranded_at_risk": int(rt.unbranded_at_risk[k]),
            "flows": own_flows.get(int(k), []),
        })

    stations = []
    for k, s in enumerate(sc.stations):
        stations.append({
            "line_id": s.line_id, "idx": s.idx, "lat": s.lat, "lng": s.lng,
            "h3": world.hex_ids[s.hex_idx],
            "riders_daily": round(float(ff.riders_daily[k])),
            "catchment_pop10": round(float(ff.catchment_pop10[k])),
            "usefulness": round(float(ff.usefulness[k]), 3),
            "vendor_pitches": round(float(inf.vendor_pitches[k]), 1),
            "zone_radius_m": round(float(zn.radius_m[k])),
            "zone_users": round(float(zn.users[k])),
            "zone_pop": round(float(zn.pop_in[k])),
            "zone_mean_p": round(float(zn.mean_p[k]), 3),
            "snap_m": None if not np.isfinite(s.snap_m) else round(s.snap_m, 1),
        })

    rent_by_hex = dict(zip(rent_cells.tolist(), rt.rent_cells.tolist()))
    min_hex = float(params["output.min_abs_hex_change"])
    hex_rows = set(np.flatnonzero(np.abs(sp.d_reach) > min_hex).tolist())
    hex_rows |= {h for h, v in rent_by_hex.items() if abs(v) > min_hex}
    hexes = [{
        "h3": world.hex_ids[h],
        "d_spend_reach": round(float(sp.d_reach[h]), 4),
        "rent_pct": round(float(rent_by_hex.get(h, 0.0) * 100), 3),
        "rent_pct_opening": round(float(rent_by_hex.get(h, 0.0) * 100
                                        * float(params["rents.short_run_share"])), 3),
    } for h in sorted(hex_rows)]

    min_ff = float(params["output.min_footfall_daily"])
    footfall_cells = [
        {"h3": str(world.r9["h3"].iat[c]), "footfall_daily": round(float(v), 1)}
        for c, v in zip(ff.footfall_r9, ff.footfall_daily) if v >= min_ff
    ]

    mb = world.microbus
    microbus_routes = [
        {"route_id": str(mb["route_id"].iat[r]), "status": str(inf.status[r]),
         "share_at_risk": round(float(inf.share_at_risk[r]), 3)}
        for r in np.flatnonzero(inf.status != "unaffected")
    ]
    status_counts = {k: int((inf.status == k).sum()) for k in ("competing", "feeder", "unaffected")}

    min_dR = np.abs(dR_pct) > min_pct
    totals = {
        "dR_egp": float(dR_total.sum()),
        "abs_dR_egp": float(np.abs(dR_total).sum()),
        "R0_egp": float(R0_total.sum()),
        "passby_egp": float(ff.passby_egp.sum()),
        "total_revenue_change": float(dR_total.sum() + ff.passby_egp.sum()),
        "riders_daily": float(ff.riders_daily.sum()),
        "streets_gaining": int(((dR_total > 0) & min_dR).sum()),
        "streets_losing": int(((dR_total < 0) & min_dR).sum()),
        "local_losers": int(local_loser.sum()),
        "red_streets": int((rt.risk == "red").sum()),
        "amber_streets": int((rt.risk == "amber").sum()),
        "red_streets_opening": int((rt.risk_opening == "red").sum()),
        "unbranded_at_risk": int(rt.unbranded_at_risk.sum()),
        "max_rent_pct": float(rt.rent_cells.max() * 100) if len(rt.rent_cells) else 0.0,
        "vendor_pitches": float(inf.vendor_pitches.sum()),
        "vendor_pitches_lost": inf.vendor_pitches_lost,
        "microbus": status_counts,
        "zone_users": float(zn.users.sum()),
        "zone_threshold": float(params["zones.threshold"]),
    }

    lines_out = [{
        "line_id": ln.line_id, "name": ln.name, "n_stations": len(ln.stations),
        "length_km": round(line_length_m(ln.stations) / 1000, 2),
        "headway_min": ln.headway_min, "speed_kmh": ln.speed_kmh, "dwell_s": ln.dwell_s,
    } for ln in sc.lines]

    return {
        "warnings": list(sc.warnings),
        "lines": lines_out,
        "totals": totals,
        "streets": streets,
        "stations": stations,
        "hexes": hexes,
        "footfall_cells": footfall_cells,
        "flows": spending.top_flows(world, sp, int(params["spending.n_flows"]), F,
                                   float(params["spending.flow_min_share"])),
        "microbus_routes": microbus_routes,
        "synergy": [],
        "attribution": {},
    }
