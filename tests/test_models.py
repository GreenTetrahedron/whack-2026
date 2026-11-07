"""Footfall (2), rents (3), informal (4) and catchments."""

import numpy as np
import pytest

from khatt.engine.catchment import Catchment, WalkGraph, assign_nearest, reached_cells
from khatt.evaluate import build_scenario, evaluate
from khatt.models import informal
from khatt.models.footfall import riders_per_resident
from khatt.models.rents import classify, market_potential


# ---------------- catchments ----------------
def _line_graph(n=10, step_min=1.0):
    x = np.arange(n) * 75.0
    src = np.concatenate([np.arange(n - 1), np.arange(1, n)])
    dst = np.concatenate([np.arange(1, n), np.arange(n - 1)])
    return WalkGraph.from_edges(x, np.zeros(n), src, dst, np.full(len(src), step_min),
                                node_r9=np.arange(n) // 2)


def test_dijkstra_cutoff_and_cache():
    g = _line_graph()
    nodes, t = g.reach(0, 3.5)
    assert sorted(nodes.tolist()) == [0, 1, 2, 3]
    assert np.allclose(np.sort(t), [0, 1, 2, 3])
    again = g.reach(0, 3.5)
    assert again[0] is nodes, "second call served from cache"
    nodes5, _ = g.reach(5, 2.0)
    assert sorted(nodes5.tolist()) == [3, 4, 5, 6, 7]
    assert np.isinf(g._dist).all(), "workspace restored"


def test_reached_cells_takes_min_time():
    g = _line_graph()
    cells, t = reached_cells(g, np.array([0, 1, 2, 3]), np.array([0.0, 1.0, 2.0, 3.0]))
    assert cells.tolist() == [0, 1] and t.tolist() == [0.0, 2.0]


def test_assign_nearest_counts_each_cell_once():
    a = Catchment(0, 0, np.array([]), np.array([]), np.array([0, 1, 2]), np.array([1.0, 2, 9]))
    b = Catchment(1, 0, np.array([]), np.array([]), np.array([2, 3]), np.array([3.0, 4]))
    owner, best = assign_nearest([a, b], 5, cutoff_min=10)
    assert owner.tolist() == [0, 0, 1, 1, -1]
    assert best[2] == 3.0


# ---------------- footfall ----------------
def test_riders_calibration(world, params):
    r = riders_per_resident(world, params)
    assert r > 0
    assert np.isclose(r * world.existing["catchment_pop10"].sum(), 4.5e6)


def test_station_riders_and_passby(world, params, downtown_line):
    out = evaluate(world, [downtown_line], params)
    riders = [s["riders_daily"] for s in out["stations"]]
    assert all(r >= 0 for r in riders) and sum(riders) > 0
    assert out["totals"]["passby_egp"] > 0
    assert all(s["catchment_pop10"] >= 0 for s in out["stations"])
    lo, hi = params["footfall.usefulness_clip"]
    assert all(lo <= s["usefulness"] <= hi for s in out["stations"])
    assert out["footfall_cells"], "heatmap cells for the web"
    # pass-by is reported separately from dR and only summed in total_revenue_change
    t = out["totals"]
    assert np.isclose(t["total_revenue_change"], t["dR_egp"] + t["passby_egp"])


def test_more_riders_with_higher_calibration(world, params, downtown_line):
    lo = evaluate(world, [downtown_line], params.with_overrides(
        {"footfall.metro_daily_riders": 4.0e6}))
    hi = evaluate(world, [downtown_line], params.with_overrides(
        {"footfall.metro_daily_riders": 5.0e6}))
    assert hi["totals"]["riders_daily"] > lo["totals"]["riders_daily"]


# ---------------- rents ----------------
def test_classify_thresholds():
    risk = classify(np.array([0.01, 0.0, -0.01, -0.02, -0.05]), 0.02)
    assert risk.tolist() == ["green", "green", "amber", "amber", "red"]


def test_market_potential_rises_when_times_fall():
    S = np.ones((3, 3))
    T = np.full((3, 2), 30.0)
    beta = np.array([0.1, 0.05, 0.08])
    assert (market_potential(S, T - 10, beta) > market_potential(S, T, beta)).all()


def test_rents_rise_near_stations(world, params, downtown_line):
    out = evaluate(world, [downtown_line], params)
    rents = [h["rent_pct"] for h in out["hexes"] if h["rent_pct"] != 0]
    assert rents and max(rents) > 0
    for h in out["hexes"]:
        assert np.isclose(h["rent_pct_opening"], h["rent_pct"] * params["rents.short_run_share"],
                          atol=2e-3)
    for s in out["streets"]:
        assert s["risk"] in ("green", "amber", "red")
        if s["risk"] != "red":
            assert s["unbranded_at_risk"] == 0


def test_higher_elasticity_more_red(world, params, downtown_line):
    lo = evaluate(world, [downtown_line], params.with_overrides({"rents.elasticity": 0.1}))
    hi = evaluate(world, [downtown_line], params.with_overrides({"rents.elasticity": 3.0}))
    assert hi["totals"]["max_rent_pct"] > lo["totals"]["max_rent_pct"]
    assert hi["totals"]["red_streets"] >= lo["totals"]["red_streets"]


# ---------------- informal ----------------
def test_microbus_classification(world, params, downtown_line):
    out = evaluate(world, [downtown_line], params)
    counts = out["totals"]["microbus"]
    assert sum(counts.values()) == len(world.microbus)
    assert counts["feeder"] + counts["competing"] > 0
    for r in out["microbus_routes"]:
        assert r["status"] in ("competing", "feeder")
        if r["status"] == "competing":
            assert r["share_at_risk"] >= params["informal.compete_threshold"]
    pitches = sum(s["vendor_pitches"] for s in out["stations"])
    expected = out["totals"]["riders_daily"] / 1000 * params["informal.vendors_per_1000_exits"]
    assert np.isclose(pitches, expected, rtol=0.01, atol=0.5)


def test_competing_when_threshold_zero(world, params, downtown_line):
    p = params.with_overrides({"informal.compete_threshold": 0.0})
    sc = build_scenario(world, [downtown_line], p)
    res = informal.run(world, p, sc.overlay, sc.station_x, sc.station_y,
                       np.zeros(len(sc.stations)))
    assert (res.status == "competing").all()


@pytest.mark.parametrize("gap", [1, 3])
def test_pairs_respect_gap(world, gap):
    rows, _a, _b = world.microbus_pairs(gap)
    assert len(rows) > 0
