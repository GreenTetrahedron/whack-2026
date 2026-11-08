"""Station zones (area of effect): kernel vs brute force, bounds, invariants."""

import numpy as np
import pytest

from khatt.engine.lines import LineSpec
from khatt.evaluate import build_scenario, evaluate, station_catchments
from khatt.models import zones


def _brute(Tb, A, G, H, row_parent, extras, mass, beta, pw, max_trip, lo, ramp):
    R, S, K = len(row_parent), G.shape[0], mass.shape[0]
    out = np.zeros((R, S))
    for r, p in enumerate(row_parent):
        tot, acc = np.zeros(K), np.zeros((K, S))
        for d in range(Tb.shape[1]):
            options = [(A[p, i] + G[i, d], i) for i in range(S)]
            options += [(cost + G[i, d], i) for i, cost, _ in extras[r]]
            tl, a = min(options)
            tb = min([Tb[p, d]] + [base + H[i, d] for i, _, base in extras[r]])
            t1 = min(tb, tl)
            if t1 > max_trip:
                continue
            pl = float(np.clip((tb - tl - lo) / ramp, 0, 1))
            for k in range(K):
                if mass[k, d] > 0:
                    w = mass[k, d] * np.exp(-beta[k] * t1)
                    tot[k] += w
                    acc[k, a] += w * pl
        live = tot > 0
        weights = pw[p] * live / (pw[p] * live).sum()
        for k in np.flatnonzero(live):
            out[r] += weights[k] * acc[k] / tot[k]
    return out


def test_kernel_matches_brute_force():
    rng = np.random.default_rng(3)
    P, S, D, K = 4, 3, 9, 2
    Tb = rng.uniform(5, 90, (P, D)).astype(np.float32)
    A = rng.uniform(3, 30, (P, S)).astype(np.float32)
    G = rng.uniform(5, 60, (S, D)).astype(np.float32)
    H = rng.uniform(5, 60, (S, D)).astype(np.float32)
    mass = rng.uniform(0, 5, (K, D))
    mass[0, :2] = 0
    beta, pw = np.array([0.05, 0.2]), rng.uniform(0.1, 1, (P, K))
    row_parent = np.array([0, 1, 2, 3, 1, 2])
    extras = [[], [], [], [], [(0, 2.0, 1.0), (2, 4.0, 30.0)], [(1, 1.0, 0.5)]]
    ptr = np.concatenate([[0], np.cumsum([len(e) for e in extras])]).astype(np.int64)
    ex_station = np.array([x[0] for e in extras for x in e], np.int64)
    ex_cost = np.array([x[1] for e in extras for x in e], np.float32)
    ex_base = np.array([x[2] for e in extras for x in e], np.float32)

    TL, ARG = zones.best_line(A, G)
    got = zones.row_use(Tb, TL, ARG, row_parent, ptr, ex_station, ex_cost, ex_base, G, H, mass,
                        beta, pw, 80.0, 2.0, 8.0)
    want = _brute(Tb, A, G, H, row_parent, extras, mass, beta, pw, 80.0, 2.0, 8.0)
    assert got.max() > 0.05, "test inputs should produce some line use"
    assert np.allclose(got, want, atol=1e-6)


def test_line_times_match_overlay(world, params, downtown_line):
    sc = build_scenario(world, [downtown_line], params)
    parents = np.arange(0, world.n, 7)
    dest, _ = zones.destination_mass(world, params)
    Tb, _, _, TL, _ = zones.line_times(world, sc.overlay, parents, dest)
    _, T1 = sc.overlay.to(dest)
    assert np.allclose(np.minimum(Tb, TL), T1[parents], atol=1e-4)


def _zones(world, params, lines):
    sc = build_scenario(world, lines, params)
    walk = station_catchments(world, sc, params, float(params["zones.walk_max_min"]))
    return zones.run(world, params, sc, walk)


def test_probabilities_bounded(world, params, downtown_line, cross_line):
    z = _zones(world, params, [downtown_line, cross_line])
    assert len(z.cells) and z.p_use.min() >= 0
    assert z.p_use.sum(axis=1).max() <= 1 + 1e-9
    assert z.users.sum() > 0 and (z.radius_m > 0).any()
    assert (z.radius_m <= float(params["zones.max_radius_m"])).all()
    inside = z.radius_m > 0
    assert (z.mean_p[inside] >= float(params["zones.threshold"]) - 1e-9).all()


def test_no_lines_no_zones(world, params):
    out = evaluate(world, [], params)
    assert out["stations"] == [] and out["totals"]["zone_users"] == 0


def test_far_line_has_no_zone(world, params):
    far = LineSpec("F", "Far", [(29.0, 32.5), (29.02, 32.55), (29.04, 32.6)])
    out = evaluate(world, [far], params)
    assert all(s["zone_radius_m"] == 0 and s["zone_users"] == 0 for s in out["stations"])


def test_useless_line_has_no_users(world, params, downtown_line):
    crawl = LineSpec(**{**downtown_line.__dict__, "speed_kmh": 1.0})
    assert _zones(world, params, [crawl]).users.sum() == pytest.approx(0, abs=1e-6)


def test_faster_line_has_more_users(world, params, downtown_line):
    slow = LineSpec(**{**downtown_line.__dict__, "speed_kmh": 20.0})
    fast = LineSpec(**{**downtown_line.__dict__, "speed_kmh": 60.0})
    assert _zones(world, params, [fast]).users.sum() > _zones(world, params, [slow]).users.sum()


def test_higher_threshold_smaller_circles(world, params, downtown_line):
    z = _zones(world, params, [downtown_line])
    sc = build_scenario(world, [downtown_line], params)
    sxy = np.column_stack([sc.station_x, sc.station_y])
    strict = zones.circles(world, params.with_overrides({"zones.threshold": 0.5}), sxy, z.cells,
                           z.p_use)
    assert (strict.radius_m <= z.radius_m + 1e-9).all()


def test_zones_order_independent(world, params, downtown_line, cross_line):
    def by_station(out):
        return {(s["line_id"], s["idx"]): (s["zone_radius_m"], s["zone_users"])
                for s in out["stations"]}

    a = evaluate(world, [downtown_line, cross_line], params)
    b = evaluate(world, [cross_line, downtown_line], params)
    assert by_station(a) == by_station(b)
