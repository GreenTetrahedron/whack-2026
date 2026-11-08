import numpy as np
import pytest

from khatt.engine.lines import LineSpec
from khatt.evaluate import build_scenario, evaluate
from khatt.models.spending import huff_probs, run, run_huff


def _street_times(world, sc):
    dest = np.unique(world.street_hex)
    T0_d, T1_d = sc.overlay.to(dest)
    cols = np.searchsorted(dest, world.street_hex)
    return T0_d[:, cols], T1_d[:, cols]


def test_probabilities_sum_to_one(world, params, downtown_line):
    sc = build_scenario(world, [downtown_line], params)
    _T0_s, T1_s = _street_times(world, sc)
    huff = run_huff(world.spend, world.street_A, T1_s, params)
    for P in huff.P:
        sums = P.sum(axis=1)
        reachable = np.isfinite(huff.log_reach).all(axis=1)
        assert reachable.mean() > 0.95
        np.testing.assert_allclose(sums[reachable], 1.0, atol=1e-5)
        assert (P >= 0).all()


def test_huff_excludes_far_and_zero_attractiveness():
    with np.errstate(divide="ignore"):
        logA = np.log(np.array([1.0, 2.0, 0.0]))
    T = np.array([[10.0, 200.0, 5.0], [500.0, 600.0, 700.0]])
    with np.errstate(divide="ignore"):
        P, lr = huff_probs(logA, T, 1.0, 0.1, 120.0)
    assert np.allclose(P[0], [1.0, 0.0, 0.0])
    assert np.allclose(P[1], 0.0) and lr[1] == -np.inf


def test_sum_dR_is_zero_without_induced_demand(world, params, downtown_line):
    assert params["spending.induced_pct"] == 0
    sc = build_scenario(world, [downtown_line], params)
    T0_s, T1_s = _street_times(world, sc)
    res = run(world, params, T0_s, T1_s)
    total = res.base.R.sum()
    assert abs(res.dR.sum()) < 1e-6 * total
    assert np.abs(res.dR).sum() > 1e-4 * total, "the line should move some spending"


def test_induced_demand_adds_spending(world, params, downtown_line):
    p = params.with_overrides({"spending.induced_pct": 0.05})
    sc = build_scenario(world, [downtown_line], p)
    T0_s, T1_s = _street_times(world, sc)
    res = run(world, p, T0_s, T1_s)
    assert res.dR.sum() > 0


def test_far_line_changes_nothing(world, params):
    far = LineSpec("F", "Far", [(29.0, 32.5), (29.02, 32.55), (29.04, 32.6)])
    out = evaluate(world, [far], params)
    assert out["warnings"], "stations outside the area must be flagged"
    assert all(abs(s["dR_pct"]) < 0.1 for s in out["streets"])
    assert out["totals"]["streets_losing"] == 0 and out["totals"]["streets_gaining"] == 0


def test_closer_streets_gain(world, params, downtown_line):
    out = evaluate(world, [downtown_line], params)
    assert out["totals"]["streets_gaining"] > 0
    assert out["totals"]["streets_losing"] > 0
    assert out["flows"] and len(out["flows"]) <= params["spending.n_flows"]
    f = out["flows"][0]
    assert len(f["from"]) == 2 and 31 < f["from"][0] < 32, "flows are [lng, lat]"


def test_street_flows_are_each_streets_largest(world, params, downtown_line):
    from khatt.evaluate import build_scenario, destinations
    from khatt.models.spending import flow_matrix, street_flows

    sc = build_scenario(world, [downtown_line], params)
    dest, _ = destinations(world, sc, params)
    T0_d, T1_d = sc.overlay.to(dest)
    cols = np.searchsorted(dest, world.street_hex)
    F = flow_matrix(world, run(world, params, T0_d[:, cols], T1_d[:, cols]))
    rows = np.flatnonzero(np.abs(F).max(axis=0) > 0)[:10]
    got = street_flows(world, F, rows, 5)
    for row in rows:
        flows = got[int(row)]
        mags = [abs(f["egp"]) for f in flows]
        assert 0 < len(flows) <= 5 and mags == sorted(mags, reverse=True)
        assert mags[0] == pytest.approx(np.abs(F[:, row]).max(), abs=1)
        assert all(np.sign(f["egp"]) in (-1, 0, 1) for f in flows)


def test_flows_drop_the_tiny_tail(world, params, downtown_line):
    """Map flows skip shifts below flow_min_share × the largest (overview and per street)."""
    share = float(params["spending.flow_min_share"])
    assert share > 0
    out = evaluate(world, [downtown_line], params)
    mags = [abs(f["egp"]) for f in out["flows"]]
    assert mags and min(mags) >= share * max(mags) - 1e-6
    for s in out["streets"]:
        own = [abs(f["egp"]) for f in s["flows"]]
        assert not own or min(own) >= share * max(own) - 1
    loose = evaluate(world, [downtown_line], params.with_overrides({"spending": {"flow_min_share": 0}}))
    assert len(loose["flows"]) >= len(out["flows"])


def test_evaluate_returns_street_flows(world, params, downtown_line):
    out = evaluate(world, [downtown_line], params)
    k = int(params["spending.n_street_flows"])
    with_flows = [s for s in out["streets"] if s["flows"]]
    assert with_flows and all(len(s["flows"]) <= k for s in out["streets"])
    f = with_flows[0]["flows"][0]
    assert len(f["from"]) == 2 and isinstance(f["egp"], int)
