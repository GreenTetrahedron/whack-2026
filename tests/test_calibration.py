import itertools

import numpy as np

from khatt.calibrate import fit_beta, mean_trip_time, street_times
from khatt.montecarlo import run as run_mc


def test_mean_trip_time_decreases_with_beta(world, params):
    T = street_times(world)
    times = [mean_trip_time(world, params, 0, b, T) for b in (0.01, 0.05, 0.2, 1.0)]
    assert all(a > b for a, b in itertools.pairwise(times))


def test_fit_hits_targets(world, params):
    out = fit_beta(world, params)
    for cat, target in out["effective_target_min"].items():
        assert abs(out["achieved_mean_min"][cat] - target) < 0.05, cat
        assert 0 < out["beta"][cat] < 3.0, "never pinned at the cap"
        assert target >= out["target_mean_min"][cat]


def test_montecarlo_reproducible(world, params, downtown_line):
    a = run_mc(world, [downtown_line], params, n_runs=6, seed=42)
    b = run_mc(world, [downtown_line], params, n_runs=6, seed=42)
    assert a["totals"] == b["totals"]
    q = a["totals"]["riders_daily"]
    assert q["p10"] <= q["p50"] <= q["p90"]
    assert all(0 <= s["p_loss"] <= 1 for s in a["streets"].values())
    assert np.isfinite(a["totals"]["abs_dR_egp"]["p50"])
