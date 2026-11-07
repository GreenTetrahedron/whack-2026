"""Offline uncertainty runs for hero scenarios (CLAUDE.md §13).

Every uncertain parameter is drawn from triangular(low, central, high), seed 42. Reports
P10/P50/P90 of headline totals and, per street, the fraction of runs in which it loses (p_loss).
"""

from __future__ import annotations

import time

import numpy as np

from khatt.config import Params
from khatt.engine.lines import LineSpec
from khatt.evaluate import evaluate

HEADLINES = [
    "abs_dR_egp", "passby_egp", "total_revenue_change", "riders_daily", "streets_gaining",
    "streets_losing", "local_losers", "red_streets", "unbranded_at_risk", "max_rent_pct",
    "vendor_pitches", "vendor_pitches_lost",
]


def run(world, specs: list[LineSpec], params: Params, n_runs: int | None = None,
        seed: int | None = None, progress: bool = False) -> dict:
    n_runs = int(n_runs or params["montecarlo.n_runs"])
    seed = int(params["montecarlo.seed"] if seed is None else seed)
    rng = np.random.default_rng(seed)
    min_pct = float(params["output.min_abs_dR_pct"])
    samples = {k: [] for k in HEADLINES + ["microbus_competing"]}
    losses: dict[int, int] = {}
    t = time.perf_counter()
    for k in range(n_runs):
        out = evaluate(world, specs, params.sample(rng))
        tot = out["totals"]
        for key in HEADLINES:
            samples[key].append(float(tot[key]))
        samples["microbus_competing"].append(float(tot["microbus"]["competing"]))
        for s in out["streets"]:
            if s["dR_egp"] < 0 and abs(s["dR_pct"]) > min_pct:
                losses[s["street_id"]] = losses.get(s["street_id"], 0) + 1
        if progress and (k + 1) % 20 == 0:
            print(f"  monte carlo {k + 1}/{n_runs} ({time.perf_counter() - t:.0f} s)")
    totals = {
        key: {f"p{q}": float(np.percentile(vals, q)) for q in (10, 50, 90)}
        for key, vals in samples.items()
    }
    return {
        "n_runs": n_runs,
        "seed": seed,
        "totals": totals,
        "streets": {str(sid): {"p_loss": round(c / n_runs, 3)} for sid, c in losses.items()},
        "elapsed_s": round(time.perf_counter() - t, 1),
    }
