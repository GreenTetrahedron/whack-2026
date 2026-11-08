"""Fit beta per category with Hyman's method (CLAUDE.md §13).

Bisection on beta_c until the spending-weighted mean baseline shopping trip time equals
spending.target_mean_trip_min[c]. Writes data/processed/calibrated.json.

    python -m khatt.calibrate [processed_dir]
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from khatt.config import CATEGORIES, Params, processed_dir
from khatt.data import load_world
from khatt.models.spending import huff_probs

BETA_RANGE = (1e-4, 3.0)
TOLERANCE_MIN = 0.01
MAX_ITER = 80


def street_times(world) -> np.ndarray:
    return world.T0[:, world.street_hex]


def mean_trip_time(world, params: Params, c: int, beta: float, T: np.ndarray) -> float:
    with np.errstate(divide="ignore"):
        logA = np.log(world.street_A[:, c])
    P, _ = huff_probs(logA, T, float(params["spending.alpha"]), beta,
                      float(params["spending.max_trip_min"]))
    S = world.spend[:, c]
    trips = S[:, None] * P
    total = trips.sum()
    return float((trips * T).sum() / total) if total > 0 else float("nan")


def fit_beta(world, params: Params) -> dict:
    """Bisection per category. If the target is below what any beta can reach (the time to the
    nearest street cluster, ``floor``), aim for floor × (1 + calibration_floor_margin) instead of
    pinning beta at the cap, which would make everyone shop only at the nearest cluster."""
    T = street_times(world)
    margin = float(params["spending.calibration_floor_margin"])
    out = {"beta": {}, "achieved_mean_min": {}, "target_mean_min": {},
           "effective_target_min": {}, "notes": []}
    for c, cat in enumerate(CATEGORIES):
        target = float(params[f"spending.target_mean_trip_min.{cat}"])
        out["target_mean_min"][cat] = target
        lo, hi = BETA_RANGE
        f_lo = mean_trip_time(world, params, c, lo, T)
        f_hi = mean_trip_time(world, params, c, hi, T)
        if target <= f_hi * (1 + margin):
            new_target = round(f_hi * (1 + margin), 2)
            out["notes"].append(f"{cat}: target {target} min is below the nearest-cluster floor "
                                f"{f_hi:.1f} min; calibrated to {new_target} (floor + {margin:.0%})")
            target = new_target
        out["effective_target_min"][cat] = target
        if target >= f_lo:
            beta, achieved = lo, f_lo
            out["notes"].append(f"{cat}: target {target} ≥ max achievable {f_lo:.1f}; beta at floor")
        else:
            for _ in range(MAX_ITER):
                mid = np.sqrt(lo * hi)  # geometric bisection: beta spans orders of magnitude
                f_mid = mean_trip_time(world, params, c, mid, T)
                if abs(f_mid - target) < TOLERANCE_MIN:
                    break
                if f_mid > target:
                    lo = mid
                else:
                    hi = mid
            beta, achieved = float(mid), float(f_mid)
        out["beta"][cat] = float(beta)
        out["achieved_mean_min"][cat] = round(float(achieved), 3)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("processed", nargs="?", default=None)
    args = ap.parse_args()
    path = Path(args.processed) if args.processed else processed_dir()
    t = time.perf_counter()
    params = Params.load(calibrated=False)
    world = load_world(path)
    result = fit_beta(world, params)
    result["fitted_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    (path / "calibrated.json").write_text(json.dumps(result, indent=2))
    print(f"calibrated.json written to {path} in {time.perf_counter() - t:.1f} s")
    for cat in CATEGORIES:
        print(f"  {cat:5s} beta={result['beta'][cat]:.4f}  mean trip "
              f"{result['achieved_mean_min'][cat]:.1f} min (target "
              f"{result['effective_target_min'][cat]}, spec {result['target_mean_min'][cat]})")
    for note in result["notes"]:
        print("  NOTE:", note)


if __name__ == "__main__":
    main()
