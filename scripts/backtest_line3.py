"""Line 3 west-branch plausibility backtest (CLAUDE.md §13).

1. Build a T0 variant without the manual feed (the branch is in it):
       python pipeline/08_baseline_matrix.py --exclude-feed manual --out data/processed/T0_no_manual.npy
2. python scripts/backtest_line3.py
Evaluates the branch as a drawn line on that baseline and saves predicted riders plus the top
winner/loser streets to scenarios/backtest_line3.json. A plausibility test only.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import yaml

from khatt.config import ROOT, Params, processed_dir, scenarios_dir
from khatt.data import load_world
from khatt.engine.lines import LineSpec
from khatt.evaluate import evaluate


def branch_line(route_id: str) -> LineSpec | None:
    spec = yaml.safe_load((ROOT / "config" / "manual_lines.yaml").read_text())
    for ln in spec.get("lines", []):
        if ln["route_id"] == route_id:
            st = [(s.get("lat"), s.get("lng")) for s in ln.get("stations", [])]
            if len(st) < 2 or any(a is None or b is None for a, b in st):
                return None
            return LineSpec(route_id, ln.get("name", route_id),
                            [(float(a), float(b)) for a, b in st],
                            ln.get("headway_min"), ln.get("speed_kmh"), ln.get("dwell_s"))
    return None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--route", default="L3W")
    ap.add_argument("--t0", default=str(processed_dir() / "T0_no_manual.npy"))
    args = ap.parse_args()
    line = branch_line(args.route)
    if line is None:
        sys.exit(f"{args.route}: fill station coordinates in config/manual_lines.yaml first")
    if not Path(args.t0).exists():
        sys.exit(f"missing {args.t0}: run pipeline/08_baseline_matrix.py --exclude-feed manual "
                 f"--out {args.t0}")
    variant = np.load(args.t0)
    world = load_world(processed_dir())
    if variant.shape != world.T0.shape:
        sys.exit(f"T0 variant shape {variant.shape} != {world.T0.shape}")
    world.T0 = variant.astype(np.float32)
    params = Params.load()
    result = evaluate(world, [line], params)
    streets = sorted(result["streets"], key=lambda s: s["dR_egp"])
    doc = {
        "id": "backtest_line3", "name": "Backtest: Line 3 west branch",
        "description": "Branch evaluated as a drawn line on a baseline without it. "
                       "Plausibility test only.",
        "lines": [{"line_id": line.line_id, "name": line.name,
                   "stations": [{"lat": a, "lng": b} for a, b in line.stations]}],
        "predicted_riders_daily": result["totals"]["riders_daily"],
        "stations": result["stations"],
        "top_winners": streets[::-1][:10],
        "top_losers": streets[:10],
        "result": result,
        "generated_at": time.strftime("%Y-%m-%d %H:%M"),
    }
    out = scenarios_dir() / "backtest_line3.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(doc))
    print(f"predicted riders/day: {doc['predicted_riders_daily']:,.0f} → {out}")


if __name__ == "__main__":
    main()
