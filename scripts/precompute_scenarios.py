"""Precompute hero scenarios (CLAUDE.md §17) with Monte Carlo ranges → scenarios/{id}.json.

    python scripts/precompute_scenarios.py [--runs 200] [--only underserved] [--no-mc]

These files are the demo fallback: GET /scenarios/{id} serves them even if the engine is down.
"""

from __future__ import annotations

import argparse
import json
import time

import numpy as np
import yaml

from khatt import montecarlo
from khatt.config import ROOT, Params, processed_dir, scenarios_dir
from khatt.data import load_world
from khatt.engine.lines import LineSpec
from khatt.evaluate import evaluate
from khatt.geo import from_utm

STATION_SPACING_M = 1200.0
UNDERSERVED_LENGTH_M = 12000.0
SEARCH_RADIUS_M = 6000.0


def underserved_line(world) -> LineSpec:
    """Densest area furthest from any existing station; stations every ~1.2 km along the
    principal axis of high-scoring cells around it."""
    ex = world.existing
    if len(ex):
        from khatt.geo import to_utm

        ex_x, ex_y = to_utm(ex["lat"], ex["lng"])
        d = np.min(np.hypot(world.hex_x[:, None] - np.atleast_1d(ex_x)[None, :],
                            world.hex_y[:, None] - np.atleast_1d(ex_y)[None, :]), axis=1)
    else:
        d = np.ones(world.n)
    score = world.pop * d
    # smooth over neighbours so one dense cell does not win alone
    nb = world.hex_tree.query_ball_point(np.column_stack([world.hex_x, world.hex_y]), r=1000)
    smooth = np.array([score[n].sum() for n in nb])
    seed = int(np.argmax(smooth))
    near = np.array(world.hex_tree.query_ball_point([world.hex_x[seed], world.hex_y[seed]],
                                                    r=SEARCH_RADIUS_M))
    w = smooth[near]
    xy = np.column_stack([world.hex_x[near], world.hex_y[near]])
    centre = np.average(xy, axis=0, weights=w)
    cov = np.cov((xy - centre).T, aweights=w)
    axis = np.linalg.eigh(cov)[1][:, -1]
    n = int(UNDERSERVED_LENGTH_M // STATION_SPACING_M) + 1
    offsets = (np.arange(n) - (n - 1) / 2) * STATION_SPACING_M
    pts = centre[None, :] + offsets[:, None] * axis[None, :]
    lat, lng = from_utm(pts[:, 0], pts[:, 1])
    return LineSpec("U", "Underserved", [(float(a), float(b)) for a, b in zip(lat, lng)])


def spec_lines(sc: dict) -> list[LineSpec]:
    out = []
    for ln in sc.get("lines") or []:
        stations = [(float(s["lat"]), float(s["lng"])) for s in ln.get("stations") or []
                    if s.get("lat") is not None and s.get("lng") is not None]
        out.append(LineSpec(str(ln["line_id"]), ln.get("name", ln["line_id"]), stations,
                            ln.get("headway_min"), ln.get("speed_kmh"), ln.get("dwell_s")))
    return out


def as_request(lines: list[LineSpec]) -> list[dict]:
    return [{"line_id": ln.line_id, "name": ln.name,
             "stations": [{"lat": a, "lng": b} for a, b in ln.stations],
             "headway_min": ln.headway_min, "speed_kmh": ln.speed_kmh, "dwell_s": ln.dwell_s}
            for ln in lines]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--runs", type=int, default=None)
    ap.add_argument("--only", default="")
    ap.add_argument("--no-mc", action="store_true")
    args = ap.parse_args()
    params = Params.load()
    world = load_world(processed_dir())
    out_dir = scenarios_dir()
    out_dir.mkdir(parents=True, exist_ok=True)
    config = yaml.safe_load((ROOT / "config" / "scenarios.yaml").read_text())
    only = {x for x in args.only.split(",") if x}
    synthetic = (world.path / "SYNTHETIC").exists()
    for sc in config["scenarios"]:
        if only and sc["id"] not in only:
            continue
        lines = [underserved_line(world)] if sc.get("auto") == "underserved" else spec_lines(sc)
        if not lines or any(len(ln.stations) < 2 for ln in lines):
            print(f"skip {sc['id']}: needs station coordinates in config/scenarios.yaml")
            continue
        t = time.perf_counter()
        attribution = bool(sc.get("attribution")) or len(lines) >= 2
        result = evaluate(world, lines, params, attribution=attribution)
        doc = {"id": sc["id"], "name": sc["name"], "description": sc.get("description", ""),
               "lines": as_request(lines), "result": result,
               "generated_at": time.strftime("%Y-%m-%d %H:%M"), "synthetic_data": synthetic}
        if len(lines) >= 2:
            doc["alone"] = {ln.line_id: evaluate(world, [ln], params)["totals"] for ln in lines}
        if not args.no_mc:
            doc["montecarlo"] = montecarlo.run(world, lines, params, n_runs=args.runs,
                                               progress=True)
        (out_dir / f"{sc['id']}.json").write_text(json.dumps(doc))
        print(f"{sc['id']}: written in {time.perf_counter() - t:.0f} s "
              f"(riders {result['totals']['riders_daily']:,.0f}/day)")


if __name__ == "__main__":
    main()
