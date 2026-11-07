"""Write example /baseline and /evaluate responses (synthetic data) for front-end work.

    python scripts/write_mock.py data/synthetic scenarios/mock
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from khatt.api.main import build_baseline
from khatt.config import Params
from khatt.data import load_world
from khatt.engine.lines import LineSpec
from khatt.evaluate import evaluate


def main() -> None:
    src = Path(sys.argv[1] if len(sys.argv) > 1 else "data/synthetic")
    out = Path(sys.argv[2] if len(sys.argv) > 2 else "scenarios/mock")
    out.mkdir(parents=True, exist_ok=True)
    world = load_world(src)
    params = Params.load(calibrated=src / "calibrated.json")
    a = LineSpec("A", "Line A", [(29.95 + 0.009 * k, 31.15 + 0.008 * k) for k in range(20)])
    b = LineSpec("B", "Line B", [(30.13 - 0.008 * k, 31.12 + 0.011 * k) for k in range(20)])
    request = {"lines": [{"line_id": ln.line_id, "name": ln.name,
                          "stations": [{"lat": x, "lng": y} for x, y in ln.stations]}
                         for ln in (a, b)], "attribution": True}
    (out / "evaluate_request.json").write_text(json.dumps(request, indent=1))
    (out / "evaluate.json").write_text(json.dumps(evaluate(world, [a, b], params,
                                                           attribution=True)))
    (out / "baseline.json").write_text(json.dumps(build_baseline(world, params)))
    print(f"mock responses written to {out} (synthetic data, not Cairo)")


if __name__ == "__main__":
    main()
