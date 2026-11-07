import json

import numpy as np
import pytest

from khatt.engine.lines import LineSpec
from khatt.evaluate import evaluate


def test_response_shape(world, params, downtown_line):
    out = evaluate(world, [downtown_line], params)
    for key in ("elapsed_ms", "warnings", "totals", "streets", "stations", "hexes", "flows",
                "microbus_routes", "synergy", "attribution"):
        assert key in out
    t = out["totals"]
    for key in ("dR_egp", "passby_egp", "riders_daily", "streets_gaining", "streets_losing",
                "red_streets", "unbranded_at_risk", "microbus"):
        assert key in t
    s = out["streets"][0]
    for key in ("street_id", "name", "dR_egp", "dR_pct", "by_category", "passby_egp",
                "rent_pct", "revenue_pct", "risk", "local_loser", "unbranded_at_risk"):
        assert key in s
    assert len(out["stations"]) == len(downtown_line.stations)
    payload = json.dumps(out)  # native types only
    assert len(payload) < 2_000_000


def test_no_lines(world, params):
    out = evaluate(world, [], params)
    assert out["totals"]["dR_egp"] == 0
    assert out["streets"] == [] or all(s["dR_egp"] == 0 for s in out["streets"])
    assert out["stations"] == []


def test_order_independent(world, params, downtown_line, cross_line):
    a = evaluate(world, [downtown_line, cross_line], params)
    b = evaluate(world, [cross_line, downtown_line], params)
    da = {s["street_id"]: s["dR_egp"] for s in a["streets"]}
    db = {s["street_id"]: s["dR_egp"] for s in b["streets"]}
    assert da.keys() == db.keys()
    for k, v in da.items():
        assert abs(v - db[k]) <= 1 + 1e-6 * abs(v)
    assert np.isclose(a["totals"]["riders_daily"], b["totals"]["riders_daily"], rtol=1e-6)


def test_local_loser_flag(world, params, downtown_line):
    out = evaluate(world, [downtown_line], params)
    for s in out["streets"]:
        if s["local_loser"]:
            assert s["dR_egp"] < 0


def test_line_defaults_and_overrides(world, params):
    spec = LineSpec("X", "X", [(30.03, 31.22), (30.05, 31.25)], headway_min=10, speed_kmh=None)
    out = evaluate(world, [spec], params)
    assert out["lines"][0]["headway_min"] == 10
    assert out["lines"][0]["speed_kmh"] == params["line.speed_kmh"]


def test_faster_line_moves_more(world, params, downtown_line):
    slow = LineSpec(**{**downtown_line.__dict__, "speed_kmh": 15.0})
    fast = LineSpec(**{**downtown_line.__dict__, "speed_kmh": 60.0})
    assert (evaluate(world, [fast], params)["totals"]["abs_dR_egp"]
            > evaluate(world, [slow], params)["totals"]["abs_dR_egp"])


def test_invalid_lines_raise(world, params):
    with pytest.raises(ValueError):
        evaluate(world, [LineSpec("A", "A", [(30.0, 31.2)])], params)
