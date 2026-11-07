import numpy as np

from khatt.attribution import shapley
from khatt.engine.lines import LineSpec
from khatt.evaluate import evaluate


def test_shapley_additive_game():
    # v(S) = sum of member weights → Shapley value = own weight
    w = np.array([[1.0, 2.0], [3.0, -1.0], [0.5, 0.5]])
    values = {m: sum(w[i] for i in range(3) if m >> i & 1) + np.zeros(2) for m in range(8)}
    np.testing.assert_allclose(shapley(values, 3), w)


def test_shapley_parts_sum_to_joint():
    rng = np.random.default_rng(0)
    values = {m: rng.normal(size=5) for m in range(16)}
    values[0] = np.zeros(5)
    phi = shapley(values, 4)
    np.testing.assert_allclose(phi.sum(axis=0), values[15])


def test_evaluate_attribution(world, params, downtown_line, cross_line):
    out = evaluate(world, [downtown_line, cross_line], params, attribution=True)
    assert len(out["synergy"]) == 1
    assert out["synergy"][0]["lines"] == ["A", "B"]
    joint = {str(s["street_id"]): s["dR_egp"] for s in out["streets"]}
    assert out["attribution"]
    for sid, parts in out["attribution"].items():
        assert set(parts) == {"A", "B"}
        total = sum(parts.values())
        assert abs(total - joint[sid]) <= 1e-3 * abs(joint[sid]) + 1.0


def test_attribution_skipped_above_four_lines(world, params):
    lines = [LineSpec(str(k), str(k), [(30.0 + 0.01 * k, 31.2), (30.0 + 0.01 * k, 31.26)])
             for k in range(5)]
    out = evaluate(world, lines, params, attribution=True)
    assert out["attribution"] == {}
    assert any("Attribution skipped" in w for w in out["warnings"])


def test_attribution_off_by_default(world, params, downtown_line, cross_line):
    out = evaluate(world, [downtown_line, cross_line], params)
    assert out["synergy"] == [] and out["attribution"] == {}
