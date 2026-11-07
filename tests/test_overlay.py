import numpy as np
import pytest

from khatt.engine.multiline import joint_rstar, wait_board
from khatt.engine.overlay import Overlay
from tests.toy import brute_force_T1, engine_T1, toy_base, toy_line


@pytest.fixture(scope="module")
def base():
    return toy_base(n=26, seed=3)


def _lines(lat, lng):
    return [
        toy_line("A", [0, 5, 9, 14], lat, lng),
        toy_line("B", [2, 9, 20], lat, lng, headway=6.0, speed=30.0),
        toy_line("C", [17, 11, 3, 22, 25], lat, lng, headway=3.0, dwell=45.0),
    ]


@pytest.mark.parametrize("n_lines", [1, 2, 3])
def test_matches_brute_force(base, params, n_lines):
    lat, lng, w, T0 = base
    lines = _lines(lat, lng)[:n_lines]
    nodes = [[0, 5, 9, 14], [2, 9, 20], [17, 11, 3, 22, 25]][:n_lines]
    station_hex = np.array([k for ns in nodes for k in ns])
    T1 = engine_T1(T0, lines, station_hex, params)
    expected = brute_force_T1(w, lines, nodes, params)
    np.testing.assert_allclose(T1, expected, atol=1e-4, rtol=1e-6)


@pytest.mark.parametrize("seed", [0, 1, 2, 7])
def test_brute_force_random_graphs(params, seed):
    lat, lng, w, T0 = toy_base(n=30, seed=seed)
    rng = np.random.default_rng(seed + 100)
    nodes = [list(rng.choice(30, 5, replace=False)), list(rng.choice(30, 4, replace=False))]
    lines = [toy_line(chr(65 + k), ns, lat, lng) for k, ns in enumerate(nodes)]
    station_hex = np.array([k for ns in nodes for k in ns])
    T1 = engine_T1(T0, lines, station_hex, params)
    np.testing.assert_allclose(T1, brute_force_T1(w, lines, nodes, params), atol=1e-4, rtol=1e-6)


def test_never_slower(base, params):
    lat, lng, _, T0 = base
    lines = _lines(lat, lng)
    station_hex = np.array([0, 5, 9, 14, 2, 9, 20, 17, 11, 3, 22, 25])
    T1 = engine_T1(T0, lines, station_hex, params)
    assert (T1 <= T0 + 1e-4).all()
    assert (T1 < T0 - 1).any(), "a fast line should speed up at least one trip"


def test_no_lines_is_identity(base, params):
    T0 = base[3]
    ov = Overlay(T0, np.zeros(0, np.int64), np.zeros(0), np.zeros((0, 0)))
    _t0, t1 = ov.to(np.arange(T0.shape[0]))
    np.testing.assert_array_equal(t1, T0)
    np.testing.assert_array_equal(engine_T1(T0, [], np.zeros(0, int), params), T0)


def test_order_independent(base, params):
    lat, lng, _, T0 = base
    lines = _lines(lat, lng)
    nodes = [[0, 5, 9, 14], [2, 9, 20], [17, 11, 3, 22, 25]]
    order = [2, 0, 1]
    hex_a = np.array([k for ns in nodes for k in ns])
    hex_b = np.array([k for i in order for k in nodes[i]])
    T1a = engine_T1(T0, lines, hex_a, params)
    T1b = engine_T1(T0, [lines[i] for i in order], hex_b, params)
    np.testing.assert_allclose(T1a, T1b, atol=1e-5)


def test_pairs_match_full(base, params):
    lat, lng, _, T0 = base
    lines = _lines(lat, lng)[:2]
    station_hex = np.array([0, 5, 9, 14, 2, 9, 20])
    rstar = joint_rstar(lines, station_hex, T0, params)
    ov = Overlay(T0, station_hex, wait_board(lines, params), rstar)
    _, T1 = ov.to(np.arange(T0.shape[0]))
    rng = np.random.default_rng(0)
    a, b = rng.integers(0, T0.shape[0], 50), rng.integers(0, T0.shape[0], 50)
    t0, t1 = ov.pairs(a, b)
    np.testing.assert_allclose(t1, T1[a, b], atol=1e-6)
    np.testing.assert_allclose(t0, T0[a, b])


def test_destination_subset_matches_full(base, params):
    lat, lng, _, T0 = base
    lines = _lines(lat, lng)
    station_hex = np.array([0, 5, 9, 14, 2, 9, 20, 17, 11, 3, 22, 25])
    rstar = joint_rstar(lines, station_hex, T0, params)
    ov = Overlay(T0, station_hex, wait_board(lines, params), rstar)
    _, T1_full = ov.to(np.arange(T0.shape[0]))
    dest = np.array([3, 7, 19])
    _, T1_sub = ov.to(dest)
    np.testing.assert_allclose(T1_sub, T1_full[:, dest])
