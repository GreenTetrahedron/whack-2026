import numpy as np

from khatt.engine.lines import LineSpec, line_length_m, ride_matrix, validate_lines
from khatt.engine.multiline import floyd_warshall, joint_rstar, minplus, wait_board
from khatt.geo import haversine_m
from tests.toy import engine_T1


def _grid_world(n_side=9, spacing_deg=0.01):
    """Square grid of 'cells' with walking-only T0 (Manhattan × 1.3 / 4.5 km/h)."""
    lat0, lng0 = 30.0, 31.2
    ii, jj = np.meshgrid(np.arange(n_side), np.arange(n_side), indexing="ij")
    lat = (lat0 + ii * spacing_deg).ravel()
    lng = (lng0 + jj * spacing_deg).ravel()
    d = haversine_m(lat[:, None], lng[:, None], lat[None, :], lng[None, :])
    T0 = (d * 1.3 / 75.0).astype(np.float32)
    np.fill_diagonal(T0, 4.0)
    return lat, lng, T0


def test_ride_matrix_dwell_and_symmetry():
    stations = [(30.0, 31.2), (30.0, 31.21), (30.0, 31.22), (30.0, 31.23)]
    ride = ride_matrix(stations, speed_kmh=36.0, dwell_s=30.0)
    assert np.allclose(ride, ride.T)
    assert np.allclose(np.diag(ride), 0)
    seg = haversine_m(30.0, 31.2, 30.0, 31.21)
    assert np.isclose(ride[0, 1], seg / 600.0)
    # two intermediate stops between 0 and 3
    assert np.isclose(ride[0, 3], 3 * seg / 600.0 + 2 * 0.5, rtol=1e-6)
    assert np.isclose(line_length_m(stations), 3 * seg, rtol=1e-9)


def test_validate_lines(params):
    good = LineSpec("A", "A", [(30.0, 31.2), (30.01, 31.21)])
    validate_lines([good], params)
    import pytest

    with pytest.raises(ValueError):
        validate_lines([LineSpec("A", "A", [(30.0, 31.2)])], params)
    with pytest.raises(ValueError):
        validate_lines([good, LineSpec("A", "B", [(30.0, 31.2), (30.1, 31.2)])], params)
    with pytest.raises(ValueError):
        validate_lines([LineSpec("A", "A", [(130.0, 31.2), (30.1, 31.2)])], params)


def test_minplus_and_fw():
    inf = np.inf
    m = np.array([[0, 1, inf], [inf, 0, 2], [5, inf, 0]], float)
    fw = floyd_warshall(m)
    assert fw[0, 2] == 3 and fw[2, 1] == 6
    assert np.array_equal(minplus(np.eye(2) * 0, np.ones((2, 2))), np.ones((2, 2)))


def test_two_crossing_lines_transfer_beats_either(params):
    lat, lng, T0 = _grid_world()
    idx = lambda i, j: i * 9 + j
    west_east = [idx(4, j) for j in range(9)]
    south_north = [idx(i, 4) for i in range(9)]
    a = LineSpec("A", "A", [(lat[k], lng[k]) for k in west_east], 4.0, 35.0, 30.0)
    b = LineSpec("B", "B", [(lat[k], lng[k]) for k in south_north], 4.0, 35.0, 30.0)
    o, d = idx(4, 0), idx(8, 4)  # west end of A → north end of B
    T1_a = engine_T1(T0, [a], np.array(west_east), params)
    T1_b = engine_T1(T0, [b], np.array(south_north), params)
    T1_ab = engine_T1(T0, [a, b], np.array(west_east + south_north), params)
    assert T1_ab[o, d] < T1_a[o, d] - 1
    assert T1_ab[o, d] < T1_b[o, d] - 1
    assert T1_ab[o, d] < T0[o, d]


def test_rstar_symmetric_for_symmetric_inputs(params):
    lat, lng, T0 = _grid_world()
    idx = lambda i, j: i * 9 + j
    l1 = [idx(1, j) for j in range(0, 9, 2)]
    l2 = [idx(i, 3) for i in range(0, 9, 2)]
    l3 = [idx(i, i) for i in range(0, 9, 3)]
    lines = [
        LineSpec(n, n, [(lat[k], lng[k]) for k in ks], 4.0, 35.0, 30.0)
        for n, ks in zip("ABC", [l1, l2, l3])
    ]
    hexes = np.array(l1 + l2 + l3)
    rstar = joint_rstar(lines, hexes, T0, params)
    assert np.allclose(rstar, rstar.T, atol=1e-9)
    assert np.allclose(np.diag(rstar), 0)
    assert np.isfinite(rstar).all(), "every station reachable through transfers"


def test_rstar_same_line_equals_ride(params):
    lat, lng, T0 = _grid_world()
    ks = [0, 10, 20, 30, 40]
    line = LineSpec("A", "A", [(lat[k], lng[k]) for k in ks], 4.0, 35.0, 30.0)
    rstar = joint_rstar([line], np.array(ks), T0, params)
    assert np.allclose(rstar, ride_matrix(line.stations, 35.0, 30.0))


def test_wait_board(params):
    a = LineSpec("A", "A", [(30, 31), (30.01, 31)], 4.0, 35.0, 30.0)
    b = LineSpec("B", "B", [(30, 31), (30.01, 31), (30.02, 31)], 10.0, 35.0, 30.0)
    wb = wait_board([a, b], params)
    board = params["line.board_penalty_min"]
    assert np.allclose(wb, [2 + board] * 2 + [5 + board] * 3)
