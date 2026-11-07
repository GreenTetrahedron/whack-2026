"""Joint station graph for all drawn lines → Rstar (on board at i → alighted at j).

Exact formulation (no dwell double counting when chaining stops on one line):
  L[i, j]  ride on the same line (from lines.ride_matrix; inf across lines; 0 on diagonal)
  X[a, b]  transfer: alight at a, travel by baseline network, wait and board at b
           = T0[hex_a, hex_b] + wait_b + board_penalty   (a != b)
  M  = X ⊗ L         (alighted at a → alighted at c with one transfer + one ride)
  M* = Floyd–Warshall closure of M with zero diagonal
  Rstar = L ⊗ M*     (⊗ = min-plus product)
"""

from __future__ import annotations

import numpy as np

from khatt.config import Params
from khatt.engine.lines import LineSpec, Station, ride_matrix

INF = np.inf


def minplus(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """(a ⊗ b)[i, j] = min_k a[i, k] + b[k, j]."""
    return (a[:, :, None] + b[None, :, :]).min(axis=1)


def floyd_warshall(m: np.ndarray) -> np.ndarray:
    d = m.copy()
    np.fill_diagonal(d, 0.0)
    for k in range(d.shape[0]):
        np.minimum(d, d[:, k : k + 1] + d[k : k + 1, :], out=d)
    return d


def stations_of(lines: list[LineSpec]) -> list[Station]:
    out: list[Station] = []
    for ln in lines:
        for k, (lat, lng) in enumerate(ln.stations):
            out.append(Station(line_id=ln.line_id, idx=k, lat=lat, lng=lng, global_idx=len(out)))
    return out


def same_line_matrix(lines: list[LineSpec]) -> np.ndarray:
    """Block-diagonal ride matrix L (inf between different lines)."""
    sizes = [len(ln.stations) for ln in lines]
    total = sum(sizes)
    big = np.full((total, total), INF)
    start = 0
    for ln, size in zip(lines, sizes):
        big[start : start + size, start : start + size] = ride_matrix(
            ln.stations, ln.speed_kmh, ln.dwell_s
        )
        start += size
    return big


def wait_board(lines: list[LineSpec], params: Params) -> np.ndarray:
    """Per-station expected wait + boarding penalty (minutes)."""
    irregularity = float(params["routing.irregularity_factor.metro"])
    board = float(params["line.board_penalty_min"])
    return np.concatenate(
        [np.full(len(ln.stations), ln.headway_min / 2.0 * irregularity + board) for ln in lines]
    )


def joint_rstar(
    lines: list[LineSpec], station_hex: np.ndarray, T0: np.ndarray, params: Params,
    access: np.ndarray | None = None,
) -> np.ndarray:
    """Rstar[i, j]: best time from on board at i to alighted at j (minutes, float64).

    `access[i]` is extra walking between station i and its cell centre (0 for stations inside
    a modelled cell). It is added on egress (column j) and to both ends of every transfer;
    add it to wait_board for the boarding side.
    """
    L = same_line_matrix(lines)
    S = L.shape[0]
    if S == 0:
        return np.zeros((0, 0))
    acc = np.zeros(S) if access is None else np.asarray(access, float)
    wb = wait_board(lines, params)
    X = (T0[np.ix_(station_hex, station_hex)].astype(np.float64)
         + acc[:, None] + acc[None, :] + wb[None, :])
    np.fill_diagonal(X, INF)
    M = minplus(X, L)
    Mstar = floyd_warshall(M)
    rstar = minplus(L, Mstar)
    np.fill_diagonal(rstar, 0.0)
    return rstar + acc[None, :]
