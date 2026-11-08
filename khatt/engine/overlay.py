"""Min-plus overlay of drawn lines on the baseline matrix (numba, parallel over origins).

T1[o, d] = min( T0[o, d], min_{i,j} T0[o, s_i] + wait_board[i] + Rstar[i, j] + T0[s_j, d] )

Split in two passes so cost is O(N·S² + N·S·D):
  via[o, j] = min_i T0[o, s_i] + wait_board[i] + Rstar[i, j]   (best arrival, alighted at j)
  T1[o, d]  = min(T0[o, d], min_j via[o, j] + T0[s_j, d])
"""

from __future__ import annotations

import numpy as np
from numba import njit, prange


@njit(parallel=True, fastmath=False, cache=True)
def compute_via(T0_os, wait_board, Rstar):
    N, S = T0_os.shape
    via = np.empty((N, S), np.float32)
    for o in prange(N):
        for j in range(S):
            via[o, j] = np.float32(1e9)
        for i in range(S):
            a = T0_os[o, i] + wait_board[i]
            for j in range(S):
                v = a + Rstar[i, j]
                via[o, j] = min(via[o, j], v)
    return via


@njit(parallel=True, fastmath=False, cache=True)
def overlay_from_via(T0_od, via, T0_sd):
    N, D = T0_od.shape
    S = via.shape[1]
    T1 = T0_od.copy()
    for o in prange(N):
        for d in range(D):
            best = T1[o, d]
            for j in range(S):
                v = via[o, j] + T0_sd[j, d]
                best = min(best, v)
            T1[o, d] = best
    return T1


def overlay(T0_od, T0_os, T0_sd, wait_board, Rstar):
    """Spec signature (CLAUDE.md §7). All inputs float32."""
    if Rstar.shape[0] == 0:
        return T0_od.copy()
    via = compute_via(
        np.ascontiguousarray(T0_os, np.float32),
        np.ascontiguousarray(wait_board, np.float32),
        np.ascontiguousarray(Rstar, np.float32),
    )
    return overlay_from_via(
        np.ascontiguousarray(T0_od, np.float32), via, np.ascontiguousarray(T0_sd, np.float32)
    )


@njit(parallel=True, fastmath=False, cache=True)
def overlay_pairs(T0, a_idx, b_idx, via, station_hex):
    """T1 for an explicit list of (a, b) pairs. `via` rows are indexed by a (full N rows)."""
    P = a_idx.shape[0]
    S = station_hex.shape[0]
    out = np.empty(P, np.float32)
    for p in prange(P):
        a = a_idx[p]
        b = b_idx[p]
        best = T0[a, b]
        for j in range(S):
            v = via[a, j] + T0[station_hex[j], b]
            best = min(best, v)
        out[p] = best
    return out


class Overlay:
    """Holds `via` for one scenario so callers can ask for any destination subset."""

    def __init__(self, T0: np.ndarray, station_hex: np.ndarray, wait_board, Rstar):
        self.T0 = T0
        self.station_hex = np.asarray(station_hex, np.int64)
        self.S = len(self.station_hex)
        self.wait_board = np.asarray(wait_board, np.float32)  # kept for the zones model
        self.Rstar = np.asarray(Rstar, np.float32)
        if self.S:
            self.via = compute_via(
                np.ascontiguousarray(T0[:, self.station_hex], np.float32),
                np.ascontiguousarray(wait_board, np.float32),
                np.ascontiguousarray(Rstar, np.float32),
            )
        else:
            self.via = np.zeros((T0.shape[0], 0), np.float32)

    def to(self, dest_idx: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """(T0[:, dest], T1[:, dest]) for a destination subset."""
        dest_idx = np.asarray(dest_idx, np.int64)
        T0_od = np.ascontiguousarray(self.T0[:, dest_idx])
        if self.S == 0:
            return T0_od, T0_od.copy()
        T0_sd = np.ascontiguousarray(self.T0[np.ix_(self.station_hex, dest_idx)])
        return T0_od, overlay_from_via(T0_od, self.via, T0_sd)

    def pairs(self, a_idx: np.ndarray, b_idx: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        a_idx = np.asarray(a_idx, np.int64)
        b_idx = np.asarray(b_idx, np.int64)
        t0 = self.T0[a_idx, b_idx]
        if self.S == 0 or len(a_idx) == 0:
            return t0, t0.copy()
        return t0, overlay_pairs(self.T0, a_idx, b_idx, self.via, self.station_hex)


def warmup() -> None:
    """Trigger JIT compilation (or load from cache) so the first request is fast."""
    T0 = np.ones((3, 3), np.float32)
    ov = Overlay(T0, np.array([0, 1]), np.ones(2, np.float32), np.zeros((2, 2), np.float32))
    ov.to(np.array([0, 2]))
    ov.pairs(np.array([0]), np.array([2]))
