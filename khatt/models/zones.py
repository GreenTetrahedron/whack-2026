"""Station zones (area of effect): where residents are likely to use each new station.

For a res-9 cell c and new station s, P_use[c, s] is the chance that a typical trip by a resident
of c boards the new line at s:

  T_line[c, d] = min_i access[c, i] + wait_board[i] + G[i, d],   G[i, d] = min_j Rstar[i, j] + T0[s_j, d]
  P_line[c, d] = clip((T0[c, d] - T_line[c, d] - min_saving) / (full_saving - min_saving), 0, 1)
  P_use[c, s]  = sum_k weight_k * sum_d w_k[c, d] * P_line[c, d] * [argmin_i = s]

access[c, i] is the walk time on the walk graph (up to zones.walk_max_min) or, if shorter, the
baseline time T0 from c's parent res-8 cell to the station's cell (e.g. a microbus to the station).
T0[c, d] is the parent's T0, or walking to a station's area and continuing on today's network if
that is faster, so walk precision never favours the line. Paths must ride at least one stop.
w_k is the destination choice for purpose k over scenario times T1 = min(T0, T_line):
work trips go to mapped places (jobs stand-in, n_places · exp(-beta_work · T1)); shopping trips
use the Huff model of model 1 (sum of street A^alpha · exp(-beta_c · T1)). weight_k blends them:
zones.work_trip_share for work, the rest split by the origin's spending mix.

Each station's circle is the largest radius whose population-weighted mean P_use is at least
zones.threshold, while the band just inside the edge (zones.edge_band_m) still averages at least
zones.edge_share × threshold, so a busy town 3 km out cannot stretch the circle over empty rings. The ramp (not a logit) gives exactly 0 to trips the line does not speed up.
"""

from __future__ import annotations

from dataclasses import dataclass

import h3
import numpy as np
from numba import njit, prange

from khatt.config import CATEGORIES, Params

R9_EQUIV_RADIUS_M = float(np.sqrt(h3.average_hexagon_area(9, unit="m^2") / np.pi))


@dataclass
class ZonesResult:
    radius_m: np.ndarray  # (S,) circle radius, 0 = no zone
    users: np.ndarray  # (S,) sum over nearby cells of residents × P_use
    pop_in: np.ndarray  # (S,) residents inside the circle
    mean_p: np.ndarray  # (S,) population-weighted mean P_use inside the circle
    cells: np.ndarray  # candidate r9 rows
    p_use: np.ndarray  # (len(cells), S)


def empty(S: int) -> ZonesResult:
    z = np.zeros(S)
    return ZonesResult(z, z.copy(), z.copy(), z.copy(), np.zeros(0, np.int64), np.zeros((0, S)))


# ---------------------------------------------------------------- kernels
@njit(parallel=True, fastmath=False, cache=True)
def best_line(A, G):
    """A: (P, S) access + wait/board per origin; G: (S, D). → T_line (P, D), boarding station."""
    P, S = A.shape
    D = G.shape[1]
    T = np.empty((P, D), np.float32)
    arg = np.empty((P, D), np.int32)
    for p in prange(P):
        for d in range(D):
            best = np.float32(1e9)
            a = -1
            for i in range(S):
                v = A[p, i] + G[i, d]
                if v < best:
                    best = v
                    a = i
            T[p, d] = best
            arg[p, d] = a
    return T, arg


@njit(parallel=True, fastmath=False, cache=True)
def row_use(Tb, TL, ARG, row_parent, ex_ptr, ex_station, ex_cost, ex_base, G, H, mass, beta,
            pw, max_trip, min_saving, ramp):
    """P_use per row. A row is an origin: a parent res-8 cell (no extras) or a res-9 cell with
    extra walk-access options that beat the parent's T0 access to a station.

    Tb, TL, ARG: (P, D) baseline time, line time and boarding station per parent.
    Extra e: station ex_station[e]; line access ex_cost[e] (walk + wait/board) and baseline
    access ex_base[e] (walk to the station's area, then today's network H = T0[s_i, d]), so the
    finer walk helps today's trip as much as the line's and only riding can win.
    mass: (K, D) destination mass per purpose; beta: (K,); pw: (P, K) purpose weights.
    """
    R = row_parent.shape[0]
    D = Tb.shape[1]
    K = mass.shape[0]
    S = G.shape[0]
    out = np.zeros((R, S), np.float64)
    for r in prange(R):
        p = row_parent[r]
        tot = np.zeros(K)
        acc = np.zeros((K, S))
        for d in range(D):
            tl = TL[p, d]
            a = ARG[p, d]
            for e in range(ex_ptr[r], ex_ptr[r + 1]):
                v = ex_cost[e] + G[ex_station[e], d]
                if v < tl:
                    tl = v
                    a = ex_station[e]
            tb = Tb[p, d]
            for e in range(ex_ptr[r], ex_ptr[r + 1]):
                tb = min(tb, ex_base[e] + H[ex_station[e], d])
            t1 = min(tb, tl)
            if t1 > max_trip:
                continue
            pl = (tb - tl - min_saving) / ramp
            pl = min(max(pl, 0.0), 1.0)
            for k in range(K):
                m = mass[k, d]
                if m <= 0.0:
                    continue
                w = m * np.exp(-beta[k] * t1)
                tot[k] += w
                if pl > 0.0:
                    acc[k, a] += w * pl
        wsum = 0.0
        for k in range(K):
            if tot[k] > 0.0:
                wsum += pw[p, k]
        if wsum <= 0.0:
            continue
        for k in range(K):
            if tot[k] > 0.0:
                f = pw[p, k] / wsum / tot[k]
                for s in range(S):
                    out[r, s] += f * acc[k, s]
    return out


def warmup() -> None:
    """Compile the kernels (or load them from the numba cache) before the first request."""
    A, G = np.ones((2, 2), np.float32), np.ones((2, 3), np.float32)
    TL, ARG = best_line(A, G)
    one = np.ones(1, np.float32)
    row_use(np.ones((2, 3), np.float32), TL, ARG, np.array([0, 1]), np.array([0, 0, 1]),
            np.array([0]), one, one, G, G, np.ones((1, 3)), np.ones(1), np.ones((2, 1)),
            120.0, 2.0, 8.0)


# ---------------------------------------------------------------- inputs
def destination_mass(world, params: Params) -> tuple[np.ndarray, np.ndarray]:
    """→ (dest cell idx, mass (K, D)) for K = work + the 3 shopping categories."""
    N = world.n
    alpha = float(params["spending.alpha"])
    with np.errstate(divide="ignore"):
        A_alpha = np.where(world.street_A > 0, world.street_A ** alpha, 0.0)
    shop = np.column_stack([np.bincount(world.street_hex, weights=A_alpha[:, c], minlength=N)[:N]
                            for c in range(len(CATEGORIES))])
    dest = np.flatnonzero((world.places > 0) | (shop.sum(axis=1) > 0))
    mass = np.vstack([world.places[dest][None, :], shop[dest].T])
    return dest, mass


def purpose_weights(world, params: Params, parents: np.ndarray) -> np.ndarray:
    work = float(params["zones.work_trip_share"]) if world.places.sum() > 0 else 0.0
    S = world.spend[parents]
    total = S.sum(axis=1, keepdims=True)
    mix = np.divide(S, total, out=np.full_like(S, 1.0 / S.shape[1]), where=total > 0)
    return np.column_stack([np.full(len(parents), work), (1.0 - work) * mix])


def betas(params: Params) -> np.ndarray:
    return np.concatenate([[float(params["zones.beta_work"])], params.beta])


def line_times(world, ov, parents: np.ndarray, dest: np.ndarray):
    """Baseline Tb (P, D), access A (P, S), G (S, D), line time TL (P, D) and boarding station.

    Paths must ride at least one stop: boarding and alighting at the same station is not a
    train trip, and with the fine walk access of res-9 cells it would act as a walking shortcut.
    Otherwise min(Tb, TL) is the overlay's T1 for these origins and destinations."""
    T0p = world.T0[parents]
    Tb = np.ascontiguousarray(T0p[:, dest], np.float32)
    A = np.ascontiguousarray(T0p[:, ov.station_hex] + ov.wait_board[None, :], np.float32)
    ride = ov.Rstar.astype(np.float32).copy()
    np.fill_diagonal(ride, np.inf)
    G = (ride[:, :, None] + world.T0[np.ix_(ov.station_hex, dest)][None, :, :]).min(axis=1)
    G = np.ascontiguousarray(G, np.float32)
    TL, ARG = best_line(A, G)
    return Tb, A, G, TL, ARG


# ---------------------------------------------------------------- main entry
def run(world, params: Params, sc, walk_catches) -> ZonesResult:
    """sc: evaluate.Scenario; walk_catches: station catchments with cutoff zones.walk_max_min."""
    S = len(sc.stations)
    if S == 0 or len(world.r9) == 0:
        return empty(S)
    ov = sc.overlay
    max_r = float(params["zones.max_radius_m"])
    sxy = np.column_stack([sc.station_x, sc.station_y])

    # candidate origins: parents within max radius (+ one cell) of any station, and their r9 cells
    hits = world.hex_tree.query_ball_point(sxy, r=max_r + 1000.0)
    parents = np.unique(np.concatenate([np.asarray(h, np.int64) for h in hits]))
    if len(parents) == 0:
        return empty(S)
    ppos = np.full(world.n, -1, np.int64)
    ppos[parents] = np.arange(len(parents))
    cells = np.flatnonzero(ppos[world.r9_parent] >= 0)

    dest, mass = destination_mass(world, params)
    Tb, A, G, TL, ARG = line_times(world, ov, parents, dest)
    H = np.ascontiguousarray(world.T0[np.ix_(ov.station_hex, dest)], np.float32)

    # walk access that beats the parent's T0 access → extra options for that r9 cell
    access = np.asarray(sc.access, np.float32)
    wb_walk = ov.wait_board - access  # walk straight to the platform
    extras: dict[int, list[tuple[int, float, float]]] = {}
    for i, c in enumerate(walk_catches):
        if len(c.r9_idx) == 0:
            continue
        cost = c.r9_min + wb_walk[i]
        par = ppos[world.r9_parent[c.r9_idx]]
        ok = par >= 0
        better = ok & (cost < np.where(ok, A[np.maximum(par, 0), i], np.inf))
        for cell, v, walk in zip(c.r9_idx[better], cost[better], c.r9_min[better]):
            extras.setdefault(int(cell), []).append((i, float(v), float(walk + access[i])))
    ex_cells = np.array(sorted(extras), np.int64)
    row_parent = np.concatenate([np.arange(len(parents)), ppos[world.r9_parent[ex_cells]]])
    counts = np.concatenate([np.zeros(len(parents), np.int64),
                             [len(extras[c]) for c in ex_cells]]).astype(np.int64)
    ex_ptr = np.concatenate([[0], np.cumsum(counts)]).astype(np.int64)
    flat = [e for c in ex_cells for e in extras[c]]
    ex_station = np.array([e[0] for e in flat], np.int64)
    ex_cost = np.array([e[1] for e in flat], np.float32)
    ex_base = np.array([e[2] for e in flat], np.float32)

    lo, hi = float(params["zones.min_saving_min"]), float(params["zones.full_saving_min"])
    use = row_use(Tb, TL, ARG, row_parent.astype(np.int64), ex_ptr, ex_station, ex_cost, ex_base,
                  G, H, np.ascontiguousarray(mass, np.float64), betas(params),
                  purpose_weights(world, params, parents),
                  float(params["spending.max_trip_min"]), lo, max(hi - lo, 1e-6))

    # r9 cell → its row (own row if it has walk extras, else its parent's)
    row_of = ppos[world.r9_parent[cells]]
    if len(ex_cells):
        own = np.minimum(np.searchsorted(ex_cells, cells), len(ex_cells) - 1)
        row_of = np.where(ex_cells[own] == cells, len(parents) + own, row_of)
    p_use = use[row_of]
    return circles(world, params, sxy, cells, p_use)


def circles(world, params: Params, sxy: np.ndarray, cells: np.ndarray,
            p_use: np.ndarray) -> ZonesResult:
    S = sxy.shape[0]
    thr = float(params["zones.threshold"])
    edge_thr = thr * float(params["zones.edge_share"])
    band = float(params["zones.edge_band_m"])
    max_r = float(params["zones.max_radius_m"])
    rx, ry = world.r9_xy()
    rx, ry = rx[cells], ry[cells]
    pop = world.r9_pop[cells]
    radius, users, pop_in, mean_p = (np.zeros(S) for _ in range(4))
    for s in range(S):
        users[s] = float((pop * p_use[:, s]).sum())
        dist = np.hypot(rx - sxy[s, 0], ry - sxy[s, 1])
        near = np.flatnonzero((dist <= max_r) & (pop > 0))
        if len(near) == 0:
            continue
        order = near[np.argsort(dist[near], kind="stable")]
        d_sorted = dist[order]
        cum_pop = np.cumsum(pop[order])
        cum_pw = np.cumsum(pop[order] * p_use[order, s])
        cum_mean = cum_pw / cum_pop
        # mean chance in the band just inside each radius: the edge must still matter
        start = np.searchsorted(d_sorted, d_sorted - band, side="right")
        prev_pw = np.where(start > 0, cum_pw[np.maximum(start - 1, 0)], 0.0)
        prev_pop = np.where(start > 0, cum_pop[np.maximum(start - 1, 0)], 0.0)
        edge_mean = (cum_pw - prev_pw) / np.maximum(cum_pop - prev_pop, 1e-9)
        ok = np.flatnonzero((cum_mean >= thr) & (edge_mean >= edge_thr))
        if len(ok) == 0:
            continue
        k = ok[-1]
        radius[s] = min(float(dist[order[k]]) + R9_EQUIV_RADIUS_M, max_r)
        pop_in[s] = float(cum_pop[k])
        mean_p[s] = float(cum_mean[k])
    return ZonesResult(radius, users, pop_in, mean_p, cells, p_use)
