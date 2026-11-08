"""Model 1 — spending shifts between shopping streets (Huff model).

logit[o, d] = alpha · ln A[d, c] − beta_c · T[o, d]     (d with T > max_trip excluded)
P[o, d]     = softmax_d(logit[o, :])
R[d, c]     = Σ_o S[o, c] · P[o, d]
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from khatt.config import CATEGORIES, Params


@dataclass
class Huff:
    P: list[np.ndarray]  # per category, (N, n_streets) float32; 0 for excluded streets
    log_reach: np.ndarray  # (N, 3) log Σ_d exp(logit); -inf if nothing reachable
    R: np.ndarray  # (n_streets, 3) EGP/yr
    S: np.ndarray  # (N, 3) spending used (after induced demand)


def huff_probs(logA: np.ndarray, T: np.ndarray, alpha: float, beta: float, max_trip: float):
    """Row-softmax of alpha·logA − beta·T with T > max_trip (or logA = -inf) excluded."""
    logit = alpha * logA[None, :] - beta * T.astype(np.float64)
    logit[T > max_trip] = -np.inf
    m = logit.max(axis=1, keepdims=True)
    ok = np.isfinite(m[:, 0])
    e = np.exp(logit - np.where(ok[:, None], m, 0.0))
    s = e.sum(axis=1, keepdims=True)
    P = np.divide(e, s, out=np.zeros_like(e), where=s > 0)
    with np.errstate(divide="ignore"):
        log_reach = np.where(ok, m[:, 0] + np.log(np.where(s[:, 0] > 0, s[:, 0], 1.0)), -np.inf)
    return P.astype(np.float32), log_reach


def run_huff(S: np.ndarray, A: np.ndarray, T_streets: np.ndarray, params: Params) -> Huff:
    """S (N, 3) spending; A (n_streets, 3) attractiveness; T_streets (N, n_streets) minutes."""
    alpha = float(params["spending.alpha"])
    beta = params.beta
    max_trip = float(params["spending.max_trip_min"])
    P_list, log_reach = [], np.full((S.shape[0], len(CATEGORIES)), -np.inf)
    R = np.zeros((A.shape[0], len(CATEGORIES)))
    with np.errstate(divide="ignore"):
        logA = np.log(A)  # A = 0 → -inf → excluded for that category
    for c in range(len(CATEGORIES)):
        P, lr = huff_probs(logA[:, c], T_streets, alpha, beta[c], max_trip)
        P_list.append(P)
        log_reach[:, c] = lr
        R[:, c] = S[:, c].astype(np.float64) @ P.astype(np.float64)
    return Huff(P_list, log_reach, R, S)


def reach_gain(base: Huff, scen: Huff, S: np.ndarray) -> np.ndarray:
    """Spending-weighted relative gain in accessibility (Huff denominator) per origin."""
    with np.errstate(invalid="ignore", over="ignore"):
        ratio = np.exp(np.clip(scen.log_reach - base.log_reach, -50, 50)) - 1.0
    ratio[~np.isfinite(ratio)] = 0.0
    w = S.sum(axis=1)
    return np.divide((S * ratio).sum(axis=1), w, out=np.zeros(len(w)), where=w > 0)


@dataclass
class SpendingResult:
    base: Huff
    scen: Huff
    dR: np.ndarray  # (n_streets, 3)
    d_reach: np.ndarray  # (N,) relative accessibility gain
    use_share: np.ndarray  # (N,) spending-weighted share of trips whose best path uses a new line


def run(world, params: Params, T0_streets: np.ndarray, T1_streets: np.ndarray,
        base: Huff | None = None) -> SpendingResult:
    """T*_streets: (N, n_streets) times to each street's hex."""
    S = world.spend
    if base is None:
        base = run_huff(S, world.street_A, T0_streets, params)
    scen = run_huff(S, world.street_A, T1_streets, params)
    induced = float(params["spending.induced_pct"])
    if induced > 0:
        gain = np.clip(np.exp(np.clip(scen.log_reach - base.log_reach, -50, 50)) - 1.0, 0, None)
        gain[~np.isfinite(gain)] = 0.0
        S1 = S * (1.0 + induced * gain)
        scen = run_huff(S1, world.street_A, T1_streets, params)
    d_reach = reach_gain(base, scen, S)

    margin = float(params["footfall.use_margin_min"])
    uses = T1_streets < (T0_streets - margin)
    num = np.zeros(S.shape[0])
    den = np.zeros(S.shape[0])
    for c in range(len(CATEGORIES)):
        P = scen.P[c]
        num += S[:, c] * (P * uses).sum(axis=1)
        den += S[:, c] * P.sum(axis=1)
    use_share = np.divide(num, den, out=np.zeros_like(num), where=den > 0)
    return SpendingResult(base, scen, scen.R - base.R, d_reach, use_share)


def flow_matrix(world, res: SpendingResult) -> np.ndarray:
    """F[o, d] = Σ_c S[o, c] · ΔP[o, d, c]: EGP/yr that origin o moves to (+) or away from (−)
    street d."""
    F = np.zeros(res.scen.P[0].shape, np.float64)
    for c in range(len(CATEGORIES)):
        F += world.spend[:, c : c + 1] * (res.scen.P[c].astype(np.float64) - res.base.P[c])
    return F


def street_flows(world, F: np.ndarray, rows: np.ndarray, k: int) -> dict[int, list[dict]]:
    """For each street row in `rows`: its k largest |F| origins → {row: [{from, egp}]}, so the
    map can show where one street's spending comes from or goes."""
    rows = np.asarray(rows, np.int64)
    k = min(k, F.shape[0])
    if len(rows) == 0 or k <= 0:
        return {}
    sub = np.abs(F[:, rows])
    top = np.argpartition(-sub, k - 1, axis=0)[:k]  # (k, len(rows)), unsorted
    out: dict[int, list[dict]] = {}
    for j, row in enumerate(rows):
        o = top[:, j]
        o = o[np.argsort(-sub[o, j])]
        o = o[sub[o, j] > 0]
        out[int(row)] = [{"from": [round(float(world.lng[oi]), 5), round(float(world.lat[oi]), 5)],
                          "egp": round(float(F[oi, row]))} for oi in o]
    return out


def top_flows(world, res: SpendingResult, n: int, F: np.ndarray | None = None) -> list[dict]:
    """Largest |Σ_c S[o, c] · ΔP[o, d, c]| origin → street shifts (EGP/yr, signed)."""
    if F is None:
        F = flow_matrix(world, res)
    flat = np.abs(F).ravel()
    n = min(n, int((flat > 0).sum()))
    if n == 0:
        return []
    top = np.argpartition(flat, -n)[-n:]
    top = top[np.argsort(-flat[top])]
    o, d = np.unravel_index(top, F.shape)
    st = world.streets
    return [
        {
            "from": [float(world.lng[oi]), float(world.lat[oi])],
            "to": [float(st["lng"].iat[di]), float(st["lat"].iat[di])],
            "street_id": int(st["street_id"].iat[di]),
            "egp": float(F[oi, di]),
        }
        for oi, di in zip(o, d)
    ]
