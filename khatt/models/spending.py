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


def net_inflow(world, res: SpendingResult, rows: np.ndarray) -> np.ndarray:
    """Net EGP/yr each street moves *into* each street in `rows`: (n_streets, len(rows)), < 0 =
    money moving out of rows[j] to that street.

    An origin's spending only moves between streets (ΔP sums to 0 per origin and category), so
    what origin o adds at street b it takes from the streets where it now spends less. Each gain
    is split across those losses in proportion, per category, and summed:
        moved[a → b] = Σ_c Σ_o S[o,c]·ΔP⁺[o,b,c] · ΔP⁻[o,a,c] / Σ_d ΔP⁻[o,d,c]
    and net[a, j] = moved[a → rows[j]] − moved[rows[j] → a]. Summed over a, it is dR of rows[j].
    """
    rows = np.asarray(rows, np.int64)
    net = np.zeros((res.dR.shape[0], len(rows)))
    for c in range(len(CATEGORIES)):
        F = world.spend[:, c : c + 1] * (res.scen.P[c].astype(np.float64) - res.base.P[c])
        gain = np.maximum(F, 0.0)
        loss = np.maximum(-F, 0.0)
        total = loss.sum(axis=1, keepdims=True)
        share = np.divide(loss, total, out=np.zeros_like(loss), where=total > 0)
        net += share.T @ gain[:, rows] - gain.T @ share[:, rows]
    return net


def street_transfers(world, res: SpendingResult, rows: np.ndarray, k: int,
                     min_share: float = 0.0) -> dict[int, list[dict]]:
    """For each street row in `rows`: its k largest net money moves with other streets →
    {row: [{from, to, street_id, egp}]}: `from` is the street that loses the spending, `to` the
    one that gains it, `street_id` the other street, `egp` > 0 (EGP/yr). Moves smaller than
    `min_share` × that street's largest are dropped."""
    rows = np.asarray(rows, np.int64)
    if len(rows) == 0 or k <= 0:
        return {}
    net = net_inflow(world, res, rows)
    st = world.streets
    ids = st["street_id"].to_numpy()
    lnglat = np.column_stack([st["lng"].to_numpy(), st["lat"].to_numpy()]).round(5).tolist()
    out: dict[int, list[dict]] = {}
    for j, row in enumerate(rows):
        mag = np.abs(net[:, j])
        top = np.argsort(-mag)[: min(k, len(mag))]
        top = top[(mag[top] > 0) & (mag[top] >= min_share * mag[top[0]])]
        here = lnglat[row]
        out[int(row)] = [
            {"from": lnglat[a] if net[a, j] > 0 else here,
             "to": here if net[a, j] > 0 else lnglat[a],
             "street_id": int(ids[a]),
             "egp": round(float(mag[a]))}
            for a in top
        ]
    return out


def top_flows(world, res: SpendingResult, n: int, F: np.ndarray | None = None,
              min_share: float = 0.0) -> list[dict]:
    """Largest |Σ_c S[o, c] · ΔP[o, d, c]| origin → street shifts (EGP/yr, signed): at most n,
    and none smaller than `min_share` × the largest (a long tail of tiny shifts draws as a
    tangle of hairlines on the map)."""
    if F is None:
        F = flow_matrix(world, res)
    flat = np.abs(F).ravel()
    n = min(n, int((flat > 0).sum()))
    if n == 0:
        return []
    top = np.argpartition(flat, -n)[-n:]
    top = top[np.argsort(-flat[top])]
    top = top[flat[top] >= min_share * flat[top[0]]]
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
