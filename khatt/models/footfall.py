"""Model 2 — station riders and pass-by spending (CLAUDE.md §8)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from khatt.config import Params
from khatt.engine.catchment import Catchment, assign_nearest


@dataclass
class FootfallResult:
    riders_per_resident: float
    catchment_pop10: np.ndarray  # (S,)
    usefulness: np.ndarray  # (S,)
    riders_daily: np.ndarray  # (S,) boardings ≈ exits
    passby_egp: np.ndarray  # (n_streets,) EGP/yr
    footfall_r9: np.ndarray  # r9 rows
    footfall_daily: np.ndarray  # passers-by per day (decayed station exits)


def riders_per_resident(world, params: Params) -> float:
    total = float(world.existing["catchment_pop10"].sum()) if len(world.existing) else 0.0
    return float(params["footfall.metro_daily_riders"]) / total if total > 0 else 0.0


def run(world, params: Params, catches: list[Catchment], use_share: np.ndarray,
        street_walk_min: list[tuple[np.ndarray, np.ndarray]],
        priority: list | None = None) -> FootfallResult:
    """street_walk_min[s] = (street rows, walk minutes) within the pass-by radius of station s."""
    S = len(catches)
    n_streets = len(world.streets)
    r = riders_per_resident(world, params)
    cutoff = float(params["footfall.catchment_min"])
    n_r9 = len(world.r9_pop)

    owner, _ = assign_nearest(catches, n_r9, cutoff, priority)
    has = owner >= 0
    pop10 = np.bincount(owner[has], weights=world.r9_pop[has], minlength=S)[:S]

    # usefulness: spending-weighted share of trips from the catchment that use the new line(s)
    ref = float(params["footfall.reference_use_share"])
    lo, hi = params["footfall.usefulness_clip"]
    usefulness = np.full(S, float(lo))
    if has.any():
        parents = world.r9_parent[has]
        w = world.r9_pop[has]
        num = np.bincount(owner[has], weights=w * use_share[parents], minlength=S)[:S]
        share = np.divide(num, pop10, out=np.zeros(S), where=pop10 > 0)
        usefulness = np.clip(share / ref, lo, hi)
    riders = r * pop10 * usefulness

    m_per_min = params.walk_m_per_min
    decay = float(params["footfall.passby_decay_m"])
    per_exit = (float(params["footfall.buy_rate"]) * float(params["footfall.basket_egp"])
                * float(params["footfall.days_per_year"]))
    passby = np.zeros(n_streets)
    for s, (rows, minutes) in enumerate(street_walk_min):
        if len(rows) == 0:
            continue
        walk_m = minutes * m_per_min
        np.add.at(passby, rows, riders[s] * np.exp(-walk_m / decay) * per_exit)

    # heatmap: decayed exits over each station's walking catchment (r9 cells)
    heat = {}
    for s, c in enumerate(catches):
        if riders[s] <= 0 or len(c.r9_idx) == 0:
            continue
        vals = riders[s] * np.exp(-(c.r9_min * m_per_min) / decay)
        for cell, v in zip(c.r9_idx, vals):
            heat[cell] = heat.get(cell, 0.0) + v
    cells = np.fromiter(heat.keys(), np.int64, len(heat))
    vals = np.fromiter(heat.values(), np.float64, len(heat))
    return FootfallResult(r, pop10, usefulness, riders, passby, cells, vals)
