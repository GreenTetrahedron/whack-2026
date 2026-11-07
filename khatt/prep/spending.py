"""Dasymetric spending allocation (CLAUDE.md §6, 03)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from khatt.config import CATEGORIES


def category_totals(coicop: dict[str, float], share_09_electronics: float) -> dict[str, float]:
    """Per-capita annual EGP by COICOP division ('01', '03', ...) → khatt categories."""
    get = lambda k: float(coicop.get(k, 0.0))
    return {
        "daily": get("01"),
        "comp": get("03") + get("05") + get("09") * share_09_electronics,
        "eat": get("11"),
    }


def allocate(pop, wealth, governorate, per_capita: dict[str, dict[str, float]]) -> dict[str, np.ndarray]:
    """spend_c[h] = pop × per_capita_c[gov] × wealth, rescaled per governorate to pop_gov × pc.

    per_capita: {governorate: {category: EGP/person/yr}}. Unknown governorates use the
    population-weighted mean of the known ones.
    """
    pop = np.asarray(pop, float)
    wealth = np.asarray(wealth, float)
    gov = np.asarray(governorate, dtype=object)
    known = [g for g in per_capita if g in set(gov)]
    out = {}
    for cat in CATEGORIES:
        if known:
            weights = np.array([pop[gov == g].sum() for g in known])
            fallback = np.average([per_capita[g][cat] for g in known], weights=weights + 1e-9)
        else:
            fallback = np.mean([v[cat] for v in per_capita.values()])
        pc = np.array([per_capita.get(g, {}).get(cat, fallback) for g in gov], float)
        raw = pop * pc * wealth
        spend = np.zeros_like(raw)
        for g in pd.unique(gov):
            mask = gov == g
            target = (pop[mask] * pc[mask]).sum()
            total = raw[mask].sum()
            spend[mask] = raw[mask] * (target / total) if total > 0 else 0.0
        out[cat] = spend.astype(np.float32)
    return out


def normalise_wealth(wealth, pop) -> np.ndarray:
    """Scale so the population-weighted mean is 1.0 (missing → 1.0)."""
    wealth = np.asarray(wealth, float).copy()
    wealth[~np.isfinite(wealth) | (wealth <= 0)] = np.nan
    pop = np.asarray(pop, float)
    ok = np.isfinite(wealth) & (pop > 0)
    if not ok.any():
        return np.ones_like(wealth, dtype=np.float32)
    mean = np.average(wealth[ok], weights=pop[ok])
    wealth = wealth / mean
    wealth[~np.isfinite(wealth)] = 1.0
    return wealth.astype(np.float32)
