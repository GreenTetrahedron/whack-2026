"""Model 3 — rent pressure and margin squeeze (CLAUDE.md §8). Rents are an index, never EGP."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from khatt.config import CATEGORIES, Params

RISK_LEVELS = np.array(["green", "amber", "red"], dtype=object)


@dataclass
class RentResult:
    cells: np.ndarray  # hex rows within rents.radius_m of a new station
    rent_cells: np.ndarray  # fraction, settled, per cell
    street_rent: np.ndarray  # fraction, settled, per street
    street_rent_opening: np.ndarray
    revenue: np.ndarray  # fraction per street
    profit: np.ndarray  # settled profit signal
    risk: np.ndarray  # str per street (settled)
    risk_opening: np.ndarray
    unbranded_at_risk: np.ndarray


def market_potential(S: np.ndarray, T: np.ndarray, beta: np.ndarray) -> np.ndarray:
    """M[h] = Σ_c Σ_o S[o, c] · exp(−beta_c · T[o, h]); T is (N, H)."""
    M = np.zeros(T.shape[1])
    T = T.astype(np.float64)
    for c in range(len(CATEGORIES)):
        M += S[:, c] @ np.exp(-beta[c] * T)
    return M


def classify(profit: np.ndarray, red_threshold: float, amber_threshold: float = 0.0) -> np.ndarray:
    """green if profit ≥ −amber_threshold, amber if ≥ −red_threshold, else red. The amber
    tolerance stops far-away streets whose share moved by float noise from showing as amber."""
    level = np.where(profit >= -amber_threshold, 0, np.where(profit >= -red_threshold, 1, 2))
    return RISK_LEVELS[level]


def run(world, params: Params, cells: np.ndarray, T0_cells: np.ndarray, T1_cells: np.ndarray,
        S1: np.ndarray, R0_total: np.ndarray, dR_total: np.ndarray,
        passby: np.ndarray) -> RentResult:
    beta = params.beta
    elasticity = float(params["rents.elasticity"])
    fterm = float(params["rents.footfall_term"])
    short = float(params["rents.short_run_share"])

    passby_share = np.divide(passby, R0_total, out=np.zeros_like(passby), where=R0_total > 0)
    if len(cells):
        M0 = market_potential(world.spend, T0_cells, beta)
        M1 = market_potential(S1, T1_cells, beta)
        access = elasticity * np.divide(M1 - M0, M0, out=np.zeros_like(M0), where=M0 > 0)
        # cell-level pass-by share: Σ pass-by / Σ R0 of the streets in that cell
        pb = np.bincount(world.street_hex, weights=passby, minlength=world.n)[cells]
        r0 = np.bincount(world.street_hex, weights=R0_total, minlength=world.n)[cells]
        rent_cells = access + fterm * np.divide(pb, r0, out=np.zeros_like(pb), where=r0 > 0)
    else:
        access = np.zeros(0)
        rent_cells = np.zeros(0)

    access_by_hex = np.zeros(world.n)
    access_by_hex[cells] = access
    street_rent = access_by_hex[world.street_hex] + fterm * passby_share
    revenue = np.divide(dR_total + passby, R0_total, out=np.zeros_like(R0_total),
                        where=R0_total > 0)

    gm = float(params["rents.gross_margin"])
    rcs = float(params["rents.rent_cost_share"])
    red = float(params["rents.red_threshold"])
    amber = float(params["rents.amber_threshold"])
    profit = gm * revenue - rcs * street_rent
    profit_opening = gm * revenue - rcs * street_rent * short
    risk = classify(profit, red, amber)
    n_unbranded = world.streets["n_unbranded"].to_numpy()
    return RentResult(
        cells=cells, rent_cells=rent_cells, street_rent=street_rent,
        street_rent_opening=street_rent * short, revenue=revenue, profit=profit, risk=risk,
        risk_opening=classify(profit_opening, red, amber),
        unbranded_at_risk=np.where(risk == "red", n_unbranded, 0),
    )
