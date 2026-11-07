"""Multi-line attribution (CLAUDE.md §9): pairwise synergy and Shapley split of dR per street.

The joint evaluation is the source of truth; this module only re-runs the spending model on
subsets of lines to explain it.
"""

from __future__ import annotations

from itertools import combinations
from math import factorial

import numpy as np

from khatt.config import Params
from khatt.engine.lines import LineSpec
from khatt.engine.multiline import joint_rstar, wait_board
from khatt.engine.overlay import Overlay
from khatt.models import spending

MAX_SHAPLEY_LINES = 4


def shapley(values: dict[int, np.ndarray], n: int) -> np.ndarray:
    """values[mask] → (n, n_items) Shapley values. mask 0 must be present (usually zeros)."""
    phi = np.zeros((n,) + np.shape(values[0]))
    for i in range(n):
        bit = 1 << i
        for mask in range(1 << n):
            if mask & bit:
                continue
            k = (mask).bit_count()
            weight = factorial(k) * factorial(n - k - 1) / factorial(n)
            phi[i] += weight * (values[mask | bit] - values[mask])
    return phi


def subset_dR(world, params: Params, lines: list[LineSpec], station_hex_by_line: list[np.ndarray],
              access_by_line: list[np.ndarray], mask: int, base: spending.Huff,
              street_dest: np.ndarray, street_cols: np.ndarray) -> np.ndarray:
    chosen = [k for k in range(len(lines)) if mask >> k & 1]
    sub_lines = [lines[k] for k in chosen]
    hexes = np.concatenate([station_hex_by_line[k] for k in chosen])
    access = np.concatenate([access_by_line[k] for k in chosen])
    rstar = joint_rstar(sub_lines, hexes, world.T0, params, access=access)
    ov = Overlay(world.T0, hexes, wait_board(sub_lines, params) + access, rstar)
    T0_d, T1_d = ov.to(street_dest)
    res = spending.run(world, params, T0_d[:, street_cols], T1_d[:, street_cols], base=base)
    return res.dR.sum(axis=1)


def run(world, params: Params, lines: list[LineSpec], station_hex_by_line: list[np.ndarray],
        access_by_line: list[np.ndarray], base: spending.Huff, street_dest: np.ndarray, street_cols: np.ndarray,
        joint_dR: np.ndarray, street_ids: np.ndarray, report_rows: np.ndarray) -> dict:
    """Returns {"synergy": [...], "attribution": {street_id: {line_id: egp}}, "warnings": [...]}."""
    n = len(lines)
    out: dict = {"synergy": [], "attribution": {}, "warnings": []}
    if n < 2:
        return out
    if n > MAX_SHAPLEY_LINES:
        out["warnings"].append(
            f"Attribution skipped: {n} lines (Shapley split runs for at most {MAX_SHAPLEY_LINES})")
        return out
    full = (1 << n) - 1
    values = {0: np.zeros_like(joint_dR), full: joint_dR}
    for mask in range(1, full):
        values[mask] = subset_dR(world, params, lines, station_hex_by_line, access_by_line,
                                 mask, base, street_dest, street_cols)

    for i, j in combinations(range(n), 2):
        a, b, ab = values[1 << i], values[1 << j], values[(1 << i) | (1 << j)]
        per_street = ab - a - b
        top = np.argsort(-np.abs(per_street))[:20]
        out["synergy"].append({
            "lines": [lines[i].line_id, lines[j].line_id],
            "abs_dR_egp": float(np.abs(ab).sum() - np.abs(a).sum() - np.abs(b).sum()),
            "streets": [{"street_id": int(street_ids[k]), "egp": float(per_street[k])}
                        for k in top if abs(per_street[k]) > 0],
        })

    phi = shapley(values, n)
    total = phi.sum(axis=0)
    tol = 1e-3 * np.abs(joint_dR) + 1e-6
    if not np.all(np.abs(total - joint_dR) <= tol):
        raise AssertionError("Shapley parts do not sum to the joint effect")
    for k in report_rows:
        out["attribution"][str(int(street_ids[k]))] = {
            lines[i].line_id: float(phi[i, k]) for i in range(n)
        }
    return out
