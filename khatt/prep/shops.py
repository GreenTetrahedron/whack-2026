"""Shop category crosswalk and cross-source de-duplication (CLAUDE.md §6, 04)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from khatt.prep.text import normalize_name, similarity

KHATT_CATEGORIES = ("daily", "comp", "eat", "other")
SOURCE_PRIORITY = {"overture": 0, "osm": 1}  # keep the lower number when two records match


def load_crosswalk(path: Path) -> dict[tuple[str, str], str]:
    cw = pd.read_csv(path, dtype=str).fillna("")
    bad = set(cw["khatt_category"]) - set(KHATT_CATEGORIES)
    if bad:
        raise ValueError(f"Unknown khatt_category values in {path}: {sorted(bad)}")
    return {(r.source.strip(), r.raw_category.strip().lower()): r.khatt_category.strip()
            for r in cw.itertuples()}


def map_category(crosswalk: dict, source: str, raw_categories) -> str:
    """First raw category (primary, then alternates) found in the crosswalk wins; else 'other'."""
    if raw_categories is None:
        return "other"
    if isinstance(raw_categories, str):
        raw_categories = [raw_categories]
    for raw in raw_categories:
        if raw is None:
            continue
        hit = crosswalk.get((source, str(raw).strip().lower()))
        if hit:
            return hit
    return "other"


def dedupe(shops: pd.DataFrame, radius_m: float, min_similarity: float) -> pd.DataFrame:
    """Drop cross-source duplicates: same category, within radius_m, name similarity ≥ threshold.

    Needs columns: x, y (metres), khatt_category, name, source.
    """
    if shops.empty:
        return shops.copy()
    df = shops.reset_index(drop=True).copy()
    df["name_norm"] = [normalize_name(n) for n in df["name"]]
    tree = cKDTree(df[["x", "y"]].to_numpy())
    pairs = tree.query_pairs(r=radius_m, output_type="ndarray")
    drop = np.zeros(len(df), bool)
    src = df["source"].to_numpy()
    cat = df["khatt_category"].to_numpy()
    names = df["name_norm"].to_numpy()
    for i, j in pairs:
        if src[i] == src[j] or cat[i] != cat[j] or drop[i] or drop[j]:
            continue
        if similarity(names[i], names[j]) >= min_similarity:
            loser = j if SOURCE_PRIORITY.get(src[i], 9) <= SOURCE_PRIORITY.get(src[j], 9) else i
            drop[loser] = True
    return df.loc[~drop].reset_index(drop=True)
