"""Per-capita spending by governorate (CAPMAS HIECS 2021) → dasymetric allocation to res-8 cells.

Input: data/raw/spending_manual.csv (template written by 00_download.py), else the committed
       config/spending_hiecs_2019_20.csv (CAPMAS HIECS 2019/20 urban per-capita averages)
Optional wealth: data/raw/rwi_egypt.csv (Meta Relative Wealth Index: latitude, longitude, rwi)
Output: data/interim/spending_r8.parquet
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from khatt.config import CATEGORIES, ROOT
from khatt.geo import cells_for_points
from khatt.prep.common import INTERIM, RAW, decision, require, setup, summary
from khatt.prep.spending import allocate, category_totals, normalise_wealth

RWI_SCALE = "spending.rwi_scale"
COMMITTED_TABLE = ROOT / "config" / "spending_hiecs_2019_20.csv"


def per_capita_table(path, price_factor_to_2021: float = 1.0) -> tuple[dict, list[str]]:
    """CSV rows → {governorate: {category: EGP/person/yr at 2021 prices}} plus conversion notes.

    Rows whose `prices` column is not "2021" are multiplied by price_factor_to_2021.
    """
    df = pd.read_csv(path, comment="#", dtype={"coicop": str, "prices": str})
    df = df.dropna(subset=["value_egp"])
    if df.empty:
        raise SystemExit(f"{path} has no values yet — fill it from the CAPMAS HIECS tables")
    notes = []
    df["coicop"] = df["coicop"].str.zfill(2)
    factor = np.where(df["period"].str.lower() == "monthly", 12.0, 1.0)
    hh = np.where(df["basis"].str.lower() == "household",
                  pd.to_numeric(df["household_size"], errors="coerce"), 1.0)
    if np.isnan(hh).any():
        raise SystemExit("household_size is required for rows with basis=household")
    prices = df["prices"].fillna("2021") if "prices" in df.columns else pd.Series("2021", df.index)
    to_2021 = np.where(prices.str.strip() == "2021", 1.0, price_factor_to_2021)
    df["per_person_annual"] = df["value_egp"].astype(float) * factor / hh * to_2021
    if (factor == 12).any():
        notes.append("monthly CAPMAS values ×12 → annual")
    if (df["basis"].str.lower() == "household").any():
        notes.append("household values ÷ mean household size → per person")
    if (to_2021 != 1.0).any():
        years = sorted(set(prices[to_2021 != 1.0]))
        notes.append(f"{years} prices × {price_factor_to_2021} → 2021 EGP")
    out = {}
    for gov, g in df.groupby("governorate"):
        coicop = dict(zip(g["coicop"], g["per_person_annual"]))
        out[gov] = coicop
    return out, notes


def wealth_factor(grid: pd.DataFrame, params, log) -> tuple[np.ndarray, str]:
    rwi_file = RAW / "rwi_egypt.csv"
    if not rwi_file.exists():
        return np.ones(len(grid), np.float32), "none (1.0 everywhere)"
    rwi = pd.read_csv(rwi_file)
    rwi["h3"] = cells_for_points(rwi["latitude"], rwi["longitude"], int(params["area.h3_res"]))
    mean = rwi.groupby("h3")["rwi"].mean()
    raw = np.exp(float(params[RWI_SCALE]) * grid["h3"].map(mean).to_numpy(float))
    return normalise_wealth(raw, grid["pop"].to_numpy()), "Meta RWI"


def main() -> None:
    log, params, t0 = setup("03_spending")
    path = RAW / "spending_manual.csv"
    if not path.exists():
        path = require(COMMITTED_TABLE, "fill data/raw/spending_manual.template.csv from CAPMAS "
                       "HIECS tables and save it as data/raw/spending_manual.csv")
        decision("03_spending", f"no data/raw/spending_manual.csv: used {COMMITTED_TABLE.name} "
                 "(CAPMAS HIECS 2019/20 urban per-capita averages, same for all governorates)")
    grid = pd.read_parquet(require(INTERIM / "grid_r8.parquet", "run pipeline/01_grid.py"))
    pop = pd.read_parquet(require(INTERIM / "pop_r8.parquet", "run pipeline/02_population.py"))
    grid = grid.merge(pop, on="h3", how="left").fillna({"pop": 0.0})

    coicop, notes = per_capita_table(path, float(params["spending.price_factor_to_2021"]))
    share09 = float(params["spending.share_09_electronics"])
    per_capita = {gov: category_totals(v, share09) for gov, v in coicop.items()}
    missing = sorted(set(grid["governorate"]) - set(per_capita))
    if missing:
        notes.append(f"no CAPMAS row for {missing}: population-weighted mean of the others used")
    grid["wealth"], wealth_src = wealth_factor(grid, params, log)
    spend = allocate(grid["pop"].to_numpy(), grid["wealth"].to_numpy(),
                     grid["governorate"].to_numpy(), per_capita)
    for c in CATEGORIES:
        grid[f"spend_{c}"] = spend[c]
    grid["pop"] = grid["pop"].astype("float32")
    grid["wealth"] = grid["wealth"].astype("float32")
    grid.to_parquet(INTERIM / "spending_r8.parquet", index=False)

    for n in notes + [f"wealth factor source: {wealth_src}"]:
        decision("03_spending", n)
    totals = {c: f"{grid[f'spend_{c}'].sum() / 1e9:.1f} bn" for c in CATEGORIES}
    summary("03_spending", [f"governorates with CAPMAS data: {sorted(per_capita)}",
                            f"annual spending (EGP 2021): {totals}",
                            f"wealth: {wealth_src}"], t0)


if __name__ == "__main__":
    main()
