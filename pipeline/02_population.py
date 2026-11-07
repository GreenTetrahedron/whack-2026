"""WorldPop 100 m raster → population per res-8 and res-9 cell (data/interim/pop_r{8,9}.parquet)."""

from __future__ import annotations

from khatt.prep.common import INTERIM, RAW, require, setup, summary
from khatt.prep.population import raster_to_cells

RASTER = "egy_pop_2026_CN_100m_R2025A_v1.tif"


def main() -> None:
    _log, params, t0 = setup("02_population")
    path = require(RAW / RASTER, "run pipeline/00_download.py --only worldpop")
    res8, res9 = int(params["area.h3_res"]), int(params["area.h3_res_fine"])
    cells, raster_total = raster_to_cells(path, tuple(params["area.bbox"]), (res8, res9))
    for res, df in cells.items():
        total = df["pop"].sum()
        err = abs(total - raster_total) / raster_total
        assert err < 0.01, f"res {res}: hex total {total:.0f} vs raster {raster_total:.0f}"
        df["pop"] = df["pop"].astype("float32")
        df.to_parquet(INTERIM / f"pop_r{res}.parquet", index=False)
    summary("02_population", [
        f"raster total in bbox: {raster_total:,.0f}",
        f"res-8 populated cells: {len(cells[res8])}; res-9: {len(cells[res9])}",
        "hex totals within 1 % of raster: OK",
    ], t0)


if __name__ == "__main__":
    main()
