"""Res-8 grid over the bbox with governorate labels → data/interim/grid_r8.parquet.

The final cell filter (pop > 0 or shopping street or transit stop) happens at the end of 07.
"""

from __future__ import annotations

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import Point

from khatt.geo import bbox_cells, cell_center
from khatt.prep.common import INTERIM, RAW, decision, setup, summary


def approximate_governorate(lat: np.ndarray, lng: np.ndarray) -> np.ndarray:
    """Rough rule when no boundaries are available: west of the Nile ≈ Giza, north ≈ Qalyubia."""
    nile_lng = 31.225 + (lat - 30.05) * 0.15  # the river runs roughly SSW→NNE through the city
    return np.where(lng < nile_lng, "Giza", np.where(lat > 30.13, "Qalyubia", "Cairo"))


def main() -> None:
    _log, params, t0 = setup("01_grid")
    cells = bbox_cells(tuple(params["area.bbox"]), int(params["area.h3_res"]))
    centres = np.array([cell_center(c) for c in cells])
    grid = pd.DataFrame({"h3": cells, "lat": centres[:, 0], "lng": centres[:, 1]})

    gov_file = RAW / "governorates.geojson"
    if gov_file.exists():
        gov = gpd.read_file(gov_file)[["governorate", "geometry"]].to_crs("EPSG:4326")
        pts = gpd.GeoDataFrame(grid, geometry=[Point(x, y) for x, y in zip(grid.lng, grid.lat)],
                               crs="EPSG:4326")
        joined = gpd.sjoin(pts, gov, how="left", predicate="within")
        joined = joined[~joined.index.duplicated()]
        grid["governorate"] = joined["governorate"].fillna("Other").to_numpy()
        source = "OSM boundaries"
    else:
        grid["governorate"] = approximate_governorate(grid.lat.to_numpy(), grid.lng.to_numpy())
        source = "approximate rule"
        decision("01_grid", "No governorate boundaries: used the approximate Nile/latitude rule.")
    grid.to_parquet(INTERIM / "grid_r8.parquet", index=False)
    counts = grid["governorate"].value_counts().to_dict()
    summary("01_grid", [f"{len(grid)} res-8 cells in bbox", f"governorates ({source}): {counts}"],
            t0)


if __name__ == "__main__":
    main()
