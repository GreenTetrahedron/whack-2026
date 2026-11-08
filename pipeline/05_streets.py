"""DBSCAN shop clusters → shopping streets (data/interim/streets_raw.parquet).

hex_idx is attached at the end of 07 once the cell index is final.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import shapely

from khatt.geo import to_utm
from khatt.prep.common import INTERIM, RAW, decision, require, setup, summary
from khatt.prep.streets import build_streets, cluster, footprints_near

BUILDINGS = RAW / "overture_buildings.parquet"


def building_areas(shops: pd.DataFrame, radius_m: float):
    """(area per building, nearest building per shop) using only buildings near shops."""
    import pyarrow.parquet as pq
    import pyproj
    from scipy.spatial import cKDTree

    pf = pq.ParquetFile(BUILDINGS)
    sx, sy = shops["x"].to_numpy(), shops["y"].to_numpy()
    shop_tree = cKDTree(np.column_stack([sx, sy]))
    tr = pyproj.Transformer.from_crs("EPSG:4326", "EPSG:32636", always_xy=True)

    def project(coords: np.ndarray) -> np.ndarray:
        return np.column_stack(tr.transform(coords[:, 0], coords[:, 1]))

    areas, bx, by = [], [], []
    for batch in pf.iter_batches(columns=["geometry"], batch_size=200_000):
        geom = shapely.from_wkb(batch.column("geometry").to_numpy(zero_copy_only=False))
        cent = shapely.centroid(geom)
        x, y = to_utm(shapely.get_y(cent), shapely.get_x(cent))
        near = np.isfinite(shop_tree.query(np.column_stack([x, y]),
                                           distance_upper_bound=radius_m + 50)[0])
        if not near.any():
            continue
        # projected area: transform polygons to UTM for the near ones only
        areas.append(shapely.area(shapely.transform(geom[near], project)))
        bx.append(x[near])
        by.append(y[near])
    if not areas:
        return np.zeros(0), np.full(len(shops), -1)
    area = np.concatenate(areas)
    idx = footprints_near(sx, sy, np.concatenate(bx), np.concatenate(by), area, radius_m)
    return area, idx


def main() -> None:
    _log, params, t0 = setup("05_streets")
    shops = pd.read_parquet(require(INTERIM / "shops.parquet", "run pipeline/04_shops.py"))
    shops = shops.reset_index(drop=True)
    labels = cluster(shops["x"].to_numpy(), shops["y"].to_numpy(),
                     float(params["streets.eps_m"]), int(params["streets.min_shops"]),
                     max_extent_m=float(params["streets.max_extent_m"]),
                     shrink=float(params["streets.eps_shrink"]),
                     floor_m=float(params["streets.eps_floor_m"]))
    if BUILDINGS.exists():
        area, shop_bld = building_areas(shops, float(params["streets.footprint_join_m"]))
    else:
        area, shop_bld = None, None
        decision("05_streets", "No Overture buildings: footprint_m2 = 0 for every street")
    streets = build_streets(shops, labels, area, shop_bld)
    streets.to_parquet(INTERIM / "streets_raw.parquet", index=False)
    n_noise = int((labels < 0).sum())
    summary("05_streets", [
        f"{len(streets)} streets from {len(shops)} shops ({n_noise} noise points dropped)",
        f"eps={params['streets.eps_m']} m, min_shops={params['streets.min_shops']}",
        f"shops per street: median {streets[['n_daily', 'n_comp', 'n_eat']].sum(axis=1).median():.0f}"
        if len(streets) else "no streets!",
        f"named streets: {(~streets['name'].str.startswith('Cluster')).sum() if len(streets) else 0}",
    ], t0)


if __name__ == "__main__":
    main()
