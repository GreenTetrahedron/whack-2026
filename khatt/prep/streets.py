"""Shopping streets = DBSCAN clusters of shops (CLAUDE.md §6, 05)."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from sklearn.cluster import DBSCAN

from khatt.geo import cells_for_points, from_utm

CATS = ("daily", "comp", "eat")


def cluster(x: np.ndarray, y: np.ndarray, eps_m: float, min_samples: int) -> np.ndarray:
    """DBSCAN labels; -1 = noise."""
    if len(x) == 0:
        return np.zeros(0, int)
    return DBSCAN(eps=eps_m, min_samples=int(min_samples)).fit_predict(np.column_stack([x, y]))


def footprints_near(shop_x, shop_y, bld_x, bld_y, bld_area, radius_m: float) -> np.ndarray:
    """Area of the nearest building centroid within radius_m of each shop (0 if none).

    Each building is counted once per cluster by the caller (de-duplicated on building id).
    Returns (building index or -1) per shop.
    """
    if len(bld_x) == 0:
        return np.full(len(shop_x), -1)
    tree = cKDTree(np.column_stack([bld_x, bld_y]))
    dist, idx = tree.query(np.column_stack([shop_x, shop_y]), distance_upper_bound=radius_m)
    return np.where(np.isfinite(dist), idx, -1)


def build_streets(shops: pd.DataFrame, labels: np.ndarray, building_area: np.ndarray | None = None,
                  shop_building: np.ndarray | None = None) -> pd.DataFrame:
    """Aggregate clustered shops into the streets table (without hex_idx; added once the index
    is final). `shops` needs x, y, khatt_category, unbranded, and optionally street_name."""
    df = shops.assign(label=labels)
    df = df[df["label"] >= 0]
    rows = []
    for new_id, (label, g) in enumerate(df.groupby("label", sort=True)):
        cx, cy = g["x"].mean(), g["y"].mean()
        lat, lng = from_utm(cx, cy)
        names = g.get("street_name")
        names = names.dropna().astype(str).str.strip() if names is not None else pd.Series([])
        names = names[names != ""]
        name = names.mode().iat[0] if len(names) else f"Cluster {new_id}"
        counts = {c: int((g["khatt_category"] == c).sum()) for c in CATS}
        if building_area is not None and shop_building is not None:
            bld = np.unique(shop_building[g.index.to_numpy()])
            bld = bld[bld >= 0]
            footprint = float(building_area[bld].sum())
        else:
            footprint = 0.0
        rows.append({
            "street_id": new_id, "name": name, "lat": float(lat), "lng": float(lng),
            **{f"n_{c}": counts[c] for c in CATS},
            "n_unbranded": int(g["unbranded"].sum()),
            "footprint_m2": footprint,
        })
    st = pd.DataFrame(rows)
    if st.empty:
        return st
    st["h3_r8"] = cells_for_points(st["lat"], st["lng"], 8)
    st["h3_r9"] = cells_for_points(st["lat"], st["lng"], 9)
    for c in CATS:
        st[f"A_{c}"] = st[f"n_{c}"].astype(np.float32)  # attractiveness default = shop counts
    return st


def finalize_dtypes(st: pd.DataFrame, hex_pos: dict[str, int]) -> pd.DataFrame:
    """Attach hex_idx and cast to the §5 contract dtypes."""
    st = st.copy()
    st["hex_idx"] = st["h3_r8"].map(hex_pos).astype("int32")
    for c in ("street_id", "n_daily", "n_comp", "n_eat", "n_unbranded"):
        st[c] = st[c].astype("int32")
    for c in ("footprint_m2", "A_daily", "A_comp", "A_eat"):
        st[c] = st[c].astype("float32")
    st["lat"] = st["lat"].astype("float64")
    st["lng"] = st["lng"].astype("float64")
    cols = ["street_id", "name", "lat", "lng", "h3_r8", "h3_r9", "hex_idx", "n_daily", "n_comp",
            "n_eat", "n_unbranded", "footprint_m2", "A_daily", "A_comp", "A_eat"]
    return st[cols]
