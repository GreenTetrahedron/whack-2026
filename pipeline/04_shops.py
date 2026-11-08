"""Overture places + OSM shops → categorised, de-duplicated shops (data/interim/shops.parquet)."""

from __future__ import annotations

import re

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely

from khatt.config import ROOT
from khatt.geo import to_utm
from khatt.prep.common import INTERIM, RAW, decision, require, setup, summary
from khatt.prep.shops import dedupe, load_crosswalk, map_category

LEADING_NUMBER = re.compile(r"^\s*[\d٠-٩]+[\s,،-]*")


def _get(obj, *keys):
    for k in keys:
        if obj is None:
            return None
        obj = obj.get(k) if isinstance(obj, dict) else None
    return obj


def _street_from_address(addresses) -> str | None:
    if addresses is None or len(addresses) == 0:
        return None
    free = _get(addresses[0], "freeform")
    if not free:
        return None
    return LEADING_NUMBER.sub("", str(free)).split(",")[0].strip() or None


def overture_categories(df: pd.DataFrame) -> list[list[str]]:
    """Most specific first. Releases since 2025 use `taxonomy` {primary, hierarchy, alternates} +
    `basic_category`; older ones use `categories` {primary, alternate}."""
    if "taxonomy" in df.columns:
        basic = df["basic_category"] if "basic_category" in df.columns else [None] * len(df)
        out = []
        for tax, b in zip(df["taxonomy"], basic):
            tax = tax or {}
            hierarchy = list(tax.get("hierarchy") if tax.get("hierarchy") is not None else [])
            alternates = list(tax.get("alternates") if tax.get("alternates") is not None else [])
            out.append([tax.get("primary"), *alternates, b, *hierarchy[::-1]])
        return out
    return [[_get(c, "primary")] + list(_get(c, "alternate") or []) if c is not None else []
            for c in df["categories"]]


def overture_places(path, params, cw, log) -> pd.DataFrame:
    df = pd.read_parquet(path)
    n_raw = len(df)
    df = df[df["confidence"].astype(float) >= float(params["shops.min_confidence"])]
    if "operating_status" in df.columns:
        closed = df["operating_status"].astype(str).str.contains("closed", case=False, na=False)
        df = df[~closed]
    geom = shapely.from_wkb(df["geometry"].to_numpy()) if df["geometry"].dtype == object else \
        gpd.GeoSeries(df["geometry"]).to_numpy()
    pts = shapely.point_on_surface(geom)
    cats = overture_categories(df)
    brand = df["brand"] if "brand" in df.columns else pd.Series([None] * len(df))
    out = pd.DataFrame({
        "source": "overture",
        "src_id": df["id"].astype(str).to_numpy(),
        "name": [_get(n, "primary") or "" for n in df["names"]],
        "lng": shapely.get_x(pts), "lat": shapely.get_y(pts),
        "khatt_category": [map_category(cw, "overture", c) for c in cats],
        "unbranded": [b is None or not (_get(b, "names", "primary") or _get(b, "wikidata"))
                      for b in brand],
        "street_name": [_street_from_address(a) for a in df.get("addresses", [None] * len(df))],
    })
    log.info("overture: %d raw → %d after confidence/closed filter", n_raw, len(out))
    return out


def osm_places(path, cw) -> pd.DataFrame:
    gdf = gpd.read_parquet(path)
    raw = [[s, a] for s, a in zip(gdf.get("shop", [None] * len(gdf)),
                                  gdf.get("amenity", [None] * len(gdf)))]
    name = gdf["name:ar"].fillna(gdf["name"]) if "name:ar" in gdf else gdf.get("name")
    return pd.DataFrame({
        "source": "osm",
        "src_id": gdf["id"].astype(str).to_numpy() if "id" in gdf else np.arange(len(gdf)).astype(str),
        "name": name.fillna("").astype(str).to_numpy() if name is not None else "",
        "lng": gdf.geometry.x.to_numpy(), "lat": gdf.geometry.y.to_numpy(),
        "khatt_category": [map_category(cw, "osm", r) for r in raw],
        "unbranded": gdf["brand"].isna().to_numpy() if "brand" in gdf else True,
        "street_name": gdf["addr:street"].to_numpy() if "addr:street" in gdf else None,
    })


def main() -> None:
    log, params, t0 = setup("04_shops")
    cw = load_crosswalk(ROOT / "config" / "category_crosswalk.csv")
    parts = []
    ov = RAW / "overture_places.parquet"
    osm = RAW / "osm_shops.parquet"
    if not ov.exists() and not osm.exists():
        require(ov, "run pipeline/00_download.py --only overture_places,osm_shops")
    if ov.exists():
        parts.append(overture_places(ov, params, cw, log))
    else:
        decision("04_shops", "Overture places missing: shops from OSM only")
    if osm.exists():
        parts.append(osm_places(osm, cw))
    else:
        decision("04_shops", "OSM shops missing: shops from Overture only")
    shops = pd.concat(parts, ignore_index=True)
    lng_min, lat_min, lng_max, lat_max = params["area.bbox"]
    shops = shops[shops.lat.between(lat_min, lat_max) & shops.lng.between(lng_min, lng_max)]
    x, y = to_utm(shops["lat"], shops["lng"])
    shops = shops.assign(x=x, y=y)
    before = len(shops)
    shops = dedupe(shops, float(params["shops.dedupe_radius_m"]),
                   float(params["shops.name_similarity"]))
    shops.to_parquet(INTERIM / "shops.parquet", index=False)
    by_cat = shops["khatt_category"].value_counts().to_dict()
    summary("04_shops", [f"{before} shops before de-dup, {len(shops)} after",
                         f"by category: {by_cat}",
                         f"by source: {shops['source'].value_counts().to_dict()}",
                         f"unbranded share: {shops['unbranded'].mean():.0%}"], t0)


if __name__ == "__main__":
    main()
