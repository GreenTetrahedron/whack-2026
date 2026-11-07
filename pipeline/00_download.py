"""Download raw inputs into data/raw/ (CLAUDE.md §4). Idempotent: existing files are kept.

    python pipeline/00_download.py [--only gtfs_bus,worldpop] [--force]

Anything that needs a login or fails is listed at the end with what to fetch by hand.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import urllib.request
from pathlib import Path

from khatt.prep.common import RAW, decision, setup, summary

BBOX_KEY = "area.bbox"
URLS = {
    # TfC's own links (data.transportforcairo.com/documents/41|42/download) return 404 as of
    # Nov 2026; the Mobility Database mirrors the same feeds (mdb-1825 bus, mdb-786 metro 2022).
    "gtfs_bus": ("gtfs_bus.zip", "https://files.mobilitydatabase.org/mdb-1825/latest.zip"),
    "gtfs_metro": ("gtfs_metro.zip", "https://files.mobilitydatabase.org/mdb-786/latest.zip"),
    "osm_pbf": ("egypt-latest.osm.pbf",
                "https://download.geofabrik.de/africa/egypt-latest.osm.pbf"),
    "worldpop": ("egy_pop_2026_CN_100m_R2025A_v1.tif",
                 ("https://data.worldpop.org/GIS/Population/Global_2015_2030/R2025A/2026/EGY/v1/"
                 "100m/constrained/egy_pop_2026_CN_100m_R2025A_v1.tif")),
}
SPENDING_TEMPLATE = """governorate,coicop,value_egp,period,basis,household_size,source
# Fill from CAPMAS HIECS 2021 (censusinfo.capmas.gov.eg, catalog 747), URBAN, 2021 EGP.
# coicop: 01 food, 03 clothing, 05 furnishings, 09 recreation, 11 restaurants & hotels
# period: annual | monthly ; basis: household | person ; household_size needed when basis=household
Cairo,01,,annual,household,,
Cairo,03,,annual,household,,
Cairo,05,,annual,household,,
Cairo,09,,annual,household,,
Cairo,11,,annual,household,,
Giza,01,,annual,household,,
Giza,03,,annual,household,,
Giza,05,,annual,household,,
Giza,09,,annual,household,,
Giza,11,,annual,household,,
Qalyubia,01,,annual,household,,
Qalyubia,03,,annual,household,,
Qalyubia,05,,annual,household,,
Qalyubia,09,,annual,household,,
Qalyubia,11,,annual,household,,
"""
UA = "khatt-pipeline/0.1 (hackathon research)"


def fetch(url: str, dest: Path, log) -> None:
    tmp = dest.with_suffix(dest.suffix + ".part")
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=120) as resp, open(tmp, "wb") as fh:
        ctype = resp.headers.get("Content-Type", "")
        if "text/html" in ctype:
            raise RuntimeError(f"got an HTML page ({ctype}); probably needs a login")
        total = int(resp.headers.get("Content-Length") or 0)
        done = 0
        while chunk := resp.read(1 << 20):
            fh.write(chunk)
            done += len(chunk)
            if total and done % (50 << 20) < (1 << 20):
                log.info("  %s: %.0f / %.0f MB", dest.name, done / 1e6, total / 1e6)
    tmp.rename(dest)


def overture(kind: str, bbox, dest: Path, log) -> None:
    from overturemaps import core

    reader = core.record_batch_reader(kind, bbox=tuple(bbox))
    if reader is None:
        raise RuntimeError(f"no Overture data returned for {kind}")
    import pyarrow.parquet as pq

    writer = None
    rows = 0
    for batch in reader:
        if writer is None:
            writer = pq.ParquetWriter(dest.with_suffix(".part"), batch.schema)
        writer.write_batch(batch)
        rows += batch.num_rows
    if writer is None:
        raise RuntimeError(f"Overture {kind}: empty result")
    writer.close()
    dest.with_suffix(".part").rename(dest)
    log.info("  overture %s: %d rows", kind, rows)


def osm_shops(bbox, dest: Path) -> None:
    import osmnx as ox

    tags = {"shop": True, "amenity": ["restaurant", "cafe", "fast_food", "marketplace",
                                      "food_court", "ice_cream"]}
    gdf = ox.features_from_bbox(tuple(bbox), tags=tags).reset_index()
    keep = [c for c in ("element", "id", "name", "name:ar", "name:en", "shop", "amenity", "brand",
                        "addr:street", "geometry") if c in gdf.columns]
    gdf = gdf[keep].copy()
    gdf["geometry"] = gdf.geometry.representative_point()
    gdf.to_parquet(dest)


def governorates(dest: Path) -> None:
    import osmnx as ox

    names = ["Cairo Governorate, Egypt", "Giza Governorate, Egypt", "Qalyubia Governorate, Egypt"]
    gdf = ox.geocode_to_gdf(names)
    gdf["governorate"] = ["Cairo", "Giza", "Qalyubia"]
    gdf[["governorate", "geometry"]].to_file(dest, driver="GeoJSON")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--only", default="", help="comma-separated keys")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    log, params, t0 = setup("00_download")
    bbox = params[BBOX_KEY]
    only = {k for k in args.only.split(",") if k}
    want = lambda key: not only or key in only
    failures: list[str] = []
    done: list[str] = []

    def step(key: str, dest: Path, action, manual: str) -> None:
        if not want(key):
            return
        if dest.exists() and not args.force:
            log.info("%s: already present (%s)", key, dest.name)
            done.append(key)
            return
        log.info("%s: downloading → %s", key, dest)
        try:
            action()
            done.append(key)
        except Exception as exc:  # noqa: BLE001 — report every failure, keep going
            for leftover in (dest.with_suffix(dest.suffix + ".part"), dest.with_suffix(".part")):
                leftover.unlink(missing_ok=True)
            failures.append(f"{key}: {exc}\n    → {manual}")

    for key, (name, url) in URLS.items():
        dest = RAW / name
        step(key, dest, lambda u=url, d=dest: fetch(u, d, log),
             f"download {url} in a browser and save it as {dest}")

    step("overture_places", RAW / "overture_places.parquet",
         lambda: overture("place", bbox, RAW / "overture_places.parquet", log),
         f"overturemaps download --bbox={','.join(map(str, bbox))} -f geoparquet --type=place "
         f"-o {RAW / 'overture_places.parquet'}")
    step("overture_buildings", RAW / "overture_buildings.parquet",
         lambda: overture("building", bbox, RAW / "overture_buildings.parquet", log),
         f"overturemaps download --bbox={','.join(map(str, bbox))} -f geoparquet --type=building "
         f"-o {RAW / 'overture_buildings.parquet'}")
    step("osm_shops", RAW / "osm_shops.parquet", lambda: osm_shops(bbox, RAW / "osm_shops.parquet"),
         "check Overpass availability and re-run with --only osm_shops")
    step("governorates", RAW / "governorates.geojson",
         lambda: governorates(RAW / "governorates.geojson"),
         "optional: save Cairo/Giza/Qalyubia boundaries as GeoJSON with a 'governorate' column; "
         "01_grid.py otherwise uses an approximate rule")

    # OSM clip for r5py (smaller network, less memory)
    clip = RAW / "cairo.osm.pbf"
    if want("osm_pbf") and (RAW / URLS["osm_pbf"][0]).exists() and not clip.exists():
        if shutil.which("osmium"):
            b = ",".join(map(str, bbox))
            subprocess.run(["osmium", "extract", "-b", b, str(RAW / URLS["osm_pbf"][0]), "-o",
                            str(clip)], check=False)
        else:
            decision("00_download", "osmium not installed: r5py uses the full Egypt OSM extract "
                     "(more memory). `brew install osmium-tool` to clip to the bbox.")

    # Spending: CAPMAS tables are not machine-downloadable → human fills a CSV
    manual = RAW / "spending_manual.csv"
    if want("spending") and not manual.exists():
        template = RAW / "spending_manual.template.csv"
        template.write_text(SPENDING_TEMPLATE)
        failures.append(f"spending: fill {template} from CAPMAS HIECS 2021 tables and save it as "
                        f"{manual}")

    if want("wealth") and not any((RAW / n).exists() for n in ("rwi_egypt.csv", "viirs.tif")):
        log.info("wealth (optional): no Meta RWI (data/raw/rwi_egypt.csv) or VIIRS "
                 "(data/raw/viirs.tif); wealth factor will be 1.0 everywhere")

    summary("00_download", [f"ok: {', '.join(done) or '-'}",
                            f"failed/manual: {len(failures)}"], t0)
    if failures:
        print("\nACTION NEEDED — fetch these by hand, then re-run:", file=sys.stderr)
        for f in failures:
            print(f"  - {f}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
