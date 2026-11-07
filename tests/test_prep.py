"""Pipeline helpers (khatt/prep) on tiny hand-made inputs."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from khatt.config import ROOT
from khatt.geo import to_utm
from khatt.prep import gtfs
from khatt.prep.fallback_matrix import Route, build_fallback_T0
from khatt.prep.shops import dedupe, load_crosswalk, map_category
from khatt.prep.spending import allocate, category_totals, normalise_wealth
from khatt.prep.streets import build_streets, cluster, finalize_dtypes
from khatt.prep.text import normalize_name, similarity


# ---------------- text ----------------
@pytest.mark.parametrize("raw, expected", [
    ("أحمد", "احمد"),
    ("إسلام", "اسلام"),
    ("آمال", "امال"),
    ("مكتبةُ", "مكتبه"),
    ("مـصـر", "مصر"),
    ("مستشفى", "مستشفي"),
    ("  Cafe   ROYAL ", "cafe royal"),
    (None, ""),
])
def test_normalize_name(raw, expected):
    assert normalize_name(raw) == expected


def test_similarity():
    assert similarity("كشري التحرير", "كشري التحرير") == 1.0
    assert similarity("", "x") == 0.0
    assert similarity(normalize_name("مطعم أبو طارق"), normalize_name("مطعم ابو طارق")) == 1.0


# ---------------- shops ----------------
def test_crosswalk_file():
    cw = load_crosswalk(ROOT / "config" / "category_crosswalk.csv")
    assert map_category(cw, "overture", ["supermarket"]) == "daily"
    assert map_category(cw, "overture", ["unknown_thing", "clothing_store"]) == "comp"
    assert map_category(cw, "osm", "restaurant") == "eat"
    assert map_category(cw, "osm", "car_repair") == "other"
    assert map_category(cw, "osm", None) == "other"


def test_dedupe_across_sources_only():
    x0, y0 = to_utm(30.05, 31.24)
    shops = pd.DataFrame({
        "x": [x0, x0 + 10, x0 + 5, x0 + 500, x0 + 8],
        "y": [y0, y0, y0 + 3, y0, y0 + 2],
        "khatt_category": ["eat", "eat", "eat", "eat", "comp"],
        "name": ["مطعم أبو طارق", "مطعم ابو طارق", "مطعم ابو طارق", "مطعم ابو طارق",
                 "مطعم ابو طارق"],
        "source": ["overture", "osm", "overture", "osm", "osm"],
    })
    out = dedupe(shops, radius_m=40, min_similarity=0.8)
    # one OSM duplicate near the Overture pair is dropped; far OSM and other category stay
    assert len(out) == 4
    assert (out["source"] == "overture").sum() == 2


# ---------------- streets ----------------
def test_cluster_and_build_streets():
    rng = np.random.default_rng(0)
    cx, cy = to_utm(30.05, 31.24)
    blobs = [(cx, cy), (cx + 3000, cy + 1000)]
    xs, ys = [], []
    for bx, by in blobs:
        xs.append(bx + rng.normal(0, 30, 25))
        ys.append(by + rng.normal(0, 30, 25))
    xs.append(cx + rng.uniform(-8000, 8000, 10))
    ys.append(cy + rng.uniform(-8000, 8000, 10))
    x, y = np.concatenate(xs), np.concatenate(ys)
    shops = pd.DataFrame({
        "x": x, "y": y,
        "khatt_category": rng.choice(["daily", "comp", "eat", "other"], len(x)),
        "unbranded": rng.random(len(x)) < 0.7,
        "street_name": ["Qasr El Nil"] * 25 + [""] * 35,
    })
    labels = cluster(x, y, eps_m=150, min_samples=15)
    assert len(set(labels) - {-1}) == 2
    st = build_streets(shops, labels)
    assert len(st) == 2
    assert st.loc[0, "name"] == "Qasr El Nil" and st.loc[1, "name"] == "Cluster 1"
    assert (st[["n_daily", "n_comp", "n_eat"]].sum(axis=1) <= 25).all()
    final = finalize_dtypes(st, {h: k for k, h in enumerate(st["h3_r8"])})
    assert final["hex_idx"].dtype == np.int32 and final["A_comp"].dtype == np.float32


# ---------------- spending ----------------
def test_allocate_preserves_governorate_totals():
    pop = np.array([100, 200, 300, 400.0])
    wealth = np.array([0.5, 1.5, 1.0, 2.0])
    gov = np.array(["Cairo", "Cairo", "Giza", "Giza"])
    pc = {"Cairo": {"daily": 10, "comp": 5, "eat": 2}, "Giza": {"daily": 8, "comp": 4, "eat": 1}}
    out = allocate(pop, wealth, gov, pc)
    assert np.isclose(out["daily"][:2].sum(), 300 * 10)
    assert np.isclose(out["comp"][2:].sum(), 700 * 4)
    # richer cell gets more per person
    assert out["daily"][1] / 200 > out["daily"][0] / 100


def test_category_totals_and_wealth():
    t = category_totals({"01": 100, "03": 10, "05": 20, "09": 50, "11": 7}, 0.3)
    assert t == {"daily": 100.0, "comp": 45.0, "eat": 7.0}
    w = normalise_wealth([1, 2, np.nan], [10, 10, 10])
    assert np.isclose(np.average(w[:2], weights=[10, 10]), 1.0) and w[2] == 1.0


# ---------------- population ----------------
def test_raster_to_cells_preserves_total(tmp_path):
    rasterio = pytest.importorskip("rasterio")
    from rasterio.transform import from_origin

    from khatt.prep.population import raster_to_cells

    data = np.random.default_rng(0).random((40, 40)).astype(np.float32) * 50
    data[0, :] = -99999  # nodata row
    path = tmp_path / "pop.tif"
    with rasterio.open(path, "w", driver="GTiff", height=40, width=40, count=1, dtype="float32",
                       crs="EPSG:4326", transform=from_origin(31.2, 30.06, 0.001, 0.001),
                       nodata=-99999) as dst:
        dst.write(data, 1)
    cells, total = raster_to_cells(path, (31.0, 29.9, 31.4, 30.2))
    expected = data[1:].sum()
    assert np.isclose(total, expected, rtol=1e-6)
    assert np.isclose(cells[8]["pop"].sum(), total, rtol=1e-6)
    assert np.isclose(cells[9]["pop"].sum(), total, rtol=1e-6)


# ---------------- GTFS ----------------
def _tiny_feed(tmp: Path) -> Path:
    d = tmp / "feed"
    d.mkdir()
    files = {
        "agency.txt": "agency_id,agency_name,agency_url,agency_timezone\n"
                      "cta,CTA,http://x,Africa/Cairo\nmb,Microbus Operators,http://x,Africa/Cairo\n",
        "routes.txt": "route_id,agency_id,route_short_name,route_type\n"
                      "M1,cta,M1,1\nB7,mb,B7,3\n",
        "stops.txt": "stop_id,stop_name,stop_lat,stop_lon\n"
                     "s1,One,30.05,31.23\ns2,Two,30.06,31.24\ns3,Three,31.25,30.07\n",
        "trips.txt": "route_id,service_id,trip_id,direction_id\n"
                     "M1,wk,t1,0\nB7,wk,t2,0\nB7,wk,bad,1\n",
        "stop_times.txt": "trip_id,arrival_time,departure_time,stop_id,stop_sequence\n"
                          "t1,23:58:00,23:58:00,s1,1\nt1,24:02:00,24:02:30,s2,2\n"
                          "t1,24:05:00,24:05:00,s3,3\n"
                          "t2,08:00:00,08:00:00,s1,1\nt2,08:10:00,08:10:00,s2,2\n"
                          "bad,08:10:00,08:10:00,s2,1\nbad,08:05:00,08:05:00,s1,2\n",
        "frequencies.txt": "trip_id,start_time,end_time,headway_secs,exact_times\n"
                           "t1,06:00:00,22:00:00,300,0\nt2,07:00:00,10:00:00,600,0\n",
        "calendar.txt": "service_id,monday,tuesday,wednesday,thursday,friday,saturday,sunday,"
                        "start_date,end_date\nwk,1,1,1,1,1,1,1,20210101,20220630\n",
    }
    for name, text in files.items():
        (d / name).write_text(text)
    return d


def test_gtfs_checks(tmp_path):
    from datetime import date

    feed = gtfs.read_feed(_tiny_feed(tmp_path))
    assert gtfs.non_monotonic_trips(feed) == ["bad"], "times past 24:00 are valid"
    swapped, outside = gtfs.fix_stop_coordinates(feed, (30.85, 29.80, 31.85, 30.35))
    assert (swapped, outside) == (1, 0)
    assert feed["stops.txt"].set_index("stop_id").at["s3", "stop_lat"] == "30.07"
    assert gtfs.has_service_on(feed, date(2022, 3, 15))
    assert not gtfs.has_service_on(feed, date(2023, 3, 15))
    pre = gtfs.prefix_ids(feed, "bus:")
    assert pre["stop_times.txt"]["trip_id"].str.startswith("bus:").all()
    assert pre["routes.txt"]["agency_id"].str.startswith("bus:").all()
    gtfs.drop_routes(feed, ["B7"])
    assert set(feed["trips.txt"]["trip_id"]) == {"t1"}
    out = tmp_path / "out.zip"
    gtfs.write_feed(pre, out)
    assert set(gtfs.read_feed(out)) == set(pre)


def test_representative_routes(tmp_path):
    feed = gtfs.read_feed(_tiny_feed(tmp_path))
    gtfs.fix_stop_coordinates(feed, (30.85, 29.80, 31.85, 30.35))
    gtfs.drop_trips(feed, gtfs.non_monotonic_trips(feed))
    routes = {r.route_id: r for r in gtfs.representative_routes(feed, 8 * 3600, 3600, 15)}
    assert routes["M1|0"].mode == "metro" and routes["M1|0"].headway_min == 5
    assert routes["M1|0"].ride_min == [4.0, 3.0]
    assert routes["B7|0"].mode == "microbus" and routes["B7|0"].headway_min == 10
    assert gtfs.route_mode("3", "Cairo Transport Authority") == "bus"


def test_manual_feed(tmp_path):
    spec = {"lines": [
        {"route_id": "L3W", "name": "west", "headway_min": 6,
         "stations": [{"name": "A", "lat": 30.07, "lng": 31.21},
                      {"name": "B", "lat": 30.08, "lng": 31.20},
                      {"name": "C", "lat": 30.09, "lng": 31.19}]},
        {"route_id": "EMPTY", "stations": [{"name": "X", "lat": None, "lng": None}]},
    ]}
    path = tmp_path / "manual.yaml"
    path.write_text(yaml.safe_dump(spec))
    feed, warnings = gtfs.manual_feed(path, "2022-03-15")
    assert any("EMPTY" in w for w in warnings)
    assert len(feed["routes.txt"]) == 1
    assert feed["routes.txt"]["route_id"].iat[0] == "manual:L3W"
    assert len(feed["trips.txt"]) == 2
    from datetime import date

    assert gtfs.has_service_on(feed, date(2022, 3, 15))
    assert gtfs.non_monotonic_trips(feed) == []


def test_repo_manual_lines_yaml_parses():
    spec = yaml.safe_load((ROOT / "config" / "manual_lines.yaml").read_text())
    assert spec["lines"][0]["stations"][0]["lat"] is None


# ---------------- fallback matrix ----------------
def test_fallback_matrix(params):
    import h3

    centre = h3.latlng_to_cell(30.05, 31.24, 8)
    cells = sorted(h3.grid_disk(centre, 4))
    lat = np.array([h3.cell_to_latlng(c)[0] for c in cells])
    lng = np.array([h3.cell_to_latlng(c)[1] for c in cells])
    far = np.argmax(np.hypot(lat - 30.05, lng - 31.24))
    T_walk = build_fallback_T0(cells, lat, lng, [], params)
    route = Route("r", "metro", 4.0, [30.05, float(lat[far])], [31.24, float(lng[far])],
                  speed_kmh=60.0, bidirectional=True)
    T_metro = build_fallback_T0(cells, lat, lng, [route], params)
    c = cells.index(centre)
    assert T_metro[c, far] < T_walk[c, far]
    assert (T_metro <= T_walk + 1e-4).all()
    assert np.allclose(np.diag(T_walk), params["access.intracell_min"])
    # isolated extra cell → unreachable
    lone = h3.latlng_to_cell(30.30, 31.80, 8)
    T_iso = build_fallback_T0(cells + [lone], np.append(lat, 30.30), np.append(lng, 31.80), [],
                              params)
    assert T_iso[0, -1] == 999.0
