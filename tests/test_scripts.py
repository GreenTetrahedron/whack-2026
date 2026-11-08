"""Pipeline and helper scripts: import smoke tests + runs on synthetic / tiny inputs."""

import importlib.util
import json
import os
import sys
from pathlib import Path

import pytest

from khatt.config import ROOT
from tests.test_prep import _tiny_feed

SCRIPTS = sorted((ROOT / "pipeline").glob("*.py")) + sorted((ROOT / "scripts").glob("*.py"))


def _load(path: Path):
    spec = importlib.util.spec_from_file_location(f"_script_{path.stem}", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.mark.parametrize("path", SCRIPTS, ids=lambda p: p.name)
def test_script_imports(path):
    mod = _load(path)
    assert callable(getattr(mod, "main", None)), f"{path.name} has no main()"


def test_rail_stations_from_feed(tmp_path):
    from khatt.prep import gtfs

    mod = _load(ROOT / "pipeline" / "09_existing_stations.py")
    feed = gtfs.read_feed(_tiny_feed(tmp_path))
    gtfs.fix_stop_coordinates(feed, (30.85, 29.80, 31.85, 30.35))
    stations, features = mod.rail_stations(feed)
    assert stations["stop_id"].tolist() == ["s1", "s2", "s3"]
    assert len(features) == 1 and len(features[0]["geometry"]["coordinates"]) == 3


def test_underserved_line(world):
    mod = _load(ROOT / "scripts" / "precompute_scenarios.py")
    line = mod.underserved_line(world)
    assert len(line.stations) >= 5
    lat, lng = line.stations[0]
    assert 29.5 < lat < 30.5 and 30.8 < lng < 31.9


def test_precompute_and_mock(world_dir, tmp_path, monkeypatch):
    monkeypatch.setenv("KHATT_PROCESSED_DIR", str(world_dir))
    monkeypatch.setenv("KHATT_SCENARIOS_DIR", str(tmp_path / "scen"))
    mod = _load(ROOT / "scripts" / "precompute_scenarios.py")
    monkeypatch.setattr(sys, "argv", ["x", "--only", "underserved", "--runs", "3"])
    mod.main()
    doc = json.loads((tmp_path / "scen" / "underserved.json").read_text())
    assert doc["synthetic_data"] is True
    assert doc["result"]["stations"] and doc["montecarlo"]["n_runs"] == 3

    mock = _load(ROOT / "scripts" / "write_mock.py")
    monkeypatch.setattr(sys, "argv", ["x", str(world_dir), str(tmp_path / "mock")])
    mock.main()
    for name in ("evaluate.json", "baseline.json", "evaluate_request.json"):
        assert (tmp_path / "mock" / name).stat().st_size > 100
    assert os.environ["KHATT_PROCESSED_DIR"] == str(world_dir)


def test_spending_table_conversions(tmp_path):
    mod = _load(ROOT / "pipeline" / "03_spending.py")
    committed, notes = mod.per_capita_table(ROOT / "config" / "spending_hiecs_2019_20.csv", 1.06)
    assert set(committed) == {"Cairo", "Giza", "Qalyubia"}
    assert committed["Cairo"]["01"] == pytest.approx(5055.0 * 1.06)
    assert committed["Giza"]["11"] == pytest.approx(967.1 * 1.06)
    assert any("2019/20" in n for n in notes)

    csv = tmp_path / "s.csv"
    csv.write_text("governorate,coicop,value_egp,period,basis,household_size,prices,source\n"
                   "Cairo,1,1200,monthly,household,4,2021,x\n"
                   "Cairo,11,800,annual,person,,2019/20,x\n")
    table, _ = mod.per_capita_table(csv, 1.1)
    assert table["Cairo"]["01"] == pytest.approx(1200 * 12 / 4)
    assert table["Cairo"]["11"] == pytest.approx(880.0)


def test_fill_from_fallback():
    import numpy as np

    mod = _load(ROOT / "pipeline" / "08_baseline_matrix.py")
    T0 = np.array([[4, 10, 999, 999], [12, 4, 999, 999], [999, 999, 4, 999],
                   [999, 50, 999, 4]], np.float32)
    F = np.array([[4, 20, 30, 300], [20, 4, 40, 90], [35, 45, 4, 60], [60, 55, 65, 4]], np.float32)
    out, n_dead, n_pairs = mod.fill_from_fallback(T0, F, 999.0, 120.0)
    assert n_dead == 1, "cell 2 is unreachable from everywhere"
    assert out[2].tolist() == [35, 45, 4, 60] and out[:, 2].tolist() == [30, 40, 4, 65]
    assert out[0, 1] == 10 and out[3, 1] == 50, "pairs r5py routed keep the r5py value"
    assert out[1, 3] == 90 and out[3, 0] == 60, "gaps filled where the fallback is <= 120 min"
    assert out[0, 3] == 999, "fallback > 120 min stays unreachable"
    assert n_pairs == 2


def test_routing_points_snap_to_walk_network():
    import numpy as np

    from khatt.engine.catchment import WalkGraph
    from khatt.geo import haversine_m, to_utm

    mod = _load(ROOT / "pipeline" / "08_baseline_matrix.py")
    # a two-node walk graph near Tahrir; one cell 150 m away, one 5 km away
    nlat, nlng = np.array([30.0444, 30.0450]), np.array([31.2357, 31.2360])
    x, y = to_utm(nlat, nlng)
    g = WalkGraph.from_edges(np.asarray(x), np.asarray(y), [0, 1], [1, 0], [1.0, 1.0],
                             np.full(2, -1))
    lat, lng = np.array([30.0457, 30.0900]), np.array([31.2370, 31.2357])
    rlat, rlng, moved = mod.routing_points(lat, lng, g, max_m=2500)
    assert np.allclose([rlat[0], rlng[0]], [nlat[1], nlng[1]], atol=1e-6), "moved to nearest node"
    assert 100 < moved[0] < 200
    assert (rlat[1], rlng[1], moved[1]) == (lat[1], lng[1], 0.0), "too far: centroid kept"
    assert abs(haversine_m(rlat[0], rlng[0], lat[0], lng[0]) - moved[0]) < 2, "moved_m in metres"
    same = mod.routing_points(lat, lng, None, 2500)
    assert same[0] is lat and not same[2].any(), "no walk graph: centroids unchanged"
