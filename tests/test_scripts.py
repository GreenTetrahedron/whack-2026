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
