import json
import os

import pytest
from fastapi.testclient import TestClient


@pytest.fixture(scope="module")
def client(world_dir, tmp_path_factory):
    scen = tmp_path_factory.mktemp("scenarios")
    (scen / "demo.json").write_text(json.dumps(
        {"name": "Demo", "description": "test", "lines": [{}], "result": {"totals": {}}}))
    old = {k: os.environ.get(k) for k in ("KHATT_PROCESSED_DIR", "KHATT_SCENARIOS_DIR")}
    os.environ["KHATT_PROCESSED_DIR"] = str(world_dir)
    os.environ["KHATT_SCENARIOS_DIR"] = str(scen)
    from khatt.api.main import app

    with TestClient(app) as c:
        yield c
    for k, v in old.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v


def _line(line_id, pts):
    return {"line_id": line_id, "name": f"Line {line_id}",
            "stations": [{"lat": a, "lng": b} for a, b in pts]}


A = [(30.045 + 0.012 * k, 31.215 + 0.010 * k) for k in range(-3, 4)]
B = [(30.045 - 0.011 * k, 31.200 + 0.012 * k) for k in range(-3, 5)]


def test_health(client):
    h = client.get("/health").json()
    assert h["ok"] and h["synthetic"]
    assert h["t0_shape"] == [h["n_hexes"], h["n_hexes"]]


def test_baseline(client):
    b = client.get("/baseline").json()
    assert b["streets"] and "R0" in b["streets"][0]
    assert b["existing_network"]["features"]
    assert b["microbus_routes"]["features"]
    assert "CC BY-NC 4.0" in b["credits"]


def test_evaluate(client):
    r = client.post("/evaluate", json={"lines": [_line("A", A), _line("B", B)]})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["elapsed_ms"] < 2000
    assert body["streets"] and body["stations"]
    assert len(r.content) < 2_000_000


def test_evaluate_with_overrides_and_attribution(client):
    r = client.post("/evaluate", json={
        "lines": [_line("A", A), _line("B", B)],
        "params": {"spending.beta.daily": 0.3, "rents.elasticity": 0.5},
        "attribution": True,
    })
    assert r.status_code == 200, r.text
    assert r.json()["synergy"]


def test_evaluate_validation(client):
    assert client.post("/evaluate", json={"lines": [_line("A", A[:1])]}).status_code == 422
    bad = client.post("/evaluate", json={"lines": [_line("A", A)], "params": {"nope.x": 1}})
    assert bad.status_code == 422 and "Unknown parameter" in bad.text
    dup = client.post("/evaluate", json={"lines": [_line("A", A), _line("A", B)]})
    assert dup.status_code == 422


def test_params(client):
    p = client.get("/params").json()
    keys = {x["key"] for x in p["params"]}
    assert set(p["sliders"]) <= keys
    assert all(x["low"] <= x["central"] <= x["high"] for x in p["params"])


def test_scenarios(client):
    lst = client.get("/scenarios").json()
    assert [s["id"] for s in lst] == ["demo"]
    assert client.get("/scenarios/demo").json()["name"] == "Demo"
    assert client.get("/scenarios/nope").status_code == 404
    assert client.get("/scenarios/..%2Fsecret").status_code == 404


def test_engine_down_still_serves_scenarios(client, tmp_path):
    from khatt.api import main

    saved = main.STATE.world, main.STATE.error
    main.STATE.world, main.STATE.error = None, "no data"
    try:
        assert client.get("/health").json()["ok"] is False
        assert client.post("/evaluate", json={"lines": []}).status_code == 503
        assert client.get("/scenarios").status_code == 200
    finally:
        main.STATE.world, main.STATE.error = saved
