"""/evaluate < 2 s and < 2 MB on a full-size (≈3,000-cell) world. Run: pytest -m slow."""

import os
import time

import pytest
from fastapi.testclient import TestClient

pytestmark = pytest.mark.slow


@pytest.fixture(scope="module")
def big_client(tmp_path_factory, params):
    from khatt.synthetic import make_synthetic_world

    path = make_synthetic_world(tmp_path_factory.mktemp("big"), rings=31, seed=2,
                                walk_spacing_m=200, params=params)
    old = os.environ.get("KHATT_PROCESSED_DIR")
    os.environ["KHATT_PROCESSED_DIR"] = str(path)
    from khatt.api.main import app

    with TestClient(app) as c:
        assert c.get("/health").json()["n_hexes"] > 2900
        yield c
    if old is None:
        os.environ.pop("KHATT_PROCESSED_DIR", None)
    else:
        os.environ["KHATT_PROCESSED_DIR"] = old


def _line(line_id, lat0, lng0, dlat, dlng, n):
    return {"line_id": line_id, "name": line_id,
            "stations": [{"lat": lat0 + dlat * k, "lng": lng0 + dlng * k} for k in range(n)]}


def _timed(client, body):
    t = time.perf_counter()
    r = client.post("/evaluate", json=body)
    return r, time.perf_counter() - t


def test_two_lines_twenty_stations(big_client):
    body = {"lines": [_line("A", 29.95, 31.15, 0.009, 0.008, 20),
                      _line("B", 30.13, 31.12, -0.008, 0.011, 20)]}
    _timed(big_client, body)  # warm caches
    r, seconds = _timed(big_client, body)
    assert r.status_code == 200
    assert seconds < 2.0, f"/evaluate took {seconds:.2f} s"
    assert len(r.content) < 2_000_000


def test_four_lines_forty_stations_with_attribution(big_client):
    body = {"lines": [_line("A", 29.95, 31.15, 0.018, 0.016, 10),
                      _line("B", 30.13, 31.12, -0.016, 0.022, 10),
                      _line("C", 30.0, 31.10, 0.0, 0.025, 10),
                      _line("D", 29.93, 31.30, 0.02, 0.0, 10)],
            "attribution": True}
    r, seconds = _timed(big_client, body)
    assert r.status_code == 200
    assert seconds < 2.0, f"/evaluate with attribution took {seconds:.2f} s"
    assert len(r.content) < 2_000_000
