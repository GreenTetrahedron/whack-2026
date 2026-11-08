"""FastAPI app. The World is loaded once at start-up; the API is stateless.

    uvicorn khatt.api.main:app --reload --port 8000
"""

from __future__ import annotations

import json
import logging
import math
import os
import re
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from khatt.api.schemas import EvaluateRequest, Health, ParamRange
from khatt.config import CATEGORIES, Params, processed_dir, scenarios_dir
from khatt.data import DataMissing, load_world
from khatt.engine.overlay import warmup
from khatt.evaluate import evaluate as run_evaluate
from khatt.models.zones import warmup as zones_warmup

log = logging.getLogger("khatt.api")
SLIDERS = [f"spending.beta.{c}" for c in CATEGORIES] + [
    "rents.elasticity", "spending.induced_pct", "zones.threshold", "zones.work_trip_share"]
SCENARIO_ID = re.compile(r"^[A-Za-z0-9_-]{1,64}$")
SIMPLIFY_DEG = 0.0003  # ≈ 30 m, for microbus geometries in /baseline


class State:
    world = None
    params: Params | None = None
    error: str | None = None
    baseline_cache: dict | None = None


STATE = State()


def load_state() -> None:
    STATE.params = Params.load()
    STATE.baseline_cache = None
    try:
        STATE.world = load_world(processed_dir())
        STATE.error = None
        STATE.world.r9_xy()  # used by the zones model; compute before the first request
        warmup()
        zones_warmup()
    except (DataMissing, ValueError, OSError) as exc:
        STATE.world = None
        STATE.error = str(exc)
        log.warning("Engine disabled (scenarios still served): %s", exc)


@asynccontextmanager
async def lifespan(_app: FastAPI):
    load_state()
    yield


app = FastAPI(title="Khatt — Cairo Metro What-If", version="0.1.0", lifespan=lifespan)
origins = [o.strip() for o in os.environ.get(
    "KHATT_CORS_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173").split(",") if o.strip()]
app.add_middleware(CORSMiddleware, allow_origins=origins, allow_methods=["*"],
                   allow_headers=["*"])


def _finite(obj):
    """Replace NaN/inf with None (JSON has no NaN)."""
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, dict):
        return {k: _finite(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_finite(v) for v in obj]
    return obj


def json_response(content) -> JSONResponse:
    try:
        return JSONResponse(content)
    except ValueError:
        return JSONResponse(_finite(content))


def _world():
    if STATE.world is None:
        raise HTTPException(503, f"Engine unavailable: {STATE.error}. Scenarios still work.")
    return STATE.world


@app.get("/health", response_model=Health)
def health():
    w = STATE.world
    if w is None:
        return Health(ok=False, error=STATE.error)
    return Health(ok=True, n_hexes=w.n, n_streets=len(w.streets), t0_shape=list(w.T0.shape),
                  synthetic=(w.path / "SYNTHETIC").exists(), load_s=round(w.load_s, 2))


@app.get("/baseline")
def baseline():
    w = _world()
    if STATE.baseline_cache is None:
        STATE.baseline_cache = build_baseline(w, STATE.params)
    return json_response(STATE.baseline_cache)


def build_baseline(w, params: Params) -> dict:
    from khatt.models.spending import run_huff

    huff = run_huff(w.spend, w.street_A, w.T0[:, w.street_hex], params)
    st = w.streets
    streets = [{
        "street_id": int(st["street_id"].iat[k]), "name": str(st["name"].iat[k]),
        "lat": float(st["lat"].iat[k]), "lng": float(st["lng"].iat[k]),
        "R0": {c: round(float(huff.R[k, i])) for i, c in enumerate(CATEGORIES)},
        "R0_total": round(float(huff.R[k].sum())),
        "n_shops": {c: int(st[f"n_{c}"].iat[k]) for c in CATEGORIES},
        "n_unbranded": int(st["n_unbranded"].iat[k]),
    } for k in range(len(st))]
    ex = w.existing
    stations = [{
        "stop_id": str(r.stop_id), "line": str(r.line), "name": str(r.name),
        "lat": float(r.lat), "lng": float(r.lng), "catchment_pop10": float(r.catchment_pop10),
    } for r in ex.itertuples()]
    mb = w.microbus
    routes = {"type": "FeatureCollection", "features": [{
        "type": "Feature",
        "properties": {"route_id": str(r.route_id), "agency": str(r.agency),
                       "headway_min": float(r.headway_min)},
        "geometry": json.loads(json.dumps(r.geometry.simplify(SIMPLIFY_DEG).__geo_interface__)),
    } for r in mb.itertuples() if r.geometry is not None]}
    return {
        "streets": streets,
        "existing_network": w.existing_lines or {"type": "FeatureCollection", "features": []},
        "existing_stations": stations,
        "microbus_routes": routes,
        "credits": "Transit data: Transport for Cairo (CC BY-NC 4.0)",
    }


@app.post("/evaluate")
def evaluate(req: EvaluateRequest):
    w = _world()
    try:
        params = STATE.params.with_overrides(req.params)
        result = run_evaluate(w, [ln.to_spec() for ln in req.lines], params,
                              attribution=req.attribution)
    except (ValueError, KeyError) as exc:
        raise HTTPException(422, str(exc).strip("'\"")) from exc
    return json_response(result)


def _scenario_files() -> list[Path]:
    d = scenarios_dir()
    return sorted(p for p in d.glob("*.json")) if d.exists() else []


@app.get("/scenarios")
def scenarios():
    out = []
    for f in _scenario_files():
        try:
            data = json.loads(f.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        out.append({"id": f.stem, "name": data.get("name", f.stem),
                    "description": data.get("description", ""),
                    "n_lines": len(data.get("lines", [])),
                    "has_montecarlo": bool(data.get("montecarlo"))})
    return out


@app.get("/scenarios/{scenario_id}")
def scenario(scenario_id: str):
    if not SCENARIO_ID.match(scenario_id):
        raise HTTPException(404, "Unknown scenario")
    f = scenarios_dir() / f"{scenario_id}.json"
    if not f.exists():
        raise HTTPException(404, "Unknown scenario")
    return json_response(json.loads(f.read_text()))


@app.get("/params")
def params():
    p = STATE.params or Params.load()
    ranges = [ParamRange(key=k, low=v["low"], central=v["central"], high=v["high"],
                         value=v["value"], source=str(v["source"]))
              for k, v in p.uncertain().items()]
    return {"params": [r.model_dump() for r in ranges], "sliders": SLIDERS}
