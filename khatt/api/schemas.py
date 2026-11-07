"""Pydantic request/response models for the API."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from khatt.engine.lines import LineSpec


class StationIn(BaseModel):
    """Stations are objects with explicit lat/lng to avoid [lng, lat] vs (lat, lng) mix-ups."""

    lat: float = Field(ge=-90, le=90)
    lng: float = Field(ge=-180, le=180)


class LineIn(BaseModel):
    model_config = ConfigDict(extra="ignore")

    line_id: str = Field(min_length=1, max_length=40)
    name: str = Field(default="", max_length=120)
    stations: list[StationIn] = Field(min_length=2, max_length=150)
    headway_min: float | None = Field(default=None, gt=0, le=120)
    speed_kmh: float | None = Field(default=None, gt=0, le=200)
    dwell_s: float | None = Field(default=None, ge=0, le=600)

    def to_spec(self) -> LineSpec:
        return LineSpec(
            line_id=self.line_id,
            name=self.name or self.line_id,
            stations=[(s.lat, s.lng) for s in self.stations],
            headway_min=self.headway_min,
            speed_kmh=self.speed_kmh,
            dwell_s=self.dwell_s,
        )


class EvaluateRequest(BaseModel):
    lines: list[LineIn] = Field(default_factory=list, max_length=12)
    params: dict[str, Any] | None = Field(
        default=None, description='Overrides, e.g. {"spending.beta.daily": 0.2}'
    )
    attribution: bool = False


class Health(BaseModel):
    ok: bool
    n_hexes: int = 0
    n_streets: int = 0
    t0_shape: list[int] = []
    synthetic: bool = False
    load_s: float = 0.0
    error: str | None = None


class ParamRange(BaseModel):
    key: str
    low: float
    central: float
    high: float
    value: float
    source: str
