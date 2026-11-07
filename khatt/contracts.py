"""Processed-data contracts (CLAUDE.md §5) as code, used by tests and at the end of pipeline steps."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

STR = "str"
CONTRACTS: dict[str, dict[str, str]] = {
    "hexes_r8.parquet": {
        "h3": STR, "idx": "int32", "lat": "float64", "lng": "float64", "governorate": STR,
        "pop": "float32", "wealth": "float32", "spend_daily": "float32", "spend_comp": "float32",
        "spend_eat": "float32",
    },
    "streets.parquet": {
        "street_id": "int32", "name": STR, "lat": "float64", "lng": "float64", "h3_r8": STR,
        "h3_r9": STR, "hex_idx": "int32", "n_daily": "int32", "n_comp": "int32", "n_eat": "int32",
        "n_unbranded": "int32", "footprint_m2": "float32", "A_daily": "float32",
        "A_comp": "float32", "A_eat": "float32",
    },
    "existing_stations.parquet": {
        "stop_id": STR, "line": STR, "name": STR, "lat": "float64", "lng": "float64",
        "h3_r8": STR, "hex_idx": "int32", "catchment_pop10": "float32",
    },
    "microbus_routes.parquet": {
        "route_id": STR, "agency": STR, "headway_min": "float64", "stop_hex_idx": "list",
    },
    "r9_near.parquet": {"h3": STR, "pop": "float32", "parent_r8_idx": "int32"},
    "walk_nodes_r9.parquet": {"node_id": "int", "h3_r9": STR},
}


def _dtype_ok(series: pd.Series, expected: str) -> bool:
    if expected == STR:
        return pd.api.types.is_string_dtype(series) or series.dtype == object
    if expected == "list":
        return series.dtype == object and all(hasattr(v, "__len__") for v in series.head(50))
    if expected == "int":
        return pd.api.types.is_integer_dtype(series)
    return str(series.dtype) == expected


def check_table(path: Path, contract: dict[str, str]) -> list[str]:
    df = pd.read_parquet(path)
    problems = []
    for col, dtype in contract.items():
        if col not in df.columns:
            problems.append(f"{path.name}: missing column '{col}'")
        elif not _dtype_ok(df[col], dtype):
            problems.append(f"{path.name}: '{col}' is {df[col].dtype}, expected {dtype}")
    return problems


def check_processed(path: Path, intracell_min: float | None = None) -> list[str]:
    """Return a list of contract violations (empty = OK). Optional files are checked if present."""
    path = Path(path)
    problems: list[str] = []
    index_file = path / "hex_index.json"
    if not index_file.exists():
        return [f"missing {index_file}"]
    hex_ids = json.loads(index_file.read_text())
    if not all(isinstance(h, str) for h in hex_ids):
        problems.append("hex_index.json must be a list of strings")
    n = len(hex_ids)

    for name, contract in CONTRACTS.items():
        f = path / name
        if f.exists():
            problems += check_table(f, contract)
        elif name in ("hexes_r8.parquet", "streets.parquet"):
            problems.append(f"missing {name}")

    hex_file = path / "hexes_r8.parquet"
    if hex_file.exists():
        hexes = pd.read_parquet(hex_file).sort_values("idx")
        if hexes["h3"].tolist() != hex_ids:
            problems.append("hexes_r8.h3 order != hex_index.json")
        if (hexes["pop"] < 0).any():
            problems.append("negative population")
        for c in ("spend_daily", "spend_comp", "spend_eat", "pop", "wealth"):
            if c in hexes and not np.isfinite(hexes[c]).all():
                problems.append(f"hexes_r8.{c} has NaN/inf")

    st_file = path / "streets.parquet"
    if st_file.exists():
        st = pd.read_parquet(st_file)
        if "hex_idx" in st and ((st["hex_idx"] < 0) | (st["hex_idx"] >= n)).any():
            problems.append("streets.hex_idx out of range")
        if "street_id" in st and st["street_id"].duplicated().any():
            problems.append("duplicate street_id")

    t0_file = path / "T0.npy"
    if t0_file.exists():
        T0 = np.load(t0_file, mmap_mode="r")
        if T0.dtype != np.float32:
            problems.append(f"T0 dtype {T0.dtype}, expected float32")
        if T0.shape != (n, n):
            problems.append(f"T0 shape {T0.shape}, expected ({n}, {n})")
        elif not np.isfinite(T0).all():
            problems.append("T0 has NaN/inf (use 999.0 for unreachable)")
        elif (T0 < 0).any():
            problems.append("T0 has negative times")
        elif intracell_min is not None and not np.allclose(np.diag(T0), intracell_min):
            problems.append("T0 diagonal != access.intracell_min")
    else:
        problems.append("missing T0.npy")

    for name in ("existing_stations.parquet", "microbus_routes.parquet"):
        f = path / name
        if f.exists():
            df = pd.read_parquet(f)
            col = "hex_idx" if name.startswith("existing") else None
            if col and len(df) and ((df[col] < 0) | (df[col] >= n)).any():
                problems.append(f"{name}: hex_idx out of range")
            if name.startswith("microbus") and len(df):
                flat = np.concatenate([np.asarray(v) for v in df["stop_hex_idx"]])
                if len(flat) and (flat.min() < 0 or flat.max() >= n):
                    problems.append("microbus stop_hex_idx out of range")
    return problems
