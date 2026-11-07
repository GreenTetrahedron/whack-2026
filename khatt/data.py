"""Load data/processed into one in-memory `World` (CLAUDE.md §5 contracts)."""

from __future__ import annotations

import json
import logging
import pickle
import time
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from khatt.config import CATEGORIES, processed_dir
from khatt.engine.catchment import WalkGraph
from khatt.geo import to_utm

log = logging.getLogger(__name__)

REQUIRED_FILES = ["hex_index.json", "hexes_r8.parquet", "streets.parquet", "T0.npy"]


class DataMissing(RuntimeError):
    pass


@dataclass
class World:
    path: Path
    hex_ids: list[str]
    hexes: pd.DataFrame
    T0: np.ndarray
    streets: pd.DataFrame
    r9: pd.DataFrame
    existing: pd.DataFrame
    microbus: gpd.GeoDataFrame
    walk: WalkGraph | None
    existing_lines: dict | None = None
    load_s: float = 0.0
    # derived arrays (filled in __post_init__)
    lat: np.ndarray = field(init=False)
    lng: np.ndarray = field(init=False)
    hex_x: np.ndarray = field(init=False)
    hex_y: np.ndarray = field(init=False)
    pop: np.ndarray = field(init=False)
    spend: np.ndarray = field(init=False)  # (N, 3) EGP/yr
    street_hex: np.ndarray = field(init=False)
    street_A: np.ndarray = field(init=False)  # (n_streets, 3)
    street_x: np.ndarray = field(init=False)
    street_y: np.ndarray = field(init=False)
    street_node: np.ndarray = field(init=False)
    street_node_m: np.ndarray = field(init=False)
    r9_pop: np.ndarray = field(init=False)
    r9_parent: np.ndarray = field(init=False)
    hex_tree: cKDTree = field(init=False)
    _pair_cache: dict = field(init=False, default_factory=dict)

    def __post_init__(self):
        h = self.hexes
        self.lat = h["lat"].to_numpy(np.float64)
        self.lng = h["lng"].to_numpy(np.float64)
        self.hex_x, self.hex_y = (np.asarray(a) for a in to_utm(self.lat, self.lng))
        self.pop = h["pop"].to_numpy(np.float64)
        self.spend = np.column_stack([h[f"spend_{c}"].to_numpy(np.float64) for c in CATEGORIES])
        self.hex_tree = cKDTree(np.column_stack([self.hex_x, self.hex_y]))

        s = self.streets
        self.street_hex = s["hex_idx"].to_numpy(np.int64)
        self.street_A = np.column_stack([s[f"A_{c}"].to_numpy(np.float64) for c in CATEGORIES])
        self.street_x, self.street_y = (np.asarray(a) for a in to_utm(s["lat"], s["lng"]))
        if self.walk is not None and self.walk.n and len(s):
            self.street_node, self.street_node_m = self.walk.snap_xy(self.street_x, self.street_y)
        else:
            self.street_node = np.full(len(s), -1, np.int64)
            self.street_node_m = np.full(len(s), np.inf)

        self.r9_pop = self.r9["pop"].to_numpy(np.float64) if len(self.r9) else np.zeros(0)
        self.r9_parent = (
            self.r9["parent_r8_idx"].to_numpy(np.int64) if len(self.r9) else np.zeros(0, np.int64)
        )

    @property
    def n(self) -> int:
        return len(self.hex_ids)

    def microbus_pairs(self, max_gap: int):
        """(route_row, a_hex, b_hex) for stop-hex pairs a before b, ≤ max_gap stops apart.

        Consecutive duplicate hexes are collapsed first so a route is a path of distinct cells.
        """
        key = int(max_gap)
        if key in self._pair_cache:
            return self._pair_cache[key]
        rows, a_list, b_list = [], [], []
        for r, stops in enumerate(self.microbus["stop_hex_idx"]):
            seq = collapse_repeats(np.asarray(stops, np.int64))
            for i in range(len(seq)):
                for j in range(i + 1, min(i + key, len(seq) - 1) + 1):
                    if seq[i] != seq[j]:
                        rows.append(r)
                        a_list.append(seq[i])
                        b_list.append(seq[j])
        out = (np.array(rows, np.int64), np.array(a_list, np.int64), np.array(b_list, np.int64))
        self._pair_cache[key] = out
        return out


def collapse_repeats(seq: np.ndarray) -> np.ndarray:
    if len(seq) == 0:
        return seq
    keep = np.concatenate([[True], seq[1:] != seq[:-1]])
    return seq[keep]


def _load_walk(path: Path, r9: pd.DataFrame) -> WalkGraph | None:
    csr = path / "walk_csr.npz"
    pkl = path / "walk_graph.pkl"
    if csr.exists() and (not pkl.exists() or csr.stat().st_mtime >= pkl.stat().st_mtime):
        return WalkGraph.load(csr)
    if not pkl.exists():
        log.warning("No walk graph in %s: catchments and pass-by spending disabled", path)
        return None
    with open(pkl, "rb") as fh:
        G = pickle.load(fh)
    node_r9_map = {}
    nodes_file = path / "walk_nodes_r9.parquet"
    if nodes_file.exists() and len(r9):
        r9_row = {h: k for k, h in enumerate(r9["h3"])}
        nodes = pd.read_parquet(nodes_file)
        node_r9_map = {
            nid: r9_row.get(cell, -1) for nid, cell in zip(nodes["node_id"], nodes["h3_r9"])
        }
    walk = WalkGraph.from_networkx(G, node_r9_map)
    try:
        walk.save(csr)
    except OSError:
        log.warning("Could not cache %s", csr)
    return walk


def load_world(path: Path | str | None = None) -> World:
    t = time.perf_counter()
    path = Path(path) if path else processed_dir()
    missing = [f for f in REQUIRED_FILES if not (path / f).exists()]
    if missing:
        raise DataMissing(f"Missing in {path}: {', '.join(missing)}. Run `make data baseline`.")

    hex_ids = json.loads((path / "hex_index.json").read_text())
    hexes = pd.read_parquet(path / "hexes_r8.parquet").sort_values("idx").reset_index(drop=True)
    if len(hexes) != len(hex_ids) or not (hexes["h3"].tolist() == hex_ids):
        raise ValueError("hexes_r8.parquet does not match hex_index.json order")
    T0 = np.load(path / "T0.npy", mmap_mode=None).astype(np.float32, copy=False)
    if T0.shape != (len(hex_ids), len(hex_ids)):
        raise ValueError(f"T0 shape {T0.shape} != ({len(hex_ids)}, {len(hex_ids)})")
    streets = pd.read_parquet(path / "streets.parquet").reset_index(drop=True)
    if "geometry" in streets.columns:
        streets = streets.drop(columns="geometry")

    r9_file = path / "r9_near.parquet"
    r9 = pd.read_parquet(r9_file) if r9_file.exists() else pd.DataFrame(
        {"h3": [], "pop": [], "parent_r8_idx": []}
    )
    ex_file = path / "existing_stations.parquet"
    existing = pd.read_parquet(ex_file) if ex_file.exists() else pd.DataFrame(
        columns=["stop_id", "line", "name", "lat", "lng", "h3_r8", "hex_idx", "catchment_pop10"]
    )
    mb_file = path / "microbus_routes.parquet"
    if mb_file.exists():
        microbus = gpd.read_parquet(mb_file).reset_index(drop=True)
    else:
        microbus = gpd.GeoDataFrame(
            {"route_id": [], "agency": [], "headway_min": [], "stop_hex_idx": []},
            geometry=[], crs="EPSG:4326",
        )
    lines_file = path / "existing_lines.geojson"
    existing_lines = json.loads(lines_file.read_text()) if lines_file.exists() else None

    walk = _load_walk(path, r9)
    world = World(
        path=path, hex_ids=hex_ids, hexes=hexes, T0=T0, streets=streets, r9=r9,
        existing=existing, microbus=microbus, walk=walk, existing_lines=existing_lines,
    )
    world.load_s = time.perf_counter() - t
    log.info("World loaded from %s in %.1f s (N=%d, streets=%d)", path, world.load_s,
             world.n, len(streets))
    return world


@lru_cache(maxsize=4)
def get_world(path: str | None = None) -> World:
    return load_world(Path(path) if path else None)
