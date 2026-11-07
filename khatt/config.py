"""Model parameters: load config/params.yaml, apply overrides, sample for Monte Carlo.

A leaf with a ``central`` key is *uncertain* ({low, central, high, source}); everything else is a
fixed setting. ``Params[...]`` takes a dotted path and returns the chosen value (central unless
overridden or sampled), so model code never sees the low/high wrappers.
"""

from __future__ import annotations

import copy
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parent.parent
CATEGORIES: tuple[str, ...] = ("daily", "comp", "eat")


def _resolve_path(env_key: str, default: Path) -> Path:
    value = os.environ.get(env_key)
    if not value:
        return default
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def params_path() -> Path:
    return _resolve_path("KHATT_PARAMS", ROOT / "config" / "params.yaml")


def data_dir() -> Path:
    return _resolve_path("KHATT_DATA_DIR", ROOT / "data")


def processed_dir() -> Path:
    return _resolve_path("KHATT_PROCESSED_DIR", data_dir() / "processed")


def scenarios_dir() -> Path:
    return _resolve_path("KHATT_SCENARIOS_DIR", ROOT / "scenarios")


def is_uncertain(node: Any) -> bool:
    return isinstance(node, dict) and "central" in node


def _flatten_uncertain(tree: dict, prefix: str = "") -> dict[str, dict]:
    out: dict[str, dict] = {}
    for key, node in tree.items():
        path = f"{prefix}{key}"
        if is_uncertain(node):
            out[path] = node
        elif isinstance(node, dict):
            out.update(_flatten_uncertain(node, path + "."))
    return out


def _resolve(tree: Any, chosen: dict[str, Any], prefix: str = "") -> Any:
    if is_uncertain(tree):
        return chosen.get(prefix.rstrip("."), tree["central"])
    if isinstance(tree, dict):
        return {k: _resolve(v, chosen, f"{prefix}{k}.") for k, v in tree.items()}
    return tree


def _flatten_overrides(overrides: dict, prefix: str = "") -> dict[str, Any]:
    """Accept {"spending.beta.daily": 0.2} and/or {"spending": {"beta": {"daily": 0.2}}}."""
    flat: dict[str, Any] = {}
    for key, value in overrides.items():
        path = f"{prefix}{key}"
        if isinstance(value, dict):
            flat.update(_flatten_overrides(value, path + "."))
        else:
            flat[path] = value
    return flat


class Params:
    """Resolved parameter set. Immutable: overrides and samples return new objects."""

    def __init__(self, tree: dict, chosen: dict[str, Any] | None = None):
        self.tree = tree
        self.chosen = dict(chosen or {})
        self.values = _resolve(tree, self.chosen)

    # ---------- construction ----------
    @classmethod
    def load(cls, path: Path | str | None = None, calibrated: Path | str | None | bool = True):
        path = Path(path) if path else params_path()
        with open(path, encoding="utf-8") as fh:
            tree = yaml.safe_load(fh)
        if calibrated is True:
            calibrated = processed_dir() / "calibrated.json"
        if calibrated and Path(calibrated).exists():
            tree = apply_calibration(tree, json.loads(Path(calibrated).read_text()))
        return cls(tree)

    def with_overrides(self, overrides: dict | None) -> Params:
        if not overrides:
            return self
        flat = _flatten_overrides(overrides)
        uncertain = _flatten_uncertain(self.tree)
        tree = self.tree
        chosen = dict(self.chosen)
        for path, value in flat.items():
            if path in uncertain:
                chosen[path] = float(value)
                continue
            node = self._node(path)  # raises KeyError for unknown paths
            if isinstance(node, dict):
                raise KeyError(f"'{path}' is a group, not a parameter")
            if tree is self.tree:
                tree = copy.deepcopy(self.tree)
            parent = tree
            *head, last = path.split(".")
            for part in head:
                parent = parent[part]
            parent[last] = type(node)(value) if isinstance(node, (int, float)) else value
        return Params(tree, chosen)

    def sample(self, rng: np.random.Generator) -> Params:
        """Draw every uncertain parameter from triangular(low, central, high)."""
        chosen = {}
        for path, node in _flatten_uncertain(self.tree).items():
            low, mode, high = float(node["low"]), float(node["central"]), float(node["high"])
            chosen[path] = mode if high <= low else float(rng.triangular(low, mode, high))
        return Params(self.tree, chosen)

    # ---------- access ----------
    def _node(self, path: str) -> Any:
        node: Any = self.tree
        for part in path.split("."):
            if not isinstance(node, dict) or part not in node:
                raise KeyError(f"Unknown parameter '{path}'")
            node = node[part]
        return node

    def __getitem__(self, path: str) -> Any:
        node: Any = self.values
        for part in path.split("."):
            if not isinstance(node, dict) or part not in node:
                raise KeyError(f"Unknown parameter '{path}'")
            node = node[part]
        return node

    def get(self, path: str, default: Any = None) -> Any:
        try:
            return self[path]
        except KeyError:
            return default

    def uncertain(self) -> dict[str, dict]:
        """Flat {path: {low, central, high, source, value}} for the /params endpoint."""
        out = {}
        for path, node in _flatten_uncertain(self.tree).items():
            out[path] = {
                "low": node["low"],
                "central": node["central"],
                "high": node["high"],
                "source": node.get("source", "assumption"),
                "value": self[path],
            }
        return out

    def key(self) -> str:
        """Stable cache key for the resolved values."""
        return json.dumps(self.values, sort_keys=True, default=str)

    # ---------- convenience ----------
    @property
    def walk_m_per_min(self) -> float:
        return float(self["walk.speed_kmh"]) * 1000.0 / 60.0

    @property
    def beta(self) -> np.ndarray:
        return np.array([self[f"spending.beta.{c}"] for c in CATEGORIES], dtype=np.float64)


def apply_calibration(tree: dict, calibrated: dict) -> dict:
    """Replace spending.beta with fitted values; low/high = ±50 % (CLAUDE.md §13)."""
    tree = copy.deepcopy(tree)
    for cat, value in (calibrated.get("beta") or {}).items():
        value = float(value)
        tree["spending"]["beta"][cat] = {
            "low": 0.5 * value,
            "central": value,
            "high": 1.5 * value,
            "source": "calibrated (Hyman's method), ±50 %",
        }
    return tree
