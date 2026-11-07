from __future__ import annotations

import pytest

from khatt.config import ROOT, Params


@pytest.fixture(scope="session")
def params() -> Params:
    """Central values from config/params.yaml, ignoring any calibrated.json on disk."""
    return Params.load(ROOT / "config" / "params.yaml", calibrated=False)


@pytest.fixture(scope="session")
def world_dir(tmp_path_factory, params):
    """Small synthetic world (≈330 cells) shared by the whole test session."""
    from khatt.synthetic import make_synthetic_world

    return make_synthetic_world(tmp_path_factory.mktemp("world"), rings=10, seed=1, params=params)


@pytest.fixture(scope="session")
def world(world_dir):
    from khatt.data import load_world

    return load_world(world_dir)


@pytest.fixture(scope="session")
def downtown_line():
    """A north-west → south-east line through the synthetic centre (lat, lng)."""
    from khatt.engine.lines import LineSpec

    stations = [(30.045 + 0.012 * k, 31.215 + 0.010 * k) for k in range(-3, 4)]
    return LineSpec("A", "Line A", stations)


@pytest.fixture(scope="session")
def cross_line():
    from khatt.engine.lines import LineSpec

    stations = [(30.045 - 0.011 * k, 31.200 + 0.012 * k) for k in range(-3, 5)]
    return LineSpec("B", "Line B", stations)
