import json

import numpy as np
import pandas as pd
import pytest

from khatt.config import processed_dir
from khatt.contracts import check_processed
from khatt.data import collapse_repeats


def test_synthetic_world_meets_contracts(world_dir, params):
    assert check_processed(world_dir, params["access.intracell_min"]) == []


def test_world_loads(world):
    assert world.n == len(json.loads((world.path / "hex_index.json").read_text()))
    assert world.T0.dtype == np.float32
    assert np.isfinite(world.T0).all()
    assert world.walk is not None and world.walk.n > 0
    assert (world.street_node >= 0).all()
    assert len(world.microbus) > 0 and len(world.existing) > 0
    assert world.existing["catchment_pop10"].sum() > 0


def test_walk_csr_cache_round_trip(world_dir):
    from khatt.data import load_world

    assert (world_dir / "walk_csr.npz").exists(), "first load caches the CSR graph"
    w2 = load_world(world_dir)
    assert w2.walk.n > 0


def test_population_total(world_dir):
    hexes = pd.read_parquet(world_dir / "hexes_r8.parquet")
    r9 = pd.read_parquet(world_dir / "r9_near.parquet")
    assert np.isclose(r9["pop"].sum(), hexes["pop"].sum(), rtol=1e-3)


def test_microbus_pairs(world, params):
    rows, a, b = world.microbus_pairs(int(params["informal.max_pair_gap"]))
    assert len(rows) == len(a) == len(b) > 0
    assert (a != b).all()
    assert collapse_repeats(np.array([1, 1, 2, 2, 2, 3, 1])).tolist() == [1, 2, 3, 1]


def test_missing_data_raises(tmp_path):
    from khatt.data import DataMissing, load_world

    with pytest.raises(DataMissing):
        load_world(tmp_path)


@pytest.mark.skipif(not (processed_dir() / "T0.npy").exists(), reason="real data not built yet")
def test_real_processed_data_meets_contracts(params):
    assert check_processed(processed_dir()) == []
    T0 = np.load(processed_dir() / "T0.npy", mmap_mode="r")
    assert not np.isnan(T0).any()
