import json

import numpy as np
import pytest

from khatt.config import CATEGORIES, ROOT, Params, is_uncertain


def test_every_uncertain_param_has_range_and_source(params):
    unc = params.uncertain()
    assert len(unc) > 20
    for path, node in unc.items():
        for k in ("low", "central", "high"):
            # PyYAML reads 4.0e6 as a string; write 4.0e+6 or 4000000
            assert isinstance(node[k], (int, float)), f"{path}.{k} is not numeric"
        assert node["low"] <= node["central"] <= node["high"], path
        assert node["source"], path


def test_central_values(params):
    assert params["spending.alpha"] == 1.0
    assert params["routing.irregularity_factor.metro"] == 1.0
    assert params["area.bbox"] == [30.85, 29.80, 31.85, 30.35]
    assert np.isclose(params.walk_m_per_min, 75.0)
    assert params.beta.shape == (len(CATEGORIES),)
    with pytest.raises(KeyError):
        params["nope.nothing"]


def test_overrides_flat_and_nested(params):
    p = params.with_overrides({"spending.beta.daily": 0.3, "rents": {"elasticity": 0.5}})
    assert p["spending.beta.daily"] == 0.3
    assert p["rents.elasticity"] == 0.5
    assert params["rents.elasticity"] == 0.3, "original unchanged"
    p2 = params.with_overrides({"footfall.catchment_min": 12})
    assert p2["footfall.catchment_min"] == 12
    assert params["footfall.catchment_min"] == 10
    with pytest.raises(KeyError):
        params.with_overrides({"spending.nope": 1})
    with pytest.raises(KeyError):
        params.with_overrides({"spending.beta": 1})


def test_sample_within_bounds_and_reproducible(params):
    a = params.sample(np.random.default_rng(42))
    b = params.sample(np.random.default_rng(42))
    assert a.key() == b.key()
    for path, node in params.uncertain().items():
        assert node["low"] <= a[path] <= node["high"], path
    assert a.key() != params.key()


def test_calibration_file_overrides_beta(tmp_path):
    cal = tmp_path / "calibrated.json"
    cal.write_text(json.dumps({"beta": {"daily": 0.2, "comp": 0.04, "eat": 0.1}}))
    p = Params.load(ROOT / "config" / "params.yaml", calibrated=cal)
    assert p["spending.beta.daily"] == 0.2
    node = p.uncertain()["spending.beta.comp"]
    assert np.isclose(node["low"], 0.02) and np.isclose(node["high"], 0.06)
    assert is_uncertain(p.tree["spending"]["beta"]["eat"])
