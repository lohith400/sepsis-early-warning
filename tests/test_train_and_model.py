import json

import numpy as np
import pytest

from sepsis.config import TrainConfig
from sepsis.metrics import auc_fast, pick_threshold
from sepsis.model import SepsisModel
from sepsis.samples import make_patient
from sepsis.train import save_artifact, train
from synthetic import synthetic_cohort


@pytest.fixture(scope="module")
def tiny_artifact():
    cfg = TrainConfig(ga_pop=8, ga_gen=3, pso_particles=8, pso_iters=3, rf_trees=20, n_rules=4,
                      fit_pos=800, fit_neg=1600, sel_pos=400, sel_neg=800)
    return train(synthetic_cohort(n_patients=400), cfg, source="synthetic-test", log=lambda *_: None)


def test_training_pipeline_produces_a_valid_artifact(tiny_artifact, tmp_path):
    a = tiny_artifact
    assert 3 <= len(a["features"]["selected"]) <= 6
    assert len(a["anfis"]["centers"]) == 4 and 0 < a["alert"]["threshold"] < 1
    assert a["metrics"]["anfis"]["auroc"] > 0.6
    assert len(a["metrics"]["models"]) == 5 and "roc" in a["curves"]
    p = tmp_path / "m.json"; save_artifact(a, str(p)); json.loads(p.read_text())     # JSON round trip
    assert SepsisModel.load(str(p)).threshold == a["alert"]["threshold"]


def test_model_scores_a_deteriorating_patient_higher(test_model_path):
    m = SepsisModel.load(test_model_path)
    stable, age = make_patient("stable"); sick, _ = make_patient("deteriorating")
    r0, r1 = m.predict(stable, age=age), m.predict(sick, age=age)
    assert r1["summary"]["max_risk"] > r0["summary"]["max_risk"]


def test_predict_output_structure_and_activation_sums(test_model_path):
    m = SepsisModel.load(test_model_path)
    hours, age = make_patient("shock")
    r = m.predict(hours, age=age)
    assert len(r["hours"]) == len(hours)
    for h in r["hours"]:
        assert 0 <= h["risk"] <= 1 and abs(sum(h["rule_activation"]) - 1) < 0.01
        assert set(h["inputs"]) == set(m.selected)


def test_sustained_alert_needs_consecutive_hours(test_model_path):
    m = SepsisModel.load(test_model_path)
    hours, age = make_patient("shock")
    r = m.predict(hours, age=age)
    run = 0
    for h in r["hours"]:
        run = run + 1 if h["alert"] else 0
        assert h["sustained_alert"] == (run >= r["sustained_hours"])


def test_missing_values_and_empty_input(test_model_path):
    m = SepsisModel.load(test_model_path)
    r = m.predict([{}, {"HR": None}], age=None)            # nothing measured -> medians, no crash
    assert len(r["hours"]) == 2
    with pytest.raises(ValueError):
        m.predict([])


def test_single_patient_scoring_matches_batch_logic(test_model_path):
    """Scoring hour-by-hour prefixes gives the same risk as scoring the full record (no look-ahead)."""
    m = SepsisModel.load(test_model_path)
    hours, age = make_patient("deteriorating")
    full = m.predict(hours, age=age)
    prefix = m.predict(hours[:15], age=age)
    np.testing.assert_allclose([h["risk"] for h in full["hours"][:15]], [h["risk"] for h in prefix["hours"]], atol=1e-9)


def test_metrics_helpers():
    y = np.array([0, 0, 1, 1]); s = np.array([.1, .4, .35, .8])
    assert abs(auc_fast(y, s) - 0.75) < 1e-9
    assert pick_threshold(y, s, 1.0) <= 0.35
