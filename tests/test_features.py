import numpy as np
import pandas as pd

from sepsis.features import FEATURE_NAMES, build_features, describe_feature, prepare, required_inputs


def _patient(n=20, pid=0, seed=0):
    r = np.random.default_rng(seed)
    return pd.DataFrame({"pid": pid, "HR": 80 + r.normal(0, 5, n), "SBP": 120 + r.normal(0, 5, n),
                         "Resp": 16 + r.normal(0, 1, n), "Temp": 37 + r.normal(0, .2, n),
                         "Lactate": np.where(np.arange(n) % 5 == 0, 1.5, np.nan),
                         "Age": 60.0, "ICULOS": np.arange(1, n + 1, dtype=float)})


def test_feature_count_and_order():
    F = build_features(_patient())
    assert list(F.columns) == FEATURE_NAMES and len(FEATURE_NAMES) == 37


def test_no_look_ahead():
    """Features of the first rows must not change when later rows change (no future information)."""
    a = _patient(); b = a.copy()
    b.loc[12:, ["HR", "Resp", "Temp", "SBP"]] += 50
    Fa, Fb = build_features(a), build_features(b)
    pd.testing.assert_frame_equal(Fa.iloc[:12], Fb.iloc[:12])


def test_patients_do_not_leak_into_each_other():
    p0, p1 = _patient(pid=0, seed=1), _patient(pid=1, seed=2)
    both = build_features(pd.concat([p0, p1], ignore_index=True))
    alone = build_features(p1)
    np.testing.assert_allclose(both.iloc[len(p0):].reset_index(drop=True).fillna(-9).values, alone.fillna(-9).values)


def test_lab_count_is_cumulative_and_forward_filled():
    F = build_features(_patient())
    assert F["Lactate_count"].is_monotonic_increasing and F["Lactate_count"].iloc[-1] == 4
    assert F["Lactate"].iloc[1] == 1.5                       # carried forward


def test_prepare_has_no_infinities_and_zero_trends():
    F = prepare(build_features(_patient(n=3)))
    assert not np.isinf(F.values.astype(float)).any()
    assert (F["HR_d6"] == 0).all()                          # fewer than 6 hours of history -> no trend


def test_describe_and_required_inputs():
    assert describe_feature("HR_m6")["label"].startswith("Heart rate")
    assert describe_feature("Lactate_count")["group"] == "Test ordering"
    assert required_inputs(["ShockIndex", "Resp_m6", "ICULOS"]) == ["HR", "SBP", "Resp"]
