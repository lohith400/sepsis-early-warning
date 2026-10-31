"""TEST-ONLY helper: a small artificial cohort so the unit tests can exercise the pipeline quickly.

Nothing here is shipped as a model and the application never uses it. Real results come only from the
Colab-trained model (``artifacts/model.json``) or from ``make train`` on PhysioNet data.
"""
import numpy as np
import pandas as pd

from sepsis.config import TrainConfig
from sepsis.features import RAW_INFO
from sepsis.train import save_artifact, train


def synthetic_cohort(n_patients: int = 1500, sepsis_fraction: float = 0.2, seed: int = 11) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    frames = []
    for pid in range(n_patients):
        n = int(rng.integers(24, 80))
        sep = rng.random() < sepsis_fraction
        onset = int(rng.integers(14, n)) if sep else None
        t = np.arange(n)
        drift = np.zeros(n)
        if sep:
            drift = np.clip((t - (onset - 12)) / 12.0, 0, 1) * rng.uniform(0.4, 1.2)
        base = {"HR": rng.normal(80, 8), "Resp": rng.normal(16, 2), "Temp": rng.normal(36.9, 0.3),
                "SBP": rng.normal(122, 12), "MAP": rng.normal(85, 8), "DBP": rng.normal(68, 8),
                "O2Sat": rng.normal(97, 1.2)}
        d = pd.DataFrame({
            "HR": base["HR"] + rng.normal(0, 6, n) + 20 * drift,
            "Resp": base["Resp"] + rng.normal(0, 2.0, n) + 6 * drift,
            "Temp": base["Temp"] + rng.normal(0, 0.35, n) + 0.9 * drift,
            "SBP": base["SBP"] + rng.normal(0, 8, n) - 20 * drift,
            "MAP": base["MAP"] + rng.normal(0, 5, n) - 15 * drift,
            "DBP": base["DBP"] + rng.normal(0, 5, n) - 9 * drift,
            "O2Sat": base["O2Sat"] + rng.normal(0, 1.2, n) - 3 * drift,
        })
        p_lab = 0.15
        lab_mask = rng.random(n) < p_lab
        for c, (m, s, k) in {"WBC": (8, 2.5, 4), "Lactate": (1.2, 0.4, 1.4), "Creatinine": (1.0, 0.25, 0.5),
                             "Platelets": (230, 50, -50), "BUN": (16, 5, 8)}.items():
            vals = rng.normal(m, s, n) + k * drift
            d[c] = np.where(lab_mask, vals, np.nan)
        # random missing vitals
        for c in ("HR", "Resp", "Temp", "SBP", "MAP", "DBP", "O2Sat"):
            d.loc[rng.random(n) < 0.08, c] = np.nan
        d["Age"] = float(rng.integers(30, 92))
        d["ICULOS"] = (t + 1).astype(float)
        d["SepsisLabel"] = ((t + 1) >= (onset - 6)).astype(int) if sep else 0
        d["pid"] = pid
        frames.append(d)
    df = pd.concat(frames, ignore_index=True)
    for k, (_, _, lo, hi) in RAW_INFO.items():
        df[k] = df[k].clip(lo, hi)
    return df


def test_config() -> TrainConfig:
    return TrainConfig(ga_pop=12, ga_gen=5, pso_particles=12, pso_iters=6, rf_trees=30, n_rules=5,
                       fit_pos=1500, fit_neg=3000, sel_pos=700, sel_neg=1400)


def build_test_model(out: str, log=lambda *_: None):
    art = train(synthetic_cohort(n_patients=700), test_config(), source="synthetic-test",
                name="SYNTHETIC TEST MODEL (unit tests only)", log=log)
    save_artifact(art, out)
    return art
