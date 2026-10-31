"""Shared constants and the training configuration."""
from dataclasses import dataclass

VITALS = ["HR", "O2Sat", "Temp", "SBP", "MAP", "DBP", "Resp"]
LABS = ["WBC", "Lactate", "Creatinine", "Platelets", "BUN"]
BASE = VITALS + LABS                      # raw measurements the model understands
TREND_MEAN_DIFF = ["HR", "O2Sat", "Temp", "SBP", "MAP", "Resp"]   # 6 h change + 6 h mean
TREND_STD = ["HR", "Temp", "SBP", "Resp", "MAP"]                  # 6 h variability
WINDOW = 6                                # hours used for trend features
CLIP = 5.0                                # standardised values are clipped to +/- CLIP
SUSTAINED_HOURS = 3                       # consecutive alert hours for a "sustained" alert


@dataclass
class TrainConfig:
    seed: int = 42
    target_sens: float = 0.80      # validation sensitivity used to choose the alert threshold
    k_min: int = 3                 # GA: minimum features
    k_max: int = 6                 # GA: maximum features
    n_rules: int = 8               # fuzzy rules (k-means clusters)
    ga_pop: int = 30
    ga_gen: int = 25
    pso_particles: int = 40
    pso_iters: int = 60
    rf_trees: int = 200
    fit_pos: int = 4000            # sepsis rows used to learn parameters
    fit_neg: int = 8000
    sel_pos: int = 2500            # sepsis rows used to judge GA / PSO candidates
    sel_neg: int = 5000
    n_patients: int | None = None  # limit number of patients (quick test)
