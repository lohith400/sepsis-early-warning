"""Feature engineering shared by training and serving (identical code path).

Input: a DataFrame with a ``pid`` column (patient id, rows in time order), the raw
columns in ``config.BASE`` plus ``Age`` and ``ICULOS``. Missing columns are treated as missing values.
"""
import numpy as np
import pandas as pd

from .config import BASE, LABS, TREND_MEAN_DIFF, TREND_STD, WINDOW

FEATURE_NAMES = (
    BASE
    + ["ShockIndex", "Age", "ICULOS"]
    + [f"{c}_count" for c in LABS]
    + [n for c in TREND_MEAN_DIFF for n in (f"{c}_d6", f"{c}_m6")]
    + [f"{c}_s6" for c in TREND_STD]
)

RAW_INFO = {
    "HR": ("Heart rate", "bpm", 20, 250),
    "O2Sat": ("Oxygen saturation", "%", 50, 100),
    "Temp": ("Temperature", "°C", 30, 43),
    "SBP": ("Systolic blood pressure", "mmHg", 40, 280),
    "MAP": ("Mean arterial pressure", "mmHg", 20, 200),
    "DBP": ("Diastolic blood pressure", "mmHg", 20, 200),
    "Resp": ("Respiration rate", "breaths/min", 4, 70),
    "WBC": ("White blood cell count", "10³/µL", 0, 100),
    "Lactate": ("Lactate", "mmol/L", 0, 30),
    "Creatinine": ("Creatinine", "mg/dL", 0, 20),
    "Platelets": ("Platelet count", "10³/µL", 0, 1000),
    "BUN": ("Blood urea nitrogen", "mg/dL", 0, 250),
}


def raw_schema():
    """Fields shown in the UI for manual entry."""
    return [
        {"name": k, "label": v[0], "unit": v[1], "min": v[2], "max": v[3],
         "group": "vital" if k in ("HR", "O2Sat", "Temp", "SBP", "MAP", "DBP", "Resp") else "lab"}
        for k, v in RAW_INFO.items()
    ]


def describe_feature(name: str) -> dict:
    """Human-readable label, unit, group and explanation for any engineered feature."""
    if name in RAW_INFO:
        label, unit, *_ = RAW_INFO[name]
        grp = "Vital sign" if name in ("HR", "O2Sat", "Temp", "SBP", "MAP", "DBP", "Resp") else "Lab value"
        return {"name": name, "label": label, "unit": unit, "group": grp,
                "desc": f"Last known {label.lower()} of the patient (carried forward when not re-measured)."}
    if name == "ShockIndex":
        return {"name": name, "label": "Shock index", "unit": "HR/SBP", "group": "Derived",
                "desc": "Heart rate divided by systolic blood pressure; rises when circulation is failing."}
    if name == "Age":
        return {"name": name, "label": "Age", "unit": "years", "group": "Patient",
                "desc": "Age of the patient in years."}
    if name == "ICULOS":
        return {"name": name, "label": "Hours in ICU", "unit": "h", "group": "Time",
                "desc": "Hours since ICU admission. A time indicator, not a body measurement: sepsis becomes "
                        "more likely the longer a patient stays."}
    if name.endswith("_count"):
        b = name[:-6]
        return {"name": name, "label": f"{RAW_INFO[b][0]} tests so far", "unit": "count", "group": "Test ordering",
                "desc": f"How many times {RAW_INFO[b][0].lower()} has been measured so far. "
                        "Doctors order tests when they are worried."}
    suffix = name.rsplit("_", 1)[-1]
    b = name[: -len(suffix) - 1]
    lab, unit = RAW_INFO[b][0], RAW_INFO[b][1]
    kinds = {
        "d6": ("change over 6 h", "Difference between now and 6 hours ago (trend)."),
        "m6": ("6 h average", "Average of the last 6 hours (smooths noise)."),
        "s6": ("6 h variability", "Standard deviation of the last 6 hours (instability)."),
    }
    k_label, k_desc = kinds[suffix]
    return {"name": name, "label": f"{lab}, {k_label}", "unit": unit, "group": "Trend", "desc": k_desc}


def feature_catalog():
    return [describe_feature(n) for n in FEATURE_NAMES]


def required_inputs(selected_names):
    """Raw input columns (and patient fields) needed to compute the selected features."""
    need = set()
    for n in selected_names:
        if n == "ShockIndex":
            need.update(["HR", "SBP"])
        elif n in ("Age", "ICULOS"):
            continue
        elif n.endswith("_count"):
            need.add(n[:-6])
        elif n in RAW_INFO:
            need.add(n)
        else:
            need.add(n.rsplit("_", 1)[0])
    return [c for c in BASE if c in need]


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Engineer the 37 candidate features. Only past values of each patient are used."""
    df = df.reset_index(drop=True).copy()
    for c in BASE + ["Age", "ICULOS"]:
        if c not in df:
            df[c] = np.nan
    g = df["pid"]
    ff = df.groupby("pid")[BASE].ffill()               # last known value of the patient
    F = pd.DataFrame(index=df.index)
    F[BASE] = ff
    F["ShockIndex"] = ff["HR"] / ff["SBP"]
    F["Age"] = df["Age"]
    F["ICULOS"] = df["ICULOS"]
    for c in LABS:                                      # number of times each lab was measured so far
        F[c + "_count"] = df[c].notna().groupby(g).cumsum().astype(float)
    gb = ff.groupby(g)
    for c in TREND_MEAN_DIFF:
        F[c + "_d6"] = ff[c] - gb[c].shift(WINDOW)
        F[c + "_m6"] = gb[c].rolling(WINDOW, min_periods=1).mean().reset_index(level=0, drop=True)
    for c in TREND_STD:
        F[c + "_s6"] = gb[c].rolling(WINDOW, min_periods=2).std().reset_index(level=0, drop=True)
    return F[FEATURE_NAMES]


def prepare(F: pd.DataFrame) -> pd.DataFrame:
    """Replace infinities and set missing trend values to 0 (no change / no variability known)."""
    F = F.replace([np.inf, -np.inf], np.nan).copy()
    trend_cols = [c for c in F.columns if c.endswith("_d6") or c.endswith("_s6")]
    F[trend_cols] = F[trend_cols].fillna(0)
    return F
