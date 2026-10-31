"""Three illustrative SYNTHETIC patients (not real data) used by the UI and the tests."""
import numpy as np

from .config import BASE

SAMPLES = {
    "stable": {"title": "Stable patient", "age": 64, "hours": 30,
               "description": "Vital signs stay in a normal range for the whole stay."},
    "deteriorating": {"title": "Gradual deterioration", "age": 71, "hours": 30,
                      "description": "Normal for 12 hours, then heart rate, respiration and temperature creep up while blood pressure falls."},
    "shock": {"title": "Rapid septic shock", "age": 58, "hours": 30,
              "description": "Stable until hour 18, then a fast collapse with high lactate."},
}
ONSET = {"stable": None, "deteriorating": 28, "shock": 24}      # illustrative sepsis onset hour


def make_patient(kind: str, seed: int = 7):
    """Return (list of hourly dicts with raw measurements, age)."""
    rng = np.random.default_rng({"stable": 1, "deteriorating": 2, "shock": 3}[kind] + seed)
    n = SAMPLES[kind]["hours"]
    start = {"stable": n + 1, "deteriorating": 12, "shock": 18}[kind]
    rate = {"stable": 0.0, "deteriorating": 1.0, "shock": 2.4}[kind]
    rows = []
    for t in range(n):
        d = max(0, t - start) * rate
        d = min(d, 14 if kind == "deteriorating" else 12)
        row = {
            "HR": 78 + rng.normal(0, 3) + 2.4 * d,
            "O2Sat": 97.5 + rng.normal(0, 0.8) - 0.55 * d,
            "Temp": 36.8 + rng.normal(0, 0.15) + 0.11 * d,
            "SBP": 122 + rng.normal(0, 5) - 2.6 * d,
            "MAP": 86 + rng.normal(0, 3) - 2.2 * d,
            "DBP": 70 + rng.normal(0, 3) - 1.5 * d,
            "Resp": 15 + rng.normal(0, 1) + 1.0 * d,
        }
        lab_hour = t % 6 == 0 and (t >= start or t % 12 == 0)
        if lab_hour:
            row.update({"WBC": 7.5 + 0.8 * d + rng.normal(0, 0.4), "Lactate": 1.0 + 0.3 * d + rng.normal(0, 0.1),
                        "Creatinine": 0.9 + 0.06 * d, "Platelets": 230 - 6 * d, "BUN": 14 + 1.4 * d})
        rows.append({k: (round(float(row[k]), 1) if k in row else None) for k in BASE})
    return rows, SAMPLES[kind]["age"]


def sample_list():
    return [{"id": k, "title": v["title"], "description": v["description"], "age": v["age"], "hours": v["hours"]}
            for k, v in SAMPLES.items()]


def sample_labels(kind: str):
    """Illustrative SepsisLabel column (1 from 6 h before onset), PhysioNet convention."""
    n, onset = SAMPLES[kind]["hours"], ONSET[kind]
    return [int(onset is not None and (t + 1) >= onset - 6) for t in range(n)]


PSV_COLUMNS = ("HR,O2Sat,Temp,SBP,MAP,DBP,Resp,EtCO2,BaseExcess,HCO3,FiO2,pH,PaCO2,SaO2,AST,BUN,Alkalinephos,"
               "Calcium,Chloride,Creatinine,Bilirubin_direct,Glucose,Lactate,Magnesium,Phosphate,Potassium,"
               "Bilirubin_total,TroponinI,Hct,Hgb,PTT,WBC,Fibrinogen,Platelets,Age,Gender,Unit1,Unit2,"
               "HospAdmTime,ICULOS,SepsisLabel").split(",")


def to_psv(kind: str) -> str:
    """Render a sample as a PhysioNet-format .psv text (pipe separated, NaN for missing)."""
    rows, age = make_patient(kind)
    labels = sample_labels(kind)
    lines = ["|".join(PSV_COLUMNS)]
    for i, r in enumerate(rows):
        vals = []
        for c in PSV_COLUMNS:
            if c == "Age":
                vals.append(str(age))
            elif c == "ICULOS":
                vals.append(str(i + 1))
            elif c == "SepsisLabel":
                vals.append(str(labels[i]))
            elif c in r and r[c] is not None:
                vals.append(str(r[c]))
            else:
                vals.append("NaN")
        lines.append("|".join(vals))
    return "\n".join(lines) + "\n"
