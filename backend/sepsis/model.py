"""Inference: load a JSON artifact, score a patient's hourly record, explain with fuzzy rules."""
import json

import numpy as np
import pandas as pd

from .anfis import phi
from .config import BASE, SUSTAINED_HOURS
from .features import (FEATURE_NAMES, build_features, describe_feature, feature_catalog, prepare,
                       raw_schema, required_inputs)


class SepsisModel:
    def __init__(self, art: dict):
        self.art = art
        self.selected = art["features"]["selected"]
        self.sel_idx = art["features"]["selected_idx"]
        pre = art["preprocessing"]
        self.medians = pd.Series(pre["medians"])
        self.mean = pd.Series(pre["mean"])
        self.std = pd.Series(pre["std"])
        self.clip = pre["clip"]
        a = art["anfis"]
        self.C = np.array(a["centers"], dtype=float)
        self.S = np.array(a["sigmas"], dtype=float)
        self.theta = np.array(a["theta"], dtype=float)
        self.coef = art["calibration"]["coef"]
        self.intercept = art["calibration"]["intercept"]
        self.threshold = art["alert"]["threshold"]
        self.is_demo = str(art.get("source", "")).startswith(("synthetic", "demo"))   # test-only models
        self.rules = self._describe_rules()

    @classmethod
    def load(cls, path: str) -> "SepsisModel":
        with open(path, "r", encoding="utf-8") as fh:
            return cls(json.load(fh))

    # ---------- helpers ----------
    def _risk(self, raw_score):
        return 1.0 / (1.0 + np.exp(-(self.coef * raw_score + self.intercept)))

    def _describe_rules(self):
        pre = self.art["preprocessing"]
        rules = []
        for r in range(self.C.shape[0]):
            conds = []
            for j, name in enumerate(self.selected):
                z = self.C[r, j]
                raw = float(np.clip(z * self.std[name] + self.mean[name], pre["range_min"][name], pre["range_max"][name]))
                info = describe_feature(name)
                if abs(z) > 0.5:
                    conds.append({"feature": name, "label": info["label"], "unit": info["unit"],
                                  "level": "high" if z > 0 else "low", "value": round(raw, 2)})
            x = np.concatenate([[1.0], self.C[r]])
            centre_risk = float(self._risk(float(x @ self.theta[r * (len(x)): (r + 1) * len(x)])))
            text = " AND ".join(f"{c['label']} is {c['level']} (about {c['value']:g} {c['unit']})" for c in conds)
            rules.append({
                "id": r + 1,
                "centres": {n: round(float(np.clip(self.C[r, j] * self.std[n] + self.mean[n],
                                                   pre["range_min"][n], pre["range_max"][n])), 2)
                            for j, n in enumerate(self.selected)},
                "conditions": conds,
                "text": text or "all selected inputs are close to typical values",
                "centre_risk": centre_risk,
            })
        return rules

    # ---------- public API ----------
    def required_inputs(self):
        return required_inputs(self.selected)

    def info(self) -> dict:
        a = self.art
        sel = []
        for n in self.selected:
            d = describe_feature(n)
            d["min"] = a["preprocessing"]["range_min"][n]
            d["max"] = a["preprocessing"]["range_max"][n]
            sel.append(d)
        return {
            "name": a["name"], "version": a["version"], "source": a["source"], "created": a["created"],
            "is_demo": self.is_demo, "config": a["config"], "dataset": a["dataset"],
            "selected_features": sel, "required_inputs": self.required_inputs(), "rules": self.rules,
            "alert": a["alert"], "metrics": a["metrics"], "curves": a["curves"],
            "history": a["history"], "timing": a["timing"],
        }

    def feature_catalog(self):
        sel = set(self.selected)
        return [dict(c, selected=c["name"] in sel) for c in feature_catalog()]

    def raw_schema(self):
        need = set(self.required_inputs())
        return [dict(f, used=f["name"] in need) for f in raw_schema()]

    def predict(self, hours: list, age: float | None = None, first_icu_hour: int = 1) -> dict:
        """Score every hour of one patient. ``hours`` = list of dicts with raw measurements (None = not measured)."""
        n = len(hours)
        if n == 0:
            raise ValueError("At least one hourly record is required.")
        df = pd.DataFrame(hours)
        for c in BASE:
            if c not in df:
                df[c] = np.nan
        cols = BASE + [c for c in ("Age", "ICULOS") if c in df]
        df[cols] = df[cols].apply(pd.to_numeric, errors="coerce")
        df["pid"] = 0
        default_icu = first_icu_hour + np.arange(n, dtype=float)
        df["ICULOS"] = df["ICULOS"].fillna(pd.Series(default_icu)) if "ICULOS" in df else default_icu
        if "Age" not in df:
            df["Age"] = np.nan if age is None else float(age)
        elif age is not None:
            df["Age"] = df["Age"].fillna(float(age))

        F = prepare(build_features(df)).fillna(self.medians)
        X = ((F - self.mean) / self.std).clip(-self.clip, self.clip)
        Xs = X.values[:, self.sel_idx].astype(float)
        w, P = phi(Xs, self.C, self.S)
        risk = self._risk(P @ self.theta)
        alert = risk >= self.threshold

        run, sustained = 0, []
        for a in alert:
            run = run + 1 if a else 0
            sustained.append(run >= SUSTAINED_HOURS)

        inputs = F[self.selected]
        rows = []
        for i in range(n):
            top = int(np.argmax(w[i]))
            rows.append({
                "hour": i + 1,
                "iculos": float(df["ICULOS"].iloc[i]),
                "risk": float(risk[i]),
                "alert": bool(alert[i]),
                "sustained_alert": bool(sustained[i]),
                "top_rule": top + 1,
                "rule_activation": [round(float(v), 4) for v in w[i]],
                "inputs": {k: round(float(inputs[k].iloc[i]), 3) for k in self.selected},
            })
        first_alert = next((r["hour"] for r in rows if r["alert"]), None)
        first_sustained = next((r["hour"] for r in rows if r["sustained_alert"]), None)
        return {
            "model": {"name": self.art["name"], "source": self.art["source"], "is_demo": self.is_demo},
            "threshold": self.threshold,
            "sustained_hours": SUSTAINED_HOURS,
            "hours": rows,
            "summary": {
                "n_hours": n,
                "last_risk": rows[-1]["risk"],
                "max_risk": float(risk.max()),
                "hours_in_alert": int(alert.sum()),
                "first_alert_hour": first_alert,
                "first_sustained_alert_hour": first_sustained,
                "last_status": "alert" if alert[-1] else "normal",
            },
        }
