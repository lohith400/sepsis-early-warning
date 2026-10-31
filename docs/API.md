# API reference

Base URL: `http://localhost:8000`. Interactive docs (Swagger): `/docs`, OpenAPI JSON: `/openapi.json`. All responses are JSON.

## `GET /api/health`
```json
{"status":"ok","version":"2.0.0","model":"GA + ANFIS + PSO ...","model_source":"colab-notebook-v2","is_demo":false,"model_file":"model.json"}
```
When no model is installed: `{"status":"no_model", "detail": "...instructions..."}`, and every model route answers `503`.

## `GET /api/model`
Everything shown on the *Model* and *Performance* pages: `name`, `source`, `is_demo`, `config`, `dataset`, `selected_features` (with labels, units, descriptions, ranges), `required_inputs`, `rules` (centres in real units, plain-language text, `centre_risk`), `alert` (threshold, target sensitivity), `metrics` (ours, baselines, confusion matrix), `curves` (ROC, PR), `history` (GA, PSO), `timing`.

## `GET /api/features` · `GET /api/schema`
The 37 candidate features with a `selected` flag, and the 12 raw input fields (label, unit, plausible range, `used` by the loaded model).

## `GET /api/samples` · `GET /api/samples/{id}`
Three **synthetic** example patients (`stable`, `deteriorating`, `shock`). The detail route returns `hours`, `age` and illustrative `labels`.

## `POST /api/predict`
Request:
```json
{
  "age": 71,
  "first_icu_hour": 1,
  "hours": [
    {"HR": 78, "Temp": 36.8, "Resp": 15, "SBP": 121, "WBC": 7.4},
    {"HR": 81, "Temp": 36.9, "Resp": 16, "SBP": 119}
  ]
}
```
- `hours`: 1-500 records in time order. Every measurement is optional (`null` or absent = not measured; the last known value is carried forward).
- Allowed keys: `HR, O2Sat, Temp, SBP, MAP, DBP, Resp, WBC, Lactate, Creatinine, Platelets, BUN`, optionally `ICULOS` and `Age`. Unknown keys are ignored.

Response (abridged):
```json
{
  "model": {"name": "...", "source": "colab-notebook-v2", "is_demo": false},
  "threshold": 0.0133,
  "sustained_hours": 3,
  "summary": {"n_hours": 2, "last_risk": 0.011, "max_risk": 0.012, "hours_in_alert": 0,
              "first_alert_hour": null, "first_sustained_alert_hour": null, "last_status": "normal"},
  "hours": [
    {"hour": 1, "iculos": 1.0, "risk": 0.012, "alert": false, "sustained_alert": false,
     "top_rule": 4, "rule_activation": [0.0, 0.1, 0.0, 0.8, 0.1, 0.0, 0.0, 0.0],
     "inputs": {"HR": 78.0, "BUN": 14.0, "ICULOS": 1.0, "Resp_m6": 15.0, "Temp_s6": 0.0}}
  ]
}
```
Errors: `422` for invalid values (wrong type, empty `hours`, age outside 0-120), `400` for domain errors.

## `POST /api/predict/psv`
`multipart/form-data` with a `file` field holding **one patient** in PhysioNet `.psv` format (pipe-separated, `NaN` for missing, up to 500 rows, 2 MB). Same response as `/api/predict` plus `filename`, `age`, `hours_raw` and `true_labels` (when a `SepsisLabel` column is present).
```bash
curl -F "file=@samples/patient_septic_shock.psv" localhost:8000/api/predict/psv
```

## `GET /api/reference`
Static results of the Colab run reported in the AAT report (independent of the loaded model).
