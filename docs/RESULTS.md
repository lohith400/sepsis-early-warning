# Results

Reference run reported in the AAT report (Google Colab, PhysioNet 2019 Training Set A, all 20,336 patients, seed 42). The notebook with its outputs (the run that produced `artifacts/model.json`) is in [`notebooks/`](../notebooks/Sepsis_GA_ANFIS_PSO_v2_colab_run.ipynb); the machine-readable numbers are in [`backend/app/data/reference_results.json`](../backend/app/data/reference_results.json).

## Setup
Test set: 159,397 hourly rows from unseen patients, sepsis rate 2.14 %. Alert threshold 1.33 % (chosen on validation patients for ≈ 80 % sensitivity). Selected features: HR, BUN, ICULOS, Resp_m6, Temp_s6. GA 14.6 s, PSO 98.1 s, one prediction 1.48 µs (timings vary slightly from run to run; all other numbers are fixed by the seed).

## Comparison

| Model | AUROC | AUPRC | Sens. | Spec. | Prec. | Acc. | FA / TA |
|---|---:|---:|---:|---:|---:|---:|---:|
| **GA + ANFIS + PSO** | **0.7547** | 0.0860 | 83.7 % | 45.8 % | 3.3 % | 46.6 % | 29.6 |
| Logistic Regression (GA features) | 0.7270 | 0.0835 | 84.3 % | 37.4 % | 2.9 % | 38.4 % | 33.9 |
| Random Forest (GA features) | 0.7642 | 0.0994 | 85.0 % | 45.6 % | 3.3 % | 46.5 % | 29.2 |
| Logistic Regression (all features) | 0.7483 | 0.0929 | 82.6 % | 45.3 % | 3.2 % | 46.1 % | 30.3 |
| Random Forest (all features) | 0.8100 | 0.1107 | 82.5 % | 61.3 % | 4.5 % | 61.8 % | 21.4 |

FA / TA = false alarms per true alert. AUPRC of a random model = 0.021.

Confusion matrix of the proposed model: TN 71,461 · FP 84,522 · FN 556 · TP 2,858.

## Stage-by-stage
| Item | Value |
|---|---|
| Quick logistic regression AUROC, 5 GA features vs all 37 | 0.7213 vs 0.7287 |
| PSO loss, k-means start → after 60 rounds | 0.2848 → 0.2435 |
| Validation AUROC, k-means rules only → after PSO | 0.7164 → 0.7283 |

## Interpretation
- AUROC rose from 0.61 (first version) to 0.755, and the model beats logistic regression on the same and on all features. Random Forest on the same 5 features is slightly better (0.764) but cannot explain an alert with rules; Random Forest on all features is the best model (0.810).
- Five features carry almost all the signal that 37 do (0.7213 vs 0.7287), which supports the GA step.
- PSO improved the model over the k-means rules alone, and its loss was still falling after 60 rounds, so more rounds should help a little.

## Limitations (honest list)
1. **Low precision.** About 3 %: roughly 30 false alarms for every true alert, and with hourly checking 98.8 % of non-sepsis patients receive at least one alert. The model is a risk-ranking aid, not a diagnosis.
2. **ICU hours was selected.** `ICULOS` is a time proxy (risk grows with length of stay), not a physiological measurement. In the example patient flagged in the report, the alert was driven mainly by long stay and BUN.
3. **One run, one dataset.** Single seed, Training Set A only, no external validation. Validation AUROC (0.728) and test AUROC (0.755) differ because they are different groups of patients.
4. **No alert-persistence rule was evaluated** (the UI's sustained-alert flag is an untested extension).

## Reproducing
`make train` regenerates a model with the same pipeline. Small differences are normal (library versions, parallel Random Forest, sampling); the data split and the seed are fixed.
