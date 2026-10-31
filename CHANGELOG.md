# Changelog

## 2.0.0 (2026-10-31)
- Rebuilt the pipeline on all 20,336 PhysioNet 2019 Set A patients with a **patient-level** train / validation / test split.
- 37 candidate features including 6-hour trends and lab-ordering counts.
- GA wrapper fitness (AUROC), ANFIS rules from k-means with least-squares consequents, PSO with warm start and decaying inertia.
- Alert threshold chosen on validation data; Logistic Regression and Random Forest baselines.
- FastAPI backend and light web frontend (patient monitor, model explanation, performance).
- The app runs only on the real trained model (`artifacts/model.json`, exported from the Colab notebook with `scripts/colab_export_model_cell.py` or trained with `make train`); no stand-in model.
- Tests (26), Docker, CI, dev container.
- Shipped the trained `artifacts/model.json` (exported from the notebook in `notebooks/`). Timings in the report, docs and reference data now match that run (GA 14.6 s, PSO 98.1 s, 1.48 µs per record); all other numbers are fixed by the seed.

## 1.0.0
- First version: 3 features, 4 rules, 2,000 patients, balanced test split (AUROC 0.61).
