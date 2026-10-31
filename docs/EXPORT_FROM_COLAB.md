# Using the model trained in the Colab notebook

The web app runs **only** on a real trained model stored in `artifacts/model.json`. This guide exports the model that your Colab notebook already trained. Nothing is retrained.

## Steps
1. Open the Colab notebook (`notebooks/Sepsis_GA_ANFIS_PSO_v2_colab_run.ipynb` is the same code) in the session where it ran. **All cells must have been executed and the runtime must still be connected**, because the trained parameters live in the session's memory.
2. Add a **new last cell**, paste the full contents of [`scripts/colab_export_model_cell.py`](../scripts/colab_export_model_cell.py) and run it.
3. It prints the selected features, the test AUROC and the alert threshold, then downloads `model.json`. Compare the AUROC with the report (0.7547).
4. Copy `model.json` into `artifacts/` of this repository, then `git add artifacts/model.json`, commit and push.
5. Start the app (`make run`). `GET /api/health` must show `"model_source": "colab-notebook-v2"` and `"is_demo": false`.

## If the Colab session has been closed
The parameters are lost with the session, so run the notebook again from the top (Runtime → Run all, about 10-15 minutes including the data download) and then run the export cell. The random seed is fixed (42), so the ANFIS results should be the same or very close; the Random Forest baselines can differ slightly.

## What the export cell does and does not do
- It writes out variables the notebook already holds: the PSO result (`gbest`), the rule output coefficients (`th1`), Platt scaling (`platt`), the scaling statistics (`mu`, `sd`), the medians, the chosen features, the alert threshold and all metrics, curves and histories.
- It stops with a clear message if a variable is missing or if the feature list differs from the app's 37 features.
- **Verified:** the notebook code and the export cell were run together and the exported model was loaded by the app's inference code. Across 734 hourly predictions of 25 test patients the app reproduced the notebook's probabilities with a maximum difference of 2e-7 (float32 rounding).

## Alternative: retrain locally
`make train` runs the same pipeline on PhysioNet Training Set A and writes `artifacts/model.json`. It is designed to match the notebook but numbers may differ slightly; use the Colab export if you need the exact model from the report.
