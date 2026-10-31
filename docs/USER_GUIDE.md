# User guide

## Starting the app
`make run` (or `scripts/run.sh`, `scripts\run.bat`, `docker compose up`) and open http://localhost:8000. The app needs `artifacts/model.json` (see [EXPORT_FROM_COLAB.md](EXPORT_FROM_COLAB.md)); without it a page explains what to do and nothing is scored.

## Patient monitor
1. **Load an example** patient (stable, gradual deterioration, rapid septic shock) or **upload** a PhysioNet `.psv` file, or type values into the table. Leave a cell empty if the value was not measured; the last known value is carried forward, exactly as in training.
2. Columns marked ● are the ones the loaded model actually uses. Other columns are accepted but do not change the score.
3. Set **Age** and **First ICU hour** if known (ICU hours is a model input in the reference model).
4. Press **Assess risk**. The tiles show the current risk, the peak, the number of alert hours, the first alert and the *sustained alert*. The line chart shows risk per hour with the alert threshold and shaded alert periods.
5. **Click a point** (or a table row) to open *Why hour N?*: bars show how strongly each fuzzy rule fired, and the sentence below describes the strongest rule as a patient profile with the risk of that profile.

## How the model works
Follows the pipeline from data to threshold with numbers from the loaded model, the GA and PSO convergence curves, the chosen features with their meaning, and a table of all fuzzy rules.

## Performance
Test-set metrics of the loaded model, ROC and precision-recall curves, baseline comparison and confusion matrix. The lower section shows the static reference results of the project report.

## Reading a risk value
Risk is the model's probability that the current hour belongs to the pre-sepsis window. At a 2 % base rate, even a good alert is more often wrong than right, so compare the risk with the **threshold**, not with 50 %.

## Troubleshooting
| Problem | Fix |
|---|---|
| Page says "Could not reach the backend" | The server is not running; start it with `make run`. |
| Page says "No trained model found" | Export `model.json` from the Colab notebook into `artifacts/` (docs/EXPORT_FROM_COLAB.md) or run `make train`, then restart. |
| Port 8000 busy | `PORT=8001 ./scripts/run.sh` or change the port in the Makefile. |
| Download fails during `make train` | Check your internet connection and re-run; existing files are skipped. |
| Uploaded file rejected | It must be one patient, pipe-separated, with columns such as HR, Resp, Temp, up to 500 rows. |
