# artifacts/

Put the **trained model** here as `model.json`. The app loads it automatically; there is deliberately **no fallback model**.

Two ways to get it:

1. **Export from the Colab notebook** where the model was trained: run `scripts/colab_export_model_cell.py` as the last cell, download `model.json`, copy it here. See [`docs/EXPORT_FROM_COLAB.md`](../docs/EXPORT_FROM_COLAB.md).
2. **Retrain locally:** `make train` (downloads PhysioNet Training Set A, about 10-15 minutes).

`model.json` is a small, human-readable JSON file (no pickle) and is meant to be committed to Git so Codespaces and Docker work out of the box.
