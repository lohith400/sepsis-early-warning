# Contributing

1. Fork, create a branch, run `make install`.
2. Keep feature engineering in `backend/sepsis/features.py` only: training and serving must share it.
3. Add or update tests in `tests/` and run `make test` and `make lint-js` before opening a pull request.
4. Never commit patient data (`data/` is ignored). `artifacts/model.json` is the trained model and is tracked.
5. Describe behaviour changes in `CHANGELOG.md`.
