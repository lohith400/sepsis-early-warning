# Development guide

## Setup
```bash
python3 -m venv .venv && source .venv/bin/activate
make install            # runtime + test dependencies
make test               # 26 tests
make dev                # auto-reload server on :8000
```
Python 3.10 or newer. Node is only needed for the optional `make lint-js`.

## Common tasks
| Task | Command |
|---|---|
| Export the Colab-trained model | see [EXPORT_FROM_COLAB.md](EXPORT_FROM_COLAB.md) |
| Train the real model | `make train` / `make train-quick` |
| Regenerate sample `.psv` files | `make samples` |
| Use another model file | `MODEL_PATH=path/to/model.json make run` |
| Train from Python | `from sepsis.train import train; art = train(df, TrainConfig(...))` |

## Training options
`python -m sepsis.train --help`: `--data-dir`, `--download`, `--quick` (3,000 patients), `--patients N`, `--out`, `--seed`. Hyper-parameters (GA population, PSO particles, rules, feature limits, target sensitivity) live in `sepsis/config.py::TrainConfig`.

## Adding a feature
1. Add its computation to `features.build_features` and its name to `FEATURE_NAMES`.
2. Add a description in `describe_feature` (and `required_inputs` if it needs a new raw input).
3. Retrain; old artifacts will no longer match the 37-feature list.
4. Extend `tests/test_features.py`. Keep the *no look-ahead* test green.

## Tests
| File | Covers |
|---|---|
| `test_features.py` | feature count/order, no look-ahead, per-patient isolation, lab counts |
| `test_anfis.py` | normalised rule strengths, least squares, class weights |
| `test_train_and_model.py` | end-to-end tiny training run, artifact round trip, prediction structure, sustained alerts, prefix consistency |
| `test_api.py` | every endpoint, validation errors, `.psv` upload, static frontend |

## Code layout rules
- `sepsis/` has no web dependencies; `app/` has no ML logic.
- Training and serving share `features.py`; never duplicate feature code.
- The frontend has no build step: edit `frontend/` and refresh.
