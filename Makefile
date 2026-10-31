# Sepsis Early Warning - common tasks. On Windows use scripts\run.bat or WSL.
PY ?= python3
export PYTHONPATH := backend

.PHONY: help install run dev train train-quick samples test lint-js docker-build docker-run clean

help:            ## show this help
	@grep -E '^[a-z-]+:.*##' $(MAKEFILE_LIST) | awk -F':.*## ' '{printf "  make %-12s %s\n", $$1, $$2}'

install:         ## install Python dependencies
	$(PY) -m pip install -r requirements-dev.txt

run:             ## start the app at http://localhost:8000 (needs artifacts/model.json)
	@test -f artifacts/model.json || echo ">>> artifacts/model.json missing: export it from Colab (docs/EXPORT_FROM_COLAB.md) or run 'make train'"
	$(PY) -m uvicorn app.main:app --app-dir backend --host 0.0.0.0 --port 8000

dev:             ## start with auto-reload
	$(PY) -m uvicorn app.main:app --app-dir backend --host 0.0.0.0 --port 8000 --reload

train:           ## train the REAL model on all 20,336 patients (downloads data, ~10-15 min)
	$(PY) -m sepsis.train --download --out artifacts/model.json

train-quick:     ## train the real model on 3,000 patients (~5 min)
	$(PY) -m sepsis.train --download --quick --out artifacts/model.json

samples:         ## regenerate the example .psv patient files
	$(PY) scripts/make_samples.py

test:            ## run the test-suite
	$(PY) -m pytest

lint-js:         ## syntax-check the frontend scripts (needs node)
	node --check frontend/js/charts.js && node --check frontend/js/app.js

docker-build:    ## build the Docker image
	docker build -t sepsis-early-warning .

docker-run:      ## run the Docker image on port 8000
	docker run --rm -p 8000:8000 -v $(PWD)/artifacts:/app/artifacts sepsis-early-warning

clean:           ## remove caches
	find . -name __pycache__ -type d -prune -exec rm -rf {} + ; rm -rf .pytest_cache
