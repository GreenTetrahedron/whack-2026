# Khatt — see CLAUDE.md §16. Variables from .env are exported to every recipe.
-include .env
export

PY      := uv run --frozen python
PYTEST  := uv run --frozen pytest
SYNTH   := data/synthetic

.PHONY: setup data baseline baseline-fallback calibrate api api-synthetic web scenarios test \
        test-fast lint synthetic mock backtest

setup:            ## python env + web deps
	uv sync --frozen
	@if [ -f web/package.json ]; then cd web && npm install; else echo "web/ not created yet"; fi

data:             ## pipeline 00–07, 09, 10
	$(PY) pipeline/00_download.py
	$(PY) pipeline/01_grid.py
	$(PY) pipeline/02_population.py
	$(PY) pipeline/03_spending.py
	$(PY) pipeline/04_shops.py
	$(PY) pipeline/05_streets.py
	$(PY) pipeline/06_walk_graph.py
	$(PY) pipeline/07_gtfs_clean.py
	$(PY) pipeline/09_existing_stations.py
	$(PY) pipeline/10_microbus_routes.py

baseline:         ## pipeline 08 (long) in the background → logs/baseline.log
	@mkdir -p logs
	nohup $(PY) pipeline/08_baseline_matrix.py > logs/baseline.log 2>&1 &
	@echo "running in background: tail -f logs/baseline.log"

baseline-fallback: ## pipeline 08 with the scipy fallback only (minutes, not hours)
	$(PY) pipeline/08_baseline_matrix.py --fallback

calibrate:        ## fit beta, write calibrated.json
	$(PY) -m khatt.calibrate

api:              ## FastAPI on :8000
	uv run --frozen uvicorn khatt.api.main:app --reload --port $${KHATT_API_PORT:-8000} \
		--host $${KHATT_API_HOST:-127.0.0.1}

api-synthetic: synthetic  ## API on invented data (for front-end work before the pipeline runs)
	KHATT_PROCESSED_DIR=$(SYNTH) uv run --frozen uvicorn khatt.api.main:app --reload --port 8000

web:
	cd web && npm run dev

scenarios:        ## hero scenarios + Monte Carlo
	$(PY) scripts/precompute_scenarios.py

backtest:
	$(PY) scripts/backtest_line3.py

test:
	$(PYTEST)

test-fast:        ## skip the slow performance test
	$(PYTEST) -m "not slow"

lint:
	uv run --frozen ruff check khatt pipeline scripts tests

synthetic: $(SYNTH)/T0.npy

$(SYNTH)/T0.npy:
	$(PY) -m khatt.synthetic $(SYNTH) --rings 31

mock: synthetic   ## example /baseline and /evaluate JSON for the web team
	$(PY) scripts/write_mock.py $(SYNTH) scenarios/mock
