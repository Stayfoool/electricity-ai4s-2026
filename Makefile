PY ?= /root/miniconda3/envs/electricity/bin/python
PIP ?= /root/miniconda3/envs/electricity/bin/pip
PYTHONPATH ?= src
CONFIG ?= configs/base.yaml
CHAMPION_CONFIG ?= configs/ensemble_champion_and_segmented_6.yaml

.PHONY: verify data-check test lint typecheck gate backtest window-backtest pair-window-backtest segmented-backtest ensemble tune submit champion-submit mlflow-ui

verify:
	$(PY) scripts/verify_env.py

data-check:
	$(PY) scripts/check_data_manifest.py

test:
	$(PY) -m pytest -q

lint:
	$(PY) -m ruff check src scripts tests

typecheck:
	$(PY) -m mypy src scripts tests

gate: lint test

backtest:
	PYTHONPATH=$(PYTHONPATH) $(PY) -m electricity.cli backtest --config $(CONFIG)

window-backtest:
	PYTHONPATH=$(PYTHONPATH) $(PY) -m electricity.cli window-backtest --config $(CONFIG)

pair-window-backtest:
	PYTHONPATH=$(PYTHONPATH) $(PY) -m electricity.cli pair-window-backtest --config $(CONFIG)

segmented-backtest:
	PYTHONPATH=$(PYTHONPATH) $(PY) -m electricity.cli segmented-backtest --config $(CONFIG)

ensemble:
	PYTHONPATH=$(PYTHONPATH) $(PY) -m electricity.cli ensemble --config $(CONFIG)

tune:
	PYTHONPATH=$(PYTHONPATH) $(PY) -m electricity.cli tune --config $(CONFIG)

submit:
	PYTHONPATH=$(PYTHONPATH) $(PY) -m electricity.cli submit --config $(CONFIG)

champion-submit:
	PYTHONPATH=$(PYTHONPATH) $(PY) -m electricity.cli submit --config $(CHAMPION_CONFIG)

mlflow-ui:
	/root/miniconda3/envs/electricity/bin/mlflow ui --backend-store-uri mlruns --host 0.0.0.0 --port 5000
