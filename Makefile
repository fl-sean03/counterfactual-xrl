.PHONY: install lint format test test-fast smoke train eval explain report clean

ENV_NAME := xrl
CONDA_RUN := conda run -n $(ENV_NAME) --no-capture-output

install:
	conda env create -f environment.yml || conda env update -n $(ENV_NAME) -f environment.yml --prune
	$(CONDA_RUN) pip install -e .

lint:
	$(CONDA_RUN) python -m ruff check src tests scripts
	$(CONDA_RUN) python -m black --check src tests scripts

format:
	$(CONDA_RUN) python -m ruff check --fix src tests scripts
	$(CONDA_RUN) python -m black src tests scripts

test:
	$(CONDA_RUN) python -m pytest

test-fast:
	$(CONDA_RUN) python -m pytest -m "not slow and not api"

smoke:
	$(CONDA_RUN) python scripts/smoke_env.py

train:
	$(CONDA_RUN) python scripts/train_dqn.py --config configs/dqn_baseline.yaml

eval:
	$(CONDA_RUN) python scripts/eval.py

explain:
	$(CONDA_RUN) python scripts/explain.py

report:
	cd report && latexmk -pdf main.tex

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.py[cod]" -delete
	rm -rf .pytest_cache .mypy_cache .ruff_cache
