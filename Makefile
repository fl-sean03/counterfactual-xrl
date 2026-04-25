.PHONY: install lint format test test-fast smoke train eval explain report clean

# Project venv created by `make install`. Override on the command line if
# you keep the env elsewhere: `make test PY=/path/to/python`.
VENV ?= .venv
PY ?= $(VENV)/bin/python
PIP ?= $(VENV)/bin/pip

install:
	python3 -m venv $(VENV)
	$(PIP) install --upgrade pip wheel setuptools
	$(PIP) install \
	    --extra-index-url https://download.pytorch.org/whl/cu128 \
	    numpy pandas matplotlib seaborn pyyaml tqdm jsonschema pytest pytest-cov \
	    torch==2.9.1 gymnasium==0.29.1 minigrid==2.3.1 stable-baselines3==2.3.2 \
	    "anthropic>=0.34.0" "openai>=1.40.0" tensorboard \
	    ruff==0.6.9 black==24.8.0 mypy==1.11.2
	$(PIP) install -e .

lint:
	$(PY) -m ruff check src tests scripts
	$(PY) -m black --check src tests scripts

format:
	$(PY) -m ruff check --fix src tests scripts
	$(PY) -m black src tests scripts

test:
	$(PY) -m pytest

test-fast:
	$(PY) -m pytest -m "not slow and not api"

smoke:
	$(PY) scripts/smoke_env.py

train:
	$(PY) scripts/train_ppo.py --config configs/ppo_tuned.yaml --n-envs 16

eval:
	$(PY) scripts/eval.py

explain:
	$(PY) scripts/explain.py

report:
	cd report && latexmk -pdf main.tex

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.py[cod]" -delete
	rm -rf .pytest_cache .mypy_cache .ruff_cache
