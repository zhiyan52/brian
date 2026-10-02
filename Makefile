.PHONY: help install install-dev lint format check test prepare n4 train evaluate predict clean

PYTHON := python
CONFIG ?= configs/train.yaml

help:
	@echo "IBSR18 3D U-Net"
	@echo ""
	@echo "Available commands:"
	@echo "  make install              Install the project"
	@echo "  make install-dev          Install project with development tools"
	@echo "  make lint                 Run Ruff linting"
	@echo "  make format               Format source code"
	@echo "  make check                Run linting and tests"
	@echo "  make test                 Run tests"
	@echo "  make prepare              Validate and prepare dataset"
	@echo "  make n4                   Run N4 bias-field correction"
	@echo "  make train                Train the 3D U-Net"
	@echo "  make evaluate             Evaluate the model"
	@echo "  make predict              Run inference"
	@echo "  make clean                Remove generated outputs"
	@echo ""
	@echo "Example:"
	@echo "  make train CONFIG=configs/train.yaml"

install:
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -e .

install-dev:
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -e ".[dev]"

lint:
	$(PYTHON) -m ruff check src scripts tests

format:
	$(PYTHON) -m ruff format src scripts tests

check:
	$(PYTHON) -m ruff check src scripts tests
	$(PYTHON) -m pytest

test:
	$(PYTHON) -m pytest

prepare:
	$(PYTHON) scripts/prepare_data.py

n4:
	$(PYTHON) scripts/precompute_n4.py

train:
	$(PYTHON) scripts/train.py --config $(CONFIG)

evaluate:
	$(PYTHON) scripts/evaluate.py --config $(CONFIG)

predict:
	$(PYTHON) scripts/predict.py --config $(CONFIG)

clean:
	$(PYTHON) -c "import shutil; from pathlib import Path; [shutil.rmtree(p) for p in [Path('runs'), Path('outputs'), Path('logs')] if p.exists()]"