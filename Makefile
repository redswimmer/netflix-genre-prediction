.PHONY: help install preprocess train lint format clean all

help:
	@echo "Netflix Genre Prediction - Available Commands"
	@echo ""
	@echo "  make install       - Install all dependencies (includes dev tools)"
	@echo "  make preprocess    - Run data preprocessing pipeline"
	@echo "  make train         - Train and evaluate models"
	@echo "  make lint          - Check code with ruff linter"
	@echo "  make format        - Format code with ruff"
	@echo "  make clean         - Remove generated files and cache"
	@echo "  make all           - Run full pipeline (preprocess + train)"
	@echo ""

install:
	uv sync --extra dev

preprocess:
	python netflix_preprocessing.py

train:
	python netflix_training.py

lint:
	uv run ruff check .

format:
	uv run ruff format .

clean:
	rm -rf processed_data/
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete

all: preprocess train
