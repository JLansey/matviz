# Matviz Testing Makefile

.PHONY: help install test lint clean

help:
	@echo "Available commands:"
	@echo "  make install    - Install package and dev dependencies"
	@echo "  make test       - Run tests"
	@echo "  make lint       - Run linting"
	@echo "  make clean      - Clean up generated files"

install:
	pip install -e ".[dev]"

test:
	pytest tests/ -v

lint:
	flake8 matviz/ --max-line-length=127

clean:
	rm -rf build/ dist/ *.egg-info/ .pytest_cache/ .coverage htmlcov/
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
