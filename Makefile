# Matviz Testing Makefile

.PHONY: help install test lint clean tag-version

help:
	@echo "Available commands:"
	@echo "  make install      - Install package and dev dependencies"
	@echo "  make test         - Run tests"
	@echo "  make lint         - Run linting"
	@echo "  make clean        - Clean up generated files"
	@echo "  make tag-version  - Create git tag from current auto version"

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

tag-version:
	@VERSION=$$(python -c "from matviz._version import __version__; print(__version__)"); \
	echo "Creating tag: v$$VERSION"; \
	git tag "v$$VERSION"
