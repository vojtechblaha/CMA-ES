.PHONY: install activate lint test

install:
	uv venv
	uv sync --extra dev
	uv tool install pre-commit
	uv tool run pre-commit install

activate:
	@echo "Run: source .venv/bin/activate"

lint:
	uv tool run pre-commit run --all-files

test:
	uv run pytest tests/
