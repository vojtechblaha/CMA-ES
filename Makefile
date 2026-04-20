.PHONY: install lint test

install:
	uv venv
	uv pip install -e ".[dev]"
	uv tool install pre-commit
	uv tool run pre-commit install

lint:
	uv tool run pre-commit run --all-files

test:
	uv run pytest tests/
