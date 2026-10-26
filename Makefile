.PHONY: setup fmt lint type test test-all check
setup:     ; uv sync && uv run pre-commit install
fmt:       ; uv run ruff format . && uv run ruff check --fix .
lint:      ; uv run ruff check . && uv run ruff format --check .
type:      ; uv run mypy src
test:      ; uv run pytest -m "not slow" --cov=btcx --cov-report=term-missing
test-all:  ; uv run pytest
check: lint type test
