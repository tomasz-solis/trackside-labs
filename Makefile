.PHONY: fmt lint test

fmt:
	black src tests
	ruff check src tests --fix

lint:
	ruff check src tests

test:
	pytest -q
