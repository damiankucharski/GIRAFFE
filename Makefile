.PHONY: test
.PHONY: mypy
.PHONY: all_test

test:
	@echo "Running tests..."
	uv run pytest test

mypy:
	@echo "Running mypy..."
	uv run mypy giraffe


all_test: test mypy


