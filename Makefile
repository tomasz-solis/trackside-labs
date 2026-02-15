dev-sync:
	uv sync --extra dev

fmt:
	uv run --extra dev ruff check src tests scripts app.py predict_weekend.py --fix
	uv run --extra dev ruff format src tests scripts app.py predict_weekend.py

lint:
	uv run --extra dev ruff check src tests scripts app.py predict_weekend.py

test:
	uv run --extra dev pytest -q
