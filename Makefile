fmt:
	ruff check src tests scripts app.py predict_weekend.py --fix
	ruff format src tests scripts app.py predict_weekend.py

lint:
	ruff check src tests scripts app.py predict_weekend.py

test:
	pytest -q
