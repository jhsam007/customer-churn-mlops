.PHONY: install train test run docker-build docker-run

install:
	pip install --upgrade pip
	pip install -e .

train:
	python -m customer_churn.pipeline

test:
	pytest tests -v

run:
	uvicorn app:app --reload

docker-build:
	docker build -t churn-api .

docker-run:
	docker run -p 8000:8000 churn-api
