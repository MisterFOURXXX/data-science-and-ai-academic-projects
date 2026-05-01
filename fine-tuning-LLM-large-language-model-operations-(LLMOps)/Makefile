.PHONY: help install test train evaluate deploy clean

help:
	@echo "Available commands:"
	@echo "  install     - Install dependencies"
	@echo "  test        - Run tests"
	@echo "  train       - Train model"
	@echo "  evaluate    - Evaluate model"
	@echo "  optimize    - Run hyperparameter optimization"
	@echo "  deploy      - Deploy to AWS"
	@echo "  clean       - Clean cache files"
	@echo "  docker-build - Build Docker image"
	@echo "  docker-run   - Run Docker container"

install:
	pip install -r requirements.txt
	pip install -e .

test:
	pytest tests/ -v --cov=src --cov-report=html

train:
	python scripts/run_pipeline.py

evaluate:
	python scripts/run_evaluation.py

optimize:
	python scripts/run_optimization.py

deploy:
	bash scripts/deploy.sh

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	rm -rf .pytest_cache
	rm -rf .coverage
	rm -rf htmlcov
	rm -rf mlruns
	rm -rf metrics

docker-build:
	docker build -t llmops-rag-model:latest .

docker-run:
	docker-compose up inference

docker-stop:
	docker-compose down

preprocess:
	dvc repro preprocess

pipeline:
	dvc repro train
	dvc repro evaluate

mlflow-ui:
	mlflow ui --backend-store-uri sqlite:///mlruns/mlflow.db --port 5000