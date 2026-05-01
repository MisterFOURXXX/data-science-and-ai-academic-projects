.PHONY: help setup install test lint format clean run-pipeline run-api docker-build docker-run deploy-aws

help:
	@echo "Available commands:"
	@echo "  make setup        - Setup virtual environment and install dependencies"
	@echo "  make install      - Install all dependencies"
	echo "  make test         - Run tests"
	@echo "  make lint         - Run linters"
	@echo "  make format       - Format code"
	@echo "  make clean        - Clean cache and temporary files"
	@echo "  make run-pipeline - Run full DVC pipeline"
	@echo "  make run-api      - Run FastAPI server"
	@echo "  make docker-build - Build Docker image"
	@echo "  make docker-run   - Run Docker container"
	@echo "  make deploy-aws   - Deploy to AWS ECS"

setup:
	python -m venv venv
	. venv/bin/activate && pip install --upgrade pip

# Make scripts executable
chmod +x scripts/setup_cuda.sh
# Execute script
make setup-cuda

install:
	pip install -r requirements.txt
	pip install -r requirements_dev.txt
	pip install dvc dvc-s3 mlflow dagshub

test:
	pytest tests/ -v --cov=src --cov-report=html

test-unit:
	pytest tests/test_retrieval.py tests/test_evaluation.py -v

test-integration:
	pytest tests/test_rag_pipeline.py tests/test_vectorstore.py -v

lint:
	flake8 src/ tests/
	mypy src/ --ignore-missing-imports
	black --check src/ tests/

format:
	black src/ tests/
	isort src/ tests/

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	rm -rf .pytest_cache/ .coverage htmlcov/ .mypy_cache/
	rm -rf data/raw/ data/processed/ data/splits/
	rm -rf metrics/ *.log
	rm -rf chroma_db/ qwen-dpo-final/

run-pipeline:
	python scripts/run_pipeline.py

run-pipeline-skip-data:
	python scripts/run_pipeline.py --skip-stages load_data preprocess_data split_data

run-api:
	python -m uvicorn deployment.api:app --host 0.0.0.0 --port 8000 --reload

docker-build:
	docker build -t llmops-rag:latest .

docker-run:
	docker run --gpus all -p 8000:8000 -p 5000:5000 llmops-rag:latest

docker-compose-up:
	docker-compose up --build

docker-compose-down:
	docker-compose down

deploy-aws:
	chmod +x deployment/aws_deploy.sh
	./deployment/aws_deploy.sh

dvc-init:
	dvc init
	dvc remote add -d storage s3://your-bucket/dvc-storage

dvc-repro:
	dvc repro

dvc-metrics:
	dvc metrics show

mlflow-ui:
	mlflow ui --backend-store-uri $(MLFLOW_TRACKING_URI)

dagshub-init:
	python dagshub_config.py