# RAG Pipeline (LLMOps)

## Table of Contents

- Introduction
- Methodology
  - Design
  - Core Components
  - High-Level Workflow
  - Hypotheses
  - Data Collection and Preprocessing
  - Vector Database and Retrieval
  - Model and Training
  - Experiment Tracking and Versioning
  - Evaluation
  - Experimental Conditions and Ablations
  - Deployment as a Production Artifact
  - Reproducibility Plan
  - Hardware Requirements
  - Ethical Considerations
  - Limitations
- Features
- Project Structure
- Get Started - Step-By-Step
  - Initial Setup Checklist
  - Model Experiment
    - Step 1: Environment Setup
    - Step 2: Install Dependencies
    - Step 3: Configure DagsHub and MLflow
    - Step 4: Prepare Data
    - Step 5: Run Data Pipeline
    - Step 6: Build Vector Store
    - Step 7: Fine-Tune Model
    - Step 8: Evaluate RAG Pipeline
    - Step 9: View MLflow Dashboard
    - Step 10: DVC Pipeline Commands
- Docker Deployment
  - Build and Run Locally
  - Docker Compose Services
  - Docker Environment Variables
  - Test the API
  - Docker Commands Reference
- AWS Deployment
  - Setup AWS Requirements
  - Deploy with Shell Script
  - Deploy with Python Script
  - Monitor Deployment
  - AWS Architecture
  - Required IAM Roles
  - Production Recommendation
- API Usage
  - Base URL
  - Endpoints
    - Health Check
    - Single Query
    - Batch Query
  - Python Client Example
- Testing
  - Run All Tests
  - Test Categories
  - Generate Coverage Report
- Environment Variables
- Monitoring and Logging
  - MLflow Metrics Tracked
    - Data Quality Metrics
    - Vectorstore Metrics
    - Retrieval Metrics
    - Training Metrics
    - Generation Metrics
    - Performance Metrics
  - View MLflow Dashboard
  - View DVC Pipeline Status
  - Production Monitoring Recommendations
- Troubleshooting
  - Common Issues and Solutions
  - Debug Mode
  - DagsHub Authentication Issues
  - MLflow Permission Issues
  - Out of Memory (OOM)
  - Docker Issues
  - AWS ECS Task Fails to Start
- Contributing
- License

## Introduction

A production-ready Retrieval-Augmented Generation (RAG) pipeline for coding question answering, featuring comprehensive experiment tracking, model fine-tuning, and cloud deployment. This pipeline includes experiment tracking with MLflow on DagsHub, data version control with DVC, containerization with Docker, and deployment on AWS ECS with GPU support.

This project is a **reproducible LLMOps pipeline** for building a **coding assistant RAG system** using **ChromaDB** for retrieval and a **fine-tuned Qwen3-0.6B** model (via QLoRA + DPO) for generation. It is designed for production deployment and educational purposes, with containerized inference on AWS ECS. The pipeline covers the full lifecycle of a RAG-based language model service:

1. Raw data ingestion and preprocessing.
2. Vector database construction with BAAI embeddings.
3. Model fine-tuning with QLoRA and DPO.
4. Retrieval and generation pipeline with structured output.
5. Experiment tracking with MLflow on DagsHub.
6. Data and model versioning with DVC.
7. Multi-metric evaluation for retrieval and generation.
8. Dockerized inference.
9. AWS ECS deployment with GPU support.

Retrieval-Augmented Generation systems are difficult to build reliably because retrieval quality, prompt design, and generation quality interact in complex ways. This project investigates how **optimized retrieval combined with preference-aligned generation** can improve coding-assistant answers while remaining reproducible and deployable. It also demonstrates how MLOps practices can make RAG research more transparent, comparable, and production-ready.

---

## Methodology

**Design**

This project follows a **production-grade, reproducible ML engineering design**. The central goal is to build an end-to-end RAG pipeline that retrieves relevant Stack Overflow Q&A chunks and generates accurate coding answers with a fine-tuned Qwen3-0.6B model. The study compares retrieval strategies and generation configurations using retrieval and generation metrics.

**Core Components**

| Component | Purpose |
|---|---|
| **Data Pipeline** | Cleans HTML, filters Stack Overflow Q&A pairs, and splits data. |
| **Vector Database** | Builds ChromaDB index with BAAI embeddings. |
| **Model Fine-tuning** | Trains Qwen3-0.6B with QLoRA and DPO. |
| **RAG Pipeline** | Optimized retrieval and generation with structured output. |
| **MLflow + DagsHub** | Tracks parameters, metrics, and artifacts. |
| **DVC** | Versions datasets, embeddings, and model artifacts. |
| **Evaluation Suite** | Measures hit rate, MRR, MAP, nDCG, BLEU, ROUGE, BERTScore, perplexity. |
| **Docker Inference API** | Serves the RAG pipeline via FastAPI. |
| **AWS Deployment** | ECR, ECS Fargate with GPU, CloudWatch, and ALB. |
| **Tests** | Pytest coverage for data, retrieval, model, RAG, and evaluation. |

**High-Level Workflow**

```text
Stack Overflow CSVs
        |
        V
DVC-tracked raw data
        |
        V
Preprocessing → filtered Q&A pairs → train/val/test splits
        |
        V
ChromaDB + BAAI embeddings → vector index
        |
        V
Qwen3-0.6B + QLoRA + DPO fine-tuning
        |
        V
MLflow tracking + DVC artifacts
        |
        V
RAG pipeline: Query → Retrieve → Generate → Answer
        |
        V
Evaluation (Hit Rate, MRR, MAP, nDCG, BLEU, ROUGE, BERTScore, perplexity)
        |
        V
Dockerized FastAPI inference
        |
        V
AWS ECS deployment with GPU support
```

**Hypotheses**

- **H1:** A fine-tuned Qwen3-0.6B with DPO improves generation quality (BLEU, ROUGE, BERTScore) over the base model.
- **H2:** ChromaDB with BAAI embeddings provides strong retrieval quality (high hit rate, MRR, nDCG) for coding questions.
- **H3:** Combining retrieval with fine-tuned generation outperforms generation-only baselines.
- **H4:** MLflow + DVC enable reliable reproduction of retrieval and generation experiments across machines.

**Data Collection and Preprocessing**

- **Source:** Stack Overflow `stacksample` dataset from Kaggle.
- **Files:** `Questions.csv`, `Answers.csv`.
- **Cleaning:** Remove HTML tags from question and answer bodies.
- **Filtering:** Keep answers with score > 5; select top-N questions (default 50).
- **Splits:** 70% train, 15% validation, 15% test.
- **Format:** Convert Q&A pairs into structured documents for retrieval and training.
- **Versioning:** Track raw and processed data with DVC.

**Vector Database and Retrieval**

- **Database:** ChromaDB (persistent local index).
- **Embedding Model:** `BAAI/bge-small-en-v1.5` (384-dim).
- **Chunking:** `chunk_size = 512`, `chunk_overlap = 64`.
- **Retrieval:** Top-k = 10 chunks per query.
- **Visualization:** PCA/UMAP projections of embeddings.
- **Evaluation:** Hit rate@k (1, 3, 5, 10), MRR, MAP@10, nDCG@10.

**Model and Training**

- **Base Model:** Qwen/Qwen3-0.6B.
- **Quantization:** 4-bit QLoRA.
- **Adapters:** LoRA with `r=16`, `alpha=32`, `dropout=0.1`.
- **Objective:** DPO loss (`beta=0.1`).
- **Key Hyperparameters:**
  - `epochs`: 1 (increase for production)
  - `batch_size`: 4
  - `learning_rate`: 5e-5
  - `max_seq_length`: 512
- **Hardware:** NVIDIA GPU with at least 12GB VRAM.
- **Monitoring:** GPU usage via `nvidia-smi`; training metrics via MLflow.

**Experiment Tracking and Versioning**

- **MLflow:** Logs parameters, metrics, and artifacts for every retrieval, training, and evaluation run.
- **DagsHub:** Remote MLflow tracking server with experiment comparison.
- **DVC:** Versions raw data, processed splits, vector index, and model artifacts.
- **Configuration:** All hyperparameters stored in `params.yaml`.
- **DVC Experiments:** `dvc exp run --set-param ...` for parameter sweeps.

**Evaluation**

Evaluation is performed on the held-out test set using:

- **Retrieval Metrics:** Hit rate@k (1, 3, 5, 10), MRR, MAP@10, nDCG@10.
- **Generation Metrics:** BLEU, ROUGE-1/2/L, BERTScore precision/recall/F1, perplexity.
- **Performance Metrics:** Average query time (ms), retrieval time (s), generation time (s), GPU memory (GB), CPU percent.
- **Training Metrics:** Final train loss, final eval loss, best eval loss, total steps.

Metrics are saved to DVC-tracked files and logged to MLflow.

**Experimental Conditions and Ablations**

- **Baseline:** Base Qwen3-0.6B without retrieval.
- **Main Condition:** RAG pipeline with fine-tuned Qwen3-0.6B.
- **Ablations:**
  - Retrieval top-k: 5, 10, 20.
  - Chunk size: 256, 512, 1024.
  - Embedding model: BAAI bge-small vs. bge-base.
  - Without retrieval (generation-only).
  - Without DPO (SFT only).
- **Replication:** Run multiple seeds where compute permits and report mean ± standard deviation.

**Deployment as a Production Artifact**

- **Inference API:** FastAPI service with `/health`, `/query`, and `/batch_query` endpoints.
- **Containerization:** Docker with GPU support and Nginx reverse proxy.
- **Cloud Deployment:** AWS ECR for images, ECS Fargate (or EC2 GPU) for serving, CloudWatch for logs.
- **Process Management:** Supervisor for multi-process container orchestration.
- **Purpose:** Production-ready coding assistant API.

**Reproducibility Plan**

- All configurations stored in version-controlled `params.yaml`.
- Raw data, processed splits, vector index, and models tracked with DVC.
- Every run logged to MLflow with parameters, metrics, and artifacts.
- Docker image provided for consistent inference.
- Tests included for data, vectorstore, retrieval, model, RAG, and evaluation.
- Environment variables documented in `.env.example`.

**Hardware Requirements**

- **GPU**: NVIDIA GPU with at least 12GB VRAM (for training).
- **RAM**: 16GB minimum, 32GB recommended.
- **Storage**: 50GB free space.
- **OS**: Ubuntu 22.04+ / Windows WSL2 / macOS.
- **Python**: 3.10 or higher.
- **CUDA**: 12.1+ with cuDNN 8.9+.
- **Docker**: 24.0+ (for container deployment).
- **AWS CLI**: 2.0+ (for AWS deployment).
- **Accounts**: DagsHub token, Hugging Face token (optional), AWS account.

**Ethical Considerations**

- Stack Overflow data is used under its original license; users must verify terms.
- Generated code may contain insecure, biased, or outdated patterns.
- Automatic metrics are proxies and should not replace human evaluation.
- The model should not be deployed in high-stakes settings without further validation.

**Limitations**

- BLEU, ROUGE, and BERTScore may not fully reflect code correctness.
- Retrieval quality depends on the embedding model and chunking strategy.
- 4-bit quantization may affect generation quality.
- Default training runs use 1 epoch; production requires more.
- AWS ECS with GPU is more expensive than CPU-only inference.
- Stack Overflow data may contain biases and outdated solutions.

---

## Features

- **End-to-End RAG Pipeline**: Data → Vector DB → Fine-tuned LLM → Answer.
- **Vector Database**: ChromaDB with BAAI embeddings for semantic search.
- **DPO + QLoRA Fine-tuning**: Efficient 4-bit quantization + LoRA adapters.
- **Experiment Tracking**: MLflow with DagsHub integration.
- **Data Version Control**: DVC for dataset, index, and model versioning.
- **Comprehensive Metrics**: Hit rate, MRR, MAP, nDCG, BLEU, ROUGE, BERTScore, perplexity.
- **Performance Monitoring**: Query time, retrieval time, generation time, GPU memory, CPU.
- **Docker Containerization**: GPU-enabled Docker image with Nginx and Supervisor.
- **AWS Deployment**: ECR, ECS Fargate with GPU, CloudWatch, ALB.
- **Integration Tests**: Pytest with coverage reporting.

---

## Project Structure

```text
llmops-rag-pipeline/
├── .dvc/                      # DVC cache and configuration
├── .dvcignore                 # DVC ignore patterns
├── dvc.yaml                   # DVC pipeline stages
├── dvc.lock                   # DVC lock file
├── params.yaml                # Pipeline parameters
├── metrics.yaml               # DVC metrics configuration
├── dagshub_config.py          # DagsHub integration
├── Dockerfile                 # Docker image definition
├── docker-compose.yml         # Docker Compose services
├── requirements.txt           # Python dependencies
├── requirements_dev.txt       # Development dependencies
├── setup.py                   # Package setup
├── Makefile                   # Make commands
├── src/
│   ├── config.py              # Configuration management
│   ├── data/                  # Data pipeline modules
│   │   ├── load_data.py
│   │   ├── preprocess.py
│   │   └── split_data.py
│   ├── vectorstore/           # Vector database modules
│   │   ├── create_vectorstore.py
│   │   ├── visualize_vectorstore.py
│   │   └── retrieval_eval.py
│   ├── model/                 # Model training modules
│   │   ├── load_model.py
│   │   ├── fine_tune.py
│   │   └── inference.py
│   ├── rag/                   # RAG pipeline modules
│   │   ├── pipeline.py
│   │   ├── prompt_templates.py
│   │   └── generation.py
│   └── evaluation/            # Evaluation modules
│       ├── retrieval_metrics.py
│       ├── generation_metrics.py
│       ├── performance_monitor.py
│       └── rag_evaluation.py
├── tests/                     # Unit and integration tests
├── scripts/                   # Utility scripts
│   ├── run_pipeline.py
│   ├── deploy_aws.py
│   ├── monitor_experiment.py
│   └── test_api.py
├── deployment/                # Production deployment files
│   ├── Dockerfile.aws
│   ├── nginx.conf
│   ├── supervisor.conf
│   └── aws_deploy.sh
├── notebooks/                 # Jupyter notebooks
│   └── development_notebook.ipynb
└── mlflow/                    # Local MLflow storage
    └── mlflow_server.py
```

---

## Get Started - Step-By-Step

### Initial Setup Checklist

```bash
# 1. Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 2. Configure credentials
cp .env.example .env
nano .env  # Add your DagsHub token

# 3. Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
pip install -r requirements_dev.txt

# 4. Download data
kaggle datasets download stackoverflow/stacksample
unzip stacksample.zip
mkdir -p data/download
mv Questions.csv Answers.csv data/download/

# 5. Install DVC with S3 support
pip install dvc dvc-s3

# 6. Configure DagsHub
export DAGSHUB_USERNAME=fourapiwit
export DAGSHUB_TOKEN=your_token_here

# 7. Configure DVC remote
dvc remote add -d storage https://dagshub.com/fourapiwit/llmops-rag-pipeline.dvc
dvc remote modify storage --local auth basic
dvc remote modify storage --local user $DAGSHUB_USERNAME
dvc remote modify storage --local password $DAGSHUB_TOKEN

# 8. Configure MLflow
export MLFLOW_TRACKING_URI=https://dagshub.com/fourapiwit/llmops-rag-pipeline.mlflow
export MLFLOW_TRACKING_USERNAME=$DAGSHUB_USERNAME
export MLFLOW_TRACKING_PASSWORD=$DAGSHUB_TOKEN

# 9. Run the full pipeline
python scripts/run_pipeline.py
```

### Model Experiment

**Step 1: Environment Setup**

```bash
# Check NVIDIA driver version
nvidia-smi

# If driver is too old, run CUDA setup script
chmod +x setup_cuda.sh
sudo ./setup_cuda.sh

# Verify PyTorch CUDA
python -c "import torch; print(torch.cuda.is_available())"
```

**Step 2: Install Dependencies**

```bash
python -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
pip install -r requirements_dev.txt
```

**Step 3: Configure DagsHub and MLflow**

**3.1 Create a DagsHub Repository**

1. Go to [dagshub.com](https://dagshub.com) and sign up/log in.
2. Create a new repository (e.g., `llmops-rag-pipeline`).
3. Note your username and repository name.

**3.2 Get DagsHub Token**

1. Go to Settings → Tokens.
2. Generate a new token with write permissions.
3. Save the token securely.

**3.3 Set Environment Variables**

```bash
export DAGSHUB_USERNAME=fourapiwit
export DAGSHUB_TOKEN=your_token_here
export MLFLOW_TRACKING_URI=https://dagshub.com/${DAGSHUB_USERNAME}/llmops-rag-pipeline.mlflow
export MLFLOW_TRACKING_USERNAME=${DAGSHUB_USERNAME}
export MLFLOW_TRACKING_PASSWORD=${DAGSHUB_TOKEN}
```

**3.4 Login to Hugging Face (Optional)**

```bash
huggingface-cli login
export HF_TOKEN=hf_your_token_here
```

**3.5 Configure DVC Remote**

```bash
dvc remote add -d storage https://dagshub.com/${DAGSHUB_USERNAME}/llmops-rag-pipeline.dvc
dvc remote modify storage --local auth basic
dvc remote modify storage --local user ${DAGSHUB_USERNAME}
dvc remote modify storage --local password ${DAGSHUB_TOKEN}
```

**Step 4: Prepare Data**

**4.1 Download Stack Overflow Dataset**

```bash
kaggle datasets download stackoverflow/stacksample
unzip stacksample.zip
mkdir -p data/download
mv Questions.csv Answers.csv data/download/
```

**4.2 Verify Data Files**

Ensure the following files exist in `data/download/`:
- `Questions.csv`
- `Answers.csv`

**4.3 Track Data with DVC**

```bash
dvc add data/download
git add data/download.dvc .gitignore
git commit -m "Track raw data with DVC"
dvc push
```

**Step 5: Run Data Pipeline**

```bash
# Run individual stages
python -m src.data.load_data
python -m src.data.preprocess
python -m src.data.split_data

# Or via DVC
dvc repro split_data
```

**What the data pipeline does:**
1. Loads `Questions.csv` and `Answers.csv`.
2. Cleans HTML tags.
3. Filters answers with score > 5.
4. Selects top-N questions (default 50).
5. Splits into train/val/test (70/15/15).
6. Saves processed data for vectorstore and training.

**Expected output:**

```text
Loading questions from data/download/Questions.csv
Loading answers from data/download/Answers.csv
Filtering answers with score > 5...
Selecting top 50 questions...
Cleaning HTML content...
Total QA pairs created: 50
Splitting data (70/15/15)...
Train: 35, Val: 7, Test: 8
Data pipeline completed successfully
```

**Step 6: Build Vector Store**

```bash
python -m src.vectorstore.create_vectorstore
python -m src.vectorstore.visualize_vectorstore
python -m src.vectorstore.retrieval_eval

# Or via DVC
dvc repro create_vectorstore
```

**What vectorstore does:**
1. Chunks documents (`chunk_size=512`, `chunk_overlap=64`).
2. Embeds with `BAAI/bge-small-en-v1.5`.
3. Builds ChromaDB persistent index.
4. Visualizes embeddings with PCA/UMAP.
5. Evaluates retrieval metrics.

**Expected output:**

```text
Creating ChromaDB vectorstore...
Chunking documents (size=512, overlap=64)...
Number of chunks: 128
Embedding with BAAI/bge-small-en-v1.5...
Embedding dimension: 384
Vectorstore created successfully
Retrieval metrics:
  hit_rate@1: 0.625
  hit_rate@3: 0.875
  hit_rate@5: 1.000
  mrr: 0.756
  map@10: 0.712
  ndcg@10: 0.789
```

**Step 7: Fine-Tune Model**

```bash
python -m src.model.fine_tune

# Or via DVC
dvc repro fine_tune_model
```

**What fine-tuning does:**
1. Loads Qwen3-0.6B with 4-bit quantization.
2. Applies LoRA adapters (`r=16`, `alpha=32`, `dropout=0.1`).
3. Trains with DPO loss (`beta=0.1`).
4. Saves fine-tuned model to `models/qwen-dpo-final/`.
5. Logs metrics to MLflow.

**Key training parameters (configurable in `params.yaml`):**

- `epochs`: 1
- `batch_size`: 4
- `learning_rate`: 5e-5
- `dpo_beta`: 0.1
- `max_seq_length`: 512

**Training progress monitoring:**

```bash
watch -n 1 nvidia-smi
mlflow ui --backend-store-uri $MLFLOW_TRACKING_URI
```

**Expected output:**

```text
============================================================
FINE-TUNING QWEN3-0.6B WITH DPO + QLORA
============================================================
Loading base model with 4-bit quantization...
Applying LoRA adapters (r=16, alpha=32)...
Training for 1 epoch...
Step 10/35: train_loss=0.6931, eval_loss=0.6821
Step 20/35: train_loss=0.5124, eval_loss=0.5034
Step 30/35: train_loss=0.4231, eval_loss=0.4189
Step 35/35: train_loss=0.3912, eval_loss=0.3901
Model saved to models/qwen-dpo-final/
============================================================
TRAINING COMPLETED
============================================================
```

**Step 8: Evaluate RAG Pipeline**

```bash
python -m src.evaluation.rag_evaluation

# Or via DVC
dvc repro evaluate_rag
```

**What evaluation does:**
1. Loads the fine-tuned model, vectorstore, and test set.
2. Runs retrieval + generation on test queries.
3. Computes retrieval metrics (hit rate, MRR, MAP, nDCG).
4. Computes generation metrics (BLEU, ROUGE, BERTScore, perplexity).
5. Measures performance (query time, retrieval time, generation time).
6. Saves metrics to `metrics/rag_evaluation.json`.
7. Logs metrics to MLflow.

**Expected output:**

```text
============================================================
RAG EVALUATION RESULTS
============================================================
Retrieval:
  hit_rate@1: 0.625
  hit_rate@3: 0.875
  hit_rate@5: 1.000
  mrr: 0.756
  map@10: 0.712
  ndcg@10: 0.789

Generation:
  bleu: 0.0412
  rouge1: 0.2513
  rouge2: 0.0512
  rougeL: 0.1321
  bertscore_f1: 0.8123
  perplexity: 2.7123

Performance:
  avg_query_time_ms: 1234.56
  avg_retrieval_time_s: 0.045
  avg_generation_time_s: 1.189
  avg_gpu_memory_gb: 4.21
  avg_cpu_percent: 34.2

============================================================
PIPELINE COMPLETED SUCCESSFULLY
============================================================
```

**Step 9: View MLflow Dashboard**

```bash
# Local MLflow UI
mlflow ui --backend-store-uri ./mlruns

# DagsHub MLflow (read-only)
mlflow ui --backend-store-uri $MLFLOW_TRACKING_URI
```

**What you can see in MLflow:**

- **Parameters**: All hyperparameters used in retrieval, training, and evaluation.
- **Metrics**: Retrieval metrics, generation metrics, performance metrics.
- **Artifacts**: Model files, vectorstore index, evaluation JSON, plots.
- **Tags**: Run name, stage, version.
- **Search**: Filter runs by parameters or metrics.
- **Compare**: Compare multiple runs side-by-side.

**Step 10: DVC Pipeline Commands**

```bash
# Show pipeline DAG
dvc dag

# Reproduce specific stage
dvc repro create_vectorstore
dvc repro fine_tune_model
dvc repro evaluate_rag

# Show metrics
dvc metrics show
dvc metrics diff

# Push/pull data to remote
dvc push
dvc pull

# Run experiments with different parameters
dvc exp run --set-param data.top_n_questions=100
dvc exp run --set-param training.epochs=3
```

---

## Docker Deployment

**Build and Run Locally**

```bash
# Build Docker image
docker build -t llmops-rag:latest .

# Run with GPU support
docker run --gpus all -p 8000:8000 -p 5000:5000 \
  -e DAGSHUB_USERNAME=fourapiwit \
  -e DAGSHUB_TOKEN=your_token \
  llmops-rag:latest

# Run with Docker Compose
docker-compose up --build

# Stop containers
docker-compose down
```

**Docker Compose Services**

| Service | Port | Description |
|---|---|---|
| `rag-api` | 8000 | FastAPI RAG service |
| `mlflow-server` | 5001 | MLflow tracking UI |
| `dvc-server` | - | DVC metrics viewer |

**Docker Environment Variables**

```bash
CUDA_VISIBLE_DEVICES=0
MLFLOW_TRACKING_URI=https://dagshub.com/fourapiwit/llmops-rag-pipeline.mlflow
MLFLOW_TRACKING_USERNAME=fourapiwit
MLFLOW_TRACKING_PASSWORD=your_token
HF_TOKEN=hf_your_token
```

**Test the API**

```bash
# Health check
curl -X GET http://localhost:8000/health

# Single query
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "How to reverse a string in Python?"}'

# Batch query
curl -X POST http://localhost:8000/batch_query \
  -H "Content-Type: application/json" \
  -d '{"queries": ["How to sort a list in Python?", "How to read a CSV file?"]}'
```

**Docker Commands Reference**

| Command | Description |
|---|---|
| `docker build -t llmops-rag .` | Build image |
| `docker run --gpus all -p 8000:8000 llmops-rag` | Run container with GPU |
| `docker-compose up --build` | Run all services |
| `docker-compose down` | Stop all services |
| `docker ps` | List running containers |
| `docker stop <container_id>` | Stop container |
| `docker rm <container_id>` | Remove container |
| `docker rmi llmops-rag` | Remove image |

---

## AWS Deployment

**Setup AWS Requirements**

```bash
# Install AWS CLI
curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"
unzip awscliv2.zip
sudo ./aws/install

# Configure AWS credentials
aws configure
# Enter: AWS Access Key ID, Secret Key, region (us-east-1)

# Install jq for JSON parsing
sudo apt-get install jq
```

**Deploy with Shell Script**

```bash
# Set AWS environment variables
export AWS_ACCESS_KEY_ID=your_access_key
export AWS_SECRET_ACCESS_KEY=your_secret_key
export AWS_DEFAULT_REGION=us-east-1

# Run deployment script
chmod +x deployment/aws_deploy.sh
./deployment/aws_deploy.sh
```

**Deploy with Python Script**

```bash
python scripts/deploy_aws.py \
  --repository llmops-rag \
  --cluster llmops-cluster \
  --service llmops-rag-service
```

**Monitor Deployment**

```bash
# Check service status
aws ecs describe-services \
  --cluster llmops-cluster \
  --services llmops-rag-service

# View logs
aws logs get-log-events \
  --log-group-name /ecs/llmops-rag-task \
  --log-stream-name ecs/llmops-rag/container-id

# Get public IP
aws ecs describe-tasks \
  --cluster llmops-cluster \
  --tasks $(aws ecs list-tasks --cluster llmops-cluster --query 'taskArns[0]' --output text) \
  --query 'tasks[0].attachments[0].details[?name==`networkInterfaceId`].value' --output text
```

**AWS Architecture**

```text
┌─────────────────────────────────────────────────────────────────┐
│                         AWS Cloud                               │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐       │
│  │    ECR       │    │    ECS       │    │  CloudWatch  │       │
│  │  Container   │───>│   Fargate    │───>│    Logs      │       │
│  │  Registry    │    │   Service    │    │              │       │
│  └──────────────┘    └──────────────┘    └──────────────┘       │
│         │                   │                   │               │
│         v                   v                   v               │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐       │
│  │   ALB        │    │   Security   │    │    VPC       │       │
│  │  (Optional)  │    │   Groups     │    │  Networking  │       │
│  └──────────────┘    └──────────────┘    └──────────────┘       │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

**Required IAM Roles**

- `ecsTaskExecutionRole`: For ECS to pull images from ECR.
- `ecsTaskRole`: For container to access AWS services.

**Production Recommendation**

- **ECS Fargate with GPU:** Recommended for production RAG serving.
- **ALB:** Add an Application Load Balancer for traffic distribution.
- **Auto-scaling:** Configure target-tracking on CPU/GPU utilization.
- **CloudWatch:** Monitor API latency, error rates, and GPU usage.

---

## API Usage

**Base URL**

- Local: `http://localhost:8000`
- AWS: `http://<ec2-public-ip>:8000`

**Endpoints**

**Health Check**

```bash
curl -X GET http://localhost:8000/health
```

**Response:**

```json
{
  "status": "healthy",
  "model_loaded": true,
  "vectorstore_loaded": true,
  "device": "cuda"
}
```

**Single Query**

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "How to reverse a string in Python?"}'
```

**Response:**

```json
{
  "answer": "def reverse_string(s):\n    return s[::-1]",
  "query_time_ms": 1234.56
}
```

**Batch Query**

```bash
curl -X POST http://localhost:8000/batch_query \
  -H "Content-Type: application/json" \
  -d '{
    "queries": [
      "How to sort a list in Python?",
      "How to read a CSV file?"
    ]
  }'
```

**Response:**

```json
{
  "answers": [
    "def sort_list(lst):\n    return sorted(lst)",
    "import csv\nwith open('file.csv') as f:\n    reader = csv.reader(f)"
  ],
  "total_time_ms": 2345.67
}
```

**Python Client Example**

```python
import requests

API_URL = "http://localhost:8000"

def query_rag(question: str) -> str:
    response = requests.post(
        f"{API_URL}/query",
        json={"query": question}
    )
    return response.json()["answer"]

answer = query_rag("How to create a list comprehension?")
print(answer)
```

---

## Testing

**Run All Tests**

```bash
# Run with coverage
pytest tests/ -v --cov=src --cov-report=html

# Run specific test file
pytest tests/test_retrieval.py -v

# Run with verbose output
pytest tests/ -v -s
```

**Test Categories**

| Test File | Description |
|---|---|
| `test_data.py` | Data loading, filtering, and preprocessing |
| `test_vectorstore.py` | Vector database creation and embedding |
| `test_retrieval.py` | Retrieval metrics (hit rate, MRR, MAP, nDCG) |
| `test_model.py` | Model configuration and quantization |
| `test_rag_pipeline.py` | RAG pipeline components |
| `test_evaluation.py` | Generation metrics (BLEU, ROUGE, BERTScore) |

**Generate Coverage Report**

```bash
pytest tests/ --cov=src --cov-report=html
open htmlcov/index.html
```

---

## Environment Variables

Create a `.env` file for persistent configuration:

```bash
# DagsHub Configuration
DAGSHUB_USERNAME=fourapiwit
DAGSHUB_TOKEN=your-token
MLFLOW_TRACKING_URI=https://dagshub.com/fourapiwit/llmops-rag-pipeline.mlflow
MLFLOW_TRACKING_USERNAME=fourapiwit
MLFLOW_TRACKING_PASSWORD=your-token

# HuggingFace Configuration
HF_TOKEN=hf_your-token

# AWS Configuration
AWS_ACCESS_KEY_ID=your-access-key
AWS_SECRET_ACCESS_KEY=your-secret-key
AWS_DEFAULT_REGION=us-east-1

# Local Paths
MODEL_PATH=./models/qwen-dpo-final
DATA_PATH=./data/download
VECTORSTORE_PATH=./chroma_db
OUTPUT_PATH=./models

# MLflow Configuration
MLFLOW_EXPERIMENT_NAME=llmops-rag-pipeline-experiment
```

Load environment variables:

```bash
source .env
# Or use python-dotenv
```

---

## Monitoring and Logging

**MLflow Metrics Tracked**

**Data Quality Metrics:**

- `num_questions_filtered`: Questions after filtering.
- `num_answers_filtered`: Answers after filtering.
- `questions_score_mean`: Mean question score.
- `answers_score_mean`: Mean answer score.
- `total_qa_pairs`: Total QA pairs created.

**Vectorstore Metrics:**

- `num_documents`: Source documents.
- `num_chunks`: Text chunks.
- `avg_chunk_length`: Average chunk length.
- `embedding_dimension`: 384.

**Retrieval Metrics:**

- `hit_rate@k`: Hit rate at k (1, 3, 5, 10).
- `mrr`: Mean Reciprocal Rank.
- `map@10`: Mean Average Precision at 10.
- `ndcg@10`: Normalized Discounted Cumulative Gain at 10.

**Training Metrics:**

- `final_train_loss`, `final_eval_loss`, `best_eval_loss`.
- `total_training_steps`.

**Generation Metrics:**

- `bleu`, `rouge1`, `rouge2`, `rougeL`.
- `bertscore_precision`, `bertscore_recall`, `bertscore_f1`.
- `perplexity`.

**Performance Metrics:**

- `avg_query_time_ms`, `avg_retrieval_time_s`, `avg_generation_time_s`.
- `avg_gpu_memory_gb`, `avg_cpu_percent`.

**View MLflow Dashboard**

```bash
# Local tracking
mlflow ui --backend-store-uri ./mlruns

# DagsHub tracking
# Visit: https://dagshub.com/fourapiwit/llmops-rag-pipeline
```

**View DVC Pipeline Status**

```bash
dvc dag
dvc status
dvc metrics show
dvc plots show
```

**Production Monitoring Recommendations**

- **API Latency:** Track p50/p95/p99 latency for `/query` and `/batch_query`.
- **Error Rates:** Monitor 4xx/5xx responses via ALB or API Gateway logs.
- **GPU Utilization:** Use CloudWatch or Prometheus with NVIDIA DCGM exporter.
- **Retrieval Quality:** Periodically re-run retrieval evaluation on fresh queries.
- **Cost:** Track GPU hours, ECS task hours, and ECR storage.

---

## Troubleshooting

**Common Issues and Solutions**

| Issue | Solution |
|---|---|
| `ModuleNotFoundError: No module named 'src'` | Run from project root directory, use `python -m` |
| CUDA out of memory | Reduce `batch_size` in `params.yaml` |
| ChromaDB index corruption | Delete `chroma_db/` and recreate: `rm -rf chroma_db/` |
| MLflow 403 error | Check DagsHub token is valid and has correct permissions |
| DVC push fails | Run `dvc remote modify storage --local auth basic` |
| Git push 500 error | Remove large files from git history (use DVC instead) |

**Debug Mode**

```bash
# Enable debug logging
export LOG_LEVEL=DEBUG
python scripts/run_pipeline.py

# Run single stage with verbose output
python -m src.model.fine_tune 2>&1 | tee training.log
```

**DagsHub Authentication Issues**

```bash
# Reconfigure DVC remote
dvc remote remove storage
dvc remote add -d storage https://dagshub.com/${DAGSHUB_USERNAME}/llmops-rag-pipeline.dvc
dvc remote modify storage --local auth basic
dvc remote modify storage --local user ${DAGSHUB_USERNAME}
dvc remote modify storage --local password ${DAGSHUB_TOKEN}

# Test connection
dvc pull -r storage
```

**MLflow Permission Issues**

```bash
# Use local tracking instead of remote
export MLFLOW_TRACKING_URI="file:./mlruns"

# Or create a new experiment
mlflow experiments create -n llmops-rag-pipeline-experiment
```

**Out of Memory (OOM)**

```yaml
# Reduce batch size in params.yaml
training:
  batch_size: 2  # instead of 4

# Reduce sequence length
training:
  max_seq_length: 256  # instead of 512
```

**Docker Issues**

```bash
# Check if Docker is installed
docker --version

# Verify GPU is available to Docker
docker run --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi

# Install NVIDIA Container Toolkit if GPU is not detected
# https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html
```

**AWS ECS Task Fails to Start**

```bash
# Check CloudWatch logs
aws logs describe-log-groups --log-group-name-prefix /ecs/llmops-rag

# Get task failure reason
aws ecs describe-tasks --cluster llmops-cluster --tasks TASK_ID
```

---

## Contributing

Contributions are welcome for production hardening, retrieval metrics, ablation support, and documentation. Please open an issue or pull request with a clear description, and ensure tests pass before submitting.

---

## License

MIT License - see `LICENSE` file for details.

---

**Note**: This pipeline is designed for production and educational purposes. For real deployments, ensure proper security, monitoring, scaling, access control, and compliance with the licenses of the data and base model.
