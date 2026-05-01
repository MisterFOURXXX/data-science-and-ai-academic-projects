## RAG Pipeline (LLMOps)

A production-ready Retrieval-Augmented Generation (RAG) pipeline for coding question answering, featuring comprehensive experiment tracking, model fine-tuning, and cloud deployment.

### Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Prerequisites](#prerequisites)
- [Quick Start](#quick-start)
- [Detailed Setup](#detailed-setup)
- [Configuration](#configuration)
- [Running the Pipeline](#running-the-pipeline)
- [Monitoring Experiments](#monitoring-experiments)
- [Testing](#testing)
- [Docker Deployment](#docker-deployment)
- [AWS Production Deployment](#aws-production-deployment)
- [API Usage](#api-usage)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)
- [License](#license)

### Overview

This project implements an end-to-end LLMOps pipeline for a coding assistant RAG system with:

- **Data Processing**: HTML cleaning, filtering, and quality thresholding of Stack Overflow Q&A pairs
- **Vector Database**: ChromaDB with BAAI embeddings for semantic search
- **Model Fine-tuning**: Qwen3-0.6B with QLoRA and DPO (Direct Preference Optimization)
- **RAG Pipeline**: Optimized retrieval and generation with structured output
- **Experiment Tracking**: MLflow and DVC integration with DagsHub
- **Production Deployment**: Docker containers on AWS ECS with GPU support

### Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           LLMOps RAG Pipeline                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐                   │
│  │   Data       │    │   Vector     │    │   Model      │                   │
│  │   Pipeline   │──> │   Database   │───>│   Training   │                   │
│  └──────────────┘    └──────────────┘    └──────────────┘                   │
│         │                   │                   │                           │
│         ▼                   ▼                   ▼                           │
│  ┌──────────────────────────────────────────────────────────────┐           │
│  │                    RAG Pipeline                              │           │
│  │  ┌─────────┐    ┌─────────┐    ┌─────────┐    ┌─────────┐    │           │
│  │  │ Query   │───>│ Retrieve│───>│ Generate│───>│ Answer  │    │           │
│  │  └─────────┘    └─────────┘    └─────────┘    └─────────┘    │           │
│  └──────────────────────────────────────────────────────────────┘           │
│                                                                             │
│  ┌──────────────────────────────────────────────────────────────┐           │
│  │                    Monitoring & Tracking                     │           │
│  │  ┌─────────┐    ┌─────────┐    ┌─────────┐    ┌─────────┐    │           │
│  │  │ MLflow  │    │  DVC    │    │ DagsHub │    │  AWS    │    │           │
│  │  └─────────┘    └─────────┘    └─────────┘    └─────────┘    │           │
│  └──────────────────────────────────────────────────────────────┘           │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Project Structure

```
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
│   │   ├── load_data.py       # Load and filter CSV files
│   │   ├── preprocess.py      # Clean HTML and prepare QA pairs
│   │   └── split_data.py      # Train/val/test split
│   ├── vectorstore/           # Vector database modules
│   │   ├── create_vectorstore.py    # Build ChromaDB index
│   │   ├── visualize_vectorstore.py # PCA/UMAP visualization
│   │   └── retrieval_eval.py        # Retrieval metrics
│   ├── model/                 # Model training modules
│   │   ├── load_model.py      # Load fine-tuned model
│   │   ├── fine_tune.py       # DPO/QLoRA training
│   │   └── inference.py       # Model inference utilities
│   ├── rag/                   # RAG pipeline modules
│   │   ├── pipeline.py        # Optimized RAG pipeline
│   │   ├── prompt_templates.py # Chat templates
│   │   └── generation.py      # Response generation
│   └── evaluation/            # Evaluation modules
│       ├── retrieval_metrics.py    # Hit rate, MRR, MAP, nDCG
│       ├── generation_metrics.py   # BLEU, ROUGE, BERTScore
│       ├── performance_monitor.py  # CPU/GPU monitoring
│       └── rag_evaluation.py       # End-to-end evaluation
├── tests/                     # Unit and integration tests
├── scripts/                   # Utility scripts
│   ├── run_pipeline.py        # Pipeline orchestrator
│   ├── deploy_aws.py          # AWS deployment
│   ├── monitor_experiment.py  # MLflow monitoring
│   └── test_api.py            # API testing
├── deployment/                # Production deployment files
│   ├── Dockerfile.aws         # AWS-optimized Dockerfile
│   ├── nginx.conf             # Nginx reverse proxy config
│   ├── supervisor.conf        # Process management
│   └── aws_deploy.sh          # AWS deployment script
├── notebooks/                 # Jupyter notebooks
│   └── development_notebook.ipynb
└── mlflow/                    # Local MLflow storage
    └── mlflow_server.py
```

### Metrics Tracked

**Data Quality Metrics**
- `num_questions_filtered`: Number of questions after filtering
- `num_answers_filtered`: Number of answers after filtering
- `questions_score_mean`: Mean question score
- `answers_score_mean`: Mean answer score
- `total_qa_pairs`: Total QA pairs created

**Vectorstore Metrics**
- `num_documents`: Number of source documents
- `num_chunks`: Number of text chunks
- `avg_chunk_length`: Average chunk length in characters
- `embedding_dimension`: Embedding dimension (384)

**Retrieval Metrics**
- `hit_rate@k`: Hit rate at k (1, 3, 5, 10)
- `mrr`: Mean Reciprocal Rank
- `map@10`: Mean Average Precision at 10
- `ndcg@10`: Normalized Discounted Cumulative Gain at 10

**Training Metrics**
- `final_train_loss`: Final training loss
- `final_eval_loss`: Final evaluation loss
- `best_eval_loss`: Best evaluation loss
- `total_training_steps`: Total steps completed

**Generation Metrics**
- `bleu`: BLEU score
- `rouge1`, `rouge2`, `rougeL`: ROUGE scores
- `bertscore_precision`: BERTScore precision
- `bertscore_recall`: BERTScore recall
- `bertscore_f1`: BERTScore F1
- `perplexity`: Language model perplexity

**Performance Metrics**
- `avg_query_time_ms`: Average query time
- `avg_retrieval_time_s`: Average retrieval time
- `avg_generation_time_s`: Average generation time
- `avg_gpu_memory_gb`: Average GPU memory usage
- `avg_cpu_percent`: Average CPU usage

### Project Requirements

#### Hardware Requirements
- **GPU**: NVIDIA GPU with at least 12GB VRAM (for training)
- **RAM**: 16GB minimum, 32GB recommended
- **Storage**: 50GB free space

#### Software Requirements
- **OS**: Ubuntu 22.04+ / Windows WSL2 / macOS
- **Python**: 3.10 or higher
- **CUDA**: 12.1+ with cuDNN 8.9+
- **Docker**: 24.0+ (for container deployment)
- **AWS CLI**: 2.0+ (for AWS deployment)

#### API Tokens Required
- DagsHub account and token
- Hugging Face account and token (optional)
- AWS account with appropriate permissions (for deployment)

### Detailed Setup

#### 1. Environment Setup

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Configure credentials
cp .env.example .env
nano .env  # Add your DagsHub token

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
pip install -r requirements_dev.txt

# Download data
kaggle datasets download stackoverflow/stacksample
unzip stacksample.zip
mkdir -p data/download
mv Questions.csv Answers.csv data/download/

# Install DVC with S3 support
pip install dvc dvc-s3
```

#### 2. CUDA Setup (for GPU)

```bash
# Check NVIDIA driver version
nvidia-smi

# If driver is too old, run CUDA setup script
chmod +x setup_cuda.sh
sudo ./setup_cuda.sh

# Verify PyTorch CUDA
python -c "import torch; print(torch.cuda.is_available())"
```

#### 3. DagsHub Configuration

```bash
# Login to DagsHub
export DAGSHUB_USERNAME=fourapiwit
export DAGSHUB_TOKEN=your_token_here

# Configure DVC remote
dvc remote add -d storage https://dagshub.com/fourapiwit/llmops-rag-pipeline.dvc
dvc remote modify storage --local auth basic
dvc remote modify storage --local user $DAGSHUB_USERNAME
dvc remote modify storage --local password $DAGSHUB_TOKEN

# Configure MLflow
export MLFLOW_TRACKING_URI=https://dagshub.com/fourapiwit/llmops-rag-pipeline.mlflow
export MLFLOW_TRACKING_USERNAME=$DAGSHUB_USERNAME
export MLFLOW_TRACKING_PASSWORD=$DAGSHUB_TOKEN
```

#### 4. Hugging Face Setup (Optional)

```bash
# Login to Hugging Face
huggingface-cli login
# Enter your HF token when prompted

# Set environment variable
export HF_TOKEN=hf_your_token_here
```

### Configuration

### params.yaml Structure

**Example of params.yaml Structure**
```yaml
data:
  questions_path: "data/download/Questions.csv"
  answers_path: "data/download/Answers.csv"
  score_threshold: 5          # Filter answers with score > 5
  top_n_questions: 50         # Number of top questions to use
  train_ratio: 0.70
  val_ratio: 0.15
  test_ratio: 0.15

vectorstore:
  chunk_size: 512             # Document chunk size in tokens
  chunk_overlap: 64           # Overlap between chunks
  embedding_model: "BAAI/bge-small-en-v1.5"
  retrieval_k: 10             # Number of chunks to retrieve

model:
  base_model_name: "Qwen/Qwen3-0.6B"
  lora_r: 16                  # LoRA rank
  lora_alpha: 32              # LoRA alpha
  lora_dropout: 0.1
  quantization_4bit: true

training:
  epochs: 1                   # Number of training epochs
  batch_size: 4
  learning_rate: 5.0e-5
  dpo_beta: 0.1              # DPO temperature parameter
  max_seq_length: 512

inference:
  max_new_tokens: 512
  temperature: 0.7
  top_p: 0.95
  batch_size: 8

monitoring:
  mlflow_tracking_uri: "https://dagshub.com/fourapiwit/llmops-rag-pipeline.mlflow"
  mlflow_experiment_name: "llmops-rag-pipeline-experiment"
```

### Running the Pipeline

#### Full Pipeline Execution

```bash
# Run entire pipeline with MLflow tracking
python scripts/run_pipeline.py

# Run with specific stages skipped
python scripts/run_pipeline.py --skip-stages fine_tune_model evaluate_rag

# Initialize DVC and DagsHub tracking
python scripts/run_pipeline.py --setup-dvc --setup-dagshub
```

#### Individual Stage Execution

```bash
# Data pipeline stages
python -m src.data.load_data
python -m src.data.preprocess
python -m src.data.split_data

# Vector database
python -m src.vectorstore.create_vectorstore
python -m src.vectorstore.visualize_vectorstore
python -m src.vectorstore.retrieval_eval

# Model fine-tuning
python -m src.model.fine_tune

# RAG evaluation
python -m src.evaluation.rag_evaluation
```

#### DVC Pipeline Commands

```bash
# Show pipeline DAG
dvc dag

# Reproduce specific stage
dvc repro create_vectorstore
dvc repro fine_tune_model

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

### Monitoring Experiments

#### DagsHub Web Interface

1. Navigate to: `https://dagshub.com/fourapiwit/llmops-rag-pipeline`
2. Click on **"Experiments"** tab to view MLflow runs
3. Click on **"Metrics"** tab to view DVC metrics
4. Click on **"Data"** tab to view versioned datasets

#### Local MLflow UI

```bash
# Start MLflow server with local backend
mlflow ui --backend-store-uri ./mlruns

# Or with DagsHub remote (read-only)
mlflow ui --backend-store-uri $MLFLOW_TRACKING_URI
```

#### View Specific Metrics

```bash
# Via script
python scripts/monitor_experiment.py --experiment llmops-rag-pipeline-experiment

# Compare runs
python scripts/monitor_experiment.py --compare --metric bleu

# Follow live updates
python scripts/monitor_experiment.py --follow
```

### Testing

#### Run All Tests

```bash
# Run with coverage
pytest tests/ -v --cov=src --cov-report=html

# Run specific test file
pytest tests/test_retrieval.py -v

# Run with verbose output
pytest tests/ -v -s
```

#### Test Categories

| Test File | Description |
|-----------|-------------|
| `test_data.py` | Data loading, filtering, and preprocessing |
| `test_vectorstore.py` | Vector database creation and embedding |
| `test_retrieval.py` | Retrieval metrics (hit rate, MRR, MAP, nDCG) |
| `test_model.py` | Model configuration and quantization |
| `test_rag_pipeline.py` | RAG pipeline components |
| `test_evaluation.py` | Generation metrics (BLEU, ROUGE, BERTScore) |

#### Generate Coverage Report

```bash
pytest tests/ --cov=src --cov-report=html
open htmlcov/index.html
```

### Docker Deployment

#### Local Docker Build and Run

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

#### Docker Compose Services

| Service | Port | Description |
|---------|------|-------------|
| `rag-api` | 8000 | FastAPI RAG service |
| `mlflow-server` | 5001 | MLflow tracking UI |
| `dvc-server` | - | DVC metrics viewer |

#### Docker Environment Variables

```bash
# Required environment variables
CUDA_VISIBLE_DEVICES=0
MLFLOW_TRACKING_URI=https://dagshub.com/fourapiwit/llmops-rag-pipeline.mlflow
MLFLOW_TRACKING_USERNAME=fourapiwit
MLFLOW_TRACKING_PASSWORD=your_token
HF_TOKEN=hf_your_token
```

### AWS Production Deployment

#### Prerequisites for AWS

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

#### Deployment Steps

1. **Set AWS Environment Variables**

```bash
export AWS_ACCESS_KEY_ID=your_access_key
export AWS_SECRET_ACCESS_KEY=your_secret_key
export AWS_DEFAULT_REGION=us-east-1
```

2. **Run Deployment Script**

```bash
chmod +x deployment/aws_deploy.sh
./deployment/aws_deploy.sh
```

3. **Or Use Python Deployment Script**

```bash
python scripts/deploy_aws.py \
  --repository llmops-rag \
  --cluster llmops-cluster \
  --service llmops-rag-service
```

4. **Monitor Deployment**

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

#### AWS Architecture

```
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

#### Required IAM Roles

- `ecsTaskExecutionRole`: For ECS to pull images from ECR
- `ecsTaskRole`: For container to access AWS services

### API Usage

#### Base URL

- Local: `http://localhost:8000`
- AWS: `http://<ec2-public-ip>:8000`

### Endpoints

**Health Check**

```bash
curl -X GET http://localhost:8000/health
```

Response:
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

Response:
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

Response:
```json
{
  "answers": ["def sort_list(lst):\n    return sorted(lst)", "import csv\nwith open('file.csv') as f:\n    reader = csv.reader(f)"],
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

### Troubleshooting

**Common Issues and Solutions**

| Issue | Solution |
|-------|----------|
| `ModuleNotFoundError: No module named 'src'` | Run from project root directory, use `python -m` |
| `CUDA out of memory` | Reduce `batch_size` in params.yaml |
| `ChromaDB index corruption` | Delete `chroma_db/` and recreate: `rm -rf chroma_db/` |
| `MLflow 403 error` | Check DagsHub token is valid and has correct permissions |
| `DVC push fails` | Run `dvc remote modify storage --local auth basic` |
| `Git push 500 error` | Remove large files from git history (use DVC instead) |

**Debug Mode**

```bash
# Enable debug logging
export LOG_LEVEL=DEBUG
python scripts/run_pipeline.py

# Run single stage with verbose output
python -m src.model.fine_tune 2>&1 | tee training.log
```

### License

MIT License - see LICENSE file for details.