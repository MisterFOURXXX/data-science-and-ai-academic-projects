### Fine Tuning LLM (LLMOps Pipeline)

A production-ready LLMOps pipeline for fine-tuning Qwen3-0.6B with Direct Preference Optimization (DPO) and deploying to AWS. This pipeline includes experiment tracking with MLflow, data version control with DVC, containerization with Docker, and comprehensive model evaluation.

**Table of Contents**

- [Features](#features)
- [Prerequisites](#prerequisites)
- [Project Structure](#project-structure)
- [Quick Start](#quick-start)
- [Detailed Setup Guide](#detailed-setup-guide)
  - [Step 1: Environment Setup](#step-1-environment-setup)
  - [Step 2: Install Dependencies](#step-2-install-dependencies)
  - [Step 3: Configure DagsHub and MLflow](#step-3-configure-dagshub-and-mlflow)
  - [Step 4: Prepare Data](#step-4-prepare-data)
  - [Step 5: Run Preprocessing](#step-5-run-preprocessing)
  - [Step 6: Log Hyperparameters (Manual)](#step-6-log-hyperparameters-manual)
  - [Step 7: Train Model](#step-7-train-model)
  - [Step 8: Evaluate Model](#step-8-evaluate-model)
  - [Step 9: View MLflow Dashboard](#step-9-view-mlflow-dashboard)
- [Docker Deployment](#docker-deployment)
- [AWS Deployment](#aws-deployment)
- [Testing](#testing)
- [Environment Variables](#environment-variables)
- [Monitoring and Logging](#monitoring-and-logging)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)

**Features**

- **DPO Fine-tuning with QLoRA**: Efficient 4-bit quantization + LoRA adapters
- **Experiment Tracking**: MLflow with DagsHub integration
- **Data Version Control**: DVC for dataset versioning
- **Hyperparameter Logging**: Manual parameter logging (no Optuna optimization)
- **Comprehensive Metrics**: BLEU, ROUGE, BERTScore, perplexity, code quality metrics
- **Multi-language Support**: Python, JavaScript, Java, C++, Go, Rust, SQL detection
- **Docker Containerization**: Easy deployment with Docker Compose
- **AWS Deployment**: ECR, Lambda, API Gateway with Terraform
- **Integration Tests**: Pytest with coverage reporting

**Hardware Requirements**

- **Hardware**: NVIDIA GPU with at least 8GB VRAM (RTX A4000 recommended)
- **CUDA**: 12.4 or higher
- **Python**: 3.11 or higher
- **Docker**: 24.0 or higher (for container deployment)
- **AWS CLI**: Configured with appropriate credentials (for AWS deployment)
- **DagsHub Account**: For experiment tracking and DVC remote storage

### Project Structure
```
llmops-rag-pipeline/
├── .gitignore                   # Git ignore rules
├── dvc.yaml                     # DVC pipeline stages
├── requirements.txt             # Python dependencies
├── Dockerfile                   # Docker build instructions
├── docker-compose.yml           # Docker Compose configuration
├── setup_env.sh                 # CUDA and environment setup script
├── config/
│   ├── config.yaml              # Main configuration file
│   └── params.yaml              # Hyperparameter search space (for logging)
├── data/
│   └── raw/                     # Raw CSV files (Questions.csv, Answers.csv, Tags.csv)
├── models/                      # Trained model artifacts
├── notebooks/                   # Jupyter notebooks for development
├── src/                         # Source code
│   ├── __init__.py
│   ├── data/                    # Data processing modules
│   │   ├── __init__.py
│   │   ├── preprocessing.py
│   │   └── dataset_builder.py
│   ├── models/                  # Model training and evaluation
│   │   ├── __init__.py
│   │   ├── model_trainer.py
│   │   └── model_evaluator.py
│   ├── utils/                   # Utility functions and metrics
│   │   ├── __init__.py
│   │   ├── metrics.py
│   │   └── helpers.py
│   └── tracking/                # MLflow tracking utilities
│       ├── __init__.py
│       ├── mlflow_tracker.py
│       └── optuna_optimizer.py
├── tests/                       # Unit and integration tests
│   ├── __init__.py
│   ├── test_data.py
│   ├── test_model.py
│   └── test_api.py
├── deployment/                  # Deployment scripts and Terraform configs
│   ├── aws_deploy.py
│   ├── inference_api.py
│   └── terraform/
│       ├── main.tf
│       ├── variables.tf
│       └── outputs.tf
└── scripts/                     # Pipeline execution scripts
    ├── run_pipeline.py          # Main training pipeline
    ├── run_optimization.py      # Hyperparameter logging (manual)
    └── deploy.sh                # Deployment helper script
```

### Detailed Setup Guide

#### Model Experiment

**Step 1: Environment Setup**

First, run the CUDA environment setup script:

```bash
# Make script executable
chmod +x setup_env.sh

# Run the setup script
./setup_env.sh
```

The `setup_env.sh` script performs the following:
- Removes conflicting NVIDIA packages
- Updates system dependencies
- Cleans apt cache
- Removes conflicting deepspeed packages

**Step 2: Install Dependencies**

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Upgrade pip
pip install --upgrade pip

# Install requirements
pip install -r requirements.txt

# Install the package in editable mode
pip install -e .
```

**Step 3: Configure DagsHub and MLflow**

**3.1 Create a DagsHub Repository**

1. Go to [dagshub.com](https://dagshub.com) and sign up/log in
2. Create a new repository (e.g., `llmops-rag-pipeline`)
3. Note your username and repository name

**3.2 Get DagsHub Token**

1. Go to Settings → Tokens
2. Generate a new token with write permissions
3. Save the token securely

**3.3 Set Environment Variables**

```bash
# Set DagsHub authentication
export DAGSHUB_USER="your-dagshub-username"
export DAGSHUB_TOKEN="your-dagshub-token"

# Set MLflow tracking URI
export MLFLOW_TRACKING_URI="https://dagshub.com/${DAGSHUB_USER}/${DAGSHUB_REPO}.mlflow"

# Optional: Save to .env file for persistence
echo "DAGSHUB_USER=${DAGSHUB_USER}" >> .env
echo "DAGSHUB_TOKEN=${DAGSHUB_TOKEN}" >> .env
echo "MLFLOW_TRACKING_URI=${MLFLOW_TRACKING_URI}" >> .env
```

**3.4 Login to Hugging Face**

```bash
# Install huggingface hub CLI
pip install huggingface-hub

# Login (you'll need an HF token from https://huggingface.co/settings/tokens)
hf login
```

**3.5 Configure DVC Remote**

```bash
# Initialize DVC (if not already done)
dvc init

# Add DagsHub DVC remote
dvc remote add origin https://dagshub.com/${DAGSHUB_USER}/${DAGSHUB_REPO}.dvc

# Set authentication for the remote
dvc remote modify origin --local auth basic
dvc remote modify origin --local user ${DAGSHUB_USER}
dvc remote modify origin --local password ${DAGSHUB_TOKEN}

# Set as default remote
dvc remote default origin

# Commit DVC config
git add .dvc/config .dvcignore .gitignore
git commit -m "Configure DVC with DagsHub remote"
```

**Step 4: Prepare Data**

**4.1 Download Stack Overflow Dataset**

Download the Stack Overflow dataset from Kaggle:

```bash
# Install kaggle CLI
pip install kaggle

# Configure Kaggle API (place kaggle.json in ~/.kaggle/)
mkdir -p ~/.kaggle
cp /path/to/kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json

# Download dataset
kaggle datasets download stackoverflow/stacksample

# Extract files
unzip stacksample.zip -d data/raw/
```

**4.2 Verify Data Files**

Ensure the following files exist in `data/raw/`:
- `Questions.csv` (~1.92 GB)
- `Answers.csv` (~1.61 GB)
- `Tags.csv`

**4.3 Track Data with DVC**

```bash
# Add raw data directory to DVC
dvc add data/raw

# Commit DVC tracking files
git add data/raw.dvc .gitignore
git commit -m "Track raw data with DVC"

# Push data to DagsHub remote
dvc push -r origin
```

**Step 5: Run Preprocessing**

```bash
# Run preprocessing stage
dvc repro preprocess

# Or run directly
python scripts/run_preprocessing.py
```

**What preprocessing does:**
1. Cleans HTML tags from question and answer bodies
2. Filters questions with score > 5
3. Creates DPO preference pairs (chosen = highest score answer, rejected = lowest score answer)
4. Splits data into train/validation/test (80/10/10)
5. Converts to conversational format for the model
6. Saves processed datasets to `data/processed/`

**Expected output:**
```
Building datasets from CSV files...
Loading questions from data/raw/Questions.csv
Loading answers from data/raw/Answers.csv
Sampling top 1000 questions...
Cleaning HTML content...
Joining questions and answers...
Creating DPO dataset...
Created 971 DPO examples
Splitting dataset...
Converting to conversational format...
Train samples: 776
Validation samples: 97
Test samples: 98
Preprocessing completed successfully
```

**Step 6: Log Hyperparameters For Experiments Tracking (Manual)**

Instead of running computationally expensive Optuna optimization, we manually log all hyperparameters for experiment tracking:

```bash
# Run hyperparameter logging
dvc repro optimize

# Or run directly
python scripts/run_optimization.py
```

**What the optimization script does now (manual logging):**
1. Logs all configuration parameters from `config/config.yaml`
2. Logs the search space from `config/params.yaml`
3. Logs current training parameters
4. Saves everything to `metrics/optimization_results.json`
5. Creates an MLflow run with all parameters logged
6. **Does NOT perform actual optimization or model training**

**Expected output:**
```
============================================================
HYPERPARAMETER LOGGING (No Optimization)
============================================================

[Logging Configuration Parameters]
  config.project.name: llmops-rag-pipeline
  config.project.version: 1.0.0
  ...

[Logging Search Space Parameters]
  learning_rate: min=1e-06, max=0.0001, type=loguniform
  beta: min=0.01, max=0.5, type=uniform
  lora_r: min=4, max=32, type=int
  ...

[Logging Current Training Parameters]
  current.learning_rate: 5e-05
  current.beta: 0.1
  current.lora_r: 16
  ...

[Parameters saved to metrics/optimization_results.json]

============================================================
PARAMETER LOGGING COMPLETED
============================================================
```

**Step 7: Train Model**

```bash
# Run training pipeline
dvc repro train

# Or run directly
python scripts/run_pipeline.py
```

**What training does:**
1. Loads processed datasets from `data/processed/`
2. Loads Qwen3-0.6B model with 4-bit quantization (QLoRA)
3. Applies LoRA adapters to specific modules
4. Trains with DPO loss for preference alignment
5. Saves fine-tuned model to `models/qwen-dpo-final/`
6. Logs all metrics to MLflow

**Key training parameters (configurable in `config/config.yaml`):**
- `num_train_epochs`: 30
- `learning_rate`: 5e-5
- `per_device_train_batch_size`: 2
- `gradient_accumulation_steps`: 8
- `beta`: 0.1 (DPO temperature)
- `max_length`: 1024

**Training progress monitoring:**
```bash
# Monitor GPU usage in another terminal
watch -n 1 nvidia-smi

# View MLflow logs
mlflow ui --backend-store-uri sqlite:///mlruns/mlflow.db --port 5000
```

**Step 8: Evaluate Model((

```bash
# Run evaluation
dvc repro evaluate

# Or run directly
python scripts/run_evaluation.py
```

**What evaluation does:**
1. Loads the fine-tuned model and test dataset
2. Generates responses for test prompts
3. Computes similarity metrics (BLEU, ROUGE, BERTScore)
4. Calculates perplexity
5. Analyzes code quality (functions, imports, comments)
6. Detects programming languages in generated code
7. Saves metrics to `metrics/evaluation_metrics.json`
8. Creates a metrics comparison plot

**Expected output:**
```
============================================================
EVALUATION RESULTS
============================================================
bleu: 0.0303
rouge1: 0.2324
rouge2: 0.0447
rougeL: 0.1218
bertscore_f1: 0.7956
perplexity: 2.8343
has_thinking_ratio: 0.5
has_solution_ratio: 0.4
has_explanation_ratio: 0.5
code_presence_ratio: 0.3
avg_comment_ratio: 0.4
has_imports_ratio: 0.2000
has_functions_ratio: 0.4000
avg_function_count: 0.4000
avg_line_count: 36.0000
valid_samples_ratio: 1.0000

Language Distribution:
  no_code: 1

============================================================
PIPELINE COMPLETED SUCCESSFULLY
============================================================
```

**Step 9: View MLflow Dashboard**

```bash
# Launch MLflow UI
mlflow ui --backend-store-uri sqlite:///mlruns/mlflow.db --port 5000

# Open in browser
# http://localhost:5000
```

**What you can see in MLflow:**
- **Parameters**: All hyperparameters used in training
- **Metrics**: Loss, rewards/accuracies, BLEU, ROUGE, perplexity, etc.
- **Artifacts**: Model files, evaluation JSON, plots
- **Tags**: Run name, stage, version
- **Search**: Filter runs by parameters or metrics
- **Compare**: Compare multiple runs side-by-side

### Docker Deployment

**Build and Run Locally**

```bash
# Build Docker image
docker build -t llmops-rag-model:latest .

# Run with Docker Compose
docker-compose up inference

# Or run directly
docker run --gpus all -p 8000:8000 llmops-rag-model:latest
```

**Test the API**

```bash
# Health check
curl http://localhost:8000/health

# Generate response
curl -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "How to reverse a list in Python?",
    "max_new_tokens": 512,
    "temperature": 0.7,
    "top_p": 0.95,
    "top_k": 50
  }'
```

**Stop Containers**

```bash
# Stop all services
docker-compose down

# Remove volumes
docker-compose down -v
```

### AWS Deployment

**Setup AWS Requirements**

```bash
# Install AWS CLI
pip install awscli

# Configure AWS credentials
aws configure
# Enter: AWS Access Key ID, Secret Access Key, region (us-east-1)

# Create IAM role for Lambda (or use existing)
# Role ARN needed for deployment
```

**Deploy with Python Script**

```bash
# Set AWS credentials
export AWS_ACCESS_KEY_ID="your-access-key"
export AWS_SECRET_ACCESS_KEY="your-secret-key"
export AWS_DEFAULT_REGION="us-east-1"
export LAMBDA_ROLE_ARN="arn:aws:iam::account-id:role/lambda-execution-role"

# Run deployment script
python deployment/aws_deploy.py
```

**Deploy with Terraform**

```bash
# Navigate to terraform directory
cd deployment/terraform

# Initialize Terraform
terraform init

# Plan deployment
terraform plan

# Apply deployment
terraform apply -auto-approve

# Get outputs
terraform output

# Destroy resources when done
terraform destroy
```

**API Gateway Endpoint**

After deployment, you'll receive an API Gateway URL:
```
https://{api-id}.execute-api.{region}.amazonaws.com/prod/generate
```

### Testing

**Run All Tests**

```bash
# Run all tests with coverage
pytest tests/ -v --cov=src --cov-report=html

# Open coverage report
open htmlcov/index.html
```

**Run Specific Tests**

```bash
# Test data preprocessing
pytest tests/test_data.py -v

# Test model loading and generation
pytest tests/test_model.py -v

# Test API endpoints
pytest tests/test_api.py -v
```

**Test with Sample Data**

```bash
# Run a quick test with minimal data
python -c "
from src.data.preprocessing import clean_html
print(clean_html('<p>Hello <b>World</b></p>'))
# Expected: 'Hello World'
"
```

### Environment Variables

Create a `.env` file for persistent configuration:

```bash
# DagsHub Configuration
DAGSHUB_USER=your-username
DAGSHUB_TOKEN=your-token
MLFLOW_TRACKING_URI=https://dagshub.com/your-username/your-repo.mlflow

# HuggingFace Configuration
HF_TOKEN=hf_your-token

# AWS Configuration
AWS_ACCESS_KEY_ID=your-access-key
AWS_SECRET_ACCESS_KEY=your-secret-key
AWS_DEFAULT_REGION=us-east-1
LAMBDA_ROLE_ARN=arn:aws:iam::account-id:role/lambda-execution-role

# Local Paths
MODEL_PATH=./models/qwen-dpo-final
DATA_PATH=./data/raw
OUTPUT_PATH=./models

# MLflow Configuration
MLFLOW_EXPERIMENT_NAME=llmops-rag-pipeline
```

Load environment variables:
```bash
source .env
# Or use python-dotenv
```

### Monitoring and Logging

**MLflow Metrics Tracked**

**Training Metrics:**
- `loss`: Cross-entropy loss
- `rewards/accuracies`: Preference accuracy
- `rewards/margins`: Margin between chosen and rejected
- `grad_norm`: Gradient norm
- `learning_rate`: Current learning rate
- `entropy`: Model entropy

**Evaluation Metrics:**
- `bleu`: BLEU score for generated text
- `rouge1`, `rouge2`, `rougeL`: ROUGE scores
- `bertscore_f1`: BERTScore F1
- `perplexity`: Model perplexity

**Code Quality Metrics:**
- `has_functions_ratio`: Percentage with functions
- `has_imports_ratio`: Percentage with imports
- `has_comments_ratio`: Percentage with comments
- `avg_function_count`: Average number of functions
- `avg_line_count`: Average lines of code

**View MLflow Dashboard**

```bash
# Local tracking
mlflow ui --backend-store-uri sqlite:///mlruns/mlflow.db --port 5000

# DagsHub tracking (if configured)
# Visit: https://dagshub.com/{username}/{repo}.mlflow
```

**View DVC Pipeline Status**

```bash
# Show pipeline DAG
dvc dag

# Show pipeline status
dvc status

# Show metrics
dvc metrics show

# Show plots
dvc plots show
```

### Troubleshooting

**DagsHub Authentication Issues**

```bash
# Reconfigure DVC remote
dvc remote remove origin
dvc remote add origin https://dagshub.com/${DAGSHUB_USER}/${DAGSHUB_REPO}.dvc
dvc remote modify origin --local auth basic
dvc remote modify origin --local user ${DAGSHUB_USER}
dvc remote modify origin --local password ${DAGSHUB_TOKEN}

# Test connection
dvc pull -r origin
```

**MLflow Permission Issues**

```bash
# Use local tracking instead of remote
export MLFLOW_TRACKING_URI="file:./mlruns"

# Or create a new experiment
mlflow experiments create -n llmops-rag-pipeline
```

**Out of Memory (OOM)**

```bash
# Reduce batch size in config/config.yaml
per_device_train_batch_size: 1
gradient_accumulation_steps: 16

# Reduce max length
max_length: 512

# Enable gradient checkpointing (already enabled)
gradient_checkpointing: true
```

**Data Loading Issues**

```bash
# Verify CSV files exist
ls -la data/raw/

# Check CSV encoding
file data/raw/Questions.csv

# Re-run preprocessing with smaller sample
# In config/config.yaml, set:
sample_size: 100  # instead of 1000
```

**Note**: This pipeline is designed for educational and research purposes. For production use, ensure proper security, monitoring, and scaling configurations.