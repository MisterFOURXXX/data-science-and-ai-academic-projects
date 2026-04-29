## Hospital Stay Prediction - Optimized Production Pipeline

A complete MLOps pipeline for predicting patient hospital stay duration using stacked ensemble learning with LightGBM, XGBoost, and CatBoost, featuring experiment tracking with MLflow on DagsHub, hyperparameter optimization with Optuna, data version control with DVC, containerization with Docker, and production deployment on AWS ECS.

### Project Overview

This project implements a production-ready machine learning pipeline that:
- Predicts patient stay categories using hospital admission data
- Uses stacked ensemble of three gradient boosting models (LightGBM, XGBoost, CatBoost)
- Tracks every experiment, trial, and metric with MLflow on DagsHub
- Version controls data and models with DVC
- Runs automated tests for pipeline validation
- Deploys as a containerized FastAPI service on AWS ECS

### Table of Contents

1. [Prerequisites](#prerequisites)
2. [Initial Setup](#initial-setup)
3. [Configuration](#configuration)
4. [Running Experiments](#running-experiments)
5. [Monitoring with MLflow](#monitoring-with-mlflow)
6. [Testing](#testing)
7. [Docker Containerization](#docker-containerization)
8. [AWS Deployment](#aws-deployment)
9. [API Usage](#api-usage)
10. [Troubleshooting](#troubleshooting)

### Requirements

- Python 3.11 or higher
- Git
- Docker (for local containerization)
- AWS CLI (for deployment)
- AWS account with ECR and ECS access
- DagsHub account (free tier available)

### Project Structure Summary

```
hospital-stay-prediction/
├── data/
│   ├── raw/                 # Input data (train_data.csv)
│   └── processed/          # Processed numpy arrays
├── src/
│   ├── data/               # Data preparation
│   ├── features/           # Feature engineering
│   ├── models/             # Training and tuning
│   ├── evaluation/         # Metrics calculation
│   ├── deployment/         # FastAPI application
│   └── utils/              # MLflow utilities
├── models/                 # Saved models (DVC tracked)
├── metrics/                # Metrics JSON files (DVC tracked)
├── tests/                  # Pytest suite
├── dvc.yaml               # DVC pipeline stages
├── params.yaml            # Configuration parameters
├── requirements.txt       # Python dependencies
├── Dockerfile             # Container definition
├── .env.example           # Environment template
├── .gitignore             # Git ignore rules
└── README.md              # This file
```

### Initial Setup

**1. Create DagsHub Repository**

1. Go to [https://dagshub.com](https://dagshub.com) and sign up/login
2. Create a new repository named `hospital-stay-prediction`
3. Note your:
   - Username (e.g., `your-dagshub-username`)
   - Repository name (`hospital-stay-prediction`)

**2. Clone Repository and Add Data**

```bash
# Clone the repository
git clone https://dagshub.com/YOUR_DAGSHUB_USERNAME/hospital-stay-prediction.git
cd hospital-stay-prediction

# Create data directory structure
mkdir -p data/raw data/processed models metrics logs

# Download the dataset and place it in the raw directory
# Replace with your actual dataset path
cp /path/to/train_data.csv data/raw/
```

**3. Configure DagsHub Authentication**

Create a `.env` file from the template:

```bash
cp .env.example .env
```

Edit `.env` with your DagsHub credentials:

```bash
DAGSHUB_USER=your-dagshub-username
DAGSHUB_REPO=hospital-stay-prediction
```

**4. Update MLflow Configuration**

Edit `src/utils/mlflow_utils.py` with your DagsHub credentials:

```python
def init_mlflow():
    dagshub.init(repo_owner="YOUR_DAGSHUB_USERNAME", repo_name="hospital-stay-prediction")
    mlflow.set_tracking_uri(f"https://dagshub.com/YOUR_DAGSHUB_USERNAME/hospital-stay-prediction.mlflow")
    mlflow.set_experiment("hospital-stay-stacking")
```

**5. Install Dependencies**

```bash
# Install Python dependencies
pip install -r requirements.txt

# Initialize DVC
dvc init

# Configure DVC remote for DagsHub
dvc remote add origin https://dagshub.com/YOUR_DAGSHUB_USERNAME/hospital-stay-prediction.dvc
dvc remote modify origin --local auth basic

# If you need to reconnect to DagsHub
dvc remote add -d origin https://dagshub.com/YOUR_DAGSHUB_USERNAME/hospital-stay-prediction.dvc
dvc remote modify origin --projecturl https://dagshub.com/YOUR_DAGSHUB_USERNAME/hospital-stay-prediction
dvc remote modify origin auth basic
dvc remote modify origin user YOUR_DAGSHUB_USERNAME
dvc remote modify origin password YOUR_DAGSHUB_TOKEN
```

**6. Verify Installation**

```bash
# Check DVC configuration
dvc remote list

# Check MLflow configuration
python -c "from src.utils.mlflow_utils import init_mlflow; init_mlflow(); print('MLflow configured successfully')"

# Verify dataset exists
ls -la data/raw/
```

### Configuration

**Pipeline Parameters (`params.yaml`)**

```yaml
data:
  sample_size: 2000        # Number of samples to use (set to -1 for full dataset)
  test_size: 0.2          # Test split ratio
  random_state: 42        # Reproducibility seed

preprocess:
  pca_variance: 0.95      # PCA variance retention (95% of variance)

models:
  n_trials: 5             # Number of Optuna trials per model (increase for better results)
  cv: 3                   # Cross-validation folds
  random_state: 42        # Reproducibility seed
```

**Customization Tips:**
- Increase `n_trials` to 20-50 for production-grade models
- Set `sample_size: -1` to use all available data
- Adjust `pca_variance` between 0.90-0.99 based on dimensionality reduction needs

## Running Experiments

**Run Complete Pipeline**

```bash
# Run all stages (prepare, preprocess, tune, train, evaluate, test)
dvc repro

# Or run specific stages
dvc repro prepare
dvc repro preprocess
dvc repro tune_weak
dvc repro train_final
dvc repro evaluate
dvc repro test
```

**Understanding Pipeline Stages**

| Stage | Description | Key Outputs |
|-------|-------------|-------------|
| `prepare` | Splits data into train/test sets | `train.parquet`, `test.parquet` |
| `preprocess` | Applies transformations, PCA, SMOTE | `X_train.npy`, `y_train.npy`, `preprocessor.pkl` |
| `tune_weak` | Optimizes hyperparameters for 3 models | `best_weak_params.json` |
| `train_final` | Trains stacking ensemble | `final_stacking_model.pkl` |
| `evaluate` | Calculates test metrics | `evaluation.json` |
| `test` | Runs pytest validation | `test_report.xml` |

**What Gets Logged to MLflow**

Each Optuna trial logs:
- Model parameters (learning_rate, max_depth, n_estimators, etc.)
- Mean macro F1 score from cross-validation

Final stacking model logs:
- Complete ensemble configuration
- CV parameters
- Meta-learner parameters

Evaluation logs:
- Accuracy
- Precision (macro)
- Recall (macro)
- F1 Score (macro)
- AUC-ROC (macro)

**Resume Failed Runs**

If a stage fails, fix the issue and resume:

```bash
# Resume from the last successful stage
dvc repro --downstream

# Force rerun a specific stage
dvc repro -f tune_weak
```

### Monitoring with MLflow

**View Experiments**

1. Navigate to: `https://dagshub.com/YOUR_DAGSHUB_USERNAME/hospital-stay-prediction/experiments`

2. You'll see:
   - **LightGBM trials** - All hyperparameter combinations tested
   - **XGBoost trials** - All XGBoost optimization runs
   - **CatBoost trials** - All CatBoost optimization runs
   - **final_stacking** - The final ensemble model run
   - **final_evaluation** - Test metrics run

**Compare Runs**

1. Click on the experiment name
2. Select multiple runs using checkboxes
3. Click "Compare" to see side-by-side parameter and metric comparison

**Access Run Details**

Each run contains:
- **Parameters** - Hyperparameters used
- **Metrics** - Performance scores
- **Artifacts** - Saved models (for final run)
- **Tags** - Metadata for organization

**MLflow UI Locally (Optional)**

```bash
# If you want to view locally instead of DagsHub
mlflow ui --backend-store-uri sqlite:///mlruns/mlflow.db --port 5000
```

### Testing

**Run All Tests**

```bash
# Run tests as part of DVC pipeline (automatically runs after training)
dvc repro test

# Run tests independently
pytest tests/ -v

# Run with coverage report
pytest tests/ --cov=src --cov-report=html
```

**Test Coverage**

The test suite validates:
- Data shapes after preprocessing
- Preprocessor functionality
- Model prediction capabilities

**Add Custom Tests**

Add new test functions to `tests/test_pipeline.py`:

```python
def test_model_performance():
    # Load metrics
    with open("metrics/evaluation.json", "r") as f:
        metrics = json.load(f)
    # Assert minimum performance
    assert metrics["f1_macro"] > 0.7
```

### Docker Containerization

**Build Docker Image**

```bash
# Pull the final model from DVC (if not already available)
dvc pull

# Build the Docker image
docker build -t hospital-stay-predictor:latest .

# Verify image was created
docker images | grep hospital-stay-predictor
```

**Run Locally**

```bash
# Run container with port mapping
docker run -p 8000:8000 hospital-stay-predictor:latest

# With custom port mapping (host:container)
docker run -p 8080:8000 hospital-stay-predictor:latest
```

**Test Local Container**

```bash
# Health check
curl http://localhost:8000/docs

# Test prediction endpoint
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "Hospital_code": 1,
    "Hospital_type_code": "A",
    "Hospital_region_code": "Region1",
    "Available_Extra_Rooms_in_Hospital": 10,
    "Department": "General",
    "Ward_Type": "General",
    "Bed_Grade": 2.0,
    "Type_of_Admission": "Emergency",
    "Severity_of_Illness": "Moderate",
    "Visitors_with_Patient": 2,
    "Age": "40-50",
    "Admission_Deposit": 5000.0
  }'
```

**Docker Commands Reference**

| Command | Description |
|---------|-------------|
| `docker build -t hospital-stay-predictor .` | Build image |
| `docker run -p 8000:8000 hospital-stay-predictor` | Run container |
| `docker ps` | List running containers |
| `docker stop <container_id>` | Stop container |
| `docker rm <container_id>` | Remove container |
| `docker rmi hospital-stay-predictor` | Remove image |

**Troubleshooting Docker Issues**

**Issue:** `docker: command not found`
```bash
# Check if Docker is installed
docker --version

# Install Docker (Ubuntu/Debian)
sudo apt-get update
sudo apt-get install docker.io

# Add user to docker group (to run without sudo)
sudo usermod -aG docker $USER
newgrp docker
```

**Issue:** Docker-in-Docker (DinD) in cloud environments
- Cloud notebooks typically don't support Docker-in-Docker
- Use Option A: Build locally and push to ECR
- Use Option B: Use GitHub Actions or AWS CodeBuild for building

### AWS Deployment

**Prerequisites for AWS Deployment**

```bash
# Install and configure AWS CLI
pip install awscli
aws configure

# Enter your credentials:
AWS Access Key ID: YOUR_ACCESS_KEY
AWS Secret Access Key: YOUR_SECRET_KEY
Default region name: us-east-1
Default output format: json
```

**Step 1: Create ECR Repository**

```bash
# Create repository
aws ecr create-repository \
  --repository-name hospital-stay-predictor \
  --region us-east-1

# Note the repository URI from output (e.g., 123456789012.dkr.ecr.us-east-1.amazonaws.com/hospital-stay-predictor)
```

**Step 2: Authenticate Docker to ECR**

```bash
# Get login password and authenticate
aws ecr get-login-password --region us-east-1 | \
  docker login --username AWS --password-stdin \
  YOUR_AWS_ACCOUNT_ID.dkr.ecr.us-east-1.amazonaws.com
```

**Step 3: Tag and Push Docker Image**

```bash
# Tag the image
docker tag hospital-stay-predictor:latest \
  YOUR_AWS_ACCOUNT_ID.dkr.ecr.us-east-1.amazonaws.com/hospital-stay-predictor:latest

# Push to ECR
docker push YOUR_AWS_ACCOUNT_ID.dkr.ecr.us-east-1.amazonaws.com/hospital-stay-predictor:latest
```

**Step 4: Deploy to ECS Fargate**

**Option A: Using AWS Console (Easier)**

1. Go to **ECS Console** → **Clusters** → **Create Cluster**
   - Cluster name: `hospital-stay-cluster`
   - Infrastructure: AWS Fargate (serverless)
   - Create cluster

2. **Create Task Definition**
   - Task definition family: `hospital-stay-task`
   - Launch type: AWS Fargate
   - Task role: Create new role with ECR access
   - Network mode: awsvpc
   - Operating system: Linux
   - Task memory: 1GB
   - Task CPU: 0.5 vCPU

3. **Add Container**
   - Container name: `hospital-stay-api`
   - Image URI: `YOUR_AWS_ACCOUNT_ID.dkr.ecr.us-east-1.amazonaws.com/hospital-stay-predictor:latest`
   - Port mappings: 8000 TCP
   - Environment variables: Copy from `.env`

4. **Create Service**
   - Service name: `hospital-stay-service`
   - Number of tasks: 1
   - Security group: Allow inbound on port 8000
   - Load balancer: None (or create Application Load Balancer for production)

**Option B: Using AWS CLI**

```bash
# Create cluster
aws ecs create-cluster --cluster-name hospital-stay-cluster

# Register task definition
aws ecs register-task-definition \
  --family hospital-stay-task \
  --network-mode awsvpc \
  --requires-compatibilities FARGATE \
  --cpu "512" \
  --memory "1024" \
  --execution-role-arn arn:aws:iam::YOUR_ACCOUNT_ID:role/ecsTaskExecutionRole \
  --container-definitions '[
    {
      "name": "hospital-stay-api",
      "image": "YOUR_AWS_ACCOUNT_ID.dkr.ecr.us-east-1.amazonaws.com/hospital-stay-predictor:latest",
      "portMappings": [{"containerPort": 8000, "protocol": "tcp"}],
      "environment": [
        {"name": "DAGSHUB_USER", "value": "YOUR_DAGSHUB_USERNAME"},
        {"name": "DAGSHUB_REPO", "value": "hospital-stay-prediction"}
      ]
    }
  ]'

# Create service
aws ecs create-service \
  --cluster hospital-stay-cluster \
  --service-name hospital-stay-service \
  --task-definition hospital-stay-task \
  --launch-type FARGATE \
  --network-configuration "awsvpcConfiguration={subnets=[subnet-xxxxx],securityGroups=[sg-xxxxx],assignPublicIp=ENABLED}" \
  --desired-count 1
```

**Step 5: Access Deployed API**

After deployment, get the public IP or load balancer DNS:

```bash
# Get task public IP (if using public IP assignment)
aws ecs list-tasks --cluster hospital-stay-cluster
aws ecs describe-tasks --cluster hospital-stay-cluster --tasks TASK_ID

# Test the API
curl -X POST "http://PUBLIC_IP:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "Hospital_code": 1,
    "Hospital_type_code": "A",
    "Hospital_region_code": "Region1",
    "Available_Extra_Rooms_in_Hospital": 10,
    "Department": "General",
    "Ward_Type": "General",
    "Bed_Grade": 2.0,
    "Type_of_Admission": "Emergency",
    "Severity_of_Illness": "Moderate",
    "Visitors_with_Patient": 2,
    "Age": "40-50",
    "Admission_Deposit": 5000.0
  }'
```

**Step 6: Set Up Auto-scaling (Optional)**

```bash
# Register scalable target
aws application-autoscaling register-scalable-target \
  --service-namespace ecs \
  --scalable-dimension ecs:service:DesiredCount \
  --resource-id service/hospital-stay-cluster/hospital-stay-service \
  --min-capacity 1 \
  --max-capacity 5

# Define scaling policy
aws application-autoscaling put-scaling-policy \
  --service-namespace ecs \
  --scalable-dimension ecs:service:DesiredCount \
  --resource-id service/hospital-stay-cluster/hospital-stay-service \
  --policy-name cpu-scaling \
  --policy-type TargetTrackingScaling \
  --target-tracking-scaling-policy-configuration '{
    "TargetValue": 70.0,
    "PredefinedMetricSpecification": {"PredefinedMetricType": "ECSServiceAverageCPUUtilization"}
  }'
```

### API Usage

**Swagger Documentation**

Once deployed, access the interactive API documentation at:
- Local: `http://localhost:8000/docs`
- AWS: `http://PUBLIC_IP:8000/docs`

**Prediction Endpoint**

**POST** `/predict`

**Request Body:**
```json
{
  "Hospital_code": 1,
  "Hospital_type_code": "A",
  "Hospital_region_code": "Region1",
  "Available_Extra_Rooms_in_Hospital": 10,
  "Department": "General",
  "Ward_Type": "General",
  "Bed_Grade": 2.0,
  "Type_of_Admission": "Emergency",
  "Severity_of_Illness": "Moderate",
  "Visitors_with_Patient": 2,
  "Age": "40-50",
  "Admission_Deposit": 5000.0
}
```

**Response:**
```json
{
  "stay_category": "0-10 days"
}
```

**Python Client Example**

```python
import requests
import json

url = "http://localhost:8000/predict"
data = {
    "Hospital_code": 1,
    "Hospital_type_code": "A",
    "Hospital_region_code": "Region1",
    "Available_Extra_Rooms_in_Hospital": 10,
    "Department": "General",
    "Ward_Type": "General",
    "Bed_Grade": 2.0,
    "Type_of_Admission": "Emergency",
    "Severity_of_Illness": "Moderate",
    "Visitors_with_Patient": 2,
    "Age": "40-50",
    "Admission_Deposit": 5000.0
}

response = requests.post(url, json=data)
print(response.json())
```

### Troubleshooting

**Common Issues and Solutions**

| Issue | Solution |
|-------|----------|
| `ModuleNotFoundError` | Run `pip install -r requirements.txt` |
| DVC remote authentication failed | Reconfigure with `dvc remote modify origin --local auth basic` |
| MLflow tracking not working | Check `src/utils/mlflow_utils.py` for correct credentials |
| Out of memory during training | Reduce `sample_size` in `params.yaml` |
| Docker build fails | Ensure Docker is installed and running: `docker ps` |
| ECR push unauthorized | Run `aws ecr get-login-password` again |
| ECS task fails to start | Check CloudWatch logs for error messages |

**Getting DagsHub Token**

1. Go to https://dagshub.com/user/settings/tokens
2. Click "Generate New Token"
3. Copy the token immediately

**Checking Pipeline Logs**

```bash
# DVC shows which stage failed
dvc repro -v

# Check individual stage logs
cat .dvc/tmp/run.log

# MLflow logs (stored locally or on DagsHub)
mlflow ui
```

**Clean and Rerun**

```bash
# Remove all outputs and rerun
dvc repro --downstream --force

# Clean DVC cache (if needed)
dvc gc -w
```

### Support and Resources

- **DagsHub Documentation**: https://dagshub.com/docs
- **MLflow Documentation**: https://mlflow.org/docs/latest/index.html
- **DVC Documentation**: https://dvc.org/doc
- **AWS ECS Documentation**: https://docs.aws.amazon.com/ecs

### License

This project is licensed under the MIT License.