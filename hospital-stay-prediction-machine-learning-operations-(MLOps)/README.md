# Hospital Stay Prediction - Optimized Production Pipeline

A production-ready MLOps pipeline for predicting patient hospital stay duration using stacked ensemble learning with LightGBM, XGBoost, and CatBoost. This pipeline includes experiment tracking with MLflow on DagsHub, hyperparameter optimization with Optuna, data version control with DVC, containerization with Docker, and production deployment on AWS ECS.

This project is a **reproducible MLOps pipeline** for predicting **hospital stay categories** using a **stacked ensemble of gradient boosting models** (LightGBM, XGBoost, CatBoost). It is designed for production deployment and educational purposes, with containerized inference on AWS ECS. The pipeline covers the full lifecycle of a machine learning experiment:

1. Raw data ingestion and preprocessing.
2. Feature engineering with PCA and SMOTE.
3. Hyperparameter optimization with Optuna.
4. Stacked ensemble training with meta-learner.
5. Experiment tracking with MLflow on DagsHub.
6. Data and model versioning with DVC.
7. Multi-metric evaluation.
8. Dockerized inference.
9. AWS ECS deployment.

Healthcare providers need accurate predictions of patient stay duration to optimize resource allocation, staffing, and bed management. This project investigates whether **stacked ensemble learning** can improve prediction accuracy over individual models while remaining reproducible and deployable. It also demonstrates how MLOps practices can make healthcare ML research more transparent, comparable, and production-ready.

---

## Methodology

**Design**

This project follows an **experimental, reproducible MLOps design**. The central goal is to evaluate whether a stacked ensemble of gradient boosting models can outperform individual classifiers for hospital stay prediction. The study compares the ensemble against its constituent models using cross-validation and held-out test metrics.

**Core Components**

| Component | Purpose |
|---|---|
| **Data Pipeline** | Cleans hospital admission data and creates train/test splits. |
| **Feature Engineering** | Applies PCA for dimensionality reduction and SMOTE for class balancing. |
| **Optuna Tuning** | Optimizes hyperparameters for LightGBM, XGBoost, and CatBoost. |
| **Stacked Ensemble** | Combines base models with a meta-learner for final predictions. |
| **MLflow + DagsHub** | Tracks parameters, metrics, and artifacts. |
| **DVC** | Versions datasets and model artifacts. |
| **Evaluation Suite** | Measures accuracy, precision, recall, F1, and AUC-ROC. |
| **Docker Inference API** | Serves the trained model via FastAPI. |
| **AWS ECS Deployment** | Production deployment with ECR and Fargate. |
| **Tests** | Pytest coverage for data, model, and API behavior. |

**High-Level Workflow**

```text
Hospital Admission CSV
        |
        V
DVC-tracked raw data
        |
        V
Prepare → Preprocess → PCA + SMOTE → train/test splits
        |
        V
Optuna tuning (LightGBM, XGBoost, CatBoost)
        |
        V
Stacked ensemble training with meta-learner
        |
        V
MLflow tracking + DVC artifacts
        |
        V
Evaluation (accuracy, precision, recall, F1, AUC-ROC)
        |
        V
Dockerized FastAPI inference
        |
        V
AWS ECS deployment
```

**Hypotheses**

- **H1:** Stacked ensemble improves macro F1 over individual gradient boosting models.
- **H2:** Optuna hyperparameter optimization improves model performance over default parameters.
- **H3:** PCA and SMOTE improve generalization on imbalanced hospital stay categories.
- **H4:** MLflow + DVC enable reliable reproduction of experiments across machines.

**Data Collection and Preprocessing**

- **Source:** Hospital admission dataset with patient demographics, admission details, and stay categories.
- **Files:** `train_data.csv`.
- **Cleaning:** Handle missing values, encode categorical variables, normalize numerical features.
- **Splits:** 80% train, 20% test with stratified sampling.
- **Format:** Convert to NumPy arrays for model training.
- **Versioning:** Track raw and processed data with DVC.

**Model and Training**

- **Base Models:** LightGBM, XGBoost, CatBoost.
- **Ensemble Method:** Stacking with logistic regression meta-learner.
- **Feature Engineering:** PCA (95% variance retention), SMOTE for class balancing.
- **Objective:** Multi-class classification for stay categories.
- **Key Hyperparameters:**
  - `n_trials`: 5 (increase to 20-50 for production)
  - `cv`: 3-fold cross-validation
  - `random_state`: 42
  - `pca_variance`: 0.95
- **Hardware:** CPU-based training (GPU optional for CatBoost).
- **Monitoring:** Training metrics via MLflow; pipeline status via DVC.

**Experiment Tracking and Versioning**

- **MLflow:** Logs parameters, metrics, and artifacts for every Optuna trial and final model.
- **DagsHub:** Remote MLflow tracking server with experiment comparison.
- **DVC:** Versions datasets, processed arrays, and model artifacts.
- **Configuration:** All hyperparameters stored in `params.yaml`.
- **Optuna Integration:** Each trial logged as a separate MLflow run.

**Evaluation**

Evaluation is performed on the held-out test set using:

- **Classification Metrics:** Accuracy, Precision (macro), Recall (macro), F1 Score (macro), AUC-ROC (macro).
- **Cross-Validation:** 3-fold CV during hyperparameter tuning.
- **Confusion Matrix:** Per-class performance analysis.
- **Feature Importance:** SHAP values for model interpretability.

Metrics are saved to `metrics/evaluation.json` and logged to MLflow.

**Experimental Conditions and Ablations**

- **Baseline:** Individual LightGBM, XGBoost, CatBoost with default parameters.
- **Main Condition:** Stacked ensemble with Optuna-tuned base models.
- **Ablations:**
  - Without PCA: Compare dimensionality reduction impact.
  - Without SMOTE: Compare class balancing impact.
  - Different meta-learners: Logistic regression vs. gradient boosting.
  - Number of Optuna trials: 5, 20, 50.
- **Replication:** Run multiple seeds where compute permits and report mean ± standard deviation.

**Deployment as a Production Artifact**

- **Inference API:** FastAPI service with `/predict` and `/docs` endpoints.
- **Containerization:** Docker with multi-stage builds.
- **Cloud Deployment:** AWS ECR for image storage, ECS Fargate for serverless containers.
- **Auto-scaling:** CPU-based target tracking scaling policy.
- **Purpose:** Production healthcare prediction service.

**Reproducibility Plan**

- All configurations stored in version-controlled `params.yaml`.
- Raw and processed data tracked with DVC.
- Every run logged to MLflow with parameters, metrics, and artifacts.
- Docker image provided for consistent inference.
- Tests included for data processing, model loading, and API behavior.
- Environment variables documented in `.env.example`.

**Hardware Requirements**

- **Hardware**: CPU with at least 4 cores (GPU optional for CatBoost).
- **Memory**: 8GB RAM minimum, 16GB recommended.
- **Python**: 3.11 or higher.
- **Docker**: 24.0 or higher (for container deployment).
- **AWS CLI**: Configured with appropriate credentials (for AWS deployment).
- **DagsHub Account**: For experiment tracking and DVC remote storage.

**Ethical Considerations**

- Hospital data is used under its original license; users must verify terms.
- Predictions should not replace clinical judgment.
- Model may exhibit bias across demographic groups; fairness analysis recommended.
- The model should not be deployed in high-stakes clinical settings without further validation.

**Limitations**

- Accuracy depends on data quality and representativeness.
- Stacked ensemble increases inference latency compared to single models.
- Optuna trials limited to 5 by default; increase for production.
- AWS ECS deployment is for demonstration; production requires security hardening.
- Hospital data may contain biases and outdated patterns.

---

## Features

- **Stacked Ensemble Learning**: Combines LightGBM, XGBoost, and CatBoost with meta-learner
- **Hyperparameter Optimization**: Optuna with MLflow integration for trial tracking
- **Experiment Tracking**: MLflow with DagsHub integration
- **Data Version Control**: DVC for dataset and model versioning
- **Feature Engineering**: PCA for dimensionality reduction, SMOTE for class balancing
- **Comprehensive Metrics**: Accuracy, precision, recall, F1, AUC-ROC
- **Docker Containerization**: Multi-stage builds for production deployment
- **AWS Deployment**: ECR, ECS Fargate with auto-scaling
- **Integration Tests**: Pytest with coverage reporting

---

## Project Structure

```text
hospital-stay-prediction/
├── .gitignore                   # Git ignore rules
├── dvc.yaml                     # DVC pipeline stages
├── params.yaml                  # Configuration parameters
├── requirements.txt             # Python dependencies
├── Dockerfile                   # Docker build instructions
├── .env.example                 # Environment template
├── README.md                    # Project documentation
├── data/
│   ├── raw/                     # Input data (train_data.csv)
│   └── processed/               # Processed numpy arrays
├── models/                      # Saved models (DVC tracked)
├── metrics/                     # Metrics JSON files (DVC tracked)
├── logs/                        # Pipeline logs
├── src/
│   ├── __init__.py
│   ├── data/                    # Data preparation
│   │   ├── __init__.py
│   │   └── prepare.py
│   ├── features/                # Feature engineering
│   │   ├── __init__.py
│   │   └── preprocess.py
│   ├── models/                  # Training and tuning
│   │   ├── __init__.py
│   │   ├── tune.py
│   │   └── train.py
│   ├── evaluation/              # Metrics calculation
│   │   ├── __init__.py
│   │   └── evaluate.py
│   ├── deployment/              # FastAPI application
│   │   ├── __init__.py
│   │   └── app.py
│   └── utils/                   # MLflow utilities
│       ├── __init__.py
│       └── mlflow_utils.py
├── tests/                       # Pytest suite
│   ├── __init__.py
│   └── test_pipeline.py
└── deployment/                  # AWS deployment scripts
    └── aws_deploy.sh
```

---

## Get Started - Step-By-Step

### Initial Setup Checklist

```bash
# 1. Create DagsHub repository
# Go to https://dagshub.com and create "hospital-stay-prediction"

# 2. Clone repository and add data
git clone https://dagshub.com/YOUR_DAGSHUB_USERNAME/hospital-stay-prediction.git
cd hospital-stay-prediction
mkdir -p data/raw data/processed models metrics logs
cp /path/to/train_data.csv data/raw/

# 3. Configure DagsHub authentication
cp .env.example .env
# Edit .env with your credentials

# 4. Install dependencies
pip install -r requirements.txt

# 5. Initialize DVC
dvc init

# 6. Configure DVC remote
dvc remote add origin https://dagshub.com/YOUR_DAGSHUB_USERNAME/hospital-stay-prediction.dvc
dvc remote modify origin --local auth basic
dvc remote modify origin --local user YOUR_DAGSHUB_USERNAME
dvc remote modify origin --local password YOUR_DAGSHUB_TOKEN

# 7. Verify installation
dvc remote list
python -c "from src.utils.mlflow_utils import init_mlflow; init_mlflow(); print('MLflow configured')"
ls -la data/raw/
```

### Model Experiment

**Step 1: Environment Setup**

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Upgrade pip
pip install --upgrade pip

# Install requirements
pip install -r requirements.txt
```

**Step 2: Configure DagsHub and MLflow**

**2.1 Create a DagsHub Repository**

1. Go to [dagshub.com](https://dagshub.com) and sign up/log in
2. Create a new repository (e.g., `hospital-stay-prediction`)
3. Note your username and repository name

**2.2 Get DagsHub Token**

1. Go to Settings → Tokens
2. Generate a new token with write permissions
3. Save the token securely

**2.3 Set Environment Variables**

```bash
# Set DagsHub authentication
export DAGSHUB_USER="your-dagshub-username"
export DAGSHUB_REPO="hospital-stay-prediction"
export DAGSHUB_TOKEN="your-dagshub-token"

# Optional: Save to .env file for persistence
echo "DAGSHUB_USER=${DAGSHUB_USER}" >> .env
echo "DAGSHUB_REPO=${DAGSHUB_REPO}" >> .env
echo "DAGSHUB_TOKEN=${DAGSHUB_TOKEN}" >> .env
```

**2.4 Update MLflow Configuration**

Edit `src/utils/mlflow_utils.py`:

```python
def init_mlflow():
    dagshub.init(repo_owner="YOUR_DAGSHUB_USERNAME", repo_name="hospital-stay-prediction")
    mlflow.set_tracking_uri(f"https://dagshub.com/YOUR_DAGSHUB_USERNAME/hospital-stay-prediction.mlflow")
    mlflow.set_experiment("hospital-stay-stacking")
```

**2.5 Configure DVC Remote**

```bash
# Add DagsHub DVC remote
dvc remote add -d origin https://dagshub.com/${DAGSHUB_USER}/${DAGSHUB_REPO}.dvc
dvc remote modify origin --projecturl https://dagshub.com/${DAGSHUB_USER}/${DAGSHUB_REPO}
dvc remote modify origin auth basic
dvc remote modify origin user ${DAGSHUB_USER}
dvc remote modify origin password ${DAGSHUB_TOKEN}

# Commit DVC config
git add .dvc/config .dvcignore .gitignore
git commit -m "Configure DVC with DagsHub remote"
```

**Step 3: Prepare Data**

**3.1 Verify Data Files**

Ensure the following file exists in `data/raw/`:
- `train_data.csv` (~hospital admission records)

**3.2 Track Data with DVC**

```bash
# Add raw data directory to DVC
dvc add data/raw

# Commit DVC tracking files
git add data/raw.dvc .gitignore
git commit -m "Track raw data with DVC"

# Push data to DagsHub remote
dvc push -r origin
```

**Step 4: Configure Pipeline Parameters**

Edit `params.yaml`:

```yaml
data:
  sample_size: 2000        # Number of samples (-1 for full dataset)
  test_size: 0.2          # Test split ratio
  random_state: 42        # Reproducibility seed

preprocess:
  pca_variance: 0.95      # PCA variance retention

models:
  n_trials: 5             # Optuna trials per model (increase for production)
  cv: 3                   # Cross-validation folds
  random_state: 42        # Reproducibility seed
```

**Step 5: Run Data Preparation**

```bash
# Run prepare stage
dvc repro prepare

# Or run directly
python src/data/prepare.py
```

**What prepare does:**
1. Loads raw hospital admission data.
2. Handles missing values and encodes categorical variables.
3. Splits data into train/test sets (80/20).
4. Saves processed parquet files to `data/processed/`.

**Expected output:**

```text
Loading data from data/raw/train_data.csv
Sampling 2000 records...
Encoding categorical variables...
Splitting data (80/20)...
Train samples: 1600
Test samples: 400
Data preparation completed successfully
```

**Step 6: Run Preprocessing**

```bash
# Run preprocess stage
dvc repro preprocess

# Or run directly
python src/features/preprocess.py
```

**What preprocess does:**
1. Applies PCA for dimensionality reduction (95% variance).
2. Applies SMOTE for class balancing.
3. Saves NumPy arrays to `data/processed/`.
4. Saves preprocessor pipeline to `models/preprocessor.pkl`.

**Expected output:**

```text
Loading train/test data...
Applying PCA (95% variance retention)...
PCA components: 15
Applying SMOTE for class balancing...
Original class distribution: [800, 500, 300]
Balanced class distribution: [800, 800, 800]
Saving processed arrays...
Preprocessing completed successfully
```

**Step 7: Run Hyperparameter Tuning**

```bash
# Run tuning stage
dvc repro tune_weak

# Or run directly
python src/models/tune.py
```

**What tuning does:**
1. Runs Optuna optimization for LightGBM, XGBoost, and CatBoost.
2. Logs each trial to MLflow with parameters and CV scores.
3. Saves best parameters to `models/best_weak_params.json`.

**Expected output:**

```text
============================================================
HYPERPARAMETER OPTIMIZATION WITH OPTUNA
============================================================

[Tuning LightGBM]
  Trial 1/5: macro_f1=0.7234
  Trial 2/5: macro_f1=0.7456
  Trial 3/5: macro_f1=0.7389
  Trial 4/5: macro_f1=0.7512
  Trial 5/5: macro_f1=0.7478
  Best LightGBM: macro_f1=0.7512

[Tuning XGBoost]
  Trial 1/5: macro_f1=0.7189
  Trial 2/5: macro_f1=0.7401
  Trial 3/5: macro_f1=0.7334
  Trial 4/5: macro_f1=0.7467
  Trial 5/5: macro_f1=0.7423
  Best XGBoost: macro_f1=0.7467

[Tuning CatBoost]
  Trial 1/5: macro_f1=0.7267
  Trial 2/5: macro_f1=0.7445
  Trial 3/5: macro_f1=0.7398
  Trial 4/5: macro_f1=0.7490
  Trial 5/5: macro_f1=0.7456
  Best CatBoost: macro_f1=0.7490

[Best parameters saved to models/best_weak_params.json]

============================================================
TUNING COMPLETED
============================================================
```

**Step 8: Train Final Stacked Ensemble**

```bash
# Run training stage
dvc repro train_final

# Or run directly
python src/models/train.py
```

**What training does:**
1. Loads best parameters from tuning stage.
2. Trains LightGBM, XGBoost, and CatBoost with best parameters.
3. Creates stacked ensemble with logistic regression meta-learner.
4. Saves final model to `models/final_stacking_model.pkl`.
5. Logs all metrics to MLflow.

**Key training parameters (configurable in `params.yaml`):**
- `n_trials`: 5
- `cv`: 3
- `random_state`: 42

**Training progress monitoring:**

```bash
# View MLflow logs
mlflow ui --backend-store-uri sqlite:///mlruns/mlflow.db --port 5000
```

**Expected output:**

```text
============================================================
TRAINING STACKED ENSEMBLE
============================================================

[Training LightGBM with best parameters]
  CV macro_f1: 0.7512

[Training XGBoost with best parameters]
  CV macro_f1: 0.7467

[Training CatBoost with best parameters]
  CV macro_f1: 0.7490

[Training meta-learner (LogisticRegression)]
  Meta-learner CV macro_f1: 0.7634

[Stacked ensemble saved to models/final_stacking_model.pkl]

============================================================
TRAINING COMPLETED
============================================================
```

**Step 9: Evaluate Model**

```bash
# Run evaluation
dvc repro evaluate

# Or run directly
python src/evaluation/evaluate.py
```

**What evaluation does:**
1. Loads the stacked ensemble and test dataset.
2. Generates predictions for test samples.
3. Computes accuracy, precision, recall, F1, and AUC-ROC.
4. Saves metrics to `metrics/evaluation.json`.
5. Logs all metrics to MLflow.

**Expected output:**

```text
============================================================
EVALUATION RESULTS
============================================================
accuracy: 0.7850
precision_macro: 0.7623
recall_macro: 0.7512
f1_macro: 0.7567
auc_roc_macro: 0.8912

Per-Class Metrics:
  Class 0-10 days: precision=0.81, recall=0.79, f1=0.80
  Class 11-20 days: precision=0.74, recall=0.72, f1=0.73
  Class 21-30 days: precision=0.71, recall=0.70, f1=0.70
  Class 31+ days: precision=0.78, recall=0.76, f1=0.77

============================================================
PIPELINE COMPLETED SUCCESSFULLY
============================================================
```

**Step 10: Run Tests**

```bash
# Run tests as part of DVC pipeline
dvc repro test

# Or run directly
pytest tests/ -v --cov=src --cov-report=html
```

**Test Coverage:**

The test suite validates:
- Data shapes after preprocessing.
- Preprocessor functionality.
- Model prediction capabilities.
- API endpoint responses.

**Step 11: View MLflow Dashboard**

```bash
# Launch MLflow UI
mlflow ui --backend-store-uri sqlite:///mlruns/mlflow.db --port 5000

# Open in browser
# http://localhost:5000
```

**What you can see in MLflow:**
- **Parameters**: All hyperparameters for each Optuna trial and final model.
- **Metrics**: CV scores, test metrics, training time.
- **Artifacts**: Model files, evaluation JSON, plots.
- **Tags**: Run name, stage, version.
- **Search**: Filter runs by parameters or metrics.
- **Compare**: Compare multiple runs side-by-side.

---

## Docker Deployment

**Build and Run Locally**

```bash
# Pull the final model from DVC (if not already available)
dvc pull

# Build Docker image
docker build -t hospital-stay-predictor:latest .

# Run container with port mapping
docker run -p 8000:8000 hospital-stay-predictor:latest

# With custom port mapping (host:container)
docker run -p 8080:8000 hospital-stay-predictor:latest
```

**Test the API**

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

**Expected response:**

```json
{
  "stay_category": "0-10 days"
}
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

**Stop Containers**

```bash
# Stop all services
docker stop $(docker ps -q)

# Remove all containers
docker rm $(docker ps -aq)

# Remove image
docker rmi hospital-stay-predictor:latest
```

---

## AWS Deployment

**Setup AWS Requirements**

```bash
# Install AWS CLI
pip install awscli

# Configure AWS credentials
aws configure
# Enter: AWS Access Key ID, Secret Access Key, region (us-east-1)
```

**Deploy with AWS CLI**

**Step 1: Create ECR Repository**

```bash
# Create repository
aws ecr create-repository \
  --repository-name hospital-stay-predictor \
  --region us-east-1

# Note the repository URI from output
# e.g., 123456789012.dkr.ecr.us-east-1.amazonaws.com/hospital-stay-predictor
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

```bash
# Get task public IP
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

**API Gateway Endpoint:**

After deployment, you'll receive a public IP or load balancer DNS:

```text
http://PUBLIC_IP:8000/predict
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

---

## API Usage

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

---

## Testing

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
pytest tests/test_pipeline.py -v

# Test model prediction
pytest tests/test_pipeline.py::test_model_prediction -v

# Test API endpoints
pytest tests/test_pipeline.py::test_api_endpoint -v
```

**Test with Sample Data**

```bash
# Run a quick test with minimal data
python -c "
from src.features.preprocess import load_preprocessor
preprocessor = load_preprocessor('models/preprocessor.pkl')
print('Preprocessor loaded successfully')
"
```

---

## Environment Variables

Create a `.env` file for persistent configuration:

```bash
# DagsHub Configuration
DAGSHUB_USER=your-username
DAGSHUB_REPO=hospital-stay-prediction
DAGSHUB_TOKEN=your-token
MLFLOW_TRACKING_URI=https://dagshub.com/your-username/hospital-stay-prediction.mlflow

# AWS Configuration
AWS_ACCESS_KEY_ID=your-access-key
AWS_SECRET_ACCESS_KEY=your-secret-key
AWS_DEFAULT_REGION=us-east-1

# Local Paths
MODEL_PATH=./models/final_stacking_model.pkl
DATA_PATH=./data/raw
OUTPUT_PATH=./models

# MLflow Configuration
MLFLOW_EXPERIMENT_NAME=hospital-stay-stacking
```

Load environment variables:

```bash
source .env
# Or use python-dotenv
```

---

## Monitoring and Logging

**MLflow Metrics Tracked**

**Tuning Metrics:**

- `macro_f1`: Cross-validation macro F1 score per trial
- `params`: Hyperparameters for each trial

**Training Metrics:**

- `cv_macro_f1`: Cross-validation score for final models
- `meta_learner_cv_macro_f1`: Meta-learner cross-validation score

**Evaluation Metrics:**

- `accuracy`: Overall accuracy
- `precision_macro`: Macro-averaged precision
- `recall_macro`: Macro-averaged recall
- `f1_macro`: Macro-averaged F1 score
- `auc_roc_macro`: Macro-averaged AUC-ROC

**View MLflow Dashboard**

```bash
# Local tracking
mlflow ui --backend-store-uri sqlite:///mlruns/mlflow.db --port 5000

# DagsHub tracking (if configured)
# Visit: https://dagshub.com/{username}/{repo}/experiments
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

---

## Troubleshooting

**DagsHub Authentication Issues**

```bash
# Reconfigure DVC remote
dvc remote remove origin
dvc remote add -d origin https://dagshub.com/${DAGSHUB_USER}/${DAGSHUB_REPO}.dvc
dvc remote modify origin --projecturl https://dagshub.com/${DAGSHUB_USER}/${DAGSHUB_REPO}
dvc remote modify origin auth basic
dvc remote modify origin user ${DAGSHUB_USER}
dvc remote modify origin password ${DAGSHUB_TOKEN}

# Test connection
dvc pull -r origin
```

**MLflow Permission Issues**

```bash
# Use local tracking instead of remote
export MLFLOW_TRACKING_URI="file:./mlruns"

# Or create a new experiment
mlflow experiments create -n hospital-stay-stacking
```

**Out of Memory (OOM)**

```yaml
# Reduce sample size in params.yaml
data:
  sample_size: 500  # instead of 2000

# Reduce Optuna trials
models:
  n_trials: 3  # instead of 5
```

**Data Loading Issues**

```bash
# Verify CSV files exist
ls -la data/raw/

# Check CSV encoding
file data/raw/train_data.csv

# Re-run preprocessing with smaller sample
# In params.yaml, set:
sample_size: 100  # instead of 2000
```

**Docker Issues**

```bash
# Check if Docker is installed
docker --version

# Install Docker (Ubuntu/Debian)
sudo apt-get update
sudo apt-get install docker.io

# Add user to docker group
sudo usermod -aG docker $USER
newgrp docker
```

**ECS Task Fails to Start**

```bash
# Check CloudWatch logs
aws logs describe-log-groups --log-group-name-prefix /ecs/hospital-stay

# Get task failure reason
aws ecs describe-tasks --cluster hospital-stay-cluster --tasks TASK_ID
```

---

## Contributing

Contributions are welcome for research reproducibility, evaluation metrics, ablation support, and documentation. Please open an issue or pull request with a clear description, and ensure tests pass before submitting.

---

**Note**: This pipeline is designed for educational and production purposes. For clinical deployment, ensure proper security, monitoring, scaling, and regulatory compliance configurations.
