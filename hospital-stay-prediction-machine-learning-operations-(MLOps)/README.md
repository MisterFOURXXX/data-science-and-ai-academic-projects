**Project Instruction**

**Project Structure**
```
hospital-stay-prediction/
├── data/
│   └── raw/
│       └── train_data.csv
├── src/
│   ├── data/prepare.py
│   ├── features/preprocess.py
│   ├── models/tune_weak_learners.py
│   ├── models/train_final.py
│   ├── evaluation/evaluate.py
│   ├── deployment/app.py
│   └── utils/mlflow_utils.py    # Your DAGSHUB Information
├── tests/
│   └── test_pipeline.py
├── models/                  # DVC tracked
├── metrics/                 # DVC tracked
├── dvc.yaml
├── params.yaml
├── requirements.txt
├── Dockerfile
├── .env                     # Your DAGSHUB Information
└── README.md
```

**Step-by-Step Setup**

1. Create a DagsHub repository named `YOUR_DAGSHUB_REPOSITORY_NAME` (Recommend 'hospital-stay-prediction') at https://dagshub.com. Note your username and repo name.

2. On your local computer:
   ```
   git clone https://dagshub.com/YOUR_DAGSHUB_USERNAME/YOUR_DAGSHUB_REPOSITORY_NAME.git
   cd YOUR_DAGSHUB_REPOSITORY_NAME
   ```
   Download the dataset and place it as `data/raw/train_data.csv`.

3. Revise `.env` and `\src\utils\mlflow_utils.py` and fill your DagsHub credentials.

4. Install dependencies:
   ```
   pip install -r requirements.txt
   dvc init
   dvc remote add origin https://dagshub.com/YOUR_DAGSHUB_USERNAME/YOUR_DAGSHUB_REPOSITORY_NAME.dvc
   dvc remote modify origin --local auth basic
   ```

5. In case, you have to re-connect to DagsHub repositories.
   ```
   dvc remote add -d origin https://dagshub.com/YOUR_DAGSHUB_USERNAME/YOUR_DAGSHUB_REPOSITORY_NAME.dvc
   dvc remote modify origin --projecturl https://dagshub.com/YOUR_DAGSHUB_USERNAME/YOUR_DAGSHUB_REPOSITORY_NAME
   dvc remote modify origin auth basic
   dvc remote modify origin user YOUR_DAGSHUB_USERNAME
   dvc remote modify origin password YOUR_DAGSHUB_TOKEN
   ```

**Run Experiments**

Run the full pipeline (includes weak learner tuning, ensemble training, evaluation, and tests):
   ```
   dvc repro
   ```
   All Optuna trials, stacking parameters, and evaluation metrics are logged to MLflow. View them at https://dagshub.com/YOUR_DAGSHUB_USERNAME/YOUR_DAGSHUB_REPOSITORY_NAME/experiments.

**Containerized with Docker**

Pull final model from DVC 
   ```
   dvc pull
   ```

Containerized with Docker:
   ```
   docker build -t YOUR_DAGSHUB_REPOSITORY_NAME .
   docker run -p 8000:8000 YOUR_DAGSHUB_REPOSITORY_NAME
   ```
   
**Check if Docker is available via sudo**

Sometimes the docker command is restricted to the root user or requires an explicit path. Try:
    ```
    Bash
    sudo docker --version
    ```
If that still says command not found, Docker is simply not installed on this specific machine/pod.

**If you are using a Cloud Environment (The "Dind" Problem)**

Most cloud notebook instances don't support Docker-in-Docker (DinD) for security reasons. To build your image, you usually have to:
- Option A: Use a local machine where Docker is installed.
- Option B: Use a cloud-based builder like AWS CodeBuild or GitHub Actions to build the image and push it to ECR directly from your Git repo.

**Final Model Deployment**

Deploy to AWS:
   ```
   aws ecr create-repository --repository-name hospital-stay-predictor --region us-east-1
   aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin YOUR_AWS_ACCOUNT_ID.dkr.ecr.us-east-1.amazonaws.com
   docker tag hospital-stay-predictor:latest YOUR_AWS_ACCOUNT_ID.dkr.ecr.us-east-1.amazonaws.com/hospital-stay-predictor:latest
   docker push YOUR_AWS_ACCOUNT_ID.dkr.ecr.us-east-1.amazonaws.com/hospital-stay-predictor:latest
   ```
   Create an ECS Fargate service using the ECR image URI, port 8000, and environment variables from `.env`. The model is now running on AWS.