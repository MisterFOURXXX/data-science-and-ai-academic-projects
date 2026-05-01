import os
import dagshub
import mlflow
from dotenv import load_dotenv

def setup_dagshub():
    load_dotenv()
    
    username = os.getenv('DAGSHUB_USERNAME', 'your-username')
    repo_name = os.getenv('DAGSHUB_REPO_NAME', 'llmops-rag-pipeline')
    token = os.getenv('DAGSHUB_TOKEN')
    
    dagshub.init(
        repo_name=repo_name,
        repo_owner=username,
        mlflow=True
    )
    
    tracking_uri = f"https://dagshub.com/{username}/{repo_name}.mlflow"
    mlflow.set_tracking_uri(tracking_uri)
    
    if token:
        os.environ['MLFLOW_TRACKING_USERNAME'] = username
        os.environ['MLFLOW_TRACKING_PASSWORD'] = token
    
    print(f"DagsHub initialized: {tracking_uri}")
    return tracking_uri

def get_experiment(experiment_name="llmops-rag-experiment"):
    mlflow.set_experiment(experiment_name)
    return mlflow.get_experiment_by_name(experiment_name)

def log_dvc_metrics(metrics_file: str, step_name: str):
    import json
    with open(metrics_file, 'r') as f:
        metrics = json.load(f)
    
    for key, value in metrics.items():
        if isinstance(value, (int, float)):
            mlflow.log_metric(f"{step_name}_{key}", value)

if __name__ == "__main__":
    setup_dagshub()