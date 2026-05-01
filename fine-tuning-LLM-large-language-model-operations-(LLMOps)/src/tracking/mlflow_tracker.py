import mlflow
from mlflow.tracking import MlflowClient
import dagshub
from typing import Dict, Any
import json
import os

class MLflowTracker:
    def __init__(self, tracking_uri: str, experiment_name: str, repo_owner: str = None, repo_name: str = None):
        # Initialize DagsHub if credentials are provided
        if repo_owner and repo_name:
            try:
                dagshub.init(repo_owner=repo_owner, repo_name=repo_name, mlflow=True)
            except Exception as e:
                print(f"DagsHub initialization warning: {e}")
        
        mlflow.set_tracking_uri(tracking_uri)
        self.experiment_name = experiment_name
        
        # Check if experiment exists, create if not
        experiment = mlflow.get_experiment_by_name(experiment_name)
        if experiment is None:
            mlflow.create_experiment(experiment_name)
        
        self.client = MlflowClient()
        
    def start_run(self, run_name: str = None, tags: Dict[str, str] = None):
        self.run = mlflow.start_run(run_name=run_name, tags=tags)
        return self.run.info.run_id
    
    def log_params(self, params: Dict[str, Any]):
        for key, value in params.items():
            mlflow.log_param(key, value)
    
    def log_metrics(self, metrics: Dict[str, float], step: int = None):
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                mlflow.log_metric(key, value, step=step)
    
    def log_artifacts(self, local_path: str, artifact_path: str = None):
        if os.path.exists(local_path):
            mlflow.log_artifacts(local_path, artifact_path)
    
    def log_model(self, model_path: str, model_name: str):
        if os.path.exists(model_path):
            mlflow.log_artifacts(model_path, artifact_path=model_name)
    
    def end_run(self):
        mlflow.end_run()
    
    def get_best_run(self, metric_name: str):
        experiment = self.client.get_experiment_by_name(self.experiment_name)
        runs = self.client.search_runs(
            experiment_ids=[experiment.experiment_id],
            order_by=[f"metrics.{metric_name} DESC"],
            max_results=1
        )
        return runs[0] if runs else None

def log_all_parameters(config: Dict[str, Any]):
    flattened_params = {}
    def flatten_dict(d, parent_key=''):
        for k, v in d.items():
            new_key = f"{parent_key}.{k}" if parent_key else k
            if isinstance(v, dict):
                flatten_dict(v, new_key)
            else:
                flattened_params[new_key] = v
    flatten_dict(config)
    mlflow.log_params(flattened_params)

def log_all_metrics(metrics_dict: Dict[str, Any], step: int = None):
    flattened_metrics = {}
    def flatten_metrics(d, parent_key=''):
        for k, v in d.items():
            new_key = f"{parent_key}.{k}" if parent_key else k
            if isinstance(v, dict):
                flatten_metrics(v, new_key)
            elif isinstance(v, (int, float)):
                flattened_metrics[new_key] = v
            elif isinstance(v, list) and len(v) > 0 and isinstance(v[0], (int, float)):
                for i, item in enumerate(v):
                    flattened_metrics[f"{new_key}_{i}"] = item
    flatten_metrics(metrics_dict)
    for key, value in flattened_metrics.items():
        mlflow.log_metric(key, value, step=step)