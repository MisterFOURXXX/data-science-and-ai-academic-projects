import argparse
import subprocess
import json
import mlflow
import torch
import sys
import os
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import config

def setup_mlflow():
    tracking_uri = config.monitoring.mlflow_tracking_uri
    
    if "dagshub" in tracking_uri:
        username = os.getenv("MLFLOW_TRACKING_USERNAME", os.getenv("DAGSHUB_USERNAME", ""))
        password = os.getenv("MLFLOW_TRACKING_PASSWORD", os.getenv("DAGSHUB_TOKEN", ""))
        if username and password:
            os.environ["MLFLOW_TRACKING_USERNAME"] = username
            os.environ["MLFLOW_TRACKING_PASSWORD"] = password
    
    mlflow.set_tracking_uri(tracking_uri)
    
    experiment_name = config.monitoring.mlflow_experiment_name
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        mlflow.create_experiment(experiment_name)
    mlflow.set_experiment(experiment_name)

def log_metrics_to_mlflow(metrics_file, step_name):
    if not Path(metrics_file).exists():
        print(f"Metrics file not found: {metrics_file}")
        return
    
    with open(metrics_file, 'r') as f:
        metrics = json.load(f)
    
    def flatten_dict(d, parent_key=''):
        items = []
        for k, v in d.items():
            new_key = f"{parent_key}_{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(flatten_dict(v, new_key).items())
            elif isinstance(v, (int, float)):
                items.append((new_key, v))
        return dict(items)
    
    flat_metrics = flatten_dict(metrics)
    for key, value in flat_metrics.items():
        mlflow.log_metric(f"{step_name}_{key}", value)

def run_full_pipeline(skip_stages=None):
    if skip_stages is None:
        skip_stages = []
    
    with mlflow.start_run(run_name="full_rag_pipeline"):
        mlflow.log_params({
            "data_score_threshold": config.data.score_threshold,
            "data_top_n": config.data.top_n_questions,
            "chunk_size": config.vectorstore.chunk_size,
            "embedding_model": config.vectorstore.embedding_model,
            "lora_r": config.model.lora_r,
            "learning_rate": config.training.learning_rate,
            "dpo_beta": getattr(config.training, 'dpo_beta', 0.1),
            "epochs": config.training.epochs,
            "temperature": config.inference.temperature,
            "max_seq_length": config.training.max_seq_length,
            "batch_size": config.training.batch_size,
            "gradient_accumulation_steps": config.training.gradient_accumulation_steps
        })
        
        stages = [
            ("load_data", "src.data.load_data"),
            ("preprocess_data", "src.data.preprocess"),
            ("split_data", "src.data.split_data"),
            ("create_vectorstore", "src.vectorstore.create_vectorstore"),
            ("evaluate_retrieval", "src.vectorstore.retrieval_eval"),
            ("fine_tune_model", "src.model.fine_tune"),
            ("evaluate_rag", "src.evaluation.rag_evaluation")
        ]
        
        for stage_name, module_name in stages:
            if stage_name in skip_stages:
                print(f"Skipping {stage_name}")
                continue
            
            print(f"\n{'='*60}")
            print(f"Running stage: {stage_name}")
            print(f"{'='*60}")
            
            result = subprocess.run(
                [sys.executable, "-m", module_name],
                capture_output=True,
                text=True
            )
            print(result.stdout)
            if result.stderr:
                print(result.stderr)
            
            metrics_file = f"metrics/{stage_name}_metrics.json"
            log_metrics_to_mlflow(metrics_file, stage_name)
            
            if result.returncode != 0:
                print(f"Stage {stage_name} failed")
                return False
            
            if torch.cuda.is_available():
                mlflow.log_metric(f"{stage_name}_gpu_memory_gb", torch.cuda.max_memory_allocated() / 1024**3)
                torch.cuda.reset_peak_memory_stats()
    
    print("\n" + "="*60)
    print("PIPELINE COMPLETED SUCCESSFULLY")
    print("="*60)
    return True

def main():
    parser = argparse.ArgumentParser(description="Run LLMOps RAG Pipeline")
    parser.add_argument("--skip-stages", nargs="+", default=[], help="Stages to skip")
    parser.add_argument("--setup-dvc", action="store_true", help="Initialize DVC")
    parser.add_argument("--setup-dagshub", action="store_true", help="Setup DagsHub tracking")
    
    args = parser.parse_args()
    
    if args.setup_dvc:
        if not os.path.exists(".dvc"):
            subprocess.run(["dvc", "init"], check=True)
            subprocess.run(["dvc", "remote", "add", "-d", "storage", config.monitoring.mlflow_tracking_uri], check=True)
            print("DVC initialized successfully")
        else:
            print("DVC already initialized")
    
    if args.setup_dagshub:
        import dagshub
        dagshub.init(
            repo_name="llmops-rag-pipeline",
            repo_owner="fourapiwit",
            mlflow=True
        )
        print("DagsHub tracking initialized")
    
    setup_mlflow()
    success = run_full_pipeline(skip_stages=args.skip_stages)
    
    return 0 if success else 1

if __name__ == "__main__":
    exit(main())