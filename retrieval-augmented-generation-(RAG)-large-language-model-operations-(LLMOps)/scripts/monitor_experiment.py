#!/usr/bin/env python
import mlflow
import json
import argparse
from datetime import datetime
from typing import Dict, Any

def get_latest_run(experiment_name: str):
    mlflow.set_experiment(experiment_name)
    experiment = mlflow.get_experiment_by_name(experiment_name)
    
    client = mlflow.tracking.MlflowClient()
    runs = client.search_runs(experiment.experiment_id, order_by=["start_time DESC"], max_results=1)
    
    if runs:
        return runs[0]
    return None

def display_metrics(run, metrics_to_show=None):
    if not run:
        print("No runs found")
        return
    
    print(f"\n{'='*60}")
    print(f"Experiment: {run.info.experiment_id}")
    print(f"Run ID: {run.info.run_id}")
    print(f"Status: {run.info.status}")
    print(f"Start Time: {datetime.fromtimestamp(run.info.start_time/1000)}")
    print(f"{'='*60}\n")
    
    print("Parameters:")
    for key, value in run.data.params.items():
        print(f"  {key}: {value}")
    
    print("\nMetrics:")
    metrics = run.data.metrics
    
    if metrics_to_show:
        metrics = {k: v for k, v in metrics.items() if any(m in k for m in metrics_to_show)}
    
    for key, value in sorted(metrics.items()):
        print(f"  {key}: {value:.4f}" if isinstance(value, float) else f"  {key}: {value}")

def compare_runs(experiment_name: str, metric: str = "bleu"):
    mlflow.set_experiment(experiment_name)
    experiment = mlflow.get_experiment_by_name(experiment_name)
    
    client = mlflow.tracking.MlflowClient()
    runs = client.search_runs(experiment.experiment_id, order_by=["start_time DESC"], max_results=10)
    
    print(f"\n{'='*60}")
    print(f"Comparing last {len(runs)} runs by {metric}")
    print(f"{'='*60}")
    
    for i, run in enumerate(runs):
        metric_value = run.data.metrics.get(metric, 0)
        print(f"{i+1}. Run {run.info.run_id[:8]} - {metric}: {metric_value:.4f}")

def monitor_experiment(experiment_name: str, follow: bool = False):
    while True:
        run = get_latest_run(experiment_name)
        display_metrics(run)
        
        if not follow:
            break
        
        print("\nWaiting for updates (Ctrl+C to stop)...")
        time.sleep(30)

if __name__ == "__main__":
    import time
    parser = argparse.ArgumentParser(description="Monitor MLflow experiments")
    parser.add_argument("--experiment", default="llmops-rag-experiment", help="Experiment name")
    parser.add_argument("--follow", action="store_true", help="Continue monitoring")
    parser.add_argument("--compare", action="store_true", help="Compare recent runs")
    parser.add_argument("--metric", default="bleu", help="Metric for comparison")
    
    args = parser.parse_args()
    
    if args.compare:
        compare_runs(args.experiment, args.metric)
    else:
        monitor_experiment(args.experiment, args.follow)