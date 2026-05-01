#!/usr/bin/env python
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import yaml
import torch
import mlflow
import json
from datetime import datetime
from datasets import Dataset
from transformers import AutoTokenizer
from peft import PeftModel
from src.models.model_evaluator import ModelEvaluator
from src.models.model_trainer import load_quantized_model
from src.tracking.mlflow_tracker import log_all_metrics

def load_config():
    with open("config/config.yaml", "r") as f:
        config = yaml.safe_load(f)
    return config

def main():
    config = load_config()
    
    os.makedirs("metrics", exist_ok=True)
    os.makedirs("plots", exist_ok=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Set up MLflow
    mlflow.set_tracking_uri(config["mlflow"]["tracking_uri"])
    mlflow.set_experiment(f"{config['mlflow']['experiment_name']}_evaluation")
    
    with mlflow.start_run(run_name="model_evaluation", tags={
        "stage": "evaluation",
        "timestamp": datetime.now().isoformat()
    }) as run:
        
        print("Loading test dataset...")
        test_dataset = Dataset.load_from_disk(os.path.join(config["data"]["processed_path"], "test"))
        print(f"Test samples: {len(test_dataset)}")
        mlflow.log_metric("dataset.test_size", len(test_dataset))
        
        print("Loading base model...")
        base_model, tokenizer = load_quantized_model(config["model"]["base_model"], config["quantization"])
        
        print(f"Loading LoRA adapter from {config['model']['output_dir']}...")
        model = PeftModel.from_pretrained(base_model, config["model"]["output_dir"])
        model.to(device)
        model.eval()
        
        print("Starting evaluation...")
        evaluator = ModelEvaluator(model, tokenizer, config)
        metrics, language_stats = evaluator.evaluate_on_dataset(test_dataset)
        
        # Log all metrics to MLflow
        log_all_metrics(metrics, step=0)
        
        for lang, count in language_stats.items():
            mlflow.log_metric(f"language_distribution.{lang}", count)
        
        # Save evaluation results
        eval_results = {
            "metrics": metrics,
            "language_distribution": language_stats,
            "total_samples": len(test_dataset),
            "timestamp": datetime.now().isoformat()
        }
        
        with open("metrics/evaluation_metrics.json", "w") as f:
            json.dump(eval_results, f, indent=2)
        
        mlflow.log_artifact("metrics/evaluation_metrics.json")
        
        # Create and save plot
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(10, 6))
            metric_names = ['bleu', 'rouge1', 'rouge2', 'rougeL', 'bertscore_f1', 'perplexity']
            metric_values = [metrics.get(m, 0) for m in metric_names]
            ax.bar(metric_names, metric_values)
            ax.set_xlabel('Metrics')
            ax.set_ylabel('Score')
            ax.set_title('Model Evaluation Metrics')
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig('plots/metrics_comparison.png')
            plt.close()
            mlflow.log_artifact('plots/metrics_comparison.png')
            print("Plot saved to plots/metrics_comparison.png")
        except Exception as e:
            print(f"Could not create plot: {e}")
        
        print("\n" + "="*60)
        print("EVALUATION RESULTS")
        print("="*60)
        for key, value in metrics.items():
            if isinstance(value, float):
                print(f"{key}: {value:.4f}")
            else:
                print(f"{key}: {value}")
        
        print("\nLanguage Distribution:")
        for lang, count in language_stats.items():
            print(f"  {lang}: {count}")
        
        print(f"\nMLflow Run ID: {run.info.run_id}")
        print("="*60)

if __name__ == "__main__":
    main()