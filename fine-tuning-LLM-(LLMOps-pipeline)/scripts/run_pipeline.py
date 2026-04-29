#!/usr/bin/env python
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import yaml
import torch
import mlflow
import json
from datasets import Dataset
from src.models.model_trainer import load_quantized_model, train_model
from src.models.model_evaluator import ModelEvaluator
from src.tracking.mlflow_tracker import MLflowTracker, log_all_parameters, log_all_metrics

def load_config():
    with open("config/config.yaml", "r") as f:
        config = yaml.safe_load(f)
    return config

def main():
    config = load_config()
    
    mlflow.set_tracking_uri("file:./mlruns")
    mlflow.set_experiment(config["mlflow"]["experiment_name"])
    
    with mlflow.start_run(run_name="full_pipeline_run", tags={"stage": "training", "version": config["project"]["version"]}):
        
        for key, value in config.items():
            if isinstance(value, dict):
                for sub_key, sub_value in value.items():
                    mlflow.log_param(f"{key}.{sub_key}", sub_value)
            else:
                mlflow.log_param(key, value)
        
        print("Loading datasets from disk...")
        train_dataset = Dataset.load_from_disk(os.path.join(config["data"]["processed_path"], "train"))
        val_dataset = Dataset.load_from_disk(os.path.join(config["data"]["processed_path"], "validation"))
        test_dataset = Dataset.load_from_disk(os.path.join(config["data"]["processed_path"], "test"))
        
        print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")
        
        print("Loading model and tokenizer...")
        if torch.cuda.is_available():
            torch.cuda.set_device(0)
            print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        else:
            print("CUDA not available, using CPU")
        
        model, tokenizer = load_quantized_model(config["model"]["base_model"], config["quantization"])
        
        print("Starting model training...")
        trainer = train_model(model, tokenizer, train_dataset, val_dataset, config, config["lora"])
        trainer.train()
        
        print("Saving model...")
        os.makedirs(config["model"]["output_dir"], exist_ok=True)
        trainer.save_model(config["model"]["output_dir"])
        tokenizer.save_pretrained(config["model"]["output_dir"])
        
        print("Evaluating model...")
        evaluator = ModelEvaluator(model, tokenizer, config)
        metrics, language_stats = evaluator.evaluate_on_dataset(test_dataset)
        
        for key, value in metrics.items():
            if isinstance(value, float):
                mlflow.log_metric(key, value)
        
        for lang, count in language_stats.items():
            mlflow.log_metric(f"language_count_{lang}", count)
        
        training_metrics = {
            "final_loss": trainer.state.log_history[-1].get("loss", 0.0) if trainer.state.log_history else 0.0,
            "final_reward_accuracy": trainer.state.log_history[-1].get("rewards/accuracies", 0.0) if trainer.state.log_history else 0.0,
            **metrics
        }
        
        os.makedirs("metrics", exist_ok=True)
        with open("metrics/training_metrics.json", "w") as f:
            json.dump(training_metrics, f, indent=2)
        
        mlflow.log_artifacts(config["model"]["output_dir"], artifact_path="model")
        
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
        
        print("\n" + "="*60)
        print("PIPELINE COMPLETED SUCCESSFULLY")
        print("="*60)

if __name__ == "__main__":
    main()