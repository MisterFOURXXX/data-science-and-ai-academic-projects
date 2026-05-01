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
from src.models.model_trainer import load_quantized_model, train_model
from src.models.model_evaluator import ModelEvaluator
from src.tracking.mlflow_tracker import MLflowTracker, log_all_parameters, log_all_metrics

def load_config():
    with open("config/config.yaml", "r") as f:
        config = yaml.safe_load(f)
    return config

def main():
    config = load_config()
    
    # Ensure directories exist for DVC tracking
    os.makedirs("metrics", exist_ok=True)
    os.makedirs("models", exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    
    # Set up MLflow tracking with DagsHub
    mlflow.set_tracking_uri(config["mlflow"]["tracking_uri"])
    mlflow.set_experiment(config["mlflow"]["experiment_name"])
    
    # Start MLflow run
    with mlflow.start_run(run_name="full_pipeline_run", tags={
        "stage": "training", 
        "version": config["project"]["version"],
        "timestamp": datetime.now().isoformat()
    }) as run:
        
        # Log all configuration parameters
        log_all_parameters(config)
        
        # Log system information
        mlflow.log_param("system.cuda_available", torch.cuda.is_available())
        if torch.cuda.is_available():
            mlflow.log_param("system.cuda_device_name", torch.cuda.get_device_name(0))
            mlflow.log_param("system.cuda_memory_gb", torch.cuda.get_device_properties(0).total_memory / 1024**3)
        
        # Load datasets
        print("Loading datasets from disk...")
        train_dataset = Dataset.load_from_disk(os.path.join(config["data"]["processed_path"], "train"))
        val_dataset = Dataset.load_from_disk(os.path.join(config["data"]["processed_path"], "validation"))
        test_dataset = Dataset.load_from_disk(os.path.join(config["data"]["processed_path"], "test"))
        
        print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")
        
        # Log dataset sizes
        mlflow.log_metric("dataset.train_size", len(train_dataset))
        mlflow.log_metric("dataset.val_size", len(val_dataset))
        mlflow.log_metric("dataset.test_size", len(test_dataset))
        
        # Load model and tokenizer
        print("Loading model and tokenizer...")
        if torch.cuda.is_available():
            torch.cuda.set_device(0)
            print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        else:
            print("CUDA not available, using CPU")
        
        model, tokenizer = load_quantized_model(config["model"]["base_model"], config["quantization"])
        
        # Train model
        print("Starting model training...")
        trainer = train_model(model, tokenizer, train_dataset, val_dataset, config, config["lora"])
        
        # Train and log metrics during training
        trainer.train()
        
        # Log training metrics from trainer state
        if trainer.state.log_history:
            for i, log_entry in enumerate(trainer.state.log_history):
                if "loss" in log_entry:
                    mlflow.log_metric("train.loss", log_entry["loss"], step=log_entry.get("step", i))
                if "grad_norm" in log_entry:
                    mlflow.log_metric("train.grad_norm", log_entry["grad_norm"], step=log_entry.get("step", i))
                if "learning_rate" in log_entry:
                    mlflow.log_metric("train.learning_rate", log_entry["learning_rate"], step=log_entry.get("step", i))
                if "rewards/accuracies" in log_entry:
                    mlflow.log_metric("dpo.reward_accuracy", log_entry["rewards/accuracies"], step=log_entry.get("step", i))
                if "rewards/margins" in log_entry:
                    mlflow.log_metric("dpo.reward_margin", log_entry["rewards/margins"], step=log_entry.get("step", i))
                if "eval_loss" in log_entry:
                    mlflow.log_metric("eval.loss", log_entry["eval_loss"], step=log_entry.get("step", i))
                if "epoch" in log_entry:
                    mlflow.log_metric("train.epoch", log_entry["epoch"], step=log_entry.get("step", i))
        
        # Save model
        print("Saving model...")
        os.makedirs(config["model"]["output_dir"], exist_ok=True)
        trainer.save_model(config["model"]["output_dir"])
        tokenizer.save_pretrained(config["model"]["output_dir"])
        
        # Log model artifacts
        mlflow.log_artifacts(config["model"]["output_dir"], artifact_path="model")
        
        # Evaluate model
        print("Evaluating model...")
        evaluator = ModelEvaluator(model, tokenizer, config)
        metrics, language_stats = evaluator.evaluate_on_dataset(test_dataset)
        
        # Log evaluation metrics
        log_all_metrics(metrics, step=0)
        
        # Log language distribution
        for lang, count in language_stats.items():
            mlflow.log_metric(f"language.{lang}", count)
        
        # Log final training metrics
        final_metrics = {
            "train.final_loss": trainer.state.log_history[-1].get("loss", 0.0) if trainer.state.log_history else 0.0,
            "train.final_reward_accuracy": trainer.state.log_history[-1].get("rewards/accuracies", 0.0) if trainer.state.log_history else 0.0,
            "eval.final_loss": trainer.state.log_history[-1].get("eval_loss", 0.0) if trainer.state.log_history else 0.0,
        }
        for key, value in final_metrics.items():
            if isinstance(value, (int, float)):
                mlflow.log_metric(key, value)
        
        # Save training metrics to file for DVC tracking
        training_metrics = {
            "train_size": len(train_dataset),
            "val_size": len(val_dataset),
            "test_size": len(test_dataset),
            "final_train_loss": final_metrics["train.final_loss"],
            "final_eval_loss": final_metrics["eval.final_loss"],
            "final_reward_accuracy": final_metrics["train.final_reward_accuracy"],
            **metrics
        }
        
        with open("metrics/training_metrics.json", "w") as f:
            json.dump(training_metrics, f, indent=2)
        
        # Print results
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
        print(f"MLflow Tracking URI: {config['mlflow']['tracking_uri']}")
        print("\n" + "="*60)
        print("PIPELINE COMPLETED SUCCESSFULLY")
        print("="*60)

if __name__ == "__main__":
    main()