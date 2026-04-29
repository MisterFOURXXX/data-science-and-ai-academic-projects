#!/usr/bin/env python
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import yaml
import mlflow
import json
from datetime import datetime

def load_config():
    with open("config/config.yaml", "r") as f:
        config = yaml.safe_load(f)
    with open("config/params.yaml", "r") as f:
        params = yaml.safe_load(f)
    return config, params

def flatten_dict(d, parent_key='', sep='.'):
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)

def main():
    config, params = load_config()
    
    # Set up MLflow tracking
    mlflow.set_tracking_uri("file:./mlruns")
    mlflow.set_experiment("hyperparameter_logging")
    
    with mlflow.start_run(run_name="parameter_logging_run", tags={"stage": "logging", "timestamp": str(datetime.now())}):
        
        print("\n" + "="*60)
        print("HYPERPARAMETER LOGGING (No Optimization)")
        print("="*60)
        
        # Log all configuration parameters
        print("\n[Logging Configuration Parameters]")
        flat_config = flatten_dict(config)
        for key, value in flat_config.items():
            mlflow.log_param(f"config.{key}", value)
            print(f"  config.{key}: {value}")
        
        # Log search space parameters
        print("\n[Logging Search Space Parameters]")
        search_space = params.get("search_space", {})
        for param_name, param_config in search_space.items():
            param_type = param_config.get("type", "unknown")
            if param_type == "loguniform" or param_type == "uniform":
                mlflow.log_param(f"search_space.{param_name}.min", param_config.get("min"))
                mlflow.log_param(f"search_space.{param_name}.max", param_config.get("max"))
                mlflow.log_param(f"search_space.{param_name}.type", param_type)
                print(f"  {param_name}: min={param_config.get('min')}, max={param_config.get('max')}, type={param_type}")
            elif param_type == "int":
                mlflow.log_param(f"search_space.{param_name}.min", param_config.get("min"))
                mlflow.log_param(f"search_space.{param_name}.max", param_config.get("max"))
                mlflow.log_param(f"search_space.{param_name}.type", param_type)
                print(f"  {param_name}: min={param_config.get('min')}, max={param_config.get('max')}, type={param_type}")
            elif param_type == "categorical":
                mlflow.log_param(f"search_space.{param_name}.choices", param_config.get("choices"))
                mlflow.log_param(f"search_space.{param_name}.type", param_type)
                print(f"  {param_name}: choices={param_config.get('choices')}, type={param_type}")
        
        # Log current training parameters (from config)
        print("\n[Logging Current Training Parameters]")
        current_params = {
            "learning_rate": config["training"]["learning_rate"],
            "beta": config["training"]["beta"],
            "lora_r": config["lora"]["r"],
            "lora_alpha": config["lora"]["lora_alpha"],
            "lora_dropout": config["lora"]["lora_dropout"],
            "gradient_accumulation_steps": config["training"]["gradient_accumulation_steps"],
            "warmup_steps": config["training"]["warmup_steps"],
            "max_length": config["training"]["max_length"],
            "num_train_epochs": config["training"]["num_train_epochs"],
            "per_device_train_batch_size": config["training"]["per_device_train_batch_size"],
        }
        for key, value in current_params.items():
            mlflow.log_param(f"current.{key}", value)
            print(f"  current.{key}: {value}")
        
        # Save parameters to JSON file
        os.makedirs("metrics", exist_ok=True)
        optimization_results = {
            "config": flat_config,
            "search_space": search_space,
            "current_parameters": current_params,
            "timestamp": str(datetime.now()),
            "note": "This is a parameter logging run only. No actual optimization was performed."
        }
        
        with open("metrics/optimization_results.json", "w") as f:
            json.dump(optimization_results, f, indent=2)
        
        print("\n[Parameters saved to metrics/optimization_results.json]")
        
        # Log as artifact
        mlflow.log_artifact("metrics/optimization_results.json")
        
        print("\n" + "="*60)
        print("PARAMETER LOGGING COMPLETED")
        print("="*60)
        print("\nTo view logged parameters, run:")
        print("  mlflow ui --backend-store-uri sqlite:///mlruns/mlflow.db --port 5000")

if __name__ == "__main__":
    main()