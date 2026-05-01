#!/usr/bin/env python
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import yaml
import json
from src.data.dataset_builder import DatasetBuilder
from src.utils.helpers import create_directory

def load_config():
    with open("config/config.yaml", "r") as f:
        config = yaml.safe_load(f)
    return config

def main():
    config = load_config()
    
    create_directory(config["data"]["processed_path"])
    create_directory("metrics")
    
    builder = DatasetBuilder(config)
    
    print("Building datasets from CSV files...")
    datasets = builder.build_from_csv(
        os.path.join(config["data"]["raw_path"], config["data"]["questions_file"]),
        os.path.join(config["data"]["raw_path"], config["data"]["answers_file"])
    )
    
    print("Saving datasets to disk...")
    builder.save_datasets(config["data"]["processed_path"])
    
    print(f"Train samples: {len(datasets['train'])}")
    print(f"Validation samples: {len(datasets['validation'])}")
    print(f"Test samples: {len(datasets['test'])}")
    
    preprocess_metrics = {
        "train_size": len(datasets['train']),
        "val_size": len(datasets['validation']),
        "test_size": len(datasets['test']),
        "status": "completed"
    }
    
    with open("metrics/preprocessing_metrics.json", "w") as f:
        json.dump(preprocess_metrics, f, indent=2)
    
    print("Preprocessing completed successfully")

if __name__ == "__main__":
    main()