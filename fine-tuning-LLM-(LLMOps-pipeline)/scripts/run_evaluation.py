#!/usr/bin/env python
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import yaml
import torch
import json
from datasets import Dataset
from transformers import AutoTokenizer
from peft import PeftModel
from src.models.model_evaluator import ModelEvaluator
from src.models.model_trainer import load_quantized_model

def load_config():
    with open("config/config.yaml", "r") as f:
        config = yaml.safe_load(f)
    return config

def main():
    config = load_config()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    print("Loading test dataset...")
    test_dataset = Dataset.load_from_disk(os.path.join(config["data"]["processed_path"], "test"))
    print(f"Test samples: {len(test_dataset)}")
    
    print("Loading base model...")
    base_model, tokenizer = load_quantized_model(config["model"]["base_model"], config["quantization"])
    
    print(f"Loading LoRA adapter from {config['model']['output_dir']}...")
    model = PeftModel.from_pretrained(base_model, config["model"]["output_dir"])
    model.to(device)
    model.eval()
    
    print("Starting evaluation...")
    evaluator = ModelEvaluator(model, tokenizer, config)
    metrics, language_stats = evaluator.evaluate_on_dataset(test_dataset)
    
    eval_results = {
        "metrics": metrics,
        "language_distribution": language_stats,
        "total_samples": len(test_dataset)
    }
    
    os.makedirs("metrics", exist_ok=True)
    with open("metrics/evaluation_metrics.json", "w") as f:
        json.dump(eval_results, f, indent=2)
    
    os.makedirs("plots", exist_ok=True)
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(10, 6))
        metric_names = ['bleu', 'rouge1', 'rouge2', 'rougeL', 'bertscore_f1']
        metric_values = [metrics.get(m, 0) for m in metric_names]
        ax.bar(metric_names, metric_values)
        ax.set_xlabel('Metrics')
        ax.set_ylabel('Score')
        ax.set_title('Model Evaluation Metrics')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig('plots/metrics_comparison.png')
        plt.close()
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
    
    print("="*60)

if __name__ == "__main__":
    main()