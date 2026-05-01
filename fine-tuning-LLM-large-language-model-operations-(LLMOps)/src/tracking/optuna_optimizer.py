import optuna
import mlflow
from typing import Dict, Any, Callable
import yaml
import torch

class OptunaOptimizer:
    def __init__(self, study_name: str, storage: str, direction: str = "maximize"):
        self.study = optuna.create_study(
            study_name=study_name,
            storage=storage,
            direction=direction,
            load_if_exists=True
        )
        
    def suggest_hyperparameters(self, trial, search_space: Dict[str, Any]):
        params = {}
        for param_name, param_config in search_space.items():
            param_type = param_config["type"]
            
            if param_type == "loguniform":
                params[param_name] = trial.suggest_float(
                    param_name, 
                    float(param_config["min"]), 
                    float(param_config["max"]), 
                    log=True
                )
            elif param_type == "uniform":
                params[param_name] = trial.suggest_float(
                    param_name, 
                    float(param_config["min"]), 
                    float(param_config["max"])
                )
            elif param_type == "int":
                params[param_name] = trial.suggest_int(
                    param_name, 
                    int(param_config["min"]), 
                    int(param_config["max"])
                )
            elif param_type == "categorical":
                params[param_name] = trial.suggest_categorical(
                    param_name, 
                    param_config["choices"]
                )
        return params
    
    def optimize(self, objective_func: Callable, n_trials: int = 20):
        self.study.optimize(objective_func, n_trials=n_trials)
        return self.study.best_params, self.study.best_value
    
    def log_best_params_to_mlflow(self):
        best_params = self.study.best_params
        for param_name, param_value in best_params.items():
            mlflow.log_param(f"best_{param_name}", param_value)
        mlflow.log_metric("best_objective_value", self.study.best_value)

def objective_function(trial, config, train_dataset, val_dataset, base_model, tokenizer):
    from src.models.model_trainer import load_quantized_model, train_model
    
    search_space = config["search_space"]
    hyperparams = {}
    
    for param_name, param_config in search_space.items():
        param_type = param_config["type"]
        if param_type == "loguniform":
            hyperparams[param_name] = trial.suggest_float(
                param_name, 
                float(param_config["min"]), 
                float(param_config["max"]), 
                log=True
            )
        elif param_type == "uniform":
            hyperparams[param_name] = trial.suggest_float(
                param_name, 
                float(param_config["min"]), 
                float(param_config["max"])
            )
        elif param_type == "int":
            hyperparams[param_name] = trial.suggest_int(
                param_name, 
                int(param_config["min"]), 
                int(param_config["max"])
            )
        elif param_type == "categorical":
            hyperparams[param_name] = trial.suggest_categorical(
                param_name, 
                param_config["choices"]
            )
    
    config["training"]["learning_rate"] = hyperparams["learning_rate"]
    config["training"]["beta"] = hyperparams["beta"]
    config["lora"]["r"] = hyperparams["lora_r"]
    config["lora"]["lora_alpha"] = hyperparams["lora_alpha"]
    config["lora"]["lora_dropout"] = hyperparams["lora_dropout"]
    config["training"]["gradient_accumulation_steps"] = hyperparams["gradient_accumulation_steps"]
    config["training"]["warmup_steps"] = hyperparams["warmup_steps"]
    config["training"]["max_length"] = hyperparams["max_length"]
    
    model, tokenizer = load_quantized_model(
        config["model"]["base_model"], 
        config["quantization"]
    )
    
    trainer = train_model(
        model, tokenizer, train_dataset, val_dataset, config, config["lora"]
    )
    
    trainer.train()
    
    eval_metrics = trainer.evaluate()
    
    torch.cuda.empty_cache()
    
    objective_value = eval_metrics.get("rewards/accuracies", 0.0)
    
    return objective_value