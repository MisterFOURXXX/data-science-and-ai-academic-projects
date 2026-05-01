import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    EarlyStoppingCallback
)
from peft import LoraConfig
from trl import DPOTrainer, DPOConfig
import mlflow

def load_quantized_model(model_name: str, quantization_config: dict):
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=quantization_config["load_in_4bit"],
        bnb_4bit_quant_type=quantization_config["bnb_4bit_quant_type"],
        bnb_4bit_compute_dtype=torch.bfloat16 if quantization_config["bnb_4bit_compute_dtype"] == "bfloat16" else torch.float16,
        bnb_4bit_use_double_quant=quantization_config["bnb_4bit_use_double_quant"],
    )
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    
    return model, tokenizer

def create_dpo_config(config: dict, steps_per_epoch: int):
    dpo_config = DPOConfig(
        output_dir=config["training"]["output_dir"],
        num_train_epochs=config["training"]["num_train_epochs"],
        per_device_train_batch_size=config["training"]["per_device_train_batch_size"],
        per_device_eval_batch_size=config["training"]["per_device_eval_batch_size"],
        gradient_accumulation_steps=config["training"]["gradient_accumulation_steps"],
        learning_rate=config["training"]["learning_rate"],
        lr_scheduler_type=config["training"]["lr_scheduler_type"],
        warmup_steps=config["training"]["warmup_steps"],
        logging_steps=config["training"]["logging_steps"],
        eval_strategy=config["training"]["eval_strategy"],
        eval_steps=config["training"]["eval_steps"],
        save_strategy=config["training"]["save_strategy"],
        save_steps=config["training"]["save_steps"],
        gradient_checkpointing=config["training"]["gradient_checkpointing"],
        bf16=config["training"]["bf16"],
        fp16=config["training"]["fp16"],
        max_grad_norm=config["training"]["max_grad_norm"],
        max_length=config["training"]["max_length"],
        precompute_ref_log_probs=config["training"]["precompute_ref_log_probs"],
        loss_type=[config["training"]["loss_type"]],
        beta=config["training"]["beta"],
        label_smoothing=config["training"]["label_smoothing"],
        report_to=config["training"]["report_to"],
        remove_unused_columns=config["training"]["remove_unused_columns"],
        load_best_model_at_end=config["training"]["load_best_model_at_end"],
        metric_for_best_model=config["training"]["metric_for_best_model"],
        greater_is_better=config["training"]["greater_is_better"],
        save_total_limit=config["training"]["save_total_limit"],
    )
    return dpo_config

def train_model(model, tokenizer, train_dataset, val_dataset, config, lora_config_params):
    lora_config = LoraConfig(
        r=lora_config_params["r"],
        lora_alpha=lora_config_params["lora_alpha"],
        target_modules=lora_config_params["target_modules"],
        lora_dropout=lora_config_params["lora_dropout"],
        bias=lora_config_params["bias"],
        task_type=lora_config_params["task_type"],
    )
    
    steps_per_epoch = len(train_dataset) // (config["training"]["per_device_train_batch_size"] * config["training"]["gradient_accumulation_steps"]) + 1
    dpo_config = create_dpo_config(config, steps_per_epoch)
    
    trainer = DPOTrainer(
        model=model,
        ref_model=None,
        args=dpo_config,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        processing_class=tokenizer,
        peft_config=lora_config,
        callbacks=[EarlyStoppingCallback(
            early_stopping_patience=config["training"]["early_stopping_patience"],
            early_stopping_threshold=config["training"]["early_stopping_threshold"]
        )],
    )
    
    return trainer

def log_training_metrics(trainer, mlflow_client):
    for metric_name, metric_value in trainer.state.log_history[-1].items():
        if isinstance(metric_value, (int, float)):
            mlflow_client.log_metric(metric_name, metric_value)