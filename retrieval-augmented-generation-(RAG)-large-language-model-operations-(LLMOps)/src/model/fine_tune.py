import pickle
import json
import torch
import sys
import os
from pathlib import Path
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    EarlyStoppingCallback
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import DPOTrainer, DPOConfig

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.config import config

def format_dpo_example(example, tokenizer):
    system_message = "You are an expert coding assistant."
    
    prompt_messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": example["question_body"]}
    ]
    
    chosen_messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": example["question_body"]},
        {"role": "assistant", "content": example["answer_body"]}
    ]
    
    rejected_messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": example["question_body"]},
        {"role": "assistant", "content": "I cannot answer this question."}
    ]
    
    prompt = tokenizer.apply_chat_template(prompt_messages, tokenize=False, add_generation_prompt=True)
    chosen = tokenizer.apply_chat_template(chosen_messages, tokenize=False, add_generation_prompt=False)
    rejected = tokenizer.apply_chat_template(rejected_messages, tokenize=False, add_generation_prompt=False)
    
    return {
        "prompt": prompt,
        "chosen": chosen,
        "rejected": rejected,
    }

def fine_tune_model():
    Path(config.training.output_dir).mkdir(parents=True, exist_ok=True)
    Path("metrics").mkdir(parents=True, exist_ok=True)
    
    with open("data/splits/train.pkl", "rb") as f:
        train_pl = pickle.load(f)
    
    with open("data/splits/val.pkl", "rb") as f:
        val_pl = pickle.load(f)
    
    train_df = train_pl.to_pandas()
    val_df = val_pl.to_pandas()
    
    tokenizer = AutoTokenizer.from_pretrained(config.model.base_model_name, trust_remote_code=True)
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    tokenizer.truncation_side = "left"
    
    def apply_format(example):
        return format_dpo_example(example, tokenizer)
    
    train_dataset = Dataset.from_pandas(train_df).map(apply_format)
    val_dataset = Dataset.from_pandas(val_df).map(apply_format)
    
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=config.model.quantization_4bit,
        bnb_4bit_quant_type=config.model.bnb_4bit_quant_type,
        bnb_4bit_compute_dtype=getattr(torch, config.model.bnb_4bit_compute_dtype),
        bnb_4bit_use_double_quant=config.model.bnb_4bit_use_double_quant,
    )
    
    model = AutoModelForCausalLM.from_pretrained(
        config.model.base_model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    )
    
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.bos_token_id = tokenizer.bos_token_id
    model.config.eos_token_id = tokenizer.eos_token_id
    
    model = prepare_model_for_kbit_training(model)
    
    lora_config = LoraConfig(
        r=config.model.lora_r,
        lora_alpha=config.model.lora_alpha,
        target_modules=config.model.lora_target_modules,
        lora_dropout=config.model.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )
    
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    steps_per_epoch = len(train_dataset) // (config.training.batch_size * config.training.gradient_accumulation_steps) + 1
    eval_steps = max(1, steps_per_epoch // 2)
    
    dpo_config = DPOConfig(
        output_dir=config.training.output_dir,
        num_train_epochs=config.training.epochs,
        per_device_train_batch_size=config.training.batch_size,
        per_device_eval_batch_size=config.training.batch_size,
        gradient_accumulation_steps=config.training.gradient_accumulation_steps,
        learning_rate=config.training.learning_rate,
        lr_scheduler_type="cosine",
        warmup_ratio=config.training.warmup_ratio,
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=eval_steps,
        save_strategy="steps",
        save_steps=eval_steps,
        save_total_limit=3,
        gradient_checkpointing=config.training.gradient_checkpointing,
        bf16=True,
        fp16=False,
        max_grad_norm=1.0,
        max_length=config.training.max_seq_length,
        beta=config.training.dpo_beta,
        loss_type="sigmoid",
        label_smoothing=0.0,
        report_to="none",
        remove_unused_columns=False,
        dataloader_num_workers=0,
        logging_first_step=True,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
    )
    
    trainer = DPOTrainer(
        model=model,
        ref_model=None,
        args=dpo_config,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        processing_class=tokenizer,
        callbacks=[EarlyStoppingCallback(
            early_stopping_patience=config.training.early_stopping_patience,
            early_stopping_threshold=config.training.early_stopping_threshold
        )],
    )
    
    trainer.train()
    
    trainer.save_model(config.training.output_dir)
    tokenizer.save_pretrained(config.training.output_dir)
    
    train_losses = [log["loss"] for log in trainer.state.log_history if "loss" in log]
    eval_losses = [log["eval_loss"] for log in trainer.state.log_history if "eval_loss" in log]
    
    metrics = {
        "final_train_loss": float(train_losses[-1]) if train_losses else None,
        "final_eval_loss": float(eval_losses[-1]) if eval_losses else None,
        "best_eval_loss": float(min(eval_losses)) if eval_losses else None,
        "total_training_steps": trainer.state.global_step,
        "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad)
    }
    
    with open("metrics/training_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    
    print(f"Training completed. Best eval loss: {metrics['best_eval_loss']:.4f}")
    return metrics

if __name__ == "__main__":
    fine_tune_model()