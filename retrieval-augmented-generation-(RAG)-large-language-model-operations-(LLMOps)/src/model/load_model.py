import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
from pathlib import Path

def load_fine_tuned_model(config):
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=config.model.quantization_4bit,
        bnb_4bit_quant_type=config.model.bnb_4bit_quant_type,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=config.model.bnb_4bit_use_double_quant,
    )
    
    base_model = AutoModelForCausalLM.from_pretrained(
        config.model.base_model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    )
    
    tokenizer = AutoTokenizer.from_pretrained(config.training.output_dir, trust_remote_code=True)
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    tokenizer.truncation_side = "left"
    
    model = PeftModel.from_pretrained(base_model, config.training.output_dir)
    model.eval()
    
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.bos_token_id = tokenizer.bos_token_id
    model.config.eos_token_id = tokenizer.eos_token_id
    
    print(f"Model loaded successfully. Device: {model.device}")
    print(f"Model dtype: {model.dtype}")
    
    return model, tokenizer

if __name__ == "__main__":
    from config import config
    model, tokenizer = load_fine_tuned_model(config)