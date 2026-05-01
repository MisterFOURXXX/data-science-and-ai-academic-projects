import torch
import time
import numpy as np
from typing import List, Dict, Any

class ModelInference:
    def __init__(self, model, tokenizer, config):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
    
    def generate_single(self, prompt: str, **kwargs) -> Dict[str, Any]:
        max_new_tokens = kwargs.get('max_new_tokens', self.config.inference.max_new_tokens)
        temperature = kwargs.get('temperature', self.config.inference.temperature)
        top_p = kwargs.get('top_p', self.config.inference.top_p)
        top_k = kwargs.get('top_k', self.config.inference.top_k)
        
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024).to(self.model.device)
        
        start_time = time.time()
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=True,
                top_p=top_p,
                top_k=top_k,
                repetition_penalty=self.config.inference.repetition_penalty,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                use_cache=True,
            )
        inference_time = (time.time() - start_time) * 1000
        
        generated_ids = outputs[0][len(inputs.input_ids[0]):]
        generated_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
        
        return {
            "generated_text": generated_text,
            "inference_time_ms": inference_time,
            "num_tokens": len(generated_ids)
        }
    
    def generate_batch(self, prompts: List[str], **kwargs) -> List[Dict[str, Any]]:
        max_new_tokens = kwargs.get('max_new_tokens', self.config.inference.max_new_tokens)
        temperature = kwargs.get('temperature', self.config.inference.temperature)
        
        inputs = self.tokenizer(prompts, return_tensors="pt", truncation=True, padding=True, max_length=1024).to(self.model.device)
        
        start_time = time.time()
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=True,
                top_p=self.config.inference.top_p,
                top_k=self.config.inference.top_k,
                repetition_penalty=self.config.inference.repetition_penalty,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                use_cache=True,
            )
        total_time = (time.time() - start_time) * 1000
        
        results = []
        for i, input_ids in enumerate(inputs.input_ids):
            generated_ids = outputs[i][len(input_ids):]
            generated_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
            results.append({
                "generated_text": generated_text,
                "inference_time_ms": total_time / len(prompts),
                "num_tokens": len(generated_ids)
            })
        
        return results
    
    def compute_perplexity(self, text: str) -> float:
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(self.model.device)
        with torch.no_grad():
            outputs = self.model(**inputs, labels=inputs["input_ids"])
            loss = outputs.loss
        return torch.exp(loss).item()

if __name__ == "__main__":
    from config import config
    from load_model import load_fine_tuned_model
    
    model, tokenizer = load_fine_tuned_model(config)
    inference = ModelInference(model, tokenizer, config)
    
    test_prompt = "What is a closure in Python?"
    result = inference.generate_single(test_prompt)
    print(f"Generated: {result['generated_text'][:200]}...")
    print(f"Inference time: {result['inference_time_ms']:.2f} ms")