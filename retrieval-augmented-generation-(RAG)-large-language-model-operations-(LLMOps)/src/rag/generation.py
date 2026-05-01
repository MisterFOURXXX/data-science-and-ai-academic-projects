import torch
import re
from typing import List, Dict, Any
from concurrent.futures import ThreadPoolExecutor

class RAGGenerator:
    def __init__(self, model, tokenizer, config):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
    
    def extract_solution(self, generated_text: str) -> str:
        solution_match = re.search(r'<solution>(.*?)</solution>', generated_text, re.DOTALL | re.IGNORECASE)
        if solution_match:
            solution_content = solution_match.group(1).strip()
            solution_content = re.sub(r'```\w*\n?|```', '', solution_content).strip()
            return solution_content
        
        thinking_match = re.search(r'<thinking>(.*?)</thinking>', generated_text, re.DOTALL | re.IGNORECASE)
        if thinking_match:
            return thinking_match.group(1).strip()
        
        return generated_text
    
    def extract_thinking(self, generated_text: str) -> str:
        thinking_match = re.search(r'<thinking>(.*?)</thinking>', generated_text, re.DOTALL | re.IGNORECASE)
        if thinking_match:
            return thinking_match.group(1).strip()
        return ""
    
    def extract_explanation(self, generated_text: str) -> str:
        explanation_match = re.search(r'<explanation>(.*?)</explanation>', generated_text, re.DOTALL | re.IGNORECASE)
        if explanation_match:
            return explanation_match.group(1).strip()
        return ""
    
    def generate_with_context(self, context: str, query: str) -> Dict[str, Any]:
        from src.rag.prompt_templates import create_prompt_text
        
        prompt_text = create_prompt_text(query, context, self.tokenizer)
        
        inputs = self.tokenizer(prompt_text, return_tensors="pt", truncation=True, max_length=1024).to(self.model.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.config.inference.max_new_tokens,
                temperature=self.config.inference.temperature,
                do_sample=True,
                top_p=self.config.inference.top_p,
                top_k=self.config.inference.top_k,
                repetition_penalty=self.config.inference.repetition_penalty,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                use_cache=True,
            )
        
        generated_ids = outputs[0][len(inputs.input_ids[0]):]
        generated_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
        
        return {
            "full_response": generated_text,
            "solution": self.extract_solution(generated_text),
            "thinking": self.extract_thinking(generated_text),
            "explanation": self.extract_explanation(generated_text)
        }
    
    def batch_generate_with_context(self, contexts: List[str], queries: List[str]) -> List[Dict[str, Any]]:
        from src.rag.prompt_templates import create_prompt_text
        
        prompt_texts = [create_prompt_text(q, c, self.tokenizer) for q, c in zip(queries, contexts)]
        
        inputs = self.tokenizer(prompt_texts, return_tensors="pt", truncation=True, padding=True, max_length=1024).to(self.model.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.config.inference.max_new_tokens,
                temperature=self.config.inference.temperature,
                do_sample=True,
                top_p=self.config.inference.top_p,
                top_k=self.config.inference.top_k,
                repetition_penalty=self.config.inference.repetition_penalty,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                use_cache=True,
            )
        
        results = []
        for i, input_ids in enumerate(inputs.input_ids):
            generated_ids = outputs[i][len(input_ids):]
            generated_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
            results.append({
                "full_response": generated_text,
                "solution": self.extract_solution(generated_text),
                "thinking": self.extract_thinking(generated_text),
                "explanation": self.extract_explanation(generated_text)
            })
        
        return results

if __name__ == "__main__":
    from config import config
    from src.model.load_model import load_fine_tuned_model
    
    model, tokenizer = load_fine_tuned_model(config)
    generator = RAGGenerator(model, tokenizer, config)
    
    test_context = "Python closures are functions that remember the environment in which they were created."
    test_query = "What is a closure in Python?"
    result = generator.generate_with_context(test_context, test_query)
    print(f"Solution: {result['solution'][:200]}...")