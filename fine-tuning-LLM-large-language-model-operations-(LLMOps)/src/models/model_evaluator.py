import torch
import numpy as np
from tqdm import tqdm
from src.utils.metrics import (
    detect_language,
    compute_perplexity,
    compute_reasoning_quality,
    compute_multilingual_similarity,
    evaluate_code_quality
)
import mlflow

class ModelEvaluator:
    def __init__(self, model, tokenizer, config):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
    
    def generate_response(self, prompt: str, max_new_tokens: int = 512):
        enhanced_prompt = f"""<thinking>
Let me solve this coding problem step by step:
1. Problem Analysis:
   - Understand the requirement: {prompt}
   - Identify input/output specifications
   - Determine constraints and edge cases
2. Solution Approach:
   - Consider multiple algorithms and data structures
   - Evaluate time and space complexity trade-offs
   - Choose optimal solution based on constraints
3. Implementation Strategy:
   - Break down into logical steps
   - Handle edge cases
   - Add error handling where necessary
4. Code Optimization:
   - Review for efficiency improvements
   - Check for redundant operations
   - Ensure clean, readable code
</thinking>
<solution>
Now implementing the optimal solution with detailed reasoning:
```python
# Step-by-step implementation with comments
def solution():
    # Implementation here
    pass
</solution>
<explanation>
Complexity Analysis:
- Time Complexity: O(n) with justification
- Space Complexity: O(1) with justification
Edge Cases Covered:
- Case 1: Empty input
- Case 2: Single element
Alternative Approaches Considered:
- Approach A: Pros - simple, Cons - inefficient
- Approach B: Pros - optimal, Cons - complex
</explanation>"""
        
        prompt_text = self.tokenizer.apply_chat_template(
            [{"role": "system", "content": "You are an expert coding assistant. Always use Chain-of-Thought reasoning with structured output blocks (<thinking>, <solution>, <explanation>). Provide optimal, well-commented code."},
             {"role": "user", "content": enhanced_prompt}],
            tokenize=False,
            add_generation_prompt=True
        )
        
        inputs = self.tokenizer(prompt_text, return_tensors="pt", truncation=True, max_length=512).to(self.model.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=self.config["evaluation"]["temperature"],
                do_sample=True,
                top_p=self.config["evaluation"]["top_p"],
                top_k=self.config["evaluation"]["top_k"],
                repetition_penalty=self.config["evaluation"]["repetition_penalty"],
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        
        generated_ids = outputs[0][inputs.input_ids.shape[1]:]
        generated_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
        
        return generated_text
    
    def evaluate_on_dataset(self, test_dataset):
        self.model.eval()
        predictions = []
        references = []
        perplexities = []
        code_quality_scores = []
        reasoning_scores = []
        language_stats = {}
        
        for example in tqdm(test_dataset, desc="Evaluating"):
            original_prompt = example["prompt"][0]["content"] if isinstance(example["prompt"], list) else example["prompt"]
            generated_text = self.generate_response(original_prompt)
            
            reasoning_score, solution_content = compute_reasoning_quality(generated_text)
            reasoning_scores.append(reasoning_score)
            
            if solution_content:
                detected_lang = detect_language(solution_content)
                language_stats[detected_lang] = language_stats.get(detected_lang, 0) + 1
                code_quality = evaluate_code_quality(solution_content, detected_lang)
            else:
                code_quality = evaluate_code_quality(generated_text, "unknown")
                solution_content = generated_text
            
            code_quality_scores.append(code_quality)
            
            reference_text = self.tokenizer.apply_chat_template(
                example["chosen"],
                tokenize=False,
                add_generation_prompt=False
            ).strip()
            
            predictions.append(solution_content if solution_content else generated_text)
            references.append([reference_text])
            
            ppl = compute_perplexity(self.model, self.tokenizer, generated_text)
            perplexities.append(ppl)
        
        valid_indices = [i for i, pred in enumerate(predictions) if pred and references[i][0]]
        valid_predictions = [predictions[i] for i in valid_indices]
        valid_references = [references[i] for i in valid_indices]
        
        similarity_metrics = compute_multilingual_similarity(valid_predictions, valid_references)
        
        metrics = {
            "bleu": similarity_metrics["bleu"],
            "rouge1": similarity_metrics["rouge1"],
            "rouge2": similarity_metrics["rouge2"],
            "rougeL": similarity_metrics["rougeL"],
            "bertscore_f1": similarity_metrics["bertscore_f1"],
            "perplexity": np.mean(perplexities) if perplexities else 0.0,
            "has_thinking_ratio": np.mean([s["has_thinking"] for s in reasoning_scores]),
            "has_solution_ratio": np.mean([s["has_solution"] for s in reasoning_scores]),
            "has_explanation_ratio": np.mean([s["has_explanation"] for s in reasoning_scores]),
            "code_presence_ratio": np.mean([s["code_presence"] for s in reasoning_scores]),
            "avg_thinking_length": np.mean([s["thinking_length"] for s in reasoning_scores]),
            "avg_solution_length": np.mean([s["solution_length"] for s in reasoning_scores]),
            "avg_explanation_length": np.mean([s["explanation_length"] for s in reasoning_scores]),
            "avg_comment_ratio": np.mean([s["comment_ratio"] for s in code_quality_scores]),
            "has_imports_ratio": np.mean([s["has_imports"] for s in code_quality_scores]),
            "has_functions_ratio": np.mean([s["has_functions"] for s in code_quality_scores]),
            "has_classes_ratio": np.mean([s["has_classes"] for s in code_quality_scores]),
            "has_comments_ratio": np.mean([s["has_comments"] for s in code_quality_scores]),
            "avg_function_count": np.mean([s["function_count"] for s in code_quality_scores]),
            "avg_line_count": np.mean([s["line_count"] for s in code_quality_scores]),
            "valid_samples_ratio": len(valid_indices) / len(test_dataset) if test_dataset else 0
        }
        
        return metrics, language_stats