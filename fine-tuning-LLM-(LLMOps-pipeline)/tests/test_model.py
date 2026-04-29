import pytest
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from src.utils.metrics import compute_reasoning_quality, detect_language, evaluate_code_quality

class TestModel:
    @pytest.fixture
    def model_and_tokenizer(self):
        base_model_name = "Qwen/Qwen3-0.6B"
        adapter_path = "./qwen-dpo-final"
        
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            device_map="cpu",
            trust_remote_code=True
        )
        
        model = PeftModel.from_pretrained(base_model, adapter_path)
        tokenizer = AutoTokenizer.from_pretrained(adapter_path, trust_remote_code=True)
        
        return model, tokenizer
    
    def test_model_loading(self, model_and_tokenizer):
        model, tokenizer = model_and_tokenizer
        assert model is not None
        assert tokenizer is not None
    
    def test_model_generation(self, model_and_tokenizer):
        model, tokenizer = model_and_tokenizer
        model.eval()
        
        prompt = "How to reverse a list in Python?"
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=128)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=50,
                do_sample=False
            )
        
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        assert len(generated_text) > 0
    
    def test_reasoning_quality_extraction(self):
        test_text = """<thinking>Test thinking content</thinking>
        <solution>Test solution content</solution>
        <explanation>Test explanation content</explanation>"""
        
        reasoning_score, solution_content = compute_reasoning_quality(test_text)
        
        assert reasoning_score["has_thinking"] == 1
        assert reasoning_score["has_solution"] == 1
        assert reasoning_score["has_explanation"] == 1
    
    def test_language_detection_python(self):
        python_code = "def test_function():\n    return True"
        detected = detect_language(python_code)
        assert detected == "python"
    
    def test_language_detection_javascript(self):
        js_code = "function testFunction() {\n    return true;\n}"
        detected = detect_language(js_code)
        assert detected == "javascript"
    
    def test_code_quality_metrics(self):
        python_code = """import sys

def main():
    # This is a comment
    print("Hello World")

if __name__ == "__main__":
    main()"""
        
        metrics = evaluate_code_quality(python_code, "python")
        
        assert metrics["has_imports"] == 1
        assert metrics["has_functions"] == 1
        assert metrics["has_comments"] == 1
        assert metrics["function_count"] == 1
        assert metrics["line_count"] == 8
        assert metrics["comment_ratio"] > 0