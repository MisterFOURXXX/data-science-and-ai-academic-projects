import re
import math
import torch
import numpy as np
from evaluate import load

bleu_metric = load("bleu")
rouge_metric = load("rouge")
bertscore_metric = load("bertscore")

def detect_language(code: str) -> str:
    if not code:
        return "unknown"
    
    language_patterns = {
        "python": [r"def\s+\w+\(", r"import\s+\w+", r"from\s+\w+\s+import", r"print\(", r"if\s+__name__", r"self\."],
        "javascript": [r"function\s+\w+\s*\(", r"const\s+\w+\s*=", r"let\s+\w+\s*=", r"console\.log", r"=>\s*{", r"document\."],
        "java": [r"public\s+class\s+\w+", r"public\s+static\s+void\s+main", r"System\.out\.println", r"import\s+java\.", r"@Override"],
        "cpp": [r"#include\s+<", r"int\s+main\s*\(", r"std::", r"cout\s*<<", r"class\s+\w+\s*{", r"->"],
        "go": [r"func\s+\w+\s*\(", r"package\s+main", r"import\s+\(", r"go\s+func", r"chan\s+", r"defer\s+"],
        "rust": [r"fn\s+\w+\s*\(", r"let\s+mut", r"println!", r"impl\s+\w+", r"use\s+std::", r"->\s+\w+", r"match\s+{"],
        "sql": [r"SELECT\s+", r"INSERT\s+INTO", r"UPDATE\s+", r"DELETE\s+FROM", r"CREATE\s+TABLE", r"JOIN\s+"],
    }
    
    scores = {}
    for lang, patterns in language_patterns.items():
        score = sum(1 for pattern in patterns if re.search(pattern, code, re.IGNORECASE))
        if score > 0:
            scores[lang] = score
    
    if scores:
        return max(scores, key=scores.get)
    return "unknown"

def compute_perplexity(model, tokenizer, text, max_length=512):
    model.eval()
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length).to(model.device)
    with torch.no_grad():
        outputs = model(**inputs, labels=inputs["input_ids"])
        loss = outputs.loss
    perplexity = math.exp(loss.item())
    return perplexity

def compute_reasoning_quality(generated_text: str):
    thinking_match = re.search(r'<thinking>(.*?)</thinking>', generated_text, re.DOTALL | re.IGNORECASE)
    solution_match = re.search(r'<solution>(.*?)</solution>', generated_text, re.DOTALL | re.IGNORECASE)
    explanation_match = re.search(r'<explanation>(.*?)</explanation>', generated_text, re.DOTALL | re.IGNORECASE)
    
    thinking_content = thinking_match.group(1).strip() if thinking_match else ""
    solution_content = solution_match.group(1).strip() if solution_match else ""
    explanation_content = explanation_match.group(1).strip() if explanation_match else ""
    
    if solution_content:
        solution_content = re.sub(r'```\w*\n?|```', '', solution_content).strip()
    
    reasoning_score = {
        "has_thinking": 1 if thinking_content else 0,
        "has_solution": 1 if solution_content else 0,
        "has_explanation": 1 if explanation_content else 0,
        "thinking_length": len(thinking_content.split()),
        "solution_length": len(solution_content.split()),
        "explanation_length": len(explanation_content.split()),
        "code_presence": 1 if solution_content else 0
    }
    
    return reasoning_score, solution_content

def compute_multilingual_similarity(predictions, references):
    all_predictions = [p for p, r in zip(predictions, references) if p and r[0]]
    all_references = [r[0] for p, r in zip(predictions, references) if p and r[0]]
    
    if not all_predictions:
        return {"bleu": 0.0, "rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0, "bertscore_f1": 0.0}
    
    bleu = bleu_metric.compute(predictions=all_predictions, references=[[r] for r in all_references])
    rouge = rouge_metric.compute(predictions=all_predictions, references=all_references)
    bertscore = bertscore_metric.compute(predictions=all_predictions, references=all_references, lang="en")
    
    return {
        "bleu": bleu.get("bleu", 0.0),
        "rouge1": rouge.get("rouge1", 0.0),
        "rouge2": rouge.get("rouge2", 0.0),
        "rougeL": rouge.get("rougeL", 0.0),
        "bertscore_f1": np.mean(bertscore["f1"]) if bertscore["f1"] else 0.0
    }

def evaluate_code_quality(code: str, language: str):
    metrics = {
        "has_imports": 0,
        "has_functions": 0,
        "has_classes": 0,
        "has_comments": 0,
        "function_count": 0,
        "line_count": len(code.split('\n')) if code else 0,
        "comment_ratio": 0,
        "detected_language": language
    }
    
    if not code:
        return metrics
    
    lines = code.split('\n')
    
    comment_patterns = {
        "python": r"^\s*#",
        "javascript": r"^\s*//",
        "java": r"^\s*//",
        "cpp": r"^\s*//",
        "rust": r"^\s*//",
        "sql": r"^\s*--",
    }
    
    pattern = comment_patterns.get(language, r"^\s*#|^\s*//")
    comment_lines = sum(1 for line in lines if re.search(pattern, line))
    metrics["comment_ratio"] = comment_lines / len(lines) if lines else 0
    metrics["has_comments"] = 1 if comment_lines > 0 else 0
    
    function_patterns = {
        "python": r"^\s*def\s+\w+\s*\(",
        "javascript": r"^\s*function\s+\w+\s*\(|^\s*const\s+\w+\s*=\s*\(.*\)\s*=>",
        "java": r"^\s*(public|private|protected)?\s*(static)?\s*\w+\s+\w+\s*\(",
        "cpp": r"^\s*\w+\s+\w+\s*\(",
        "rust": r"^\s*fn\s+\w+\s*\(",
    }
    
    pattern = function_patterns.get(language, r"^\s*def\s+\w+\s*\(|^\s*function\s+\w+\s*\(")
    metrics["has_functions"] = 1 if any(re.search(pattern, line) for line in lines) else 0
    metrics["function_count"] = sum(1 for line in lines if re.search(pattern, line))
    
    class_patterns = {
        "python": r"^\s*class\s+\w+",
        "java": r"^\s*(public|private|protected)?\s*class\s+\w+",
    }
    
    pattern = class_patterns.get(language, r"^\s*class\s+\w+")
    metrics["has_classes"] = 1 if any(re.search(pattern, line) for line in lines) else 0
    
    import_patterns = {
        "python": r"^\s*(import|from)\s+\w+",
        "javascript": r"^\s*(import|const|var|let).*require|^\s*import\s+.*from",
        "java": r"^\s*import\s+",
        "cpp": r"^\s*#include",
        "rust": r"^\s*use\s+",
    }
    
    pattern = import_patterns.get(language, r"^\s*(import|#include|using)")
    metrics["has_imports"] = 1 if any(re.search(pattern, line) for line in lines) else 0
    
    return metrics