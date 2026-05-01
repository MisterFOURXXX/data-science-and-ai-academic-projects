import numpy as np
from evaluate import load
from typing import List, Dict, Any

bleu_metric = load("bleu")
rouge_metric = load("rouge")
bertscore_metric = load("bertscore")

class GenerationMetrics:
    def __init__(self):
        self.bleu = bleu_metric
        self.rouge = rouge_metric
        self.bertscore = bertscore_metric
    
    def compute_bleu(self, predictions: List[str], references: List[str]) -> float:
        clean_preds = [p if p.strip() else "empty" for p in predictions]
        clean_refs = [[r if r.strip() else "empty"] for r in references]
        result = self.bleu.compute(predictions=clean_preds, references=clean_refs)
        return result.get('bleu', 0.0)
    
    def compute_rouge(self, predictions: List[str], references: List[str]) -> Dict[str, float]:
        clean_preds = [p if p.strip() else "empty" for p in predictions]
        clean_refs = [r if r.strip() else "empty" for r in references]
        result = self.rouge.compute(predictions=clean_preds, references=clean_refs)
        return {
            'rouge1': result.get('rouge1', 0.0),
            'rouge2': result.get('rouge2', 0.0),
            'rougeL': result.get('rougeL', 0.0)
        }
    
    def compute_bertscore(self, predictions: List[str], references: List[str], model_type: str = "bert-base-uncased") -> Dict[str, float]:
        clean_preds = [p if p.strip() else "empty" for p in predictions]
        clean_refs = [r if r.strip() else "empty" for r in references]
        result = self.bertscore.compute(predictions=clean_preds, references=clean_refs, lang="en", model_type=model_type)
        return {
            'precision': float(np.mean(result['precision'])),
            'recall': float(np.mean(result['recall'])),
            'f1': float(np.mean(result['f1']))
        }
    
    def evaluate_all(self, predictions: List[str], references: List[str]) -> Dict[str, float]:
        metrics = {}
        
        metrics['bleu'] = self.compute_bleu(predictions, references)
        
        rouge_scores = self.compute_rouge(predictions, references)
        metrics.update(rouge_scores)
        
        bert_scores = self.compute_bertscore(predictions, references)
        metrics['bertscore_precision'] = bert_scores['precision']
        metrics['bertscore_recall'] = bert_scores['recall']
        metrics['bertscore_f1'] = bert_scores['f1']
        
        return metrics

if __name__ == "__main__":
    predictions = ["This is a test prediction.", "Another test."]
    references = ["This is a test reference.", "Another reference."]
    
    metrics = GenerationMetrics()
    results = metrics.evaluate_all(predictions, references)
    for name, value in results.items():
        print(f"{name}: {value:.4f}")