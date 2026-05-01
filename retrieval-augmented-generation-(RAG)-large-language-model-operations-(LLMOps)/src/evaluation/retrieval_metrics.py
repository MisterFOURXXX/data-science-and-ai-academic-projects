import numpy as np
from typing import List, Dict, Any

class RetrievalMetrics:
    @staticmethod
    def hit_rate(retrieved_ids: List[str], relevant_ids: List[str], k: int = 5) -> float:
        retrieved_at_k = retrieved_ids[:k]
        for rid in retrieved_at_k:
            if rid in relevant_ids:
                return 1.0
        return 0.0
    
    @staticmethod
    def reciprocal_rank(retrieved_ids: List[str], relevant_ids: List[str]) -> float:
        for rank, rid in enumerate(retrieved_ids, 1):
            if rid in relevant_ids:
                return 1.0 / rank
        return 0.0
    
    @staticmethod
    def mean_average_precision(retrieved_ids: List[str], relevant_ids: List[str], k: int = 10) -> float:
        if not relevant_ids:
            return 0.0
        
        relevant_count = 0
        sum_precision = 0
        
        for i, rid in enumerate(retrieved_ids[:k], 1):
            if rid in relevant_ids:
                relevant_count += 1
                sum_precision += relevant_count / i
        
        return sum_precision / min(len(relevant_ids), k)
    
    @staticmethod
    def ndcg_at_k(retrieved_ids: List[str], relevant_ids: List[str], k: int = 10) -> float:
        relevance_scores = [1 if rid in relevant_ids else 0 for rid in retrieved_ids[:k]]
        
        if sum(relevance_scores) == 0:
            return 0.0
        
        dcg = sum([rel / np.log2(idx + 2) for idx, rel in enumerate(relevance_scores)])
        
        ideal_scores = sorted(relevance_scores, reverse=True)
        idcg = sum([1 / np.log2(idx + 2) for idx in range(len(ideal_scores))])
        
        return dcg / idcg if idcg > 0 else 0.0
    
    @staticmethod
    def precision_at_k(retrieved_ids: List[str], relevant_ids: List[str], k: int = 5) -> float:
        retrieved_at_k = retrieved_ids[:k]
        relevant_retrieved = sum(1 for rid in retrieved_at_k if rid in relevant_ids)
        return relevant_retrieved / k
    
    @staticmethod
    def recall_at_k(retrieved_ids: List[str], relevant_ids: List[str], k: int = 10) -> float:
        retrieved_at_k = retrieved_ids[:k]
        relevant_retrieved = sum(1 for rid in retrieved_at_k if rid in relevant_ids)
        return relevant_retrieved / len(relevant_ids) if relevant_ids else 0.0
    
    @staticmethod
    def evaluate_all(retrieved_ids: List[str], relevant_ids: List[str], k_values: List[int] = [1, 3, 5, 10]) -> Dict[str, float]:
        metrics = {}
        
        for k in k_values:
            metrics[f"hit_rate@{k}"] = RetrievalMetrics.hit_rate(retrieved_ids, relevant_ids, k)
            metrics[f"precision@{k}"] = RetrievalMetrics.precision_at_k(retrieved_ids, relevant_ids, k)
            metrics[f"recall@{k}"] = RetrievalMetrics.recall_at_k(retrieved_ids, relevant_ids, k)
        
        metrics["mrr"] = RetrievalMetrics.reciprocal_rank(retrieved_ids, relevant_ids)
        metrics["map@10"] = RetrievalMetrics.mean_average_precision(retrieved_ids, relevant_ids, 10)
        metrics["ndcg@10"] = RetrievalMetrics.ndcg_at_k(retrieved_ids, relevant_ids, 10)
        
        return metrics

if __name__ == "__main__":
    retrieved = ["doc1", "doc2", "doc3", "doc4", "doc5"]
    relevant = ["doc1", "doc3"]
    
    metrics = RetrievalMetrics.evaluate_all(retrieved, relevant)
    for name, value in metrics.items():
        print(f"{name}: {value:.4f}")