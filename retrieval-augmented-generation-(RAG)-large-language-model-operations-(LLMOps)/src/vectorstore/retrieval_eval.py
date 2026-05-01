import pickle
import json
import time
import numpy as np
import hashlib
import psutil
import torch
import sys
import os
from pathlib import Path
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.config import config

def monitor_resources(func):
    def wrapper(*args, **kwargs):
        process = psutil.Process()
        cpu_start = process.cpu_percent(interval=0.1)
        memory_start = process.memory_info().rss / 1024 / 1024
        
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            gpu_memory_start = torch.cuda.memory_allocated() / 1024 / 1024
        else:
            gpu_memory_start = 0
        
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        
        cpu_end = process.cpu_percent(interval=0.1)
        memory_end = process.memory_info().rss / 1024 / 1024
        
        if torch.cuda.is_available():
            gpu_memory_end = torch.cuda.max_memory_allocated() / 1024 / 1024
        else:
            gpu_memory_end = 0
        
        return result, {
            "time_ms": (end_time - start_time) * 1000,
            "cpu_percent": cpu_end - cpu_start,
            "memory_mb": memory_end - memory_start,
            "gpu_memory_mb": gpu_memory_end - gpu_memory_start
        }
    return wrapper

def calculate_hit_rate(retrieved_docs, relevant_ids, k=5):
    retrieved_at_k = retrieved_docs[:k]
    for doc in retrieved_at_k:
        if str(doc.metadata.get('question_id')) in relevant_ids:
            return 1.0
    return 0.0

def calculate_mrr(retrieved_docs, relevant_ids):
    for rank, doc in enumerate(retrieved_docs, 1):
        if str(doc.metadata.get('question_id')) in relevant_ids:
            return 1.0 / rank
    return 0.0

def calculate_map_at_k(retrieved_docs, relevant_ids, k=10):
    relevant_count = 0
    sum_precision = 0
    for i, doc in enumerate(retrieved_docs[:k], 1):
        if str(doc.metadata.get('question_id')) in relevant_ids:
            relevant_count += 1
            sum_precision += relevant_count / i
    return sum_precision / min(len(relevant_ids), k) if relevant_ids else 0

def calculate_ndcg_at_k(retrieved_docs, relevant_ids, k=10):
    relevance_scores = [1 if str(doc.metadata.get('question_id')) in relevant_ids else 0 for doc in retrieved_docs[:k]]
    if sum(relevance_scores) == 0:
        return 0.0
    dcg = sum([rel / np.log2(idx + 2) for idx, rel in enumerate(relevance_scores)])
    ideal_scores = sorted(relevance_scores, reverse=True)
    idcg = sum([1 / np.log2(idx + 2) for idx in range(sum(relevance_scores))])
    return dcg / idcg if idcg > 0 else 0

def evaluate_retrieval():
    embedding_model = HuggingFaceEmbeddings(
        model_name=config.vectorstore.embedding_model,
        model_kwargs={'device': 'cuda' if torch.cuda.is_available() else 'cpu'},
        encode_kwargs={'normalize_embeddings': True},
    )
    
    vectorstore = Chroma(
        persist_directory=config.vectorstore.chroma_persist_dir,
        embedding_function=embedding_model,
        collection_name="stackoverflow_coding_train"
    )
    
    with open("data/splits/val.pkl", "rb") as f:
        val_pl = pickle.load(f)
    
    retriever = vectorstore.as_retriever(search_kwargs={"k": config.vectorstore.retrieval_k})
    
    hit_rates = {k: [] for k in config.evaluation.retrieval_k_values}
    mrr_scores = []
    map_scores = []
    ndcg_scores = []
    query_times = []
    resource_metrics = []
    
    @monitor_resources
    def retrieve_with_monitoring(query):
        return retriever.invoke(query)
    
    for row in val_pl.iter_rows(named=True):
        query = row['question_body']
        ground_truth_id = hashlib.md5(query.encode()).hexdigest()
        relevant_ids = [str(ground_truth_id)]
        
        retrieved_docs, metrics = retrieve_with_monitoring(query)
        query_times.append(metrics["time_ms"])
        resource_metrics.append(metrics)
        
        mrr_scores.append(calculate_mrr(retrieved_docs, relevant_ids))
        map_scores.append(calculate_map_at_k(retrieved_docs, relevant_ids, k=10))
        ndcg_scores.append(calculate_ndcg_at_k(retrieved_docs, relevant_ids, k=10))
        
        for k in config.evaluation.retrieval_k_values:
            hit_rates[k].append(calculate_hit_rate(retrieved_docs, relevant_ids, k=k))
    
    results = {
        "hit_rates": {f"hit_rate@{k}": float(np.mean(v)) for k, v in hit_rates.items()},
        "mrr": float(np.mean(mrr_scores)),
        "map_at_10": float(np.mean(map_scores)),
        "ndcg_at_10": float(np.mean(ndcg_scores)),
        "avg_query_time_ms": float(np.mean(query_times)),
        "max_query_time_ms": float(max(query_times)),
        "min_query_time_ms": float(min(query_times)),
        "total_queries": len(val_pl),
        "avg_cpu_percent": float(np.mean([m["cpu_percent"] for m in resource_metrics])),
        "avg_memory_mb": float(np.mean([m["memory_mb"] for m in resource_metrics])),
        "avg_gpu_memory_mb": float(np.mean([m["gpu_memory_mb"] for m in resource_metrics]))
    }
    
    with open("metrics/retrieval_metrics.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print("\nRetrieval Evaluation Results:")
    for metric, value in results.items():
        if isinstance(value, dict):
            for k, v in value.items():
                print(f"  {k}: {v:.4f}")
        elif isinstance(value, float):
            print(f"  {metric}: {value:.4f}")
        else:
            print(f"  {metric}: {value}")
    
    return results

if __name__ == "__main__":
    evaluate_retrieval()