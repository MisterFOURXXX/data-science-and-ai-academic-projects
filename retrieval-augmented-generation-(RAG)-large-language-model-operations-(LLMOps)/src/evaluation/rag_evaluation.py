import pickle
import json
import time
import math
import warnings
import numpy as np
import torch
import psutil
import os
from pathlib import Path
from evaluate import load
from sentence_transformers import SentenceTransformer, util

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

bleu_metric = load("bleu")
rouge_metric = load("rouge")
bertscore_metric = load("bertscore")

class RAGEvaluator:
    def __init__(self, config, rag_pipeline):
        self.config = config
        self.rag_pipeline = rag_pipeline
        self.semantic_model = SentenceTransformer(config.evaluation.semantic_model)
    
    def compute_perplexity(self, text: str) -> float:
        if not text or len(text.strip()) == 0:
            return 100.0
        inputs = self.rag_pipeline.tokenizer(
            text, 
            return_tensors="pt", 
            truncation=True, 
            max_length=512
        ).to(self.rag_pipeline.model.device)
        
        with torch.no_grad():
            outputs = self.rag_pipeline.model(**inputs, labels=inputs["input_ids"])
            loss = outputs.loss
        return math.exp(loss.item())
    
    def compute_semantic_similarity(self, prediction: str, reference: str) -> float:
        pred_embedding = self.semantic_model.encode(prediction, convert_to_tensor=True)
        ref_embedding = self.semantic_model.encode(reference, convert_to_tensor=True)
        return util.pytorch_cos_sim(pred_embedding, ref_embedding).item()
    
    def monitor_resources(self, func):
        def wrapper(*args, **kwargs):
            process = psutil.Process()
            cpu_start = process.cpu_percent(interval=0.5)
            memory_start = process.memory_info().rss / 1024 / 1024 / 1024
            
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
                gpu_memory_start = torch.cuda.memory_allocated() / 1024 / 1024 / 1024
            else:
                gpu_memory_start = 0
            
            start_time = time.time()
            result = func(*args, **kwargs)
            end_time = time.time()
            
            cpu_end = process.cpu_percent(interval=0.5)
            memory_end = process.memory_info().rss / 1024 / 1024 / 1024
            
            if torch.cuda.is_available():
                gpu_memory_end = torch.cuda.max_memory_allocated() / 1024 / 1024 / 1024
            else:
                gpu_memory_end = 0
            
            return result, {
                "time_s": end_time - start_time,
                "cpu_percent": cpu_end,
                "memory_gb": memory_end,
                "gpu_memory_gb": gpu_memory_end
            }
        return wrapper
    
    def safe_retrieve(self, retriever, query):
        try:
            return retriever.invoke(query)
        except Exception as e:
            print(f"Retrieval error for query: {query[:50]}... Error: {str(e)[:100]}")
            return []
    
    def evaluate_generation_metrics(self, predictions, references):
        clean_preds = [p if p.strip() else "Empty response" for p in predictions]
        clean_refs = [r if r.strip() else "Empty reference" for r in references]
        
        bleu_result = bleu_metric.compute(predictions=clean_preds, references=[[ref] for ref in clean_refs])
        rouge_result = rouge_metric.compute(predictions=clean_preds, references=clean_refs)
        bertscore_result = bertscore_metric.compute(
            predictions=clean_preds, 
            references=clean_refs, 
            lang="en", 
            model_type="bert-base-uncased"
        )
        
        return {
            "bleu": bleu_result.get('bleu', 0.0),
            "rouge1": rouge_result.get('rouge1', 0.0),
            "rouge2": rouge_result.get('rouge2', 0.0),
            "rougeL": rouge_result.get('rougeL', 0.0),
            "bertscore_precision": float(np.mean(bertscore_result['precision'])),
            "bertscore_recall": float(np.mean(bertscore_result['recall'])),
            "bertscore_f1": float(np.mean(bertscore_result['f1']))
        }
    
    def evaluate(self, test_queries, reference_answers, batch_size=8):
        results = {
            "generation_metrics": {},
            "perplexity": {},
            "semantic_similarity": {},
            "performance": {}
        }
        
        all_predictions = []
        all_perplexities = []
        all_semantic_scores = []
        retrieval_times = []
        generation_times = []
        query_times = []
        resource_metrics = []
        
        @self.monitor_resources
        def retrieve_with_monitoring(retriever, query):
            return self.safe_retrieve(retriever, query)
        
        retriever = self.rag_pipeline.vectorstore.as_retriever(
            search_kwargs={"k": self.config.vectorstore.retrieval_k}
        )
        
        total_start = time.time()
        
        for batch_start in range(0, len(test_queries), batch_size):
            batch_end = min(batch_start + batch_size, len(test_queries))
            batch_queries = test_queries[batch_start:batch_end]
            batch_references = reference_answers[batch_start:batch_end]
            
            batch_retrieved_docs = []
            for query in batch_queries:
                retrieved_docs, rm = retrieve_with_monitoring(retriever, query)
                batch_retrieved_docs.append(retrieved_docs)
                resource_metrics.append(rm)
                retrieval_times.append(rm["time_s"])
            
            generation_start = time.time()
            batch_predictions = self.rag_pipeline.batch_query(batch_queries)
            generation_time = (time.time() - generation_start) / len(batch_queries)
            generation_times.append(generation_time)
            
            all_predictions.extend(batch_predictions)
            
            for idx in range(len(batch_queries)):
                query_time = retrieval_times[-len(batch_retrieved_docs) + idx] + generation_time
                query_times.append(query_time)
            
            for pred, ref in zip(batch_predictions, batch_references):
                all_perplexities.append(self.compute_perplexity(pred))
                all_semantic_scores.append(self.compute_semantic_similarity(pred, ref))
            
            print(f"Processed {batch_end}/{len(test_queries)} queries")
        
        total_time = time.time() - total_start
        
        generation_metrics = self.evaluate_generation_metrics(all_predictions, reference_answers)
        
        results["generation_metrics"] = generation_metrics
        results["perplexity"] = {
            "mean": float(np.mean(all_perplexities)),
            "min": float(np.min(all_perplexities)),
            "max": float(np.max(all_perplexities)),
            "std": float(np.std(all_perplexities))
        }
        results["semantic_similarity"] = {
            "mean": float(np.mean(all_semantic_scores)),
            "min": float(np.min(all_semantic_scores)),
            "max": float(np.max(all_semantic_scores)),
            "std": float(np.std(all_semantic_scores))
        }
        results["performance"] = {
            "total_time_s": total_time,
            "avg_retrieval_time_s": float(np.mean(retrieval_times)),
            "avg_generation_time_s": float(np.mean(generation_times)),
            "avg_query_time_s": float(np.mean(query_times)),
            "max_query_time_s": float(max(query_times)),
            "min_query_time_s": float(min(query_times)),
            "avg_cpu_percent": float(np.mean([m["cpu_percent"] for m in resource_metrics])),
            "avg_memory_gb": float(np.mean([m["memory_gb"] for m in resource_metrics])),
            "avg_gpu_memory_gb": float(np.mean([m["gpu_memory_gb"] for m in resource_metrics])),
            "peak_cpu_percent": float(max([m["cpu_percent"] for m in resource_metrics])),
            "peak_memory_gb": float(max([m["memory_gb"] for m in resource_metrics])),
            "peak_gpu_memory_gb": float(max([m["gpu_memory_gb"] for m in resource_metrics]))
        }
        
        with open("metrics/rag_metrics.json", "w") as f:
            json.dump(results, f, indent=2)
        
        return results

def evaluate_rag():
    from src.config import config
    from src.rag.pipeline import OptimizedRAGPipeline
    
    model_path = config.training.output_dir
    if not Path(model_path).exists():
        print(f"Model not found at {model_path}. Using base model without fine-tuning.")
        model_path = config.model.base_model_name
    
    with open("data/splits/test.pkl", "rb") as f:
        test_pl = pickle.load(f)
    
    test_queries = []
    reference_answers = []
    for row in test_pl.iter_rows(named=True):
        test_queries.append(row['question_body'])
        reference_answers.append(row['answer_body'])
    
    test_queries = test_queries[:config.evaluation.test_size]
    reference_answers = reference_answers[:config.evaluation.test_size]
    
    rag_pipeline = OptimizedRAGPipeline(
        config, 
        config.vectorstore.chroma_persist_dir, 
        model_path
    )
    
    evaluator = RAGEvaluator(config, rag_pipeline)
    results = evaluator.evaluate(test_queries, reference_answers, batch_size=config.inference.batch_size)
    
    print("\n" + "="*60)
    print("RAG EVALUATION RESULTS")
    print("="*60)
    print("\nGeneration Metrics:")
    for k, v in results["generation_metrics"].items():
        print(f"  {k}: {v:.4f}")
    
    print("\nPerplexity:")
    for k, v in results["perplexity"].items():
        print(f"  {k}: {v:.4f}")
    
    print("\nSemantic Similarity:")
    for k, v in results["semantic_similarity"].items():
        print(f"  {k}: {v:.4f}")
    
    print("\nPerformance:")
    for k, v in results["performance"].items():
        print(f"  {k}: {v:.4f}")
    
    return results

if __name__ == "__main__":
    evaluate_rag()