import pickle
import json
import polars as pl
import sys
import os
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import config

def preprocess_qa_pairs():
    Path("data/processed").mkdir(parents=True, exist_ok=True)
    Path("metrics").mkdir(parents=True, exist_ok=True)
    
    with open("data/raw/questions_filtered.pkl", "rb") as f:
        questions = pickle.load(f)
    
    with open("data/raw/answers_filtered.pkl", "rb") as f:
        answers = pickle.load(f)
    
    qa_pairs = answers.join(
        questions, left_on="ParentId", right_on="Id", how="inner"
    ).select([
        pl.col("Body_right").alias("question_body"),
        pl.col("Body").alias("answer_body"),
        pl.col("Score").alias("answer_score")
    ])
    
    with open("data/processed/qa_pairs.pkl", "wb") as f:
        pickle.dump(qa_pairs, f)
    
    metrics = {
        "total_qa_pairs": len(qa_pairs),
        "avg_question_length": float(qa_pairs["question_body"].str.len_bytes().mean()),
        "avg_answer_length": float(qa_pairs["answer_body"].str.len_bytes().mean()),
        "avg_answer_score": float(qa_pairs["answer_score"].mean()),
        "max_answer_score": int(qa_pairs["answer_score"].max()),
        "min_answer_score": int(qa_pairs["answer_score"].min())
    }
    
    with open("metrics/preprocess_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    
    print(f"Created {len(qa_pairs)} QA pairs")
    return metrics

if __name__ == "__main__":
    preprocess_qa_pairs()