import polars as pl
from bs4 import BeautifulSoup
from sklearn.model_selection import train_test_split
from datasets import Dataset
import re

def clean_html(text: str) -> str:
    if not text:
        return ""
    soup = BeautifulSoup(text, "html.parser")
    return soup.get_text(separator=" ", strip=True)

def load_and_filter_data(questions_path: str, answers_path: str, score_threshold: int = 5, sample_size: int = 1000):
    questions = pl.read_csv(
        questions_path,
        encoding="utf8-lossy",
        columns=["Id", "Body", "Score"]
    ).filter(pl.col("Score") > score_threshold)
    
    answers = pl.read_csv(
        answers_path,
        encoding="utf8-lossy",
        columns=["Id", "ParentId", "Body", "Score"]
    ).filter(pl.col("Score") > score_threshold)
    
    questions = questions.sort("Score", descending=True).head(sample_size)
    
    questions = questions.with_columns(
        pl.col("Body").map_elements(clean_html, return_dtype=pl.Utf8)
    )
    answers = answers.with_columns(
        pl.col("Body").map_elements(clean_html, return_dtype=pl.Utf8)
    )
    
    return questions, answers

def join_qa_pairs(answers, questions):
    qa_pairs = answers.join(
        questions, left_on="ParentId", right_on="Id", how="inner"
    ).select([
        pl.col("Body_right").alias("question_body"),
        pl.col("Body").alias("answer_body"),
        pl.col("Score").alias("answer_score")
    ])
    return qa_pairs

def create_dpo_dataset(qa_pairs: pl.DataFrame):
    records = []
    for question_id, group in qa_pairs.group_by("question_body"):
        if group.height < 2:
            continue
        best = group.filter(pl.col("answer_score") == pl.max("answer_score")).head(1)
        worst = group.filter(pl.col("answer_score") == pl.min("answer_score")).head(1)
        if best["answer_score"][0] == worst["answer_score"][0]:
            continue
        records.append({
            "prompt": best["question_body"][0],
            "chosen": best["answer_body"][0],
            "rejected": worst["answer_body"][0]
        })
    return pl.DataFrame(records)

def format_conversational(example):
    return {
        "prompt": [{"role": "user", "content": example["prompt"]}],
        "chosen": [{"role": "assistant", "content": example["chosen"]}],
        "rejected": [{"role": "assistant", "content": example["rejected"]}],
    }

def prepare_datasets(dpo_pl, test_size=0.2, val_split=0.5, random_state=42):
    train_pl, temp_pl = train_test_split(dpo_pl.to_pandas(), test_size=test_size, random_state=random_state)
    val_pl, test_pl = train_test_split(temp_pl, test_size=val_split, random_state=random_state)
    
    train_dataset = Dataset.from_pandas(train_pl)
    val_dataset = Dataset.from_pandas(val_pl)
    test_dataset = Dataset.from_pandas(test_pl)
    
    train_dataset = train_dataset.map(format_conversational)
    val_dataset = val_dataset.map(format_conversational)
    test_dataset = test_dataset.map(format_conversational)
    
    return train_dataset, val_dataset, test_dataset