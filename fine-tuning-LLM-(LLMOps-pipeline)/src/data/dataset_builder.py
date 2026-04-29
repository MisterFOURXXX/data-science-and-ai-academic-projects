import polars as pl
import numpy as np
from datasets import Dataset, DatasetDict
from sklearn.model_selection import train_test_split
from src.data.preprocessing import clean_html, format_conversational
import os

class DatasetBuilder:
    def __init__(self, config):
        self.config = config
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
    
    def build_from_csv(self, questions_path, answers_path):
        print(f"Loading questions from {questions_path}")
        questions = pl.read_csv(
            questions_path,
            encoding="utf8-lossy",
            columns=["Id", "Body", "Score"]
        ).filter(pl.col("Score") > self.config["data"]["score_threshold"])
        
        print(f"Loading answers from {answers_path}")
        answers = pl.read_csv(
            answers_path,
            encoding="utf8-lossy",
            columns=["Id", "ParentId", "Body", "Score"]
        ).filter(pl.col("Score") > self.config["data"]["score_threshold"])
        
        print(f"Sampling top {self.config['data']['sample_size']} questions...")
        questions = questions.sort("Score", descending=True).head(self.config["data"]["sample_size"])
        
        print("Cleaning HTML content...")
        questions = questions.with_columns(
            pl.col("Body").map_elements(clean_html, return_dtype=pl.Utf8)
        )
        answers = answers.with_columns(
            pl.col("Body").map_elements(clean_html, return_dtype=pl.Utf8)
        )
        
        print("Joining questions and answers...")
        qa_pairs = answers.join(
            questions, left_on="ParentId", right_on="Id", how="inner"
        ).select([
            pl.col("Body_right").alias("question_body"),
            pl.col("Body").alias("answer_body"),
            pl.col("Score").alias("answer_score")
        ])
        
        print("Creating DPO dataset...")
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
        
        dpo_pl = pl.DataFrame(records)
        print(f"Created {len(dpo_pl)} DPO examples")
        
        print("Splitting dataset...")
        train_pl, temp_pl = train_test_split(dpo_pl.to_pandas(), test_size=0.2, random_state=self.config["project"]["seed"])
        val_pl, test_pl = train_test_split(temp_pl, test_size=0.5, random_state=self.config["project"]["seed"])
        
        print("Converting to conversational format...")
        self.train_dataset = Dataset.from_pandas(train_pl).map(format_conversational)
        self.val_dataset = Dataset.from_pandas(val_pl).map(format_conversational)
        self.test_dataset = Dataset.from_pandas(test_pl).map(format_conversational)
        
        return DatasetDict({
            "train": self.train_dataset,
            "validation": self.val_dataset,
            "test": self.test_dataset
        })
    
    def get_dataset_splits(self):
        return self.train_dataset, self.val_dataset, self.test_dataset
    
    def save_datasets(self, output_dir):
        os.makedirs(output_dir, exist_ok=True)
        
        if self.train_dataset:
            self.train_dataset.save_to_disk(os.path.join(output_dir, "train"))
            print(f"Saved train dataset to {output_dir}/train")
        if self.val_dataset:
            self.val_dataset.save_to_disk(os.path.join(output_dir, "validation"))
            print(f"Saved validation dataset to {output_dir}/validation")
        if self.test_dataset:
            self.test_dataset.save_to_disk(os.path.join(output_dir, "test"))
            print(f"Saved test dataset to {output_dir}/test")