import pytest
import pandas as pd
import polars as pl
from unittest.mock import Mock, patch

def clean_html(text: str) -> str:
    if not text:
        return ""
    import re
    text = re.sub(r'<[^>]+>', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def test_clean_html():
    html_text = "<p>Hello <b>World</b></p>"
    cleaned = clean_html(html_text)
    assert "Hello" in cleaned
    assert "World" in cleaned
    assert "<p>" not in cleaned

def test_clean_html_empty():
    assert clean_html("") == ""
    assert clean_html(None) == ""

def test_data_filtering():
    df = pl.DataFrame({
        "Id": [1, 2, 3],
        "Body": ["text1", "text2", "text3"],
        "Score": [10, 3, 15]
    })
    
    filtered = df.filter(pl.col("Score") > 5)
    assert len(filtered) == 2
    assert filtered["Score"].to_list() == [10, 15]

def test_data_sampling():
    df = pl.DataFrame({
        "Id": range(100),
        "Score": range(100)
    })
    
    sampled = df.sort("Score", descending=True).head(10)
    assert len(sampled) == 10
    assert sampled["Score"].to_list() == list(range(99, 89, -1))

def test_qa_pair_creation():
    questions = pl.DataFrame({
        "Id": [1, 2],
        "Body": ["Question 1", "Question 2"],
        "Score": [10, 20]
    })
    
    answers = pl.DataFrame({
        "Id": [101, 102],
        "ParentId": [1, 2],
        "Body": ["Answer 1", "Answer 2"],
        "Score": [5, 15]
    })
    
    qa_pairs = answers.join(questions, left_on="ParentId", right_on="Id", how="inner")
    assert len(qa_pairs) == 2
    assert "Body_right" in qa_pairs.columns