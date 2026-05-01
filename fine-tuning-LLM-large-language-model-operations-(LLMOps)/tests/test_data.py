import pytest
import polars as pl
from src.data.preprocessing import clean_html, create_dpo_dataset

class TestDataPreprocessing:
    def test_clean_html_with_tags(self):
        html_text = "<p>Hello <b>World</b></p>"
        cleaned = clean_html(html_text)
        assert cleaned == "Hello World"
    
    def test_clean_html_empty(self):
        assert clean_html("") == ""
        assert clean_html(None) == ""
    
    def test_clean_html_strip_whitespace(self):
        html_text = "   <p>  Text with spaces  </p>   "
        cleaned = clean_html(html_text)
        assert cleaned == "Text with spaces"
    
    def test_create_dpo_dataset_skip_single_answer(self):
        data = pl.DataFrame({
            "question_body": ["Q1", "Q1"],
            "answer_body": ["A1", "A2"],
            "answer_score": [10, 5]
        })
        
        result = create_dpo_dataset(data)
        assert len(result) == 1
    
    def test_create_dpo_dataset_skip_equal_scores(self):
        data = pl.DataFrame({
            "question_body": ["Q1", "Q1"],
            "answer_body": ["A1", "A2"],
            "answer_score": [5, 5]
        })
        
        result = create_dpo_dataset(data)
        assert len(result) == 0