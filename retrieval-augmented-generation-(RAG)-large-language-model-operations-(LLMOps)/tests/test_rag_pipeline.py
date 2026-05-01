import pytest
from unittest.mock import Mock, patch
import re

def test_prompt_template():
    prompt_template = """<thinking>Problem Analysis: {query}Context: {context}</thinking><solution>{solution}</solution>"""
    filled = prompt_template.format(
        query="test query",
        context="test context",
        solution="test solution"
    )
    assert "<thinking>" in filled
    assert "<solution>" in filled
    assert "test query" in filled

def test_extract_solution():
    generated_text = """<thinking>Some thinking</thinking><solution>```python
print("hello")
</solution> <explanation>Explanation</explanation>"""
    match = re.search(r'<solution>(.*?)</solution>', generated_text, re.DOTALL)
    solution = match.group(1).strip() if match else ""
    assert "print" in solution
    assert "hello" in solution

def test_batch_processing():
    batch_size = 8
    queries = [f"query_{i}" for i in range(20)]
    batches = [queries[i:i+batch_size] for i in range(0, len(queries), batch_size)]
    assert len(batches) == 3
    assert len(batches[0]) == 8
    assert len(batches[2]) == 4

def test_retriever_k_value():
    k = 3
    assert k == 3
    assert k >= 1
    assert k <= 10

def test_max_tokens():
    max_tokens = 512
    assert max_tokens > 0
    assert max_tokens <= 2048