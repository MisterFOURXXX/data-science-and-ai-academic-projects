import pytest
import numpy as np
import hashlib
from unittest.mock import Mock, patch

class MockDocument:
    def __init__(self, metadata):
        self.metadata = metadata

def calculate_hit_rate(retrieved_docs, relevant_ids, k=5):
    retrieved_at_k = retrieved_docs[:k]
    for doc in retrieved_at_k:
        if doc.metadata.get('question_id') in relevant_ids:
            return 1.0
    return 0.0

def calculate_mrr(retrieved_docs, relevant_ids):
    for rank, doc in enumerate(retrieved_docs, 1):
        if doc.metadata.get('question_id') in relevant_ids:
            return 1.0 / rank
    return 0.0

def calculate_map_at_k(retrieved_docs, relevant_ids, k=10):
    relevant_count = 0
    sum_precision = 0
    for i, doc in enumerate(retrieved_docs[:k], 1):
        if doc.metadata.get('question_id') in relevant_ids:
            relevant_count += 1
            sum_precision += relevant_count / i
    return sum_precision / min(len(relevant_ids), k) if relevant_ids else 0

def test_hit_rate_calculation():
    relevant_ids = ["id1"]
    
    docs_hit = [MockDocument({"question_id": "id1"}), MockDocument({"question_id": "id2"})]
    assert calculate_hit_rate(docs_hit, relevant_ids, k=2) == 1.0
    
    docs_miss = [MockDocument({"question_id": "id2"}), MockDocument({"question_id": "id3"})]
    assert calculate_hit_rate(docs_miss, relevant_ids, k=2) == 0.0

def test_mrr_calculation():
    relevant_ids = ["id1"]
    
    docs_first = [MockDocument({"question_id": "id1"}), MockDocument({"question_id": "id2"})]
    assert calculate_mrr(docs_first, relevant_ids) == 1.0
    
    docs_second = [MockDocument({"question_id": "id2"}), MockDocument({"question_id": "id1"})]
    assert calculate_mrr(docs_second, relevant_ids) == 0.5
    
    docs_none = [MockDocument({"question_id": "id2"}), MockDocument({"question_id": "id3"})]
    assert calculate_mrr(docs_none, relevant_ids) == 0.0

def test_map_calculation():
    relevant_ids = ["id1", "id2"]
    
    docs = [
        MockDocument({"question_id": "id1"}),
        MockDocument({"question_id": "id3"}),
        MockDocument({"question_id": "id2"})
    ]
    
    map_score = calculate_map_at_k(docs, relevant_ids, k=3)
    expected = (1/1 + 2/3) / 2
    assert abs(map_score - expected) < 0.001

def test_context_precision():
    relevant_ids = ["id1"]
    
    docs_five = [MockDocument({"question_id": "id1"}) for _ in range(5)]
    precision = calculate_hit_rate(docs_five, relevant_ids, k=5)
    assert precision == 1.0
    
    docs_two_relevant = [MockDocument({"question_id": "id1"}), MockDocument({"question_id": "id1"})]
    precision = calculate_hit_rate(docs_two_relevant, relevant_ids, k=5)
    assert precision == 1.0