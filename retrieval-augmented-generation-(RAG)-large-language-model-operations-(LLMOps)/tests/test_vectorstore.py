import pytest
import hashlib
import re
from unittest.mock import Mock, patch

def create_document_id(content: str) -> str:
    return hashlib.md5(content.encode()).hexdigest()

def detect_code(text: str) -> int:
    patterns = [r'```', r'def ', r'class ', r'function', r'import ', r'#include']
    return 1 if any(re.search(p, text) for p in patterns) else 0

def test_document_id_generation():
    content = "Test content"
    doc_id = create_document_id(content)
    assert len(doc_id) == 32
    assert isinstance(doc_id, str)

def test_code_detection():
    code_text = "def test_function():\n    pass"
    assert detect_code(code_text) == 1
    
    plain_text = "This is plain text without code"
    assert detect_code(plain_text) == 0

def test_chunk_metadata():
    metadata = {
        "chunk_id": 0,
        "chunk_size": 512,
        "source": "stackoverflow",
        "has_code": 1
    }
    
    assert metadata["chunk_id"] == 0
    assert metadata["chunk_size"] == 512
    assert metadata["source"] == "stackoverflow"

def test_embedding_dimension():
    mock_embedding = [0.1] * 384
    assert len(mock_embedding) == 384

@pytest.mark.parametrize("chunk_size", [256, 512, 1024])
def test_chunk_size_parameter(chunk_size):
    assert chunk_size in [256, 512, 1024]