import pytest
from fastapi.testclient import TestClient
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class TestAPI:
    @pytest.fixture
    def client(self):
        from deployment.inference_api import app
        return TestClient(app)
    
    def test_health_endpoint(self, client):
        response = client.get("/health")
        assert response.status_code == 200
        assert "status" in response.json()
    
    def test_generate_endpoint_validation(self, client):
        response = client.post("/generate", json={})
        assert response.status_code == 422
    
    def test_generate_endpoint_with_prompt(self, client):
        response = client.post("/generate", json={
            "prompt": "How to reverse a list in Python?",
            "max_new_tokens": 100
        })
        assert response.status_code in [200, 503]
    
    def test_generate_with_custom_parameters(self, client):
        response = client.post("/generate", json={
            "prompt": "Write a function to sort a list",
            "max_new_tokens": 200,
            "temperature": 0.5,
            "top_p": 0.9,
            "top_k": 40
        })
        assert response.status_code in [200, 503]