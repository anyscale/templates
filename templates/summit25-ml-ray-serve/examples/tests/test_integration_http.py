"""HTTP integration tests."""

import pytest
import requests
from ray import serve
from tests.example_app import MyModelDeployment
import time

class TestMyModelHTTP:
    """HTTP integration tests"""

    @pytest.fixture
    def serve_app(self):
        """Setup HTTP server for testing"""
        app = MyModelDeployment.bind("test_model_path")
        serve.run(app, name="test_model", blocking=False)
        yield
        serve.shutdown()

    def test_http_prediction_endpoint(self, serve_app):
        """Test HTTP prediction endpoint"""
        input_data = {"input": "data"}
        response = requests.post("http://localhost:8000/", json=input_data, timeout=10)

        assert response.status_code == 200
        result = response.json()
        assert result == {"prediction": "test_result", "confidence": 0.95}

    def test_http_error_handling(self, serve_app):
        """Test HTTP error handling"""
        invalid_data = {"invalid": "data"}
        response = requests.post(
            "http://localhost:8000/", json=invalid_data, timeout=10
        )

        # Test appropriate error response
        assert response.status_code in [400, 422, 500]
