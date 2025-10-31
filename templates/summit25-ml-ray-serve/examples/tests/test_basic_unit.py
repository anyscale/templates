"""Unit tests for basic MyModel business logic."""

from tests.example_app import MyModel


class TestMyModel:
    """Unit tests for core business logic"""

    def test_model_initialization(self):
        """Test model loading and initialization"""
        model = MyModel("test_model_path")
        assert model.model_path == "test_model_path"
        assert model.model == "model_loaded_from_test_model_path"

    def test_end_to_end_prediction(self):
        """Test complete prediction pipeline"""
        model = MyModel("test_model_path")
        input_data = {"test": "input"}
        result = model.predict(input_data)
        assert result == {"prediction": "test_result", "confidence": 0.95}
