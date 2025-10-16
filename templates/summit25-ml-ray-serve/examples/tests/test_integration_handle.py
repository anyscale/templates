"""Integration tests using DeploymentHandle."""

import pytest
from ray import serve
from tests.example_app import MyModelDeployment


class TestMyModelDeployment:
    """Integration tests using DeploymentHandle"""

    @pytest.fixture
    def deployment_handle(self):
        """Setup deployment for testing"""
        app = MyModelDeployment.bind("test_model_path")
        handle = serve.run(
            app, name="test_model", blocking=False, _local_testing_mode=True
        )
        yield handle
        serve.shutdown()

    def test_deployment_prediction(self, deployment_handle):
        """Test prediction through deployment handle"""
        input_data = {"input": "data"}
        result = deployment_handle.predict.remote(input_data).result()
        assert result == {"prediction": "test_result", "confidence": 0.95}
