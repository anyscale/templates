"""Advanced tests for composition of deployments."""

import pytest
from unittest.mock import Mock
from ray import serve
from tests.example_composition import (
    TextClassifier,
    TextClassifierDeployment,
    TextEmbedder,
    TextEmbedderDeployment,
    TextPreprocessor,
    TextPreprocessorDeployment,
    TextPipelineDeployment,
)


class TestTextPreprocessor:
    """Unit tests for text preprocessor"""

    def test_preprocessing(self):
        preprocessor = TextPreprocessor()
        result = preprocessor.preprocess("  Hello World  ")
        assert result["tokens"] == ["hello", "world"]
        assert result["length"] == 2

    def test_empty_text(self):
        preprocessor = TextPreprocessor()
        result = preprocessor.preprocess("")
        assert result["tokens"] == []
        assert result["length"] == 0


class TestTextEmbedder:
    """Unit tests for text embedder"""

    def test_embedding_generation(self):
        embedder = TextEmbedder()
        tokens = {"tokens": ["hello", "world"], "length": 2}
        embeddings = embedder.embed(tokens)
        assert len(embeddings) == 2
        assert all(isinstance(x, float) for x in embeddings)


class TestTextClassifier:
    """Unit tests for text classifier"""

    def test_classification(self):
        classifier = TextClassifier()
        embeddings = [0.1, 0.2, 0.3]
        result = classifier.classify(embeddings)
        assert "label" in result
        assert "confidence" in result
        assert isinstance(result["confidence"], float)


class TestTextPipelineComposition:
    """Integration tests for deployment composition"""

    @pytest.fixture
    def pipeline_handle(self):
        """Setup composed pipeline for testing"""
        # Create individual deployments
        preprocessor_app = TextPreprocessorDeployment.bind()
        embedder_app = TextEmbedderDeployment.bind()
        classifier_app = TextClassifierDeployment.bind()

        # Create composed pipeline
        pipeline_app = TextPipelineDeployment.bind(
            preprocessor=preprocessor_app,
            embedder=embedder_app,
            classifier=classifier_app,
        )

        handle = serve.run(pipeline_app, name="text_pipeline", blocking=False)
        yield handle
        serve.shutdown()

    def test_end_to_end_pipeline(self, pipeline_handle):
        """Test complete pipeline execution"""
        input_text = "This is a test message"

        # Execute pipeline
        result = pipeline_handle.orchestrate.remote({"text": input_text}).result()

        # Verify result structure
        assert "preprocessed" in result
        assert "embeddings" in result
        assert "classification" in result

        # Verify preprocessing step
        preprocessed = result["preprocessed"]
        assert preprocessed["tokens"] == ["this", "is", "a", "test", "message"]
        assert preprocessed["length"] == 5

        # Verify embedding step
        embeddings = result["embeddings"]
        assert len(embeddings) == 5

        # Verify classification step
        classification = result["classification"]
        assert "label" in classification
        assert "confidence" in classification

    def test_pipeline_with_empty_text(self, pipeline_handle):
        """Test pipeline with edge case input"""
        result = pipeline_handle.orchestrate.remote({"text": ""}).result()

        # Should handle empty text gracefully
        assert result["preprocessed"]["length"] == 0
        assert len(result["embeddings"]) == 0


class TestIndividualDeployments:
    """Test individual deployment components"""

    @pytest.fixture
    def preprocessor_handle(self):
        app = TextPreprocessorDeployment.bind()
        handle = serve.run(
            app, name="preprocessor", blocking=False, _local_testing_mode=True
        )
        yield handle
        serve.shutdown()

    @pytest.fixture
    def embedder_handle(self):
        app = TextEmbedderDeployment.bind()
        handle = serve.run(
            app, name="embedder", blocking=False, _local_testing_mode=True
        )
        yield handle
        serve.shutdown()

    @pytest.fixture
    def classifier_handle(self):
        app = TextClassifierDeployment.bind()
        handle = serve.run(
            app, name="classifier", blocking=False, _local_testing_mode=True
        )
        yield handle
        serve.shutdown()

    def test_preprocessor_deployment(self, preprocessor_handle):
        """Test preprocessor deployment"""
        result = preprocessor_handle.preprocess.remote("Hello World").result()
        assert result["tokens"] == ["hello", "world"]

    def test_embedder_deployment(self, embedder_handle):
        """Test embedder deployment"""
        tokens = {"tokens": ["hello", "world"], "length": 2}
        embeddings = embedder_handle.embed.remote(tokens).result()
        assert len(embeddings) == 2

    def test_classifier_deployment(self, classifier_handle):
        """Test classifier deployment"""
        embeddings = [0.1, 0.2, 0.3]
        result = classifier_handle.classify.remote(embeddings).result()
        assert "label" in result


class TestPipelineWithMocks:
    """Test pipeline with mocked dependencies"""

    def test_pipeline_with_mocked_components(self):
        """Test pipeline using mocked deployment handles"""
        # Create mock handles
        mock_preprocessor = Mock()
        mock_embedder = Mock()
        mock_classifier = Mock()

        # Configure mock return values
        mock_preprocessor.preprocess.remote.return_value = {
            "tokens": ["test"],
            "length": 1,
        }
        mock_embedder.embed.remote.return_value = [0.5]
        mock_classifier.classify.remote.return_value = {
            "label": "positive",
            "confidence": 0.9,
        }

        # Create a simple pipeline class for testing
        class TestPipeline:
            def __init__(self, preprocessor, embedder, classifier):
                self.preprocessor = preprocessor
                self.embedder = embedder
                self.classifier = classifier

            def __call__(self, data):
                text = data["text"]
                preprocessed = self.preprocessor.preprocess.remote(text)
                embeddings = self.embedder.embed.remote(preprocessed)
                result = self.classifier.classify.remote({"embeddings": embeddings})
                return {
                    "preprocessed": preprocessed,
                    "embeddings": embeddings,
                    "classification": result,
                }

        # Create pipeline with mocked dependencies
        pipeline = TestPipeline(mock_preprocessor, mock_embedder, mock_classifier)

        # Test pipeline execution
        result = pipeline({"text": "test"})

        # Verify mock calls
        mock_preprocessor.preprocess.remote.assert_called_once_with("test")
        mock_embedder.embed.remote.assert_called_once()
        mock_classifier.classify.remote.assert_called_once()

        # Verify result
        assert result["classification"]["label"] == "positive"
