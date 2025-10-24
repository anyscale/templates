"""Pytest configuration and shared fixtures for Ray Serve tests."""

import pytest
import tempfile
import os
from ray import serve


@pytest.fixture(scope="session")
def test_model_path():
    """Create a temporary test model file."""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pt") as f:
        # Create a dummy model file for testing
        f.write(b"dummy_model_data")
        yield f.name
    os.unlink(f.name)


@pytest.fixture(autouse=True)
def cleanup_serve():
    """Clean up Serve after each test."""
    yield
    serve.shutdown()
