# Ray Serve Training Tests

This directory contains executable test examples extracted from the Ray Serve training materials.

## Test Structure

- `test_basic_unit.py` - Unit tests for basic business logic
- `test_integration_handle.py` - Integration tests using DeploymentHandle
- `test_integration_http.py` - HTTP integration tests
- `test_composition.py` - Advanced tests for deployment composition
- `example_app.py` - Example application code used in tests
- `conftest.py` - Pytest configuration and shared fixtures

## Running Tests

### Prerequisites

Make sure you have the required dependencies installed:

```bash
pip install pytest "ray[serve]" requests
```

### Run All Tests

```bash
pytest tests/
```

### Run Specific Test Categories

```bash
# Unit tests only
pytest tests/test_basic_unit.py -m unit

# Integration tests only  
pytest tests/test_integration_*.py -m integration

# Composition tests only
pytest tests/test_composition.py
```

### Run with Verbose Output

```bash
pytest tests/ -v
```

## Test Examples

The tests demonstrate:

1. **Unit Testing**: Testing business logic without Ray Serve dependencies
2. **Integration Testing**: Testing deployments using DeploymentHandle
3. **HTTP Testing**: Testing full HTTP stack with requests library
4. **Composition Testing**: Testing multi-deployment applications
5. **Mocking**: Using mocks for faster testing

Each test file contains comprehensive examples that can be run independently to verify Ray Serve testing patterns work correctly.
