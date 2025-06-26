# Testing Guide

This directory contains the test suite for the project. The testing infrastructure is set up with pytest, pytest-cov, and pytest-mock.

## Running Tests

```bash
# Run all tests
poetry run test
# or
poetry run tests

# Run only unit tests
poetry run pytest -m unit

# Run only integration tests  
poetry run pytest -m integration

# Run tests excluding slow ones
poetry run pytest -m "not slow"

# Run specific test file
poetry run pytest tests/test_specific.py

# Run with verbose output
poetry run pytest -v

# Run without coverage
poetry run pytest --no-cov
```

## Test Structure

```
tests/
├── __init__.py
├── conftest.py          # Shared fixtures and configuration
├── test_setup_validation.py  # Validation tests for the infrastructure
├── unit/                # Unit tests directory
│   └── __init__.py
└── integration/         # Integration tests directory
    └── __init__.py
```

## Available Fixtures

The `conftest.py` file provides several useful fixtures:

- `temp_dir`: Creates a temporary directory
- `mock_model`: Mock Lasagne model
- `mock_layer`: Mock Lasagne layer
- `sample_input_data`: Sample neural network input data
- `sample_labels`: Sample labels
- `mock_config`: Configuration dictionary
- `mock_weights_file`: Mock weights file path
- `mock_dataset`: Mock dataset object
- `mock_theano_function`: Mock Theano function
- `disable_gpu`: Force CPU execution for tests
- `test_data_dir`: Session-scoped test data directory
- `test_output_dir`: Session-scoped test output directory

## Writing Tests

Example test file:

```python
import pytest
import numpy as np
from modelzoo.vgg16 import build_model

@pytest.mark.unit
class TestVGG16:
    def test_model_structure(self):
        """Test that VGG16 model builds correctly."""
        model = build_model()
        assert model is not None
        # Add more assertions
    
    @pytest.mark.slow
    def test_model_forward_pass(self, sample_input_data):
        """Test forward pass through the model."""
        model = build_model()
        # Test implementation
```

## Test Markers

- `@pytest.mark.unit`: Fast, isolated unit tests
- `@pytest.mark.integration`: Integration tests (may be slower)
- `@pytest.mark.slow`: Slow-running tests

## Coverage Requirements

- Minimum coverage threshold: 80%
- Coverage reports are generated in:
  - Terminal output
  - HTML report: `htmlcov/index.html`
  - XML report: `coverage.xml`