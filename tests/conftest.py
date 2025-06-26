import pytest
import tempfile
import shutil
import os
from pathlib import Path
from unittest.mock import Mock, MagicMock
import numpy as np


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files."""
    temp_path = tempfile.mkdtemp()
    yield Path(temp_path)
    shutil.rmtree(temp_path)


@pytest.fixture
def mock_model():
    """Create a mock Lasagne model for testing."""
    model = Mock()
    model.predict = MagicMock(return_value=np.array([0.1, 0.9]))
    model.get_output = MagicMock()
    model.get_params = MagicMock(return_value=[])
    return model


@pytest.fixture
def mock_layer():
    """Create a mock Lasagne layer for testing."""
    layer = Mock()
    layer.input_shape = (None, 3, 224, 224)
    layer.output_shape = (None, 1000)
    layer.name = "test_layer"
    return layer


@pytest.fixture
def sample_input_data():
    """Generate sample input data for neural network testing."""
    # Typical ImageNet-style input: batch_size x channels x height x width
    return np.random.rand(2, 3, 224, 224).astype(np.float32)


@pytest.fixture
def sample_labels():
    """Generate sample labels for testing."""
    return np.array([0, 1], dtype=np.int32)


@pytest.fixture
def mock_config():
    """Create a mock configuration dictionary."""
    return {
        'learning_rate': 0.01,
        'batch_size': 32,
        'num_epochs': 10,
        'model_name': 'test_model',
        'data_dir': '/tmp/test_data',
        'output_dir': '/tmp/test_output'
    }


@pytest.fixture
def mock_weights_file(temp_dir):
    """Create a mock weights file for testing model loading."""
    weights_path = temp_dir / "mock_weights.pkl"
    # Create empty file to simulate weights file
    weights_path.touch()
    return str(weights_path)


@pytest.fixture
def mock_dataset():
    """Create a mock dataset for testing."""
    class MockDataset:
        def __init__(self):
            self.X_train = np.random.rand(100, 3, 32, 32).astype(np.float32)
            self.y_train = np.random.randint(0, 10, 100)
            self.X_test = np.random.rand(20, 3, 32, 32).astype(np.float32)
            self.y_test = np.random.randint(0, 10, 20)
            self.num_classes = 10
            self.input_shape = (3, 32, 32)
        
        def __len__(self):
            return len(self.X_train)
        
        def get_batch(self, indices):
            return self.X_train[indices], self.y_train[indices]
    
    return MockDataset()


@pytest.fixture(autouse=True)
def reset_modules(monkeypatch):
    """Reset any global state between tests."""
    # This can be expanded based on specific needs
    pass




@pytest.fixture
def mock_theano_function():
    """Create a mock Theano function for testing."""
    func = Mock()
    func.return_value = np.array([[0.1, 0.9]])
    return func


@pytest.fixture
def disable_gpu(monkeypatch):
    """Disable GPU for testing to ensure tests run on CPU."""
    monkeypatch.setenv('THEANO_FLAGS', 'device=cpu,floatX=float32')


# Markers for test organization
def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line(
        "markers", "unit: mark test as a unit test"
    )
    config.addinivalue_line(
        "markers", "integration: mark test as an integration test"
    )
    config.addinivalue_line(
        "markers", "slow: mark test as slow running"
    )


# Test session fixtures
@pytest.fixture(scope="session")
def test_data_dir(tmp_path_factory):
    """Create a session-wide test data directory."""
    return tmp_path_factory.mktemp("test_data")


@pytest.fixture(scope="session")
def test_output_dir(tmp_path_factory):
    """Create a session-wide test output directory."""
    return tmp_path_factory.mktemp("test_output")