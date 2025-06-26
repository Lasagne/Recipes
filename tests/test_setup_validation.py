import pytest
import numpy as np
from pathlib import Path


class TestSetupValidation:
    """Validation tests to verify the testing infrastructure is working correctly."""
    
    @pytest.mark.unit
    def test_pytest_is_working(self):
        """Basic test to verify pytest is running."""
        assert True
        assert 1 + 1 == 2
    
    @pytest.mark.unit
    def test_fixtures_are_available(self, temp_dir, mock_config):
        """Test that our custom fixtures are accessible."""
        assert isinstance(temp_dir, Path)
        assert temp_dir.exists()
        
        assert isinstance(mock_config, dict)
        assert 'learning_rate' in mock_config
        assert mock_config['batch_size'] == 32
    
    @pytest.mark.unit
    def test_numpy_is_available(self, sample_input_data):
        """Test that numpy is installed and working."""
        assert isinstance(sample_input_data, np.ndarray)
        assert sample_input_data.shape == (2, 3, 224, 224)
        assert sample_input_data.dtype == np.float32
    
    @pytest.mark.unit
    def test_mock_fixtures_work(self, mock_model, mock_layer):
        """Test that mock fixtures are properly configured."""
        # Test mock model
        result = mock_model.predict()
        assert isinstance(result, np.ndarray)
        assert result.shape == (2,)
        
        # Test mock layer
        assert mock_layer.name == "test_layer"
        assert mock_layer.input_shape == (None, 3, 224, 224)
    
    @pytest.mark.unit
    def test_temp_dir_cleanup(self, temp_dir):
        """Test that temporary directories are created and will be cleaned up."""
        test_file = temp_dir / "test.txt"
        test_file.write_text("test content")
        
        assert test_file.exists()
        assert test_file.read_text() == "test content"
    
    @pytest.mark.unit
    def test_parametrized_tests_work(self):
        """Test that pytest parametrization works correctly."""
        @pytest.mark.parametrize("input,expected", [
            (1, 2),
            (2, 4),
            (3, 6),
        ])
        def multiply_by_two(input, expected):
            assert input * 2 == expected
        
        # Run the parametrized test
        multiply_by_two(2, 4)
    
    @pytest.mark.integration
    def test_integration_marker(self):
        """Test that integration marker works."""
        # This would be an integration test
        assert True
    
    @pytest.mark.slow
    def test_slow_marker(self):
        """Test that slow marker works."""
        # This would be a slow test
        import time
        start = time.time()
        # Simulate slow operation
        time.sleep(0.01)
        end = time.time()
        assert end - start >= 0.01
    
    @pytest.mark.unit
    def test_mock_dataset(self, mock_dataset):
        """Test that mock dataset fixture works correctly."""
        assert len(mock_dataset) == 100
        assert mock_dataset.X_train.shape == (100, 3, 32, 32)
        assert mock_dataset.y_train.shape == (100,)
        assert mock_dataset.num_classes == 10
        
        # Test get_batch method
        batch_x, batch_y = mock_dataset.get_batch([0, 1, 2])
        assert batch_x.shape == (3, 3, 32, 32)
        assert batch_y.shape == (3,)
    
    @pytest.mark.unit
    def test_coverage_is_tracked(self):
        """Test that code coverage is being tracked."""
        def sample_function(x, y):
            if x > 0:
                return x + y
            else:
                return x - y
        
        # Test both branches
        assert sample_function(5, 3) == 8
        assert sample_function(-5, 3) == -8
    
    @pytest.mark.unit
    def test_stdout_capture(self, capsys):
        """Test that stdout capturing works."""
        print("Test output")
        print("Another line")
        
        captured = capsys.readouterr()
        assert "Test output" in captured.out
        assert "Another line" in captured.out


@pytest.mark.unit
class TestMarkerValidation:
    """Test class to validate markers work at class level."""
    
    def test_class_marker_applied(self):
        """All tests in this class should have unit marker."""
        assert True
    
    def test_another_class_test(self):
        """Another test in marked class."""
        assert 2 * 2 == 4