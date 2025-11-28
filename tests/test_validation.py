"""Validation tests to ensure the testing infrastructure works correctly."""

import pytest
import sys
import subprocess
from pathlib import Path


class TestInfrastructureValidation:
    """Test class to validate the testing infrastructure setup."""

    def test_pytest_runs(self):
        """Test that pytest is properly installed and can run."""
        assert pytest is not None
        assert hasattr(pytest, 'main')

    def test_python_version(self):
        """Test that Python version is compatible."""
        version = sys.version_info
        assert version.major == 3
        assert version.minor >= 9, f"Python 3.9+ required, got {version.major}.{version.minor}"

    def test_imports_work(self):
        """Test that basic imports work."""
        try:
            import numpy as np
            import pandas as pd
            import torch
            assert True
        except ImportError as e:
            pytest.fail(f"Failed to import required packages: {e}")

    def test_project_structure(self):
        """Test that the project structure is set up correctly."""
        project_root = Path(__file__).parent.parent
        
        # Check essential files exist
        assert (project_root / "pyproject.toml").exists()
        assert (project_root / "tests").exists()
        assert (project_root / "tests" / "__init__.py").exists()
        assert (project_root / "tests" / "conftest.py").exists()
        assert (project_root / "tests" / "unit").exists()
        assert (project_root / "tests" / "integration").exists()

    def test_fixtures_work(self, temp_dir, mock_coordinates, mock_sequence):
        """Test that pytest fixtures are working."""
        assert temp_dir.exists()
        assert isinstance(mock_coordinates, type(None)) or hasattr(mock_coordinates, 'shape')
        assert isinstance(mock_sequence, str)

    def test_markers_defined(self):
        """Test that custom pytest markers are properly defined."""
        # This test will pass if markers are properly configured in pyproject.toml
        # Markers are: unit, integration, slow, gpu, network
        pytest.mark.unit
        pytest.mark.integration 
        pytest.mark.slow
        pytest.mark.gpu
        pytest.mark.network

    @pytest.mark.unit
    def test_unit_marker(self):
        """Test with unit marker."""
        assert True

    @pytest.mark.integration
    def test_integration_marker(self):
        """Test with integration marker."""
        assert True

    def test_temp_directory_fixture(self, temp_dir):
        """Test that temporary directory fixture works."""
        assert temp_dir.exists()
        assert temp_dir.is_dir()
        
        # Test we can write to it
        test_file = temp_dir / "test.txt"
        test_file.write_text("test content")
        assert test_file.exists()
        assert test_file.read_text() == "test content"

    def test_mock_data_fixtures(self, mock_dataset_csv, mock_pdb_file):
        """Test that mock data fixtures work."""
        assert mock_dataset_csv.exists()
        assert mock_pdb_file.exists()
        
        # Test content
        import pandas as pd
        df = pd.read_csv(mock_dataset_csv)
        assert len(df) > 0
        assert 'aa_seq' in df.columns

    def test_validation_data_fixture(self, validation_test_data):
        """Test validation data fixture."""
        assert 'simple_addition' in validation_test_data
        assert 'string_processing' in validation_test_data
        assert 'list_operations' in validation_test_data
        
        # Test the validation data
        add_test = validation_test_data['simple_addition']
        assert add_test['a'] + add_test['b'] == add_test['expected']
        
        str_test = validation_test_data['string_processing']
        assert str_test['input'].upper() == str_test['expected']
        
        list_test = validation_test_data['list_operations']
        result = [x * 2 for x in list_test['input']]
        assert result == list_test['expected']


class TestSimpleOperations:
    """Simple tests to validate basic functionality."""

    def test_basic_math(self):
        """Test basic mathematical operations."""
        assert 2 + 2 == 4
        assert 10 - 5 == 5
        assert 3 * 4 == 12
        assert 8 / 2 == 4

    def test_string_operations(self):
        """Test basic string operations."""
        test_str = "hello world"
        assert test_str.upper() == "HELLO WORLD"
        assert test_str.split() == ["hello", "world"]
        assert "hello" in test_str

    def test_list_operations(self):
        """Test basic list operations."""
        test_list = [1, 2, 3, 4, 5]
        assert len(test_list) == 5
        assert max(test_list) == 5
        assert min(test_list) == 1
        assert sum(test_list) == 15

    def test_dictionary_operations(self):
        """Test basic dictionary operations."""
        test_dict = {"a": 1, "b": 2, "c": 3}
        assert len(test_dict) == 3
        assert test_dict["a"] == 1
        assert "b" in test_dict
        assert list(test_dict.keys()) == ["a", "b", "c"]


@pytest.mark.slow
class TestSlowOperations:
    """Tests marked as slow for demonstration."""

    def test_marked_as_slow(self):
        """A test marked as slow."""
        import time
        time.sleep(0.1)  # Simulate a slow operation
        assert True


if __name__ == "__main__":
    # Allow running this file directly
    pytest.main([__file__])