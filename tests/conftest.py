"""Shared pytest fixtures for the protein scoring project."""

import os
import tempfile
import shutil
from pathlib import Path
from typing import Dict, Any, Generator
import pytest
import numpy as np
import pandas as pd
import torch


@pytest.fixture
def temp_dir() -> Generator[Path, None, None]:
    """Create a temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def sample_data_dir(temp_dir: Path) -> Path:
    """Create a directory with sample test data."""
    data_dir = temp_dir / "data"
    data_dir.mkdir()
    return data_dir


@pytest.fixture
def mock_pdb_file(sample_data_dir: Path) -> Path:
    """Create a mock PDB file for testing."""
    pdb_content = """HEADER    TEST PDB                                01-JAN-23   TEST
ATOM      1  N   ALA A   1      20.154  -6.351   1.000  1.00 10.00           N  
ATOM      2  CA  ALA A   1      21.618  -6.351   1.000  1.00 10.00           C  
ATOM      3  C   ALA A   1      22.200  -4.951   1.000  1.00 10.00           C  
ATOM      4  O   ALA A   1      21.618  -3.868   1.000  1.00 10.00           O  
ATOM      5  CB  ALA A   1      22.200  -7.151   2.165  1.00 10.00           C  
ATOM      6  N   VAL A   2      23.518  -4.951   1.000  1.00 10.00           N  
ATOM      7  CA  VAL A   2      24.280  -3.715   1.000  1.00 10.00           C  
ATOM      8  C   VAL A   2      23.718  -2.715   2.000  1.00 10.00           C  
ATOM      9  O   VAL A   2      23.718  -1.495   1.750  1.00 10.00           O  
ATOM     10  CB  VAL A   2      25.757  -4.051   1.333  1.00 10.00           C  
END
"""
    pdb_file = sample_data_dir / "test_protein.pdb"
    pdb_file.write_text(pdb_content)
    return pdb_file


@pytest.fixture
def mock_dataset_csv(sample_data_dir: Path) -> Path:
    """Create a mock dataset CSV file for testing."""
    data = {
        'WT_name': ['test_protein.pdb', 'test_protein.pdb', 'test_protein.pdb'],
        'aa_seq': ['ARLQIVNL', 'VRLQIVNL', 'ARLQIVYL'],
        'fitness': [1.5, 2.0, 0.8],
        'mut_type': ['wt', 'A1V', 'N7Y'],
        'chains': ['A', 'A', 'A']
    }
    df = pd.DataFrame(data)
    csv_file = sample_data_dir / "test_dataset.csv"
    df.to_csv(csv_file, index=False)
    return csv_file


@pytest.fixture
def mock_complex_dataset_csv(sample_data_dir: Path) -> Path:
    """Create a mock complex dataset CSV file for testing."""
    data = {
        'WT_name': ['complex1.pdb', 'complex1.pdb'],
        'A': ['ARLQIVNL', 'VRLQIVNL'],
        'B': ['MKLLVFGL', 'MKLLVFGL'],
        'chains': ['A,B', 'A,B'],
        'muts': ['A1V', ''],
        'mut_chain_idx': ['0', ''],
        'affinity': [5.2, 6.1]
    }
    df = pd.DataFrame(data)
    csv_file = sample_data_dir / "test_complex_dataset.csv"
    df.to_csv(csv_file, index=False)
    return csv_file


@pytest.fixture
def mock_weights_file(sample_data_dir: Path) -> Path:
    """Create a mock model weights file for testing."""
    # Create a simple state dict for testing
    state_dict = {
        'encoder.layer.0.weight': torch.randn(10, 5),
        'encoder.layer.0.bias': torch.randn(10),
        'decoder.weight': torch.randn(5, 10),
    }
    weights_file = sample_data_dir / "mock_weights.pt"
    torch.save(state_dict, weights_file)
    return weights_file


@pytest.fixture
def mock_fasta_sequences(sample_data_dir: Path) -> Path:
    """Create a mock FASTA file with protein sequences."""
    fasta_content = """>seq1
MKLLVFGLSWICLPQHKQNSSNLSTAFLFLLG
>seq2
ARLQIVNLLPLEKVSAGHIYWPVSQNSVVKE
>seq3
VRLQIVYLPLEKVSAGHIYWPVSQNSVVKED
"""
    fasta_file = sample_data_dir / "test_sequences.fasta"
    fasta_file.write_text(fasta_content)
    return fasta_file


@pytest.fixture
def mock_args_namespace():
    """Create a mock argument namespace for testing CLI scripts."""
    from argparse import Namespace
    
    return Namespace(
        pdbfile="test_protein.pdb",
        chain="A",
        temperature=1.0,
        outpath="output/sampled_seqs.fasta",
        num_samples=10,
        weights_path=None,
        fixed_pos=None
    )


@pytest.fixture
def mock_scoring_args():
    """Create mock arguments for scoring scripts."""
    from argparse import Namespace
    
    return Namespace(
        weights_path=None,
        dataset_path="test_dataset.csv",
        feature="fitness",
        normalize=False,
        whole_seq=False,
        sum=False,
        out_path="results.csv"
    )


@pytest.fixture
def mock_torch_device():
    """Mock torch device for testing."""
    return torch.device("cpu")  # Always use CPU for tests


@pytest.fixture
def mock_coordinates():
    """Create mock protein coordinates for testing."""
    # Mock coordinates for a small protein (2 residues, 3 atoms each)
    coords = np.array([
        [[20.154, -6.351, 1.000],  # N
         [21.618, -6.351, 1.000],  # CA  
         [22.200, -4.951, 1.000]], # C
        [[23.518, -4.951, 1.000],  # N
         [24.280, -3.715, 1.000],  # CA
         [23.718, -2.715, 2.000]]  # C
    ], dtype=np.float32)
    return coords


@pytest.fixture
def mock_sequence():
    """Create a mock protein sequence."""
    return "ARLQIVNL"


@pytest.fixture
def sample_output_dir(temp_dir: Path) -> Path:
    """Create a sample output directory."""
    output_dir = temp_dir / "output"
    output_dir.mkdir()
    return output_dir


@pytest.fixture(autouse=True)
def set_test_environment():
    """Set up test environment variables."""
    old_env = os.environ.copy()
    # Set test environment variables
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # Disable CUDA for tests
    
    yield
    
    # Restore original environment
    os.environ.clear()
    os.environ.update(old_env)


@pytest.fixture
def mock_esm_model():
    """Mock ESM model for testing (requires actual ESM to be installed)."""
    try:
        import esm
        # This would require ESM to be properly installed
        # For now, we'll return a mock object
        class MockESMModel:
            def eval(self):
                pass
                
            def to(self, device):
                return self
                
            def sample(self, coords, partial_seq=None, temperature=1.0, device=None):
                return "MOCKSEQ"
                
        return MockESMModel()
    except ImportError:
        pytest.skip("ESM not available for testing")


@pytest.fixture
def validation_test_data():
    """Create minimal test data for validation tests."""
    return {
        'simple_addition': {'a': 2, 'b': 3, 'expected': 5},
        'string_processing': {'input': "hello", 'expected': "HELLO"},
        'list_operations': {'input': [1, 2, 3], 'expected': [2, 4, 6]}
    }


# Pytest configuration fixtures
@pytest.fixture(scope="session")
def test_session_config():
    """Session-wide test configuration."""
    config = {
        'test_data_version': '1.0',
        'temp_file_cleanup': True,
        'mock_external_apis': True
    }
    return config


# GPU-related fixtures
@pytest.fixture
def skip_if_no_gpu():
    """Skip test if GPU is not available."""
    if not torch.cuda.is_available():
        pytest.skip("GPU not available")


# Network-related fixtures  
@pytest.fixture
def skip_if_no_network():
    """Skip test if network is not available."""
    try:
        import requests
        requests.get("https://httpbin.org/get", timeout=5)
    except:
        pytest.skip("Network not available")