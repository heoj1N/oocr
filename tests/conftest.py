import pytest
from pathlib import Path

@pytest.fixture
def test_data_dir():
    return Path(__file__).parent / "data"

@pytest.fixture
def sample_image(test_data_dir):
    return test_data_dir / "sample.png"

@pytest.fixture
def device():
    import torch
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")