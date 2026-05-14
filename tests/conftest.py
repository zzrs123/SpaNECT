"""
Pytest configuration and fixtures for SpaNECT tests.
"""

import pytest
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parents[1]
src_path = project_root / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))


@pytest.fixture(scope="session")
def dlpfc_data_path():
    """Path to DLPFC dataset."""
    return "/home/dugaoyuan/AAA-new-base/ztest_tmvc_model/1-04-full-model-can/FullModel/DLPFC"


@pytest.fixture(scope="session")
def dlpfc_data_name():
    """Default DLPFC slice name."""
    return "151507"


@pytest.fixture(scope="session")
def test_output_dir(tmp_path_factory):
    """Output directory for test results."""
    return tmp_path_factory.mktemp("testresults")


@pytest.fixture(scope="session")
def device():
    """PyTorch device for testing."""
    import torch
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
