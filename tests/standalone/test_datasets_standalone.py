"""
Standalone test for datasets module.
Tests that datasets module can load data independently.

Usage:
    pytest tests/standalone/test_datasets_standalone.py -v
    python tests/standalone/test_datasets_standalone.py
"""

import sys
from pathlib import Path

project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root / "src"))

import pytest


class TestDatasetsRawData:
    """Test datasets module with raw Visium data."""

    @pytest.fixture
    def data_path(self):
        return "/home/dugaoyuan/AAA-new-base/ztest_tmvc_model/1-04-full-model-can/FullModel/DLPFC"

    @pytest.fixture
    def data_name(self):
        return "151507"

    def test_load_adata(self, data_path, data_name):
        """Test loading raw Visium adata."""
        from spanect import datasets

        adata = datasets.load_adata(
            platform='Visium',
            data_path=data_path,
            data_name=data_name
        )

        assert adata is not None, "adata should not be None"
        assert adata.n_obs > 0, "adata should have observations"
        assert adata.n_vars > 0, "adata should have variables"
        assert 'spatial' in adata.obsm, "adata should have spatial coordinates"
        print(f"[PASS] Loaded adata: {adata.shape}")

    def test_load_adata_has_required_fields(self, data_path, data_name):
        """Test that loaded adata has required fields."""
        from spanect import datasets

        adata = datasets.load_adata(
            platform='Visium',
            data_path=data_path,
            data_name=data_name
        )

        assert 'spatial' in adata.obsm, "Missing spatial coordinates"
        assert 'layer_guess' in adata.obs, "Missing layer_guess labels"
        print("[PASS] adata has required fields")


class TestDatasetsH5ad:
    """Test datasets module with h5ad data."""

    @pytest.fixture
    def h5ad_path(self):
        return Path("/home/dugaoyuan/AAA-new-base/ztest_tmvc_model/1-04-full-model-can/FullModel/Train_Data/TONSIL_NEW/combined_adata_s2.h5ad")

    def test_load_h5ad(self, h5ad_path):
        """Test loading h5ad file."""
        if not h5ad_path.exists():
            pytest.skip(f"h5ad file not found: {h5ad_path}")

        from spanect import datasets

        adata = datasets.load_h5ad(h5ad_path)

        assert adata is not None, "adata should not be None"
        assert adata.n_obs > 0, "adata should have observations"
        print(f"[PASS] Loaded h5ad: {adata.shape}")


def main():
    """Run tests manually."""
    print("=" * 60)
    print("Datasets Module Standalone Test")
    print("=" * 60)

    data_path = "/home/dugaoyuan/AAA-new-base/ztest_tmvc_model/1-04-full-model-can/FullModel/DLPFC"
    data_name = "151507"

    print("\n[INFO] Test 1: load_adata")
    test = TestDatasetsRawData()
    test.test_load_adata(data_path, data_name)
    test.test_load_adata_has_required_fields(data_path, data_name)

    print("\n[INFO] Test 2: load_h5ad")
    h5ad_path = Path("/home/dugaoyuan/AAA-new-base/ztest_tmvc_model/1-04-full-model-can/FullModel/Train_Data/TONSIL_NEW/combined_adata_s2.h5ad")
    test_h5ad = TestDatasetsH5ad()
    test_h5ad.test_load_h5ad(h5ad_path)

    print("\n[SUCCESS] All datasets tests passed!")


if __name__ == "__main__":
    main()
