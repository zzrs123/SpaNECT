"""
Standalone test for pp (preprocess) module.
Tests that pp module can preprocess data independently.

Usage:
    pytest tests/standalone/test_pp_standalone.py -v
    python tests/standalone/test_pp_standalone.py
"""

import sys
from pathlib import Path

project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root / "src"))

import numpy as np
import pytest


class TestPPPreprocess:
    """Test pp preprocessing functions."""

    def test_preprocess_rna(self):
        """Test preprocess_rna function with synthetic data."""
        from spanect.pp import preprocess_rna
        import scanpy as sc
        import anndata as ad

        np.random.seed(42)
        n_cells = 500
        n_genes = 1000
        X = np.random.negative_binomial(5, 0.3, size=(n_cells, n_genes)).astype(np.float32)
        
        adata = ad.AnnData(X)
        adata.var_names = [f"gene_{i}" for i in range(n_genes)]
        adata.obs_names = [f"cell_{i}" for i in range(n_cells)]
        
        preprocess_rna(adata)

        assert 'highly_variable' in adata.var, "HVG should be computed"
        print(f"[PASS] preprocess_rna: {adata.shape}")

    def test_preprocess_alias(self):
        """Test preprocess alias."""
        from spanect.pp import preprocess, preprocess_rna

        assert preprocess is preprocess_rna, "preprocess should be the same function as preprocess_rna"
        print("[PASS] preprocess alias works")

    def test_preprocess_rna_from_X(self):
        """Test preprocess_rna_from_X function with synthetic data."""
        from spanect.pp import preprocess_rna_from_X
        import anndata as ad

        np.random.seed(42)
        n_cells = 500
        n_genes = 1000
        X = np.random.negative_binomial(5, 0.3, size=(n_cells, n_genes)).astype(np.float32)
        
        adata = ad.AnnData(X)
        adata.var_names = [f"gene_{i}" for i in range(n_genes)]
        adata.obs_names = [f"cell_{i}" for i in range(n_cells)]
        
        X_out = preprocess_rna_from_X(adata, rna_pca_dim=50, seed=0)

        assert X_out.shape[0] == n_cells, "Output should have same n_obs"
        assert X_out.shape[1] == 50, "Output should have pca_dim features"
        print(f"[PASS] preprocess_rna_from_X: {X_out.shape}")

    def test_preprocess_omics2_from_obsm(self):
        """Test preprocess_omics2_from_obsm function."""
        from spanect.pp import preprocess_omics2_from_obsm

        class MockAdata:
            pass

        adata = MockAdata()
        adata.obsm = {'X_atac': np.random.rand(100, 500).astype(np.float32)}

        X = preprocess_omics2_from_obsm(
            adata,
            key='X_atac',
            pipeline='pca',
            omics_dim=30,
            seed=0
        )

        assert X.shape == (100, 30), f"Expected (100, 30), got {X.shape}"
        print(f"[PASS] preprocess_omics2_from_obsm: {X.shape}")

    def test_preprocess_atac_alias(self):
        """Test preprocess_atac_from_obsm alias."""
        from spanect.pp import preprocess_atac_from_obsm

        assert preprocess_atac_from_obsm is not None, "Alias should exist"
        print("[PASS] preprocess_atac_from_obsm alias exists")


class TestPPRawInputs:
    """Test pp raw inputs preparation."""

    @pytest.fixture
    def data_path(self):
        return "/home/dugaoyuan/AAA-new-base/ztest_tmvc_model/1-04-full-model-can/FullModel/DLPFC"

    @pytest.fixture
    def data_name(self):
        return "151507"

    def test_prepare_raw_inputs(self, data_path, data_name, tmp_path):
        """Test prepare_raw_inputs function."""
        from spanect import datasets
        from spanect.tl import get_model_input

        adata = datasets.load_adata(
            platform='Visium',
            data_path=data_path,
            data_name=data_name
        )

        Xs, Y, adata_filtered = get_model_input(
            adata=adata,
            data_path=data_path,
            data_name=data_name,
            device='cpu',
            multiomics=False,
            use_h5ad=False,
            has_labels=True,
        )

        assert len(Xs) == 3, "Should have 3 modalities"
        assert Xs[0].shape[0] == adata.n_obs, "X_gene count mismatch"
        assert Xs[1].shape[0] == adata.n_obs, "X_img count mismatch"
        assert Xs[2].shape[0] == adata.n_obs, "X_cell count mismatch"
        assert len(Y) == adata.n_obs, "Y count mismatch"

        print(f"[PASS] get_model_input:")
        print(f"  Xs[0]: {Xs[0].shape}")
        print(f"  Xs[1]: {Xs[1].shape}")
        print(f"  Xs[2]: {Xs[2].shape}")
        print(f"  Y: {len(Y)}")


def main():
    """Run tests manually."""
    print("=" * 60)
    print("PP Module Standalone Test")
    print("=" * 60)

    print("\n[INFO] Test 1: preprocess functions")
    test = TestPPPreprocess()
    test.test_preprocess_rna()
    test.test_preprocess_alias()
    test.test_preprocess_rna_from_X()
    test.test_preprocess_omics2_from_obsm()
    test.test_preprocess_atac_alias()

    print("\n[SUCCESS] All pp tests passed!")


if __name__ == "__main__":
    main()
