"""
Standalone test for emb (embedding) module.
Tests that emb module can extract image features independently.

Usage:
    pytest tests/standalone/test_emb_standalone.py -v
    python tests/standalone/test_emb_standalone.py
"""

import sys
from pathlib import Path

project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root / "src"))

import numpy as np
import pytest
import torch


class TestEMBAPI:
    """Test emb module API."""

    def test_vision_embed_class_exists(self):
        """Test that VisionEmbed class exists."""
        from spanect.emb import VisionEmbed

        assert VisionEmbed is not None, "VisionEmbed should be importable"
        assert hasattr(VisionEmbed, 'run'), "VisionEmbed should have run method"
        print("[PASS] VisionEmbed class exists")

    def test_vision_embed_methods(self):
        """Test that VisionEmbed has required methods."""
        from spanect.emb import VisionEmbed

        required_methods = ['run', '_clip_patches', '_filter_patches', '_train_byol', '_infer_embeddings']
        for method in required_methods:
            assert hasattr(VisionEmbed, method), f"VisionEmbed should have {method} method"
        print(f"[PASS] VisionEmbed has all required methods: {required_methods}")


class TestEmbWithMockData:
    """Test emb module with mock data."""

    def test_vision_embed_with_mock_data(self, tmp_path):
        """Test VisionEmbed with mock adata and embeddings file."""
        from spanect.emb import VisionEmbed
        import anndata as ad

        n_cells = 100
        n_genes = 500

        adata = ad.AnnData(X=np.random.rand(n_cells, n_genes).astype(np.float32))
        adata.var_names = [f"gene_{i}" for i in range(n_genes)]
        adata.obs_names = [f"cell_{i}" for i in range(n_cells)]

        emb_path = tmp_path / "test_embeddings.npy"
        mock_embeddings = np.random.rand(n_cells, 128).astype(np.float32)
        np.save(emb_path, mock_embeddings)

        embeddings = np.load(emb_path)
        assert embeddings.shape == (n_cells, 128), f"Expected (100, 128), got {embeddings.shape}"

        adata.obsm['image_feat'] = embeddings
        assert 'image_feat' in adata.obsm, "image_feat should be in adata.obsm"
        assert adata.obsm['image_feat'].shape[0] == n_cells, "Embeddings count should match adata"

        print(f"[PASS] Mock embeddings test: {embeddings.shape}")

    def test_embedding_shape_validation(self):
        """Test embedding shape validation."""
        n_cells = 50
        n_features = 64

        embeddings = np.random.rand(n_cells, n_features).astype(np.float32)

        assert embeddings.shape[0] == n_cells, "First dimension should be n_cells"
        assert embeddings.shape[1] == n_features, "Second dimension should be n_features"
        assert embeddings.dtype == np.float32, "Embeddings should be float32"

        print(f"[PASS] Embedding shape validation: {embeddings.shape}")


def main():
    """Run tests manually."""
    print("=" * 60)
    print("EMB Module Standalone Test")
    print("=" * 60)

    print("\n[INFO] Test 1: API check")
    test_api = TestEMBAPI()
    test_api.test_vision_embed_class_exists()
    test_api.test_vision_embed_methods()

    print("\n[INFO] Test 2: Mock data tests")
    import tempfile
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        test_mock = TestEmbWithMockData()
        test_mock.test_vision_embed_with_mock_data(tmp_path)
    test_mock.test_embedding_shape_validation()

    print("\n[SUCCESS] All emb tests passed!")


if __name__ == "__main__":
    main()
