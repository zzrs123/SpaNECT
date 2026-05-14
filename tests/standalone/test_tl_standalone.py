"""
Standalone test for tl (tools/training) module.
Tests that tl module can run training and evaluation independently.

Usage:
    pytest tests/standalone/test_tl_standalone.py -v
    python tests/standalone/test_tl_standalone.py
"""

import sys
from pathlib import Path

project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root / "src"))

import numpy as np
import pytest
import torch


class TestTLGraphOps:
    """Test tl graph operations."""

    def test_normalize(self):
        """Test normalize function."""
        from spanect.tl import normalize

        indices = torch.tensor([[0, 0, 1, 1], [0, 1, 0, 1]])
        values = torch.tensor([1.0, 0.5, 0.5, 1.0])
        A = torch.sparse_coo_tensor(indices, values, (2, 2))
        
        A_norm = normalize(A, add_self_loops=True)

        assert A_norm.shape == A.shape, "Shape should be preserved"
        print(f"[PASS] normalize: {A_norm.shape}")

    def test_augment(self):
        """Test augment function."""
        from spanect.tl import augment

        indices = torch.tensor([[0, 0, 1, 1], [0, 1, 0, 1]])
        values = torch.tensor([1.0, 0.5, 0.5, 1.0])
        A = torch.sparse_coo_tensor(indices, values, (2, 2))
        X = torch.rand(2, 5)

        A_aug, X_aug = augment(A, X, edge_mask_rate=0.1, feat_drop_rate=0.1)

        assert A_aug.shape == A.shape, "A shape should be preserved"
        assert X_aug.shape == X.shape, "X shape should be preserved"
        print(f"[PASS] augment: A={A_aug.shape}, X={X_aug.shape}")


class TestTLClusterEvaluators:
    """Test tl cluster evaluation functions."""

    def test_evaluate_kmeans(self):
        """Test evaluate_kmeans function."""
        from spanect.tl import evaluate_kmeans
        import scanpy as sc

        adata = sc.datasets.pbmc3k()[:100].copy()
        adata.obsm['embed'] = np.random.rand(100, 10).astype(np.float32)
        adata.obsm['spatial'] = np.random.rand(100, 2).astype(np.float32)
        adata.obs['layer_guess'] = np.random.randint(0, 5, 100)

        ari, nmi, acc, f1 = evaluate_kmeans(adata)

        assert isinstance(ari, float), "ARI should be float"
        assert isinstance(nmi, float), "NMI should be float"
        assert isinstance(acc, float), "ACC should be float"
        assert isinstance(f1, float), "F1 should be float"
        print(f"[PASS] evaluate_kmeans: ARI={ari:.4f}, NMI={nmi:.4f}")


class TestTLPreprocess:
    """Test tl preprocess functions (facade)."""

    def test_preprocess_facade(self):
        """Test that tl preprocess functions are facades to pp."""
        from spanect.tl import preprocess, preprocess_rna, preprocess_omics2_from_obsm
        from spanect.pp import preprocess as pp_preprocess, preprocess_rna as pp_preprocess_rna

        assert preprocess is pp_preprocess, "tl.preprocess should be pp.preprocess"
        assert preprocess_rna is pp_preprocess_rna, "tl.preprocess_rna should be pp.preprocess_rna"
        print("[PASS] tl preprocess facade works")


class TestTLRefine:
    """Test tl refine functions."""

    def test_refine_label(self):
        """Test refine_label function."""
        from spanect.tl import refine_label
        import scanpy as sc

        adata = sc.datasets.pbmc3k()[:50].copy()
        adata.obsm['spatial'] = np.random.rand(50, 2).astype(np.float32)
        adata.obs['label'] = np.random.randint(0, 3, 50).astype(str)

        refined = refine_label(adata, radius=5, key='label')

        assert len(refined) == adata.n_obs, "Refined labels should have same length"
        print(f"[PASS] refine_label: {len(refined)} labels")


def main():
    """Run tests manually."""
    print("=" * 60)
    print("TL Module Standalone Test")
    print("=" * 60)

    print("\n[INFO] Test 1: graph operations")
    test_graph = TestTLGraphOps()
    test_graph.test_normalize()
    test_graph.test_augment()

    print("\n[INFO] Test 2: cluster evaluators")
    test_cluster = TestTLClusterEvaluators()
    test_cluster.test_evaluate_kmeans()

    print("\n[INFO] Test 3: preprocess facade")
    test_pp = TestTLPreprocess()
    test_pp.test_preprocess_facade()

    print("\n[INFO] Test 4: refine functions")
    test_refine = TestTLRefine()
    test_refine.test_refine_label()

    print("\n[SUCCESS] All tl tests passed!")


if __name__ == "__main__":
    main()
