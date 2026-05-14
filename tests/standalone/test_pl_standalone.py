"""
Standalone test for pl (plotting) module.
Tests that pl module can plot results independently.

Usage:
    pytest tests/standalone/test_pl_standalone.py -v
    python tests/standalone/test_pl_standalone.py
"""

import sys
from pathlib import Path

project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root / "src"))

import numpy as np
import pytest
import tempfile


class TestPLCluster:
    """Test pl cluster visualization functions."""

    def test_visualize_kmeans_result(self):
        """Test visualize_kmeans_result function."""
        from spanect.pl import visualize_kmeans_result
        import anndata as ad

        X = np.random.rand(100, 10).astype(np.float32)
        adata = ad.AnnData(X)
        adata.obsm['spatial'] = np.random.rand(100, 2) * 100
        adata.obs['cluster_pred'] = np.random.randint(0, 5, 100).astype(str)

        with tempfile.TemporaryDirectory() as tmp_dir:
            save_path = tmp_dir
            visualize_kmeans_result(
                data_name="test",
                adata=adata,
                n_clusters=5,
                save_dir=save_path,
                spot_size=1.0
            )

            expected_file = Path(save_path) / "test_SpaNECT_kmeans.pdf"
            assert expected_file.exists(), f"Plot should be saved to {expected_file}"
            print(f"[PASS] visualize_kmeans_result: saved to {expected_file}")


class TestPLPipeline:
    """Test pl pipeline plotting functions."""

    def test_plot_pipeline(self):
        """Test plot_pipeline function with mock model."""
        from spanect.pl import plot_pipeline
        import anndata as ad

        X = np.random.rand(100, 10).astype(np.float32)
        adata = ad.AnnData(X)
        adata.obsm['spatial'] = np.random.rand(100, 2) * 100
        adata.obs['cluster_pred'] = np.random.randint(0, 5, 100).astype(str)

        class MockModel:
            def __init__(self, adata, save_path):
                self.adata = adata
                self.save_path = save_path
                self.data_name = "test"
                self.has_labels = False
                self.Y = None
                self.config = {"n_clusters": 5}

            def _safe_to_numpy_1d(self, y):
                return np.array([0] * 100)

        with tempfile.TemporaryDirectory() as tmp_dir:
            model = MockModel(adata, tmp_dir)
            plot_pipeline(model, spot_size=1.0)

            expected_file = Path(tmp_dir) / "test_SpaNECT_kmeans.pdf"
            assert expected_file.exists(), f"Plot should be saved to {expected_file}"
            print(f"[PASS] plot_pipeline: saved to {expected_file}")


def main():
    """Run tests manually."""
    print("=" * 60)
    print("PL Module Standalone Test")
    print("=" * 60)

    print("\n[INFO] Test 1: visualize_kmeans_result")
    test_cluster = TestPLCluster()
    test_cluster.test_visualize_kmeans_result()

    print("\n[INFO] Test 2: plot_pipeline")
    test_pipeline = TestPLPipeline()
    test_pipeline.test_plot_pipeline()

    print("\n[SUCCESS] All pl tests passed!")


if __name__ == "__main__":
    main()
