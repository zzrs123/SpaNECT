"""
Standalone test for gr (graph) module.
Tests that gr module can build spatial graphs independently.

Usage:
    pytest tests/standalone/test_gr_standalone.py -v
    python tests/standalone/test_gr_standalone.py
"""

import sys
from pathlib import Path

project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root / "src"))

import numpy as np
import pytest


class TestGRBuild:
    """Test gr graph building functions."""

    def test_build_spatial_graph(self):
        """Test build_spatial_graph function."""
        from spanect.gr import build_spatial_graph

        coords = np.random.rand(100, 2) * 100
        A = build_spatial_graph(coords, k=10)

        assert 'adj_norm' in A, "Result should have adj_norm"
        assert A['adj_norm'].shape == (100, 100), f"Expected (100, 100), got {A['adj_norm'].shape}"
        print(f"[PASS] build_spatial_graph: {A['adj_norm'].shape}")

    def test_graph_class(self):
        """Test graph class directly."""
        from spanect.gr.graph import graph

        coords = np.random.rand(50, 2) * 100
        g = graph(coords, k=5, distType='BallTree')
        A = g.main()

        assert 'adj_norm' in A, "Result should have adj_norm"
        assert A['adj_norm'].shape == (50, 50), f"Expected (50, 50), got {A['adj_norm'].shape}"
        print(f"[PASS] graph class: {A['adj_norm'].shape}")


class TestGRPipeline:
    """Test gr pipeline functions with real data."""

    @pytest.fixture
    def data_path(self):
        return "/home/dugaoyuan/AAA-new-base/ztest_tmvc_model/1-04-full-model-can/FullModel/DLPFC"

    @pytest.fixture
    def test_prepare_graph_inputs(self, data_path, data_name):
        """Test prepare_graph_inputs function."""
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
    print("GR Module Standalone Test")
    print("=" * 60)

    print("\n[INFO] Test 1: graph building")
    test = TestGRBuild()
    test.test_build_spatial_graph()
    test.test_graph_class()

    print("\n[INFO] Test 2: prepare_graph_inputs with real data")
    data_path = "/home/dugaoyuan/AAA-new-base/ztest_tmvc_model/1-04-full-model-can/FullModel/DLPFC"
    data_name = "151507"
    test_pipeline = TestGRPipeline()
    test_pipeline.test_prepare_graph_inputs(data_path, data_name)

    print("\n[SUCCESS] All gr tests passed!")


if __name__ == "__main__":
    main()
