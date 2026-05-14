"""
Standalone test for characterization modality-weight functions.
"""

import sys
from pathlib import Path

project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root / "src"))

import numpy as np
import torch
import anndata as ad

from spanect import characterization as ch


class DummyModel:
    def __init__(self):
        self.modality_names = ["gene", "omics2", "celltype"]
        self.last_attn_weights = {
            "gene": torch.tensor([[0.7, 0.2, 0.1], [0.6, 0.3, 0.1]], dtype=torch.float32),
            "omics2": torch.tensor([[0.2, 0.7, 0.1], [0.2, 0.6, 0.2]], dtype=torch.float32),
            "celltype": torch.tensor([[0.1, 0.2, 0.7], [0.2, 0.2, 0.6]], dtype=torch.float32),
        }

    def eval(self):
        return self

    def __call__(self, inputs, modality_to_decode='all'):
        return None


def test_modality_weights_writeback():
    X = np.random.rand(2, 4).astype(np.float32)
    adata = ad.AnnData(X)
    adata.obs["region"] = ["r0", "r1"]

    model = DummyModel()
    A = torch.sparse_coo_tensor(
        indices=torch.tensor([[0, 1], [1, 0]]),
        values=torch.tensor([1.0, 1.0]),
        size=(2, 2),
    )
    Xs = [torch.ones((2, 3), dtype=torch.float32) for _ in range(3)]

    W = ch.get_modality_weights_per_cell(model=model, Xs=Xs, A=A, reduce="mean")
    assert W.shape == (2, 3)
    assert np.allclose(W.sum(axis=1), np.ones(2), atol=1e-6)

    ok = ch.attach_modality_weights(model=model, adata=adata, Xs=Xs, A=A, reduce="mean", key="mw")
    assert ok is True
    assert "mw" in adata.obsm
    assert "mw_modalities" in adata.uns
    assert "mw_top" in adata.obs
    assert "mw_entropy" in adata.obs

    ok2 = ch.attach_region_modality_summary(adata=adata, region_key="region", key="mw")
    assert ok2 is True
    assert "mw_by_region_regions" in adata.uns
    assert "mw_by_region_mean" in adata.uns
    assert "mw_by_region_median" in adata.uns
