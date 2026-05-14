"""Smoke test for modality-weight characterization delegation."""

from importlib import import_module
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch

from spanect import SpaNECT


def _build_model(**kwargs):
    config = {
        "device": "cpu",
        "data_path": "/tmp/spanect",
        "k": 2,
        "seed": 0,
        "single_branch_configs": {},
    }
    config.update(kwargs.pop("config", {}))
    return SpaNECT(config=config, **kwargs)


def test_get_modality_weights_delegates_to_ch():
    impl = import_module("spanect.SpaNECT")
    model = _build_model(data_name="dummy", multiomics=False)
    model.model = SimpleNamespace()
    model.Xs = [torch.ones((2, 3), dtype=torch.float32)]
    model.A = torch.sparse_coo_tensor(
        indices=torch.tensor([[0, 1], [1, 0]]),
        values=torch.tensor([1.0, 1.0]),
        size=(2, 2),
    )

    expected = np.array([[0.7, 0.3], [0.4, 0.6]], dtype=np.float32)
    with patch.object(impl.ch, "get_modality_weights_per_cell", return_value=expected) as mocked:
        got = model.get_modality_weights_per_cell(reduce="mean")

    mocked.assert_called_once_with(model=model.model, Xs=model.Xs, A=model.A, reduce="mean")
    assert np.allclose(got, expected)


def test_attach_modality_weights_delegates_to_ch():
    impl = import_module("spanect.SpaNECT")
    model = _build_model(data_name="dummy", multiomics=False)
    model.model = SimpleNamespace()
    model.Xs = [torch.ones((2, 3), dtype=torch.float32)]
    model.A = torch.sparse_coo_tensor(
        indices=torch.tensor([[0, 1], [1, 0]]),
        values=torch.tensor([1.0, 1.0]),
        size=(2, 2),
    )
    model.adata = SimpleNamespace(obs=pd.DataFrame(index=["c0", "c1"]), obsm={}, uns={})

    with patch.object(impl.ch, "attach_modality_weights", return_value=True) as mocked:
        model.attach_modality_weights(reduce="median", key="mw")

    mocked.assert_called_once_with(
        model=model.model,
        adata=model.adata,
        Xs=model.Xs,
        A=model.A,
        reduce="median",
        key="mw",
    )


def test_attach_region_modality_summary_delegates_to_ch():
    impl = import_module("spanect.SpaNECT")
    model = _build_model(data_name="dummy", multiomics=False)
    model.adata = SimpleNamespace(obs=pd.DataFrame({"region": ["a", "b"]}), obsm={"mw": np.ones((2, 2))}, uns={"mw_modalities": ["gene", "image"]})

    with patch.object(impl.ch, "attach_region_modality_summary", return_value=True) as mocked:
        model.attach_region_modality_summary(region_key="region", key="mw")

    mocked.assert_called_once_with(adata=model.adata, region_key="region", key="mw")
