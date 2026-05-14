"""Smoke test for all.py-equivalent manual modular assembly."""

from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch

from spanect import SpaNECT
from spanect import SpaNECT as sn


class DummyAdata:
    def __init__(self):
        self.obs = pd.DataFrame(index=["cell_0", "cell_1"])
        self.obsm = {
            "spatial": np.array([[0.0, 0.0], [1.0, 1.0]], dtype=np.float32),
        }

    def write_h5ad(self, path):
        Path(path).write_text("dummy h5ad\n")


def _build_model(save_path, **kwargs):
    config = {
        "device": "cpu",
        "data_path": "/tmp/spanect",
        "k": 2,
        "seed": 0,
        "single_branch_configs": {},
    }
    config.update(kwargs.pop("config", {}))
    return SpaNECT(config=config, save_path=str(save_path), **kwargs)


def _run_manual_modular_pipeline(model, data_names, output_path):
    all_results = []

    for data_name in data_names:
        model.data_name = data_name
        model.save_path = str(output_path / data_name)

        model.adata = sn.datasets.load_adata(
            platform="Visium",
            data_path=model.config["data_path"],
            data_name=data_name,
        )

        Xs, Y, adata_filtered = sn.tl.get_model_input(
            adata=model.adata,
            data_path=model.config["data_path"],
            data_name=data_name,
            config=model.config,
            device=model.device,
            multiomics=False,
            use_h5ad=False,
            has_labels=True,
        )
        model.Xs = Xs
        model.Y = Y
        if adata_filtered is not None:
            model.adata = adata_filtered
        model.construct_graph()

        sn.tl.train_pipeline(model)

        metrics = sn.tl.evaluate_pipeline(model)
        if isinstance(metrics, dict):
            ari = float(metrics.get("ARI", 0.0))
            nmi = float(metrics.get("NMI", 0.0))
            acc = float(metrics.get("ACC", 0.0))
            f1 = float(metrics.get("F1", 0.0))
        else:
            ari, nmi, acc, f1 = metrics
        all_results.append([data_name, ari, nmi, acc, f1])

        sn.pl.plot_pipeline(model, spot_size=150)

        output_path.mkdir(parents=True, exist_ok=True)
        model.adata.obs["Proposed_cluster"] = model.adata.obs["domain_pred"]
        model.adata.obsm["Proposed"] = model.adata.obsm["embed"]
        model.adata.obs[["Proposed_cluster"]].to_csv(
            output_path / f"{data_name}_FinalLabels.csv",
            index=False,
            header=False,
        )
        model.adata.write_h5ad(output_path / f"{data_name}_processed.h5ad")

    results_df = pd.DataFrame(all_results, columns=["Slice", "ARI", "NMI", "ACC", "F1"])
    results_df.to_csv(output_path / "all_slices_metrics.csv", index=False)
    return results_df


def test_manual_modular_pipeline_matches_all_flow(tmp_path):
    model = _build_model(save_path=tmp_path / "results", data_name=None, multiomics=False)
    output_path = tmp_path / "all_dlpfc"
    data_name = "151507"
    events = []

    sparse_adj = torch.sparse_coo_tensor(
        indices=torch.tensor([[0, 1], [1, 0]]),
        values=torch.tensor([1.0, 1.0]),
        size=(2, 2),
    )

    def _load_adata(**kwargs):
        events.append(("load", kwargs["data_name"]))
        return DummyAdata()

    def _prepare_graph_inputs(adata, data_path, data_name_arg, device=None, k=None, cnnType="ResNet50"):
        events.append(("graph", data_name_arg))
        return (
            np.ones((2, 4), dtype=np.float32),
            np.ones((2, 5), dtype=np.float32),
            np.ones((2, 6), dtype=np.float32),
            {"adj_norm": sparse_adj},
            np.array([0, 1]),
        )

    def _get_model_input(
        adata,
        data_path=None,
        data_name=None,
        config=None,
        device='cuda',
        multiomics=False,
        use_h5ad=False,
        h5ad_path=None,
        omics2_key=None,
        has_labels=True,
    ):
        events.append(("graph", data_name))
        return (
            [
                torch.ones((2, 4), dtype=torch.float32),
                torch.ones((2, 5), dtype=torch.float32),
                torch.ones((2, 6), dtype=torch.float32),
            ],
            np.array([0, 1]),
            None,
        )

    def _train_pipeline(pipeline_model):
        events.append(("train", pipeline_model.data_name))
        return None

    def _evaluate_pipeline(pipeline_model):
        events.append(("evaluate", pipeline_model.data_name))
        pipeline_model.adata.obs["domain_pred"] = ["0", "1"]
        pipeline_model.adata.obsm["embed"] = np.array(
            [[1.0, 2.0], [3.0, 4.0]],
            dtype=np.float32,
        )
        return {"ARI": 11.0, "NMI": 22.0, "ACC": 33.0, "F1": 44.0}

    def _plot_pipeline(pipeline_model, spot_size=150):
        events.append(("plot", pipeline_model.data_name, spot_size))
        return None

    with patch.object(sn.datasets, "load_adata", side_effect=_load_adata) as load_mock, \
            patch.object(sn.tl, "get_model_input", side_effect=_get_model_input) as graph_mock, \
            patch.object(sn.gr, "build_spatial_graph", return_value={"adj_norm": sparse_adj}) as adj_mock, \
            patch.object(sn.tl, "train_pipeline", side_effect=_train_pipeline) as train_mock, \
            patch.object(sn.tl, "evaluate_pipeline", side_effect=_evaluate_pipeline) as eval_mock, \
            patch.object(sn.pl, "plot_pipeline", side_effect=_plot_pipeline) as plot_mock:
        results_df = _run_manual_modular_pipeline(
            model=model,
            data_names=[data_name],
            output_path=output_path,
        )

    load_mock.assert_called_once_with(
        platform="Visium",
        data_path="/tmp/spanect",
        data_name=data_name,
    )
    graph_mock.assert_called_once()
    train_mock.assert_called_once_with(model)
    eval_mock.assert_called_once_with(model)
    plot_mock.assert_called_once_with(model, spot_size=150)

    assert events == [
        ("load", data_name),
        ("graph", data_name),
        ("train", data_name),
        ("evaluate", data_name),
        ("plot", data_name, 150),
    ]

    assert list(results_df.columns) == ["Slice", "ARI", "NMI", "ACC", "F1"]
    assert results_df.iloc[0].to_dict() == {
        "Slice": data_name,
        "ARI": 11.0,
        "NMI": 22.0,
        "ACC": 33.0,
        "F1": 44.0,
    }

    labels_path = output_path / f"{data_name}_FinalLabels.csv"
    metrics_path = output_path / "all_slices_metrics.csv"
    h5ad_path = output_path / f"{data_name}_processed.h5ad"

    assert labels_path.exists(), "Final labels output should be written"
    assert metrics_path.exists(), "Slice metrics output should be written"
    assert h5ad_path.exists(), "Processed h5ad output should be written"

    labels_df = pd.read_csv(labels_path, header=None)
    metrics_df = pd.read_csv(metrics_path)

    assert labels_df.iloc[:, 0].astype(str).tolist() == ["0", "1"]
    metrics_row = metrics_df.iloc[0].to_dict()
    assert str(metrics_row["Slice"]).replace(".0", "") == data_name
    assert metrics_row["ARI"] == 11.0
    assert metrics_row["NMI"] == 22.0
    assert metrics_row["ACC"] == 33.0
    assert metrics_row["F1"] == 44.0
    assert model.adata.obs["Proposed_cluster"].tolist() == ["0", "1"]
    assert np.array_equal(model.adata.obsm["Proposed"], model.adata.obsm["embed"])
