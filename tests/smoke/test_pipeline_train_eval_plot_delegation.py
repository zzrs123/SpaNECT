"""Smoke test for train/evaluate/plot delegation."""

from importlib import import_module
from unittest.mock import patch

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


def test_train_delegates_to_tl():
    impl = import_module("spanect.SpaNECT")
    model = _build_model(data_name="dummy", multiomics=False)

    with patch.object(impl.tl, "train_pipeline", return_value="train-ok") as mocked:
        result = model.train()

    mocked.assert_called_once_with(model)
    assert result == "train-ok"


def test_single_modality_delegates_to_tl():
    impl = import_module("spanect.SpaNECT")
    model = _build_model(data_name="dummy", multiomics=False)

    with patch.object(impl.tl, "train_single_modality", return_value="single-ok") as mocked:
        result = model._train_single_modality("gene")

    mocked.assert_called_once_with(model, "gene")
    assert result == "single-ok"


def test_multimodal_delegates_to_tl():
    impl = import_module("spanect.SpaNECT")
    model = _build_model(data_name="dummy", multiomics=False)

    with patch.object(impl.tl, "train_multimodal", return_value="mm-ok") as mocked:
        result = model._train_multimodal()

    mocked.assert_called_once_with(model)
    assert result == "mm-ok"


def test_evaluate_delegates_to_tl():
    impl = import_module("spanect.SpaNECT")
    model = _build_model(data_name="dummy", multiomics=False)

    with patch.object(impl.tl, "evaluate_pipeline", return_value={"ARI": 1.0}) as mocked:
        result = model.evaluate()

    mocked.assert_called_once_with(model)
    assert result == {"ARI": 1.0}


def test_supervised_eval_dispatch_delegates_to_tl():
    impl = import_module("spanect.SpaNECT")
    model = _build_model(data_name="dummy", multiomics=False)

    with patch.object(impl.tl, "supervised_eval_dispatch", return_value={"ARI": 0.5}) as mocked:
        result = model._supervised_eval_dispatch()

    mocked.assert_called_once_with(model)
    assert result == {"ARI": 0.5}


def test_plot_delegates_to_pl():
    impl = import_module("spanect.SpaNECT")
    model = _build_model(data_name="dummy", multiomics=False)

    with patch.object(impl.pl, "plot_pipeline", return_value="plot-ok") as mocked:
        result = model.plot(spot_size=88)

    mocked.assert_called_once_with(model, spot_size=88)
    assert result == "plot-ok"


def main():
    test_train_delegates_to_tl()
    test_single_modality_delegates_to_tl()
    test_multimodal_delegates_to_tl()
    test_evaluate_delegates_to_tl()
    test_supervised_eval_dispatch_delegates_to_tl()
    test_plot_delegates_to_pl()
    print("pipeline train/evaluate/plot delegation smoke test passed")


if __name__ == "__main__":
    main()
