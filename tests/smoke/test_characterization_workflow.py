"""Smoke test for direct characterization-style access."""

from spanect import SpaNECT as sn


def main():
    checks = {
        "sn.datasets.load_adata": hasattr(sn.datasets, "load_adata"),
        "sn.gr.prepare_graph_inputs": hasattr(sn.gr, "prepare_graph_inputs"),
        "sn.gr.build_spatial_graph": hasattr(sn.gr, "build_spatial_graph"),
        "sn.tl.train_pipeline": hasattr(sn.tl, "train_pipeline"),
        "sn.tl.evaluate_pipeline": hasattr(sn.tl, "evaluate_pipeline"),
        "sn.pl.plot_pipeline": hasattr(sn.pl, "plot_pipeline"),
        "sn.ch.load_adata": hasattr(sn.ch, "load_adata"),
        "sn.ch.prepare_graph_inputs": hasattr(sn.ch, "prepare_graph_inputs"),
        "sn.ch.build_spatial_graph": hasattr(sn.ch, "build_spatial_graph"),
        "sn.ch.train_pipeline": hasattr(sn.ch, "train_pipeline"),
        "sn.ch.evaluate_pipeline": hasattr(sn.ch, "evaluate_pipeline"),
        "sn.ch.plot_pipeline": hasattr(sn.ch, "plot_pipeline"),
    }
    for key, ok in checks.items():
        print(f"{key}: {ok}")

    if not all(checks.values()):
        raise SystemExit(1)

    assert sn.ch.load_adata is sn.datasets.load_adata, "sn.ch.load_adata should forward sn.datasets.load_adata"
    assert (
        sn.ch.prepare_graph_inputs is sn.gr.prepare_graph_inputs
    ), "sn.ch.prepare_graph_inputs should forward sn.gr.prepare_graph_inputs"
    assert (
        sn.ch.build_spatial_graph is sn.gr.build_spatial_graph
    ), "sn.ch.build_spatial_graph should forward sn.gr.build_spatial_graph"
    assert sn.ch.train_pipeline is sn.tl.train_pipeline, "sn.ch.train_pipeline should forward sn.tl.train_pipeline"
    assert (
        sn.ch.evaluate_pipeline is sn.tl.evaluate_pipeline
    ), "sn.ch.evaluate_pipeline should forward sn.tl.evaluate_pipeline"
    assert sn.ch.plot_pipeline is sn.pl.plot_pipeline, "sn.ch.plot_pipeline should forward sn.pl.plot_pipeline"

    print("characterization workflow smoke test passed")


if __name__ == "__main__":
    main()
