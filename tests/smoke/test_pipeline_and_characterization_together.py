"""Smoke test for pipeline and characterization coexistence."""

from spanect import SpaNECT
from spanect import SpaNECT as sn


def main():
    # Pipeline facade remains importable as the primary class.
    assert SpaNECT is sn, "SpaNECT alias should resolve to the same class"

    # Characterization access hangs off the same facade class.
    checks = {
        "sn.datasets.load_adata": hasattr(sn.datasets, "load_adata"),
        "sn.gr.prepare_graph_inputs": hasattr(sn.gr, "prepare_graph_inputs"),
        "sn.gr.build_spatial_graph": hasattr(sn.gr, "build_spatial_graph"),
        "sn.tl.train_pipeline": hasattr(sn.tl, "train_pipeline"),
        "sn.tl.evaluate_pipeline": hasattr(sn.tl, "evaluate_pipeline"),
        "sn.pl.plot_pipeline": hasattr(sn.pl, "plot_pipeline"),
        "sn.ch.load_adata": hasattr(sn.ch, "load_adata"),
        "sn.ch.prepare_graph_inputs": hasattr(sn.ch, "prepare_graph_inputs"),
        "sn.ch.train_pipeline": hasattr(sn.ch, "train_pipeline"),
        "sn.ch.plot_pipeline": hasattr(sn.ch, "plot_pipeline"),
    }
    for key, ok in checks.items():
        print(f"{key}: {ok}")

    if not all(checks.values()):
        raise SystemExit(1)

    # Verify the facade class itself is still instantiable.
    assert callable(SpaNECT), "SpaNECT should remain callable"

    # Verify characterization aliases forward to the expected package-level APIs.
    assert sn.ch.load_adata is sn.datasets.load_adata, "sn.ch.load_adata should reuse datasets.load_adata"
    assert (
        sn.ch.prepare_graph_inputs is sn.gr.prepare_graph_inputs
    ), "sn.ch.prepare_graph_inputs should reuse gr.prepare_graph_inputs"
    assert sn.ch.train_pipeline is sn.tl.train_pipeline, "sn.ch.train_pipeline should reuse tl.train_pipeline"
    assert sn.ch.plot_pipeline is sn.pl.plot_pipeline, "sn.ch.plot_pipeline should reuse pl.plot_pipeline"

    print("pipeline and characterization coexistence smoke test passed")


if __name__ == "__main__":
    main()
