"""Smoke test for characterization-mode style access."""

from spanect import SpaNECT as sn


def main():
    checks = {
        "sn.datasets": hasattr(sn, "datasets"),
        "sn.gr": hasattr(sn, "gr"),
        "sn.tl": hasattr(sn, "tl"),
        "sn.pl": hasattr(sn, "pl"),
        "sn.ch": hasattr(sn, "ch"),
    }
    for key, ok in checks.items():
        print(f"{key}: {ok}")

    if not all(checks.values()):
        raise SystemExit(1)

    # Minimal attribute checks to ensure capability modules are reachable.
    assert hasattr(sn.datasets, "load_adata"), "sn.datasets.load_adata not found"
    assert hasattr(sn.gr, "prepare_graph_inputs"), "sn.gr.prepare_graph_inputs not found"
    assert hasattr(sn.gr, "build_spatial_graph"), "sn.gr.build_spatial_graph not found"
    assert hasattr(sn.tl, "train_pipeline"), "sn.tl.train_pipeline not found"
    assert hasattr(sn.tl, "evaluate_pipeline"), "sn.tl.evaluate_pipeline not found"
    assert hasattr(sn.pl, "plot_pipeline"), "sn.pl.plot_pipeline not found"
    assert hasattr(sn.ch, "load_adata"), "sn.ch.load_adata not found"
    assert hasattr(sn.ch, "prepare_graph_inputs"), "sn.ch.prepare_graph_inputs not found"
    assert hasattr(sn.ch, "train_pipeline"), "sn.ch.train_pipeline not found"
    assert hasattr(sn.ch, "plot_pipeline"), "sn.ch.plot_pipeline not found"

    print("characterization smoke test passed")


if __name__ == "__main__":
    main()
