"""Smoke test for training graph-ops API exposure."""

from spanect import SpaNECT as sn
from spanect.tl._graph_ops import (
    adj_matrix_to_edge_index as tl_graph_ops_adj_matrix_to_edge_index,
    augment as tl_graph_ops_augment,
    normalize as tl_graph_ops_normalize,
)


def main():
    checks = {
        "sn.tl._graph_ops": hasattr(sn.tl, "_graph_ops"),
        "sn.tl.adj_matrix_to_edge_index": hasattr(sn.tl, "adj_matrix_to_edge_index"),
        "sn.tl.augment": hasattr(sn.tl, "augment"),
        "sn.tl.normalize": hasattr(sn.tl, "normalize"),
    }
    for key, ok in checks.items():
        print(f"{key}: {ok}")

    if not all(checks.values()):
        raise SystemExit(1)

    assert (
        sn.tl.adj_matrix_to_edge_index is tl_graph_ops_adj_matrix_to_edge_index
    ), "sn.tl.adj_matrix_to_edge_index should expose tl._graph_ops"
    assert sn.tl.augment is tl_graph_ops_augment, "sn.tl.augment should expose tl._graph_ops"
    assert sn.tl.normalize is tl_graph_ops_normalize, "sn.tl.normalize should expose tl._graph_ops"

    print("training graph-ops API smoke test passed")


if __name__ == "__main__":
    main()
