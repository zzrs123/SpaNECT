"""Smoke test for graph build API exposure."""

from spanect import SpaNECT as sn
from spanect.gr import graph as build_graph
from spanect.gr import combine_graph_dict as build_combine_graph_dict


def main():
    checks = {
        "sn.gr.graph": hasattr(sn.gr, "graph"),
        "sn.gr.combine_graph_dict": hasattr(sn.gr, "combine_graph_dict"),
    }
    for key, ok in checks.items():
        print(f"{key}: {ok}")

    if not all(checks.values()):
        raise SystemExit(1)

    assert sn.gr.graph is build_graph, "sn.gr should expose graph builder"
    assert (
        sn.gr.combine_graph_dict is build_combine_graph_dict
    ), "sn.gr should expose combine_graph_dict"

    print("graph build API smoke test passed")


if __name__ == "__main__":
    main()
