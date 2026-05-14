"""Smoke test for plotting cluster API exposure."""

from spanect import SpaNECT as sn
from spanect.pl._cluster import (
    visualize_kmeans_result as cluster_visualize_kmeans_result,
    visualize_kmeans_result_old as cluster_visualize_kmeans_result_old,
)


def main():
    checks = {
        "sn.pl.visualize_kmeans_result": hasattr(sn.pl, "visualize_kmeans_result"),
        "sn.pl.visualize_kmeans_result_old": hasattr(sn.pl, "visualize_kmeans_result_old"),
        "sn.pl._cluster": hasattr(sn.pl, "_cluster"),
    }
    for key, ok in checks.items():
        print(f"{key}: {ok}")

    if not all(checks.values()):
        raise SystemExit(1)

    assert (
        sn.pl.visualize_kmeans_result is cluster_visualize_kmeans_result
    ), "sn.pl.visualize_kmeans_result should expose pl._cluster implementation"
    assert (
        sn.pl.visualize_kmeans_result_old is cluster_visualize_kmeans_result_old
    ), "sn.pl.visualize_kmeans_result_old should expose pl._cluster implementation"

    print("plot cluster API smoke test passed")


if __name__ == "__main__":
    main()
