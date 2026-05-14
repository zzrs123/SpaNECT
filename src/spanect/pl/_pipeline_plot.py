"""Plot pipeline facade functions for SpaNECT."""

import numpy as np

from ._cluster import visualize_kmeans_result


def plot_pipeline(model, spot_size=150):
    if model.has_labels and (model.Y is not None):
        n_clusters = len(np.unique(model._safe_to_numpy_1d(model.Y)))
    else:
        if "cluster_pred" in model.adata.obs:
            n_clusters = len(np.unique(model.adata.obs["cluster_pred"]))
        else:
            n_clusters = model.config.get("n_clusters", 7)

    visualize_kmeans_result(
        model.data_name,
        model.adata,
        n_clusters=n_clusters,
        save_dir=model.save_path,
        spot_size=spot_size,
    )


__all__ = ["plot_pipeline"]
