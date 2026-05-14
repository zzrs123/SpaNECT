"""Cluster post-processing helpers for refinement and resolution selection."""

import numpy as np
import pandas as pd
import scanpy as sc
from scipy.spatial import distance
from sklearn.metrics import calinski_harabasz_score


def _optimize_cluster(
        adata,
        resolution: list = list(np.arange(0.01, 2.5, 0.01)),
        ):
        scores = []
        for r in resolution:
            sc.tl.leiden(adata, resolution=r)
            s = calinski_harabasz_score(adata.X, adata.obs["leiden"])
            scores.append(s)
        cl_opt_df = pd.DataFrame({"resolution": resolution, "score": scores})
        best_idx = np.argmax(cl_opt_df["score"])
        res = cl_opt_df.iloc[best_idx, 0]
        print("Best resolution: ", res)
        return res


def _priori_cluster(
    adata,
    n_domains=7,
    ):
    for res in sorted(list(np.arange(0.01, 2.5, 0.01)), reverse=True):
        sc.tl.leiden(adata, random_state=0, resolution=res)
        count_unique_leiden = len(pd.DataFrame(adata.obs['leiden']).leiden.unique())
        if count_unique_leiden == n_domains:
            break
    print("Best resolution: ", res)
    return res


def refine(
    sample_id,
    pred,
    dis,
    shape="hexagon"
    ):
    refined_pred = []
    pred = pd.DataFrame({"pred": pred}, index=sample_id)
    dis_df = pd.DataFrame(dis, index=sample_id, columns=sample_id)
    if shape == "hexagon":
        num_nbs = 6
    elif shape == "square":
        num_nbs = 4
    else:
        print("Shape not recongized, shape='hexagon' for Visium data, 'square' for ST data.")
    for i in range(len(sample_id)):
        index = sample_id[i]
        dis_tmp = dis_df.loc[index, :].sort_values()
        nbs = dis_tmp[0:num_nbs + 1]
        nbs_pred = pred.loc[nbs.index, "pred"]
        self_pred = pred.loc[index, "pred"]
        v_c = nbs_pred.value_counts()
        if (v_c.loc[self_pred] < num_nbs / 2) and (np.max(v_c) > num_nbs / 2):
            refined_pred.append(v_c.idxmax())
        else:
            refined_pred.append(self_pred)
    return refined_pred


def refine_label(adata, radius=50, key='label'):
    n_neigh = radius
    new_type = []
    old_type = adata.obs[key].values
    position = adata.obsm['spatial']
    distance_matrix = distance.cdist(position, position, metric='euclidean')
    n_cell = distance_matrix.shape[0]

    for i in range(n_cell):
        vec = distance_matrix[i, :]
        index = vec.argsort()
        neigh_type = []
        for j in range(1, n_neigh + 1):
            neigh_type.append(old_type[index[j]])
        max_type = max(set(neigh_type), key=neigh_type.count)
        new_type.append(max_type)
    new_type = [str(i) for i in new_type]
    return new_type


__all__ = [
    "_optimize_cluster",
    "_priori_cluster",
    "refine",
    "refine_label",
    "refine_labels_knn",
]


def refine_labels_knn(labels, coords, n_neighbors=50):
    """
    Refine labels using KNN spatial smoothing.
    
    Parameters
    ----------
    labels : array-like
        Input labels to be refined.
    coords : array-like
        Spatial coordinates (N x 2).
    n_neighbors : int
        Number of neighbors for smoothing.
        
    Returns
    -------
    refined_labels : array-like
        Refined labels.
    """
    import numpy as np
    from sklearn.neighbors import NearestNeighbors

    coords = np.asarray(coords)
    labels = np.asarray(labels)
    
    if coords.ndim != 2 or coords.shape[0] != len(labels):
        raise ValueError("spatial coordinates shape mismatch with labels.")

    n = coords.shape[0]
    k = int(max(1, min(n - 1, n_neighbors)))
    nbrs = NearestNeighbors(n_neighbors=k + 1, metric='euclidean').fit(coords)
    indices = nbrs.kneighbors(return_distance=False)[:, 1:]

    refined = np.empty_like(labels)
    for i in range(n):
        neigh = labels[indices[i]]
        vals, counts = np.unique(neigh, return_counts=True)
        refined[i] = vals[np.argmax(counts)]
        
    return refined
