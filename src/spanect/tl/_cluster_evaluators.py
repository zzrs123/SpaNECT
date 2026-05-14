"""Cluster evaluators for SpaNECT."""

import os
import numpy as np
import pandas as pd
import scanpy as sc
from scipy.spatial import distance

from ._cluster_postprocess import _optimize_cluster, _priori_cluster, refine, refine_label, refine_labels_knn
from ._metrics import metrics
from ._seed import fix_seed


def get_cluster_data(
        adata,
        n_domains,
        priori=True,
        ):
    print("Embeddings shape:", adata.obsm['embed'].shape)
    print("Embeddings sample:", adata.obsm['embed'][:5, :5])
    assert not np.isnan(adata.obsm['embed']).any(), "NaN values found in embeddings!"
    assert not np.isinf(adata.obsm['embed']).any(), "Inf values found in embeddings!"
    sc.pp.neighbors(adata, use_rep='embed')
    if priori:
        res = _priori_cluster(adata, n_domains=n_domains)
    else:
        res = _optimize_cluster(adata)
    sc.tl.leiden(adata, key_added="SpaNECT_niche", resolution=res)
    adj_2d = distance.cdist(adata.obsm['spatial'], adata.obsm['spatial'], 'euclidean')
    refined_pred = refine(
        sample_id=adata.obs.index.tolist(),
        pred=adata.obs["SpaNECT_niche"].tolist(),
        dis=adj_2d,
        shape="hexagon",
    )
    adata.obs["SpaNECT_niche"] = refined_pred
    return adata


def get_evaluate_ARI_result(adata, label_key='layer_guess'):
    from sklearn import metrics as sk_metrics
    sub_adata = adata[~pd.isnull(adata.obs[label_key])]
    ARI = sk_metrics.adjusted_rand_score(sub_adata.obs[label_key], sub_adata.obs['SpaNECT_niche'])
    return ARI


def evaluate_kmeans(adata, label_key='layer_guess', clustering_method='kMeans', logger=None, radius=50):
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import LabelEncoder
    logger = print if logger is None else logger

    if label_key not in adata.obs.columns:
        raise KeyError(f"Label key '{label_key}' not found in adata.obs. Available keys: {list(adata.obs.columns)}")

    label_encoder = LabelEncoder()
    Y = label_encoder.fit_transform(adata.obs[label_key])
    print(np.unique(Y).shape)
    Z = adata.obsm['embed']
    n_clusters = np.unique(Y).shape[0]
    print(n_clusters)

    ACCs = []
    NMIs = []
    ARIs = []
    F1s = []

    best_ari = -1
    best_labels = None

    for i in range(1):
        fix_seed(i)
        # kmeans = KMeans(n_clusters=n_clusters, random_state=i, n_init=10)
        kmeans = KMeans(n_clusters=n_clusters, random_state=1, n_init=10)
        Y_pred = kmeans.fit_predict(Z)
        adata.obs['SpaNECT_niche'] = Y_pred
        new_type = refine_label(adata, radius=radius, key='SpaNECT_niche')
        adata.obs['SpaNECT_niche'] = new_type
        acc, nmi, ari, f1 = metrics(Y, new_type)
        NMIs.append(nmi)
        ARIs.append(ari)
        ACCs.append(acc)
        F1s.append(f1)

        if ari > best_ari:
            best_ari = ari
            best_labels = new_type

    adata.obs['SpaNECT_niche'] = best_labels
    adata.obs['SpaNECT_niche_inferred'] = best_labels
    adata.obs['SpaNECT_niche_inferred'] = adata.obs['SpaNECT_niche_inferred'].astype('category')

    acc_mean = np.mean(ACCs) * 100
    acc_std = np.std(ACCs) * 100
    nmi_mean = np.mean(NMIs) * 100
    nmi_std = np.std(NMIs) * 100
    ari_mean = np.mean(ARIs) * 100
    ari_std = np.std(ARIs) * 100
    f1_mean = np.mean(F1s) * 100
    f1_std = np.std(F1s) * 100

    s = (f"KMeans Evaluation: ACC={acc_mean:.2f}+-{acc_std:.2f}, "
         f"NMI={nmi_mean:.2f}+-{nmi_std:.2f}, "
         f"ARI={ari_mean:.2f}+-{ari_std:.2f}, "
         f"F1={f1_mean:.2f}+-{f1_std:.2f}")
    logger(s)
    return ari_mean, nmi_mean, acc_mean, f1_mean


def evaluate_mclust(
          adata,
          repeat=10,
          radius=50,
          embed_key='embed',
          label_key='layer_guess',
          method_name='mclust'
    ):
    import rpy2.robjects as robjects
    import rpy2.robjects.numpy2ri
    from sklearn.preprocessing import LabelEncoder

    if label_key not in adata.obs.columns:
        raise KeyError(f"Label key '{label_key}' not found in adata.obs. Available keys: {list(adata.obs.columns)}")

    # os.environ['R_HOME'] = '/home/userhome/.conda/envs/R_env/lib/R'
    # robjects.r('.libPaths("/home/userhome/.conda/envs/R_env/lib/R/library")')
    # robjects.r.library("mclust")

    r_home = os.environ.get("R_HOME")
    r_libs_user = os.environ.get("R_LIBS_USER")

    if r_home:
        os.environ["R_HOME"] = r_home

    if r_libs_user:
        robjects.r(f'.libPaths("{r_libs_user}")')

    try:
        robjects.r.library("mclust")
    except Exception as e:
        raise RuntimeError(
            "Failed to load R package 'mclust'. "
            "Please make sure R is installed, the package 'mclust' is available, "
            "and set R_HOME / R_LIBS_USER if R is not auto-detected."
        ) from e
    
    rpy2.robjects.numpy2ri.activate()
    label_encoder = LabelEncoder()
    Y = label_encoder.fit_transform(adata.obs[label_key])
    Z = adata.obsm[embed_key]
    n_clusters = len(np.unique(Y))

    best_ari = -1
    best_labels = None
    accs, nmis, aris, f1s = [], [], [], []

    for i in range(repeat):
        fix_seed(i)
        robjects.r['set.seed'](i)
        adata_copy = adata.copy()
        r_Mclust = robjects.r['Mclust']
        res = r_Mclust(rpy2.robjects.numpy2ri.numpy2rpy(Z), n_clusters, 'EEE')
        mclust_labels = np.array(res[-2])

        adata_copy.obs[method_name] = mclust_labels
        adata_copy.obs[method_name] = adata_copy.obs[method_name].astype('int')
        adata_copy.obs[method_name] = adata_copy.obs[method_name].astype('category')
        new_type = refine_label(adata_copy, radius=radius, key=method_name)
        refined_labels = [int(x) for x in new_type]
        acc, nmi, ari, f1 = metrics(Y, refined_labels)

        accs.append(acc)
        nmis.append(nmi)
        aris.append(ari)
        f1s.append(f1)

        if ari > best_ari:
            best_ari = ari
            best_labels = new_type

    adata.obs[f'{method_name}_domain_pred'] = best_labels
    adata.obs[f'{method_name}_domain_pred'] = adata.obs[f'{method_name}_domain_pred'].astype('category')
    adata.obs['domain_pred'] = adata.obs[f'{method_name}_domain_pred']

    return {
        'ACC': np.mean(accs) * 100,
        'NMI': np.mean(nmis) * 100,
        'ARI': np.mean(aris) * 100,
        'F1': np.mean(f1s) * 100,
    }


def evaluate_unlabeled(adata, embed, n_clusters, n_neighbors=50, seed=0, spatial_key='spatial'):
    """
    Evaluate unlabeled clustering with spatial refinement.
    
    Parameters
    ----------
    adata : AnnData
        Annotated data matrix.
    embed : ndarray
        Embedding matrix (N x D).
    n_clusters : int
        Number of clusters.
    n_neighbors : int
        Number of neighbors for refinement.
    seed : int
        Random seed.
    spatial_key : str
        Key in adata.obsm for spatial coordinates.
        
    Returns
    -------
    dict
        Evaluation metrics.
    """
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score, davies_bouldin_score

    if spatial_key not in adata.obsm:
        raise KeyError(f"adata.obsm['{spatial_key}'] not found.")

    km = KMeans(n_clusters=n_clusters, n_init='auto', random_state=seed)
    pred_raw = km.fit_predict(embed)
    adata.obs['cluster_pred_raw'] = pred_raw.astype(int)

    coords = np.asarray(adata.obsm[spatial_key])
    pred_refined = refine_labels_knn(pred_raw, coords, n_neighbors=n_neighbors)
    pred_refined = pred_refined.astype(int)
    adata.obs['cluster_pred'] = pred_refined

    def _safe_sc_db(X, y):
        y = np.asarray(y)
        if len(np.unique(y)) < 2 or X.shape[0] <= len(np.unique(y)):
            return None, None
        try:
            sc = silhouette_score(X, y)
        except Exception:
            sc = None
        try:
            db = davies_bouldin_score(X, y)
        except Exception:
            db = None
        return sc, db

    sc_raw, db_raw = _safe_sc_db(embed, pred_raw)
    sc_ref, db_ref = _safe_sc_db(embed, pred_refined)

    def _fmt(x): return "None" if x is None else f"{x:.4f}"
    print(f"[INFO] (raw)  Silhouette={_fmt(sc_raw)},  DaviesBouldin={_fmt(db_raw)}  (k={n_clusters})")
    print(f"[INFO] (ref.) Silhouette={_fmt(sc_ref)},  DaviesBouldin={_fmt(db_ref)}  (k={n_clusters})")

    return {
        'Silhouette_raw': sc_raw,
        'DaviesBouldin_raw': db_raw,
        'Silhouette': sc_ref,
        'DaviesBouldin': db_ref,
        'n_clusters': n_clusters
    }


__all__ = [
    "get_cluster_data",
    "get_evaluate_ARI_result",
    "evaluate_kmeans",
    "evaluate_mclust",
    "evaluate_unlabeled",
]