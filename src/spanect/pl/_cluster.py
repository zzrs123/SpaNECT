"""Cluster visualization for SpaNECT."""

import os
import numpy as np
import pandas as pd
import scanpy as sc
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances, calinski_harabasz_score
from sklearn.metrics import accuracy_score, f1_score, \
    normalized_mutual_info_score, adjusted_rand_score
from munkres import Munkres

from ..tl._cluster_postprocess import refine_label


def visualize_kmeans_result_old(data_name, adata, n_clusters, save_dir="./Results"):
    """
    利用KMeans聚类后细化标签的结果，在空间上进行可视化并保存图像
    """
    Z = adata.obsm['embed']
    kmeans = KMeans(n_clusters=n_clusters, random_state=1)
    Y_pred = kmeans.fit_predict(Z)
    adata.obs['SpaNECT_niche_inferred'] = Y_pred
    new_type = refine_label(adata, radius=50, key='SpaNECT_niche_inferred')
    adata.obs['SpaNECT_niche_inferred'] = new_type
    sc.pl.spatial(adata, color='domain_pred', frameon=False, spot_size=150)
    plt.savefig(os.path.join(save_dir, f'{data_name}_domains_kmeans.pdf'),
                bbox_inches='tight', dpi=300)


def visualize_kmeans_result(data_name, adata, n_clusters,
                            save_dir="./Results", spot_size=150):
    """
    利用KMeans聚类后细化标签的结果，在空间上进行可视化并保存图像
    """
    if 'SpaNECT_niche' in adata.obs:
        adata.obs['SpaNECT_niche_inferred'] = adata.obs['SpaNECT_niche']
    elif 'cluster_pred' in adata.obs:
        adata.obs['SpaNECT_niche_inferred'] = adata.obs['cluster_pred'].astype(str)
    else:
        raise KeyError("Neither 'SpaNECT_niche' nor 'cluster_pred' found in adata.obs")

    sc.pl.spatial(
        adata,
        color='SpaNECT_niche_inferred',
        frameon=False,
        spot_size=spot_size,
        img_key=None,
    )

    plt.savefig(
        os.path.join(save_dir, f'{data_name}_SpaNECT_kmeans.pdf'),
        bbox_inches='tight', dpi=300
    )
    plt.close()


__all__ = ["visualize_kmeans_result", "visualize_kmeans_result_old"]
