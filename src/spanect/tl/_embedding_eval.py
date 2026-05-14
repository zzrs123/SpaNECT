"""Embedding evaluation helpers for characterization-oriented workflows."""

import numpy as np
from sklearn.cluster import KMeans

from ._metrics import metrics
from ._seed import fix_seed


def evaluate(Z: np.ndarray, Y: np.ndarray, logger=None):
    """Evaluate embeddings on node clustering."""
    logger = print if logger is None else logger.info

    n_clusters = np.unique(Y).shape[0]
    ACCs = []
    NMIs = []
    ARIs = []
    F1s = []

    for i in range(10):
        fix_seed(i)

        kmeans = KMeans(n_clusters=n_clusters, random_state=i, n_init=10)
        Y_ = kmeans.fit_predict(Z)
        acc, nmi, ari, f1 = metrics(Y, Y_)
        ACCs.append(acc)
        NMIs.append(nmi)
        ARIs.append(ari)
        F1s.append(f1)
    ACCs = np.array(ACCs)
    NMIs = np.array(NMIs)
    ARIs = np.array(ARIs)
    F1s = np.array(F1s)
    acc_mean = ACCs.mean() * 100
    acc_std = ACCs.std() * 100
    nmi_mean = NMIs.mean() * 100
    nmi_std = NMIs.std() * 100
    ari_mean = ARIs.mean() * 100
    ari_std = ARIs.std() * 100
    f1_mean = F1s.mean() * 100
    f1_std = F1s.std() * 100
    s = f"ACC={acc_mean:.2f}+-{acc_std:.2f}, NMI={nmi_mean:.2f}+-{nmi_std:.2f}, " \
        f"ARI={ari_mean:.2f}+-{ari_std:.2f}, F1={f1_mean:.2f}+-{f1_std:.2f}"
    logger(s)


__all__ = ["evaluate"]
