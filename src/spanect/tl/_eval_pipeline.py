"""Evaluation pipeline facade functions for SpaNECT."""

import numpy as np
import torch

from ._graph_ops import normalize
from ._cluster_evaluators import evaluate_kmeans, evaluate_mclust, evaluate_unlabeled


def evaluate_pipeline(model):
    model.model.eval()
    A_norm = normalize(model.A, add_self_loops=True)
    inputs = [(A_norm, X) for X in model.Xs]

    with torch.no_grad():
        if model.multiomics:
            z, recon, _, _ = model.model(inputs, modality_to_decode="all")
        else:
            z, recon, _, _ = model.model(inputs, modality_to_decode="gene")

    embed = z.cpu().detach().numpy()
    model.adata.obsm["embed"] = embed

    if isinstance(recon, dict):
        for key, value in recon.items():
            model.adata.obsm[f"recon_{key}"] = value.detach().cpu().numpy()
    else:
        model.adata.obsm["recon"] = recon.detach().cpu().numpy()

    if model.has_labels and (model.Y is not None):
        y_np = model._safe_to_numpy_1d(model.Y)
        if y_np is not None:
            model.adata.obs["layer_guess"] = y_np

    if model.has_labels:
        res = supervised_eval_dispatch(model)
        print(
            f"[INFO] Eval by '{model.config.get('eval', {}).get('clusterer', 'kmeans')}': "
            f"ARI={res['ARI']:.2f}, NMI={res['NMI']:.2f}, ACC={res['ACC']:.2f}, F1={res['F1']:.2f}"
        )
        return res

    try:
        from sklearn.cluster import KMeans  # noqa: F401
    except ImportError:
        print("[ERROR] scikit-learn is required for SC/DB calculation.")
        return {"Silhouette": None, "DaviesBouldin": None}

    if model.user_n_clusters is not None:
        n_clusters = int(model.user_n_clusters)
        print(f"[INFO] Using user-specified n_clusters={n_clusters}")
    else:
        n_clusters = model.config.get("n_clusters", None)
        if n_clusters is None:
            n_clusters = model._auto_select_k(
                embed,
                k_min=2,
                k_max=10,
                random_state=model.config.get("seed", 0),
            )
        else:
            print(f"[INFO] Using config n_clusters={n_clusters}")

    refine_k = int(model.config.get("refine_k", 50))
    metrics_unlabeled = evaluate_unlabeled(
        adata=model.adata,
        embed=embed,
        n_clusters=n_clusters,
        n_neighbors=refine_k,
        seed=model.config.get("seed", 0),
    )
    return metrics_unlabeled


def supervised_eval_dispatch(model):
    eval_cfg = dict(model.config.get("eval", {}))
    clusterer = str(eval_cfg.get("clusterer", "kmeans")).lower()
    repeat = int(eval_cfg.get("repeat", 10))
    radius = int(eval_cfg.get("radius", 50))
    label_key = str(eval_cfg.get("label_key", "layer_guess"))

    if label_key not in model.adata.obs.columns:
        available_keys = list(model.adata.obs.columns)
        print(f"[WARN] label_key '{label_key}' not found in adata.obs. Available: {available_keys}")
        if "layer_guess" in available_keys:
            label_key = "layer_guess"
        elif "Layer" in available_keys:
            label_key = "Layer"
        elif "ground_truth" in available_keys:
            label_key = "ground_truth"
        else:
            for key in available_keys:
                if "layer" in key.lower() or "truth" in key.lower() or "label" in key.lower():
                    label_key = key
                    break
        print(f"[INFO] Using label_key: {label_key}")

    if clusterer == "kmeans":
        ari, nmi, acc, f1 = evaluate_kmeans(model.adata, label_key=label_key, radius=radius)
        return {"ARI": ari, "NMI": nmi, "ACC": acc, "F1": f1}

    if clusterer == "mclust":
        try:
            res = evaluate_mclust(
                adata=model.adata,
                repeat=repeat,
                radius=radius,
                embed_key="embed",
                label_key=label_key,
                method_name="mclust",
            )
            return {"ARI": res["ARI"], "NMI": res["NMI"], "ACC": res["ACC"], "F1": res["F1"]}
        except Exception as exc:
            print(f"[WARN] evaluate_mclust failed, fallback to KMeans. Reason: {exc}")
            ari, nmi, acc, f1 = evaluate_kmeans(model.adata, label_key=label_key, radius=radius)
            return {"ARI": ari, "NMI": nmi, "ACC": acc, "F1": f1}

    print(f"[WARN] Unknown eval.clusterer='{clusterer}', fallback to KMeans.")
    ari, nmi, acc, f1 = evaluate_kmeans(model.adata, label_key=label_key, radius=radius)
    return {"ARI": ari, "NMI": nmi, "ACC": acc, "F1": f1}


__all__ = ["evaluate_pipeline", "supervised_eval_dispatch"]
