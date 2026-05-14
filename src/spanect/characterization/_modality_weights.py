import numpy as np
import pandas as pd
import torch

from ..tl._graph_ops import normalize


def get_modality_weights_per_cell(model, Xs, A, reduce='mean'):
    model.eval()
    A_norm = normalize(A, add_self_loops=True)
    with torch.no_grad():
        _ = model([(A_norm, X) for X in Xs], modality_to_decode='all')

    attn = getattr(model, "last_attn_weights", None)
    if not attn:
        return None

    names = list(getattr(model, "modality_names", []))
    if not names:
        return None

    mats = [attn[q].detach().cpu().numpy() for q in names if q in attn]
    if not mats:
        return None

    W = np.stack(mats, axis=0)

    if reduce == 'mean':
        contrib = W.mean(axis=0)
    elif reduce == 'median':
        contrib = np.median(W, axis=0)
    else:
        raise ValueError("reduce must be 'mean' or 'median'.")

    contrib = np.maximum(contrib, 1e-12)
    contrib = contrib / contrib.sum(axis=1, keepdims=True)
    return contrib


def get_region_modality_weights(model, adata, Xs, A, region_key, reduce='mean'):
    W = get_modality_weights_per_cell(model=model, Xs=Xs, A=A, reduce=reduce)
    if W is None:
        return None
    names = list(getattr(model, "modality_names", []))
    df = pd.DataFrame(W, columns=names, index=adata.obs_names)
    df['region'] = adata.obs[region_key].astype(str).values
    df_long = df.melt(id_vars='region', var_name='modality', value_name='weight')
    return df_long


def attach_modality_weights(model, adata, Xs, A, reduce='mean', key='modality_weights'):
    W = get_modality_weights_per_cell(model=model, Xs=Xs, A=A, reduce=reduce)
    if W is None:
        return False

    names = list(getattr(model, "modality_names", []))
    if not names:
        return False

    adata.obsm[key] = W.astype(np.float32)
    adata.uns[f'{key}_modalities'] = names
    adata.uns[f'{key}_reduce'] = reduce

    for i, m in enumerate(names):
        adata.obs[f'{key}_{m}'] = W[:, i].astype(np.float32)

    top_idx = W.argmax(axis=1)
    adata.obs[f'{key}_top'] = pd.Categorical([names[i] for i in top_idx], categories=names)

    M = W.shape[1]
    p = np.clip(W, 1e-12, 1.0)
    ent = -(p * np.log(p)).sum(axis=1) / np.log(M)
    adata.obs[f'{key}_entropy'] = ent.astype(np.float32)

    attn = getattr(model, "last_attn_weights", None)
    if attn:
        for qname, mat in attn.items():
            adata.obsm[f'{key}_per_query_{qname}'] = mat.detach().cpu().numpy().astype(np.float32)
            adata.uns[f'{key}_per_query_{qname}_modalities'] = names

    return True


def attach_region_modality_summary(adata, region_key, key='modality_weights'):
    if key not in adata.obsm:
        raise KeyError(f"obsm['{key}'] not found, call attach_modality_weights() first.")
    if region_key not in adata.obs:
        raise KeyError(f"obs['{region_key}'] not found.")

    W = adata.obsm[key]
    names = list(adata.uns[f'{key}_modalities'])
    reg = adata.obs[region_key].astype(str).values

    regions = sorted(pd.unique(reg))
    R, M = len(regions), len(names)
    mean = np.zeros((R, M), dtype=np.float32)
    med = np.zeros((R, M), dtype=np.float32)
    for i, r in enumerate(regions):
        mask = (reg == r)
        if mask.any():
            mean[i] = W[mask].mean(axis=0)
            med[i] = np.median(W[mask], axis=0)

    adata.uns[f'{key}_by_{region_key}_regions'] = regions
    adata.uns[f'{key}_by_{region_key}_mean'] = mean
    adata.uns[f'{key}_by_{region_key}_median'] = med
    adata.uns[f'{key}_by_{region_key}_modalities'] = names
    return True


__all__ = [
    "get_modality_weights_per_cell",
    "get_region_modality_weights",
    "attach_modality_weights",
    "attach_region_modality_summary",
]
