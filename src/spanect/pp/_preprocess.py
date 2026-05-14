"""Preprocess helpers shared by raw and h5ad workflows."""

import numpy as np
import scanpy as sc


def preprocess_rna(adata, use_hvg=True):
    """
    RNA preprocessing: HVG selection, normalization, log1p.
    This is the standard RNA preprocessing pipeline.
    Modifies adata in place.

    Parameters
    ----------
    adata : AnnData
        AnnData object to preprocess.
    use_hvg : bool, optional
        If True (default), subset to highly variable genes (3000).
        If False, keep all genes (for multi-omics mode to match Final behavior).

    Original order from Final class:
    1. Highly variable genes selection
    2. Normalization
    3. Log transformation
    4. Filter to HVGs (only when use_hvg=True)
    """
    sc.pp.highly_variable_genes(adata, flavor="seurat_v3", n_top_genes=3000)
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    if use_hvg:
        adata._inplace_subset_var(adata.var['highly_variable'])


preprocess = preprocess_rna


def _standardize_cols(A: np.ndarray):
    mu = A.mean(axis=0, keepdims=True)
    sd = A.std(axis=0, keepdims=True)
    sd[sd < 1e-6] = 1e-6
    return (A - mu) / sd


def preprocess_rna_from_X(adata, rna_pca_dim: int = 0, seed: int = 0, use_hvg: bool = True):
    """
    Preprocess RNA data from adata.X on a copy.

    This function does NOT modify the original adata.
    It works on a copy and returns the processed feature matrix.

    Parameters
    ----------
    adata : AnnData
        AnnData object. Will NOT be modified.
    rna_pca_dim : int
        If > 0, apply PCA to reduce to this dimension.
    seed : int
        Random seed for PCA.
    use_hvg : bool, optional
        If True (default), subset to 3000 HVGs. If False, keep all genes
        (for multi-omics mode to match Final behavior).

    Returns
    -------
    numpy.ndarray
        Processed RNA features with shape (n_cells, n_hvgs) or (n_cells, n_genes)
        or (n_cells, rna_pca_dim).
    """
    ad = adata.copy()
    preprocess_rna(ad, use_hvg=use_hvg)
    X = ad.X.toarray() if hasattr(ad.X, "toarray") else np.asarray(ad.X)
    if rna_pca_dim and X.shape[1] > rna_pca_dim:
        try:
            from sklearn.decomposition import PCA
            X = PCA(n_components=rna_pca_dim, random_state=seed).fit_transform(X)
        except Exception as e:
            print(f"[WARN] RNA PCA failed ({e}), skip PCA.")
    return X.astype(np.float32)


def preprocess_omics2_from_obsm(
    adata,
    key: str = 'X_atac',
    pipeline: str = 'auto',
    omics_dim: int = 50,
    standardize: bool = True,
    log1p_for_rna_like: bool = True,
    seed: int = 0
):
    """
    Unified omics2 (ATAC/ADT/etc) preprocessing entry point.
    Returns np.ndarray with shape (n_cells, d).
    
    Parameters:
    - key: obsm key for omics2 data (default 'X_atac')
    - pipeline: preprocessing method ('auto', 'lsi', 'pca', 'clr', 'rna_like', 'none')
    - omics_dim: output dimension for dimensionality reduction
                 - 0 or negative: no dimensionality reduction, keep original dimension
                 - positive: reduce to min(omics_dim, original_dim)
    - standardize: whether to standardize columns
    - log1p_for_rna_like: whether to apply log1p for 'rna_like' pipeline
    - seed: random seed
    """
    if key not in adata.obsm:
        raise KeyError(f"obsm['{key}'] not found.")
    X = adata.obsm[key]
    X = X.toarray() if hasattr(X, "toarray") else np.asarray(X)
    X = X.astype(np.float32)

    if key == 'X_atac' and 'X_atac' not in adata.obsm and 'X_adt' in adata.obsm:
        X = adata.obsm['X_adt']
        X = X.toarray() if hasattr(X, "toarray") else np.asarray(X)
        X = X.astype(np.float32)

    original_dim = X.shape[1]
    p = (pipeline or 'auto').lower()

    if p == 'rna_like':
        A = X
        if log1p_for_rna_like:
            A = np.log1p(np.clip(A, a_min=0, a_max=None))
        if standardize:
            A = _standardize_cols(A)
        return A

    if p == 'clr':
        A = np.log1p(np.clip(X, a_min=0, a_max=None))
        A = A - A.mean(axis=1, keepdims=True)
        if standardize:
            A = _standardize_cols(A)
        return A

    if p in ('lsi', 'auto'):
        use_lsi = (p == 'lsi') or (X.shape[1] > 1000 or np.issubdtype(X.dtype, np.integer))
        if use_lsi:
            try:
                from sklearn.decomposition import TruncatedSVD
                rowsum = X.sum(axis=1, keepdims=True)
                rowsum[rowsum < 1e-12] = 1.0
                tf = X / rowsum
                df = (X > 0).sum(axis=0, keepdims=True).astype(np.float32) + 1.0
                idf = np.log((X.shape[0] + 1.0) / df) + 1.0
                tfidf = tf * idf
                k = max(2, min(omics_dim, tfidf.shape[1]-1))
                Z = TruncatedSVD(n_components=k, random_state=seed).fit_transform(tfidf)
                if standardize:
                    Z = _standardize_cols(Z)
                return Z.astype(np.float32)
            except Exception as e:
                print(f"[WARN] LSI failed ({e}), fall back to PCA.")
                p = 'pca'

    if p == 'pca':
        try:
            from sklearn.decomposition import PCA
            k = max(2, min(omics_dim, X.shape[1]-1))
            Z = PCA(n_components=k, random_state=seed).fit_transform(X)
            if standardize:
                Z = _standardize_cols(Z)
            return Z.astype(np.float32)
        except Exception as e:
            print(f"[WARN] Omics2 PCA failed ({e}), return raw.")

    else:
        if p != 'none':
            print(f"[WARN] Unknown omics2 pipeline '{pipeline}', return raw.")
        A = X
    if standardize:
        A = _standardize_cols(A)
    return A.astype(np.float32)


def preprocess_atac_from_obsm(
    adata,
    key: str = 'X_atac',
    pipeline: str = 'auto',
    atac_dim: int = 50,
    standardize: bool = True,
    log1p_for_rna_like: bool = True,
    seed: int = 0
):
    """
    Legacy alias for preprocess_omics2_from_obsm.
    Kept for backward compatibility.
    """
    return preprocess_omics2_from_obsm(
        adata,
        key=key,
        pipeline=pipeline,
        omics_dim=atac_dim,
        standardize=standardize,
        log1p_for_rna_like=log1p_for_rna_like,
        seed=seed,
    )


__all__ = [
    "preprocess",
    "preprocess_rna",
    "_standardize_cols",
    "preprocess_rna_from_X",
    "preprocess_omics2_from_obsm",
    "preprocess_atac_from_obsm",
]
