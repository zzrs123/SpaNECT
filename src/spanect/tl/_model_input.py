"""
Unified model input assembly module.

This module provides a single entry point for assembling all model inputs:
- RNA features (via pp)
- Omics2 features (via pp, for multi-omics)
- Image features (via tl._image_feat)
- Cell type features (via tl._cell_type)
- Labels Y

It replaces the old prepare_inputs_from_h5ad and get_input functions.
"""

from pathlib import Path
import numpy as np
import torch

from ._image_feat import get_image_feat
from ._cell_type import get_cell_type


def get_model_input(
    adata,
    data_path=None,
    data_name=None,
    config=None,
    device='cuda',
    multiomics=False,
    use_h5ad=False,
    h5ad_path=None,
    omics2_key=None,
    has_labels=True,
    cell_type_csv_pattern='cell_type_decon_{data_name}.csv',
):
    """
    Unified entry point for model input assembly.

    This function assembles all features needed for model training:
    - Xs: list of feature tensors [X_rna, X_img/omics2, X_cell]
    - Y: labels (if available)

    Parameters
    ----------
    adata : AnnData
        AnnData object with spatial transcriptomics data.
    data_path : str or Path, optional
        Path to data directory. Required for non-h5ad or when loading from files.
    data_name : str, optional
        Name of dataset/slice.
    config : dict, optional
        Configuration dict with preprocessing parameters.
    device : str, optional
        Device for tensor creation. Default: 'cuda'.
    multiomics : bool, optional
        Whether using multi-omics mode. Default: False.
    use_h5ad : bool, optional
        Whether using h5ad file. Default: False.
    h5ad_path : str or Path, optional
        Path to h5ad file.
    omics2_key : str, optional
        Key for omics2 data in adata.obsm (e.g., 'X_adt', 'X_atac').
    has_labels : bool, optional
        Whether labels are expected. Default: True.
    cell_type_csv_pattern : str, optional
        Pattern for cell type CSV file name.

    Returns
    -------
    tuple
        (Xs, Y, adata_filtered) where:
        - Xs: list of torch tensors [X_rna, X_img/omics2, X_cell]
        - Y: numpy array of labels or None
        - adata_filtered: filtered AnnData or None

    Raises
    ------
    KeyError
        If required data (cell_type, image_feat) is not found.
    ValueError
        If parameters are inconsistent.

    Notes
    -----
    This function modifies adata in place:
    - Writes to adata.obsm['image_feat'] if loaded/generated
    - Writes to adata.obsm['cell_type'] if loaded from CSV
    - Writes preprocessing results to adata.obsm
    """
    config = config or {}

    if multiomics:
        Xs, Y, adata_filtered = _assemble_multiomics_inputs(
            adata=adata,
            config=config,
            device=device,
            omics2_key=omics2_key,
            has_labels=has_labels,
        )
    else:
        Xs, Y, adata_filtered = _assemble_single_omics_inputs(
            adata=adata,
            data_path=data_path,
            data_name=data_name,
            config=config,
            device=device,
            use_h5ad=use_h5ad,
            h5ad_path=h5ad_path,
            has_labels=has_labels,
            cell_type_csv_pattern=cell_type_csv_pattern,
        )

    return Xs, Y, adata_filtered


def _assemble_single_omics_inputs(
    adata,
    data_path,
    data_name,
    config,
    device,
    use_h5ad,
    h5ad_path,
    has_labels,
    cell_type_csv_pattern,
):
    """
    Assemble inputs for single-omics mode (RNA + Image + CellType).

    Returns (Xs, Y, adata_filtered).
    """
    from ..pp import preprocess, preprocess_rna_from_X

    image_feat = get_image_feat(
        adata=adata,
        data_path=data_path,
        data_name=data_name,
        use_h5ad=use_h5ad,
        h5ad_path=h5ad_path,
        config=config,
        device=device,
    )

    if image_feat is None:
        raise ValueError(
            "Image features not found. For single-omics mode, image_feat is required. "
            "Either provide adata.obsm['image_feat'], embeddings.npy file, "
            "or allow BYOL generation."
        )

    if use_h5ad:
        X_rna_np = preprocess_rna_from_X(
            adata,
            rna_pca_dim=int(config.get('rna_pca_dim', 0) or 0),
            seed=int(config.get('seed', 0)),
        )
        adata_filtered = None
    else:
        preprocess(adata)
        adata = adata[:, adata.var['highly_variable']].copy()
        X_rna_np = adata.X.toarray() if hasattr(adata.X, 'toarray') else np.asarray(adata.X)
        adata_filtered = adata

    cell_type = get_cell_type(
        adata=adata,
        data_path=data_path,
        data_name=data_name,
        csv_pattern=cell_type_csv_pattern,
    )

    Y = _get_labels(adata, has_labels)

    X_rna = torch.tensor(X_rna_np, dtype=torch.float32, device=device)
    X_img = torch.tensor(image_feat, dtype=torch.float32, device=device)
    X_cell = torch.tensor(cell_type, dtype=torch.float32, device=device)

    Xs = [X_rna, X_img, X_cell]

    adata.obsm['X_rna_proc'] = X_rna_np
    adata.obsm['cell_type_proc'] = cell_type

    return Xs, Y, adata_filtered


def _assemble_multiomics_inputs(
    adata,
    config,
    device,
    omics2_key,
    has_labels,
):
    """
    Assemble inputs for multi-omics mode (RNA + Omics2 + CellType).

    Returns (Xs, Y, None).
    """
    from ..pp import preprocess_rna_from_X, preprocess_omics2_from_obsm

    if config.get('rna_preprocess_in_multiomics', False):
        X_rna_np = preprocess_rna_from_X(
            adata,
            rna_pca_dim=int(config.get('rna_pca_dim', 0) or 0),
            seed=int(config.get('seed', 0)),
            use_hvg=False,  # 多组学：全基因，与 Final 一致
        )
    else:
        if 'X_rna' not in adata.obsm:
            raise KeyError("multiomics mode requires adata.obsm['X_rna']")
        X_rna_np = np.asarray(adata.obsm['X_rna'], dtype=np.float32)

    final_omics2_key = omics2_key or config.get('omics2_key') or 'X_atac'
    print(f"[INFO] Using omics2_key: {final_omics2_key}")

    X_omics2_np = preprocess_omics2_from_obsm(
        adata,
        key=final_omics2_key,
        pipeline=str(config.get('omics2_pipeline', config.get('atac_pipeline', 'none'))),
        omics_dim=int(config.get('omics2_dim', config.get('atac_dim', 50))),
        standardize=bool(config.get('omics2_standardize', config.get('atac_standardize', True))),
        log1p_for_rna_like=bool(config.get('omics2_log1p', config.get('atac_log1p', True))),
        seed=int(config.get('seed', 0)),
    )

    cell_type = get_cell_type(adata=adata)

    Y = _get_labels(adata, has_labels)

    X_rna = torch.tensor(X_rna_np, dtype=torch.float32, device=device)
    X_omics2 = torch.tensor(X_omics2_np, dtype=torch.float32, device=device)
    X_cell = torch.tensor(cell_type, dtype=torch.float32, device=device)

    adata.obsm['X_rna_proc'] = X_rna_np
    adata.obsm['X_omics2_proc'] = X_omics2_np
    adata.obsm['cell_type_proc'] = cell_type

    if config.get('standardize_modalities', False):
        X_rna, X_omics2 = _standardize_modalities(X_rna, X_omics2)

    if config.get('celltype_eps', 0) > 0:
        eps = float(config['celltype_eps'])
        X_cell = X_cell + eps
        X_cell = X_cell / X_cell.sum(dim=1, keepdim=True).clamp_min(1e-12)

    Xs = [X_rna, X_omics2, X_cell]

    return Xs, Y, None


def _get_labels(adata, has_labels):
    """
    Extract labels from adata.

    Priority:
    1. adata.obs['ground_truth']
    2. adata.obs['Layer']
    3. adata.obs['layer_guess']
    4. adata.obs['spatial']

    Returns numpy array or None.
    """
    if not has_labels:
        return None

    if 'ground_truth' in adata.obs:
        return adata.obs['ground_truth'].values
    elif 'Layer' in adata.obs:
        return adata.obs['Layer'].values
    elif 'layer_guess' in adata.obs:
        return adata.obs['layer_guess'].values
    elif 'spatial' in adata.obs:
        return adata.obs['spatial'].values

    return None


def _standardize_modalities(X_rna, X_omics2):
    """
    Standardize modality features.
    """
    def _std_t(x):
        mu = x.mean(dim=0, keepdim=True)
        sd = x.std(dim=0, keepdim=True).clamp_min(1e-6)
        return (x - mu) / sd

    return _std_t(X_rna), _std_t(X_omics2)


__all__ = ["get_model_input"]
