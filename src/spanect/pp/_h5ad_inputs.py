"""h5ad input preparation: omics processing only.

This module only handles omics data processing for h5ad files.
Image features are managed by tl._image_feat module.
Cell type features are managed by tl._cell_type module.
Labels are managed by tl._model_input module.
"""

import numpy as np
import torch

from ._preprocess import preprocess, preprocess_omics2_from_obsm, preprocess_rna_from_X


def prepare_inputs_from_h5ad(
    adata,
    config,
    device,
    multiomics: bool,
    image_feat=None,
    cell_type=None,
    omics2_key=None,
):
    """
    Prepare model inputs from h5ad adata.

    This function only handles omics preprocessing.
    Image features and cell type should be provided by caller.

    Parameters
    ----------
    adata : AnnData
        AnnData object.
    config : dict
        Configuration dict with preprocessing parameters.
    device : str or torch.device
        Device for tensor creation.
    multiomics : bool
        Whether using multi-omics mode.
    image_feat : numpy.ndarray, optional
        Image features (required for single-omics mode).
    cell_type : numpy.ndarray, optional
        Cell type features (required for multi-omics mode).
    omics2_key : str, optional
        Key for omics2 data in adata.obsm.

    Returns
    -------
    tuple
        (Xs, Y, adata_filtered) where:
        - Xs: list of tensors in order [gene/omics1, image/omics2, cell_type]
        - Y: None (labels are handled by caller)
        - adata_filtered: filtered adata for single-omics, None for multi-omics

    Notes
    -----
    This function is deprecated. Use tl.get_model_input() instead.
    """
    print("[WARN] pp.prepare_inputs_from_h5ad is deprecated. Use tl.get_model_input() instead.")

    if multiomics:
        return _prepare_multiomics_inputs(adata, config, device, omics2_key, cell_type)
    else:
        return _prepare_single_omics_inputs(adata, config, device, image_feat, cell_type)


def _prepare_multiomics_inputs(adata, config, device, omics2_key, cell_type):
    """Prepare inputs for multi-omics mode."""
    if config.get("rna_preprocess_in_multiomics", False):
        X_rna_np = preprocess_rna_from_X(
            adata,
            rna_pca_dim=int(config.get("rna_pca_dim", 0) or 0),
            seed=int(config.get("seed", 0)),
        )
    else:
        X_rna_np = np.asarray(adata.obsm["X_rna"], dtype=np.float32)

    final_omics2_key = omics2_key or config.get("omics2_key") or "X_atac"
    print(f"[INFO] Using omics2_key: {final_omics2_key}")

    X_omics2_np = preprocess_omics2_from_obsm(
        adata,
        key=final_omics2_key,
        pipeline=str(config.get("omics2_pipeline", config.get("atac_pipeline", "none"))),
        omics_dim=int(config.get("omics2_dim", config.get("atac_dim", 50))),
        standardize=bool(config.get("omics2_standardize", config.get("atac_standardize", True))),
        log1p_for_rna_like=bool(config.get("omics2_log1p", config.get("atac_log1p", True))),
        seed=int(config.get("seed", 0)),
    )

    X_rna = torch.tensor(X_rna_np, dtype=torch.float32, device=device)
    X_omics2 = torch.tensor(X_omics2_np, dtype=torch.float32, device=device)
    adata.obsm["X_rna_proc"] = X_rna_np
    adata.obsm["X_omics2_proc"] = X_omics2_np

    if config.get("standardize_modalities", False):
        def _std_t(x):
            mu = x.mean(dim=0, keepdim=True)
            sd = x.std(dim=0, keepdim=True).clamp_min(1e-6)
            return (x - mu) / sd
        X_rna = _std_t(X_rna)
        X_omics2 = _std_t(X_omics2)

    if cell_type is None:
        if "cell_type" not in adata.obsm:
            raise KeyError(
                "multiomics mode requires adata.obsm['cell_type'] as celltype input."
            )
        cell_type = np.asarray(adata.obsm["cell_type"], dtype=np.float32)

    X_cell = torch.tensor(cell_type, dtype=torch.float32, device=device)

    if config.get("celltype_eps", 0) > 0:
        eps = float(config["celltype_eps"])
        X_cell = X_cell + eps
        X_cell = X_cell / X_cell.sum(dim=1, keepdim=True).clamp_min(1e-12)

    adata.obsm["cell_type_proc"] = cell_type
    Xs = [X_rna, X_omics2, X_cell]
    return Xs, None, None


def _prepare_single_omics_inputs(adata, config, device, image_feat, cell_type):
    """Prepare inputs for single-omics mode."""
    if image_feat is None:
        raise ValueError(
            "single-omics h5ad requires image_feat; "
            "pass image_feat from tl.get_image_feat()."
        )

    preprocess(adata)
    adata = adata[:, adata.var["highly_variable"]].copy()

    X_gene = (
        adata.X.toarray()
        if hasattr(adata.X, "toarray")
        else np.asarray(adata.X)
    )

    if cell_type is None:
        if "cell_type" not in adata.obsm:
            raise KeyError(
                "single-omics mode requires adata.obsm['cell_type'] or cell_type parameter."
            )
        cell_type = adata.obsm["cell_type"]

    X_gene = torch.FloatTensor(np.asarray(X_gene)).to(device)
    X_img = torch.FloatTensor(np.asarray(image_feat)).to(device)
    X_cell = torch.FloatTensor(np.asarray(cell_type)).to(device)

    Xs = [X_gene, X_img, X_cell]
    return Xs, None, adata


__all__ = ["prepare_inputs_from_h5ad"]
