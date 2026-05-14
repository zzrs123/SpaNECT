"""
Image feature source management module.

This module handles the acquisition of image features for spatial transcriptomics data.
It provides a unified interface for:
1. Loading existing image features from adata.obsm (for h5ad)
2. Loading pre-computed embeddings from .npy files (for raw Visium)
3. Generating new image features via BYOL (fallback)

Priority:
1. adata.obsm['image_feat'] - if h5ad and exists
2. {data_dir}/embeddings.npy - if raw Visium and exists
3. {data_dir}/{data_name}_embeddings.npy - alternative naming
4. Generate via emb.VisionEmbed - fallback
"""
import warnings
warnings.filterwarnings('ignore')

from pathlib import Path
import numpy as np


def get_image_feat(adata, data_path, data_name, use_h5ad=False, h5ad_path=None, config=None, device='cuda'):
    """
    Unified image feature source management entry point.

    Parameters
    ----------
    adata : AnnData
        AnnData object.
    data_path : str or Path
        Path to data directory (e.g., ".../Human_AD_brains").
    data_name : str
        Name of dataset/slice (e.g., "2-5").
    use_h5ad : bool, optional
        Whether using h5ad file. Default: False.
    h5ad_path : str or Path, optional
        Path to h5ad file (for additional context).
    config : dict, optional
        Configuration dict with optional 'img_epoch'.
    device : str, optional
        Device for computation. Default: 'cuda'.

    Returns
    -------
    numpy.ndarray
        Image features with shape (n_spots, feature_dim).

    Notes
    -----
    This function updates adata.obsm['image_feat'] in place.
    """
    config = config or {}

    if use_h5ad:
        result = _get_image_feat_from_h5ad(adata, data_path, data_name, config, device)
        if result is not None:
            return result

    result = _get_image_feat_from_npy(adata, data_path, data_name)
    if result is not None:
        return result

    return _generate_image_feat(adata, data_path, data_name, config, device)


def _get_image_feat_from_h5ad(adata, data_path, data_name, config, device):
    """
    Try to get image feature from h5ad's obsm.

    Returns None if not found, caller should try other methods.
    """
    if 'image_feat' in adata.obsm:
        print("[INFO] Found image_feat in adata.obsm, using directly")
        embeddings = _to_numpy(adata.obsm['image_feat'])
        return embeddings

    print("[INFO] No image_feat in adata.obsm, checking for embeddings.npy...")
    return None


def _get_image_feat_from_npy(adata, data_path, data_name):
    """
    Try to get image feature from embeddings.npy file.

    Checks multiple possible locations:
    1. {data_path}/embeddings.npy
    2. {data_path}/{data_name}_embeddings.npy

    Returns None if not found, caller should try other methods.
    """
    data_dir = Path(data_path)

    candidate_paths = [
        data_dir / "embeddings.npy",
        data_dir / f"{data_name}_embeddings.npy",
        data_dir / data_name / f"embeddings.npy"
    ]

    for emb_path in candidate_paths:
        if emb_path.exists():
            print(f"[INFO] Loading embeddings from {emb_path}")
            embeddings = np.load(emb_path)
            adata.obsm['image_feat'] = embeddings
            print(f"[INFO] Saved embeddings to adata.obsm['image_feat']")
            return embeddings

    print(f"[INFO] No embeddings.npy found at {data_dir}")
    print(f"[INFO] Checked: {[str(p) for p in candidate_paths]}")
    return None


def _generate_image_feat(adata, data_path, data_name, config, device):
    """
    Generate image features via BYOL.

    This is the fallback when no pre-computed features are available.
    """
    from ..emb import VisionEmbed

    data_dir = Path(data_path) / data_name
    print(f"[INFO] Generating image features via BYOL for {data_name}...")
    print(f"[INFO] Data directory: {data_dir}")
    print("[INFO] This may take a while for large datasets")

    embedder = VisionEmbed(
        data_dir,
        adata,
        data_name,
        epoch_num=config.get('img_epoch', 1),
        device=device
    )
    embeddings, adata_updated = embedder.run()

    if adata_updated is not None and 'image_feat' in adata_updated.obsm:
        adata.obsm['image_feat'] = adata_updated.obsm['image_feat']
    else:
        adata.obsm['image_feat'] = embeddings

    print(f"[INFO] Generated embeddings with shape {embeddings.shape}")
    return embeddings


def _to_numpy(x):
    """Convert array-like object to numpy array."""
    if hasattr(x, 'toarray'):
        return x.toarray()
    return np.asarray(x)


__all__ = ["get_image_feat"]
