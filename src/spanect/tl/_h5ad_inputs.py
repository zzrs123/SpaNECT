"""h5ad input preparation: compatibility facade delegating to tl._model_input.

This module is deprecated. Use tl.get_model_input() instead.
"""

from ._image_feat import get_image_feat
from ._model_input import get_model_input


def prepare_inputs_from_h5ad(model):
    """
    Delegate to tl.get_model_input and assign results to model.

    This function is deprecated. Use tl.get_model_input() directly.

    Parameters
    ----------
    model : SpaNECT
        SpaNECT model instance.

    Notes
    -----
    This function exists for backward compatibility only.
    New code should use tl.get_model_input() directly.
    """
    print("[WARN] tl.prepare_inputs_from_h5ad is deprecated. Use tl.get_model_input() instead.")

    image_feat = None
    if not model.multiomics:
        image_feat = get_image_feat(
            adata=model.adata,
            data_path=model.config.get('data_path'),
            data_name=model.data_name,
            use_h5ad=model.use_h5ad,
            h5ad_path=model.h5ad_path,
            config=model.config,
            device=model.device,
        )

    omics2_key = getattr(model, 'omics2_key', None)

    Xs, Y, adata_filtered = get_model_input(
        adata=model.adata,
        data_path=model.config.get('data_path'),
        data_name=model.data_name,
        config=model.config,
        device=model.device,
        multiomics=model.multiomics,
        use_h5ad=model.use_h5ad,
        h5ad_path=model.h5ad_path,
        omics2_key=omics2_key,
        has_labels=model.has_labels,
    )

    model.Xs = Xs
    model.Y = Y

    if adata_filtered is not None:
        model.adata = adata_filtered


__all__ = ["prepare_inputs_from_h5ad"]
