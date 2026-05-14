"""Raw-data input preparation helpers.

This module only handles omics data processing.
Image features are now managed by tl._image_feat module.
Cell type features are now managed by tl._cell_type module.
"""

import os
import numpy as np
from sklearn.preprocessing import LabelEncoder

from ._preprocess import preprocess


def prepare_raw_inputs(
        adata,
        data_path,
        data_name,
        cnnType='ResNet50',
        device='cuda:3',
    ):
        """
        Prepare model inputs from raw Visium-style data (adata + slide dir).

        This function only handles RNA preprocessing.
        Image features and cell type should be obtained separately via:
        - tl.get_image_feat() for image features
        - tl.get_cell_type() for cell type

        Parameters
        ----------
        adata : AnnData
            AnnData object from datasets.load_adata.
        data_path : str or Path
            Path to data directory.
        data_name : str
            Name of the dataset/slice.
        cnnType : str, optional
            CNN type (deprecated, kept for compatibility).
        device : str, optional
            Device (deprecated, kept for compatibility).

        Returns
        -------
        tuple
            (X_gene, Y) where:
            - X_gene: numpy array of gene expression features
            - Y: numpy array of labels

        Notes
        -----
        This function is deprecated. Use tl.get_model_input() instead.
        """
        print("[WARN] prepare_raw_inputs is deprecated. Use tl.get_model_input() instead.")

        if not isinstance(data_path, (str, bytes, os.PathLike)):
            raise TypeError("data_path must be a string, bytes, or os.PathLike object")
        if not isinstance(data_name, (str, bytes, os.PathLike)):
            raise TypeError("data_name must be a string, bytes, or os.PathLike object")

        preprocess(adata=adata)
        adata = adata[:, adata.var['highly_variable']].copy()
        X_gene = adata.X.toarray().astype(np.float64)
        print("After filtering, adata shape:", adata.shape)

        label_encoder = LabelEncoder()
        Y = label_encoder.fit_transform(adata.obs['layer_guess'])

        return X_gene, Y


__all__ = ["prepare_raw_inputs"]
