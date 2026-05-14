"""Preprocess compatibility facade: re-export from pp."""

from ..pp._preprocess import (
    preprocess,
    preprocess_rna,
    _standardize_cols,
    preprocess_rna_from_X,
    preprocess_omics2_from_obsm,
    preprocess_atac_from_obsm,
)

__all__ = [
    "preprocess",
    "preprocess_rna",
    "_standardize_cols",
    "preprocess_rna_from_X",
    "preprocess_omics2_from_obsm",
    "preprocess_atac_from_obsm",
]
