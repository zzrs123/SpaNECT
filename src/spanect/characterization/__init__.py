"""Characterization facade exposing modular capabilities."""

from .. import datasets, gr, pp, pl, tl
from ..datasets import load_adata, load_h5ad
from ..gr import build_spatial_graph
from ..pl import plot_pipeline, visualize_kmeans_result
from ..tl import (
    augment,
    evaluate_kmeans,
    evaluate_mclust,
    evaluate_pipeline,
    normalize,
    train_pipeline,
    get_image_feat,
    get_cell_type,
    get_model_input,
)

from ._modality_weights import (
    attach_modality_weights,
    attach_region_modality_summary,
    get_modality_weights_per_cell,
    get_region_modality_weights,
)


__all__ = [
    "datasets",
    "gr",
    "pp",
    "tl",
    "pl",
    "load_adata",
    "load_h5ad",
    "build_spatial_graph",
    "train_pipeline",
    "evaluate_pipeline",
    "plot_pipeline",
    "augment",
    "normalize",
    "evaluate_kmeans",
    "evaluate_mclust",
    "visualize_kmeans_result",
    "get_image_feat",
    "get_cell_type",
    "get_model_input",
    "get_modality_weights_per_cell",
    "get_region_modality_weights",
    "attach_modality_weights",
    "attach_region_modality_summary",
]
