"""Tools and training capability package for SpaNECT."""

from . import _graph_ops
from . import _cluster_evaluators
from . import _cluster_eval
from . import _cluster_postprocess
from . import _embedding_eval
from . import _eval_pipeline
from . import _logging
from . import _metrics
from . import _pipeline_io
from . import _preprocess
from . import _seed
from . import _train_pipeline
from . import evaluate
from . import utils_st
from . import _image_feat
from . import _cell_type
from . import _model_input

from ._eval_pipeline import evaluate_pipeline, supervised_eval_dispatch
from ._cluster_evaluators import (
    evaluate_kmeans,
    evaluate_mclust,
    get_cluster_data,
)
from ._cluster_postprocess import (
    refine,
    refine_label,
)
from ._metrics import metrics
from ._graph_ops import (
    adj_matrix_to_edge_index,
    augment,
    normalize,
)
from ..datasets import load_adata
from ._preprocess import (
    preprocess,
    preprocess_rna,
    preprocess_atac_from_obsm,
    preprocess_omics2_from_obsm,
    preprocess_rna_from_X,
)
from ._seed import fix_seed
from ._train_pipeline import train_multimodal, train_pipeline, train_single_modality
from ._image_feat import get_image_feat
from ._cell_type import get_cell_type
from ._model_input import get_model_input

__all__ = [
    "evaluate",
    "utils_st",
    "_graph_ops",
    "_cluster_evaluators",
    "_cluster_eval",
    "_cluster_postprocess",
    "_embedding_eval",
    "_eval_pipeline",
    "_logging",
    "_metrics",
    "_pipeline_io",
    "_preprocess",
    "_seed",
    "_train_pipeline",
    "_image_feat",
    "_cell_type",
    "_model_input",
    "augment",
    "normalize",
    "adj_matrix_to_edge_index",
    "fix_seed",
    "load_adata",
    "preprocess",
    "preprocess_rna",
    "preprocess_rna_from_X",
    "preprocess_omics2_from_obsm",
    "preprocess_atac_from_obsm",
    "evaluate_kmeans",
    "evaluate_mclust",
    "get_cluster_data",
    "metrics",
    "refine",
    "refine_label",
    "train_pipeline",
    "train_single_modality",
    "train_multimodal",
    "evaluate_pipeline",
    "supervised_eval_dispatch",
    "get_image_feat",
    "get_cell_type",
    "get_model_input",
]
