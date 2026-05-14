"""Datasets capability package for SpaNECT."""

from . import load_data
from .load_data import load_adata, load_h5ad, read_10X_Visium

__all__ = ["load_data", "load_adata", "load_h5ad", "read_10X_Visium"]
