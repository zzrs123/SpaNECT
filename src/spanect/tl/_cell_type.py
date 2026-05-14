"""Cell type source management module.

This module handles the acquisition of cell type features from various sources.
It provides a unified interface for:
1. Loading existing cell_type from adata.obsm (for h5ad)
2. Loading from CSV files (for raw Visium)
3. Writing results back to adata.obsm

Priority:
1. adata.obsm['cell_type'] - if exists
2. CSV file at {data_path}/{data_name}/{csv_pattern}
"""

import os
from pathlib import Path

import numpy as np
import pandas as pd


def get_cell_type(
    adata=None,
    data_path=None,
    data_name=None,
    csv_pattern='cell_type_decon_{data_name}.csv',
):
    """
    Unified cell type source management entry point.

    Parameters
    ----------
    adata : AnnData, optional
        AnnData object. If provided and contains cell_type, will use it.
    data_path : str or Path, optional
        Path to data directory containing CSV files.
    data_name : str, optional
        Name of the dataset/slice.
    csv_pattern : str, optional
        Pattern for CSV file name. Default: 'cell_type_decon_{data_name}.csv'
        Supports {data_name} placeholder.

    Returns
    -------
    numpy.ndarray
        Cell type matrix with shape (n_cells, n_cell_types).

    Raises
    ------
    KeyError
        If neither adata.obsm['cell_type'] nor CSV file is found.

    Notes
    -----
    If adata is provided, the result will be written to adata.obsm['cell_type'].
    """
    if adata is not None and 'cell_type' in adata.obsm:
        print("[INFO] Found cell_type in adata.obsm, using directly")
        cell_type = _to_numpy(adata.obsm['cell_type'])
        _validate_cell_type(cell_type)
        return cell_type

    if data_path is not None and data_name is not None:
        csv_filename = csv_pattern.format(data_name=data_name)
        csv_path = Path(data_path) / data_name / csv_filename

        if csv_path.exists():
            print(f"[INFO] Loading cell_type from {csv_path}")
            cell_type = _load_cell_type_from_csv(csv_path, adata)
            _validate_cell_type(cell_type)

            if adata is not None:
                adata.obsm['cell_type'] = cell_type
                print("[INFO] Saved cell_type to adata.obsm['cell_type']")

            return cell_type

    raise KeyError(
        "Cell type not found. Either provide adata.obsm['cell_type'] "
        f"or CSV file at {data_path}/{data_name}/{csv_pattern}"
    )


def _load_cell_type_from_csv(csv_path, adata=None):
    """
    Load cell type from CSV file.

    This function maintains compatibility with the original logic in
    datasets/load_data.py.

    Parameters
    ----------
    csv_path : Path
        Path to the CSV file.
    adata : AnnData, optional
        AnnData object for index alignment.

    Returns
    -------
    numpy.ndarray
        Cell type matrix.
    """
    df_cell_type = pd.read_csv(csv_path, sep=',', header=None)

    df_cell_type = df_cell_type.iloc[1:, 1:]
    df_cell_type = df_cell_type.reset_index(drop=True)

    if adata is not None:
        df_cell_type.index = adata.obs.index

    cell_type = df_cell_type.values

    if not np.issubdtype(cell_type.dtype, np.number):
        cell_type = cell_type.astype(float)

    return cell_type


def _validate_cell_type(cell_type):
    """
    Validate cell type matrix.

    Checks that each row sums to approximately 1.0 (probability distribution).

    Parameters
    ----------
    cell_type : numpy.ndarray
        Cell type matrix to validate.

    Raises
    ------
    AssertionError
        If row sums are not approximately 1.0.
    """
    row_sums = np.sum(cell_type, axis=1)
    is_sum_one = np.allclose(row_sums, 1.0, rtol=1e-5, atol=1e-5)
    if not is_sum_one:
        print(f"[WARN] Cell type rows do not sum to 1.0. Min: {row_sums.min():.4f}, Max: {row_sums.max():.4f}")


def _to_numpy(x):
    """Convert array-like object to numpy array."""
    if hasattr(x, 'toarray'):
        return x.toarray()
    return np.asarray(x, dtype=np.float32)


__all__ = ["get_cell_type"]
