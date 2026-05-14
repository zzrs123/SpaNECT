import os
import torch
import numpy as np


def adj_matrix_to_edge_index(adj_matrix):
    """
    将邻接矩阵转换为 edge_index 格式
    :param adj_matrix: torch.Tensor, 形状为 [n, n] 的邻接矩阵
    :return: torch.Tensor, 形状为 [2, E] 的 edge_index
    """
    if adj_matrix.layout == torch.sparse_coo:
        adj_matrix = adj_matrix.to_dense()
    edge_index = adj_matrix.nonzero().t()
    return edge_index


def augment(A: torch.sparse.Tensor, X: torch.Tensor,
            edge_mask_rate: float, feat_drop_rate: float):
    A = drop_edge(A, edge_mask_rate)
    X = mask_feat(X, feat_drop_rate)
    return A, X


def mask_feat(X: torch.Tensor, mask_prob: float):
    drop_mask = (
            torch.empty((X.size(1),), dtype=torch.float32, device=X.device).uniform_()
            < mask_prob
    )
    X = X.clone()
    X[:, drop_mask] = 0
    return X


def drop_edge(A: torch.sparse.Tensor, drop_prob: float):
    n_edges = A._nnz()
    mask_rates = torch.full((n_edges,), fill_value=drop_prob,
                            dtype=torch.float)
    masks = torch.bernoulli(1 - mask_rates)
    mask_idx = masks.nonzero().squeeze(1)

    E = A._indices()
    V = A._values()

    E = E[:, mask_idx]
    V = V[mask_idx]
    A = torch.sparse_coo_tensor(E, V, A.shape, device=A.device)

    return A


def add_self_loop(A: torch.sparse.Tensor):
    return A + sparse_identity(A.shape[0], device=A.device)


def normalize(A: torch.sparse.Tensor, add_self_loops=True, returnA=False):
    """Normalized the graph's adjacency matrix in the torch.sparse.Tensor format"""
    if add_self_loops:
        A_hat = add_self_loop(A)
    else:
        A_hat = A

    D_hat_invsqrt = torch.sparse.sum(A_hat, dim=0).to_dense() ** -0.5
    D_hat_invsqrt[D_hat_invsqrt == torch.inf] = 0
    D_hat_invsqrt = sparse_diag(D_hat_invsqrt)
    A_norm = D_hat_invsqrt @ A_hat @ D_hat_invsqrt
    if returnA:
        return A_hat, A_norm
    else:
        return A_norm


def sparse_identity(dim, device):
    indices = torch.arange(dim).unsqueeze(0).repeat(2, 1)
    values = torch.ones(dim)
    identity_matrix = torch.sparse_coo_tensor(indices, values,
                                              size=(dim, dim), device=device)
    return identity_matrix


def sparse_diag(V: torch.Tensor):
    size = V.size(0)
    indices = torch.arange(size).unsqueeze(0).repeat(2, 1)
    values = V
    diagonal_matrix = torch.sparse_coo_tensor(indices, values,
                                              size=(size, size), device=V.device)
    return diagonal_matrix


__all__ = [
    "adj_matrix_to_edge_index",
    "augment",
    "mask_feat",
    "drop_edge",
    "add_self_loop",
    "normalize",
    "sparse_identity",
    "sparse_diag",
]
